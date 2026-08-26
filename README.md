# HTX AI Engineering Take-Home — Parts 1, 2 & 3

**Document Extraction, MCP Date Normalisation & LLM Temporal Reasoning with LangChain + Gemini**

---

## My approach

This solution uses LLMs only where semantic interpretation genuinely adds value and keeps deterministic operations in Python. In Parts 1 and 2, the assessment specifies exactly which pages to inspect, so I avoided adding retrieval entirely — the LLM identifies evidence and Python verifies it deterministically. Part 2 adds MCP as a tool boundary so that date normalisation (a deterministic task) stays in pure Python while Gemini handles temporal reasoning. Part 3 is the first point where the relevant pages are unknown and must be found from a question alone; that is where I introduce hybrid retrieval and a LangGraph supervisor. The progression is intentional: each part adds exactly the machinery the problem requires and no more.

---

## Quick start

**Prerequisites:** Python 3.14

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install the package and all dependencies
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env
# Edit .env — set GEMINI_API_KEY to your Google AI Studio key

# 4. Run the pipelines
python scripts/run_part1.py
python scripts/run_part2.py
python scripts/run_part3.py   # also builds the semantic index (requires GEMINI_API_KEY)
```

Outputs:

| File | Contents |
|------|----------|
| `outputs/part1_result.json` | Five final extracted values |
| `outputs/part1_evidence.json` | Per-field audit trail |
| `outputs/part2_result.json` | Two normalised dates with temporal classification |
| `outputs/part2_evidence.json` | Full audit trail including MCP tool trace |
| `outputs/part3_result.json` | Answer for the required HTX query |
| `outputs/part3_trace.json` | Workflow trace from the required HTX query |
| `outputs/part3_demo_queries.json` | Results for all four demonstration queries |



## Environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_EXTRACTION_MODEL` | Model for extraction tasks (default: `gemini-3.5-flash-lite`, ~500 RPD on the development project's free-tier quota) |
| `GEMINI_EXTRACTION_FALLBACK_MODEL` | Fallback for extraction on quota/rate-limit errors (default: `gemini-3.1-flash-lite`) |
| `GEMINI_REASONING_MODEL` | Model for reasoning + tool-selection tasks (default: `gemini-3.5-flash-lite`, ~500 RPD on the development project's free-tier quota) |
| `GEMINI_REASONING_FALLBACK_MODEL` | Fallback for reasoning on quota/rate-limit errors (unset by default — quota failures surface clearly rather than silently downgrading) |
| `SOURCE_PDF` | Path to the data-source PDF (default: `data/fy2024_analysis_of_revenue_and_expenditure.pdf`) |

---

## Library choices

| Library | Reason |
|---------|--------|
| `pdfplumber` | Lightweight page-level text and table extraction; no OCR or CV overhead; output inspected page-by-page before writing any parser code |
| `langchain` + `langchain-google-genai` | Structured output via `llm.with_structured_output(PydanticModel)` keeps LLM responses schema-validated; prompt templates are version-controlled separately from orchestration logic |
| `mcp>=1.29.1,<2` | MCP SDK used by `FastMCP` to expose the date-normaliser over stdio. Pinned below 2.x because `langchain-mcp-adapters==0.3.1` is incompatible with the mcp 2.x API |
| `langchain-mcp-adapters==0.3.1` | Converts MCP tool definitions into LangChain `BaseTool` objects so they can be bound to Gemini through LangChain |
| `langgraph>=1.2.0` | Compiles the Part 3 supervisor graph; provides conditional fan-out (parallel specialist branches) and fan-in (synthesis waits for all active branches) |
| `rank-bm25` | BM25Okapi index for lexical retrieval; no server or persistent storage required |
| `numpy` | Cosine similarity computation over the in-memory embedding matrix |
| `pydantic` | Runtime-validated schemas for every LLM output and final result; field-level documentation doubles as prompt guidance |
| `python-dotenv` | Keeps secrets out of source code; standard `.env` convention |
| `pytest` | Parametrised unit tests for all deterministic components; `@pytest.mark.integration` separates API-dependent tests |

## Parser choice — why pdfplumber

I started with `pdfplumber` rather than a heavier parser because the PDF already contains machine-readable text — no OCR needed. After inspecting the required pages, I found that pages 5–6 were straightforward narrative, page 20 was extractable as a table, and page 8 had a borderless table where `extract_tables()` returned nothing. For page 8 I reconstructed the values from extracted text using a trailing-numeric-token heuristic. I kept this approach intentionally document-specific rather than introducing a general-purpose document intelligence stack for a single known file.

### What inspection revealed

| Page | `extract_tables()` | Strategy used |
|------|--------------------|---------------|
| 5–6 | N/A (narrative) | `extract_text()` |
| 8 | Returns 0 tables — no ruling lines detected | `extract_text()` parsed with trailing-numeric-token heuristic |
| 20 | Returns 1 table with sparse None-filled columns | `extract_tables()` + per-row None stripping |

Page 8 additionally contains a garbled line (`Add: 22,376, 23,480,570, 22,915, 0`) from chart-coordinate data embedded in the PDF stream.  This is detected and skipped via the `_is_garbled_coordinate_line()` function in `pdf_parser.py`.

**Page-number mapping** — verified: printed page *N* is at physical index *N − 1* (0-based). Confirmed by inspecting the page-number stamp on page 4 at physical index 3.  All page references go through the `page_index()` function; no physical index is hardcoded elsewhere.

---

## Architecture

```
Data-source PDF
      │
      ▼
  pdfplumber
      │
      ▼
Extract only required pages
      │
      ├─────────────────────┐
      ▼                     ▼
Narrative text          Financial tables
pages 5–6               pages 8 and 20
      │                     │
      └──────────┬──────────┘
                 ▼
       LangChain + Gemini
                 │
                 ▼
 Identify semantic evidence /
 relevant row and column
                 │
                 ▼
    Deterministic Python layer
                 │
        ┌────────┴────────┐
        ▼        ▼        ▼
     verify   retrieve  normalise
     source    value    units/types
        │        │        │
        └────────┴────────┘
                 │
                 ▼
          Pydantic output
```

**Core principle: the LLM performs semantic understanding; Python performs numerical correctness.**

The LLM is never asked to produce final numeric answers.  For narrative fields it returns verbatim evidence that Python verifies against the source text.  For table fields it returns row and column labels that Python uses to retrieve the exact cell value from the parsed table.

---

## Prompt strategy

Four targeted extraction tasks, each using only the relevant page content:

| Task | Pages | LLM output | Python action |
|------|-------|------------|---------------|
| A | 5 | `CorporateTaxEvidence` (evidence + sub-strings) | verify evidence in source; verify sub-strings in evidence; normalize |
| B | 5–6 | `OperatingRevenueTaxes` (list of `TaxWithEvidence`) | per-item: verify evidence in source; verify name in evidence; normalize |
| C | 8 | `TableCellSelection` (row label + column label) | retrieve cell from `ParsedTable`; normalize |
| D | 20 | `TableCellSelection` (row label + column label) | retrieve cell from `ParsedTable`; normalize |

All improved prompts share a grounding preamble that instructs the model to:

1. Use only the supplied document content.
2. Not use training data or general knowledge.
3. Not infer or calculate values.
4. Return `null` if evidence is absent.
5. Preserve exact source text (no paraphrasing or rounding).

---

## Hallucination mitigation

| Risk | Mitigation |
|------|------------|
| LLM invents a number | Numbers come from `ParsedTable.get_cell()` or raw text, not from LLM-generated floats |
| LLM paraphrases evidence | `validate_evidence_in_source()` performs verbatim (whitespace-normalised) lookup |
| LLM returns an unsupported tax name | `validate_tax_with_evidence()` checks each tax's evidence exists in source and the name appears in that evidence |
| LLM returns a non-existent row/column | `ParsedTable.get_cell()` raises `ExtractionValidationError` |
| LLM omits evidence | Extraction raises `ExtractionValidationError`; the field is `None` in the audit trail, not silently `0.0` |
| Unit confusion | `NormalizedNumber.source_unit` is preserved alongside `normalized_value` throughout the pipeline |

---

## Naive vs improved prompt comparison


| Dimension | Naive prompt | Improved prompt |
|-----------|-------------|-----------------|
| Grounding | None — model may use training data | Explicit: "use only supplied content" |
| Output structure | Free-form prose | Pydantic schema (`CorporateTaxEvidence`) |
| Evidence | Not requested | Required: verbatim source sentence |
| Missing-value behaviour | May fabricate | Returns `null` |
| Deterministic validation | Impossible | `validate_evidence_in_source()` + `validate_value_in_evidence()` |

---

## Project structure

```
src/part1/
  __init__.py
  models.py        — Pydantic models (evidence, selection, final result)
  normalizers.py   — normalize_to_float (deterministic, no LLM)
  validators.py    — source-grounding checks, ExtractionValidationError
  pdf_parser.py    — pdfplumber extraction, ParsedTable
  prompts.py       — ChatPromptTemplate instances
  extractor.py     — LangChain + Gemini orchestration

scripts/run_part1.py       — entry point
tests/                — unit + integration tests
outputs/              — part1_result.json, part1_evidence.json
```

---

## Running tests

```bash
# Unit tests only — no API key required
python -m pytest tests/ -m "not integration" -v

# Include the optional end-to-end smoke test (requires GEMINI_API_KEY)
python -m pytest tests/ -v
```

---

## Assumptions

### Year labelling mismatch (fields 1 and 2)

The assessment asks for *"Corporate Income Tax in 2024"* and specifies **page 5** as the source.  However, page 5 of the source document is within **Section 01 — Update on Financial Year 2023**, which reports the *Revised FY2023* figures.  The Corporate Income Tax amount shown on page 5 is the revised FY2023 collection ($28.4 billion).

I chose to treat the explicitly specified source page as authoritative: the values are extracted from page 5 as-is.  This assumption is documented here and in `outputs/part1_evidence.json`.

The source document was **not modified** to reconcile this discrepancy.

### YoY percentage mismatch (field 2)

The assessment asks for the *"YOY percentage difference of Corp Income Tax in 2024"*.  However, page 5 describes 17.0% as the increase from the **Estimated FY2023** figure to the **Revised FY2023** figure — not a calendar year-on-year comparison.

Since the assessment explicitly references page 5 and 17.0% is the only percentage figure associated with Corporate Income Tax on that page, this value is returned without reinterpretation.  The distinction is noted here for transparency.

### Table 1.1 "Latest Actual" column

Table 1.1 on page 8 has three value columns: *Actual FY2022*, *Estimated FY2023*, and *Revised FY2023*.  "Latest Actual" is interpreted as the **Actual FY2022** column because it is the only column labelled "Actual" (the FY2023 figures are estimates or revisions, not finalized actuals at the time of publication).

### Top-up amounts unit

Table 2.4 on page 20 states amounts in **$ million**.  The extracted `total_top_ups_2024` float is therefore in millions (20352.0), consistent with the source unit retained in `part1_evidence.json`.

### pdfplumber `extract_tables()` failure on page 8

Page 8 returns zero tables from `extract_tables()` because Table 1.1 has no visible ruling lines.  The text extraction is used instead with a trailing-numeric-token parser.  This was verified to produce correct values for all tested rows including `OVERALL FISCAL POSITION`.

---

---

# Part 2 — Data Extraction + Datetime Tool via Local MCP

## What it does

Part 2 extracts two specific dates from the PDF, normalises them to ISO 8601 format via a local MCP tool, then classifies each date against a fixed reference date using Gemini.

| Stage | What happens |
|-------|--------------|
| **1A** | Gemini extracts a date and its source sentence from page 1 (distribution date) |
| **1B** | Gemini extracts a date and its source sentence from page 36 (Estate Duty cutoff) |
| **1C** | Python validates `original_text` and `date_text` verbatim in the page source |
| **2A** | Gemini calls the `normalize_date` MCP tool with the raw date string |
| **2B** | The MCP server runs `normalize_date_value()` — purely deterministic, no LLM |
| **2C** | Python captures the MCP tool call and result for the audit trail |
| **3** | Gemini classifies each ISO date as Expired / Upcoming / Ongoing against 2024-01-01 |

```bash
python scripts/run_part2.py
```

Outputs:

| File | Contents |
|------|----------|
| `outputs/part2_normalized_dates.json` | Plain JSON list of two ISO date strings |
| `outputs/part2_result.json` | HTX-schema items: `original_text`, `normalized_date`, `status` |
| `outputs/part2_evidence.json` | Full audit trail: source page, validation flag, MCP tool trace, LLM rationale |

---

## Why local MCP for date normalisation?

Date parsing is a deterministic task — a given date string should always produce the same ISO output. I exposed `normalize_date_value()` as a local MCP tool so that Gemini's tool call is observable: I can inspect `response.tool_calls[0]` and confirm the model actually requested the tool rather than answering from training data. The normalisation itself is pure Python wrapping `datetime.strptime`, so it is predictable and unit-testable without any API calls.

MCP is admittedly more machinery than needed to call `datetime.strptime()` directly. I used it because Part 2 specifically asks for a local MCP implementation and I wanted the tool boundary to be explicit and inspectable. The underlying normalisation stays as a pure function so MCP remains an interface boundary rather than being mixed into business logic.

## Why LLM classification, not rule-based?

The distinction between **Expired** and **Ongoing** requires semantic context:

> *"Estate Duty does not apply to a person who dies after 15 February 2008."*

The date `2008-02-15` is before the reference date `2024-01-01`, which a simple date comparison classifies as Expired. However, I interpret this as **Ongoing** because the source describes an open-ended condition applying after the cutoff date, with no stated end date — the condition still governs all deaths in 2024. A regex-over-date approach cannot make this distinction.

Rather than asking the model to resolve temporal semantics and classification in one step (which a lightweight model can get wrong), the classifier is split into two calls. First it produces a structured `TemporalInterpretation` describing how the date functions grammatically — whether it is a point event, a period, or a cutoff/threshold, and what directional relation ("after", "until", "on", …) the text expresses. The classification step then receives this structured reading as explicit context, so its task is narrower: label selection given a known temporal structure. A deterministic consistency checker validates the result and issues one LLM retry if the classification contradicts the interpretation. This decomposition makes the behaviour of a lightweight model reliable on ambiguous cases without hard-coding any answer.

## Part 2 design decisions

**`original_text` in output:** The HTX schema includes `original_text` alongside `normalized_date`. Retaining the verbatim source sentence provides human-readable provenance, lets reviewers verify the classification without consulting the PDF, and gives the LLM classifier the semantic context it needs (see above).

**Reference date is fixed at 2024-01-01:** Specified by the assessment; not taken from the system clock. `REFERENCE_DATE = date(2024, 1, 1)` is a module-level constant in `src/part2/models.py`.

---

## Architecture

```
PDF pages 1 and 36
        |
        v
   pdfplumber (page.extract_text() / page.crop() for two-column pages)
        |
        v
   Gemini (with_structured_output -> ExtractedDate)
   +-- original_text: source sentence
   +-- date_text: raw date string
        |
        v
   Python validation
   +-- original_text verbatim in page source  -> validate_evidence_in_source()
   +-- date_text within original_text         -> validate_value_in_evidence()
        |
        v
   MCP stdio session (mcp 1.29.1 + FastMCP)
   +-- Gemini.bind_tools(lc_tools).ainvoke(...)
   |   +-- response.tool_calls[0] verified == "normalize_date"
   +-- session.call_tool("normalize_date", args)
       +-- normalize_date_value() -> ISO string
        |
        v
   Gemini — Step 1: temporal interpretation (with_structured_output -> TemporalInterpretation)
   +-- temporal_type: point_event | period | cutoff_or_threshold
   +-- relation: on | after | until | from | between | …
   +-- has_explicit_end: bool
        |
        v
   Gemini — Step 2: classification (with_structured_output -> InternalDateClassification)
   +-- status: Expired | Upcoming | Ongoing
   +-- reason: LLM rationale
        |
        v
   Python consistency check
   +-- detects contradictions (e.g. open-ended after-cutoff classified as Expired)
   +-- issues one LLM retry with conflict described if contradictory
        |
        v
   Three JSON output files
```

---

## Assumptions

### Reference date is fixed at 2024-01-01

The spec states the reference date must not come from the system clock. `REFERENCE_DATE = date(2024, 1, 1)` is a module-level constant in `src/part2/models.py`.

### Source pages are 1 and 36

Confirmed by inspecting the physical page content: the distribution date appears on printed page 1 (physical index 0) and the Estate Duty cutoff appears on printed page 36 (physical index 35).

### Page 36 two-column layout

`pdfplumber`'s default `extract_text()` interleaves the two columns on page 36, corrupting the Estate Duty sentence. To avoid this, the pipeline uses `pdfplumber`'s `page.crop()` to extract text from the left half of the page only (`x in [0, page.width/2]`). The cropped output contains the full clean sentence *"Estate Duty does not apply to a person who dies after 15 February 2008."* without interleaving, enabling full `original_text` verbatim grounding.

### Estate Duty: Ongoing vs Expired disambiguation

A naive date-comparison rule classifies `2008-02-15` as **Expired** (before the 2024-01-01 reference date), and this is what the model produces when asked to classify directly. I interpret this as **Ongoing** because the source describes an open-ended condition applying after the cutoff date, with no stated end date — the condition still governs all deaths in 2024.

To make this reliable without hardcoding the answer, the classifier now uses two steps. First it produces a structured `TemporalInterpretation` from the source sentence (`temporal_type=cutoff_or_threshold`, `relation=after`, `has_explicit_end=false`). Then it classifies using that interpretation as explicit context. A deterministic consistency checker then validates the result: an open-ended after-cutoff condition cannot be `Expired`. If the LLM's first classification contradicts the interpretation it was given, one retry is issued with the specific conflict described. This is sufficient — on the retry the model produces `Ongoing`.

---

## Part 2 project structure

```
src/part2/
  __init__.py
  models.py          — Pydantic schemas, REFERENCE_DATE, TemporalStatus enum
  date_normalizer.py — Deterministic ISO conversion (no LLM, no MCP)
  mcp_server.py      — FastMCP server exposing normalize_date over stdio
  mcp_client.py      — Async context manager: start server, yield (session, lc_tools)
  prompts.py         — ChatPromptTemplates for extraction, interpretation, and classification
  date_extractor.py  — Gemini extraction + evidence validation
  classifier.py      — Two-step temporal classification: interpret → classify → consistency check
  workflow.py        — Async orchestration of all stages

scripts/run_part2.py — Entry point; writes three output files

tests/part2/
  test_date_normalizer.py           — 15 unit tests, no API
  test_date_extraction_validation.py — 8 validation tests, no API
  test_models.py                    — 15 schema tests, no API
  test_classifier_consistency.py    — 13 deterministic consistency-checker tests, no API
  test_mcp_integration.py           — 5 real MCP boundary tests, no Gemini
  test_part2_integration.py         — end-to-end tests (require GEMINI_API_KEY)
```

---

## Part 3 — LangGraph multi-agent supervisor

### Overview

Part 3 implements a **LangGraph supervisor** that answers questions about the FY2024 Singapore Budget document by routing to specialised agents, running hybrid retrieval loops, and synthesising grounded answers.

```
User query
    │
    ▼
supervisor_route  ──── routes to one or both ────┐
                                                  │
                              ┌───────────────────┴──────────────────┐
                              ▼                                       ▼
                        revenue_agent                         expenditure_agent
                     (retrieval loop)                          (retrieval loop)
                              │                                       │
                              └───────────────────┬──────────────────┘
                                                  ▼
                                              synthesis
                                                  │
                                                  ▼
                                           final_answer
```

### Architecture

#### Supervisor pattern

The supervisor node (`src/part3/supervisor.py`) reads the user query, selects the appropriate specialist(s), and delegates a scoped task with a list of required aspects. Each specialist operates independently and writes to a separate key in the shared state, so parallel writes never conflict. The reducer on the `trace` key (`operator.add`) safely merges event lists from concurrent branches.

**Why a supervisor instead of a single RAG chain?** The supervisor makes domain routing explicit and testable, while allowing specialised retrieval strategies to run independently. Adding a new specialist (e.g. a tax-policy agent) does not require rewriting existing code.

#### Parallel fan-out (LangGraph)

When both revenue and expenditure are required, `add_conditional_edges` returns a list of two node names, causing LangGraph to activate both branches in parallel. The synthesis node fires only after all active branches complete (LangGraph fan-in semantics). This can reduce wall-clock time for combined queries because the independent specialists run concurrently.

```python
# Fan-out (parallel when list has two items)
graph.add_conditional_edges("supervisor_route", route_fn, ["revenue_agent", "expenditure_agent"])

# Fan-in (synthesis waits for all active branches)
graph.add_edge("revenue_agent", "synthesis")
graph.add_edge("expenditure_agent", "synthesis")
```

#### Specialist agent loop

Each specialist (`src/part3/specialist.py`) runs an internal subgraph:

```
START → retrieve → assess → [sufficient?] → extract_facts → END
                        └──→ reformulate ──→ retrieve (loop)
```

The loop terminates when any of the following conditions holds:
- Evidence is assessed as sufficient for all required aspects.
- Maximum retrieval attempts (3) are exhausted.
- No new chunks were retrieved (no new evidence available).
- The reformulated query is identical to a previously issued query (repeated query guard).

This bounded loop prevents infinite retries while still allowing the agent to recover from initial sparse retrieval.

#### Hybrid BM25 + semantic retrieval (RRF)

`src/part3/retriever.py` implements **Reciprocal Rank Fusion** (Cormack et al. 2009) over BM25 and semantic embeddings:

$$\text{RRF}(d) = \sum_{r \in \{BM25,\ \text{semantic}\}} \frac{1}{K + \text{rank}_r(d)}, \quad K = 60$$

**Why hybrid?** I compared BM25, semantic retrieval and hybrid RRF on four representative queries. Semantic retrieval already performed well on this small corpus, and hybrid RRF matched rather than exceeded its hit rate. I kept the hybrid approach because BM25 is inexpensive at this scale and provides an exact-term signal alongside semantic retrieval, but I would evaluate whether that extra complexity remains worthwhile on a larger corpus.

**Semantic embedding model:** `models/gemini-embedding-001` (supported through at least May 2028). The previously used `text-embedding-004` was deprecated by Google on 14 January 2026.

**Why no vector database?** The corpus is ~62 chunks from a single 37-page PDF. Maintaining a vector DB introduces infrastructure cost and complexity without benefit at this scale. The in-memory semantic index is built once at startup by embedding the 62 document chunks.

#### Grounding and hallucination control

`src/part3/grounding.py` validates every LLM-extracted fact before inclusion in the result:
1. The `evidence_text` must appear verbatim (modulo whitespace normalisation) in the source chunks for the stated page.
2. The `amount_text` (if provided) must appear inside `evidence_text`.

Facts that fail grounding are logged as warnings and dropped — they are never surfaced in the final answer. This ensures the answer cites only what was actually retrieved, not what the LLM "knows" from pre-training.

> **Retrieval rank ≠ truth.** A chunk appearing first in the ranking does not mean it is factually correct. Grounding validation provides an independent, deterministic check that the LLM's claim is traceable to the source document.

### Retrieval benchmark

Run `scripts/evaluate_part3_retrieval.py` to reproduce `outputs/part3_retrieval_benchmark.json`.  
Hit rate = fraction of known-relevant pages appearing (uniquely) in the top-8 results.

| Query | BM25 | Semantic | Hybrid RRF |
|-------|------|----------|------------|
| Exact: "Corporate Income Tax Operating Revenue FY2024" | 1.00 | 1.00 | **1.00** |
| Paraphrased: "main sources of government income" | 0.50 | 0.75 | **0.75** |
| Paraphrased: "How is the Future Energy Fund being financed?" | 1.00 | 1.00 | **1.00** |
| Exact: "top-ups endowment trust funds Budget 2024" | 0.33 | 1.00 | **1.00** |

Corpus: 62 chunks from 37 PDF pages. Embeddings: `gemini-embedding-001` (3072-dim).

**Interpretation:** BM25 performed well when the query shared exact terminology with the document, but recall dropped to 0.33–0.50 for the two broader queries. Semantic retrieval recovered more of the labelled relevant pages. In this small benchmark, hybrid RRF matched semantic retrieval rather than improving the measured hit rate. I retained RRF because adding the BM25 ranking is inexpensive for a 62-chunk in-memory corpus and provides an additional lexical signal, but this benchmark does not prove that hybrid retrieval is universally better.

### Running Part 3

```bash
# Build the semantic index and run the four demonstration queries
python scripts/run_part3.py

# Retrieval benchmark (BM25 vs semantic vs hybrid)
python scripts/evaluate_part3_retrieval.py
```

**Outputs:**

| File | Contents |
|------|----------|
| `outputs/part3_result.json` | Answer for the required HTX query |
| `outputs/part3_trace.json` | Full trace from the required HTX query |
| `outputs/part3_demo_queries.json` | Results for all four demonstration queries |
| `outputs/part3_retrieval_benchmark.json` | Per-query hit rates for all three retrieval methods |

### Key design decisions and trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Supervisor routes to specialist subgraphs | Separates routing logic from retrieval logic; each specialist is independently testable | Adds one LLM call per query for routing |
| Parallel fan-out for dual-domain queries | Reduces wall-clock time; agents do not wait on each other | Parallel branches share quota — if rate limits are tight, sequential execution is safer |
| Bounded retrieval loop (max 3 attempts) | Prevents infinite LLM calls; stops on no-new-evidence | May surface incomplete answers for difficult questions |
| In-memory BM25 + numpy semantic index | Fast startup; no infrastructure dependency | Must rebuild on every process restart; does not scale beyond ~10,000 chunks |
| Deterministic grounding validation | No hallucinated amounts or citations in output | Drops valid facts if LLM paraphrases evidence text rather than quoting verbatim |
| `INSUFFICIENT_EVIDENCE` status | Honest about what was not found; synthesis produces a partial answer | Users receive less complete answers than expected for hard questions |

### Assumptions

- The corpus is one known PDF (single FY2024 Budget document). A production system would maintain a persistent index and support incremental updates.
- Routing is binary (revenue / expenditure). The supervisor model can be extended with additional specialist types without changing the graph topology.
- The current implementation is stateless across queries; `SupervisorState.trace` records workflow execution events for the current run. Multi-turn dialogue would require persisting state across `ainvoke` calls.

---

## Limitations and future improvements

**Part 1 — tax extraction:** Evidence validation ensures returned values are grounded in the source, but single-pass extraction cannot guarantee 100% recall of every possible tax mention across arbitrarily structured documents.

**PDF parsing:** The page-8 table reconstruction is optimised for this specific document layout. A production system supporting arbitrary PDFs would use a more general table extraction strategy rather than token-level heuristics.

**Part 2 — temporal classification:** Classification is probabilistic because semantic interpretation is performed by an LLM, even though its inputs and allowed output values are constrained. A different model or prompt could produce a different result for ambiguous cases.

**Scalability:** The solution is scoped to one known document. Parts 1 and 2 use mostly sequential processing, while Part 3 uses parallel specialist branches when both domains are required. A production multi-document system would need persistent indexes, bounded concurrency, durable job handling and stronger observability.
