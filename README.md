# HTX AI Engineering Take-Home — Parts 1, 2 & 3

**Document Extraction, MCP Date Normalisation & LLM Temporal Reasoning with LangChain + Gemini**

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

# 4. Run the extraction pipeline
python scripts/run_part1.py
```

Outputs:

| File | Contents |
|------|----------|
| `outputs/part1_result.json` | Five final extracted values |
| `outputs/part1_evidence.json` | Per-field audit trail |

Run Part 3 (LangGraph multi-agent supervisor):

```bash
python scripts/run_part3.py
```



## Environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_EXTRACTION_MODEL` | Model for extraction tasks (default: `gemini-3.5-flash-lite`, ~500 RPD on the development project's free-tier quota) |
| `GEMINI_EXTRACTION_FALLBACK_MODEL` | Fallback for extraction on quota/rate-limit errors (default: `gemini-3.1-flash-lite`) |
| `GEMINI_REASONING_MODEL` | Model for reasoning + tool-selection tasks (default: `gemini-3.6-flash`, ~20 RPD on the development project's free-tier quota) |
| `GEMINI_REASONING_FALLBACK_MODEL` | Fallback for reasoning on quota/rate-limit errors (default: `gemini-3.5-flash-lite`) |
| `SOURCE_PDF` | Path to the data-source PDF (default: `data/fy2024_analysis_of_revenue_and_expenditure.pdf`) |

---

## Library choices

| Library | Reason |
|---------|--------|
| `pdfplumber` | Lightweight page-level text and table extraction; no OCR or CV overhead; output inspected page-by-page before writing any parser code |
| `langchain` + `langchain-google-genai` | Structured output via `llm.with_structured_output(PydanticModel)` keeps LLM responses schema-validated; prompt templates are version-controlled separately from orchestration logic |
| `mcp>=1.29.1,<2` | MCP SDK used by `FastMCP` to expose the date-normaliser over stdio. Pinned below 2.x because `langchain-mcp-adapters==0.3.1` is incompatible with the mcp 2.x API |
| `langchain-mcp-adapters==0.3.1` | Converts MCP tool definitions into LangChain `BaseTool` objects so they can be bound to Gemini through LangChain |
| `pydantic` | Runtime-validated schemas for every LLM output and final result; field-level documentation doubles as prompt guidance |
| `python-dotenv` | Keeps secrets out of source code; standard `.env` convention |
| `pytest` | Parametrised unit tests for all deterministic components; `@pytest.mark.integration` separates API-dependent tests |

## Parser choice — why pdfplumber

The source document (`fy2024_analysis_of_revenue_and_expenditure.pdf`) contains both narrative prose and financial tables.  `pdfplumber` was selected because:

- **Page-level extraction** — `page.extract_text()` and `page.extract_tables()` operate on individual pages, making it straightforward to target only the pages required.
- **Table detection** — the built-in table extractor handles ruled tables automatically; for borderless tables a word-coordinate fallback is available.
- **Lightweight** — adds no OCR, computer-vision, or document-intelligence overhead.
- **Verified output** — before any parsing code was written, the raw output of `extract_text()` and `extract_tables()` was inspected for pages 5, 6, 8, and 20 to confirm which strategy is required per page.

No alternative PDF library (PyMuPDF, Docling, etc.) was introduced because `pdfplumber` handled all required pages.

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

Date parsing is a deterministic task — a given date string should always produce the same ISO output regardless of model state or temperature.  Exposing `normalize_date_value()` as a **local MCP tool** means:

1. **Gemini's tool-call is observable** — the `tool_calls` list on the response object is inspected before execution, proving the model requested the tool (not that it answered from training data).
2. **The normalisation itself is pure Python** — no LLM involved, so the output is predictable and unit-testable without any API calls.
3. **MCP stdio transport** — the server runs as a subprocess; no network port, no auth, no persistence.

---

## Why deterministic normalisation, not LLM normalisation?

Normalising `"16 February 2024"` → `"2024-02-16"` does not require semantic understanding. Delegating it to an LLM adds latency, API cost, and non-determinism. The MCP tool wraps `datetime.strptime` over a known set of formats derived from the source document; it raises `DateNormalizationError` for anything outside that set.

---

## Why LLM classification, not rule-based?

The distinction between **Expired** and **Ongoing** requires semantic context:

> *"Estate Duty does not apply to a person who dies after 15 February 2008."*

The date `2008-02-15` is before the reference date `2024-01-01`, which a simple date comparison would classify as Expired. However, the source uses the date as a threshold — "after 15 February 2008" — and does not state an end date. I therefore interpret it as an open-ended condition that includes the reference date, resulting in `Ongoing`. A regex-over-date approach cannot make this distinction. The LLM receives both `original_text` and `normalized_date` so it can reason about whether the date marks a boundary of an ongoing rule, a past event, or a future event.

---

## Why is `original_text` retained in the output?

The HTX output schema includes `original_text` alongside `normalized_date`. Retaining the verbatim source sentence:

- Provides human-readable provenance for each date.
- Allows reviewers to verify the classification against the source without consulting the PDF.
- Enables the LLM classifier to use semantic context (see above).

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
   Gemini (with_structured_output -> InternalDateClassification)
   +-- status: Expired | Upcoming | Ongoing
   +-- reason: LLM rationale
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

A pure date-comparison rule would classify `2008-02-15` as **Expired** (before the 2024-01-01 reference date). I interpret the Estate Duty statement as an open-ended condition because the source sentence applies to persons who die "after 15 February 2008" and states no end date. Under this interpretation, the condition includes the 2024-01-01 reference date and is classified as `Ongoing`. The LLM classification is consistent with this interpretation, and the integration test asserts this outcome.

---

## Part 2 project structure

```
src/part2/
  __init__.py
  models.py          — Pydantic schemas, REFERENCE_DATE, TemporalStatus enum
  date_normalizer.py — Deterministic ISO conversion (no LLM, no MCP)
  mcp_server.py      — FastMCP server exposing normalize_date over stdio
  mcp_client.py      — Async context manager: start server, yield (session, lc_tools)
  prompts.py         — ChatPromptTemplates for extraction and classification
  date_extractor.py  — Gemini extraction + evidence validation
  classifier.py      — Gemini temporal classification
  workflow.py        — Async orchestration of all stages

scripts/run_part2.py — Entry point; writes three output files

tests/part2/
  test_date_normalizer.py           — 15 unit tests, no API
  test_date_extraction_validation.py — 8 validation tests, no API
  test_models.py                    — 12 schema tests, no API
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

**Why hybrid?** Budget documents mix exact financial terminology ("Corporate Income Tax", "Operating Revenue") and natural-language prose. BM25 dominates for exact-term queries; semantic embeddings recover paraphrased or conceptual queries. Hybrid RRF is expected to outperform either method alone across the evaluation queries.

**Semantic embedding model:** `models/gemini-embedding-001` (supported through at least May 2028). The previously used `text-embedding-004` was deprecated by Google on 14 January 2026.

**Why no vector database?** The corpus is ~62 chunks from a single 37-page PDF. Maintaining a vector DB introduces infrastructure cost and complexity without benefit at this scale. The in-memory semantic index is built once at startup in O(n) API calls.

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

**Interpretation:** BM25 alone fails on paraphrased and indirect queries (hit rates 0.33–0.50). Semantic embeddings close that gap. Hybrid RRF matches or exceeds semantic retrieval on every query, while preserving BM25's exact-term strength. The largest gain is `top_ups_endowment` (BM25 0.33 → Hybrid 1.00), driven by semantic recall of page 18 and page 20 which BM25 misses when the query wording diverges from the document's phrasing.

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

**Scalability:** The pipeline processes one known document synchronously. For high-volume or asynchronous processing, bounded concurrency, durable queues, and dead-letter handling could be introduced where appropriate.
