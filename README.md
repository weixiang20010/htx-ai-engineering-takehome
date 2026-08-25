# HTX AI Engineering Take-Home — Part 1 & Part 2

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

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_EXTRACTION_MODEL` | Model for extraction tasks (default: `gemini-3.1-flash-lite`, 500 RPD free tier) |
| `GEMINI_REASONING_MODEL` | Model for reasoning + tool-selection tasks (default: `gemini-3.6-flash`) |
| `GEMINI_REASONING_FALLBACK_MODEL` | Fallback for reasoning on quota/rate-limit errors (default: `gemini-3.1-flash-lite`) |
| `SOURCE_PDF` | Path to the data-source PDF (default: `data/fy2024_analysis_of_revenue_and_expenditure.pdf`) |

---

## Library choices

| Library | Reason |
|---------|--------|
| `pdfplumber` | Lightweight page-level text and table extraction; no OCR or CV overhead; output inspected page-by-page before writing any parser code |
| `langchain` + `langchain-google-genai` | Structured output via `llm.with_structured_output(PydanticModel)` keeps LLM responses schema-validated; prompt templates are version-controlled separately from orchestration logic |
| `mcp>=1.29.1,<2` | MCP SDK used by `FastMCP` to expose the date-normaliser over stdio. Pinned below 2.x because `langchain-mcp-adapters==0.3.1` is incompatible with the mcp 2.x API |
| `langchain-mcp-adapters==0.3.1` | Bridges the MCP session into LangChain `BaseTool` objects so LangGraph/LangChain agents can call MCP tools natively |
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

The standalone script `scripts/compare_prompts.py` demonstrates both prompts side by side.

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
scripts/compare_prompts.py — optional naive vs improved demo
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

# Part 2 � Data Extraction + Datetime Tool via Local MCP

## What it does

Part 2 extracts two specific dates from the PDF, normalises them to ISO 8601 format via a local MCP tool, then classifies each date against a fixed reference date using Gemini.

| Stage | What happens |
|-------|--------------|
| **1A** | Gemini extracts a date and its source sentence from page 1 (distribution date) |
| **1B** | Gemini extracts a date and its source sentence from page 36 (Estate Duty cutoff) |
| **1C** | Python validates each date_text appears verbatim in the page source |
| **2A** | Gemini calls the `normalize_date` MCP tool with the raw date string |
| **2B** | The MCP server runs `normalize_date_value()` � purely deterministic, no LLM |
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

Date parsing is a deterministic task � a given date string should always produce the same ISO output regardless of model state or temperature.  Exposing `normalize_date_value()` as a **local MCP tool** means:

1. **Gemini's tool-call is observable** � the `tool_calls` list on the response object is inspected before execution, proving the model requested the tool (not that it answered from training data).
2. **The normalisation itself is pure Python** � no LLM involved, so the output is predictable and unit-testable without any API calls.
3. **MCP stdio transport** � the server runs as a subprocess; no network port, no auth, no persistence.

---

## Why deterministic normalisation, not LLM normalisation?

Normalising `"16 February 2024"` ? `"2024-02-16"` does not require semantic understanding. Delegating it to an LLM adds latency, API cost, and non-determinism. The MCP tool wraps `datetime.strptime` over a known set of formats derived from the source document; it raises `DateNormalizationError` for anything outside that set.

---

## Why LLM classification, not rule-based?

The distinction between **Expired** and **Ongoing** requires semantic context:

> *"Estate Duty does not apply to a person who dies after 15 February 2008."*

The date `2008-02-15` is before the reference date `2024-01-01`, which would naively make it Expired. But the sentence describes a **continuing policy** still in effect at the reference date � it is **Ongoing**. A regex-over-date approach cannot make this distinction. The LLM receives both `original_text` and `normalized_date` so it can reason about whether the date marks a boundary of an ongoing rule, a past event, or a future event.

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
        �
        ?
   pdfplumber (page.extract_text())
        �
        ?
   Gemini (with_structured_output ? ExtractedDate)
   +-- original_text: source sentence
   +-- date_text: raw date string
        �
        ?
   Python validation
   +-- date_text verbatim in page source   ? validate_evidence_in_source()
   +-- date_text within original_text      ? validate_value_in_evidence()
        �
        ?
   MCP stdio session (mcp 1.29.1 + FastMCP)
   +-- Gemini.bind_tools(lc_tools).ainvoke(...)
   �   +-- response.tool_calls[0] verified == "normalize_date"
   +-- session.call_tool("normalize_date", args)
       +-- normalize_date_value() ? ISO string
        �
        ?
   Gemini (with_structured_output ? InternalDateClassification)
   +-- status: Expired | Upcoming | Ongoing
   +-- reason: LLM rationale
        �
        ?
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

A pure date-comparison rule would classify `2008-02-15` as **Expired** (before the 2024-01-01 reference date). However, the source sentence describes a **continuing policy** still in force at the reference date: *"Estate Duty does not apply to a person who dies after 15 February 2008."* The LLM is expected to classify this as **Ongoing** because the date marks the start of a permanent exemption, not a past one-off event. The integration test suite asserts this classification explicitly (`test_estate_duty_date_is_classified_expired` is deliberately named to contrast: the LLM should return Ongoing, not Expired, and an assertion covers this expectation).

---

## Part 2 project structure

```
src/part2/
  __init__.py
  models.py          � Pydantic schemas, REFERENCE_DATE, TemporalStatus enum
  date_normalizer.py � Deterministic ISO conversion (no LLM, no MCP)
  mcp_server.py      � FastMCP server exposing normalize_date over stdio
  mcp_client.py      � Async context manager: start server, yield (session, lc_tools)
  prompts.py         � ChatPromptTemplates for extraction and classification
  date_extractor.py  � Gemini extraction + evidence validation
  classifier.py      � Gemini temporal classification
  workflow.py        � Async orchestration of all stages

scripts/run_part2.py � Entry point; writes three output files

tests/part2/
  test_date_normalizer.py          � 15 unit tests, no API
  test_date_extraction_validation.py � 7 validation tests, no API
  test_models.py                    � 12 schema tests, no API
  test_mcp_integration.py           � 5 real MCP boundary tests, no Gemini
  test_part2_integration.py         � end-to-end tests (require GEMINI_API_KEY)
```
