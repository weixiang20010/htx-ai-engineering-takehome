# HTX AI Engineering Take-Home — Part 1

**Document Extraction & Prompt Engineering with LangChain**

---

## Quick start

```bash
cp .env.example .env          # fill in GEMINI_API_KEY
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
| `GEMINI_MODEL` | Model name (default: `gemini-3.6-flash`) |
| `SOURCE_PDF` | Path to the data-source PDF (default: `data/fy2024_analysis_of_revenue_and_expenditure.pdf`) |

---

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
| B | 5–6 | `OperatingRevenueTaxes` (tax name list) | filter: keep only names present in source |
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
| LLM returns an unsupported tax name | `validate_taxes_in_source()` filters the returned list against the source text |
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
# All tests (integration tests require GEMINI_API_KEY)
python -m pytest tests/ -v

# Unit tests only (no API key needed)
python -m pytest tests/ -m "not integration" -v
```

---

## Assumptions

### Year labelling mismatch (fields 1 and 2)

The assessment asks for *"Corporate Income Tax in 2024"* and specifies **page 5** as the source.  However, page 5 of the source document is within **Section 01 — Update on Financial Year 2023**, which reports the *Revised FY2023* figures.  The Corporate Income Tax amount shown on page 5 is the revised FY2023 collection ($28.4 billion, 17.0% above the FY2023 estimate).

Per the assessment instruction *"treat the explicitly specified source page as authoritative"*, the values are extracted from page 5 as-is.  The assumption is documented here and in `outputs/part1_evidence.json`.

The source document was **not modified** to reconcile this discrepancy.

### Table 1.1 "Latest Actual" column

Table 1.1 on page 8 has three value columns: *Actual FY2022*, *Estimated FY2023*, and *Revised FY2023*.  "Latest Actual" is interpreted as the **Actual FY2022** column because it is the only column labelled "Actual" (the FY2023 figures are estimates or revisions, not finalized actuals at the time of publication).

### Top-up amounts unit

Table 2.4 on page 20 states amounts in **$ million**.  The extracted `total_top_ups_2024` float is therefore in millions (20352.0), consistent with the source unit retained in `part1_evidence.json`.

### pdfplumber `extract_tables()` failure on page 8

Page 8 returns zero tables from `extract_tables()` because Table 1.1 has no visible ruling lines.  The text extraction is used instead with a trailing-numeric-token parser.  This was verified to produce correct values for all tested rows including `OVERALL FISCAL POSITION`.
