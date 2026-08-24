"""
pdfplumber-based PDF extraction for Part 1.

Responsibilities
----------------
- Map printed page numbers to physical PDF page indices.
- Extract narrative text (pages 5–6).
- Extract structured tables (pages 8 and 20) as ParsedTable objects.
- Serialise tables to a human-readable text format for LLM prompts.

This module converts raw PDF content into structured Python objects.
Semantic interpretation of that content is the responsibility of the LLM layer.

Page-number mapping
-------------------
Verified by inspection: the source PDF prints page number N on physical page
index N-1 (0-based).  The mapping is centralised in ``page_index()``; no other
code should hardcode physical indices.

Table extraction notes
----------------------
Page 8 (Table 1.1):
    ``extract_tables()`` returns 0 tables because the table has no ruling lines
    — pdfplumber cannot detect the boundaries automatically.  We fall back to
    parsing the text output, which contains the table data row-by-row with
    "BLANK" tokens marking empty header cells.  Column headers are reconstructed
    from the known multi-line header structure verified during PDF inspection.

Page 20 (Table 2.4):
    ``extract_tables()`` succeeds but returns sparse rows with many ``None``
    entries because pdfplumber detects more column boundaries than the table
    actually has.  We clean each row by dropping ``None`` and empty cells.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber


# ---------------------------------------------------------------------------
# Page number mapping
# ---------------------------------------------------------------------------


def page_index(printed_page: int) -> int:
    """
    Convert a 1-based printed page number to a 0-based physical page index.

    Verified against source PDF: printed page 4 is at physical index 3,
    confirming the relationship  ``physical_index = printed_page - 1``.
    """
    if printed_page < 1:
        raise ValueError(f"Page numbers are 1-based; got {printed_page}")
    return printed_page - 1


# ---------------------------------------------------------------------------
# ParsedTable
# ---------------------------------------------------------------------------


@dataclass
class ParsedTable:
    """
    Structured table extracted from a PDF page.

    Attributes
    ----------
    title:
        Human-readable table title (from the PDF).
    source_page:
        Printed page number the table was extracted from.
    headers:
        Ordered column labels, excluding the row-label column (column 0).
    rows:
        Mapping of ``row_label -> [value_col0, value_col1, …]`` in the same
        order as *headers*.  Missing cells are ``None``.
    """

    title: str
    source_page: int
    headers: list[str]
    rows: dict[str, list[str | None]]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_cell(self, row_label: str, column_label: str) -> str:
        """
        Return the raw string value at (*row_label*, *column_label*).

        Matching is case-insensitive and whitespace-normalised (substring
        matching in both directions to handle minor label variations between
        the LLM output and the parsed table).

        Raises
        ------
        ExtractionValidationError
            If the row or column label is not found, or the cell is ``None``.
        """
        from .validators import ExtractionValidationError  # local to avoid circular import

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip()).lower()

        # Locate column index.
        # Use one-directional substring check: the supplied label must be a
        # substring of the stored header (LLM may return a shorter label).
        # The reverse direction is intentionally excluded to prevent "Actual
        # FY2022" from matching "Compared to Actual FY2022".
        col_idx: int | None = None
        for i, h in enumerate(self.headers):
            if _norm(column_label) in _norm(h):
                col_idx = i
                break
        if col_idx is None:
            raise ExtractionValidationError(
                f"Column {column_label!r} not found in table headers.\n"
                f"  Available: {self.headers}"
            )

        # Locate row
        matched_values: list[str | None] | None = None
        for label, vals in self.rows.items():
            if _norm(row_label) in _norm(label) or _norm(label) in _norm(row_label):
                matched_values = vals
                break
        if matched_values is None:
            available = list(self.rows.keys())[:12]
            raise ExtractionValidationError(
                f"Row {row_label!r} not found in table.\n"
                f"  Available (first 12): {available}"
            )

        if col_idx >= len(matched_values) or matched_values[col_idx] is None:
            raise ExtractionValidationError(
                f"Cell ({row_label!r}, {column_label!r}) is None or out of range.\n"
                f"  Row values: {matched_values}"
            )
        return matched_values[col_idx]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # LLM serialisation
    # ------------------------------------------------------------------

    def to_llm_text(self) -> str:
        """
        Render the table as a pipe-delimited plain-text block suitable for
        inclusion in an LLM prompt.
        """
        header_line = "Row Label | " + " | ".join(self.headers)
        separator = "-" * min(len(header_line), 100)
        lines = [
            f"Table: {self.title} (source page {self.source_page})",
            header_line,
            separator,
        ]
        for label, vals in self.rows.items():
            cells = [v if v is not None else "-" for v in vals]
            lines.append(f"{label} | " + " | ".join(cells))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Narrative text extraction
# ---------------------------------------------------------------------------


def get_page_text(pdf_path: str | Path, printed_page: int) -> str:
    """
    Return the full text content of a single page.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    printed_page:
        1-based page number as printed in the document.
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        idx = page_index(printed_page)
        if idx >= len(pdf.pages):
            raise IndexError(
                f"Page {printed_page} (index {idx}) is out of range "
                f"(PDF has {len(pdf.pages)} pages)."
            )
        return pdf.pages[idx].extract_text() or ""


def get_pages_text(pdf_path: str | Path, printed_pages: list[int]) -> dict[int, str]:
    """
    Return a mapping of ``{printed_page: text}`` for the requested pages.

    Opens the PDF once to avoid repeated I/O.
    """
    result: dict[int, str] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pn in printed_pages:
            idx = page_index(pn)
            result[pn] = pdf.pages[idx].extract_text() or ""
    return result


# ---------------------------------------------------------------------------
# Table extraction — page 8
# ---------------------------------------------------------------------------

# Column headers for Table 1.1 (Fiscal Position in FY2022 and FY2023).
# Reconstructed from the multi-line BLANK-labelled header rows verified
# during PDF inspection.  These label the data columns (index 0 = Actual FY2022).
_PAGE8_HEADERS: list[str] = [
    "Actual FY2022",
    "Estimated FY2023",
    "Revised FY2023",
    "Compared to Actual FY2022",
    "Compared to Estimated FY2023",
]

# Lines whose labels match these prefixes are section markers, not data rows.
_SKIP_LABELS: frozenset[str] = frozenset({"Add", "Less"})

# Lines starting with these strings mark the end of the table data area.
_TABLE_END_MARKERS: tuple[str, ...] = (
    "MINISTRY OF FINANCE",
    "Note:",
    "1 Other Taxes",
    "2 Special Transfers",
    "3 Includes",
    "4 Consists",
    "5 Consists",
    "6 SINGA",
)

# Matches a single numeric token (integer or decimal, optionally parenthesised
# for negatives or prefixed with a minus sign).
_NUMERIC_TOKEN_RE = re.compile(r"^[\(\-]?\d[\d,.]*\)?$")


def _is_numeric_token(token: str) -> bool:
    """True if *token* represents a standalone number."""
    return bool(_NUMERIC_TOKEN_RE.match(token.strip()))


def _is_garbled_coordinate_line(line: str) -> bool:
    """
    Detect lines that contain embedded chart-coordinate data artefacts.

    The source PDF has a chart element on page 8 whose coordinates bleed into
    the text stream as comma-separated numbers with a trailing comma and space,
    e.g. "Add: 22,376, 23,480,570, 22,915, 0".  These are not table rows.
    """
    return bool(re.search(r"\d,\s+\d", line))


def _split_label_values(line: str) -> tuple[str, list[str]] | None:
    """
    Split a table text line into ``(row_label, [value, …])``.

    Values are the trailing numeric tokens on the line; everything before them
    is the row label.  Returns ``None`` for non-data lines (headers, section
    markers, garbled coordinate lines, lines with no numeric values).
    """
    if _is_garbled_coordinate_line(line):
        return None

    tokens = line.strip().split()
    if not tokens:
        return None

    values: list[str] = []
    label_parts = list(tokens)

    while label_parts and _is_numeric_token(label_parts[-1]):
        values.insert(0, label_parts.pop())

    if not label_parts or not values:
        return None

    label = " ".join(label_parts)
    if label.rstrip(":").strip() in _SKIP_LABELS:
        return None

    return label, values


def extract_table_page8(pdf_path: str | Path) -> ParsedTable:
    """
    Extract Table 1.1 (Fiscal Position in FY2022 and FY2023) from page 8.

    Because ``extract_tables()`` returns no results for this page (the table
    has no ruling lines), we parse the flat ``extract_text()`` output using a
    trailing-numeric-token heuristic and a known column schema.
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index(8)]
        text = page.extract_text() or ""

    lines = text.splitlines()
    n_cols = len(_PAGE8_HEADERS)

    # Skip past the multi-line BLANK header section; data rows start after the
    # last line that contains the "BLANK" accessibility token.
    data_start = 0
    for i, line in enumerate(lines):
        if "BLANK" in line:
            data_start = i + 1

    rows: dict[str, list[str | None]] = {}

    for line in lines[data_start:]:
        stripped = line.strip()
        if any(stripped.startswith(m) for m in _TABLE_END_MARKERS):
            break

        result = _split_label_values(stripped)
        if result is None:
            continue

        label, values = result
        # Pad to n_cols so every row has the same width
        padded: list[str | None] = (values + [None] * n_cols)[:n_cols]
        rows[label] = padded

    return ParsedTable(
        title="Fiscal Position in FY2022 and FY2023",
        source_page=8,
        headers=_PAGE8_HEADERS,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Table extraction — page 20
# ---------------------------------------------------------------------------


def _clean_sparse_row(row: list[Any]) -> list[str]:
    """Return non-None, non-empty, stripped cell values from a pdfplumber row."""
    return [str(c).strip() for c in row if c is not None and str(c).strip()]


def extract_table_page20(pdf_path: str | Path) -> ParsedTable:
    """
    Extract Table 2.4 (Top-ups to Endowment and Trust Funds in FY2024)
    from page 20.

    ``extract_tables()`` succeeds on this page but returns rows with many
    ``None``-filled intermediate columns.  We clean each row by keeping only
    non-None non-empty cells, which reduces the table to its two meaningful
    columns: row label and amount.
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index(20)]
        raw_tables = page.extract_tables()

    if not raw_tables:
        raise RuntimeError("No tables found on page 20 — check pdfplumber version.")

    raw = raw_tables[0]
    cleaned: list[list[str]] = [_clean_sparse_row(row) for row in raw]
    cleaned = [r for r in cleaned if r]  # drop fully-empty rows

    # Row 0: column header  e.g. ["Estimated FY2024"]
    # Row 1: unit info      e.g. ["($ million)"]
    # Row 2+: data rows     e.g. ["Goods and Services Tax Voucher Fund", "6,000"]
    if len(cleaned) < 3:
        raise RuntimeError(
            f"Page 20 table has too few rows after cleaning: {cleaned}"
        )

    col_header = cleaned[0][0] if cleaned[0] else "Estimated FY2024"

    rows: dict[str, list[str | None]] = {}
    for row in cleaned[2:]:  # skip column-header and unit rows
        if len(row) >= 2:
            rows[row[0]] = [row[1]]
        elif len(row) == 1:
            # Occasionally the label and value are in separate cleaned rows
            # for very long labels; skip single-token rows as incomplete.
            pass

    return ParsedTable(
        title="Top-ups to Endowment and Trust Funds in FY2024",
        source_page=20,
        headers=[col_header],
        rows=rows,
    )
