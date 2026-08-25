"""
PDF chunking for Part 3 retrieval.

Strategy
--------
Rather than splitting every N characters, the chunker tries to produce
meaningful units that preserve semantic context:

  - Narrative pages (mostly text): section + paragraph chunks.
  - Table-heavy pages: one chunk per table block, serialised as
    "Title | Unit | Row | Amount" so the fund name and figure stay together.
  - Statistical annex (pages 23-34): paragraph + table blocks.
  - Cover / explanatory notes / glossary: skipped or kept as single chunks.

Section headings are detected by patterns common in Singapore MOF documents
(e.g. "1.2 Operating Revenue", "02 Outlook for Financial Year 2024",
"Table 2.4"). A new chunk is started at each heading.

Within a section, text is split at blank lines to yield paragraph-level
chunks, each carrying the nearest ancestor section heading as metadata.
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from .models import DocumentChunk

# Pages to skip entirely (cover, explanatory notes, table of contents).
_SKIP_PAGES: frozenset[int] = frozenset({1, 2, 3})

# Regex that matches lines identifying section or table headings.
_HEADING_RE = re.compile(
    r"^("
    r"\d{2}\s+[A-Z]"          # "01 Update …", "02 Outlook …"
    r"|[12]\.\d+\s+[A-Z]"     # "1.2 Operating Revenue"
    r"|2\.\d+\s+[A-Z]"        # "2.3 Total Expenditure"
    r"|Table\s+\d+\.\d+"      # "Table 1.1", "Table 2.4"
    r"|Chart\s+\d+\.\d+"      # "Chart 1.1"
    r")"
)

_MIN_CHUNK_CHARS = 40   # discard trivially short fragments
_MAX_CHUNK_CHARS = 900  # hard-split overly long paragraphs at sentence boundary


def _strip_footer(text: str) -> str:
    """Remove 'MINISTRY OF FINANCE <n>' footers injected by pdfplumber."""
    return re.sub(r"\nMINISTRY OF FINANCE\s+\d+\s*", " ", text).strip()


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks ≤ max_chars, preferring sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    # Try to split at ". " boundary near the midpoint.
    mid = len(text) // 2
    split_pos = text.rfind(". ", 0, mid + 100)
    if split_pos == -1 or split_pos < 20:
        split_pos = mid
    return [text[: split_pos + 1].strip(), text[split_pos + 1 :].strip()]


def _make_id(page: int, section: str | None, seq: int) -> str:
    slug = re.sub(r"\W+", "_", (section or "").lower())[:30].strip("_")
    return f"p{page:02d}_{slug}_{seq:03d}" if slug else f"p{page:02d}_{seq:03d}"


def _chunks_from_page(
    page_num: int, text: str, section_hint: str | None = None
) -> list[DocumentChunk]:
    """Convert a single page's text into DocumentChunks."""
    text = _strip_footer(text)
    if not text.strip():
        return []

    chunks: list[DocumentChunk] = []
    current_section = section_hint
    current_lines: list[str] = []
    seq = 0

    def _flush(lines: list[str], section: str | None) -> None:
        nonlocal seq
        block = "\n".join(lines).strip()
        if len(block) < _MIN_CHUNK_CHARS:
            return
        for part in _hard_split(block, _MAX_CHUNK_CHARS):
            part = part.strip()
            if len(part) >= _MIN_CHUNK_CHARS:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_id(page_num, section, seq),
                        page=page_num,
                        section=section,
                        text=part,
                    )
                )
                seq += 1

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _HEADING_RE.match(line):
            # Flush the previous block before starting the new section.
            _flush(current_lines, current_section)
            current_section = line
            current_lines = [line]
        elif line == "":
            # Blank line: treat as paragraph separator.
            _flush(current_lines, current_section)
            current_lines = []
        else:
            current_lines.append(line)

    _flush(current_lines, current_section)
    return chunks


def _serialize_table(
    title: str, headers: list[str], rows: dict[str, list[str | None]], unit: str | None
) -> str:
    """Render a table as a plain-text block suitable for retrieval."""
    lines = [f"Table: {title}"]
    if unit:
        lines.append(f"Unit: {unit}")
    lines.append("Row | " + " | ".join(headers))
    for label, vals in rows.items():
        cells = " | ".join(v if v else "-" for v in (vals or []))
        lines.append(f"{label} | {cells}")
    return "\n".join(lines)


def _extract_page20_table(pdf_path: str | Path) -> list[DocumentChunk]:
    """
    Extract Table 2.4 (Top-ups to Endowment and Trust Funds in FY2024).

    pdfplumber can extract this table successfully on page 20.
    """
    chunks: list[DocumentChunk] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[19]  # 0-indexed → page 20
        raw_text = _strip_footer(page.extract_text() or "")
        tables = page.extract_tables() or []

    if tables:
        # Take the largest table (most rows).
        table = max(tables, key=len)
        if len(table) > 1:
            headers = [str(h).strip() for h in (table[0] or []) if h]
            unit: str | None = None
            unit_match = re.search(r"\(\$\s*million\)", raw_text, re.IGNORECASE)
            if unit_match:
                unit = "$ million"

            rows: dict[str, list[str | None]] = {}
            for row in table[1:]:
                cells = [str(c).strip() if c else None for c in row]
                label = cells[0] if cells else None
                if label:
                    rows[label] = cells[1:]

            table_text = _serialize_table(
                "Top-ups to Endowment and Trust Funds in FY2024",
                headers[1:] if len(headers) > 1 else headers,
                rows,
                unit,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id="p20_table_2_4_000",
                    page=20,
                    section="2.5 Top-ups to Endowment and Trust Funds",
                    text=table_text,
                )
            )

    # Also include the narrative text from page 20 as a separate chunk.
    for chunk in _chunks_from_page(20, raw_text, "2.5 Top-ups to Endowment and Trust Funds"):
        if chunk.chunk_id == "p20_table_2_4_000":
            continue  # avoid duplicate id (shouldn't happen but guard it)
        chunks.append(chunk)

    return chunks


def build_chunks(pdf_path: str | Path) -> list[DocumentChunk]:
    """
    Parse the source PDF and return a list of DocumentChunks for retrieval.

    The corpus covers pages 4-22 (main analysis) plus pages 23-34
    (statistical annex tables), with page 20's Table 2.4 handled specially.
    """
    pdf_path = Path(pdf_path)
    all_chunks: list[DocumentChunk] = []
    carrying_section: str | None = None

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)

    for printed_page in range(4, min(total_pages + 1, 35)):
        if printed_page in _SKIP_PAGES:
            continue

        if printed_page == 20:
            page20_chunks = _extract_page20_table(pdf_path)
            all_chunks.extend(page20_chunks)
            # Keep the last section heading as context for subsequent pages.
            for c in page20_chunks:
                if c.section:
                    carrying_section = c.section
            continue

        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[printed_page - 1]
            text = _strip_footer(page.extract_text() or "")

        page_chunks = _chunks_from_page(printed_page, text, carrying_section)
        all_chunks.extend(page_chunks)

        # Carry the last known section into the next page (sections often span pages).
        for c in reversed(page_chunks):
            if c.section:
                carrying_section = c.section
                break

    return all_chunks
