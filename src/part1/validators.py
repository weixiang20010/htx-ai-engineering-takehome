"""
Deterministic validation of LLM output against source document content.

Every function in this module operates on plain Python strings and structures
returned by the PDF parser — no LLM calls are made here.  Validation failures
raise ExtractionValidationError rather than returning a fallback value; the
caller decides how to surface or suppress the error.
"""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pdf_parser import ParsedTable


class ExtractionValidationError(Exception):
    """
    Raised when LLM-generated output cannot be grounded in the source document.

    Prefer this over returning a fallback value: a missing financial figure is
    safer than an invented one.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm_ws(text: str) -> str:
    """Normalise text for source-grounding comparison.

    Steps applied in order:
    1. NFKC Unicode normalisation (handles non-breaking spaces, ligatures, etc.)
    2. Join soft line-break hyphens: pdfplumber renders "stronger-\\nthan" as
       "stronger- than" but the LLM correctly reconstructs "stronger-than".
    3. Collapse all remaining whitespace runs to a single space and strip edges.
    """
    text = unicodedata.normalize("NFKC", text)
    # Remove the space that pdfplumber inserts after a line-continuation hyphen
    text = re.sub(r"-\s+([a-zA-Z])", r"-\1", text)
    return re.sub(r"\s+", " ", text.strip())


# ---------------------------------------------------------------------------
# Narrative evidence validators
# ---------------------------------------------------------------------------


def validate_evidence_in_source(
    evidence_text: str,
    source_text: str,
    field_name: str = "evidence",
) -> None:
    """
    Verify *evidence_text* appears verbatim in *source_text* (after
    whitespace normalisation).

    Raises
    ------
    ExtractionValidationError
        If the evidence string is absent from the source.
    """
    norm_ev = _norm_ws(evidence_text)
    if not norm_ev:
        raise ExtractionValidationError(
            f"Evidence for {field_name!r} is empty after normalisation."
        )
    norm_src = _norm_ws(source_text)
    if norm_ev not in norm_src:
        raise ExtractionValidationError(
            f"Evidence for {field_name!r} not found in source.\n"
            f"  evidence : {norm_ev[:140]!r}\n"
            f"  source   : {norm_src[:140]!r}"
        )


def validate_value_in_evidence(
    value_text: str,
    evidence_text: str,
    field_name: str = "value",
) -> None:
    """
    Verify *value_text* appears verbatim within *evidence_text* (after
    whitespace normalisation).

    Raises
    ------
    ExtractionValidationError
        If *value_text* is absent or empty.
    """
    if not value_text or not value_text.strip():
        raise ExtractionValidationError(
            f"Value text for {field_name!r} is None or empty."
        )
    norm_val = _norm_ws(value_text)
    norm_ev = _norm_ws(evidence_text)
    if norm_val not in norm_ev:
        raise ExtractionValidationError(
            f"Value {norm_val!r} not found in evidence {norm_ev[:140]!r}."
        )


def validate_taxes_in_source(taxes: list[str], source_text: str) -> list[str]:
    """
    Filter *taxes* to only those present in *source_text* (whitespace-normalised).

    Raises
    ------
    ExtractionValidationError
        If none of the returned tax names are supported by the source.
    """
    norm_src = _norm_ws(source_text)
    supported = [t for t in taxes if _norm_ws(t) in norm_src]
    if not supported:
        raise ExtractionValidationError(
            "No returned tax names were found in the source text. "
            f"Returned: {taxes}"
        )
    return supported


# ---------------------------------------------------------------------------
# Table cell validator
# ---------------------------------------------------------------------------


def validate_table_cell(
    table: "ParsedTable",
    row_label: str,
    column_label: str,
) -> str:
    """
    Retrieve the raw cell value at (*row_label*, *column_label*) from *table*.

    Delegates to ``ParsedTable.get_cell``, which performs fuzzy label matching
    and raises ExtractionValidationError on a miss.
    """
    return table.get_cell(row_label, column_label)
