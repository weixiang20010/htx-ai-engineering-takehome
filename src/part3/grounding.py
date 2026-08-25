"""
Deterministic grounding validation for Part 3 GroundedFact objects.

Every claim returned by a specialist agent must be traceable to a verbatim
sentence in the source PDF.  Validation re-uses the whitespace-normalisation
logic from Part 1 (validate_evidence_in_source / validate_value_in_evidence)
so the same grounding contract is enforced throughout the project.

Numeric conversion
------------------
When a source table uses "$ million" as the unit and the answer context
requires billions, convert_to_billions() performs the conversion
deterministically in Python instead of asking the LLM to do arithmetic.
"""
from __future__ import annotations

import re

from ..part1.validators import ExtractionValidationError, validate_evidence_in_source
from .models import DocumentChunk, GroundedFact

# Re-export so callers can import from grounding without knowing Part 1 internals.
__all__ = [
    "ExtractionValidationError",
    "validate_grounded_fact",
    "convert_to_billions",
]


def validate_grounded_fact(
    fact: GroundedFact,
    source_chunks: list[DocumentChunk],
) -> None:
    """
    Verify that *fact* is grounded in the retrieved source chunks.

    Checks:
    1. evidence_text appears verbatim (whitespace-normalised) in at least
       one chunk whose page matches fact.source_page.
    2. If amount_text is provided, it appears within evidence_text.

    Raises
    ------
    ExtractionValidationError
        If either check fails.
    """
    # Build a combined text for the claimed source page.
    page_text = " ".join(
        c.text for c in source_chunks if c.page == fact.source_page
    )
    if not page_text:
        raise ExtractionValidationError(
            f"No retrieved chunks found for page {fact.source_page} "
            f"(fact claim: {fact.claim!r})"
        )

    # Check evidence_text is in source.
    validate_evidence_in_source(
        fact.evidence_text,
        page_text,
        field_name=f"fact '{fact.claim}'",
    )

    # If amount_text is specified, check it appears in evidence_text.
    if fact.amount_text:
        norm_amount = re.sub(r"\s+", " ", fact.amount_text.strip())
        norm_evidence = re.sub(r"\s+", " ", fact.evidence_text.strip())
        if norm_amount not in norm_evidence:
            raise ExtractionValidationError(
                f"amount_text {fact.amount_text!r} not found in evidence_text "
                f"for fact {fact.claim!r}.\n"
                f"  evidence: {norm_evidence[:120]!r}"
            )


_NUMERIC_RE = re.compile(r"[\d,]+(\.\d+)?")


def convert_to_billions(amount_text: str, source_unit: str) -> float | None:
    """
    Convert an amount string to billions based on its source unit.

    Parameters
    ----------
    amount_text:
        Raw number string as it appears in the document (e.g. "5,000" or "28.4").
    source_unit:
        Unit string (e.g. "$ million", "$ billion").

    Returns
    -------
    Numeric value in billions, or None if conversion is not applicable.

    Examples
    --------
    >>> convert_to_billions("5,000", "$ million")
    5.0
    >>> convert_to_billions("28.4", "$ billion")
    28.4
    """
    match = _NUMERIC_RE.search(amount_text.replace(",", ""))
    if not match:
        return None
    raw = float(match.group().replace(",", ""))
    unit_lower = source_unit.lower()
    if "million" in unit_lower:
        return raw / 1_000.0
    if "billion" in unit_lower:
        return raw
    return None
