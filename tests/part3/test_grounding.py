"""
Tests for Part 3 grounding validation.

All tests are deterministic — no LLM or API calls.
"""
from __future__ import annotations

import pytest

from src.part3.grounding import (
    ExtractionValidationError,
    convert_to_billions,
    validate_grounded_fact,
)
from src.part3.models import DocumentChunk, GroundedFact


def _chunk(page: int, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"p{page}", page=page, section=None, text=text)


# ---------------------------------------------------------------------------
# validate_grounded_fact
# ---------------------------------------------------------------------------


def test_valid_fact_passes():
    chunk = _chunk(
        5,
        "Corporate Income Tax is expected to be $28.4 billion in FY2024.",
    )
    fact = GroundedFact(
        claim="Corporate Income Tax is $28.4 billion",
        source_page=5,
        evidence_text="Corporate Income Tax is expected to be $28.4 billion in FY2024.",
        amount_text="28.4",
        source_unit="$ billion",
    )
    validate_grounded_fact(fact, [chunk])  # should not raise


def test_invented_evidence_raises():
    chunk = _chunk(5, "Corporate Income Tax is $28.4 billion.")
    fact = GroundedFact(
        claim="CIT is huge",
        source_page=5,
        evidence_text="This sentence was completely made up by the LLM.",
    )
    with pytest.raises(ExtractionValidationError):
        validate_grounded_fact(fact, [chunk])


def test_wrong_page_raises():
    chunk = _chunk(6, "Corporate Income Tax is $28.4 billion.")
    fact = GroundedFact(
        claim="CIT",
        source_page=5,  # wrong page
        evidence_text="Corporate Income Tax is $28.4 billion.",
    )
    with pytest.raises(ExtractionValidationError, match="No retrieved chunks found for page 5"):
        validate_grounded_fact(fact, [chunk])


def test_amount_not_in_evidence_raises():
    chunk = _chunk(20, "Future Energy Fund | 5,000")
    fact = GroundedFact(
        claim="Future Energy Fund amount",
        source_page=20,
        evidence_text="Future Energy Fund | 5,000",
        amount_text="9,999",  # wrong amount
    )
    with pytest.raises(ExtractionValidationError):
        validate_grounded_fact(fact, [chunk])


def test_valid_amount_passes():
    chunk = _chunk(20, "Future Energy Fund | 5,000")
    fact = GroundedFact(
        claim="Future Energy Fund",
        source_page=20,
        evidence_text="Future Energy Fund | 5,000",
        amount_text="5,000",
        source_unit="$ million",
    )
    validate_grounded_fact(fact, [chunk])  # should not raise


# ---------------------------------------------------------------------------
# convert_to_billions
# ---------------------------------------------------------------------------


def test_million_to_billion():
    assert convert_to_billions("5,000", "$ million") == pytest.approx(5.0)


def test_billion_passthrough():
    assert convert_to_billions("28.4", "$ billion") == pytest.approx(28.4)


def test_decimal_million_to_billion():
    assert convert_to_billions("500", "$ million") == pytest.approx(0.5)


def test_invalid_amount_returns_none():
    assert convert_to_billions("N/A", "$ million") is None


def test_unknown_unit_returns_raw():
    """Units other than million/billion return None (not guessed)."""
    assert convert_to_billions("100", "$ thousand") is None
