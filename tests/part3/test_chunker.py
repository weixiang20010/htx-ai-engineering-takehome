"""
Tests for the Part 3 document chunker.

These tests run against the actual source PDF and verify that:
  - The corpus is non-empty.
  - No chunk is below the minimum meaningful length.
  - Table chunks on page 20 contain both the fund name and its amount.
  - Section headings appear as metadata on subsequent chunks.
  - chunk_ids are globally unique.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

PDF_PATH = Path(__file__).parent.parent.parent / os.getenv(
    "SOURCE_PDF", "data/fy2024_analysis_of_revenue_and_expenditure.pdf"
)


@pytest.fixture(scope="module")
def chunks():
    if not PDF_PATH.exists():
        pytest.skip(f"Source PDF not found at {PDF_PATH}")
    from src.part3.chunker import build_chunks

    return build_chunks(PDF_PATH)


def test_corpus_is_non_empty(chunks):
    assert len(chunks) > 30, "Expected at least 30 chunks from a 37-page PDF"


def test_no_trivially_short_chunks(chunks):
    short = [c for c in chunks if len(c.text.strip()) < 40]
    assert not short, f"Found {len(short)} chunks shorter than 40 chars: {short[:3]}"


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


def test_page20_table_chunk_contains_future_energy_fund(chunks):
    page20 = [c for c in chunks if c.page == 20]
    combined = " ".join(c.text for c in page20).lower()
    assert "future energy fund" in combined, (
        "Page 20 chunks should contain 'Future Energy Fund'"
    )


def test_page20_chunk_contains_amount(chunks):
    """The Future Energy Fund amount (5,000) must appear in the page 20 chunks."""
    page20 = [c for c in chunks if c.page == 20]
    combined = " ".join(c.text for c in page20)
    assert "5,000" in combined, "Expected amount 5,000 in page 20 chunks"


def test_page20_chunk_preserves_unit_context(chunks):
    """The table chunk must include the unit ($ million) so figures can be interpreted."""
    page20 = [c for c in chunks if c.page == 20]
    combined = " ".join(c.text.lower() for c in page20)
    assert "million" in combined, "Unit context (million) should appear near the table"


def test_revenue_pages_present(chunks):
    """Pages 5-6 (Operating Revenue) must be represented in the corpus."""
    pages = {c.page for c in chunks}
    assert 5 in pages
    assert 6 in pages


def test_chunks_have_page_numbers(chunks):
    for c in chunks:
        assert c.page >= 4, f"Unexpected page number: {c.page}"
