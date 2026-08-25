"""
Tests for BM25 retrieval.

Verifies that exact-term queries retrieve the expected chunks without
any LLM or API calls.
"""
from __future__ import annotations

import pytest

from src.part3.models import DocumentChunk
from src.part3.retriever import HybridRRFRetriever


def _make_chunks(texts: list[str], page: int = 1) -> list[DocumentChunk]:
    return [
        DocumentChunk(chunk_id=f"c{i}", page=page, section=None, text=t)
        for i, t in enumerate(texts)
    ]


@pytest.fixture()
def retriever():
    chunks = _make_chunks(
        [
            "Corporate Income Tax is the largest revenue source in FY2024.",
            "The Future Energy Fund received a top-up of 5,000 million dollars.",
            "Personal Income Tax collections were revised upward.",
            "Operating Revenue includes taxes and non-tax items.",
            "Total Expenditure comprises operating and development spending.",
        ]
    )
    return HybridRRFRetriever(chunks)


def test_exact_term_corporate_income_tax(retriever):
    results = retriever.retrieve_bm25_only("Corporate Income Tax", top_k=3)
    top = results[0]
    assert "Corporate Income Tax" in top.chunk.text


def test_exact_term_future_energy_fund(retriever):
    results = retriever.retrieve_bm25_only("Future Energy Fund", top_k=3)
    assert any("Future Energy Fund" in r.chunk.text for r in results)


def test_bm25_ranks_start_at_1(retriever):
    results = retriever.retrieve_bm25_only("tax", top_k=5)
    ranks = [r.bm25_rank for r in results]
    assert 1 in ranks


def test_bm25_returns_at_most_top_k(retriever):
    results = retriever.retrieve_bm25_only("revenue", top_k=2)
    assert len(results) <= 2


def test_bm25_semantic_rank_is_none(retriever):
    results = retriever.retrieve_bm25_only("revenue", top_k=3)
    for r in results:
        assert r.semantic_rank is None
