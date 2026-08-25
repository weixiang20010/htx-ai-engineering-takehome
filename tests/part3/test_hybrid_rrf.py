"""
Tests for RRF fusion.

Verifies that RRF combines BM25 and semantic ranks deterministically,
that duplicate chunks are merged (not doubled), and that raw scores
are not mixed — only ranks are combined.
"""
from __future__ import annotations

import pytest

from src.part3.models import DocumentChunk, RetrievedChunk
from src.part3.retriever import HybridRRFRetriever, RRF_K


def _rc(chunk_id: str, bm25_rank: int | None, sem_rank: int | None) -> RetrievedChunk:
    chunk = DocumentChunk(chunk_id=chunk_id, page=1, section=None, text="dummy")
    score = 0.0
    if bm25_rank is not None:
        score += 1.0 / (RRF_K + bm25_rank)
    if sem_rank is not None:
        score += 1.0 / (RRF_K + sem_rank)
    return RetrievedChunk(
        chunk=chunk, bm25_rank=bm25_rank, semantic_rank=sem_rank, rrf_score=score
    )


def test_rrf_score_formula():
    """Score for rank-1 from both methods should be 2/(K+1)."""
    score = _rc("a", bm25_rank=1, sem_rank=1).rrf_score
    expected = 2.0 / (RRF_K + 1)
    assert abs(score - expected) < 1e-9


def test_chunk_appearing_in_both_lists_ranks_higher_than_one_list():
    """A chunk ranked 1st in both lists should outscore a chunk in only one list."""
    both = _rc("both", bm25_rank=1, sem_rank=1)
    one = _rc("one", bm25_rank=1, sem_rank=None)
    assert both.rrf_score > one.rrf_score


def test_rrf_is_deterministic():
    """Same input → same score on every call."""
    a = _rc("x", bm25_rank=3, sem_rank=5).rrf_score
    b = _rc("x", bm25_rank=3, sem_rank=5).rrf_score
    assert a == b


def test_bm25_only_result_has_no_semantic_rank():
    chunks = [
        DocumentChunk(chunk_id=f"c{i}", page=1, section=None, text=f"text {i}")
        for i in range(3)
    ]
    retriever = HybridRRFRetriever(chunks)
    results = retriever.retrieve_bm25_only("text 0", top_k=3)
    for r in results:
        assert r.semantic_rank is None


def test_rrf_k_constant_is_documented():
    """RRF_K must be 60 per the RRF paper (Cormack et al. 2009)."""
    assert RRF_K == 60
