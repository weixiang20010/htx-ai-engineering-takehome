"""
Hybrid retrieval for Part 3: BM25 + semantic embeddings + Reciprocal Rank Fusion.

Architecture
------------

    query ──┬── BM25 (rank_bm25.BM25Okapi) ──────────┐
            │                                         │
            └── Semantic (GoogleGenerativeAI           │
                          gemini-embedding-001)        │
                    cosine similarity ────────────────┘
                                                      │
                                              RRF fusion (K=60)
                                                      │
                                              Top-K RetrievedChunks

RRF is preferred over direct score combination because BM25 scores and
cosine similarities have incompatible scales. RRF normalises both to ranks
before combining.

RRF score formula (from Cormack et al. 2009):
    score(d) = Σ  1 / (K + rank_i(d))
              i ∈ {bm25, semantic}

K = 60 is the standard value from the original RRF paper; it bounds the
influence of very high-ranked documents and prevents any single method from
dominating purely on rank position.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
from rank_bm25 import BM25Okapi

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

# The standard RRF constant (Cormack et al. 2009).
RRF_K: int = 60


class HybridRRFRetriever:
    """
    In-memory hybrid retriever: BM25 + semantic (Gemini gemini-embedding-001) + RRF.

    Usage
    -----
    retriever = HybridRRFRetriever(chunks)
    await retriever.build_semantic_index(api_key)   # one-time API call
    results = await retriever.retrieve(query)
    """

    _TOP_K_INITIAL: ClassVar[int] = 20  # wider pool before RRF fusion
    _TOP_K_DEFAULT: ClassVar[int] = 8   # chunks returned to the agent

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks
        self._ids = [c.chunk_id for c in chunks]
        self._texts = [c.text for c in chunks]
        self._id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(self._ids)}

        # BM25 index — built synchronously, no API calls required.
        tokenized = [t.lower().split() for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("[Part3] BM25 index built for %d chunks", len(chunks))

        # Semantic index — built lazily via build_semantic_index().
        self._embedding_model: GoogleGenerativeAIEmbeddings | None = None
        self._embedding_matrix: np.ndarray | None = None  # (n_chunks, emb_dim)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    async def build_semantic_index(self, api_key: str) -> None:
        """
        Embed all chunks and store the embedding matrix in memory.

        Requires a Gemini API key and ~1 API call per 100 chunks.
        """
        self._embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        logger.info(
            "[Part3] Building semantic index for %d chunks (gemini-embedding-001)…",
            len(self._texts),
        )
        embeddings = await self._embedding_model.aembed_documents(self._texts)
        self._embedding_matrix = np.array(embeddings, dtype=np.float32)
        logger.info("[Part3] Semantic index built — shape %s", self._embedding_matrix.shape)

    # ------------------------------------------------------------------
    # Individual retrievers
    # ------------------------------------------------------------------

    def _bm25_ranked(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, bm25_score) sorted by score descending."""
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    async def _semantic_ranked(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_similarity) sorted descending."""
        if self._embedding_model is None or self._embedding_matrix is None:
            raise RuntimeError("Call build_semantic_index() before semantic retrieval.")
        query_emb = np.array(
            await self._embedding_model.aembed_query(query), dtype=np.float32
        )
        # Normalise both query and corpus embeddings for cosine similarity.
        q_norm = query_emb / (np.linalg.norm(query_emb) or 1.0)
        row_norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        normed = self._embedding_matrix / row_norms
        similarities = normed @ q_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(i), float(similarities[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = _TOP_K_DEFAULT) -> list[RetrievedChunk]:
        """Hybrid RRF retrieval combining BM25 and semantic search."""
        pool = self._TOP_K_INITIAL
        bm25_results = self._bm25_ranked(query, pool)
        semantic_results = await self._semantic_ranked(query, pool)

        # Reciprocal Rank Fusion — sum 1/(K + rank) across both ranked lists.
        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(semantic_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        bm25_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        sem_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(semantic_results)}

        top = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        return [
            RetrievedChunk(
                chunk=self._chunks[idx],
                bm25_rank=bm25_rank_map.get(idx),
                semantic_rank=sem_rank_map.get(idx),
                rrf_score=score,
            )
            for idx, score in top
        ]

    def retrieve_bm25_only(self, query: str, top_k: int = _TOP_K_DEFAULT) -> list[RetrievedChunk]:
        """BM25-only retrieval (no API calls)."""
        results = self._bm25_ranked(query, top_k)
        return [
            RetrievedChunk(
                chunk=self._chunks[idx],
                bm25_rank=rank + 1,
                semantic_rank=None,
                rrf_score=score,
            )
            for rank, (idx, score) in enumerate(results)
        ]

    async def retrieve_semantic_only(
        self, query: str, top_k: int = _TOP_K_DEFAULT
    ) -> list[RetrievedChunk]:
        """Semantic-only retrieval."""
        results = await self._semantic_ranked(query, top_k)
        return [
            RetrievedChunk(
                chunk=self._chunks[idx],
                bm25_rank=None,
                semantic_rank=rank + 1,
                rrf_score=score,
            )
            for rank, (idx, score) in enumerate(results)
        ]
