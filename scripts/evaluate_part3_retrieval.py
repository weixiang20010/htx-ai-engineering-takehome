"""
Evaluate hybrid retrieval quality for Part 3 by comparing:
  BM25-only  vs  Semantic-only  vs  Hybrid-RRF

For each representative query the script prints the top-K chunks returned by
each method and records whether the known-relevant pages appear in the results.

This benchmark is engineering evidence for the README, not used in production.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Repository root and output path.
REPO_ROOT = Path(__file__).parent.parent
PDF_PATH = REPO_ROOT / os.getenv("SOURCE_PDF", "data/fy2024_analysis_of_revenue_and_expenditure.pdf")
OUTPUT_PATH = REPO_ROOT / "outputs" / "part3_retrieval_benchmark.json"

# Representative evaluation queries and the pages known to be relevant.
EVAL_QUERIES = [
    {
        "id": "exact_terminology",
        "query": "Corporate Income Tax Operating Revenue FY2024",
        "description": "Exact terminology — taxes, revenue",
        "relevant_pages": [5, 6, 9, 26],
    },
    {
        "id": "paraphrased_revenue",
        "query": "What are the main sources of government income?",
        "description": "Paraphrased revenue query",
        "relevant_pages": [5, 9, 13, 26],
    },
    {
        "id": "future_energy_fund",
        "query": "How is the Future Energy Fund being financed and supported?",
        "description": "Paraphrased Future Energy Fund / energy-transition support",
        "relevant_pages": [20, 18],
    },
    {
        "id": "top_ups_endowment",
        "query": "top-ups endowment trust funds Budget 2024",
        "description": "Exact terminology — top-ups",
        "relevant_pages": [20, 12, 18],
    },
]

TOP_K = 8


async def run_benchmark() -> None:
    from src.part3.chunker import build_chunks
    from src.part3.retriever import HybridRRFRetriever

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment / .env")

    logger.info("Building document chunks…")
    chunks = build_chunks(PDF_PATH)
    logger.info("Corpus size: %d chunks", len(chunks))

    retriever = HybridRRFRetriever(chunks)
    logger.info("Building semantic index (API call)…")
    await retriever.build_semantic_index(api_key)
    logger.info("Semantic index ready.")

    results: list[dict] = []

    for eq in EVAL_QUERIES:
        logger.info("Evaluating: %s", eq["id"])
        query = eq["query"]
        relevant = set(eq["relevant_pages"])

        bm25_chunks = retriever.retrieve_bm25_only(query, top_k=TOP_K)
        semantic_chunks = await retriever.retrieve_semantic_only(query, top_k=TOP_K)
        hybrid_chunks = await retriever.retrieve(query, top_k=TOP_K)

        def summarize(chunks):
            pages_found = [c.chunk.page for c in chunks]
            relevant_hit = [p for p in pages_found if p in relevant]
            return {
                "top_pages": pages_found,
                "relevant_pages_hit": relevant_hit,
                "hit_rate": len(relevant_hit) / max(len(relevant), 1),
                "top_chunks": [
                    {
                        "chunk_id": c.chunk.chunk_id,
                        "page": c.chunk.page,
                        "section": c.chunk.section,
                        "bm25_rank": c.bm25_rank,
                        "semantic_rank": c.semantic_rank,
                        "rrf_score": round(c.rrf_score, 6),
                        "text_preview": c.chunk.text[:120],
                    }
                    for c in chunks
                ],
            }

        result = {
            "query_id": eq["id"],
            "query": query,
            "description": eq["description"],
            "relevant_pages": list(relevant),
            "bm25_only": summarize(bm25_chunks),
            "semantic_only": summarize(semantic_chunks),
            "hybrid_rrf": summarize(hybrid_chunks),
        }
        results.append(result)

        logger.info(
            "  BM25 hit_rate=%.2f | Semantic hit_rate=%.2f | Hybrid hit_rate=%.2f",
            result["bm25_only"]["hit_rate"],
            result["semantic_only"]["hit_rate"],
            result["hybrid_rrf"]["hit_rate"],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark saved to %s", OUTPUT_PATH)

    # Print summary table.
    print("\n=== Retrieval Benchmark Summary ===")
    print(f"{'Query':<35} {'BM25':>8} {'Semantic':>10} {'Hybrid':>8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['query_id']:<35} "
            f"{r['bm25_only']['hit_rate']:>8.2f} "
            f"{r['semantic_only']['hit_rate']:>10.2f} "
            f"{r['hybrid_rrf']['hit_rate']:>8.2f}"
        )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
