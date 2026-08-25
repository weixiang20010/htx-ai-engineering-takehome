"""
Run Part 3 demonstration queries through the LangGraph supervisor.

Four queries are run in sequence:
  A — Revenue only
  B — Expenditure only
  C — Required HTX query (Revenue + Expenditure in parallel)
  D — Mixed comparison query

Outputs:
  outputs/part3_result.json       — result from the required HTX query (C)
  outputs/part3_trace.json        — trace from the required HTX query (C)
  outputs/part3_demo_queries.json — results from all four queries
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

REPO_ROOT = Path(__file__).parent.parent
PDF_PATH = REPO_ROOT / os.getenv("SOURCE_PDF", "data/fy2024_analysis_of_revenue_and_expenditure.pdf")
OUTPUT_DIR = REPO_ROOT / "outputs"

DEMO_QUERIES = [
    {
        "id": "A",
        "label": "Revenue only",
        "query": "What are the key government revenue streams?",
        "expected_agents": ["revenue"],
    },
    {
        "id": "B",
        "label": "Expenditure only",
        "query": "How will the Future Energy Fund be supported?",
        "expected_agents": ["expenditure"],
    },
    {
        "id": "C",
        "label": "Required HTX query",
        "query": (
            "What are the key government revenue streams, and how will the Budget "
            "for the Future Energy Fund be supported?"
        ),
        "expected_agents": ["revenue", "expenditure"],
    },
    {
        "id": "D",
        "label": "Mixed comparison",
        "query": (
            "How does the government's revenue outlook compare to its planned "
            "spending on endowment and trust funds in FY2024?"
        ),
        "expected_agents": ["revenue", "expenditure"],
    },
]


async def run_all() -> None:
    from src.part3.chunker import build_chunks
    from src.part3.graph import build_graph, run_query
    from src.part3.models import AgentStatus

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment / .env")

    logger.info("Building document chunks from %s…", PDF_PATH)
    chunks = build_chunks(PDF_PATH)
    logger.info("Corpus: %d chunks", len(chunks))

    from src.part3.retriever import HybridRRFRetriever

    retriever = HybridRRFRetriever(chunks)
    logger.info("Building semantic index…")
    await retriever.build_semantic_index(api_key)
    logger.info("Semantic index ready.")

    app = build_graph(retriever)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    demo_results: list[dict] = []
    # Retain the final state for Query C (HTX required query) — used for both
    # part3_result.json and part3_trace.json so both files describe the same run.
    htx_final_state = None

    for dq in DEMO_QUERIES:
        logger.info("\n%s", "=" * 60)
        logger.info("Query %s — %s", dq["id"], dq["label"])
        logger.info("  %s", dq["query"])

        final_state = await run_query(dq["query"], retriever, app=app)

        if dq["id"] == "C":
            htx_final_state = final_state

        agents_used: list[str] = []
        agent_statuses: dict[str, str] = {}

        if final_state.get("revenue_result") is not None:
            r = final_state["revenue_result"]
            agents_used.append("revenue")
            agent_statuses["revenue"] = r.status.value
        if final_state.get("expenditure_result") is not None:
            e = final_state["expenditure_result"]
            agents_used.append("expenditure")
            agent_statuses["expenditure"] = e.status.value

        routing = final_state.get("routing")
        result_entry = {
            "query_id": dq["id"],
            "label": dq["label"],
            "query": dq["query"],
            "expected_agents": dq["expected_agents"],
            "actual_agents": agents_used,
            "routing_correct": sorted(agents_used) == sorted(dq["expected_agents"]),
            "answer": final_state.get("final_answer"),
            "agents_used": agents_used,
            "agent_statuses": agent_statuses,
            "routing_reason": routing.reason if routing else None,
        }
        demo_results.append(result_entry)

        logger.info("  Agents used: %s", agents_used)
        logger.info("  Statuses: %s", agent_statuses)
        logger.info("  Routing correct: %s", result_entry["routing_correct"])
        if final_state.get("final_answer"):
            logger.info("  Answer (preview): %s…", final_state["final_answer"][:200])

    # Save all demo results.
    with open(OUTPUT_DIR / "part3_demo_queries.json", "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved outputs/part3_demo_queries.json")

    # Save Query C result and trace from the single retained state (no re-run).
    htx_result_entry = next(r for r in demo_results if r["query_id"] == "C")
    part3_result = {
        "query": htx_result_entry["query"],
        "answer": htx_result_entry["answer"],
        "agents_used": htx_result_entry["agents_used"],
        "agent_statuses": htx_result_entry["agent_statuses"],
    }
    with open(OUTPUT_DIR / "part3_result.json", "w", encoding="utf-8") as f:
        json.dump(part3_result, f, indent=2, ensure_ascii=False)
    logger.info("Saved outputs/part3_result.json")

    trace_records = [
        {
            "sequence": e.sequence,
            "node": e.node,
            "action": e.action,
            "details": e.details,
            "timestamp": e.timestamp,
        }
        for e in sorted(htx_final_state.get("trace", []), key=lambda x: x.timestamp)
    ]
    with open(OUTPUT_DIR / "part3_trace.json", "w", encoding="utf-8") as f:
        json.dump(trace_records, f, indent=2, ensure_ascii=False)
    logger.info("Saved outputs/part3_trace.json")

    logger.info("\n%s", "=" * 60)
    logger.info("All outputs saved to outputs/")
    logger.info("Routing accuracy: %d/%d queries routed correctly",
                sum(1 for r in demo_results if r["routing_correct"]), len(demo_results))


if __name__ == "__main__":
    asyncio.run(run_all())
