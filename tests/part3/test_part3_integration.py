"""
Integration test for Part 3 — end-to-end graph run.

Requires:
  - GEMINI_API_KEY set in environment / .env
  - Source PDF present at data/

Run with:  pytest tests/part3/test_part3_integration.py -m integration -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PDF_PATH = Path(__file__).parent.parent.parent / os.getenv(
    "SOURCE_PDF", "data/fy2024_analysis_of_revenue_and_expenditure.pdf"
)

REQUIRED_HTX_QUERY = (
    "What are the key government revenue streams, and how will the Budget "
    "for the Future Energy Fund be supported?"
)


@pytest.fixture(scope="module")
def event_loop_policy():
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="module")
async def retriever():
    if not PDF_PATH.exists():
        pytest.skip(f"Source PDF not found at {PDF_PATH}")
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    from src.part3.chunker import build_chunks
    from src.part3.retriever import HybridRRFRetriever

    chunks = build_chunks(PDF_PATH)
    r = HybridRRFRetriever(chunks)
    await r.build_semantic_index(os.environ["GEMINI_API_KEY"])
    return r


@pytest.mark.asyncio
async def test_required_htx_query_routes_both_agents(retriever):
    from src.part3.graph import run_query
    from src.part3.models import AgentName

    final_state = await run_query(REQUIRED_HTX_QUERY, retriever)

    routing = final_state.get("routing")
    assert routing is not None, "Supervisor must produce a routing decision"
    agents = routing.selected_agents
    assert AgentName.REVENUE in agents, "Revenue agent must be selected"
    assert AgentName.EXPENDITURE in agents, "Expenditure agent must be selected"


@pytest.mark.asyncio
async def test_required_htx_query_produces_final_answer(retriever):
    from src.part3.graph import run_query

    final_state = await run_query(REQUIRED_HTX_QUERY, retriever)
    answer = final_state.get("final_answer")
    assert answer, "A final answer must be produced"
    assert len(answer) > 50, "Final answer should be substantive"


@pytest.mark.asyncio
async def test_required_htx_query_emits_trace(retriever):
    from src.part3.graph import run_query

    final_state = await run_query(REQUIRED_HTX_QUERY, retriever)
    trace = final_state.get("trace", [])
    assert len(trace) >= 3, "At least supervisor + one agent + synthesis trace events expected"
    nodes_seen = {e.node for e in trace}
    assert "supervisor_route" in nodes_seen
    assert "synthesis" in nodes_seen


@pytest.mark.asyncio
async def test_revenue_only_query_routes_revenue_only(retriever):
    from src.part3.graph import run_query
    from src.part3.models import AgentName

    final_state = await run_query("What are the key government revenue streams?", retriever)
    routing = final_state.get("routing")
    assert AgentName.REVENUE in routing.selected_agents
    # Expenditure result should be absent (or None).
    assert final_state.get("expenditure_result") is None


@pytest.mark.asyncio
async def test_future_energy_fund_query_routes_expenditure_only(retriever):
    from src.part3.graph import run_query
    from src.part3.models import AgentName

    final_state = await run_query("How will the Future Energy Fund be supported?", retriever)
    routing = final_state.get("routing")
    assert AgentName.EXPENDITURE in routing.selected_agents
    assert final_state.get("revenue_result") is None
