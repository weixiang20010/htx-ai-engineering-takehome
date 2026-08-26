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


@pytest.fixture(scope="module")
async def required_htx_state(retriever):
    """Run the required HTX query exactly once; all three assertion tests share this state."""
    from src.part3.graph import run_query
    return await run_query(REQUIRED_HTX_QUERY, retriever)


@pytest.mark.asyncio
async def test_required_htx_query_routes_both_agents(required_htx_state):
    from src.part3.models import AgentName

    routing = required_htx_state.get("routing")
    assert routing is not None, "Supervisor must produce a routing decision"
    agents = routing.selected_agents
    assert AgentName.REVENUE in agents, "Revenue agent must be selected"
    assert AgentName.EXPENDITURE in agents, "Expenditure agent must be selected"


@pytest.mark.asyncio
async def test_required_htx_query_both_agents_succeed(required_htx_state):
    from src.part3.models import AgentStatus

    revenue = required_htx_state.get("revenue_result")
    expenditure = required_htx_state.get("expenditure_result")

    assert revenue is not None, "Revenue result must be present"
    assert expenditure is not None, "Expenditure result must be present"
    assert revenue.status == AgentStatus.SUCCESS, (
        f"Revenue agent must succeed for this source document; got {revenue.status}"
    )
    assert expenditure.status == AgentStatus.SUCCESS, (
        f"Expenditure agent must succeed for this source document; got {expenditure.status}"
    )
    assert revenue.facts, "Revenue agent must return at least one grounded fact"
    assert expenditure.facts, "Expenditure agent must return at least one grounded fact"


@pytest.mark.asyncio
async def test_required_htx_query_produces_final_answer(required_htx_state):
    answer = required_htx_state.get("final_answer")
    assert answer, "A final answer must be produced"
    assert len(answer) > 50, "Final answer should be substantive"


@pytest.mark.asyncio
async def test_required_htx_query_emits_trace(required_htx_state):
    trace = required_htx_state.get("trace", [])
    assert len(trace) >= 3, "At least supervisor + one agent + synthesis trace events expected"
    nodes_seen = {e.node for e in trace}
    assert "supervisor" in nodes_seen
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
