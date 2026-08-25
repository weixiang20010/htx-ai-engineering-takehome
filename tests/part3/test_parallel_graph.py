"""
Parallel graph routing tests using a compiled LangGraph.

These tests build a minimal StateGraph (no LLM, no retriever) with instrumented
async nodes to verify LangGraph's fan-out / fan-in semantics:

  - When both agents are selected, both branches execute before synthesis.
  - When only one agent is selected, only that branch executes.
  - Synthesis fires after all active branches complete.

Using an actual compiled StateGraph proves that LangGraph's conditional-edge
fan-out and fan-in work correctly for this graph topology.
"""
from __future__ import annotations

import asyncio
import pytest

from langgraph.graph import END, START, StateGraph

from src.part3.models import (
    AgentName,
    AgentResult,
    AgentStatus,
    DelegatedTask,
    RoutingDecision,
    SupervisorState,
)


def _dummy_result(agent: AgentName) -> AgentResult:
    return AgentResult(
        agent=agent,
        status=AgentStatus.SUCCESS,
        delegated_task="Test task",
        summary=None,
        facts=[],
        supported_aspects=[],
        missing_aspects=[],
        attempts=1,
    )


def _make_routing(agents: list[AgentName]) -> RoutingDecision:
    return RoutingDecision(
        selected_agents=agents,
        revenue_task=DelegatedTask(task="R", required_aspects=[])
        if AgentName.REVENUE in agents else None,
        expenditure_task=DelegatedTask(task="E", required_aspects=[])
        if AgentName.EXPENDITURE in agents else None,
        reason="Test",
    )


def _build_test_graph(call_log: list[str]) -> object:
    """
    Compiled StateGraph with the same topology as the real Part 3 graph but
    using instrumented async nodes instead of LLM/retriever calls.

    Nodes append events to call_log so tests can assert execution order.
    """
    from src.part3.graph import _build_router

    async def revenue_runner(state: SupervisorState) -> dict:
        call_log.append("revenue_start")
        await asyncio.sleep(0)  # yield to expose interleaving
        call_log.append("revenue_end")
        return {"revenue_result": _dummy_result(AgentName.REVENUE), "trace": []}

    async def expenditure_runner(state: SupervisorState) -> dict:
        call_log.append("expenditure_start")
        await asyncio.sleep(0)
        call_log.append("expenditure_end")
        return {"expenditure_result": _dummy_result(AgentName.EXPENDITURE), "trace": []}

    async def supervisor_node(state: SupervisorState) -> dict:
        call_log.append("supervisor")
        return {}  # routing already set in initial state

    async def synthesis_node(state: SupervisorState) -> dict:
        call_log.append("synthesis")
        return {"final_answer": "Test answer"}

    route_fn, rev_node, exp_node = _build_router(revenue_runner, expenditure_runner)

    g = StateGraph(SupervisorState)
    g.add_node("supervisor_route", supervisor_node)
    g.add_node("revenue_agent", rev_node)
    g.add_node("expenditure_agent", exp_node)
    g.add_node("synthesis", synthesis_node)

    g.add_edge(START, "supervisor_route")
    g.add_conditional_edges(
        "supervisor_route",
        route_fn,
        ["revenue_agent", "expenditure_agent"],
    )
    g.add_edge("revenue_agent", "synthesis")
    g.add_edge("expenditure_agent", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile()


def _initial_state(agents: list[AgentName]) -> SupervisorState:
    return {
        "query": "test query",
        "routing": _make_routing(agents),
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_branches_execute_before_synthesis():
    """
    With both agents selected, revenue and expenditure must both complete
    before synthesis runs — verified against a compiled LangGraph.
    """
    call_log: list[str] = []
    app = _build_test_graph(call_log)

    await app.ainvoke(_initial_state([AgentName.REVENUE, AgentName.EXPENDITURE]))

    assert "revenue_start" in call_log, "Revenue branch must execute"
    assert "expenditure_start" in call_log, "Expenditure branch must execute"
    assert "synthesis" in call_log, "Synthesis must execute"

    synthesis_idx = call_log.index("synthesis")
    assert call_log.index("revenue_end") < synthesis_idx
    assert call_log.index("expenditure_end") < synthesis_idx


@pytest.mark.asyncio
async def test_both_branches_populate_state():
    """Both agent results must appear in the final state."""
    call_log: list[str] = []
    app = _build_test_graph(call_log)

    final = await app.ainvoke(_initial_state([AgentName.REVENUE, AgentName.EXPENDITURE]))

    assert final["revenue_result"] is not None
    assert final["expenditure_result"] is not None
    assert final["final_answer"] == "Test answer"


@pytest.mark.asyncio
async def test_revenue_only_branch():
    """Only revenue_agent executes; expenditure_result stays None."""
    call_log: list[str] = []
    app = _build_test_graph(call_log)

    final = await app.ainvoke(_initial_state([AgentName.REVENUE]))

    assert "revenue_start" in call_log
    assert "expenditure_start" not in call_log
    assert "synthesis" in call_log
    assert final["revenue_result"] is not None
    assert final["expenditure_result"] is None


@pytest.mark.asyncio
async def test_expenditure_only_branch():
    """Only expenditure_agent executes; revenue_result stays None."""
    call_log: list[str] = []
    app = _build_test_graph(call_log)

    final = await app.ainvoke(_initial_state([AgentName.EXPENDITURE]))

    assert "expenditure_start" in call_log
    assert "revenue_start" not in call_log
    assert "synthesis" in call_log
    assert final["expenditure_result"] is not None
    assert final["revenue_result"] is None
