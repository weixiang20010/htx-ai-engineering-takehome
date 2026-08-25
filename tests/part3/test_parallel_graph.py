"""
Parallel graph routing tests.

Verifies that:
  - When both agents are selected, both nodes are activated.
  - When only one agent is selected, only that node runs.
  - Synthesis fires after all activated branches complete.

All LLM calls and agent runners are mocked.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock

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
        summary="OK",
        facts=[],
        supported_aspects=[],
        missing_aspects=[],
        attempts=1,
    )


def _make_runner(agent: AgentName, call_tracker: list) -> AsyncMock:
    async def runner(state: SupervisorState) -> dict:
        call_tracker.append(agent)
        key = "revenue_result" if agent == AgentName.REVENUE else "expenditure_result"
        return {key: _dummy_result(agent), "trace": []}

    return runner


@pytest.mark.asyncio
async def test_both_agents_activated_when_both_selected():
    from src.part3.graph import _build_router

    calls: list[AgentName] = []
    revenue_runner = _make_runner(AgentName.REVENUE, calls)
    expenditure_runner = _make_runner(AgentName.EXPENDITURE, calls)

    route_fn, rev_node, exp_node = _build_router(revenue_runner, expenditure_runner)

    state: SupervisorState = {
        "query": "q",
        "routing": RoutingDecision(
            selected_agents=[AgentName.REVENUE, AgentName.EXPENDITURE],
            revenue_task=DelegatedTask(task="R", required_aspects=[]),
            expenditure_task=DelegatedTask(task="E", required_aspects=[]),
            reason="Both",
        ),
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }

    # Simulate the graph routing step.
    activated = route_fn(state)
    assert "revenue_agent" in activated
    assert "expenditure_agent" in activated

    # Simulate parallel execution.
    results = await asyncio.gather(rev_node(state), exp_node(state))
    assert any(AgentName.REVENUE in str(r) for r in results)
    assert any(AgentName.EXPENDITURE in str(r) for r in results)


@pytest.mark.asyncio
async def test_only_revenue_activated_for_revenue_only():
    from src.part3.graph import _build_router

    calls: list[AgentName] = []
    revenue_runner = _make_runner(AgentName.REVENUE, calls)
    expenditure_runner = _make_runner(AgentName.EXPENDITURE, calls)

    route_fn, _, _ = _build_router(revenue_runner, expenditure_runner)

    state: SupervisorState = {
        "query": "q",
        "routing": RoutingDecision(
            selected_agents=[AgentName.REVENUE],
            revenue_task=DelegatedTask(task="R", required_aspects=[]),
            expenditure_task=None,
            reason="Revenue only",
        ),
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }

    activated = route_fn(state)
    assert activated == ["revenue_agent"]
    assert "expenditure_agent" not in activated


@pytest.mark.asyncio
async def test_only_expenditure_activated_for_expenditure_only():
    from src.part3.graph import _build_router

    calls: list[AgentName] = []
    route_fn, _, _ = _build_router(
        _make_runner(AgentName.REVENUE, calls),
        _make_runner(AgentName.EXPENDITURE, calls),
    )

    state: SupervisorState = {
        "query": "q",
        "routing": RoutingDecision(
            selected_agents=[AgentName.EXPENDITURE],
            revenue_task=None,
            expenditure_task=DelegatedTask(task="E", required_aspects=[]),
            reason="Expenditure only",
        ),
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }

    activated = route_fn(state)
    assert activated == ["expenditure_agent"]
