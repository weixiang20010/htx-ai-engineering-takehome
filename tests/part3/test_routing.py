"""
Tests for Supervisor routing logic.

All routing tests mock the LLM call so no Gemini API is needed.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.part3.models import (
    AgentName,
    DelegatedTask,
    RoutingDecision,
    SupervisorState,
)


def _make_routing(agents: list[AgentName], with_tasks: bool = True) -> RoutingDecision:
    return RoutingDecision(
        selected_agents=agents,
        revenue_task=DelegatedTask(
            task="Identify key revenue streams",
            required_aspects=["revenue categories", "top revenue components"],
        )
        if AgentName.REVENUE in agents and with_tasks
        else None,
        expenditure_task=DelegatedTask(
            task="Explain Future Energy Fund support",
            required_aspects=["funding mechanism", "amount", "purpose"],
        )
        if AgentName.EXPENDITURE in agents and with_tasks
        else None,
        reason="Test routing",
    )


# ---------------------------------------------------------------------------
# Routing decision model
# ---------------------------------------------------------------------------


def test_revenue_only_routing():
    r = _make_routing([AgentName.REVENUE])
    assert r.selected_agents == [AgentName.REVENUE]
    assert r.revenue_task is not None
    assert r.expenditure_task is None


def test_expenditure_only_routing():
    r = _make_routing([AgentName.EXPENDITURE])
    assert r.selected_agents == [AgentName.EXPENDITURE]
    assert r.expenditure_task is not None
    assert r.revenue_task is None


def test_both_agents_routing():
    r = _make_routing([AgentName.REVENUE, AgentName.EXPENDITURE])
    assert AgentName.REVENUE in r.selected_agents
    assert AgentName.EXPENDITURE in r.selected_agents
    assert r.revenue_task is not None
    assert r.expenditure_task is not None


# ---------------------------------------------------------------------------
# Graph routing function (no LLM)
# ---------------------------------------------------------------------------


def test_graph_route_revenue_only():
    """route_to_agents must return only revenue_agent when routing selects revenue."""
    from src.part3.graph import _build_router

    route_fn, _, _ = _build_router(AsyncMock(), AsyncMock())
    state: SupervisorState = {
        "query": "revenue query",
        "routing": _make_routing([AgentName.REVENUE]),
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }
    result = route_fn(state)
    assert result == ["revenue_agent"]
    assert "expenditure_agent" not in result


def test_graph_route_expenditure_only():
    from src.part3.graph import _build_router

    route_fn, _, _ = _build_router(AsyncMock(), AsyncMock())
    state: SupervisorState = {
        "query": "expenditure query",
        "routing": _make_routing([AgentName.EXPENDITURE]),
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }
    result = route_fn(state)
    assert result == ["expenditure_agent"]
    assert "revenue_agent" not in result


def test_graph_route_both_agents():
    from src.part3.graph import _build_router

    route_fn, _, _ = _build_router(AsyncMock(), AsyncMock())
    state: SupervisorState = {
        "query": "both query",
        "routing": _make_routing([AgentName.REVENUE, AgentName.EXPENDITURE]),
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }
    result = route_fn(state)
    assert "revenue_agent" in result
    assert "expenditure_agent" in result
    assert len(result) == 2
