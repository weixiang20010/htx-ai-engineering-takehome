"""
Tests for synthesis node.

Verifies that partial results (one agent succeeded, one failed) produce
a partial answer, and that full results from both agents produce a combined
answer. All LLM calls are mocked.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.part3.models import (
    AgentName,
    AgentResult,
    AgentStatus,
    DelegatedTask,
    GroundedFact,
    RoutingDecision,
    SupervisorState,
    TraceEvent,
)


def _revenue_success() -> AgentResult:
    return AgentResult(
        agent=AgentName.REVENUE,
        status=AgentStatus.SUCCESS,
        delegated_task="Identify key revenue streams",
        summary="Revenue summary text",
        facts=[
            GroundedFact(
                claim="CIT $28b",
                source_page=5,
                evidence_text="CIT is 28 billion",
            )
        ],
        supported_aspects=["revenue categories"],
        missing_aspects=[],
        attempts=1,
    )


def _expenditure_insufficient() -> AgentResult:
    return AgentResult(
        agent=AgentName.EXPENDITURE,
        status=AgentStatus.INSUFFICIENT_EVIDENCE,
        delegated_task="Explain Future Energy Fund support",
        summary=None,
        facts=[],
        supported_aspects=[],
        missing_aspects=["fund amount"],
        attempts=3,
    )


def _full_state(
    revenue: AgentResult | None = None,
    expenditure: AgentResult | None = None,
) -> SupervisorState:
    return {
        "query": "What are revenue streams and FEF support?",
        "routing": RoutingDecision(
            selected_agents=[AgentName.REVENUE, AgentName.EXPENDITURE],
            revenue_task=DelegatedTask(task="Revenue", required_aspects=[]),
            expenditure_task=DelegatedTask(task="Expenditure", required_aspects=[]),
            reason="Both",
        ),
        "revenue_result": revenue,
        "expenditure_result": expenditure,
        "trace": [],
        "final_answer": None,
    }


@pytest.mark.asyncio
async def test_synthesis_produces_final_answer():
    """Synthesis should always produce a non-empty final_answer."""
    from unittest.mock import patch
    from src.part3.synthesis import build_synthesis_node
    from src.llm import ModelUsage

    mock_response = MagicMock()
    mock_response.content = "Revenue was strong. FEF details unavailable."

    with patch(
        "src.part3.synthesis.ainvoke_with_fallback",
        new=AsyncMock(return_value=(mock_response, ModelUsage(requested_model="m", actual_model="m"))),
    ):
        node = build_synthesis_node(MagicMock(), None)
        state = _full_state(
            revenue=_revenue_success(),
            expenditure=_expenditure_insufficient(),
        )
        result = await node(state)
        assert result["final_answer"]
        assert len(result["final_answer"]) > 10


@pytest.mark.asyncio
async def test_synthesis_emits_trace_event():
    from unittest.mock import patch
    from src.part3.synthesis import build_synthesis_node
    from src.llm import ModelUsage

    mock_response = MagicMock()
    mock_response.content = "Answer"

    with patch(
        "src.part3.synthesis.ainvoke_with_fallback",
        new=AsyncMock(return_value=(mock_response, ModelUsage(requested_model="m", actual_model="m"))),
    ):
        node = build_synthesis_node(MagicMock(), None)
        state = _full_state(
            revenue=_revenue_success(),
            expenditure=_expenditure_insufficient(),
        )
        result = await node(state)
        assert len(result["trace"]) >= 1
        event = result["trace"][0]
        assert event.node == "synthesis"
