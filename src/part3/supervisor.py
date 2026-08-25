"""
Supervisor routing node for Part 3.

The supervisor's sole job is structured routing: it decides which agent(s)
should handle the query and what each should investigate. It does NOT answer
the financial question itself.
"""
from __future__ import annotations

import logging

from langchain_core.runnables import Runnable

from ..llm import ainvoke_with_fallback
from .models import RoutingDecision, SupervisorState, TraceEvent, now_iso
from .prompts import SUPERVISOR_ROUTING_PROMPT

logger = logging.getLogger(__name__)


def build_supervisor_node(
    reasoning_llm: Runnable,
    reasoning_fallback: Runnable | None,
):
    """Return an async LangGraph node function for the supervisor."""

    async def supervisor_route(state: SupervisorState) -> dict:
        query = state["query"]
        logger.info("[Part3][Supervisor] Routing query: %s", query)

        routing, _ = await ainvoke_with_fallback(
            lambda llm: SUPERVISOR_ROUTING_PROMPT
            | llm.with_structured_output(RoutingDecision),
            {"query": query},
            reasoning_llm,
            reasoning_fallback,
        )

        logger.info(
            "[Part3][Supervisor] Selected agents: %s — %s",
            [a.value for a in routing.selected_agents],
            routing.reason,
        )

        event = TraceEvent(
            sequence=0,
            node="supervisor",
            action="routing_decision",
            details={
                "selected_agents": [a.value for a in routing.selected_agents],
                "reason": routing.reason,
                "revenue_task": routing.revenue_task.model_dump() if routing.revenue_task else None,
                "expenditure_task": routing.expenditure_task.model_dump()
                if routing.expenditure_task
                else None,
            },
            timestamp=now_iso(),
        )

        return {"routing": routing, "trace": [event]}

    return supervisor_route
