"""
Main LangGraph graph for Part 3.

Graph structure
---------------

    START
      │
      ▼
  supervisor_route ──► [route_to_agents] ──► revenue_agent ──► synthesis ──► END
                                         └─► expenditure_agent ──┘

When both agents are selected, revenue_agent and expenditure_agent execute
in parallel (LangGraph fan-out via conditional edges returning a list).
Synthesis waits for all active branches before executing (LangGraph fan-in).

When only one agent is selected, only that branch runs; synthesis receives
the partial state.

State concurrency safety
------------------------
- revenue_result and expenditure_result are separate keys → no write conflict.
- trace uses operator.add reducer → parallel writes are merged safely.
"""
from __future__ import annotations

import logging
from typing import Callable

from langgraph.graph import END, START, StateGraph

from ..llm import build_extraction_llm_pair, build_reasoning_llm_pair
from .models import AgentName, SupervisorState
from .retriever import HybridRRFRetriever
from .specialist import EXPENDITURE_CONFIG, REVENUE_CONFIG, build_specialist_runner
from .supervisor import build_supervisor_node
from .synthesis import build_synthesis_node

logger = logging.getLogger(__name__)


def _build_router(
    revenue_runner: Callable,
    expenditure_runner: Callable,
) -> tuple[Callable, Callable, Callable]:
    """
    Return (route_fn, revenue_node, expenditure_node).

    route_fn returns the list of node names to activate (fan-out).
    """

    def route_to_agents(state: SupervisorState) -> list[str]:
        routing = state["routing"]
        names = []
        if AgentName.REVENUE in routing.selected_agents:
            names.append("revenue_agent")
        if AgentName.EXPENDITURE in routing.selected_agents:
            names.append("expenditure_agent")
        logger.info("[Part3][Graph] Routing to: %s", names)
        return names

    return route_to_agents, revenue_runner, expenditure_runner


def build_graph(retriever: HybridRRFRetriever) -> object:
    """
    Compile and return the Part 3 LangGraph application.

    The graph reads LLM configuration from environment variables via the
    existing build_extraction_llm_pair / build_reasoning_llm_pair helpers.
    """
    extraction_llm, extraction_fallback = build_extraction_llm_pair()
    reasoning_llm, reasoning_fallback = build_reasoning_llm_pair()

    supervisor_node = build_supervisor_node(reasoning_llm, reasoning_fallback)
    synthesis_node = build_synthesis_node(reasoning_llm, reasoning_fallback)

    revenue_runner = build_specialist_runner(
        config=REVENUE_CONFIG,
        retriever=retriever,
        extraction_llm=extraction_llm,
        extraction_fallback=extraction_fallback,
        reasoning_llm=reasoning_llm,
        reasoning_fallback=reasoning_fallback,
    )
    expenditure_runner = build_specialist_runner(
        config=EXPENDITURE_CONFIG,
        retriever=retriever,
        extraction_llm=extraction_llm,
        extraction_fallback=extraction_fallback,
        reasoning_llm=reasoning_llm,
        reasoning_fallback=reasoning_fallback,
    )

    route_fn, revenue_node, expenditure_node = _build_router(revenue_runner, expenditure_runner)

    g = StateGraph(SupervisorState)
    g.add_node("supervisor_route", supervisor_node)
    g.add_node("revenue_agent", revenue_node)
    g.add_node("expenditure_agent", expenditure_node)
    g.add_node("synthesis", synthesis_node)

    g.add_edge(START, "supervisor_route")

    # Fan-out: conditional edges returning a list → parallel execution.
    g.add_conditional_edges(
        "supervisor_route",
        route_fn,
        ["revenue_agent", "expenditure_agent"],
    )

    # Fan-in: both agents converge on synthesis.
    g.add_edge("revenue_agent", "synthesis")
    g.add_edge("expenditure_agent", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile()


async def run_query(
    query: str,
    retriever: HybridRRFRetriever,
    app=None,
) -> SupervisorState:
    """
    Run a single query through the Part 3 graph and return the final state.

    Parameters
    ----------
    query:
        User question.
    retriever:
        Pre-built HybridRRFRetriever.
    app:
        Pre-compiled graph (built once and reused across queries).
    """
    if app is None:
        app = build_graph(retriever)

    initial: SupervisorState = {
        "query": query,
        "routing": None,
        "revenue_result": None,
        "expenditure_result": None,
        "trace": [],
        "final_answer": None,
    }

    return await app.ainvoke(initial)
