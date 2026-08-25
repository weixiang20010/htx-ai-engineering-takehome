"""
Reusable specialist agent subgraph for Part 3.

Architecture
------------
Each specialist executes an autonomous retrieval loop:

    START
      │
      ▼
    retrieve ──► assess ──► [should_continue?]
      ▲                           │
      │              ┌────────────┼────────────┐
      │         reformulate    extract_facts   (stop)
      └──────────────┘              │
                                   END

Stop conditions:
  SUCCESS        — all required aspects are grounded in evidence
  MAX_ATTEMPTS   — reached MAX_RETRIEVAL_ATTEMPTS (3) without full coverage
  NO_NEW_EVIDENCE — retrieval returned no previously unseen chunks
  REPEATED_QUERY — reformulated query is identical to a prior query
  ERROR          — application-level failure

Configuration
-------------
Two SpecialistAgentConfig objects are pre-defined:

  REVENUE_CONFIG     — government revenue, taxes, NIRC
  EXPENDITURE_CONFIG — spending, fund allocations, top-ups

Call build_specialist_runner() to get a coroutine-returning callable
configured for one role, suitable for use as a LangGraph node.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import Runnable

from ..llm import ModelUsage, ainvoke_with_fallback
from .grounding import ExtractionValidationError, validate_grounded_fact
from .models import (
    MAX_RETRIEVAL_ATTEMPTS,
    AgentName,
    AgentResult,
    AgentStatus,
    DocumentChunk,
    EvidenceAssessment,
    GroundedFact,
    GroundedFactsResult,
    RetrievedChunk,
    SpecialistAgentConfig,
    SpecialistState,
    SupervisorState,
    TraceEvent,
    now_iso,
)
from .prompts import (
    EVIDENCE_ASSESSMENT_PROMPT,
    FACT_EXTRACTION_PROMPT,
    QUERY_REFORMULATION_PROMPT,
)
from .retriever import HybridRRFRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role configurations
# ---------------------------------------------------------------------------

REVENUE_CONFIG = SpecialistAgentConfig(
    name=AgentName.REVENUE,
    role_prompt=(
        "You are the Revenue Agent. Your domain covers government revenue: "
        "Total Revenue, Operating Revenue (taxes and non-tax revenue), "
        "and Net Investment Returns Contribution (NIRC). "
        "Do not stray into spending, fund top-ups, or expenditure unless "
        "strictly necessary to clarify a revenue figure."
    ),
)

EXPENDITURE_CONFIG = SpecialistAgentConfig(
    name=AgentName.EXPENDITURE,
    role_prompt=(
        "You are the Expenditure Agent. Your domain covers government spending: "
        "Total Expenditure, Operating Expenditure, Development Expenditure, "
        "Special Transfers, top-ups to endowment and trust funds, and specific "
        "fund allocations with their stated purposes and amounts. "
        "Pay particular attention to matching specific fund names with their "
        "correct monetary figures as stated in the source document."
    ),
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_evidence_text(pool: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a text block for LLM prompts."""
    lines: list[str] = []
    for rc in pool:
        c = rc.chunk
        lines.append(f"[Page {c.page}] {c.section or ''}\n{c.text}")
    return "\n\n---\n\n".join(lines)


def _trace(
    agent_name: str,
    action: str,
    details: dict[str, Any] | str,
    seq: int,
) -> TraceEvent:
    return TraceEvent(
        sequence=seq,
        node=agent_name,
        action=action,
        details=details,
        timestamp=now_iso(),
    )


def _normalize_query(q: str) -> str:
    return " ".join(q.lower().split())


# ---------------------------------------------------------------------------
# Subgraph node functions
# ---------------------------------------------------------------------------


async def _retrieve(
    state: SpecialistState,
    retriever: HybridRRFRetriever,
) -> dict:
    """Hybrid retrieval; accumulate unique chunks into evidence_pool."""
    query = state["current_query"]
    attempt = state["attempt"]
    seen = set(state["seen_chunk_ids"])

    retrieved = await retriever.retrieve(query)
    new_chunks = [r for r in retrieved if r.chunk.chunk_id not in seen]

    updated_seen = state["seen_chunk_ids"] + [r.chunk.chunk_id for r in new_chunks]
    updated_pool = state["evidence_pool"] + new_chunks

    stop_reason = state.get("stop_reason")
    if not new_chunks and attempt > 0 and stop_reason is None:
        stop_reason = "no_new_evidence"

    event = _trace(
        state["agent_name"],
        f"retrieval_attempt_{attempt + 1}",
        {
            "query": query,
            "chunks_retrieved": len(retrieved),
            "new_chunks": len(new_chunks),
            "evidence_pool_size": len(updated_pool),
        },
        attempt,
    )

    return {
        "attempt": attempt + 1,
        "evidence_pool": updated_pool,
        "seen_chunk_ids": updated_seen,
        "stop_reason": stop_reason,
        "trace_events": state.get("trace_events", []) + [event],
    }


async def _assess(
    state: SpecialistState,
    extraction_llm: Runnable,
    extraction_fallback: Runnable | None,
) -> dict:
    """LLM assesses whether evidence is sufficient for the required aspects."""
    evidence_text = _build_evidence_text(state["evidence_pool"])
    aspects_list = "\n".join(f"  - {a}" for a in state["required_aspects"])

    structured_llm = extraction_llm.with_structured_output(EvidenceAssessment)
    chain = EVIDENCE_ASSESSMENT_PROMPT | structured_llm

    assessment, _ = await ainvoke_with_fallback(
        lambda llm: EVIDENCE_ASSESSMENT_PROMPT
        | llm.with_structured_output(EvidenceAssessment),
        {
            "delegated_task": state["delegated_task"],
            "required_aspects": aspects_list,
            "evidence_text": evidence_text,
        },
        extraction_llm,
        extraction_fallback,
    )
    _ = chain  # chain is built dynamically; kept for clarity

    stop_reason = state.get("stop_reason")
    # Record LLM assessment as "ready_to_extract" rather than "success";
    # final SUCCESS is determined after grounding in _extract_facts.
    if assessment.sufficient and stop_reason is None:
        stop_reason = "ready_to_extract"

    event = _trace(
        state["agent_name"],
        "evidence_assessment",
        {
            "sufficient": assessment.sufficient,
            "supported": assessment.supported_aspects,
            "missing": assessment.missing_aspects,
        },
        state["attempt"],
    )

    return {
        "evidence_assessment": assessment,
        "stop_reason": stop_reason,
        "supported_aspects": assessment.supported_aspects,
        "missing_aspects": assessment.missing_aspects,
        "trace_events": state["trace_events"] + [event],
    }


def _should_continue(state: SpecialistState) -> str:
    """Routing function for the specialist loop."""
    stop = state.get("stop_reason")
    if stop:
        return "extract_facts"
    if state["attempt"] >= MAX_RETRIEVAL_ATTEMPTS:
        return "extract_facts"
    assessment = state.get("evidence_assessment")
    if assessment and not assessment.sufficient:
        return "reformulate"
    return "extract_facts"


async def _reformulate(
    state: SpecialistState,
    reasoning_llm: Runnable,
    reasoning_fallback: Runnable | None,
) -> dict:
    """Advance to the next search query using the assessment's suggestion.

    EvidenceAssessment.next_search_query is produced by the extraction LLM in
    the preceding _assess step.  Using it directly avoids an extra reasoning-
    model call.  The reasoning model is only invoked as a fallback when the
    assessment did not supply a query (None or empty string).
    """
    assessment = state["evidence_assessment"]

    # Primary: use the query already produced by the evidence-assessment LLM.
    new_query = (assessment.next_search_query or "").strip() if assessment else ""

    if not new_query:
        # Fallback: ask the reasoning model to reformulate.
        missing = assessment.missing_aspects if assessment else state["required_aspects"]
        aspects_str = "\n".join(f"  - {a}" for a in missing)
        prev_queries = "\n".join(f"  - {q}" for q in state["search_queries"])

        raw, _ = await ainvoke_with_fallback(
            lambda llm: QUERY_REFORMULATION_PROMPT | llm,
            {
                "delegated_task": state["delegated_task"],
                "missing_aspects": aspects_str,
                "previous_queries": prev_queries,
            },
            reasoning_llm,
            reasoning_fallback,
        )
        new_query = (
            raw.content.strip() if hasattr(raw, "content") else str(raw).strip()
        )

    # Repeated-query guard: compare against all prior queries AND the current
    # query so an immediate repeat is also detected.
    all_seen = {
        _normalize_query(q)
        for q in state["search_queries"] + [state["current_query"]]
    }
    stop_reason = state.get("stop_reason")
    if _normalize_query(new_query) in all_seen:
        stop_reason = "repeated_query"
        new_query = state["current_query"]  # keep current; won't be used

    updated_queries = state["search_queries"] + [state["current_query"]]
    event = _trace(
        state["agent_name"],
        "query_reformulation",
        {"new_query": new_query, "stop_reason": stop_reason},
        state["attempt"],
    )

    return {
        "current_query": new_query,
        "search_queries": updated_queries,
        "stop_reason": stop_reason,
        "trace_events": state["trace_events"] + [event],
    }


async def _extract_facts(
    state: SpecialistState,
    extraction_llm: Runnable,
    extraction_fallback: Runnable | None,
) -> dict:
    """Extract grounded facts from the accumulated evidence pool."""
    stop = state.get("stop_reason") or "max_attempts"
    assessment = state.get("evidence_assessment")

    if not state["evidence_pool"]:
        event = _trace(state["agent_name"], "extract_facts", {"result": "no_evidence"}, state["attempt"])
        return {
            "status": AgentStatus.INSUFFICIENT_EVIDENCE,
            "grounded_facts": [],
            "summary": "No evidence was retrieved from the document.",
            "stop_reason": stop,
            "trace_events": state["trace_events"] + [event],
        }

    evidence_text = _build_evidence_text(state["evidence_pool"])

    result, _ = await ainvoke_with_fallback(
        lambda llm: FACT_EXTRACTION_PROMPT
        | llm.with_structured_output(GroundedFactsResult),
        {
            "role_prompt": state["role_prompt"],
            "delegated_task": state["delegated_task"],
            "evidence_text": evidence_text,
        },
        extraction_llm,
        extraction_fallback,
    )

    # Deterministic grounding validation — drop facts that cannot be verified.
    source_chunks: list[DocumentChunk] = [rc.chunk for rc in state["evidence_pool"]]
    validated: list[GroundedFact] = []
    for fact in result.facts:
        try:
            validate_grounded_fact(fact, source_chunks)
            validated.append(fact)
        except ExtractionValidationError as exc:
            logger.warning(
                "[Part3][%s] Dropping ungrounded fact %r: %s",
                state["agent_name"],
                fact.claim,
                exc,
            )

    # SUCCESS requires at least one validated grounded fact; the LLM evidence
    # assessment alone is insufficient because grounding may reject all facts.
    if validated:
        status = AgentStatus.SUCCESS
    else:
        status = AgentStatus.INSUFFICIENT_EVIDENCE

    event = _trace(
        state["agent_name"],
        "extract_facts",
        {
            "facts_extracted": len(result.facts),
            "facts_validated": len(validated),
            "status": status,
            "stop_reason": stop,
        },
        state["attempt"],
    )

    return {
        "status": status,
        "grounded_facts": validated,
        # summary is from the LLM and is not grounded; kept for audit only.
        # synthesis reads grounded_facts directly and does not use this field.
        "summary": result.summary if validated else None,
        "trace_events": state["trace_events"] + [event],
    }


# ---------------------------------------------------------------------------
# Subgraph builder
# ---------------------------------------------------------------------------


def build_specialist_runner(
    config: SpecialistAgentConfig,
    retriever: HybridRRFRetriever,
    extraction_llm: Runnable,
    extraction_fallback: Runnable | None,
    reasoning_llm: Runnable,
    reasoning_fallback: Runnable | None,
):
    """
    Return an async callable (state: SupervisorState) -> dict that runs the
    specialist loop and returns a partial SupervisorState update.

    Using a compiled LangGraph subgraph internally for the loop.
    """
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    # Bind dependencies via partial application.
    retrieve_fn = partial(_retrieve, retriever=retriever)
    assess_fn = partial(
        _assess,
        extraction_llm=extraction_llm,
        extraction_fallback=extraction_fallback,
    )
    reformulate_fn = partial(
        _reformulate,
        reasoning_llm=reasoning_llm,
        reasoning_fallback=reasoning_fallback,
    )
    extract_fn = partial(
        _extract_facts,
        extraction_llm=extraction_llm,
        extraction_fallback=extraction_fallback,
    )

    g = StateGraph(SpecialistState)
    g.add_node("retrieve", retrieve_fn)
    g.add_node("assess", assess_fn)
    g.add_node("reformulate", reformulate_fn)
    g.add_node("extract_facts", extract_fn)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges(
        "assess",
        _should_continue,
        {"reformulate": "reformulate", "extract_facts": "extract_facts"},
    )
    g.add_edge("reformulate", "retrieve")
    g.add_edge("extract_facts", END)

    subgraph = g.compile()

    result_key = (
        "revenue_result" if config.name == AgentName.REVENUE else "expenditure_result"
    )

    async def run(state: SupervisorState) -> dict:
        routing = state["routing"]
        if config.name == AgentName.REVENUE:
            task_obj = routing.revenue_task
        else:
            task_obj = routing.expenditure_task

        if task_obj is None:
            # Should not happen if routing is correct, but guard it.
            logger.warning("[Part3][%s] No delegated task in routing.", config.name)
            return {result_key: None, "trace": []}

        initial_state: SpecialistState = {
            "agent_name": config.name.value,
            "delegated_task": task_obj.task,
            "required_aspects": task_obj.required_aspects,
            "role_prompt": config.role_prompt,
            "attempt": 0,
            "search_queries": [],
            "current_query": task_obj.task,
            "evidence_pool": [],
            "seen_chunk_ids": [],
            "evidence_assessment": None,
            "stop_reason": None,
            "status": None,
            "grounded_facts": [],
            "supported_aspects": [],
            "missing_aspects": [],
            "summary": None,
            "trace_events": [],
        }

        logger.info(
            "[Part3][%s] Starting specialist loop — task: %s",
            config.name.value,
            task_obj.task,
        )

        final = await subgraph.ainvoke(initial_state)

        agent_result = AgentResult(
            agent=config.name,
            status=AgentStatus(final["status"] or AgentStatus.INSUFFICIENT_EVIDENCE),
            delegated_task=task_obj.task,
            summary=final.get("summary"),
            facts=final.get("grounded_facts", []),
            supported_aspects=final.get("supported_aspects", []),
            missing_aspects=final.get("missing_aspects", []),
            attempts=final["attempt"],
        )

        logger.info(
            "[Part3][%s] Completed — status=%s attempts=%d facts=%d",
            config.name.value,
            agent_result.status,
            agent_result.attempts,
            len(agent_result.facts),
        )

        return {
            result_key: agent_result,
            "trace": final.get("trace_events", []),
        }

    return run
