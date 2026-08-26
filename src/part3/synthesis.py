"""
Synthesis node for Part 3.

The synthesis node receives the original query and whichever agent results
are available, then produces a grounded final answer. It uses only the
grounded facts from the specialist agents — no independent document access.
"""
from __future__ import annotations

import logging

from langchain_core.runnables import Runnable

from ..llm import ainvoke_with_fallback
from .models import AgentResult, AgentStatus, SupervisorState, TraceEvent, now_iso
from .prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


def _format_agent_findings(
    revenue_result: AgentResult | None,
    expenditure_result: AgentResult | None,
) -> str:
    """Render agent results as a text block for the synthesis prompt."""
    parts: list[str] = []

    for result in (revenue_result, expenditure_result):
        if result is None:
            continue
        header = f"=== {result.agent.value.upper()} AGENT ==="
        status_line = f"Status: {result.status}"
        parts.append(header)
        parts.append(status_line)

        if result.status == AgentStatus.INSUFFICIENT_EVIDENCE:
            parts.append(
                f"Note: Evidence was insufficient for these aspects: "
                f"{result.missing_aspects}"
            )

        for fact in result.facts:
            page_ref = f"[p. {fact.source_page}]"
            amount_note = ""
            if fact.amount_text and fact.source_unit:
                amount_note = f" ({fact.amount_text} {fact.source_unit})"
            parts.append(
                f"Fact {page_ref}: {fact.claim}{amount_note}\n"
                f"  Evidence: {fact.evidence_text!r}"
            )
        parts.append("")  # blank separator

    return "\n".join(parts).strip()


def build_synthesis_node(
    reasoning_llm: Runnable,
    reasoning_fallback: Runnable | None,
):
    """Return an async LangGraph node function for synthesis."""

    async def synthesis(state: SupervisorState) -> dict:
        query = state["query"]
        revenue_result = state.get("revenue_result")
        expenditure_result = state.get("expenditure_result")

        findings_text = _format_agent_findings(revenue_result, expenditure_result)
        logger.info("[Part3][Synthesis] Synthesizing findings for query: %s", query)

        answer_raw, _ = await ainvoke_with_fallback(
            lambda llm: SYNTHESIS_PROMPT | llm,
            {"query": query, "agent_findings": findings_text},
            reasoning_llm,
            reasoning_fallback,
        )
        content = answer_raw.content if hasattr(answer_raw, "content") else answer_raw
        if isinstance(content, list):
            # Gemini can return a list of content parts; extract text from each
            answer = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            ).strip()
        else:
            answer = str(content).strip()

        agents_used = []
        if revenue_result is not None:
            agents_used.append("revenue")
        if expenditure_result is not None:
            agents_used.append("expenditure")

        event = TraceEvent(
            sequence=0,
            node="synthesis",
            action="final_synthesis",
            details={
                "agents_used": agents_used,
                "answer_length": len(answer),
            },
            timestamp=now_iso(),
        )

        logger.info("[Part3][Synthesis] Done — answer length %d chars", len(answer))
        return {"final_answer": answer, "trace": [event]}

    return synthesis
