"""
Tests for the specialist agent retrieval loop.

All LLM calls are mocked — no Gemini API required.
Retrieval is also mocked so the loop stop conditions can be tested
in isolation.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.part3.models import (
    AgentName,
    AgentStatus,
    DocumentChunk,
    EvidenceAssessment,
    GroundedFact,
    GroundedFactsResult,
    RetrievedChunk,
    SpecialistAgentConfig,
    SpecialistState,
)


def _rc(chunk_id: str, page: int = 5, text: str = "chunk text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(chunk_id=chunk_id, page=page, section=None, text=text),
        bm25_rank=1,
        semantic_rank=1,
        rrf_score=0.01,
    )


def _sufficient_assessment() -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=True,
        supported_aspects=["revenue categories"],
        missing_aspects=[],
        next_search_query=None,
    )


def _insufficient_assessment(query: str = "next query") -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=False,
        supported_aspects=[],
        missing_aspects=["revenue breakdown"],
        next_search_query=query,
    )


def _facts_result() -> GroundedFactsResult:
    return GroundedFactsResult(
        facts=[
            GroundedFact(
                claim="CIT is $28 billion",
                source_page=5,
                evidence_text="CIT is $28 billion.",
            )
        ],
        summary="Revenue summary",
    )


# ---------------------------------------------------------------------------
# Helpers to build mocked specialist state for unit-testing internal fns
# ---------------------------------------------------------------------------


def _initial_specialist_state(aspects: list[str] | None = None) -> SpecialistState:
    return SpecialistState(
        agent_name=AgentName.REVENUE,
        delegated_task="Summarise revenue",
        required_aspects=aspects or ["revenue categories"],
        role_prompt="You are the revenue specialist.",
        attempt=1,
        search_queries=[],
        current_query="What are the revenue streams?",
        evidence_pool=[],
        seen_chunk_ids=set(),
        evidence_assessment=None,
        stop_reason=None,
        status=None,
        grounded_facts=None,
        supported_aspects=[],
        missing_aspects=[],
        summary=None,
        trace_events=[],
    )


# ---------------------------------------------------------------------------
# _should_continue decision logic
# ---------------------------------------------------------------------------


def test_should_continue_when_sufficient():
    from src.part3.specialist import _should_continue

    state = _initial_specialist_state()
    state["evidence_assessment"] = _sufficient_assessment()
    assert _should_continue(state) == "extract_facts"


def test_should_continue_when_max_attempts():
    from src.part3.specialist import _should_continue
    from src.part3.models import MAX_RETRIEVAL_ATTEMPTS

    state = _initial_specialist_state()
    state["attempt"] = MAX_RETRIEVAL_ATTEMPTS + 1
    state["evidence_assessment"] = _insufficient_assessment()
    # max attempts exceeded → extract_facts (with insufficient evidence)
    assert _should_continue(state) == "extract_facts"


def test_should_reformulate_when_insufficient():
    from src.part3.specialist import _should_continue
    from src.part3.models import MAX_RETRIEVAL_ATTEMPTS

    state = _initial_specialist_state()
    state["attempt"] = 1
    state["evidence_assessment"] = _insufficient_assessment()
    assert _should_continue(state) == "reformulate"


def test_should_extract_when_no_new_evidence():
    """If evidence pool is empty after retrieval, stop early."""
    from src.part3.specialist import _should_continue
    from src.part3.models import MAX_RETRIEVAL_ATTEMPTS

    state = _initial_specialist_state()
    state["attempt"] = 2
    state["evidence_assessment"] = _insufficient_assessment()
    state["stop_reason"] = "no_new_evidence"
    assert _should_continue(state) == "extract_facts"


def test_should_extract_when_repeated_query():
    """If the reformulated query matches a previous query, stop early."""
    from src.part3.specialist import _should_continue

    state = _initial_specialist_state()
    state["attempt"] = 2
    state["evidence_assessment"] = _insufficient_assessment()
    state["stop_reason"] = "repeated_query"
    assert _should_continue(state) == "extract_facts"
