"""
Pydantic models, TypedDict state definitions, and LangGraph reducers for Part 3.

Hierarchy
---------
DocumentChunk          — a meaningful chunk of PDF text, with page and section metadata
RetrievedChunk         — a chunk decorated with BM25/semantic ranks and RRF score
RoutingDecision        — structured Supervisor output (which agents + delegated tasks)
EvidenceAssessment     — structured specialist output (sufficient? supported/missing aspects)
GroundedFact           — a single LLM-generated claim grounded to a source page and quote
AgentResult            — final output from one specialist (facts, status, trace)
TraceEvent             — one observable workflow event (node, action, timestamp)
SupervisorState        — main LangGraph state shared across all nodes
SpecialistState        — internal state of the specialist subgraph
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentName(StrEnum):
    REVENUE = "revenue"
    EXPENDITURE = "expenditure"


class AgentStatus(StrEnum):
    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    chunk_id: str
    page: int
    section: str | None
    text: str


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    bm25_rank: int | None
    semantic_rank: int | None
    rrf_score: float


# ---------------------------------------------------------------------------
# Supervisor routing
# ---------------------------------------------------------------------------


class DelegatedTask(BaseModel):
    task: str
    required_aspects: list[str]


class RoutingDecision(BaseModel):
    selected_agents: list[AgentName]
    revenue_task: DelegatedTask | None = None
    expenditure_task: DelegatedTask | None = None
    # One-sentence routing justification only — no chain-of-thought.
    reason: str


# ---------------------------------------------------------------------------
# Specialist agent
# ---------------------------------------------------------------------------


class EvidenceAssessment(BaseModel):
    sufficient: bool
    supported_aspects: list[str]
    missing_aspects: list[str]
    # Populated only when sufficient=False and another query might help.
    next_search_query: str | None = None


class GroundedFact(BaseModel):
    claim: str
    source_page: int
    evidence_text: str
    amount_text: str | None = None
    source_unit: str | None = None


class GroundedFactsResult(BaseModel):
    """Structured output wrapper for batch fact extraction."""

    facts: list[GroundedFact] = Field(default_factory=list)
    summary: str


class AgentResult(BaseModel):
    agent: AgentName
    status: AgentStatus
    delegated_task: str
    summary: str | None
    facts: list[GroundedFact] = Field(default_factory=list)
    supported_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    attempts: int


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TraceEvent(BaseModel):
    # sequence is local to the emitting node; sort by timestamp for global order.
    sequence: int
    node: str
    action: str
    details: dict[str, Any] | str
    timestamp: str  # ISO 8601 with microseconds


def now_iso() -> str:
    """Return UTC timestamp as ISO 8601 string with microseconds."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class SupervisorState(TypedDict):
    query: str
    routing: RoutingDecision | None
    revenue_result: AgentResult | None
    expenditure_result: AgentResult | None
    # operator.add merges lists from parallel branches safely.
    trace: Annotated[list[TraceEvent], operator.add]
    final_answer: str | None


# ---------------------------------------------------------------------------
# Specialist subgraph internal state
# ---------------------------------------------------------------------------

MAX_RETRIEVAL_ATTEMPTS = 3


class SpecialistAgentConfig(BaseModel):
    name: AgentName
    role_prompt: str


class SpecialistState(TypedDict):
    # ── Inputs from supervisor (set before subgraph is invoked) ──────────────
    agent_name: str  # AgentName value
    delegated_task: str
    required_aspects: list[str]
    role_prompt: str

    # ── Loop tracking ────────────────────────────────────────────────────────
    attempt: int
    search_queries: list[str]  # history of queries issued
    current_query: str
    evidence_pool: list[RetrievedChunk]  # unique chunks accumulated across attempts
    seen_chunk_ids: list[str]  # chunk_ids already in evidence_pool
    evidence_assessment: EvidenceAssessment | None
    stop_reason: str | None  # "success"|"max_attempts"|"no_new_evidence"|"repeated_query"|"error"

    # ── Output ───────────────────────────────────────────────────────────────
    status: str | None  # AgentStatus value
    grounded_facts: list[GroundedFact]
    supported_aspects: list[str]
    missing_aspects: list[str]
    summary: str | None
    trace_events: list[TraceEvent]  # agent-local; merged into main trace on completion
