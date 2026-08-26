"""Pydantic models for the Part 2 extraction and classification pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from src.llm import ModelUsage

# Fixed assessment constant — do not derive from system clock
REFERENCE_DATE: date = date(2024, 1, 1)


class ExtractedDate(BaseModel):
    """LLM-extracted date evidence for a single PDF page."""

    source_page: int = Field(description="Page number the evidence was drawn from")
    original_text: str | None = Field(
        default=None,
        description=(
            "Verbatim phrase or sentence from the source that contains the date. "
            "Copy exactly — do not paraphrase."
        ),
    )
    date_text: str | None = Field(
        default=None,
        description=(
            "Exact date string from the source, copied without modification or normalisation. "
            "Must appear as a literal substring of original_text."
        ),
    )


class TemporalStatus(StrEnum):
    EXPIRED = "Expired"
    UPCOMING = "Upcoming"
    ONGOING = "Ongoing"


class TemporalInterpretation(BaseModel):
    """LLM-produced structural reading of how a date expression functions semantically."""

    temporal_type: Literal["point_event", "period", "cutoff_or_threshold"]
    relation: Literal["before", "after", "on", "from", "until", "between", "not_applicable"]
    has_explicit_end: bool = Field(
        description="True only if the source text states an explicit end date or expiry for the condition."
    )
    brief_reason: str = Field(description="One sentence explaining the temporal reading.")


class InternalDateClassification(BaseModel):
    """LLM classification output — only the fields the LLM actually determines."""

    status: TemporalStatus
    reason: str = Field(description="Concise rationale for the classification (one or two sentences)")


@dataclass
class ClassificationResult:
    """Full result of the interpret → classify → consistency pipeline."""

    classification: InternalDateClassification
    interpretation: TemporalInterpretation
    interpretation_usage: ModelUsage
    classification_usage: ModelUsage
    consistency_check_passed: bool
    retried: bool


class Part2ResultItem(BaseModel):
    """Final HTX-schema output item — exactly three fields, no extras."""

    original_text: str
    normalized_date: str
    status: TemporalStatus


class PerOperationModelUsage(BaseModel):
    """Model usage broken down by pipeline stage for full audit transparency."""

    date_extraction: ModelUsage
    tool_selection: ModelUsage
    interpretation: ModelUsage
    classification: ModelUsage


class Part2Evidence(BaseModel):
    """Full audit record for one extracted and classified date."""

    source_page: int
    source_statement: str
    extracted_date_text: str
    source_validation_passed: bool
    mcp_tool_requested: str
    mcp_tool_arguments: dict[str, str]
    mcp_tool_result: str
    reference_date: str
    temporal_interpretation: TemporalInterpretation
    consistency_check_passed: bool
    classification_retried: bool
    llm_status: str
    llm_rationale: str
    model_usage: PerOperationModelUsage
