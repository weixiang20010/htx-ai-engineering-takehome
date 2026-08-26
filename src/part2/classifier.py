"""
LLM temporal classification using a two-step interpret → classify pipeline.

Step 1 (interpretation): the model reads the source sentence and returns a
structured description of how the date functions semantically — whether it is a
point event, a period, or a cutoff/threshold, and which directional relation
(after, until, on, …) the text expresses.

Step 2 (classification): the model receives the interpretation alongside the
original text and produces Expired / Upcoming / Ongoing.  Because the task is
narrower — label selection given a structured reading — a lightweight model can
handle it reliably.

A deterministic consistency check then verifies that the classification does not
contradict the interpretation (e.g. calling an open-ended after-cutoff
"Expired").  If it does, one retry is issued with the specific conflict
described.  The check uses no hard-coded labels for specific dates or sentences.
"""
from __future__ import annotations

import logging
from datetime import date

from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm import ModelUsage, invoke_with_fallback
from .models import (
    REFERENCE_DATE,
    ClassificationResult,
    InternalDateClassification,
    TemporalInterpretation,
    TemporalStatus,
)
from .prompts import (
    CLASSIFICATION_PROMPT,
    CLASSIFICATION_RETRY_PROMPT,
    INTERPRETATION_PROMPT,
)

logger = logging.getLogger(__name__)


def _interpret(
    original_text: str,
    normalized_date: str,
    llm: ChatGoogleGenerativeAI,
    fallback_llm: ChatGoogleGenerativeAI | None = None,
) -> tuple[TemporalInterpretation, ModelUsage]:
    result, usage = invoke_with_fallback(
        lambda lm: INTERPRETATION_PROMPT | lm.with_structured_output(TemporalInterpretation),
        {"original_text": original_text, "normalized_date": normalized_date},
        llm,
        fallback_llm,
    )
    logger.info(
        "[Part2] Interpretation: type=%s relation=%s open_ended=%s — %s",
        result.temporal_type,
        result.relation,
        not result.has_explicit_end,
        result.brief_reason,
    )
    return result, usage


def _classify(
    original_text: str,
    normalized_date: str,
    interp: TemporalInterpretation,
    llm: ChatGoogleGenerativeAI,
    fallback_llm: ChatGoogleGenerativeAI | None = None,
    retry_context: dict | None = None,
) -> tuple[InternalDateClassification, ModelUsage]:
    """Run the classification step, optionally with retry context."""
    prompt = CLASSIFICATION_RETRY_PROMPT if retry_context else CLASSIFICATION_PROMPT
    inputs = {
        "original_text": original_text,
        "normalized_date": normalized_date,
        "reference_date": REFERENCE_DATE.isoformat(),
        "temporal_type": interp.temporal_type,
        "relation": interp.relation,
        "open_ended": str(not interp.has_explicit_end),
        **(retry_context or {}),
    }
    result, usage = invoke_with_fallback(
        lambda lm: prompt | lm.with_structured_output(InternalDateClassification),
        inputs,
        llm,
        fallback_llm,
    )
    logger.info("[Part2] Classification: %s — %s", result.status, result.reason)
    return result, usage


def _consistency_conflict(
    interp: TemporalInterpretation,
    status: TemporalStatus,
    normalized_date: str,
) -> str | None:
    """
    Return a human-readable conflict description, or None if consistent.

    Only the unambiguous cases are checked — unclear cases are accepted.
    """
    d = date.fromisoformat(normalized_date)
    ref = REFERENCE_DATE

    # Open-ended after-cutoff in the past: condition still active → must be Ongoing
    if (
        interp.temporal_type == "cutoff_or_threshold"
        and interp.relation == "after"
        and not interp.has_explicit_end
        and d < ref
        and status == TemporalStatus.EXPIRED
    ):
        return (
            f"The interpretation identifies an open-ended cutoff with relation 'after' "
            f"and no explicit end date. The condition is still active past the threshold "
            f"date {normalized_date}, so 'Expired' contradicts 'no explicit end'."
        )

    # Point event before reference date → must be Expired
    if (
        interp.temporal_type == "point_event"
        and d < ref
        and status in (TemporalStatus.UPCOMING, TemporalStatus.ONGOING)
    ):
        return (
            f"The interpretation identifies a point event on {normalized_date}, which is "
            f"before the reference date. '{status}' contradicts a past point event."
        )

    # Point event after reference date → must be Upcoming
    if (
        interp.temporal_type == "point_event"
        and d > ref
        and status in (TemporalStatus.EXPIRED, TemporalStatus.ONGOING)
    ):
        return (
            f"The interpretation identifies a point event on {normalized_date}, which is "
            f"after the reference date. '{status}' contradicts a future point event."
        )

    return None


def classify_date(
    original_text: str,
    normalized_date: str,
    llm: ChatGoogleGenerativeAI,
    fallback_llm: ChatGoogleGenerativeAI | None = None,
) -> ClassificationResult:
    """
    Classify a normalized date as Expired / Upcoming / Ongoing.

    Runs interpret → classify → consistency check → optional single retry.
    """
    logger.info(
        "[Part2] Classifying %s against reference date %s",
        normalized_date,
        REFERENCE_DATE.isoformat(),
    )

    interp, interp_usage = _interpret(original_text, normalized_date, llm, fallback_llm)
    classification, cls_usage = _classify(original_text, normalized_date, interp, llm, fallback_llm)

    conflict = _consistency_conflict(interp, classification.status, normalized_date)
    consistent = conflict is None
    retried = False

    if conflict:
        logger.warning("[Part2] Consistency conflict detected: %s — retrying", conflict)
        classification, cls_usage = _classify(
            original_text,
            normalized_date,
            interp,
            llm,
            fallback_llm,
            retry_context={
                "previous_status": str(classification.status),
                "conflict_reason": conflict,
            },
        )
        retried = True
        remaining_conflict = _consistency_conflict(interp, classification.status, normalized_date)
        if remaining_conflict:
            logger.warning(
                "[Part2] Classification still inconsistent after retry: %s", classification.status
            )

    return ClassificationResult(
        classification=classification,
        interpretation=interp,
        interpretation_usage=interp_usage,
        classification_usage=cls_usage,
        consistency_check_passed=consistent,
        retried=retried,
    )
