"""
LLM temporal classification using a fixed reference date.

The LLM receives both original_text and normalized_date. The source sentence
provides semantic context required to distinguish a past event from an ongoing
condition (e.g. a historical cutoff date that describes a continuing rule).
"""
from __future__ import annotations

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from .models import REFERENCE_DATE, InternalDateClassification
from .prompts import CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


def classify_date(
    original_text: str,
    normalized_date: str,
    llm: ChatGoogleGenerativeAI,
) -> InternalDateClassification:
    """
    Classify a normalized date as Expired / Upcoming / Ongoing.

    The LLM receives both the original source sentence and the ISO date so it can
    consider semantic context — a past cutoff date may describe an ongoing rule.
    Python enforces the output schema; it does not hardcode the classification.
    """
    logger.info(
        "[Part2] Classifying %s against reference date %s",
        normalized_date,
        REFERENCE_DATE.isoformat(),
    )

    chain = CLASSIFICATION_PROMPT | llm.with_structured_output(InternalDateClassification)
    result: InternalDateClassification = chain.invoke(
        {
            "original_text": original_text,
            "normalized_date": normalized_date,
            "reference_date": REFERENCE_DATE.isoformat(),
        }
    )

    logger.info("[Part2] Classification: %s — %s", result.status, result.reason)

    return result
