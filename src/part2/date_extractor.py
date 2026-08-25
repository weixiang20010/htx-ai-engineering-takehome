"""
Gemini-based date extraction from PDF page text.

The LLM identifies the relevant source sentence and date string.
Python validates the extraction against the source before normalization.
"""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.part1.validators import ExtractionValidationError, validate_evidence_in_source, validate_value_in_evidence
from .models import ExtractedDate

logger = logging.getLogger(__name__)


def _validate_date_extraction(extracted: ExtractedDate, page_text: str) -> None:
    """
    Verify the LLM extraction is grounded in the source page text.

    Checks:
      1. Both fields are non-null and non-empty.
      2. date_text appears verbatim in the source page.
      3. date_text appears within original_text (LLM did not fabricate the date).

    Note: original_text is NOT required to be verbatim in the page text because
    page 36 uses a two-column layout where pdfplumber interleaves columns, making
    verbatim matching of multi-sentence spans unreliable.
    """
    if not extracted.original_text:
        raise ExtractionValidationError(
            f"Page {extracted.source_page}: LLM returned null original_text"
        )
    if not extracted.date_text:
        raise ExtractionValidationError(
            f"Page {extracted.source_page}: LLM returned null date_text"
        )

    # date_text must exist verbatim in the source page — primary grounding check
    validate_evidence_in_source(
        extracted.date_text,
        page_text,
        field_name=f"date_text:page{extracted.source_page}",
    )

    # date_text must appear within original_text — LLM did not fabricate it
    validate_value_in_evidence(
        extracted.date_text,
        extracted.original_text,
        field_name=f"date_text:page{extracted.source_page}",
    )


def extract_date(
    page_text: str,
    prompt: ChatPromptTemplate,
    llm: ChatGoogleGenerativeAI,
    page_num: int,
) -> ExtractedDate:
    """
    Use Gemini to identify a target date and its source sentence on a PDF page.

    Parameters
    ----------
    page_text:
        pdfplumber-extracted text from the target page.
    prompt:
        Targeted ChatPromptTemplate for this page/date type.
    llm:
        Configured Gemini LLM instance.
    page_num:
        Printed page number (for logging and error messages).

    Returns
    -------
    ExtractedDate
        Validated extraction — never contains null fields.
    """
    logger.info("[Part2] Extracting date from page %d", page_num)

    chain = prompt | llm.with_structured_output(ExtractedDate)
    result: ExtractedDate = chain.invoke({"context": page_text})

    logger.debug("[Part2] Page %d LLM result: %s", page_num, result)

    _validate_date_extraction(result, page_text)

    logger.info("[Part2] Page %d source evidence validated", page_num)

    return result
