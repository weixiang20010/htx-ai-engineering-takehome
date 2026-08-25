"""LangChain ChatPromptTemplate instances for Part 2."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

_GROUNDING = (
    "You are an information extraction assistant. "
    "Use ONLY the document content provided. "
    "Do NOT use outside knowledge or training data. "
    "Copy text exactly as it appears in the source — do not paraphrase or normalise. "
    "Return null for both fields if the requested information is absent."
)

DISTRIBUTION_DATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _GROUNDING),
        (
            "human",
            """DOCUMENT CONTENT — page 1:
{context}

TASK:
Find the document distribution date — the date on which this document was distributed or published.

Return:
  source_page   : 1
  original_text : verbatim sentence or phrase from the source that states the distribution date
  date_text     : the exact date string from that sentence, copied without modification or normalisation

Do not infer or compute a date. Do not normalise the date format.""",
        ),
    ]
)

ESTATE_DUTY_DATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _GROUNDING),
        (
            "human",
            """DOCUMENT CONTENT — page 36:
{context}

TASK:
Find the date mentioned in relation to Estate Duty — specifically the date from which Estate Duty
rules changed, ceased to apply, or were otherwise referenced as a cutoff.

Return:
  source_page   : 36
  original_text : verbatim phrase or sentence from the source that contains this date
  date_text     : the exact date string from that phrase, copied without modification or normalisation

Do not infer or compute a date. Do not normalise the date format.""",
        ),
    ]
)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are classifying a date relative to a fixed reference date.

Reference date: {reference_date}

Allowed classifications:
  Expired  — The relevant date, event, or period has already ended before the reference date.
  Upcoming — The relevant date or event occurs after the reference date.
  Ongoing  — The source describes a period, condition, or rule that started before or on the
              reference date and remains active on the reference date.

Rules:
- Use BOTH the original_text (semantic context) and the normalized_date.
- The reference date is {reference_date}. Do NOT use today's date or any other date.
- Do not use external knowledge.
- Do not modify the normalized_date.
- Return exactly one status from: Expired, Upcoming, Ongoing.
- Give a concise rationale (one or two sentences). Do not output chain-of-thought.""",
        ),
        (
            "human",
            """Original text from source document:
"{original_text}"

Normalized date: {normalized_date}

Classify this date relative to the reference date {reference_date}.""",
        ),
    ]
)
