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

INTERPRETATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a temporal semantics analyst. Read a sentence that contains a date and
describe how that date functions grammatically and semantically.

Choose:
  temporal_type:
    point_event          — a single event or distribution that occurs on one specific date
    period               — a span of time with a defined start and optional end
    cutoff_or_threshold  — a boundary after/before which a condition, rule, or exemption applies

  relation (the grammatical role of the date relative to the described condition):
    on    — occurs on this date
    after — condition applies after this date
    before — condition applies before this date
    from  — condition begins from this date
    until — condition ends at this date
    between — condition spans from one date to another
    not_applicable — date does not express a directional relation

  has_explicit_end:
    true  — the source text contains a stated end date or expiry for this condition
    false — no end date or expiry is stated in the source text

Return a brief_reason: one sentence explaining your reading.""",
        ),
        (
            "human",
            """Source sentence:
"{original_text}"

Normalized date from that sentence: {normalized_date}

Describe the temporal semantics of this date expression.""",
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
  Expired  — A point event or period that ended before the reference date and has
              no continuing effect on or after it.
  Upcoming — An event or date that occurs after the reference date.
  Ongoing  — A period or condition that began before or on the reference date and
              remains applicable on the reference date.

You are provided with a structured temporal interpretation of the source sentence.
Use it to guide your reasoning — do not contradict it without strong justification.

Rules:
- Use BOTH the original_text (semantic context) and the normalized_date.
- The reference date is {reference_date}. Do NOT use today's date or any other date.
- Do not use external knowledge.
- Return exactly one status from: Expired, Upcoming, Ongoing.
- Give a concise rationale (one or two sentences). Do not output chain-of-thought.""",
        ),
        (
            "human",
            """Original text from source document:
"{original_text}"

Normalized date: {normalized_date}

Temporal interpretation:
  type     : {temporal_type}
  relation : {relation}
  open-ended (no explicit end): {open_ended}

Classify this date relative to the reference date {reference_date}.""",
        ),
    ]
)

CLASSIFICATION_RETRY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are classifying a date relative to a fixed reference date.

Reference date: {reference_date}

Allowed classifications: Expired, Upcoming, Ongoing.
Give a concise rationale (one or two sentences). Do not output chain-of-thought.""",
        ),
        (
            "human",
            """Your previous classification conflicted with the temporal interpretation you were given.

Original text: "{original_text}"
Normalized date: {normalized_date}

Temporal interpretation:
  type     : {temporal_type}
  relation : {relation}
  open-ended (no explicit end): {open_ended}

Previous classification: {previous_status}
Conflict: {conflict_reason}

Re-evaluate. Classify this date relative to the reference date {reference_date}.""",
        ),
    ]
)
