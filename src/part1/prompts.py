"""
LangChain prompt templates for Part 1 extraction.

All prompt strings live here so that business logic in extractor.py remains
free of large inline strings.  Each template is a ChatPromptTemplate that can
be composed with a structured-output LLM chain.

Design principles applied to every improved prompt
---------------------------------------------------
- Grounding:     model is restricted to the supplied document text only.
- No guessing:   explicit instruction to return None if evidence is absent.
- Evidence:      narrative prompts require verbatim source text to be returned
                 alongside the extracted value.
- Structured:    all improved prompts target a Pydantic output schema.
- No arithmetic: table prompts ask for row/column labels, not the value itself.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------

_GROUNDING_INSTRUCTIONS = """\
STRICT INSTRUCTIONS — follow exactly:
1. Use ONLY the document content provided below.
2. Do NOT use prior knowledge, training data, or external sources.
3. Do NOT infer, calculate, or estimate missing values.
4. If the requested information is not present in the supplied text, return \
null for that field.
5. Preserve numerical text EXACTLY as it appears in the source — do not round, \
abbreviate, or rewrite.
6. Evidence text must be copied verbatim from the document (whitespace may be \
normalised)."""

# ---------------------------------------------------------------------------
# Extraction A — Corporate Income Tax (page 5)
# ---------------------------------------------------------------------------

CORPORATE_TAX_IMPROVED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""{_GROUNDING_INSTRUCTIONS}

You are extracting two specific values from page 5 of a government budget \
document.

OUTPUT SCHEMA:
  source_page  : integer — must be 5
  evidence_text: verbatim sentence(s) from the source that contain BOTH values
  amount_text  : exact sub-string from evidence_text stating the Corporate \
Income Tax amount
  yoy_text     : exact sub-string from evidence_text stating the \
year-on-year percentage change

IMPORTANT:
- amount_text and yoy_text must both be literal sub-strings of evidence_text.
- If either value is not present, return null for that field (and for \
evidence_text if nothing supports the extraction).
- Do not combine information from multiple non-adjacent sentences into \
evidence_text.""",
        ),
        (
            "human",
            """DOCUMENT CONTENT — page 5:
{context}

TASK:
From the "Operating Revenue" section, identify:
  1. The Corporate Income Tax collection amount (as a currency figure, e.g. \
"$28.4 billion")
  2. The year-on-year (YOY) percentage change for Corporate Income Tax

Return your findings according to the output schema.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Extraction A — naive comparison prompt (free-form, no grounding)
# ---------------------------------------------------------------------------

CORPORATE_TAX_NAIVE = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Extract the Corporate Income Tax information from this text:

{context}""",
        )
    ]
)

# ---------------------------------------------------------------------------
# Extraction B — Operating Revenue taxes (pages 5–6)
# ---------------------------------------------------------------------------

OPERATING_REVENUE_TAXES_IMPROVED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""{_GROUNDING_INSTRUCTIONS}

You are identifying taxes and tax categories from a specific section of a \
government budget document.

OUTPUT SCHEMA:
  source_pages : list of page numbers searched
  taxes        : list of objects, each containing:
    name           : tax name exactly as it appears in the source
    evidence_text  : verbatim sentence from the source that names this tax

CLASSIFICATION RULES — apply strictly:
  INCLUDE items that are explicitly labelled a "tax" in the document, including:
    income taxes, consumption taxes, excise duties, asset taxes,
    gaming/betting taxes, stamp duties, carbon taxes, withholding taxes,
    and items explicitly grouped under a "tax" heading in the document.

  EXCLUDE non-tax revenue items such as fees, premiums, charges, or quotas.
    For example: "Vehicle Quota Premiums" are quota fees — NOT a tax — and
    must NOT appear in your output.

  Do NOT add taxes based on general knowledge.
  evidence_text for each item must be copied verbatim from the document.
  The tax name must appear as a literal substring of its evidence_text.""",
        ),
        (
            "human",
            """DOCUMENT CONTENT — pages 5–6 (Operating Revenue section):
{context}

TASK:
List every TAX or TAX CATEGORY explicitly named within the "Operating Revenue" \
section.
For each, return the exact name and a verbatim supporting sentence.

Do NOT include "Vehicle Quota Premiums" or any other fees, charges, or premiums.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Extraction C — Fiscal Position table (page 8)
# ---------------------------------------------------------------------------

FISCAL_POSITION_IMPROVED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""{_GROUNDING_INSTRUCTIONS}

You are identifying the correct row and column in a financial table.

OUTPUT SCHEMA:
  source_page  : integer — the page number of the table
  row_label    : exact row label from the table
  column_label : exact column header from the table

CRITICAL:
- Do NOT return the numerical value itself — return only the labels.
- "Latest Actual" means the most recent year for which ACTUAL (not estimated \
or revised) figures have been published.
- The row_label and column_label you return must appear literally in the table \
below.""",
        ),
        (
            "human",
            """TABLE CONTENT:
{table_text}

TASK:
The assessment asks for the "Latest Actual Fiscal Position in billions."

1. Identify which row contains the overall fiscal position.
2. Identify which column contains the LATEST ACTUAL (not revised, not \
estimated) values.
3. Return the exact row_label and column_label from the table.

Remember: return ONLY the labels — not the number.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Extraction D — Top-ups table (page 20)
# ---------------------------------------------------------------------------

TOP_UPS_IMPROVED = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""{_GROUNDING_INSTRUCTIONS}

You are identifying the correct row and column in a financial table.

OUTPUT SCHEMA:
  source_page  : integer — the page number of the table
  row_label    : exact row label from the table
  column_label : exact column header from the table

CRITICAL:
- Do NOT return the numerical value itself — return only the labels.
- The row_label and column_label must appear literally in the table below.""",
        ),
        (
            "human",
            """TABLE CONTENT:
{table_text}

TASK:
The assessment asks for the "Total amount of top-ups in 2024."

1. Identify the row that represents the TOTAL of all top-up fund amounts.
2. Identify the column that contains the FY2024 estimated amounts.
3. Return the exact row_label and column_label from the table.

Remember: return ONLY the labels — not the number.""",
        ),
    ]
)
