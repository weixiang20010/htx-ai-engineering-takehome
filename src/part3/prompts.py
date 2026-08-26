"""
ChatPromptTemplates for Part 3 LangGraph supervisor and specialist agents.

Design principles
-----------------
- Prompts define the general reasoning framework; they do NOT encode
  expected answers (those belong in integration tests).
- LLMs are used for semantic decisions: routing, evidence assessment,
  query reformulation, fact extraction, synthesis.
- Deterministic code handles retrieval mechanics, grounding validation,
  and unit conversion.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Supervisor — routing decision
# ---------------------------------------------------------------------------

SUPERVISOR_ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a supervisor that routes financial queries to specialist agents.

Available agents:
  revenue      — Specialises in government revenue: Total Revenue, Operating Revenue,
                  taxes, non-tax revenue, Net Investment Returns Contribution (NIRC).
  expenditure  — Specialises in government spending: Total Expenditure, Operating
                  Expenditure, Development Expenditure, Special Transfers, top-ups to
                  endowment and trust funds, specific allocations and their purposes.

Your ONLY job is to determine which agent(s) should handle the query and what each
should investigate.

Rules:
- Select revenue if the query is about government income, taxes, or revenue streams.
- Select expenditure if the query is about spending, funds, allocations, or transfers.
- Select both when the query genuinely requires both domains.
- Do NOT attempt to answer the financial question yourself.
- The reason field must be one short sentence explaining the routing choice.
- required_aspects must be specific, verifiable sub-questions derived from the query.""",
        ),
        (
            "human",
            "Query: {query}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Specialist — evidence assessment
# ---------------------------------------------------------------------------

EVIDENCE_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are assessing whether retrieved document evidence is sufficient to
support a delegated task.

Delegated task: {delegated_task}

Required aspects that must be supported:
{required_aspects}

Instructions:
- Read the retrieved evidence carefully.
- For each required aspect, determine whether it is clearly supported by the text.
- An aspect is supported only if the evidence contains specific relevant information
  about it — not just a mention.
- If all required aspects are supported, set sufficient=true.
- If any aspect is unsupported, set sufficient=false and identify exactly which
  aspects are missing.
- If sufficient=false, provide next_search_query targeting the missing aspects.
  The query must be specific and directly address the missing information.
  Do not broaden the topic.""",
        ),
        (
            "human",
            """Retrieved evidence (from document source):
{evidence_text}

Assess sufficiency against the required aspects.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Specialist — query reformulation
# ---------------------------------------------------------------------------

QUERY_REFORMULATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are reformulating a document search query to improve retrieval.

Original delegated task: {delegated_task}

Your goal is to retrieve evidence for the SAME task. Preserve the exact intent.
Do not broaden the topic or introduce concepts not present in the task or evidence.

Missing aspects still needed:
{missing_aspects}

Previous queries already tried:
{previous_queries}

Return ONLY the new search query string — no explanation.""",
        ),
        (
            "human",
            "Reformulate the query to target the missing aspects.",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Specialist — grounded fact extraction
# ---------------------------------------------------------------------------

FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are extracting structured facts from document evidence.

Role context: {role_prompt}

Delegated task: {delegated_task}

Required aspects that must be covered:
{required_aspects}

For each relevant fact:
  claim            — State the fact clearly and concisely.
  source_page      — The page number from the source document.
  evidence_text    — Copy the EXACT verbatim sentence or line from the evidence that
                     supports the claim. Do not paraphrase.
  amount_text      — If the fact involves a monetary or numerical amount, copy the
                     exact amount string as it appears (e.g. "5,000" or "28.4").
  source_unit      — If a unit is stated near the amount (e.g. "$ million", "$ billion"),
                     copy it exactly.
  supports_aspects — List which of the required aspects above this fact directly
                     addresses. Use the exact wording from the required aspects list.
                     May contain one or more aspects. Never include aspects not in
                     the required list.

Rules:
- Use ONLY information present in the supplied evidence.
- Do not invent or infer figures not explicitly stated.
- Each fact must have a unique, specific claim.
- summary must be 1-3 sentences synthesising the main findings.""",
        ),
        (
            "human",
            """Evidence:
{evidence_text}

Extract facts relevant to the delegated task.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Synthesis — final answer
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are synthesizing findings from specialist agents into a final answer.

Rules:
- Use ONLY the supplied grounded agent findings below.
- Do NOT independently search the document or add outside knowledge.
- Do not introduce monetary figures or claims absent from the findings.
- Include page references [p. N] when citing specific facts.
- If an agent's status is insufficient_evidence, clearly state that the
  corresponding part of the question could not be fully answered from the document.
- If one agent succeeded and one did not, provide the supported partial answer.
- Answer all parts of the original query that are supported.""",
        ),
        (
            "human",
            """Original query: {query}

Agent findings:
{agent_findings}

Provide the final answer.""",
        ),
    ]
)
