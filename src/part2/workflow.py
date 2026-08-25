"""
Part 2 main workflow — async orchestration of extraction, MCP normalisation, and classification.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from mcp import ClientSession

from src.part1.extractor import build_llm  # noqa: F401 — re-exported for convenience
from src.part1.pdf_parser import get_pages_text
from src.part1.validators import ExtractionValidationError

from .classifier import classify_date
from .date_extractor import extract_date
from .mcp_client import open_mcp_session
from .models import REFERENCE_DATE, Part2Evidence, Part2ResultItem
from .prompts import DISTRIBUTION_DATE_PROMPT, ESTATE_DUTY_DATE_PROMPT

logger = logging.getLogger(__name__)


async def _normalize_via_mcp(
    date_text: str,
    source_page: int,
    session: ClientSession,
    lc_tools: list[BaseTool],
    llm: ChatGoogleGenerativeAI,
) -> tuple[str, dict]:
    """
    Ask Gemini to call the normalize_date MCP tool and execute it via the session.

    The LLM's tool_calls are verified before execution — if Gemini does not call
    the expected tool, an error is raised rather than silently falling back.

    Returns (iso_date, tool_trace).
    """
    llm_with_tools = llm.bind_tools(lc_tools)
    prompt = (
        f"Use the normalize_date tool to convert this date to ISO YYYY-MM-DD format.\n"
        f'Date: "{date_text}"\n'
        f"Call the tool exactly once. Do not perform the conversion yourself."
    )
    response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])

    if not response.tool_calls:
        raise ExtractionValidationError(
            f"Page {source_page}: Gemini did not call normalize_date for {date_text!r}"
        )

    tool_call = response.tool_calls[0]

    if tool_call["name"] != "normalize_date":
        raise ExtractionValidationError(
            f"Page {source_page}: expected tool 'normalize_date', got {tool_call['name']!r}"
        )

    logger.info("[Part2] Gemini requested tool: %s", tool_call["name"])

    # Execute through MCP session directly — not via the LangChain tool wrapper —
    # so the raw call/result is captured for the audit trace
    mcp_result = await session.call_tool(tool_call["name"], tool_call["args"])

    if mcp_result.isError:
        raise ExtractionValidationError(
            f"Page {source_page}: MCP tool returned an error for {date_text!r}"
        )

    normalized = mcp_result.content[0].text
    logger.info("[Part2] MCP returned normalized date: %s", normalized)

    trace = {
        "tool_requested": tool_call["name"],
        "tool_arguments": tool_call["args"],
        "tool_result": normalized,
    }
    return normalized, trace


async def run_part2(
    pdf_path: str | Path,
    llm: ChatGoogleGenerativeAI,
) -> tuple[list[str], list[Part2ResultItem], list[Part2Evidence]]:
    """
    Execute the full Part 2 pipeline.

    Returns
    -------
    normalized_dates : list[str]
        Plain ISO strings for part2_normalized_dates.json.
    results : list[Part2ResultItem]
        HTX-schema items for part2_result.json.
    evidence : list[Part2Evidence]
        Full audit records for part2_evidence.json.
    """
    # ── Stage 1 ─ PDF extraction ─────────────────────────────────────────────
    logger.info("[Part2] Extracting pages 1 and 36 from PDF")
    texts = get_pages_text(pdf_path, printed_pages=[1, 36])

    # ── Stage 1 ─ Gemini date extraction ─────────────────────────────────────
    logger.info("[Part2] Extracting distribution date from page 1")
    extracted1 = extract_date(texts[1], DISTRIBUTION_DATE_PROMPT, llm, page_num=1)

    logger.info("[Part2] Extracting Estate Duty date from page 36")
    extracted36 = extract_date(texts[36], ESTATE_DUTY_DATE_PROMPT, llm, page_num=36)

    # ── Stage 1 ─ MCP normalisation ──────────────────────────────────────────
    logger.info("[Part2] Opening MCP session for date normalisation")
    async with open_mcp_session() as (session, lc_tools):
        normalized1, trace1 = await _normalize_via_mcp(
            extracted1.date_text, 1, session, lc_tools, llm
        )
        normalized2, trace2 = await _normalize_via_mcp(
            extracted36.date_text, 36, session, lc_tools, llm
        )

    normalized_dates = [normalized1, normalized2]

    # ── Stage 2 ─ LLM temporal classification ────────────────────────────────
    logger.info("[Part2] Classifying dates against %s", REFERENCE_DATE.isoformat())
    classification1 = classify_date(extracted1.original_text, normalized1, llm)
    classification2 = classify_date(extracted36.original_text, normalized2, llm)

    results = [
        Part2ResultItem(
            original_text=classification1.original_text,
            normalized_date=classification1.normalized_date,
            status=classification1.status,
        ),
        Part2ResultItem(
            original_text=classification2.original_text,
            normalized_date=classification2.normalized_date,
            status=classification2.status,
        ),
    ]

    evidence = [
        Part2Evidence(
            source_page=1,
            source_statement=extracted1.original_text,
            extracted_date_text=extracted1.date_text,
            source_validation_passed=True,
            mcp_tool_requested=trace1["tool_requested"],
            mcp_tool_arguments=trace1["tool_arguments"],
            mcp_tool_result=trace1["tool_result"],
            reference_date=REFERENCE_DATE.isoformat(),
            llm_status=str(classification1.status),
            llm_rationale=classification1.reason,
        ),
        Part2Evidence(
            source_page=36,
            source_statement=extracted36.original_text,
            extracted_date_text=extracted36.date_text,
            source_validation_passed=True,
            mcp_tool_requested=trace2["tool_requested"],
            mcp_tool_arguments=trace2["tool_arguments"],
            mcp_tool_result=trace2["tool_result"],
            reference_date=REFERENCE_DATE.isoformat(),
            llm_status=str(classification2.status),
            llm_rationale=classification2.reason,
        ),
    ]

    logger.info("[Part2] Final output written")

    return normalized_dates, results, evidence
