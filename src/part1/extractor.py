"""
Part 1 extraction pipeline — LangChain + Gemini.

Architecture
------------
Each extraction task follows the same pattern:

  1. PDF parser  →  structured Python object (text / ParsedTable)
  2. LLM call    →  semantic identification (evidence text or row/column labels)
  3. Validation  →  verify LLM output against source content (deterministic)
  4. Retrieval   →  obtain the raw value from the parsed source
  5. Normalise   →  convert raw string to float (deterministic)

The LLM is never trusted to produce numerical answers directly.  For narrative
extractions the evidence is verified verbatim in the source.  For table
extractions the LLM identifies the cell location and Python retrieves the
actual value from the parsed table.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from .models import (
    CorporateTaxEvidence,
    FieldEvidence,
    NormalizedNumber,
    OperatingRevenueTaxes,
    Part1Result,
    TableCellSelection,
)
from .normalizers import normalize_to_float
from .pdf_parser import (
    ParsedTable,
    extract_table_page20,
    extract_table_page8,
    get_pages_text,
)
from .prompts import (
    CORPORATE_TAX_IMPROVED,
    CORPORATE_TAX_NAIVE,
    FISCAL_POSITION_IMPROVED,
    OPERATING_REVENUE_TAXES_IMPROVED,
    TOP_UPS_IMPROVED,
)
from .validators import (
    ExtractionValidationError,
    validate_evidence_in_source,
    validate_table_cell,
    validate_taxes_in_source,
    validate_value_in_evidence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def build_llm(
    model: str | None = None,
    api_key: str | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Construct a ChatGoogleGenerativeAI instance.

    Falls back to environment variables when parameters are not supplied.
    """
    resolved_model = model or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
    if not resolved_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.  "
            "Provide it via the environment variable or the api_key argument."
        )
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=resolved_key,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_normalized(
    raw_value: str,
    source_unit: str,
) -> NormalizedNumber:
    """Wrap a raw string + unit into a NormalizedNumber."""
    return NormalizedNumber(
        raw_value=raw_value,
        source_unit=source_unit,
        normalized_value=normalize_to_float(raw_value),
    )


def _detect_unit(text: str) -> str:
    """Infer the source unit from a financial text fragment."""
    t = text.lower()
    if "billion" in t:
        return "billion"
    if "million" in t:
        return "$ million"
    if "%" in t:
        return "%"
    if "$" in t:
        return "$"
    return ""


# ---------------------------------------------------------------------------
# Extraction A — Corporate Income Tax (page 5)
# ---------------------------------------------------------------------------


def extract_corporate_tax(
    page5_text: str,
    llm: ChatGoogleGenerativeAI,
) -> tuple[NormalizedNumber, NormalizedNumber, FieldEvidence, FieldEvidence]:
    """
    Extract Corporate Income Tax amount and YOY percentage from page 5.

    Returns
    -------
    amount_num:
        Normalized amount.
    yoy_num:
        Normalized YOY percentage.
    amount_evidence:
        Audit record for the amount field.
    yoy_evidence:
        Audit record for the YOY field.
    """
    logger.info("Extraction A: Corporate Income Tax (page 5)")

    chain = CORPORATE_TAX_IMPROVED | llm.with_structured_output(CorporateTaxEvidence)
    evidence: CorporateTaxEvidence = chain.invoke({"context": page5_text})

    logger.debug("LLM evidence: %s", evidence.model_dump())

    # --- Validate ---
    amount_passed = True
    amount_note = ""
    yoy_passed = True
    yoy_note = ""

    if evidence.evidence_text is None:
        amount_passed = yoy_passed = False
        amount_note = yoy_note = "LLM returned no evidence_text"
        raise ExtractionValidationError(
            "No evidence_text returned for Corporate Income Tax."
        )

    # Verify evidence exists verbatim in source
    validate_evidence_in_source(
        evidence.evidence_text, page5_text, field_name="corporate_tax_evidence"
    )

    # Verify amount sub-string appears in evidence
    if evidence.amount_text:
        try:
            validate_value_in_evidence(
                evidence.amount_text,
                evidence.evidence_text,
                field_name="corporate_tax_amount",
            )
        except ExtractionValidationError as exc:
            amount_passed = False
            amount_note = str(exc)
            raise
    else:
        amount_passed = False
        amount_note = "LLM returned no amount_text"
        raise ExtractionValidationError("amount_text is None for Corporate Income Tax.")

    # Verify YOY sub-string appears in evidence
    if evidence.yoy_text:
        try:
            validate_value_in_evidence(
                evidence.yoy_text,
                evidence.evidence_text,
                field_name="corporate_tax_yoy",
            )
        except ExtractionValidationError as exc:
            yoy_passed = False
            yoy_note = str(exc)
            raise
    else:
        yoy_passed = False
        yoy_note = "LLM returned no yoy_text"
        raise ExtractionValidationError("yoy_text is None for Corporate Income Tax.")

    # --- Normalize ---
    amount_unit = _detect_unit(evidence.amount_text)
    yoy_unit = "%"

    amount_num = _make_normalized(evidence.amount_text, amount_unit)
    yoy_num = _make_normalized(evidence.yoy_text, yoy_unit)

    logger.info(
        "Extraction A complete: amount=%.2f %s, yoy=%.1f%%",
        amount_num.normalized_value,
        amount_unit,
        yoy_num.normalized_value,
    )

    amount_ev = FieldEvidence(
        field_name="corporate_income_tax_2024",
        source_page=5,
        source_evidence=evidence.evidence_text,
        raw_value=evidence.amount_text,
        source_unit=amount_unit,
        normalized_value=amount_num.normalized_value,
        validation_passed=amount_passed,
        validation_note=amount_note,
    )
    yoy_ev = FieldEvidence(
        field_name="corporate_income_tax_yoy_pct_2024",
        source_page=5,
        source_evidence=evidence.evidence_text,
        raw_value=evidence.yoy_text,
        source_unit=yoy_unit,
        normalized_value=yoy_num.normalized_value,
        validation_passed=yoy_passed,
        validation_note=yoy_note,
    )
    return amount_num, yoy_num, amount_ev, yoy_ev


# ---------------------------------------------------------------------------
# Extraction B — Operating Revenue taxes (pages 5–6)
# ---------------------------------------------------------------------------


def extract_operating_revenue_taxes(
    page5_text: str,
    page6_text: str,
    llm: ChatGoogleGenerativeAI,
) -> tuple[list[str], FieldEvidence]:
    """
    Extract all tax names mentioned in the Operating Revenue section (pp. 5–6).

    Returns
    -------
    taxes:
        Validated list of tax names.
    evidence:
        Audit record.
    """
    logger.info("Extraction B: Operating Revenue taxes (pages 5–6)")

    combined = page5_text + "\n" + page6_text
    chain = OPERATING_REVENUE_TAXES_IMPROVED | llm.with_structured_output(
        OperatingRevenueTaxes
    )
    result: OperatingRevenueTaxes = chain.invoke({"context": combined})

    logger.debug("LLM taxes: %s", result.taxes)

    # --- Validate each tax name against source ---
    validated_taxes = validate_taxes_in_source(result.taxes, combined)
    unsupported = [t for t in result.taxes if t not in validated_taxes]
    if unsupported:
        logger.warning(
            "Extraction B: %d tax name(s) not found in source and dropped: %s",
            len(unsupported),
            unsupported,
        )

    logger.info("Extraction B complete: %d taxes validated", len(validated_taxes))

    ev = FieldEvidence(
        field_name="operating_revenue_taxes",
        source_page=5,
        source_evidence=f"Pages 5–6 Operating Revenue section",
        raw_value=str(validated_taxes),
        source_unit="",
        normalized_value=None,
        validation_passed=True,
        validation_note=(
            f"Dropped {len(unsupported)} unverified name(s): {unsupported}"
            if unsupported
            else ""
        ),
    )
    return validated_taxes, ev


# ---------------------------------------------------------------------------
# Extraction C — Fiscal Position (page 8)
# ---------------------------------------------------------------------------


def extract_fiscal_position(
    table8: ParsedTable,
    llm: ChatGoogleGenerativeAI,
) -> tuple[NormalizedNumber, FieldEvidence]:
    """
    Extract the Latest Actual Fiscal Position from the parsed page-8 table.

    The LLM identifies the semantic row and column labels; Python retrieves
    the actual cell value from the parsed table.
    """
    logger.info("Extraction C: Latest Actual Fiscal Position (page 8)")

    chain = FISCAL_POSITION_IMPROVED | llm.with_structured_output(TableCellSelection)
    selection: TableCellSelection = chain.invoke({"table_text": table8.to_llm_text()})

    logger.debug(
        "LLM selection: row=%r, col=%r", selection.row_label, selection.column_label
    )

    # --- Deterministic retrieval ---
    raw_value = validate_table_cell(table8, selection.row_label, selection.column_label)
    logger.info("Extraction C: raw cell value=%r", raw_value)

    num = _make_normalized(raw_value, "billion")

    ev = FieldEvidence(
        field_name="latest_actual_fiscal_position_billions",
        source_page=8,
        source_evidence=f"Table: {table8.title} | row={selection.row_label!r} | col={selection.column_label!r}",
        raw_value=raw_value,
        source_unit="billion",
        normalized_value=num.normalized_value,
        validation_passed=True,
    )
    return num, ev


# ---------------------------------------------------------------------------
# Extraction D — Total top-ups (page 20)
# ---------------------------------------------------------------------------


def extract_total_top_ups(
    table20: ParsedTable,
    llm: ChatGoogleGenerativeAI,
) -> tuple[NormalizedNumber, FieldEvidence]:
    """
    Extract the total top-up amount from the parsed page-20 table.

    The LLM identifies the semantic row and column; Python retrieves the cell.
    """
    logger.info("Extraction D: Total top-ups (page 20)")

    chain = TOP_UPS_IMPROVED | llm.with_structured_output(TableCellSelection)
    selection: TableCellSelection = chain.invoke({"table_text": table20.to_llm_text()})

    logger.debug(
        "LLM selection: row=%r, col=%r", selection.row_label, selection.column_label
    )

    # --- Deterministic retrieval ---
    raw_value = validate_table_cell(table20, selection.row_label, selection.column_label)
    logger.info("Extraction D: raw cell value=%r", raw_value)

    num = _make_normalized(raw_value, "$ million")

    ev = FieldEvidence(
        field_name="total_top_ups_2024",
        source_page=20,
        source_evidence=f"Table: {table20.title} | row={selection.row_label!r} | col={selection.column_label!r}",
        raw_value=raw_value,
        source_unit="$ million",
        normalized_value=num.normalized_value,
        validation_passed=True,
    )
    return num, ev


# ---------------------------------------------------------------------------
# Naive vs improved demonstration
# ---------------------------------------------------------------------------


def demonstrate_naive_vs_improved(
    page5_text: str,
    llm: ChatGoogleGenerativeAI,
) -> dict[str, object]:
    """
    Run both the naive and improved prompts against page 5 text and return
    a side-by-side comparison dict.

    The naive prompt returns free-form prose.  The improved prompt returns
    a structured CorporateTaxEvidence with evidence grounding.  This comparison
    illustrates why the improved approach is safer for production use.
    """
    logger.info("Demonstration: naive vs improved prompt comparison")

    # Naive — plain string response
    naive_chain = CORPORATE_TAX_NAIVE | llm
    naive_response = naive_chain.invoke({"context": page5_text})
    naive_text: str = getattr(naive_response, "content", str(naive_response))

    # Improved — structured + validated
    improved_chain = CORPORATE_TAX_IMPROVED | llm.with_structured_output(
        CorporateTaxEvidence
    )
    improved_result: CorporateTaxEvidence = improved_chain.invoke(
        {"context": page5_text}
    )

    # Attempt to validate improved output
    improved_validated = False
    improved_note = ""
    try:
        if improved_result.evidence_text:
            validate_evidence_in_source(
                improved_result.evidence_text,
                page5_text,
                field_name="demo_evidence",
            )
            improved_validated = True
    except ExtractionValidationError as exc:
        improved_note = str(exc)

    return {
        "naive": {
            "prompt": "Extract the Corporate Income Tax information from this text.",
            "response_type": "free-form text",
            "response": naive_text,
            "structured": False,
            "evidence_verified": False,
            "notes": (
                "Free-form prose — no schema, no evidence grounding, "
                "no deterministic validation possible."
            ),
        },
        "improved": {
            "prompt": "Grounded structured prompt (see prompts.py CORPORATE_TAX_IMPROVED)",
            "response_type": "CorporateTaxEvidence (Pydantic)",
            "response": improved_result.model_dump(),
            "structured": True,
            "evidence_verified": improved_validated,
            "notes": (
                "Structured output enables deterministic validation of evidence. "
                + (
                    "Evidence verified in source."
                    if improved_validated
                    else f"Validation note: {improved_note}"
                )
            ),
        },
        "comparison": {
            "grounding": "Improved prompt restricts the model to the supplied context; naive allows hallucination.",
            "structured_output": "Improved uses Pydantic schema; naive returns unstructured text.",
            "evidence": "Improved requires verbatim source evidence; naive provides none.",
            "missing_value_handling": "Improved instructs the model to return null if evidence is absent; naive may fabricate.",
            "deterministic_validation": "Improved output can be programmatically verified; naive cannot.",
            "hallucination_resistance": "Improved combines grounding + evidence + deterministic validation; naive has none of these.",
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_extraction(
    pdf_path: str | Path,
    llm: ChatGoogleGenerativeAI,
) -> tuple[Part1Result, list[FieldEvidence]]:
    """
    Run the complete Part 1 extraction pipeline.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    llm:
        Configured ChatGoogleGenerativeAI instance.

    Returns
    -------
    result:
        Final validated Part1Result.
    evidence_records:
        Audit trail with one FieldEvidence per extracted field.
    """
    logger.info("Starting Part 1 extraction pipeline")

    # --- Parse PDF ---
    pages_text = get_pages_text(pdf_path, [5, 6])
    page5_text = pages_text[5]
    page6_text = pages_text[6]
    table8 = extract_table_page8(pdf_path)
    table20 = extract_table_page20(pdf_path)

    logger.info("PDF parsing complete")

    # --- Run extractions ---
    amount_num, yoy_num, amount_ev, yoy_ev = extract_corporate_tax(page5_text, llm)
    taxes, taxes_ev = extract_operating_revenue_taxes(page5_text, page6_text, llm)
    fiscal_num, fiscal_ev = extract_fiscal_position(table8, llm)
    topups_num, topups_ev = extract_total_top_ups(table20, llm)

    evidence_records = [amount_ev, yoy_ev, taxes_ev, fiscal_ev, topups_ev]

    # --- Assemble final result ---
    result = Part1Result(
        corporate_income_tax_2024=amount_num.normalized_value,
        corporate_income_tax_yoy_pct_2024=yoy_num.normalized_value,
        total_top_ups_2024=topups_num.normalized_value,
        operating_revenue_taxes=taxes,
        latest_actual_fiscal_position_billions=fiscal_num.normalized_value,
    )

    logger.info("Part 1 extraction complete: %s", result.model_dump())
    return result, evidence_records
