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

from src.llm import ModelUsage, ainvoke_with_fallback, build_llm_pair, invoke_with_fallback

from .models import (
    CorporateTaxEvidence,
    FieldEvidence,
    NormalizedNumber,
    OperatingRevenueTaxes,
    Part1Result,
    TableCellSelection,
    TaxWithEvidence,
)
from .normalizers import normalize_to_float
from .pdf_parser import (
    ParsedTable,
    extract_operating_revenue_section,
    extract_table_page20,
    extract_table_page8,
    get_pages_text,
)
from .prompts import (
    CORPORATE_TAX_IMPROVED,
    FISCAL_POSITION_IMPROVED,
    OPERATING_REVENUE_TAXES_IMPROVED,
    TOP_UPS_IMPROVED,
)
from .validators import (
    ExtractionValidationError,
    validate_evidence_in_source,
    validate_table_cell,
    validate_tax_with_evidence,
    validate_value_in_evidence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def build_llm(
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI | None]:
    """
    Return (primary_llm, fallback_llm) from environment configuration.

    Reads GEMINI_PRIMARY_MODEL and GEMINI_FALLBACK_MODEL. Accepts explicit
    overrides for testing. Returns fallback=None when GEMINI_FALLBACK_MODEL
    is not set.
    """
    import os

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Provide it via the environment variable or a .env file."
        )
    from src.llm import build_llm as _build_one

    primary_name = model or os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-3.6-flash")
    fallback_name = os.environ.get("GEMINI_FALLBACK_MODEL")

    primary = _build_one(primary_name, api_key)
    fallback = _build_one(fallback_name, api_key) if fallback_name else None
    return primary, fallback


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
    fallback_llm: ChatGoogleGenerativeAI | None = None,
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

    evidence, usage = invoke_with_fallback(
        lambda lm: CORPORATE_TAX_IMPROVED | lm.with_structured_output(CorporateTaxEvidence),
        {"context": page5_text},
        llm,
        fallback_llm,
    )

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
        requested_model=usage.requested_model,
        actual_model=usage.actual_model,
        fallback_used=usage.fallback_used,
        fallback_reason=usage.fallback_reason,
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
        requested_model=usage.requested_model,
        actual_model=usage.actual_model,
        fallback_used=usage.fallback_used,
        fallback_reason=usage.fallback_reason,
    )
    return amount_num, yoy_num, amount_ev, yoy_ev


# ---------------------------------------------------------------------------
# Extraction B — Operating Revenue taxes (pages 5–6)
# ---------------------------------------------------------------------------


def extract_operating_revenue_taxes(
    page5_text: str,
    page6_text: str,
    llm: ChatGoogleGenerativeAI,
    fallback_llm: ChatGoogleGenerativeAI | None = None,
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

    section = extract_operating_revenue_section(page5_text, page6_text)
    combined = page5_text + "\n" + page6_text  # kept for grounding validation
    result, usage = invoke_with_fallback(
        lambda lm: OPERATING_REVENUE_TAXES_IMPROVED | lm.with_structured_output(OperatingRevenueTaxes),
        {"context": section},
        llm,
        fallback_llm,
    )

    logger.debug(
        "LLM taxes (pre-validation): %s", [t.name for t in result.taxes]
    )

    # The LLM decides semantically what counts as a tax.
    # Python verifies: evidence exists in source, name exists in evidence.
    validated: list[str] = []
    dropped: list[str] = []
    for tax in result.taxes:
        try:
            validate_tax_with_evidence(tax.name, tax.evidence_text, combined)
            validated.append(tax.name)
        except ExtractionValidationError as exc:
            logger.warning("Dropping tax %r — evidence not grounded: %s", tax.name, exc)
            dropped.append(tax.name)

    if not validated:
        raise ExtractionValidationError(
            "No tax names passed evidence validation. Check the prompt and source."
        )

    logger.info("Extraction B complete: %d taxes validated", len(validated))

    ev = FieldEvidence(
        field_name="operating_revenue_taxes",
        source_page=5,
        source_evidence="Pages 5–6 Operating Revenue section",
        raw_value=str(validated),
        source_unit="",
        normalized_value=None,
        validation_passed=True,
        validation_note=(
            f"Dropped {len(dropped)} item(s) not grounded in source: {dropped}"
            if dropped else ""
        ),        per_item_evidence=[
            {"name": tax.name, "evidence_text": tax.evidence_text}
            for tax in result.taxes
            if tax.name in validated
        ],
        requested_model=usage.requested_model,
        actual_model=usage.actual_model,
        fallback_used=usage.fallback_used,
        fallback_reason=usage.fallback_reason,
    )
    return validated, ev


# ---------------------------------------------------------------------------
# Extraction C — Fiscal Position (page 8)
# ---------------------------------------------------------------------------


def extract_fiscal_position(
    table8: ParsedTable,
    llm: ChatGoogleGenerativeAI,
    fallback_llm: ChatGoogleGenerativeAI | None = None,
) -> tuple[NormalizedNumber, FieldEvidence]:
    """
    Extract the Latest Actual Fiscal Position from the parsed page-8 table.

    The LLM identifies the semantic row and column labels; Python retrieves
    the actual cell value from the parsed table.
    """
    logger.info("Extraction C: Latest Actual Fiscal Position (page 8)")

    selection, usage = invoke_with_fallback(
        lambda lm: FISCAL_POSITION_IMPROVED | lm.with_structured_output(TableCellSelection),
        {"table_text": table8.to_llm_text()},
        llm,
        fallback_llm,
    )

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
        requested_model=usage.requested_model,
        actual_model=usage.actual_model,
        fallback_used=usage.fallback_used,
        fallback_reason=usage.fallback_reason,
    )
    return num, ev


# ---------------------------------------------------------------------------
# Extraction D — Total top-ups (page 20)
# ---------------------------------------------------------------------------


def extract_total_top_ups(
    table20: ParsedTable,
    llm: ChatGoogleGenerativeAI,
    fallback_llm: ChatGoogleGenerativeAI | None = None,
) -> tuple[NormalizedNumber, FieldEvidence]:
    """
    Extract the total top-up amount from the parsed page-20 table.

    The LLM identifies the semantic row and column; Python retrieves the cell.
    """
    logger.info("Extraction D: Total top-ups (page 20)")

    selection, usage = invoke_with_fallback(
        lambda lm: TOP_UPS_IMPROVED | lm.with_structured_output(TableCellSelection),
        {"table_text": table20.to_llm_text()},
        llm,
        fallback_llm,
    )

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
        requested_model=usage.requested_model,
        actual_model=usage.actual_model,
        fallback_used=usage.fallback_used,
        fallback_reason=usage.fallback_reason,
    )
    return num, ev


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_extraction(
    pdf_path: str | Path,
    llm: ChatGoogleGenerativeAI,
    fallback_llm: ChatGoogleGenerativeAI | None = None,
) -> tuple[Part1Result, list[FieldEvidence]]:
    """
    Run the complete Part 1 extraction pipeline.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    llm:
        Primary ChatGoogleGenerativeAI instance.
    fallback_llm:
        Optional fallback LLM used only on quota/rate-limit errors.
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
    amount_num, yoy_num, amount_ev, yoy_ev = extract_corporate_tax(page5_text, llm, fallback_llm)
    taxes, taxes_ev = extract_operating_revenue_taxes(page5_text, page6_text, llm, fallback_llm)
    fiscal_num, fiscal_ev = extract_fiscal_position(table8, llm, fallback_llm)
    topups_num, topups_ev = extract_total_top_ups(table20, llm, fallback_llm)

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
