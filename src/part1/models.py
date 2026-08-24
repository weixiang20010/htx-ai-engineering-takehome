"""Pydantic models for the Part 1 extraction pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field


class NormalizedNumber(BaseModel):
    """A financial value with its original source representation preserved."""

    raw_value: str = Field(description="Exact string from the source document")
    source_unit: str = Field(
        description="Unit as stated in the source (e.g. 'billion', '$ million', '%')"
    )
    normalized_value: float = Field(
        description="Python float after deterministic normalization"
    )


class CorporateTaxEvidence(BaseModel):
    """LLM-extracted evidence for the Corporate Income Tax fields on page 5."""

    source_page: int = Field(description="Page number the evidence was drawn from")
    evidence_text: str | None = Field(
        default=None,
        description=(
            "Verbatim sentence(s) from the source that support both values. "
            "Must be copied exactly from the document — do not paraphrase."
        ),
    )
    amount_text: str | None = Field(
        default=None,
        description=(
            "Exact sub-string from evidence_text that states the Corporate Income Tax "
            "amount (e.g. '$28.4 billion').  Must appear literally in evidence_text."
        ),
    )
    yoy_text: str | None = Field(
        default=None,
        description=(
            "Exact sub-string from evidence_text that states the year-on-year "
            "percentage change (e.g. '17.0%').  Must appear literally in evidence_text."
        ),
    )


class OperatingRevenueTaxes(BaseModel):
    """LLM-extracted list of taxes from the Operating Revenue section."""

    source_pages: list[int] = Field(
        description="Page numbers that were searched"
    )
    taxes: list[str] = Field(
        description=(
            "Tax or revenue-category names exactly as they appear in the source document. "
            "Do not add categories based on general knowledge."
        )
    )


class TableCellSelection(BaseModel):
    """LLM-identified row and column for a subsequent deterministic table lookup."""

    source_page: int = Field(description="Page number of the table")
    row_label: str = Field(
        description="Exact row label text as it appears in the table"
    )
    column_label: str = Field(
        description="Exact column header text as it appears in the table"
    )


class FieldEvidence(BaseModel):
    """Audit record for a single extracted field."""

    field_name: str
    source_page: int
    source_evidence: str
    raw_value: str
    source_unit: str
    normalized_value: float | None
    validation_passed: bool
    validation_note: str = ""


class Part1Result(BaseModel):
    """Final validated output for Part 1."""

    corporate_income_tax_2024: float
    corporate_income_tax_yoy_pct_2024: float
    total_top_ups_2024: float
    operating_revenue_taxes: list[str]
    latest_actual_fiscal_position_billions: float
