"""Unit tests for src/part1/models.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.part1.models import (
    CorporateTaxEvidence,
    FieldEvidence,
    NormalizedNumber,
    OperatingRevenueTaxes,
    Part1Result,
    TableCellSelection,
)


class TestNormalizedNumber:
    def test_valid_construction(self) -> None:
        n = NormalizedNumber(raw_value="28.4", source_unit="billion", normalized_value=28.4)
        assert n.normalized_value == pytest.approx(28.4)

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            NormalizedNumber(raw_value="28.4", source_unit="billion")  # type: ignore[call-arg]

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            NormalizedNumber(
                raw_value="28.4",
                source_unit="billion",
                normalized_value="not_a_float",  # type: ignore[arg-type]
            )


class TestCorporateTaxEvidence:
    def test_optional_fields_default_none(self) -> None:
        ev = CorporateTaxEvidence(source_page=5)
        assert ev.evidence_text is None
        assert ev.amount_text is None
        assert ev.yoy_text is None

    def test_full_construction(self) -> None:
        ev = CorporateTaxEvidence(
            source_page=5,
            evidence_text="Corporate Income Tax is $28.4 billion, 17.0% higher.",
            amount_text="$28.4 billion",
            yoy_text="17.0%",
        )
        assert ev.source_page == 5

    def test_wrong_page_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            CorporateTaxEvidence(source_page="five")  # type: ignore[arg-type]


class TestOperatingRevenueTaxes:
    def test_valid(self) -> None:
        r = OperatingRevenueTaxes(source_pages=[5, 6], taxes=["Corporate Income Tax"])
        assert len(r.taxes) == 1

    def test_empty_taxes_allowed(self) -> None:
        r = OperatingRevenueTaxes(source_pages=[5], taxes=[])
        assert r.taxes == []

    def test_missing_taxes_raises(self) -> None:
        with pytest.raises(ValidationError):
            OperatingRevenueTaxes(source_pages=[5])  # type: ignore[call-arg]


class TestTableCellSelection:
    def test_valid(self) -> None:
        s = TableCellSelection(
            source_page=8,
            row_label="OVERALL FISCAL POSITION",
            column_label="Actual FY2022",
        )
        assert s.source_page == 8

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            TableCellSelection(source_page=8, row_label="some row")  # type: ignore[call-arg]


class TestPart1Result:
    def test_valid_construction(self) -> None:
        r = Part1Result(
            corporate_income_tax_2024=28.4,
            corporate_income_tax_yoy_pct_2024=17.0,
            total_top_ups_2024=20352.0,
            operating_revenue_taxes=["Corporate Income Tax", "GST"],
            latest_actual_fiscal_position_billions=1.72,
        )
        assert r.operating_revenue_taxes == ["Corporate Income Tax", "GST"]

    def test_invalid_float_raises(self) -> None:
        with pytest.raises(ValidationError):
            Part1Result(
                corporate_income_tax_2024="not_a_float",  # type: ignore[arg-type]
                corporate_income_tax_yoy_pct_2024=17.0,
                total_top_ups_2024=20352.0,
                operating_revenue_taxes=[],
                latest_actual_fiscal_position_billions=1.72,
            )

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            Part1Result(  # type: ignore[call-arg]
                corporate_income_tax_2024=28.4,
                corporate_income_tax_yoy_pct_2024=17.0,
                total_top_ups_2024=20352.0,
                operating_revenue_taxes=[],
                # missing latest_actual_fiscal_position_billions
            )
