"""Unit tests for src/part1/validators.py."""
from __future__ import annotations

import pytest

from src.part1.validators import (
    ExtractionValidationError,
    validate_evidence_in_source,
    validate_table_cell,
    validate_tax_with_evidence,
    validate_value_in_evidence,
)


class TestValidateEvidenceInSource:
    def test_exact_match_passes(self) -> None:
        validate_evidence_in_source("hello world", "prefix hello world suffix")

    def test_whitespace_normalisation_passes(self) -> None:
        # Extra internal whitespace in evidence should still be found
        validate_evidence_in_source(
            "Corporate  Income  Tax",
            "text Corporate Income Tax more text",
        )

    def test_leading_trailing_whitespace_passes(self) -> None:
        validate_evidence_in_source("  hello  ", "say hello please")

    def test_missing_evidence_raises(self) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_evidence_in_source("invented text", "source does not contain it")

    def test_empty_evidence_raises(self) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_evidence_in_source("", "source text")


class TestValidateValueInEvidence:
    def test_value_present_passes(self) -> None:
        validate_value_in_evidence("$28.4 billion", "revised to $28.4 billion which is higher")

    def test_value_absent_raises(self) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_value_in_evidence("$999 billion", "revised to $28.4 billion")

    def test_none_equivalent_raises(self) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_value_in_evidence("", "some evidence text")


class TestValidateTaxWithEvidence:
    SOURCE = (
        "Collections from Corporate Income Tax, Personal Income Tax, "
        "Assets Taxes, Goods and Services Tax, and Betting Taxes."
    )

    def test_valid_tax_passes(self) -> None:
        validate_tax_with_evidence(
            "Corporate Income Tax",
            "Collections from Corporate Income Tax, Personal Income Tax",
            self.SOURCE,
        )

    def test_name_not_in_evidence_raises(self) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_tax_with_evidence(
                "Invented Tax",
                "Collections from Corporate Income Tax",
                self.SOURCE,
            )

    def test_evidence_not_in_source_raises(self) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_tax_with_evidence(
                "Corporate Income Tax",
                "This sentence is not in the source at all.",
                self.SOURCE,
            )

    def test_empty_evidence_raises(self) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_tax_with_evidence("Corporate Income Tax", "", self.SOURCE)


class TestValidateTableCell:
    """Tests via ParsedTable.get_cell (which validate_table_cell delegates to)."""

    @pytest.fixture
    def table(self, sample_parsed_table_page8):
        return sample_parsed_table_page8

    def test_exact_row_and_column_succeeds(self, table) -> None:
        val = validate_table_cell(table, "OVERALL FISCAL POSITION", "Actual FY2022")
        assert val == "1.72"

    def test_partial_row_label_match_succeeds(self, table) -> None:
        # "OVERALL" is a substring of the full label
        val = validate_table_cell(table, "OVERALL FISCAL", "Actual FY2022")
        assert val == "1.72"

    def test_nonexistent_row_raises(self, table) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_table_cell(table, "DOES NOT EXIST ROW", "Actual FY2022")

    def test_nonexistent_column_raises(self, table) -> None:
        with pytest.raises(ExtractionValidationError):
            validate_table_cell(table, "OVERALL FISCAL POSITION", "Nonexistent Column")

    def test_none_cell_raises(self, table) -> None:
        # Column index 3 ("Compared to Actual FY2022") is None for OVERALL FISCAL POSITION
        with pytest.raises(ExtractionValidationError):
            validate_table_cell(
                table,
                "OVERALL FISCAL POSITION",
                "Compared to Actual FY2022",
            )
