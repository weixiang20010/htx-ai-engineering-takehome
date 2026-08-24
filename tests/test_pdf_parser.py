"""Unit tests for src/part1/pdf_parser.py."""
from __future__ import annotations

import pytest

from src.part1.pdf_parser import (
    ParsedTable,
    _clean_sparse_row,
    _is_garbled_coordinate_line,
    _is_numeric_token,
    _split_label_values,
    extract_table_page20,
    extract_table_page8,
    get_page_text,
    get_pages_text,
    page_index,
)
from src.part1.validators import ExtractionValidationError


# ---------------------------------------------------------------------------
# page_index
# ---------------------------------------------------------------------------


class TestPageIndex:
    def test_page_1(self) -> None:
        assert page_index(1) == 0

    def test_page_5(self) -> None:
        assert page_index(5) == 4

    def test_page_20(self) -> None:
        assert page_index(20) == 19

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            page_index(0)


# ---------------------------------------------------------------------------
# _is_numeric_token
# ---------------------------------------------------------------------------


class TestIsNumericToken:
    @pytest.mark.parametrize(
        "token",
        ["1.72", "28.4", "(3.57)", "(0.35)", "17.0", "23.07", "104.30", "20,352"],
    )
    def test_numeric_tokens(self, token: str) -> None:
        assert _is_numeric_token(token) is True

    @pytest.mark.parametrize(
        "token",
        ["BLANK", "FY2023", "$billion", "% change", "Corporate", "-", "Add:", "Note:"],
    )
    def test_non_numeric_tokens(self, token: str) -> None:
        assert _is_numeric_token(token) is False


# ---------------------------------------------------------------------------
# _is_garbled_coordinate_line
# ---------------------------------------------------------------------------


class TestIsGarbledCoordinateLine:
    def test_garbled_line_detected(self) -> None:
        assert _is_garbled_coordinate_line("Add: 22,376, 23,480,570, 22,915, 0") is True

    def test_normal_line_not_garbled(self) -> None:
        assert _is_garbled_coordinate_line("OVERALL FISCAL POSITION 1.72 (0.35) (3.57)") is False

    def test_thousands_separator_not_garbled(self) -> None:
        # "20,352" has no ", \d" pattern (no space after comma)
        assert _is_garbled_coordinate_line("Total 20,352") is False


# ---------------------------------------------------------------------------
# _split_label_values
# ---------------------------------------------------------------------------


class TestSplitLabelValues:
    def test_standard_row(self) -> None:
        result = _split_label_values("Corporate Income Tax 23.07 24.26 28.38 23.0 17.0")
        assert result is not None
        label, values = result
        assert label == "Corporate Income Tax"
        assert values == ["23.07", "24.26", "28.38", "23.0", "17.0"]

    def test_overall_fiscal_position(self) -> None:
        result = _split_label_values("OVERALL FISCAL POSITION 1.72 (0.35) (3.57)")
        assert result is not None
        label, values = result
        assert label == "OVERALL FISCAL POSITION"
        assert values == ["1.72", "(0.35)", "(3.57)"]

    def test_blank_header_line_returns_none(self) -> None:
        assert _split_label_values("BLANK $billion $billion $billion % change % change") is None

    def test_blank_label_line_returns_none(self) -> None:
        assert _split_label_values("BLANK Actual Estimated Revised Compared to") is None

    def test_garbled_line_returns_none(self) -> None:
        assert _split_label_values("Add: 22,376, 23,480,570, 22,915, 0") is None

    def test_section_marker_add_returns_none(self) -> None:
        assert _split_label_values("Add: 22.38 23.48 22.92") is None

    def test_empty_line_returns_none(self) -> None:
        assert _split_label_values("") is None

    def test_row_with_parenthesised_negatives(self) -> None:
        result = _split_label_values("OVERALL BUDGET SURPLUS / DEFICIT (0.41) (3.55) (6.84)")
        assert result is not None
        label, values = result
        assert label == "OVERALL BUDGET SURPLUS / DEFICIT"
        assert values == ["(0.41)", "(3.55)", "(6.84)"]


# ---------------------------------------------------------------------------
# _clean_sparse_row
# ---------------------------------------------------------------------------


class TestCleanSparseRow:
    def test_strips_nones(self) -> None:
        assert _clean_sparse_row(["a", None, None, "b", None]) == ["a", "b"]

    def test_strips_empty_strings(self) -> None:
        assert _clean_sparse_row(["", "Total", "", "20,352", ""]) == ["Total", "20,352"]

    def test_all_none_returns_empty(self) -> None:
        assert _clean_sparse_row([None, None, None]) == []


# ---------------------------------------------------------------------------
# ParsedTable.get_cell
# ---------------------------------------------------------------------------


class TestParsedTableGetCell:
    def test_exact_match(self, sample_parsed_table_page8: ParsedTable) -> None:
        val = sample_parsed_table_page8.get_cell(
            "OVERALL FISCAL POSITION", "Actual FY2022"
        )
        assert val == "1.72"

    def test_substring_row_match(self, sample_parsed_table_page8: ParsedTable) -> None:
        # "FISCAL POSITION" is a subset of the stored label
        val = sample_parsed_table_page8.get_cell("FISCAL POSITION", "Actual FY2022")
        assert val == "1.72"

    def test_negative_cell(self, sample_parsed_table_page8: ParsedTable) -> None:
        val = sample_parsed_table_page8.get_cell(
            "OVERALL FISCAL POSITION", "Revised FY2023"
        )
        assert val == "(3.57)"

    def test_missing_row_raises(self, sample_parsed_table_page8: ParsedTable) -> None:
        with pytest.raises(ExtractionValidationError):
            sample_parsed_table_page8.get_cell("NONEXISTENT ROW", "Actual FY2022")

    def test_missing_column_raises(self, sample_parsed_table_page8: ParsedTable) -> None:
        with pytest.raises(ExtractionValidationError):
            sample_parsed_table_page8.get_cell(
                "OVERALL FISCAL POSITION", "Nonexistent Column"
            )

    def test_none_cell_raises(self, sample_parsed_table_page8: ParsedTable) -> None:
        with pytest.raises(ExtractionValidationError):
            sample_parsed_table_page8.get_cell(
                "OVERALL FISCAL POSITION", "Compared to Actual FY2022"
            )

    def test_page20_total(self, sample_parsed_table_page20: ParsedTable) -> None:
        val = sample_parsed_table_page20.get_cell("Total", "Estimated FY2024")
        assert val == "20,352"


# ---------------------------------------------------------------------------
# ParsedTable.to_llm_text
# ---------------------------------------------------------------------------


class TestToLlmText:
    def test_contains_title(self, sample_parsed_table_page8: ParsedTable) -> None:
        text = sample_parsed_table_page8.to_llm_text()
        assert "Fiscal Position" in text

    def test_contains_headers(self, sample_parsed_table_page8: ParsedTable) -> None:
        text = sample_parsed_table_page8.to_llm_text()
        assert "Actual FY2022" in text
        assert "Revised FY2023" in text

    def test_contains_row_data(self, sample_parsed_table_page8: ParsedTable) -> None:
        text = sample_parsed_table_page8.to_llm_text()
        assert "OVERALL FISCAL POSITION" in text
        assert "1.72" in text


# ---------------------------------------------------------------------------
# Integration tests — real PDF required
# ---------------------------------------------------------------------------


class TestPdfParserIntegration:
    """These tests open the actual source PDF.  Skipped if the file is absent."""

    def test_page5_text_contains_corporate_tax(self, source_pdf_path) -> None:
        text = get_page_text(source_pdf_path, 5)
        assert "Corporate Income Tax" in text

    def test_page8_table_contains_fiscal_position(self, source_pdf_path) -> None:
        table = extract_table_page8(source_pdf_path)
        # The row must exist in the parsed table
        assert any("OVERALL FISCAL" in label for label in table.rows)

    def test_page8_table_fiscal_position_value(self, source_pdf_path) -> None:
        table = extract_table_page8(source_pdf_path)
        val = table.get_cell("OVERALL FISCAL POSITION", "Actual FY2022")
        assert val == "1.72"

    def test_page20_table_total(self, source_pdf_path) -> None:
        table = extract_table_page20(source_pdf_path)
        val = table.get_cell("Total", "Estimated FY2024")
        assert val == "20,352"

    def test_pages_text_returns_dict(self, source_pdf_path) -> None:
        texts = get_pages_text(source_pdf_path, [5, 6])
        assert 5 in texts and 6 in texts
        assert len(texts[5]) > 100
