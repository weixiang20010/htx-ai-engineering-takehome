"""Unit tests for src/part2/date_normalizer.py — no MCP, no Gemini."""
from __future__ import annotations

import pytest

from src.part2.date_normalizer import DateNormalizationError, normalize_date_value


class TestNormalizeDateValue:
    def test_dd_full_month_yyyy(self) -> None:
        assert normalize_date_value("16 February 2024") == "2024-02-16"

    def test_dd_full_month_yyyy_historic(self) -> None:
        assert normalize_date_value("15 February 2008") == "2008-02-15"

    def test_abbreviated_month(self) -> None:
        assert normalize_date_value("16 Feb 2024") == "2024-02-16"

    def test_abbreviated_month_historic(self) -> None:
        assert normalize_date_value("15 Feb 2008") == "2008-02-15"

    def test_us_long_format(self) -> None:
        assert normalize_date_value("February 16, 2024") == "2024-02-16"

    def test_us_abbreviated_format(self) -> None:
        assert normalize_date_value("Feb 16, 2024") == "2024-02-16"

    def test_already_iso_passthrough(self) -> None:
        assert normalize_date_value("2024-02-16") == "2024-02-16"

    def test_slash_format(self) -> None:
        assert normalize_date_value("16/02/2024") == "2024-02-16"

    def test_extra_whitespace_handled(self) -> None:
        assert normalize_date_value("  16 February 2024  ") == "2024-02-16"

    def test_empty_string_raises(self) -> None:
        with pytest.raises(DateNormalizationError):
            normalize_date_value("")

    def test_blank_string_raises(self) -> None:
        with pytest.raises(DateNormalizationError):
            normalize_date_value("   ")

    def test_invalid_day_raises(self) -> None:
        with pytest.raises(DateNormalizationError):
            normalize_date_value("32 February 2024")

    def test_invalid_month_name_raises(self) -> None:
        with pytest.raises(DateNormalizationError):
            normalize_date_value("16 Febrrrr 2024")

    def test_unsupported_format_raises(self) -> None:
        with pytest.raises(DateNormalizationError):
            normalize_date_value("2024 16 02")

    def test_plain_year_raises(self) -> None:
        with pytest.raises(DateNormalizationError):
            normalize_date_value("2024")
