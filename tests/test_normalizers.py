"""Unit tests for src/part1/normalizers.py."""
from __future__ import annotations

import pytest

from src.part1.normalizers import normalize_to_float


class TestNormalizeToFloat:
    """Deterministic normalization — no external dependencies."""

    # --- Thousands separators ---
    def test_thousands_separator(self) -> None:
        assert normalize_to_float("20,352") == pytest.approx(20352.0)

    def test_thousands_separator_large(self) -> None:
        assert normalize_to_float("1,234,567") == pytest.approx(1234567.0)

    # --- Percentage ---
    def test_percentage(self) -> None:
        assert normalize_to_float("17.0%") == pytest.approx(17.0)

    def test_percentage_no_decimal(self) -> None:
        assert normalize_to_float("5%") == pytest.approx(5.0)

    # --- Currency prefix ---
    def test_dollar_prefix(self) -> None:
        assert normalize_to_float("$28.4") == pytest.approx(28.4)

    def test_dollar_with_unit(self) -> None:
        assert normalize_to_float("$28.4 billion") == pytest.approx(28.4)

    def test_dollar_million(self) -> None:
        assert normalize_to_float("$100 million") == pytest.approx(100.0)

    # --- Parenthesised negatives ---
    def test_parenthesised_negative(self) -> None:
        assert normalize_to_float("(3.57)") == pytest.approx(-3.57)

    def test_parenthesised_negative_integer(self) -> None:
        assert normalize_to_float("(0.35)") == pytest.approx(-0.35)

    # --- Minus sign ---
    def test_minus_sign(self) -> None:
        assert normalize_to_float("-3.57") == pytest.approx(-3.57)

    # --- Plain float ---
    def test_plain_float(self) -> None:
        assert normalize_to_float("1.72") == pytest.approx(1.72)

    def test_plain_integer(self) -> None:
        assert normalize_to_float("28") == pytest.approx(28.0)

    # --- Combined: thousands + billion unit ---
    def test_billions_with_thousands(self) -> None:
        assert normalize_to_float("1,234.5 billion") == pytest.approx(1234.5)

    # --- Error cases ---
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_to_float("")

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_to_float("not a number")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_to_float("   ")
