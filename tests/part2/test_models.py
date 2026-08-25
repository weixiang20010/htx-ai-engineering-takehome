"""Unit tests for Part 2 Pydantic models — no LLM, no MCP, no I/O."""
from __future__ import annotations

from datetime import date

import pytest

from src.part2.models import (
    REFERENCE_DATE,
    ExtractedDate,
    InternalDateClassification,
    Part2Evidence,
    Part2ResultItem,
    TemporalStatus,
)


class TestReferenceDate:
    def test_is_exactly_2024_01_01(self) -> None:
        assert REFERENCE_DATE == date(2024, 1, 1)

    def test_is_date_instance(self) -> None:
        assert isinstance(REFERENCE_DATE, date)


class TestTemporalStatus:
    def test_expired_value(self) -> None:
        assert TemporalStatus.EXPIRED == "Expired"

    def test_upcoming_value(self) -> None:
        assert TemporalStatus.UPCOMING == "Upcoming"

    def test_ongoing_value(self) -> None:
        assert TemporalStatus.ONGOING == "Ongoing"

    def test_all_values_are_strings(self) -> None:
        for status in TemporalStatus:
            assert isinstance(status, str)


class TestExtractedDate:
    def test_valid_construction(self) -> None:
        obj = ExtractedDate(
            source_page=1,
            original_text="Distributed on Budget Day: 16 February 2024",
            date_text="16 February 2024",
        )
        assert obj.source_page == 1
        assert obj.date_text == "16 February 2024"

    def test_optional_fields_default_to_none(self) -> None:
        obj = ExtractedDate(source_page=1)
        assert obj.original_text is None
        assert obj.date_text is None


class TestPart2ResultItem:
    def test_model_dump_has_exactly_three_keys(self) -> None:
        item = Part2ResultItem(
            original_text="Distributed on Budget Day: 16 February 2024",
            normalized_date="2024-02-16",
            status=TemporalStatus.UPCOMING,
        )
        dumped = item.model_dump()
        assert set(dumped.keys()) == {"original_text", "normalized_date", "status"}

    def test_status_accepts_enum_members(self) -> None:
        for status in TemporalStatus:
            item = Part2ResultItem(
                original_text="some text",
                normalized_date="2024-01-01",
                status=status,
            )
            assert item.status == status


class TestInternalDateClassification:
    def test_valid_construction(self) -> None:
        obj = InternalDateClassification(
            original_text="Distributed on Budget Day: 16 February 2024",
            normalized_date="2024-02-16",
            status=TemporalStatus.UPCOMING,
            reason="The date is after the reference date.",
        )
        assert obj.normalized_date == "2024-02-16"
        assert obj.status == TemporalStatus.UPCOMING


class TestPart2Evidence:
    def test_valid_construction(self) -> None:
        ev = Part2Evidence(
            source_page=1,
            source_statement="Distributed on Budget Day: 16 February 2024",
            extracted_date_text="16 February 2024",
            source_validation_passed=True,
            mcp_tool_requested="normalize_date",
            mcp_tool_arguments={"date_text": "16 February 2024"},
            mcp_tool_result="2024-02-16",
            reference_date="2024-01-01",
            llm_status="Upcoming",
            llm_rationale="The date is after the reference date.",
        )
        assert ev.source_page == 1
        assert ev.source_validation_passed is True
        assert ev.mcp_tool_requested == "normalize_date"
