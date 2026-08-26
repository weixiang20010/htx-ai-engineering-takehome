"""Unit tests for Part 2 Pydantic models — no LLM, no MCP, no I/O."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from src.part2.models import (
    REFERENCE_DATE,
    ClassificationResult,
    ExtractedDate,
    InternalDateClassification,
    Part2Evidence,
    Part2ResultItem,
    PerOperationModelUsage,
    TemporalInterpretation,
    TemporalStatus,
)
from src.llm import ModelUsage


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


class TestTemporalInterpretation:
    def test_valid_point_event(self) -> None:
        obj = TemporalInterpretation(
            temporal_type="point_event",
            relation="on",
            has_explicit_end=False,
            brief_reason="The sentence describes a single event on a specific date.",
        )
        assert obj.temporal_type == "point_event"
        assert not obj.has_explicit_end

    def test_valid_cutoff_after(self) -> None:
        obj = TemporalInterpretation(
            temporal_type="cutoff_or_threshold",
            relation="after",
            has_explicit_end=False,
            brief_reason="The condition applies after the threshold.",
        )
        assert obj.relation == "after"

    def test_rejects_unknown_temporal_type(self) -> None:
        with pytest.raises(Exception):
            TemporalInterpretation(
                temporal_type="unknown",  # type: ignore[arg-type]
                relation="on",
                has_explicit_end=False,
                brief_reason=".",
            )

    def test_rejects_unknown_relation(self) -> None:
        with pytest.raises(Exception):
            TemporalInterpretation(
                temporal_type="point_event",
                relation="during",  # type: ignore[arg-type]
                has_explicit_end=False,
                brief_reason=".",
            )



    def test_valid_construction(self) -> None:
        obj = InternalDateClassification(
            status=TemporalStatus.UPCOMING,
            reason="The date is after the reference date.",
        )
        assert obj.status == TemporalStatus.UPCOMING
        assert obj.reason

    def test_only_status_and_reason_fields(self) -> None:
        """LLM output must NOT echo back inputs (trust boundary)."""
        obj = InternalDateClassification(
            status=TemporalStatus.EXPIRED,
            reason="Before the reference date.",
        )
        dumped = obj.model_dump()
        assert set(dumped.keys()) == {"status", "reason"}


class TestPart2Evidence:
    def _make_usage(self) -> ModelUsage:
        return ModelUsage(requested_model="gemini-3.5-flash-lite", actual_model="gemini-3.5-flash-lite")

    def _make_interpretation(self) -> TemporalInterpretation:
        return TemporalInterpretation(
            temporal_type="point_event",
            relation="on",
            has_explicit_end=False,
            brief_reason="Single event.",
        )

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
            temporal_interpretation=self._make_interpretation(),
            consistency_check_passed=True,
            classification_retried=False,
            llm_status="Upcoming",
            llm_rationale="The date is after the reference date.",
            model_usage=PerOperationModelUsage(
                date_extraction=self._make_usage(),
                tool_selection=self._make_usage(),
                interpretation=self._make_usage(),
                classification=self._make_usage(),
            ),
        )
        assert ev.source_page == 1
        assert ev.source_validation_passed is True
        assert ev.mcp_tool_requested == "normalize_date"
