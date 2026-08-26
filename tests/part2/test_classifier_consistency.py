"""
Deterministic tests for the Part 2 consistency checker — no LLM, no I/O.

_consistency_conflict() is pure Python: it takes an interpretation and a
classification status, compares them against the reference date, and returns
a conflict description string (or None when consistent).
"""
from __future__ import annotations

import pytest

from src.part2.classifier import _consistency_conflict
from src.part2.models import TemporalInterpretation, TemporalStatus


def _interp(
    temporal_type: str,
    relation: str,
    has_explicit_end: bool = False,
    brief_reason: str = "test",
) -> TemporalInterpretation:
    return TemporalInterpretation(
        temporal_type=temporal_type,  # type: ignore[arg-type]
        relation=relation,  # type: ignore[arg-type]
        has_explicit_end=has_explicit_end,
        brief_reason=brief_reason,
    )


class TestOpenEndedCutoffAfter:
    """cutoff_or_threshold + after + no explicit end + past date must be Ongoing."""

    PAST_DATE = "2008-02-15"
    INTERP = _interp("cutoff_or_threshold", "after", has_explicit_end=False)

    def test_ongoing_is_consistent(self) -> None:
        assert _consistency_conflict(self.INTERP, TemporalStatus.ONGOING, self.PAST_DATE) is None

    def test_expired_is_inconsistent(self) -> None:
        conflict = _consistency_conflict(self.INTERP, TemporalStatus.EXPIRED, self.PAST_DATE)
        assert conflict is not None
        assert "open-ended" in conflict.lower() or "no explicit end" in conflict.lower() or "cutoff" in conflict.lower()

    def test_upcoming_is_accepted(self) -> None:
        # Unusual but we don't flag it — the LLM may have a reason
        assert _consistency_conflict(self.INTERP, TemporalStatus.UPCOMING, self.PAST_DATE) is None

    def test_future_cutoff_not_flagged(self) -> None:
        # Cutoff date is after reference → constraint does not apply
        assert _consistency_conflict(self.INTERP, TemporalStatus.EXPIRED, "2025-01-01") is None


class TestPointEventInPast:
    """point_event before reference date must be Expired."""

    PAST_DATE = "2024-02-16"  # after reference 2024-01-01 — use a truly past one
    TRULY_PAST = "2020-06-01"
    INTERP = _interp("point_event", "on")

    def test_expired_is_consistent(self) -> None:
        assert _consistency_conflict(self.INTERP, TemporalStatus.EXPIRED, self.TRULY_PAST) is None

    def test_upcoming_contradicts_past_event(self) -> None:
        conflict = _consistency_conflict(self.INTERP, TemporalStatus.UPCOMING, self.TRULY_PAST)
        assert conflict is not None

    def test_ongoing_contradicts_past_event(self) -> None:
        conflict = _consistency_conflict(self.INTERP, TemporalStatus.ONGOING, self.TRULY_PAST)
        assert conflict is not None


class TestPointEventInFuture:
    """point_event after reference date must be Upcoming."""

    FUTURE_DATE = "2024-02-16"
    INTERP = _interp("point_event", "on")

    def test_upcoming_is_consistent(self) -> None:
        assert _consistency_conflict(self.INTERP, TemporalStatus.UPCOMING, self.FUTURE_DATE) is None

    def test_expired_contradicts_future_event(self) -> None:
        conflict = _consistency_conflict(self.INTERP, TemporalStatus.EXPIRED, self.FUTURE_DATE)
        assert conflict is not None

    def test_ongoing_contradicts_future_event(self) -> None:
        conflict = _consistency_conflict(self.INTERP, TemporalStatus.ONGOING, self.FUTURE_DATE)
        assert conflict is not None


class TestUnclearCasesAreAccepted:
    """Anything not clearly contradictory should return None (no false positives)."""

    def test_period_type_any_status_accepted(self) -> None:
        interp = _interp("period", "from")
        for status in TemporalStatus:
            assert _consistency_conflict(interp, status, "2022-01-01") is None

    def test_cutoff_with_explicit_end_accepted(self) -> None:
        interp = _interp("cutoff_or_threshold", "after", has_explicit_end=True)
        assert _consistency_conflict(interp, TemporalStatus.EXPIRED, "2008-02-15") is None

    def test_cutoff_until_past_expired_accepted(self) -> None:
        interp = _interp("cutoff_or_threshold", "until")
        assert _consistency_conflict(interp, TemporalStatus.EXPIRED, "2019-12-31") is None
