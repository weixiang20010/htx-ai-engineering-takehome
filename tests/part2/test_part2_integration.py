"""
End-to-end integration test for the Part 2 pipeline.

Requires: GEMINI_API_KEY env var and the source PDF at the expected path.
These tests are skipped automatically in CI if the key is absent.

Run locally with:
    pytest tests/part2/test_part2_integration.py -m integration -v
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from src.llm import build_llm_pair
from src.part2.models import Part2Evidence, Part2ResultItem, TemporalStatus
from src.part2.workflow import run_part2

_HAS_API_KEY = bool(os.environ.get("GEMINI_API_KEY"))
_PDF_PATH = Path("data/fy2024_analysis_of_revenue_and_expenditure.pdf")
_PDF_EXISTS = _PDF_PATH.exists()

_SKIP_REASON = (
    "Integration test requires GEMINI_API_KEY environment variable "
    "and the source PDF at data/fy2024_analysis_of_revenue_and_expenditure.pdf"
)


@pytest.fixture(scope="session")
def part2_pipeline_output():
    """Run the Part 2 pipeline once; all tests in this session share the result."""
    if not (_HAS_API_KEY and _PDF_EXISTS):
        pytest.skip(_SKIP_REASON)
    llm, fallback_llm = build_llm_pair()
    return asyncio.get_event_loop().run_until_complete(
        run_part2(_PDF_PATH, llm, fallback_llm)
    )


@pytest.mark.integration
@pytest.mark.skipif(not (_HAS_API_KEY and _PDF_EXISTS), reason=_SKIP_REASON)
class TestPart2Integration:
    def test_normalized_dates_are_iso(self, part2_pipeline_output) -> None:
        """Both normalized dates must be ISO YYYY-MM-DD strings."""
        normalized_dates, _, _ = part2_pipeline_output
        assert len(normalized_dates) == 2
        for iso_date in normalized_dates:
            parts = iso_date.split("-")
            assert len(parts) == 3, f"Not ISO format: {iso_date}"
            assert parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit()

    def test_known_dates_are_correct(self, part2_pipeline_output) -> None:
        """Verify the exact ISO dates for the two known source dates."""
        normalized_dates, _, _ = part2_pipeline_output
        assert "2024-02-16" in normalized_dates, "Distribution date missing"
        assert "2008-02-15" in normalized_dates, "Estate Duty date missing"

    def test_results_have_valid_schema(self, part2_pipeline_output) -> None:
        """Each result has all three required fields with valid enum status."""
        _, results, _ = part2_pipeline_output
        assert len(results) == 2
        valid_statuses = set(TemporalStatus)
        for item in results:
            assert isinstance(item, Part2ResultItem)
            assert item.original_text
            assert item.normalized_date
            assert item.status in valid_statuses

    def test_evidence_has_mcp_tool_trace(self, part2_pipeline_output) -> None:
        """Evidence records must record the MCP tool call as required by spec."""
        _, _, evidence = part2_pipeline_output
        assert len(evidence) == 2
        for ev in evidence:
            assert isinstance(ev, Part2Evidence)
            assert ev.mcp_tool_requested == "normalize_date"
            assert ev.mcp_tool_arguments.get("date_text"), "Tool args must have date_text"
            assert ev.source_validation_passed is True

    def test_reference_date_is_recorded_in_evidence(self, part2_pipeline_output) -> None:
        """Evidence must record the fixed reference date, not today's date."""
        _, _, evidence = part2_pipeline_output
        for ev in evidence:
            assert ev.reference_date == "2024-01-01"
