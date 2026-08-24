"""
End-to-end smoke test for the Part 1 extraction pipeline.

Marked with ``@pytest.mark.integration`` so it is skipped during local unit
test runs that do not have a Gemini API key.  Run with:

    pytest tests/test_integration.py -v
"""
from __future__ import annotations

import os

import pytest

SKIP_REASON = "GEMINI_API_KEY not set — skipping integration test"


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason=SKIP_REASON)
class TestPart1EndToEnd:
    """Full pipeline smoke test: PDF → Gemini → validation → Part1Result."""

    def test_pipeline_produces_valid_result(self, source_pdf_path, tmp_path) -> None:
        from src.part1.extractor import build_llm, run_extraction
        from src.part1.models import Part1Result

        llm = build_llm()
        result, evidence = run_extraction(source_pdf_path, llm)

        # Schema validation — Pydantic ensures types are correct
        assert isinstance(result, Part1Result)

        # Sanity-check ranges (not exact values — avoid hardcoding)
        assert 20.0 < result.corporate_income_tax_2024 < 50.0, (
            f"CIT out of plausible range: {result.corporate_income_tax_2024}"
        )
        assert 0.0 < result.corporate_income_tax_yoy_pct_2024 < 100.0, (
            f"CIT YoY out of plausible range: {result.corporate_income_tax_yoy_pct_2024}"
        )
        assert result.total_top_ups_2024 > 0, "Total top-ups should be positive"
        assert len(result.operating_revenue_taxes) >= 5, (
            f"Expected at least 5 taxes, got {len(result.operating_revenue_taxes)}"
        )
        assert result.latest_actual_fiscal_position_billions != 0.0

        # Evidence audit trail is produced for every field
        assert len(evidence) == 5
        assert all(ev.validation_passed for ev in evidence), (
            f"Validation failures: {[ev for ev in evidence if not ev.validation_passed]}"
        )
