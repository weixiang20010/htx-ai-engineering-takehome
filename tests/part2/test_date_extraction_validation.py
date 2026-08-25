"""
Tests for date extraction evidence validation — no Gemini API required.

Reuses Part 1 validators via src.part2.date_extractor._validate_date_extraction.
"""
from __future__ import annotations

import pytest

from src.part1.validators import ExtractionValidationError
from src.part2.date_extractor import _validate_date_extraction
from src.part2.models import ExtractedDate

# Actual pdfplumber output for page 1 (verified during PDF inspection)
_PAGE1_TEXT = (
    "ANALYSIS OF\nREVENUE AND\nEXPENDITURE\n"
    "Financial Year 2024\n"
    "Distributed on Budget Day: 16 February 2024"
)

# Clean left-column text from page 36 (cropped with pdfplumber; no column interleaving)
_PAGE36_TEXT = (
    "Estate Duty does not apply to a\nperson who dies after 15 February 2008."
)


class TestValidateDateExtraction:
    def test_valid_page1_extraction_passes(self) -> None:
        extracted = ExtractedDate(
            source_page=1,
            original_text="Distributed on Budget Day: 16 February 2024",
            date_text="16 February 2024",
        )
        _validate_date_extraction(extracted, _PAGE1_TEXT)

    def test_valid_page36_extraction_passes(self) -> None:
        # With the left-column crop, the full sentence is available verbatim
        extracted = ExtractedDate(
            source_page=36,
            original_text="Estate Duty does not apply to a person who dies after 15 February 2008.",
            date_text="15 February 2008",
        )
        _validate_date_extraction(extracted, _PAGE36_TEXT)

    def test_null_original_text_raises(self) -> None:
        extracted = ExtractedDate(
            source_page=1, original_text=None, date_text="16 February 2024"
        )
        with pytest.raises(ExtractionValidationError):
            _validate_date_extraction(extracted, _PAGE1_TEXT)

    def test_null_date_text_raises(self) -> None:
        extracted = ExtractedDate(
            source_page=1,
            original_text="Distributed on Budget Day: 16 February 2024",
            date_text=None,
        )
        with pytest.raises(ExtractionValidationError):
            _validate_date_extraction(extracted, _PAGE1_TEXT)

    def test_invented_date_not_in_source_raises(self) -> None:
        # "17 February 2024" does not appear in the page text
        extracted = ExtractedDate(
            source_page=1,
            original_text="Distributed on Budget Day: 17 February 2024",
            date_text="17 February 2024",
        )
        with pytest.raises(ExtractionValidationError):
            _validate_date_extraction(extracted, _PAGE1_TEXT)

    def test_date_text_not_in_original_text_raises(self) -> None:
        # date_text is in source but NOT in original_text → fabrication detected
        extracted = ExtractedDate(
            source_page=1,
            original_text="Distributed on Budget Day: some other text",
            date_text="16 February 2024",
        )
        with pytest.raises(ExtractionValidationError):
            _validate_date_extraction(extracted, _PAGE1_TEXT)

    def test_invented_sentence_raises(self) -> None:
        # date_text is correct and in source, but original_text is fabricated — must fail
        extracted = ExtractedDate(
            source_page=1,
            original_text="The budget was distributed on 16 February 2024 under fiscal law.",
            date_text="16 February 2024",
        )
        with pytest.raises(ExtractionValidationError):
            _validate_date_extraction(extracted, _PAGE1_TEXT)

    def test_empty_date_text_raises(self) -> None:
        extracted = ExtractedDate(
            source_page=1,
            original_text="Distributed on Budget Day: 16 February 2024",
            date_text="",
        )
        with pytest.raises(ExtractionValidationError):
            _validate_date_extraction(extracted, _PAGE1_TEXT)
