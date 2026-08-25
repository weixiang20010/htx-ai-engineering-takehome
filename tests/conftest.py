"""Shared pytest fixtures for Part 1 and Part 2 tests."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env before any skipif/guard reads it

from src.part1.pdf_parser import ParsedTable

# ---------------------------------------------------------------------------
# Integration test guard
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip integration tests automatically when no API key is configured."""
    if "integration" in item.keywords and not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set — skipping integration test")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SOURCE_PDF = Path(__file__).resolve().parent.parent / "data" / "fy2024_analysis_of_revenue_and_expenditure.pdf"


@pytest.fixture
def source_pdf_path() -> Path:
    """Return the path to the source PDF, skipping if absent."""
    if not SOURCE_PDF.exists():
        pytest.skip("Source PDF not found — skipping PDF-dependent test")
    return SOURCE_PDF


@pytest.fixture
def sample_page5_text() -> str:
    return (
        "1.2 Operating Revenue\n"
        "Revised FY2023 Operating Revenue is $104.3 billion, which is $7.6 billion "
        "(7.9%) higher than the Estimated FY2023 figure.\n"
        "Corporate Income Tax collections are revised to $28.4 billion, which is "
        "$4.1 billion (17.0%) higher than the Estimated FY2023 figure due to stronger-"
        "than-expected economic growth in 2022. Collections from Other Taxes, which "
        "include the Foreign Worker Levy, Water Conservation Tax, Land Betterment "
        "Charge, and Annual Tonnage Tax, are revised to $8.8 billion.\n"
        "Vehicle Quota Premiums collections are revised to $4.7 billion.\n"
        "Personal Income"
    )


@pytest.fixture
def sample_page6_text() -> str:
    return (
        "Tax collections are revised to $17.5 billion.\n"
        "Assets Taxes collections are revised to $5.9 billion.\n"
        "Betting Taxes collections are revised to $3.2 billion due to stronger-"
        "than-expected collections for Casino Taxes.\n"
        "Goods and Services Tax collections are revised to $16.4 billion."
    )


@pytest.fixture
def sample_parsed_table_page8() -> ParsedTable:
    """A minimal ParsedTable matching Table 1.1 structure."""
    return ParsedTable(
        title="Fiscal Position in FY2022 and FY2023",
        source_page=8,
        headers=[
            "Actual FY2022",
            "Estimated FY2023",
            "Revised FY2023",
            "Compared to Actual FY2022",
            "Compared to Estimated FY2023",
        ],
        rows={
            "OPERATING REVENUE": ["91.01", "96.70", "104.30", "14.6", "7.9"],
            "Corporate Income Tax": ["23.07", "24.26", "28.38", "23.0", "17.0"],
            "OVERALL FISCAL POSITION": ["1.72", "(0.35)", "(3.57)", None, None],
        },
    )


@pytest.fixture
def sample_parsed_table_page20() -> ParsedTable:
    """A minimal ParsedTable matching Table 2.4 structure."""
    return ParsedTable(
        title="Top-ups to Endowment and Trust Funds in FY2024",
        source_page=20,
        headers=["Estimated FY2024"],
        rows={
            "Goods and Services Tax Voucher Fund": ["6,000"],
            "Future Energy Fund": ["5,000"],
            "Total": ["20,352"],
        },
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """A mock ChatGoogleGenerativeAI that raises if accidentally invoked."""
    m = MagicMock(name="ChatGoogleGenerativeAI")
    m.with_structured_output.return_value = m
    return m
