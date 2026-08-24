"""
Optional demonstration: naive prompt vs improved (grounded, structured) prompt.

Shows why a naive "answer the question" prompt is insufficient for reliable
information extraction and how the improved prompt addresses each gap.

Usage
-----
    python scripts/compare_prompts.py

Reads GEMINI_API_KEY and GEMINI_MODEL from .env (or the environment).
Reads SOURCE_PDF from .env (default: data/fy2024_analysis_of_revenue_and_expenditure.pdf).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.part1.extractor import build_llm
from src.part1.models import CorporateTaxEvidence
from src.part1.pdf_parser import get_pages_text
from src.part1.prompts import CORPORATE_TAX_IMPROVED, CORPORATE_TAX_NAIVE
from src.part1.validators import ExtractionValidationError, validate_evidence_in_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _compare(page5_text: str, llm) -> dict[str, object]:
    """Run naive and improved prompts and return a side-by-side comparison."""

    # --- Naive: free-form prose ---
    naive_chain = CORPORATE_TAX_NAIVE | llm
    naive_response = naive_chain.invoke({"context": page5_text})
    naive_text: str = getattr(naive_response, "content", str(naive_response))

    # --- Improved: structured + evidence-grounded ---
    improved_chain = CORPORATE_TAX_IMPROVED | llm.with_structured_output(
        CorporateTaxEvidence
    )
    improved_result: CorporateTaxEvidence = improved_chain.invoke(
        {"context": page5_text}
    )

    improved_validated = False
    improved_note = ""
    try:
        if improved_result.evidence_text:
            validate_evidence_in_source(
                improved_result.evidence_text,
                page5_text,
                field_name="demo_evidence",
            )
            improved_validated = True
        else:
            improved_note = "LLM returned no evidence_text"
    except ExtractionValidationError as exc:
        improved_note = str(exc)

    return {
        "naive": {
            "description": "Plain question — no grounding, no schema, no evidence requirement",
            "response_type": "free-form text",
            "response": naive_text,
            "structured": False,
            "evidence_verified": False,
        },
        "improved": {
            "description": "Grounded prompt — context-only, Pydantic schema, verbatim evidence required",
            "response_type": "CorporateTaxEvidence (Pydantic)",
            "response": improved_result.model_dump(),
            "structured": True,
            "evidence_verified": improved_validated,
            "evidence_note": improved_note,
        },
        "comparison": {
            "grounding": (
                "Improved restricts the model to the supplied context; "
                "naive allows the model to draw on training-data knowledge."
            ),
            "structured_output": (
                "Improved returns a Pydantic schema (CorporateTaxEvidence); "
                "naive returns unstructured prose."
            ),
            "evidence": (
                "Improved requires a verbatim source sentence; "
                "naive provides none."
            ),
            "missing_value_handling": (
                "Improved instructs the model to return null when evidence is absent; "
                "naive may fabricate a value."
            ),
            "deterministic_validation": (
                "Improved output is programmatically verifiable via validate_evidence_in_source; "
                "naive output cannot be validated."
            ),
        },
    }


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set.  Add it to .env or the environment.")
        sys.exit(1)

    pdf_path = Path(
        os.environ.get("SOURCE_PDF", "data/fy2024_analysis_of_revenue_and_expenditure.pdf")
    )
    if not pdf_path.exists():
        logger.error("PDF not found at %s", pdf_path)
        sys.exit(1)

    logger.info("Building LLM client...")
    llm = build_llm(api_key=api_key)

    logger.info("Extracting page 5 text...")
    texts = get_pages_text(pdf_path, printed_pages=[5])
    page5_text = texts[5]

    logger.info("Running naive vs improved comparison...")
    result = _compare(page5_text, llm)

    print("\n" + "=" * 70)
    print("NAIVE vs IMPROVED PROMPT COMPARISON")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
