"""
Entry point for the Part 1 extraction pipeline.

Usage
-----
    python scripts/run_part1.py

Reads configuration from a .env file in the project root (or from the
environment).  Writes results to:

    outputs/part1_result.json    — the five final extracted values
    outputs/part1_evidence.json  — per-field audit trail
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the src package importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import build_llm_pair
from src.part1.extractor import run_extraction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_part1")


def main() -> None:
    # --- Load environment ---
    load_dotenv()

    pdf_path = Path(os.environ.get("SOURCE_PDF", "data/fy2024_analysis_of_revenue_and_expenditure.pdf"))
    if not pdf_path.exists():
        logger.error("Source PDF not found at %s", pdf_path.resolve())
        sys.exit(1)

    # --- Build LLM ---
    try:
        llm, fallback_llm = build_llm_pair()
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Primary model : %s", os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-3.6-flash"))
    logger.info("Fallback model: %s", os.environ.get("GEMINI_FALLBACK_MODEL", "none"))
    logger.info("Source PDF    : %s", pdf_path.resolve())

    # --- Run pipeline ---
    result, evidence = run_extraction(pdf_path, llm, fallback_llm)

    # --- Save outputs ---
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    result_path = outputs_dir / "part1_result.json"
    evidence_path = outputs_dir / "part1_evidence.json"

    result_path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            [ev.model_dump() for ev in evidence],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info("Results written to %s", result_path)
    logger.info("Evidence written to %s", evidence_path)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("PART 1 EXTRACTION RESULTS")
    print("=" * 60)
    for field, value in result.model_dump().items():
        print(f"  {field}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
