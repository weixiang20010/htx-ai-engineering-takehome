"""
Entry point for the Part 2 pipeline.

Usage
-----
    python scripts/run_part2.py

Reads configuration from a .env file in the project root (or from the environment).
Writes results to:

    outputs/part2_normalized_dates.json  — plain list of ISO date strings
    outputs/part2_result.json            — HTX-schema classification items
    outputs/part2_evidence.json          — full per-date audit trail
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.part1.extractor import build_llm
from src.part2.workflow import run_part2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_part2")


async def _async_main() -> None:
    load_dotenv()

    pdf_path = Path(
        os.environ.get("SOURCE_PDF", "data/fy2024_analysis_of_revenue_and_expenditure.pdf")
    )
    if not pdf_path.exists():
        logger.error("Source PDF not found at %s", pdf_path.resolve())
        sys.exit(1)

    try:
        llm = build_llm()
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Using model: %s", os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"))
    logger.info("Source PDF : %s", pdf_path.resolve())

    normalized_dates, results, evidence = await run_part2(pdf_path, llm)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Stage 1 output — plain JSON list, not wrapped in an object
    (outputs_dir / "part2_normalized_dates.json").write_text(
        json.dumps(normalized_dates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Stage 2 output — exact HTX schema: original_text, normalized_date, status only
    (outputs_dir / "part2_result.json").write_text(
        json.dumps(
            [item.model_dump() for item in results],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Audit output — richer engineering data kept separate
    (outputs_dir / "part2_evidence.json").write_text(
        json.dumps(
            [ev.model_dump() for ev in evidence],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info("[Part2] Normalized dates: %s", normalized_dates)
    logger.info("[Part2] Results written: %d items", len(results))


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
