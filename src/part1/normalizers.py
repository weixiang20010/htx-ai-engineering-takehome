"""
Deterministic numeric normalization — no LLM involvement.

All conversions in this module are pure Python and must not call any
external service.  They operate on the exact raw strings retrieved from the
source PDF.
"""
from __future__ import annotations

import re


def normalize_to_float(text: str) -> float:
    """
    Convert a formatted financial string to a plain Python float.

    Supported formats
    -----------------
    "20,352"         -> 20352.0   (thousands separator)
    "17.0%"          -> 17.0      (percentage — strip %)
    "$28.4"          -> 28.4      (currency prefix — strip $)
    "$28.4 billion"  -> 28.4      (currency + unit suffix)
    "(3.57)"         -> -3.57     (parenthesised negative)
    "-3.57"          -> -3.57     (minus sign)
    "1.72"           -> 1.72      (plain float)

    Raises
    ------
    ValueError
        If no numeric value can be extracted from *text*.
    """
    if not text or not text.strip():
        raise ValueError("Cannot normalize empty string to float")

    cleaned = text.strip()

    # Detect parenthesised negatives before stripping punctuation
    negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        negative = True
        cleaned = cleaned[1:-1]
    elif cleaned.startswith("-"):
        negative = True
        cleaned = cleaned[1:]

    # Strip currency symbols and percent signs
    cleaned = re.sub(r"[$%]", "", cleaned).strip()

    # Strip trailing unit words (billion, million, trillion)
    cleaned = re.sub(
        r"\s*(billion|million|trillion)\s*$", "", cleaned, flags=re.IGNORECASE
    ).strip()

    # Remove thousands separators
    cleaned = cleaned.replace(",", "")

    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse {text!r} as float (normalized to {cleaned!r})"
        ) from exc

    return -value if negative else value
