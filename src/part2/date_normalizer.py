"""
Pure Python deterministic date normalisation: natural-language date → ISO YYYY-MM-DD.

This module is the source of truth for date conversion. The MCP server wraps it;
unit tests call it directly without any MCP or LLM dependency.
"""
from __future__ import annotations

from datetime import datetime

# Explicit format list — fail-closed on ambiguous numeric formats (e.g. 02/03/2024)
_FORMATS: tuple[str, ...] = (
    "%d %B %Y",   # 16 February 2024
    "%d %b %Y",   # 16 Feb 2024
    "%Y-%m-%d",   # Already ISO
)


class DateNormalizationError(ValueError):
    """Raised when date_text cannot be parsed by any supported format."""


def normalize_date_value(date_text: str) -> str:
    """
    Convert a natural-language date string to ISO YYYY-MM-DD.

    Does not call date.today() or datetime.now().

    Raises
    ------
    DateNormalizationError
        If date_text is empty, blank, or matches no supported format.
    """
    if not date_text or not date_text.strip():
        raise DateNormalizationError("date_text is empty or blank")

    cleaned = " ".join(date_text.strip().split())

    for fmt in _FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue

    raise DateNormalizationError(
        f"Unsupported or ambiguous date format: {date_text!r}. "
        f"Supported formats: {_FORMATS}"
    )
