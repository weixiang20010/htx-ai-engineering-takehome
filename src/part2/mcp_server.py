"""
Local MCP server exposing the normalize_date tool over stdio transport.

Architecture:
    date_normalizer.normalize_date_value()  ← pure Python, independently testable
            |
    @mcp.tool() wrapper                     ← MCP interface only
            |
    stdio transport                         ← local, no network

Run as a module from the project root:
    python -m src.part2.mcp_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .date_normalizer import normalize_date_value

mcp = FastMCP("date-normalizer")


@mcp.tool()
def normalize_date(date_text: str) -> str:
    """
    Normalize a natural-language date string to ISO YYYY-MM-DD format.

    Args:
        date_text: A date string such as "16 February 2024" or "15 Feb 2008".

    Returns:
        ISO-formatted date string "YYYY-MM-DD".
    """
    return normalize_date_value(date_text)


if __name__ == "__main__":
    mcp.run(transport="stdio")
