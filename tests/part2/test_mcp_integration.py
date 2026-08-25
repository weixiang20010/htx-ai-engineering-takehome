"""
Real MCP integration tests — cross the actual MCP client/server boundary.

Gemini is NOT required. Each test starts the MCP server subprocess, connects
via stdio, and exercises the normalize_date tool end-to-end.

These tests verify MCP, not just the underlying Python function.
"""
from __future__ import annotations

import sys

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.part2.mcp_client import _PROJECT_ROOT


def _make_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.part2.mcp_server"],
        cwd=str(_PROJECT_ROOT),
    )


class TestMCPNormalizeDateTool:
    async def test_tool_is_discoverable(self) -> None:
        """normalize_date appears in the server's tool listing."""
        async with stdio_client(_make_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                assert "normalize_date" in tool_names

    async def test_normalize_distribution_date(self) -> None:
        """MCP call returns correct ISO date for the page 1 distribution date."""
        async with stdio_client(_make_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "normalize_date", {"date_text": "16 February 2024"}
                )
                assert not result.isError
                assert result.content[0].text == "2024-02-16"

    async def test_normalize_estate_duty_date(self) -> None:
        """MCP call returns correct ISO date for the page 36 Estate Duty date."""
        async with stdio_client(_make_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "normalize_date", {"date_text": "15 February 2008"}
                )
                assert not result.isError
                assert result.content[0].text == "2008-02-15"

    async def test_abbreviated_month_via_mcp(self) -> None:
        """MCP correctly delegates abbreviated month formats to the normalizer."""
        async with stdio_client(_make_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "normalize_date", {"date_text": "16 Feb 2024"}
                )
                assert not result.isError
                assert result.content[0].text == "2024-02-16"

    async def test_invalid_date_returns_mcp_error(self) -> None:
        """Invalid input propagates as an MCP tool error, not an uncaught exception."""
        async with stdio_client(_make_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "normalize_date", {"date_text": "not a date at all"}
                )
                assert result.isError
