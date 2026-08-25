"""
MCP client utilities — start the local server subprocess and yield a live session.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# CWD for the server subprocess — must be the project root so that
# `python -m src.part2.mcp_server` resolves the src package correctly
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def open_mcp_session(
    server_module: str = "src.part2.mcp_server",
) -> AsyncGenerator[tuple[ClientSession, list[BaseTool]], None]:
    """
    Start the MCP server subprocess and yield (session, lc_tools).

    session   — supports session.call_tool() for direct execution and trace capture
    lc_tools  — LangChain BaseTool wrappers bound to the LLM for tool discovery
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", server_module],
        cwd=str(_PROJECT_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            lc_tools = await load_mcp_tools(session)
            yield session, lc_tools
