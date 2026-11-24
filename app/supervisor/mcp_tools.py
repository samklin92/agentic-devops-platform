"""MCP-backed tools providers for the Kubernetes investigator.

A provider opens an MCP session for one investigation, builds the
agent's tools from it with the existing adapter, and closes the session
afterwards, all on the event loop the bridge runs the agent on.
"""

import asyncio
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.mcp_server.adapter import build_mcp_tools
from app.supervisor.kubernetes_bridge import ToolsProvider

INITIALIZE_TIMEOUT_SECONDS = 20

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


def mcp_session_tools_provider(
    session_factory: SessionFactory,
) -> ToolsProvider:
    """Build the agent's tools from an MCP session opened per investigation."""

    @asynccontextmanager
    async def provider(request):
        async with session_factory() as session:
            yield build_mcp_tools(
                session=session,
                pod_name=request.target.pod_name,
                namespace=request.target.namespace,
            )

    return provider


def server_parameters() -> StdioServerParameters:
    """How to start the simulated server. Run from the repository root."""
    # Use the running interpreter, never a bare "python": a bare "python"
    # can resolve to an install that lacks the mcp package.
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server.server"],
    )


@asynccontextmanager
async def stdio_session():
    """Start the simulated Kubernetes MCP server and connect to it."""
    async with stdio_client(server_parameters()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            async with asyncio.timeout(INITIALIZE_TIMEOUT_SECONDS):
                await session.initialize()
            yield session


def stdio_tools_provider() -> ToolsProvider:
    """Tools provider backed by the simulated MCP server over stdio."""
    return mcp_session_tools_provider(stdio_session)