"""MCP-backed ``InputGuard`` and ``OutputGuard`` adapters.

Each adapter opens a per-call stdio session to a ``lisa-guard`` MCP server,
calls ``validate_input``/``validate_output``, parses the JSON response, and
returns a :class:`~kernel.seams.protocols.GuardResult`. The server handles
audit logging internally — the adapter does not need an event bus.

v1 uses per-call stdio sessions (open → call → close), matching the
``MCPMemoryClient`` pattern in ``kernel/mcp_client.py``.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from kernel.models import SurfaceContext
from kernel.seams.protocols import GuardResult
from kernel.tools.mcp_client import MCPClientConfig

__all__ = ["MCPInputGuard", "MCPOutputGuard"]


def _extract_text(result: Any) -> str:
    """Extract text from an MCP call_tool result."""
    if result.content and len(result.content) > 0:
        return result.content[0].text  # type: ignore[union-attr]
    return ""


def _parse_guard_response(text: str) -> GuardResult:
    """Parse the JSON response from validate_input/validate_output."""
    if not text:
        return GuardResult(allowed=True)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return GuardResult(allowed=True)
    return GuardResult(
        allowed=bool(data.get("allowed", True)),
        blocked_reason=data.get("blocked_reason"),
        sanitized_text=data.get("sanitized_text"),
        annotations=data.get("annotations", {}),
    )


class MCPInputGuard:
    """InputGuard backed by an MCP guard server.

    The adapter opens a stdio session to the configured ``lisa-guard`` server,
    calls ``validate_input``, and returns a :class:`GuardResult`.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )

    async def validate(
        self,
        query: str,
        surface: SurfaceContext,
    ) -> GuardResult:
        async with stdio_client(self._params) as (read, write_stream):
            async with ClientSession(read, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "validate_input",
                    arguments={
                        "query": query,
                        "surface_type": surface.surface_type,
                        "surface_id": surface.surface_id,
                        "memory_scope": surface.memory_scope,
                        "thread_owner_id": surface.thread_owner_id,
                    },
                )
                return _parse_guard_response(_extract_text(result))


class MCPOutputGuard:
    """OutputGuard backed by an MCP guard server.

    The adapter opens a stdio session to the configured ``lisa-guard`` server,
    calls ``validate_output``, and returns a :class:`GuardResult`.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )

    async def validate(
        self,
        text: str,
        surface: SurfaceContext,
    ) -> GuardResult:
        async with stdio_client(self._params) as (read, write_stream):
            async with ClientSession(read, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "validate_output",
                    arguments={
                        "text": text,
                        "surface_type": surface.surface_type,
                        "surface_id": surface.surface_id,
                        "memory_scope": surface.memory_scope,
                        "thread_owner_id": surface.thread_owner_id,
                    },
                )
                return _parse_guard_response(_extract_text(result))
