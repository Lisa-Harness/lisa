"""MCP-native tool client pool for the kernel turn runner.

Manages N MCP server subprocesses and exposes a unified tool catalog. The
kernel speaks MCP natively for tool discovery and dispatch; provider-specific
schemas (OpenAI function, Anthropic tool-use) are translated at the provider
edge by ``kernel.tools.schema_translator``.

Design — MCP at the edges, Protocols in the middle:

    MCP is a wire protocol. The kernel uses it to cross process boundaries
    (external tool servers, external memory servers, kernel-owned servers
    running out-of-process). In-process seams (MemoryStore, BlobIO, etc.)
    stay as typed Protocols; they do not go through MCP.

    The ``MCPToolPool`` owns the subprocess lifecycle: spawn, health check,
    restart, and graceful shutdown. The host declares which servers to run
    via ``TurnHooks.mcp_tool_servers``; the kernel manages the rest.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MCPClientConfig",
    "MCPConnectionError",
    "MCPToolPool",
    "MCPToolResult",
    "MCPToolSpec",
]


@dataclass(frozen=True, slots=True)
class MCPClientConfig:
    """One configured MCP server connection.

    Attributes:
        server_id: Stable identifier for this server. Used to route tool
            calls and to correlate logs.
        command: Executable to spawn (e.g. ``python``, ``npx``, ``uvx``).
        args: Arguments passed to ``command``.
        env: Optional environment overrides for the subprocess. When
            ``None``, the subprocess inherits the parent environment.
        cwd: Optional working directory for the subprocess.
        tool_policy: Optional allow/deny list for filtering discovered tools.
    """

    server_id: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    tool_policy: Any = None


@dataclass(frozen=True, slots=True)
class MCPToolSpec:
    """One tool discovered from an MCP server.

    The ``input_schema`` is a JSON-Schema dict in MCP's native format. The
    schema translator converts this to provider-specific shapes.
    """

    server_id: str
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class MCPToolResult:
    """Result of one MCP tool call."""

    tool_id: str
    output_text: str = ""
    is_error: bool = False
    raw: Any = None


class MCPConnectionError(RuntimeError):
    """Raised when an MCP server cannot be reached or initialized."""


class MCPToolPool:
    """Manages N MCP server subprocesses and exposes a unified tool catalog.

    The pool is async-context-managed. On enter, it spawns every configured
    server, initializes each client session, and caches the merged tool
    catalog. On exit, it closes every session and terminates every subprocess.

    Tool dispatch routes by tool name to the server that owns the tool. If
    two servers expose the same tool name, the first one registered wins and
    a warning is logged.
    """

    def __init__(self, configs: list[MCPClientConfig]) -> None:
        self._configs = list(configs)
        self._sessions: dict[str, Any] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._tool_index: dict[str, tuple[str, MCPToolSpec]] = {}
        self._tools: list[MCPToolSpec] = []
        self._started = False

    @property
    def configs(self) -> list[MCPClientConfig]:
        return list(self._configs)

    async def __aenter__(self) -> "MCPToolPool":
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        """Spawn and initialize every configured MCP server."""

        if self._started:
            return
        self._started = True
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        for config in self._configs:
            try:
                await self._connect(config)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "MCP server %r failed to start: %s: %s",
                    config.server_id,
                    type(exc).__name__,
                    exc,
                )
        await self._refresh_tool_index()

    async def _connect(self, config: MCPClientConfig) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        assert self._exit_stack is not None
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )
        # Use AsyncExitStack to manage stdio_client + ClientSession lifetimes.
        # Manually calling __aenter__/__aexit__ on anyio-based context managers
        # breaks cancel-scope tracking; AsyncExitStack preserves the correct
        # nesting and ensures cleanup on failure.
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[config.server_id] = session

    async def _refresh_tool_index(self) -> None:
        """Merge tool lists from all connected servers into one index."""

        from kernel.mcp_launch.tool_filter import filter_tools

        self._tools = []
        self._tool_index = {}
        for config in self._configs:
            session = self._sessions.get(config.server_id)
            if session is None:
                continue
            try:
                result = await session.list_tools()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "MCP server %r list_tools failed: %s: %s",
                    config.server_id,
                    type(exc).__name__,
                    exc,
                )
                continue
            server_tools = [
                MCPToolSpec(
                    server_id=config.server_id,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {"type": "object", "properties": {}},
                )
                for tool in result.tools
            ]
            if config.tool_policy is not None:
                server_tools = filter_tools(server_tools, config.tool_policy)
            for spec in server_tools:
                if spec.name in self._tool_index:
                    existing_server = self._tool_index[spec.name][0]
                    logger.warning(
                        "MCP tool name collision: %r defined by %r and %r; %r wins.",
                        spec.name,
                        existing_server,
                        spec.server_id,
                        existing_server,
                    )
                    continue
                self._tool_index[spec.name] = (config.server_id, spec)
                self._tools.append(spec)

    async def list_tools(self) -> list[MCPToolSpec]:
        """Return the merged tool catalog from all connected servers."""

        if not self._started:
            await self.start()
        return list(self._tools)

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Route one tool call to the server that owns the tool."""

        if not self._started:
            await self.start()
        entry = self._tool_index.get(tool_id)
        if entry is None:
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"Tool '{tool_id}' is not registered on any MCP server.",
                is_error=True,
            )
        server_id, _spec = entry
        session = self._sessions.get(server_id)
        if session is None:
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"MCP server '{server_id}' for tool '{tool_id}' is not connected.",
                is_error=True,
            )
        try:
            result = await session.call_tool(tool_id, arguments=arguments)
        except Exception as exc:  # pylint: disable=broad-except
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"MCP tool '{tool_id}' failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        text_parts: list[str] = []
        for content_block in result.content or []:
            text = getattr(content_block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        return MCPToolResult(
            tool_id=tool_id,
            output_text="\n".join(text_parts) if text_parts else "Done",
            is_error=bool(getattr(result, "is_error", False)),
            raw=result,
        )

    async def aclose(self) -> None:
        """Close every session and terminate every subprocess."""

        self._sessions.clear()
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:  # pylint: disable=broad-except
                logger.debug("MCP exit stack close failed", exc_info=True)
        self._exit_stack = None
        self._tool_index.clear()
        self._tools = []
        self._started = False