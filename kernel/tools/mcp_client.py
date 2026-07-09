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

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "MCPClientConfig",
    "MCPConnectionError",
    "MCPToolPool",
    "MCPToolResult",
    "MCPToolSpec",
    "SyntheticToolSpec",
    "SYNTHETIC_SERVER_ID",
]

#: The ``server_id`` stamped on synthetic (host-owned) tool specs so the
#: catalog can distinguish them from MCP-discovered tools. Synthetic tools
#: are never routed to a subprocess — they dispatch to an in-process handler.
SYNTHETIC_SERVER_ID = "__synthetic__"


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
        warm: When True, this server is a candidate for the host-owned
            long-lived pool: it is spawned once at runtime init and kept
            alive across turns instead of spawn/teardown per turn. The
            kernel still spawns non-warm (cold) servers per turn as today.
            Default False (zero behavior change for existing callers).
        required: Meaningful only when ``warm=True``. When True, startup
            failure (after bounded retry-with-backoff) raises
            :class:`MCPConnectionError`, propagating to the caller so a
            load-bearing warm server fails the turn or startup loudly.
            When False, startup failure is logged and the pool proceeds
            without that server's tools. Ignored when ``warm=False``;
            cold servers keep their existing swallow-and-warn behavior.
            Default False.
        data_dir_path: The resolved data directory (post-subdir) the pool
            should ensure exists before spawning the server. The compiler
            sets this from ``MCPServerSpec.data_dir`` / ``data_dir_subdir``.
            May be ``None`` for servers that manage their own storage.
        ensure_data_dir: When True, the pool creates ``data_dir_path``
            (with ``parents=True, exist_ok=True``) before spawning the
            server. Lets a server own a managed subfolder that is guaranteed
            to pre-exist. Default False.
    """

    server_id: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    tool_policy: Any = None
    warm: bool = False
    required: bool = False
    data_dir_path: str | None = None
    ensure_data_dir: bool = False
    # HTTP transport (runtime_kind=HTTP). When url is set, the pool connects
    # via streamable-HTTP instead of spawning a subprocess. command/args/env
    # are ignored for HTTP configs.
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


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


@dataclass(frozen=True, slots=True)
class SyntheticToolSpec:
    """One host-owned tool exposed in the same catalog as MCP-discovered tools.

    Synthetic tools are a **generic** mechanism: any well-known or made-up
    call can be registered as a synthetic tool and dispatched in-process,
    without an MCP server subprocess backing it. The canonical use case is
    memory aliases (``memory.add`` / ``memory.search`` / ``memory.list``)
    that route to whichever backend is wired as default — but the mechanism
    is not memory-specific. A synthetic tool may also redirect to an MCP
    server tool via :func:`kernel.tools.synthetic.mcp_alias_tool`.

    On the wire to the model, a synthetic tool is indistinguishable from an
    MCP-discovered tool: it flows through the same
    ``mcp_tool_to_openai_function`` translator and ``call_tool`` dispatch.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]


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

    Warm vs cold servers (host-owned long-lived pool):

    A config with ``warm=True`` is a candidate for a host-owned pool whose
    ``start()`` is called once at runtime init and whose ``aclose()`` is
    called at process exit. The kernel's ``run_turn`` accepts such a pool
    via ``TurnHooks.mcp_tool_pool`` and reuses it across turns without
    spawning/tearing down subprocesses per turn. A config with
    ``warm=False`` (default) is spawned and torn down per turn as before.

    ``required`` (only meaningful with ``warm=True``) designates whether a
    startup failure is fatal (raise :class:`MCPConnectionError` after
    bounded retry) or non-fatal (log and proceed without the server's
    tools). Mid-session crashes of warm servers are recovered
    event-drivenly via :meth:`restart_warm` from :meth:`call_tool` /
    :meth:`list_tools`, not via a background scheduler.
    """

    # Retry-with-backoff for warm server (re)starts. Bounded so a
    # permanently-broken required server fails fast rather than hanging.
    _WARM_STARTUP_ATTEMPTS = 3
    _WARM_BACKOFF_SECONDS = (0.2, 0.4, 0.8)

    def __init__(self, configs: list[MCPClientConfig]) -> None:
        self._configs = list(configs)
        self._warm_configs = [c for c in self._configs if c.warm]
        self._cold_configs = [c for c in self._configs if not c.warm]
        self._sessions: dict[str, Any] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._tool_index: dict[str, tuple[str, MCPToolSpec]] = {}
        # Per-server tool index: server_id -> {tool_name -> spec}. Unlike
        # _tool_index (first-registered-wins across the whole pool), this keeps
        # every server's tools even when names collide across servers. Used by
        # server-scoped dispatch (call_tool(..., server_id=...)) so an alias can
        # pin a call to a specific backend regardless of bare-name collisions.
        self._tool_index_by_server: dict[str, dict[str, MCPToolSpec]] = {}
        self._tools: list[MCPToolSpec] = []
        self._started = False
        self._warm_failed: set[str] = set()
        """Server ids of warm servers whose (re)start failed non-fatally."""
        self._synthetic_handlers: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] = {}
        """Host-owned in-process handlers keyed by tool name. Populated by
        :meth:`add_synthetic_tools`. Synthetic tools dispatch through here
        instead of routing to an MCP server subprocess."""

    @property
    def configs(self) -> list[MCPClientConfig]:
        return list(self._configs)

    @property
    def warm_configs(self) -> list[MCPClientConfig]:
        return list(self._warm_configs)

    @property
    def cold_configs(self) -> list[MCPClientConfig]:
        return list(self._cold_configs)

    def warm_server_ids(self) -> list[str]:
        """Return ids of warm servers currently connected."""
        return [c.server_id for c in self._warm_configs if c.server_id in self._sessions]

    def warm_failed_ids(self) -> list[str]:
        """Return ids of warm servers that failed (re)start and are down."""
        return sorted(self._warm_failed)

    def add_synthetic_tools(self, tools: list[SyntheticToolSpec]) -> None:
        """Register host-owned in-process tools into the merged catalog.

        Synthetic tools appear alongside MCP-discovered tools in
        :meth:`list_tools` and dispatch through :meth:`call_tool` to their
        in-process handler — no subprocess backing them. The canonical use
        case is memory aliases (``memory.add`` / ``memory.search`` /
        ``memory.list``), but the mechanism is generic: any well-known or
        made-up call can be registered as a synthetic tool.

        Register synthetic tools **after** :meth:`start` so handlers that
        redirect to an MCP server tool (see
        :func:`kernel.tools.synthetic.mcp_alias_tool`) can close over the
        live pool. Synthetic tools are preserved across
        :meth:`refresh_tool_index` refreshes (they are host-owned, not
        discovered from an MCP server).

        Name collisions follow the same first-registered-wins policy as
        MCP-discovered tools, with a warning logged. Synthetic tools are
        registered after the MCP index is built, so a synthetic alias can
        intentionally override a server-native name if desired — call
        :meth:`refresh_tool_index` first to seed the MCP index, then
        register synthetic tools.
        """
        for tool in tools:
            if tool.name in self._synthetic_handlers or tool.name in self._tool_index:
                existing = (
                    "synthetic" if tool.name in self._synthetic_handlers
                    else self._tool_index[tool.name][0]
                )
                logger.warning(
                    "Synthetic tool name collision: %r already registered by %r; "
                    "existing entry wins.",
                    tool.name,
                    existing,
                )
                continue
            spec = MCPToolSpec(
                server_id=SYNTHETIC_SERVER_ID,
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            self._synthetic_handlers[tool.name] = tool.handler
            self._tool_index[tool.name] = (SYNTHETIC_SERVER_ID, spec)
            self._tools.append(spec)

    def _reindex_synthetic_tools(self) -> None:
        """Re-register synthetic tool entries after a tool-index rebuild.

        Called by :meth:`_refresh_tool_index` so host-owned synthetic tools
        survive catalog refreshes (they are not rediscovered from MCP servers).
        """
        for name, _handler in self._synthetic_handlers.items():
            # Reconstruct the MCPToolSpec and re-insert. Synthetic handlers are
            # never dropped by a refresh — only by aclose().
            if name in self._tool_index:
                continue
            # We stored no extra metadata beyond name/handler; rebuild a
            # minimal spec. The original description/schema are recovered from
            # the prior _tools list if present, else a minimal default.
            prior = next((t for t in self._tools if t.name == name), None)
            spec = MCPToolSpec(
                server_id=SYNTHETIC_SERVER_ID,
                name=name,
                description=prior.description if prior else f"Synthetic tool {name}",
                input_schema=prior.input_schema if prior else {"type": "object", "properties": {}},
            )
            self._tool_index[name] = (SYNTHETIC_SERVER_ID, spec)
            self._tools.append(spec)

    async def __aenter__(self) -> MCPToolPool:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        """Spawn and initialize every configured MCP server.

        Warm servers (``config.warm=True``) are (re)started with bounded
        retry-with-backoff. A warm server with ``required=True`` whose
        retries are exhausted raises :class:`MCPConnectionError`; a
        non-required warm server is logged and skipped. Cold servers
        keep the original swallow-and-warn behavior.
        """

        if self._started:
            return
        self._started = True
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Warm servers: bounded retry; required failures are fatal.
        for config in self._warm_configs:
            await self._connect_warm(config)

        # Cold servers: best-effort, swallow startup errors.
        for config in self._cold_configs:
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

    async def _connect_warm(self, config: MCPClientConfig) -> None:
        """Connect one warm server with bounded retry-with-backoff.

        On exhaustion: raise :class:`MCPConnectionError` when
        ``config.required`` is True, else record the server as failed and
        continue (its tools are simply absent from the catalog).
        """
        last_exc: Exception | None = None
        for attempt in range(self._WARM_STARTUP_ATTEMPTS):
            try:
                await self._connect(config)
                self._warm_failed.discard(config.server_id)
                return
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                if attempt < self._WARM_STARTUP_ATTEMPTS - 1:
                    delay = self._WARM_BACKOFF_SECONDS[min(
                        attempt, len(self._WARM_BACKOFF_SECONDS) - 1
                    )]
                    logger.warning(
                        "Warm MCP server %r start attempt %d/%d failed: %s: %s; "
                        "retrying in %.2fs",
                        config.server_id,
                        attempt + 1,
                        self._WARM_STARTUP_ATTEMPTS,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        # Exhausted retries.
        if config.required:
            raise MCPConnectionError(
                f"Required warm MCP server '{config.server_id}' failed to start "
                f"after {self._WARM_STARTUP_ATTEMPTS} attempts: "
                f"{type(last_exc).__name__ if last_exc else 'unknown'}: {last_exc}"
            ) from last_exc
        logger.warning(
            "Warm MCP server %r failed to start after %d attempts (non-required); "
            "proceeding without it.",
            config.server_id,
            self._WARM_STARTUP_ATTEMPTS,
        )
        self._warm_failed.add(config.server_id)

    async def _connect(self, config: MCPClientConfig) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        assert self._exit_stack is not None
        if config.url is not None:
            await self._connect_http(config)
            return

        # Optionally pre-create the data directory before spawning. A server
        # may own a managed subfolder under the host data-dir and expect it to
        # exist; the host guarantees it here so the server can use it (possibly
        # destructively) without a pre-flight check of its own.
        if config.ensure_data_dir and config.data_dir_path:
            Path(config.data_dir_path).mkdir(parents=True, exist_ok=True)

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

    async def _connect_http(self, config: MCPClientConfig) -> None:
        """Connect one streamable-HTTP MCP server.

        Uses the same AsyncExitStack discipline as stdio so the
        ``streamable_http_client`` + ``ClientSession`` cancel scopes live in
        the pool's own long-lived stack (entered once in ``start()``), not a
        throwaway loop. This avoids the anyio cross-task cancel-scope error
        that per-call ``asyncio.run`` caused for the old memory HTTP client.
        """
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        assert self._exit_stack is not None
        assert config.url is not None
        read, write, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(config.url, headers=config.headers)
        )
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[config.server_id] = session

    async def _refresh_tool_index(self) -> None:
        """Merge tool lists from all connected servers into one index.

        Host-owned synthetic tools are preserved across refreshes (they are
        not rediscovered from MCP servers) via :meth:`_reindex_synthetic_tools`.
        """

        from kernel.mcp_launch.tool_filter import filter_tools

        self._tools = []
        self._tool_index = {}
        self._tool_index_by_server = {}
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
                # Warm server session may have died; attempt a restart.
                if config.warm:
                    await self._maybe_restart_warm(config)
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
            server_index: dict[str, MCPToolSpec] = {}
            for spec in server_tools:
                # Per-server index keeps every tool, collisions or not.
                server_index[spec.name] = spec
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
            self._tool_index_by_server[config.server_id] = server_index

        # Re-seed host-owned synthetic tools after the MCP index is rebuilt.
        # Synthetic tools are not rediscovered from servers — they are owned
        # by the host and must survive catalog refreshes.
        self._reindex_synthetic_tools()

    async def refresh_tool_index(self) -> None:
        """Rebuild the merged tool catalog from connected sessions.

        Public entry point for hosts that change a server's tool surface
        at runtime (e.g. after installing/uninstalling packs that affect a
        cold server whose tool surface may change). Re-queries every connected
        session and rebuilds the index.
        """
        await self._refresh_tool_index()

    async def restart_warm(self, server_id: str) -> bool:
        """Re-spawn one warm server after a mid-session crash.

        Returns True if the server is (re)connected after the attempt,
        False if it remains down. Uses the same bounded retry-with-backoff
        as :meth:`start`. A non-required warm server that fails is recorded
        in ``warm_failed_ids()``; a required one raises
        :class:`MCPConnectionError`.
        """
        config = next((c for c in self._warm_configs if c.server_id == server_id), None)
        if config is None:
            logger.warning("restart_warm: %r is not a warm server in this pool", server_id)
            return False
        # Drop any stale session for this server before retrying.
        self._sessions.pop(server_id, None)
        await self._connect_warm(config)
        return server_id in self._sessions

    async def _maybe_restart_warm(self, config: MCPClientConfig) -> None:
        """Best-effort restart of a warm server whose session errored.

        Swallows failures for non-required servers; lets required ones raise
        so the caller can fail the turn loudly.
        """
        try:
            await self.restart_warm(config.server_id)
        except MCPConnectionError:
            if config.required:
                raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "Warm MCP server %r restart attempt failed: %s: %s",
                config.server_id,
                type(exc).__name__,
                exc,
            )

    async def list_tools(self) -> list[MCPToolSpec]:
        """Return the merged tool catalog from all connected servers."""

        if not self._started:
            await self.start()
        return list(self._tools)

    async def call_tool(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        server_id: str | None = None,
    ) -> MCPToolResult:
        """Route one tool call to the server that owns the tool.

        Host-owned synthetic tools (registered via :meth:`add_synthetic_tools`)
        are dispatched to their in-process handler before any MCP server
        routing — they have no subprocess backing them.

        When ``server_id`` is provided, dispatch is **server-scoped**: the
        tool is looked up in that server's own index
        (:attr:`_tool_index_by_server`) instead of the merged bare-name index,
        and the call is routed to that server's session. This bypasses the
        first-registered-wins collision policy so an alias can pin a call to
        a specific backend (e.g. ``memory.add`` → ``mem0``'s ``add_memory``
        even when ``lisa-memory`` also exposes ``add_memory``). If the target
        server is not connected, it is lazily started (cold servers) or
        restarted (warm servers) once before failing; no bare-name fallback
        occurs when ``server_id`` is set.
        """

        if not self._started:
            await self.start()
        synthetic_handler = self._synthetic_handlers.get(tool_id)
        if synthetic_handler is not None:
            try:
                output_text = await synthetic_handler(arguments)
            except Exception as exc:  # pylint: disable=broad-except
                return MCPToolResult(
                    tool_id=tool_id,
                    output_text=f"Synthetic tool '{tool_id}' failed: {type(exc).__name__}: {exc}",
                    is_error=True,
                )
            return MCPToolResult(tool_id=tool_id, output_text=output_text or "Done")

        if server_id is not None:
            return await self._call_tool_scoped(tool_id, arguments, server_id)

        entry = self._tool_index.get(tool_id)
        if entry is None:
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"Tool '{tool_id}' is not registered on any MCP server.",
                is_error=True,
            )
        server_id, _spec = entry
        config = next((c for c in self._configs if c.server_id == server_id), None)
        session = self._sessions.get(server_id)
        if session is None:
            # Warm server may have crashed; try one restart before failing.
            if config is not None and config.warm:
                await self._maybe_restart_warm(config)
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
            # Session may have died mid-call; attempt one warm restart.
            if config is not None and config.warm:
                try:
                    await self.restart_warm(server_id)
                except MCPConnectionError:
                    if config.required:
                        raise
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"MCP tool '{tool_id}' failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return self._build_result(tool_id, result)

    async def _call_tool_scoped(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        server_id: str,
    ) -> MCPToolResult:
        """Dispatch a tool call pinned to a specific server.

        Used by server-scoped aliases (e.g. ``memory.add`` → the default
        memory backend) to bypass the merged bare-name index. The target
        server is lazily started if not connected: cold servers are connected
        on first call, warm servers are restarted once. No bare-name fallback.
        """
        config = next((c for c in self._configs if c.server_id == server_id), None)
        if config is None:
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"Unknown MCP server '{server_id}'.",
                is_error=True,
            )
        session = self._sessions.get(server_id)
        if session is None:
            # Lazily start the target server. Cold servers are connected on
            # first scoped call; warm servers get one restart attempt.
            try:
                if config.warm:
                    await self._connect_warm(config)
                else:
                    await self._connect(config)
                    # New session: refresh so this server's tools are indexed.
                    await self._refresh_tool_index()
            except Exception as exc:  # pylint: disable=broad-except
                return MCPToolResult(
                    tool_id=tool_id,
                    output_text=(
                        f"MCP server '{server_id}' for tool '{tool_id}' failed to "
                        f"start: {type(exc).__name__}: {exc}"
                    ),
                    is_error=True,
                )
            session = self._sessions.get(server_id)
            if session is None:
                return MCPToolResult(
                    tool_id=tool_id,
                    output_text=(
                        f"MCP server '{server_id}' for tool '{tool_id}' is not connected."
                    ),
                    is_error=True,
                )
        # Look up the tool in the target server's own index AFTER ensuring the
        # session is connected (a cold server's index is empty until started).
        server_index = self._tool_index_by_server.get(server_id, {})
        if tool_id not in server_index:
            return MCPToolResult(
                tool_id=tool_id,
                output_text=(
                    f"Tool '{tool_id}' is not registered on server '{server_id}'."
                ),
                is_error=True,
            )
        try:
            result = await session.call_tool(tool_id, arguments=arguments)
        except Exception as exc:  # pylint: disable=broad-except
            if config.warm:
                try:
                    await self.restart_warm(server_id)
                except MCPConnectionError:
                    if config.required:
                        raise
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"MCP tool '{tool_id}' failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return self._build_result(tool_id, result)

    @staticmethod
    def _build_result(tool_id: str, result: Any) -> MCPToolResult:
        """Convert an MCP session call result into an MCPToolResult."""
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
        self._tool_index_by_server.clear()
        self._tools = []
        self._synthetic_handlers.clear()
        self._started = False
