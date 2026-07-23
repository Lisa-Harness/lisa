"""MCP-native tool client pool for the kernel turn runner.

Manages N MCP server subprocesses and exposes a unified tool catalog. The
kernel speaks MCP natively for tool discovery and dispatch; provider-specific
schemas (OpenAI function, Anthropic tool-use) are translated at the provider
edge by ``kernel.tools.schema_translator``.

Tool-id keying convention (see ``kernel.tools.registry`` for the full split):

    * **MCP-native tools** (this module) are keyed **namespaced** as
      ``server_id.tool`` in the merged ``_tool_index`` (e.g.
      ``lisa-storage.read_model_manifest``) so two servers exposing the same
      bare name never collide in the model-facing catalog. A precomputed
      ``_bare_to_server_index`` maps bare names → ``server_id`` for internal
      callers (operations) that pass bare MCP tool names; resolution is an
      O(1) dict lookup, not a side-effecting scan.
    * **Internal pack tools** (``kernel.tools.registry``) are keyed **bare**
      and dispatched in-process. They are not in this pool's index.

Design — MCP at the edges, Protocols in the middle:

    MCP is a wire protocol. The kernel uses it to cross process boundaries
    (external tool servers, external memory servers, kernel-owned servers
    running out-of-process). In-process seams (MemoryStore, StateProvider, etc.)
    stay as typed Protocols; they do not go through MCP.

    The ``MCPToolPool`` owns the subprocess lifecycle: spawn, health check,
    restart, and graceful shutdown. The host declares which servers to run
    via ``TurnHooks.mcp_tool_servers``; the kernel manages the rest.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SYNTHETIC_SERVER_ID",
    "MCPClientConfig",
    "MCPConnectionError",
    "MCPToolPool",
    "MCPToolResult",
    "MCPToolSpec",
    "SyntheticToolSpec",
    "bare_tool_name",
    "namespaced_tool_name",
]

#: The ``server_id`` stamped on synthetic (host-owned) tool specs so the
#: catalog can distinguish them from MCP-discovered tools. Synthetic tools
#: are never routed to a subprocess — they dispatch to an in-process handler.
SYNTHETIC_SERVER_ID = "__synthetic__"


def namespaced_tool_name(server_id: str, tool_name: str) -> str:
    """Build the model-facing namespaced tool name: ``server_id.tool_name``.

    Every MCP-discovered tool is advertised to the LLM under a namespaced
    name so that two servers exposing the same bare tool name (e.g.
    ``lisa-memory`` and ``mem0`` both serving ``add_memory``) never collide
    in the merged catalog.  Synthetic (host-owned) tools are **not**
    namespaced — they keep their well-known bare names (``memory.add``,
    ``memory.search``, …) which form their own implicit namespace.

    The dot separator is chosen for OpenAI function-name compatibility
    (colons are rejected by the API).  Server ids in this codebase use
    hyphens, never dots, so ``split('.', 1)`` is unambiguous.
    """
    return f"{server_id}.{tool_name}"


def bare_tool_name(tool_id: str) -> str:
    """Extract the bare MCP tool name from a namespaced ``server_id.tool`` id.

    For unnamespaced (synthetic) ids, returns the id unchanged.
    """
    if "." in tool_id:
        return tool_id.split(".", 1)[1]
    return tool_id


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
    proxy_policy: Any = None
    warm: bool = False
    required: bool = False
    data_dir_path: str | None = None
    ensure_data_dir: bool = False
    # HTTP transport (runtime_kind=HTTP). When url is set, the pool connects
    # via streamable-HTTP instead of spawning a subprocess. command/args/env
    # are ignored for HTTP configs.
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    aliases: tuple[Any, ...] = field(default_factory=tuple)
    """Generic alias specs (``AliasSpec``) mapping stable model-facing
    names to native tools on this server. The host registers each alias
    as a synthetic tool after pool start."""


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

    ``target_server_id`` is an optional hint naming the MCP server this
    synthetic forwards to (e.g. ``mem0`` for ``memory.add`` → ``mem0``'s
    ``add_memory``). The pool stamps it onto the emitted :class:`MCPToolSpec`
    so listing/grouping tools by server reflects the real destination instead
    of the opaque ``__synthetic__`` marker. Dispatch is unaffected —
    synthetics route through the in-process handler by name, not by server
    id — so a bare ``None`` is safe for synthetics that don't forward to an
    MCP server.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]
    target_server_id: str | None = None
    result_map: Callable[[str], str] | None = None
    """Optional output-normalization transform applied to the MCP tool's
    ``output_text`` after dispatch. When ``None``, the raw output passes
    through unchanged. Used by memory aliases to normalize backend-specific
    response shapes (e.g. Mem0's paginated envelope) into the canonical shape
    operations expect. The handler closure applies it; this field is the
    declarative record for catalog/diagnostics tools that surface it."""


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
    tools). Mid-session crashes of warm servers are detected by the data
    path (``call_tool`` / ``_refresh_tool_index``), which marks the
    session dead and returns an error — it does NOT restart the server.
    Recovery is the host's responsibility via :meth:`restart_warm`, which
    enforces a restart budget (OTP MaxRestarts/MaxSeconds) to prevent
    infinite spawn loops.
    """

    # Retry-with-backoff for warm server (re)starts. Bounded so a
    # permanently-broken required server fails fast rather than hanging.
    _WARM_STARTUP_ATTEMPTS = 3
    _WARM_BACKOFF_SECONDS = (0.2, 0.4, 0.8)

    # ── Supervised lifecycle: restart budget ──────────────────────────
    # At most _MAX_RESTARTS restarts within _RESTART_WINDOW_S seconds.
    # If exceeded, the server is permanently FAILED — no more spawns.
    # This is the OTP one_for_one MaxRestarts/MaxSeconds pattern: it
    # prevents infinite restart loops when a warm server starts (stdio
    # + initialize succeed) but crashes immediately after.
    _MAX_RESTARTS = 3
    _RESTART_WINDOW_S = 60.0

    def __init__(self, configs: list[MCPClientConfig]) -> None:
        self._configs = list(configs)
        self._warm_configs = [c for c in self._configs if c.warm]
        self._cold_configs = [c for c in self._configs if not c.warm]
        self._sessions: dict[str, Any] = {}
        self._child_stacks: dict[str, AsyncExitStack] = {}
        """Per-server AsyncExitStacks. Keyed by server_id. Each entry is
        an INDEPENDENT stack that owns the server's stdio_client +
        ClientSession contexts. The pool does NOT push these onto a
        parent stack — each server's lifecycle is independent so that
        restart_warm can close one server's stack and create a new one
        without affecting other servers or accumulating dead entries on
        a parent."""
        self._tool_index: dict[str, tuple[str, MCPToolSpec]] = {}
        # Per-server tool index: server_id -> {tool_name -> spec}. Unlike
        # _tool_index (first-registered-wins across the whole pool), this keeps
        # every server's tools even when names collide across servers. Used by
        # server-scoped dispatch (call_tool(..., server_id=...)) so an alias can
        # pin a call to a specific backend regardless of bare-name collisions.
        self._tool_index_by_server: dict[str, dict[str, MCPToolSpec]] = {}
        # Precomputed bare-name -> server_id index, rebuilt in
        # ``_refresh_tool_index``. Covers only currently-connected servers;
        # cold servers are merged in when they connect and the index is
        # refreshed. Used by ``_resolve_server_for_bare_name`` so bare-name
        # resolution is an O(1) dict lookup over connected servers, not a
        # linear scan. Cold-server lazy connect still happens when a bare
        # name is absent (a cold server's tools are not in this index until
        # it is connected), but only once per unknown bare name, and never
        # for names already resolvable from warm/connected servers.
        self._bare_to_server_index: dict[str, str] = {}
        self._tools: list[MCPToolSpec] = []
        self._started = False
        self._warm_failed: set[str] = set()
        """Server ids of warm servers whose (re)start failed non-fatally."""
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        """Serializes _child_stacks mutations when warm servers are started
        in parallel (see start()). Each parallel connect creates its own
        AsyncExitStack independently — the lock only guards dict access."""
        self._restarting: set[str] = set()
        """Server ids of warm servers currently being restarted. Guards against
        restart storms: if list_tools fails for a server that is already
        being restarted (by a concurrent caller), the concurrent caller
        skips the restart instead of spawning a second subprocess."""
        self._restart_timestamps: dict[str, list[float]] = {}
        """Per-server restart timestamps for budget enforcement. Keyed by
        server_id, each value is a list of monotonic timestamps. Restart
        budget: at most _MAX_RESTARTS within _RESTART_WINDOW_S seconds.
        When exceeded, the server is permanently FAILED and no more
        subprocess spawns are attempted."""
        self._permanently_failed: set[str] = set()
        """Server ids that have exhausted their restart budget. Once in this
        set, restart_warm() returns False immediately without spawning."""
        self._cold_failed: set[str] = set()
        """Server ids of cold servers that failed to connect. Once in this
        set, ``_ensure_all_cold_connected`` skips them instead of retrying
        on every ``list_tools()`` call (which would cause an unbounded
        re-spawn loop for cold servers that can't start)."""
        self._synthetic_handlers: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] = {}
        """Host-owned in-process handlers keyed by tool name. Populated by
        :meth:`add_synthetic_tools`. Synthetic tools dispatch through here
        instead of routing to an MCP server subprocess."""
        self._synthetic_targets: dict[str, str | None] = {}
        """Optional target MCP server id per synthetic tool name (from
        :attr:`SyntheticToolSpec.target_server_id`). Recovered by
        :meth:`_reindex_synthetic_tools` after a catalog refresh so the
        emitted :class:`MCPToolSpec` keeps its real ``server_id`` across
        rebuilds. ``None`` means the synthetic has no MCP forwarding target —
        the pool falls back to :data:`SYNTHETIC_SERVER_ID` for display."""
        self._dispatch_arg_maps: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        """Per-tool arg maps keyed by namespaced tool name (e.g.
        ``mem0.search_memories``). Applied by :meth:`call_tool` before
        forwarding to the MCP session, so direct calls to native server
        tools get the same ambient-metadata injection as synthetic aliases.
        Populated by :meth:`register_dispatch_arg_map`."""
        self._ambient_resolvers: dict[str, Callable[[], str]] = {}
        """Runtime resolvers for :class:`~kernel.mcp_launch.policy.AmbientSource`
        values. The host registers one callable per source (``user_slug``,
        ``mode_id``, ``app_id``) via :meth:`set_ambient_resolver`. The proxy
        calls them at dispatch time to inject current ambient values into
        outgoing arguments. When a source has no resolver, the proxy uses
        a sensible default (``"default"`` for user_slug/mode_id, ``"lisa"``
        for app_id)."""

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
                    "synthetic"
                    if tool.name in self._synthetic_handlers
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
                server_id=tool.target_server_id or SYNTHETIC_SERVER_ID,
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            self._synthetic_handlers[tool.name] = tool.handler
            self._synthetic_targets[tool.name] = tool.target_server_id
            self._tool_index[tool.name] = (spec.server_id, spec)
            self._tools.append(spec)

    def register_dispatch_arg_map(
        self,
        namespaced_tool_name: str,
        arg_map: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register an arg map applied to direct MCP dispatch for one tool.

        When the model calls a native server tool by its namespaced name
        (e.g. ``mem0.search_memories``), the host applies ``arg_map`` to
        the model-supplied arguments before forwarding them to the MCP
        session. This gives the host a single chokepoint to inject
        ambient scope boundaries (user_id, agent_id, app_id, …) that the
        model cannot escape — the same enforcement the synthetic
        ``memory.*`` aliases already apply via their own arg maps.

        Synthetic alias tools that already apply their own arg map in their
        in-process handler are unaffected — they dispatch before the MCP
        routing path, so this registry is never consulted for them.
        """
        self._dispatch_arg_maps[namespaced_tool_name] = arg_map

    def is_synthetic_tool(self, name: str) -> bool:
        """Return True when ``name`` is a host-owned synthetic (alias) tool.

        Synthetic tools are registered via :meth:`add_synthetic_tools` and
        dispatch to in-process handlers, not MCP server subprocesses. Many
        are aliases that forward to a real MCP server tool (see
        :func:`kernel.tools.synthetic.mcp_alias_tool`); others are purely
        in-process. Tool-listing UIs use this to annotate synthetic entries
        so authors can tell host-owned aliases apart from native MCP tools.
        """
        return name in self._synthetic_handlers

    def synthetic_target_server_id(self, name: str) -> str | None:
        """Return the MCP server id a synthetic alias forwards to, or None.

        Mirrors :attr:`SyntheticToolSpec.target_server_id`. ``None`` means
        the synthetic is purely in-process (no MCP forwarding target).
        """
        return self._synthetic_targets.get(name)

    def _apply_dispatch_arg_map(
        self,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply the registered dispatch arg map for ``tool_id`` if any."""
        arg_map = self._dispatch_arg_maps.get(tool_id)
        if arg_map is None:
            return arguments
        try:
            return arg_map(arguments)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Dispatch arg map for %r failed (%s: %s); passing raw arguments.",
                tool_id,
                type(exc).__name__,
                exc,
            )
            return arguments

    def set_ambient_resolver(
        self,
        source: str,
        resolver: Callable[[], str],
    ) -> None:
        """Register a runtime resolver for one :class:`AmbientSource`.

        The proxy calls the resolver at dispatch time to inject the current
        ambient value into outgoing arguments. Sources without a registered
        resolver fall back to a sensible default (``"default"`` for
        ``user_slug`` / ``mode_id``, ``"lisa"`` for ``app_id``).

        Called by the host after pool construction, before the first turn.
        """
        self._ambient_resolvers[source] = resolver

    def _resolve_ambient(self, source: str) -> str:
        """Resolve one ambient source to its current runtime value."""
        resolver = self._ambient_resolvers.get(source)
        if resolver is not None:
            try:
                return str(resolver() or "").strip() or self._default_ambient(source)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Ambient resolver for %r failed (%s: %s); using default.",
                    source,
                    type(exc).__name__,
                    exc,
                )
        return self._default_ambient(source)

    @staticmethod
    def _default_ambient(source: str) -> str:
        if source == "app_id":
            return "lisa"
        return "default"

    def _proxy_policy_for(self, server_id: str) -> Any:
        """Return the ProxyPolicy for a server, or None when not configured."""
        config = next((c for c in self._configs if c.server_id == server_id), None)
        if config is None:
            return None
        policy = getattr(config, "proxy_policy", None)
        if policy is None:
            return None
        # An empty policy (no ambient fields) is equivalent to no policy.
        if not getattr(policy, "ambient", None):
            return None
        return policy

    def _strip_ambient_from_schema(
        self,
        schema: dict[str, Any],
        policy: Any,
    ) -> dict[str, Any]:
        """Remove ambient fields from a tool's input schema.

        The model never sees fields the proxy will inject at dispatch time.
        Strips them from ``properties`` and ``required``. Returns a copy;
        the original schema is not mutated.
        """
        ambient_fields = policy.ambient_field_names()
        if not ambient_fields:
            return schema
        stripped = dict(schema)
        props = dict(schema.get("properties") or {})
        for field_name in ambient_fields:
            props.pop(field_name, None)
        stripped["properties"] = props
        required = schema.get("required")
        if isinstance(required, list):
            stripped["required"] = [r for r in required if r not in ambient_fields]
        return stripped

    def _inject_ambient_args(
        self,
        arguments: dict[str, Any],
        policy: Any,
        input_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Inject ambient field values into outgoing arguments.

        For tools that use structured ``filters.AND`` syntax (detected via
        ``structured_filters=True`` on the policy AND a ``filters`` parameter
        in the tool's input schema), ambient fields are injected as
        ``{"key": value}`` entries inside the AND list — idempotently
        (existing ambient entries are replaced, model-supplied filters
        preserved). For other tools, they are top-level argument overrides.
        """
        ambient = policy.ambient
        if not ambient:
            return arguments
        resolved = {
            native_field: self._resolve_ambient(
                source.value if hasattr(source, "value") else str(source)
            )
            for native_field, source in ambient.items()
        }
        use_structured = bool(policy.structured_filters) and (
            input_schema is not None and "filters" in (input_schema.get("properties") or {})
        )
        if use_structured:
            return self._merge_ambient_filters(arguments, resolved)
        merged = dict(arguments)
        merged.update(resolved)
        return merged

    @staticmethod
    def _merge_ambient_filters(
        arguments: dict[str, Any],
        resolved: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject ambient values into ``filters.AND`` idempotently.

        Existing AND entries whose key matches an ambient field are dropped
        (replaced by the resolved values); all other entries (model-supplied
        content filters, etc.) are preserved.
        """
        ambient_keys = set(resolved.keys())
        merged = dict(arguments)
        existing_and = list((arguments.get("filters") or {}).get("AND") or [])
        non_ambient = [entry for entry in existing_and if not any(k in ambient_keys for k in entry)]
        ambient_entries = [{k: v} for k, v in resolved.items()]
        merged["filters"] = {"AND": [*ambient_entries, *non_ambient]}
        return merged

    def _apply_proxy_policy_for_server(
        self,
        server_id: str,
        arguments: dict[str, Any],
        input_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply the server's proxy policy ambient injection to arguments."""
        policy = self._proxy_policy_for(server_id)
        if policy is None:
            return arguments
        return self._inject_ambient_args(arguments, policy, input_schema)

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
            # the prior _tools list if present, else a minimal default. The
            # target server id is recovered from _synthetic_targets so the
            # spec keeps its real server_id across rebuilds.
            prior = next((t for t in self._tools if t.name == name), None)
            target_server_id = self._synthetic_targets.get(name)
            server_id = target_server_id or SYNTHETIC_SERVER_ID
            spec = MCPToolSpec(
                server_id=server_id,
                name=name,
                description=prior.description if prior else f"Synthetic tool {name}",
                input_schema=prior.input_schema if prior else {"type": "object", "properties": {}},
            )
            self._tool_index[name] = (server_id, spec)
            self._tools.append(spec)

    async def __aenter__(self) -> MCPToolPool:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        """Spawn and initialize every **warm** MCP server in parallel.

        Warm servers (``config.warm=True``) are started concurrently via
        :func:`asyncio.gather`. Each server gets its own child
        :class:`contextlib.AsyncExitStack` (entered inside the server's
        own task) and is pushed onto the pool's parent stack under
        :attr:`_connect_lock` so anyio cancel-scope nesting stays correct
        while spawns run concurrently.

        A warm server with ``required=True`` whose retries are exhausted
        raises :class:`MCPConnectionError`; a non-required warm server is
        logged and skipped. Cold servers (``config.warm=False``) are **not**
        connected here — they are lazily connected on first ``list_tools``
        or ``call_tool`` that needs them. This keeps servers like ``mem0``
        (no API key) or ``lisa-guard`` (only needed in some modes) from
        spawning until first use.
        """

        if self._started:
            return

        if not self._warm_configs:
            self._started = True
            await self._refresh_tool_index()
            return

        # Start all warm servers concurrently. Each _connect_warm drives
        # its own child AsyncExitStack (see _connect) so concurrent
        # enter_async_context calls do not race on the parent stack.
        # Do NOT set _started=True until after gather — a required-server
        # failure leaves a partially-started pool that the caller must
        # aclose(). Setting _started early would make list_tools/call_tool
        # skip the start path and use a half-initialized pool.
        results = await asyncio.gather(
            *(self._connect_warm(c) for c in self._warm_configs),
            return_exceptions=True,
        )
        # Surface required-server failures after gather. Non-required
        # failures are already recorded in _warm_failed by _connect_warm.
        for config, result in zip(self._warm_configs, results):
            if isinstance(result, MCPConnectionError) and config.required:
                raise result
            if isinstance(result, BaseException) and not isinstance(result, MCPConnectionError):
                if config.required:
                    raise MCPConnectionError(
                        f"Required warm MCP server '{config.server_id}' "
                        f"raised {type(result).__name__}: {result}"
                    ) from result
                logger.warning(
                    "Warm MCP server %r raised %s: %s (non-required); proceeding without it.",
                    config.server_id,
                    type(result).__name__,
                    result,
                )
                self._warm_failed.add(config.server_id)

        # All warm connects completed (or failed). Mark started so
        # list_tools/call_tool proceed with the connected subset.
        self._started = True

        # Cold servers are NOT connected at start. They are lazily
        # connected on first list_tools/call_tool that needs them (see
        # _ensure_connected). Only the warm servers' tools are indexed
        # here; cold tools are merged into the catalog lazily.
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
                    delay = self._WARM_BACKOFF_SECONDS[
                        min(attempt, len(self._WARM_BACKOFF_SECONDS) - 1)
                    ]
                    logger.warning(
                        "Warm MCP server %r start attempt %d/%d failed: %s: %s; retrying in %.2fs",
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
        # Each server gets its OWN independent AsyncExitStack — NOT pushed
        # onto a parent. This is critical for restart correctness: when a
        # server is restarted, its old stack is closed (subprocess killed)
        # and a new stack is created. If children were pushed onto a parent,
        # the parent would accumulate dead entries and double-close them.
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        async with self._connect_lock:
            self._child_stacks[config.server_id] = stack
        self._sessions[config.server_id] = session

    async def _connect_http(self, config: MCPClientConfig) -> None:
        """Connect one streamable-HTTP MCP server.

        Uses the same independent-AsyncExitStack discipline as stdio.
        """
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        assert config.url is not None
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(config.url, headers=config.headers)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        async with self._connect_lock:
            self._child_stacks[config.server_id] = stack
        self._sessions[config.server_id] = session

    async def _disconnect_server(self, server_id: str) -> None:
        """Close and clean up one server's session + stack.

        Pops the session from ``_sessions`` and closes the server's
        ``AsyncExitStack`` from ``_child_stacks``. This kills the
        subprocess. Called by the data path (``call_tool`` exception) when
        a session is dead, and by ``restart_warm`` before reconnecting.
        """
        self._sessions.pop(server_id, None)
        async with self._connect_lock:
            stack = self._child_stacks.pop(server_id, None)
        if stack is not None:
            try:
                await stack.aclose()
            except asyncio.CancelledError:
                logger.debug(
                    "Stack close for %r cancelled during disconnect",
                    server_id,
                )
            except Exception:  # pylint: disable=broad-except
                logger.debug(
                    "Stack close for %r failed during disconnect",
                    server_id,
                    exc_info=True,
                )

    async def _refresh_tool_index(self) -> None:
        """Merge tool lists from all connected servers into one index.

        This is a READ-ONLY operation: it queries ``list_tools()`` on
        connected sessions and rebuilds the catalog. If a session is dead
        or ``list_tools()`` fails, the server's tools are simply absent
        from the catalog. This method NEVER spawns or restarts subprocesses
        — that is the job of :meth:`restart_warm` (called by the host, not
        by the data path). This prevents restart loops where a server
        starts but crashes immediately after, causing every
        ``list_tools()`` call to trigger a new subprocess spawn.
        """

        from kernel.mcp_launch.tool_filter import filter_tools

        self._tools = []
        self._tool_index = {}
        self._tool_index_by_server = {}
        self._bare_to_server_index = {}
        for config in self._configs:
            session = self._sessions.get(config.server_id)
            if session is None:
                continue
            try:
                result = await session.list_tools()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "MCP server %r list_tools failed: %s: %s; "
                    "tools absent from catalog (no restart from data path)",
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
            # Per-server index keeps EVERY discovered tool, collisions or
            # not.  Server-scoped dispatch (_call_tool_scoped) resolves
            # alias targets through this index, so it must not be filtered
            # by tool_policy — denying a tool from the model catalog must
            # not break internal alias forwarding (e.g. memory.list →
            # list_memories when list_memories is denied to the model).
            server_index: dict[str, MCPToolSpec] = {}
            for spec in server_tools:
                server_index[spec.name] = spec
            self._tool_index_by_server[config.server_id] = server_index
            # Bare-name index: merge this server's bare names into the
            # pool-wide bare-to-server map. First-registered-wins mirrors
            # the merged _tool_index collision policy. Bare-name collisions
            # across connected servers are logged below via the namespaced
            # index warning; here the bare index just keeps the first owner
            # so bare-name resolution is deterministic.
            for bare_name in server_index:
                if bare_name not in self._bare_to_server_index:
                    self._bare_to_server_index[bare_name] = config.server_id

            # Model-facing catalog: apply tool_policy (hide tools the model
            # should not call directly) then proxy_policy (strip ambient
            # fields from schemas so the model never sees user_id /
            # agent_id / app_id as settable parameters).
            catalog_tools = server_tools
            if config.tool_policy is not None:
                catalog_tools = filter_tools(catalog_tools, config.tool_policy)
            # Proxy policy: strip ambient fields from the tool schema so the
            # model never sees parameters it cannot control (user_id,
            # agent_id, app_id, …). The proxy injects them at dispatch time.
            proxy_policy = getattr(config, "proxy_policy", None)
            if proxy_policy is not None and getattr(proxy_policy, "ambient", None):
                catalog_tools = [
                    MCPToolSpec(
                        server_id=spec.server_id,
                        name=spec.name,
                        description=spec.description,
                        input_schema=self._strip_ambient_from_schema(
                            spec.input_schema, proxy_policy
                        ),
                    )
                    for spec in catalog_tools
                ]
            for spec in catalog_tools:
                # Merged index is keyed by the namespaced name
                # (``server_id.tool_name``).  This eliminates cross-server
                # collisions by construction — two servers exposing the
                # same bare name (e.g. lisa-memory and mem0 both serving
                # add_memory) land under different namespaces and both
                # appear in the catalog.
                namespaced = namespaced_tool_name(spec.server_id, spec.name)
                namespaced_spec = MCPToolSpec(
                    server_id=spec.server_id,
                    name=namespaced,
                    description=spec.description,
                    input_schema=spec.input_schema,
                )
                if namespaced in self._tool_index:
                    # Same server listing the same tool name twice — a
                    # genuine server-side duplicate, not a cross-server
                    # collision.  First-registered-wins with a warning.
                    logger.warning(
                        "MCP tool name collision within namespace %r: "
                        "server %r lists duplicate tool %r; first entry wins.",
                        namespaced,
                        spec.server_id,
                        spec.name,
                    )
                    continue
                self._tool_index[namespaced] = (config.server_id, namespaced_spec)
                self._tools.append(namespaced_spec)

        # Re-seed host-owned synthetic tools after the MCP index is rebuilt.
        # Synthetic tools are not rediscovered from servers — they are owned
        # by the host and must survive catalog refreshes.  They keep their
        # bare (unnamespaced) names — ``memory.add``, ``memory.search``, etc.
        self._reindex_synthetic_tools()

    async def refresh_tool_index(self) -> None:
        """Rebuild the merged tool catalog from connected sessions.

        Public entry point for hosts that change a server's tool surface
        at runtime (e.g. after installing/uninstalling packs that affect a
        cold server whose tool surface may change). Re-queries every connected
        session and rebuilds the index.
        """
        await self._refresh_tool_index()

    async def _ensure_connected(self, config: MCPClientConfig) -> bool:
        """Ensure a server has a live session, connecting or restarting if needed.

        Returns True when the server has a live session after the call
        (was already connected, or was just connected/restarted). Returns
        False when the server could not be started (logged and skipped).

        Warm servers: if the session is dead, :meth:`restart_warm` is
        called (which enforces the restart budget). If the server is
        permanently failed (budget exhausted), returns False immediately.
        Cold servers: connected on demand via :meth:`_connect` (no budget
        — cold servers are spawned and torn down per use).
        """
        if config.server_id in self._sessions:
            return True
        if config.warm:
            if config.server_id in self._permanently_failed:
                logger.debug(
                    "Warm MCP server %r is permanently failed; not restarting",
                    config.server_id,
                )
                return False
            await self._maybe_restart_warm(config)
            return config.server_id in self._sessions
        # Cold server — connect on demand, best-effort.
        try:
            await self._connect(config)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Cold MCP server %r failed to start on demand: %s: %s",
                config.server_id,
                type(exc).__name__,
                exc,
            )
            return False
        # New session: refresh so this server's tools are indexed.
        await self._refresh_tool_index()
        return config.server_id in self._sessions

    async def _ensure_all_cold_connected(self, allowed_server_ids: set[str] | None = None) -> None:
        """Best-effort connect every cold server that is not yet connected.

        Used by ``list_tools`` so the initial tool catalog reflects all
        servers' tools. Cold servers that fail to start are recorded in
        ``_cold_failed`` and skipped on subsequent calls — this prevents
        an unbounded re-spawn loop where every ``list_tools()`` call
        retries failed cold servers.

        When ``allowed_server_ids`` is provided, only servers in that set
        are connected (mode-aware cold activation).
        """
        for config in self._cold_configs:
            if config.server_id in self._sessions:
                continue
            if config.server_id in self._cold_failed:
                continue
            if allowed_server_ids is not None and config.server_id not in allowed_server_ids:
                continue
            try:
                await self._connect(config)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Cold MCP server %r failed to start: %s: %s",
                    config.server_id,
                    type(exc).__name__,
                    exc,
                )
                self._cold_failed.add(config.server_id)
        await self._refresh_tool_index()

    async def _resolve_server_for_bare_name(self, tool_id: str) -> str | None:
        """Find which server exposes a bare tool name.

        Uses the precomputed ``_bare_to_server_index`` (rebuilt in
        ``_refresh_tool_index``) for an O(1) lookup over connected servers.
        Only when the bare name is absent from that index are cold servers
        lazily connected (best-effort) and the index rebuilt so their tools
        become resolvable — this keeps the cold-connect side effect out of
        the hot path for any name already resolvable from warm/connected
        servers. Returns the owning ``server_id``, or ``None`` when no
        server exposes the tool. Used by ``call_tool`` as a bare-name
        fallback for internal callers (operations) that pass the bare MCP
        tool name; model-facing calls always use namespaced names.
        """
        # Hot path: O(1) dict lookup over already-connected servers.
        cached = self._bare_to_server_index.get(tool_id)
        if cached is not None:
            return cached
        # Cold path: the bare name is not in the precomputed index, which
        # means no connected server exposes it. Lazily connect cold servers
        # that have not been tried yet, re-indexing after each so their bare
        # names enter _bare_to_server_index, and re-check the index.
        for config in self._cold_configs:
            if config.server_id in self._sessions:
                continue
            if config.server_id in self._cold_failed:
                continue
            try:
                await self._connect(config)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Cold MCP server %r failed to start during bare-name resolution: %s: %s",
                    config.server_id,
                    type(exc).__name__,
                    exc,
                )
                self._cold_failed.add(config.server_id)
                continue
            await self._refresh_tool_index()
            cached = self._bare_to_server_index.get(tool_id)
            if cached is not None:
                return cached
        return None

    async def restart_warm(self, server_id: str) -> bool:
        """Re-spawn one warm server after a mid-session crash (supervised).

        Returns True if the server is (re)connected after the attempt,
        False if it remains down or is permanently failed.

        Supervised lifecycle — restart budget (OTP MaxRestarts/MaxSeconds):
        at most :attr:`_MAX_RESTARTS` restarts within
        :attr:`_RESTART_WINDOW_S` seconds. If exceeded, the server is
        permanently FAILED and this method returns False without
        spawning. This prevents infinite restart loops when a warm server
        starts (stdio + initialize succeed) but crashes immediately after.

        Thread-safety: uses ``_restarting`` to guard against concurrent
        restart calls. If a restart is already in flight for this server,
        returns the current connection state without spawning a duplicate.
        """
        config = next((c for c in self._warm_configs if c.server_id == server_id), None)
        if config is None:
            logger.warning("restart_warm: %r is not a warm server in this pool", server_id)
            return False

        # Permanently failed — budget exhausted. No more spawns.
        if server_id in self._permanently_failed:
            logger.warning(
                "restart_warm: %r is permanently failed (restart budget exhausted); not spawning",
                server_id,
            )
            return False

        # Guard against concurrent restart calls.
        if server_id in self._restarting:
            logger.debug("restart_warm: %r already being restarted; skip", server_id)
            return server_id in self._sessions

        # Enforce restart budget: count restarts within the window.
        now = time.monotonic()
        timestamps = self._restart_timestamps.get(server_id, [])
        # Drop timestamps outside the window.
        timestamps = [t for t in timestamps if now - t < self._RESTART_WINDOW_S]
        if len(timestamps) >= self._MAX_RESTARTS:
            logger.error(
                "Warm MCP server %r exceeded restart budget (%d restarts in %.0fs); "
                "marking permanently failed. No more spawns will be attempted.",
                server_id,
                len(timestamps),
                self._RESTART_WINDOW_S,
            )
            self._permanently_failed.add(server_id)
            self._warm_failed.add(server_id)
            return False

        self._restarting.add(server_id)
        timestamps.append(now)
        self._restart_timestamps[server_id] = timestamps
        try:
            # Close the old session + child stack (kills old subprocess)
            # before attempting a new connection.
            await self._disconnect_server(server_id)
            # SINGLE attempt — no retry loop. The _connect_warm retry loop
            # is for initial startup only. Using it here would spawn up to
            # 3 subprocesses per restart_warm call, causing a child explosion
            # when the server starts but crashes immediately after.
            try:
                await self._connect(config)
                self._warm_failed.discard(server_id)
            except Exception as exc:
                logger.warning(
                    "Warm MCP server %r restart failed: %s: %s",
                    server_id,
                    type(exc).__name__,
                    exc,
                )
                self._warm_failed.add(server_id)
                if config.required:
                    raise MCPConnectionError(
                        f"Required warm MCP server '{server_id}' restart failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            return server_id in self._sessions
        finally:
            self._restarting.discard(server_id)

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

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[MCPToolSpec]:
        """Return the merged tool catalog from all connected servers.

        Cold servers that have not yet been connected are lazily started
        here so the catalog reflects their tool surface. Cold servers that
        fail to start are skipped — their tools are absent from the catalog.

        When ``allowed_server_ids`` is provided, only cold servers in that
        set are connected. This lets the turn runner avoid spawning cold
        servers whose tools are not allowed for the active mode.
        """

        if not self._started:
            await self.start()
        # Lazily connect cold servers that have not been connected yet so
        # the catalog reflects every server's tools. Warm servers are already
        # up (or in _warm_failed); they are not touched here.
        cold_needs_connect = [
            c
            for c in self._cold_configs
            if c.server_id not in self._sessions and c.server_id not in self._cold_failed
        ]
        if allowed_server_ids is not None:
            cold_needs_connect = [
                c for c in cold_needs_connect if c.server_id in allowed_server_ids
            ]
        if cold_needs_connect:
            await self._ensure_all_cold_connected(allowed_server_ids)
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
            # Bare-name fallback: internal callers (e.g. operations calling
            # ``execute_tool_fn`` with a bare MCP tool name like
            # ``read_model_manifest``) won't find the tool in the merged
            # index because it is keyed by namespaced names
            # (``lisa-storage.read_model_manifest``). Resolve via the
            # precomputed ``_bare_to_server_index`` (O(1) over connected
            # servers; cold servers lazily connected only when the bare name
            # is absent). Model-facing calls always use namespaced names,
            # so this fallback never shadows the namespaced path.
            resolved_server_id = await self._resolve_server_for_bare_name(tool_id)
            if resolved_server_id is not None:
                return await self._call_tool_scoped(tool_id, arguments, resolved_server_id)
            return MCPToolResult(
                tool_id=tool_id,
                output_text=f"Tool '{tool_id}' is not registered on any MCP server.",
                is_error=True,
            )
        server_id, _spec = entry
        config = next((c for c in self._configs if c.server_id == server_id), None)
        session = self._sessions.get(server_id)
        if session is None:
            # Server is down. Warm servers get one restart attempt; cold
            # servers are lazily connected on demand.
            if config is not None:
                await self._ensure_connected(config)
                session = self._sessions.get(server_id)
            if session is None:
                return MCPToolResult(
                    tool_id=tool_id,
                    output_text=f"MCP server '{server_id}' for tool '{tool_id}' is not connected.",
                    is_error=True,
                )
        # The merged index is keyed by namespaced names (server_id.tool),
        # but the MCP session expects the bare tool name.  Strip the
        # namespace prefix before calling the server.  Use the resolved
        # server_id (not a naive first-dot split) so pack-contributed
        # servers with dotted ids (e.g. ``lisa_packs.cloudflare.cloudflare-api``)
        # dispatch to the correct native tool name (e.g. ``search``, not
        # ``cloudflare-api.search``).
        bare_name = tool_id
        if server_id and tool_id.startswith(server_id + "."):
            bare_name = tool_id[len(server_id) + 1 :]
        elif "." in tool_id:
            bare_name = bare_tool_name(tool_id)
        mapped = self._apply_dispatch_arg_map(tool_id, arguments)
        # Proxy policy: inject ambient scope (user_id, agent_id, app_id, …)
        # from the host runtime after the dispatch arg map. This closes the
        # privacy boundary for direct native-tool calls — the model cannot
        # escape because the schema already hides these fields (stripped in
        # _refresh_tool_index), and the values are overridden here even if
        # the model somehow sends them.
        server_spec = self._tool_index_by_server.get(server_id, {}).get(bare_name)
        input_schema = server_spec.input_schema if server_spec else None
        mapped = self._apply_proxy_policy_for_server(server_id, mapped, input_schema)
        try:
            result = await session.call_tool(bare_name, arguments=mapped)
        except Exception as exc:  # pylint: disable=broad-except
            # Session died mid-call. Close the child stack to kill the
            # subprocess immediately — do NOT restart from the data path.
            await self._disconnect_server(server_id)
            if config is not None and config.warm:
                self._warm_failed.add(server_id)
            return MCPToolResult(
                tool_id=tool_id,
                output_text=(
                    f"MCP tool '{tool_id}' failed: {type(exc).__name__}: {exc}. "
                    f"Server '{server_id}' session dropped; "
                    f"restart may be needed."
                ),
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
            connected = await self._ensure_connected(config)
            if not connected:
                return MCPToolResult(
                    tool_id=tool_id,
                    output_text=(
                        f"MCP server '{server_id}' for tool '{tool_id}' is not connected."
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
                output_text=(f"Tool '{tool_id}' is not registered on server '{server_id}'."),
                is_error=True,
            )
        namespaced = namespaced_tool_name(server_id, tool_id)
        mapped = self._apply_dispatch_arg_map(namespaced, arguments)
        # Proxy policy: inject ambient scope from the host runtime. Same
        # enforcement as the non-scoped path — the alias already translated
        # generic→native, and this injects the ambient boundary values
        # idempotently (existing ambient entries in filters.AND are replaced).
        scoped_spec = server_index.get(tool_id)
        scoped_schema = scoped_spec.input_schema if scoped_spec else None
        mapped = self._apply_proxy_policy_for_server(server_id, mapped, scoped_schema)
        try:
            result = await session.call_tool(tool_id, arguments=mapped)
        except Exception as exc:  # pylint: disable=broad-except
            # Session died mid-call. Close the child stack to kill the
            # subprocess immediately — do NOT restart from the data path.
            await self._disconnect_server(server_id)
            if config.warm:
                self._warm_failed.add(server_id)
            return MCPToolResult(
                tool_id=tool_id,
                output_text=(
                    f"MCP tool '{tool_id}' failed: {type(exc).__name__}: {exc}. "
                    f"Server '{server_id}' session dropped; "
                    f"restart may be needed."
                ),
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
        """Close every session and terminate every subprocess.

        Iterates ``_child_stacks`` and closes each one independently.
        Each server owns its own AsyncExitStack — there is no parent stack
        to close. This avoids the double-close bug that child-stacks-on-
        parent caused when servers were restarted mid-session.
        """
        self._sessions.clear()
        async with self._connect_lock:
            stacks = list(self._child_stacks.values())
            self._child_stacks.clear()
        for stack in stacks:
            try:
                await stack.aclose()
            except asyncio.CancelledError:
                # CancelledError is BaseException in 3.9+, so the broad
                # except below misses it. Suppress so shutdown never
                # crashes when the event loop cancels subprocess waits.
                logger.debug("Server stack close cancelled during aclose")
            except Exception:  # pylint: disable=broad-except
                logger.debug("Server stack close failed during aclose", exc_info=True)
        self._restarting.clear()
        self._restart_timestamps.clear()
        self._permanently_failed.clear()
        self._cold_failed.clear()
        self._tool_index.clear()
        self._tool_index_by_server.clear()
        self._tools = []
        self._synthetic_handlers.clear()
        self._synthetic_targets.clear()
        self._started = False
