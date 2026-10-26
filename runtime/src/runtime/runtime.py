"""HostRuntime — the typed composition root that replaces ``_RUNTIME_STATE``.

This is the mutable, stateful counterpart to the frozen ``HostConfig``.
It is deliberately a **thin composition root**: it owns ``config``,
``session_plan``, and ``execution_policy`` directly, and groups every other
responsibility into one of five focused sub-objects so each call site names
the domain it touches:

    runtime.paths.thread            # HostPaths — derived filesystem layout
    runtime.providers.openai_client # ProviderRegistry — lazy SDK clients
    runtime.resources.mcp_tool_pool # RuntimeResources — MCP pool lifecycle
    runtime.session.current_mode_id # SessionState — routing/diagnostics state
    runtime.services.approval_service # HostServices — durable service handles

Async lifecycle::

        runtime = HostRuntime(config)
        await runtime.start()          # starts MCP pool in the host's loop
        result = await handle_message(runtime, "/search test", transport)
        await runtime.aclose()         # closes MCP pool in the same loop

The two sub-object classes are defined in this module (not separate files)
because ``RuntimeResources`` implements the runtime's async MCP lifecycle —
keeping it beside the object that enters and exits its event loop makes the
task/cancel-scope ownership rules easy to read end to end.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.session_plan import SessionPlan

from runtime.config import HostConfig
from runtime.execution_policy import resolve_execution_policy
from runtime.mcp_binary_cache import resolve_mcp_binary_configs
from runtime.operator_signal import OperatorSignal
from runtime.providers import ProviderRegistry
from runtime.runtime_paths import HostPaths
from runtime.services import HostServices
from runtime.session_context import compile_selected_session_plan
from runtime.thread import latest_turn_owner_mode

logger = logging.getLogger(__name__)

#: Upper bound on per-dispatch-token dedupe sets retained for scoped
#: instruction injection. One token exists per turn and per fork/delegate
#: child, and only the newest are still running, so a small bound is ample.
_MAX_TRACKED_TURNS = 8

#: Tool argument keys whose value names a path directly (``str``), and keys
#: whose value is a list of records each carrying a ``path`` (the coding pack's
#: batched ``coding_edit_files``). Every coding-pack file tool names its
#: argument ``path``; ``file_path`` covers the customization reader. Directory
#: values (``find_files``, ``list_directory``, ``search_code``) match no
#: ``applyTo`` glob, so they contribute nothing.
_PATH_ARGUMENT_KEYS = ("path", "file_path")
_PATH_RECORD_LIST_KEYS = ("edits", "files")


def _tool_call_paths(arguments: Any) -> tuple[str, ...]:
    """Return the distinct paths a tool call refers to, in argument order.

    Covers the single-path tools (``path`` / ``file_path``) and the batched
    shape whose paths live inside a list of records. Non-string, empty, and
    unparseable values contribute nothing — a call that names no identifiable
    file gets no scoped guidance.
    """
    if not isinstance(arguments, dict):
        return ()
    paths: list[str] = []
    for key in _PATH_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    for key in _PATH_RECORD_LIST_KEYS:
        records = arguments.get(key)
        if not isinstance(records, (list, tuple)):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            value = record.get("path")
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
    # dict.fromkeys keeps first-seen order while removing repeats.
    return tuple(dict.fromkeys(paths))


def _workspace_relative_posix_path(file_path: str) -> str:
    """Normalize a tool-argument path for the ``applyTo`` matcher.

    Returns a POSIX-style path with no leading ``./``, or ``""`` when the value
    names no path (empty, ``"."``) or escapes the workspace root via ``..``.
    A leading slash is stripped rather than rejected: reading an absolute path
    workspace-relative can only miss a directory-anchored glob, never attach a
    body for an unrelated file.
    """
    normalized = file_path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if not normalized:
        return ""
    parts: list[str] = []
    for part in normalized.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            # Escapes the workspace root; nothing scoped can match.
            return ""
        parts.append(part)
    return "/".join(parts)


def _scoped_instruction_block(instruction: Any) -> str:
    """Render one ``applyTo``-scoped instruction as a labelled Markdown block.

    The ``applies to`` label is kept so the model can see why the guidance
    arrived with a particular tool result.
    """
    apply_label = ", ".join(instruction.apply_to)
    return (
        f"## Instruction: {instruction.name} (applies to: {apply_label})"
        f"\n\n{instruction.body.strip()}"
    )


def _unconditional_customization_blocks(bundle: Any) -> list[str]:
    """Render the customization blocks that belong in every system prompt.

    Carries always-on files, instructions that declare no ``applyTo``, and the
    skill *index*. Instructions that declare one are deferred to the dispatch
    that first touches a matching path, and skill *bodies* are loaded on demand
    (see :func:`_skill_index_block`).
    """
    parts = [
        f"## Project Instructions\n\n{f.body.strip()}"
        for f in bundle.always_on
        if f.body.strip()
    ]
    parts.extend(
        _scoped_instruction_block(inst)
        for inst in bundle.instructions
        if inst.body.strip() and not inst.apply_to
    )
    index = _skill_index_block(bundle.skills)
    if index:
        parts.append(index)
    return parts


def _skill_index_block(skills: Any) -> str:
    """Render the skill index: name, description, and path — never the body.

    Mirrors Copilot's progressive loading. The index is the discovery surface
    (~100 tokens per skill), the body loads when a turn decides it needs the
    procedure, and a skill's own reference files load only if the body names
    them. Injecting bodies eagerly costs the full corpus on every request no
    matter what the turn does.

    Skill bodies have no ``applyTo``, so unlike instructions they cannot be
    matched to a path; the model must choose. Each entry therefore names the
    file to read, and the read goes through the ordinary file-read tool in
    source scope — no dedicated loader is required.

    ``disable_model_invocation`` is honored as an opt-out of model-driven
    loading: such a skill is neither listed nor injected. Copilot keeps it
    reachable as a slash command; Lisa exposes no skill-slash surface, so the
    alternative would be listing guidance the model is forbidden to load.
    """
    usable = [
        skill
        for skill in skills
        if skill.body.strip() and not skill.disable_model_invocation
    ]
    if not usable:
        return ""
    lines = [
        "## Skills",
        "",
        "Each skill below is a procedure you can load on demand. Read the named",
        "file when your task matches its description; do not guess at its",
        "contents. Skills you do not load cost nothing.",
        "",
    ]
    for skill in usable:
        description = " ".join(skill.description.split())
        lines.append(f"- **{skill.name}** — {description}")
        lines.append(f"  Load with the file-read tool: `{skill.path}`")
    return "\n".join(lines)


def _make_in_process_resolver():
    """Build an ``InProcessResolver`` backed by ``mcp_server.registry``.

    Imported lazily so the kernel never imports from ``mcp_server`` at
    module load time. The host (which depends on both packages) bridges
    the registry's entry type to the kernel's
    :class:`~kernel.tools.mcp_client.InProcessServerEntry`.
    """
    from kernel.tools.mcp_client import InProcessServerEntry
    from mcp_server.registry import get_in_process_server

    def resolver(server_id: str) -> InProcessServerEntry | None:
        entry = get_in_process_server(server_id)
        if entry is None:
            return None
        return InProcessServerEntry(
            server_id=entry.server_id,
            mcp=entry.mcp,
            init=entry.init,
        )

    return resolver


@dataclass(slots=True)
class TokenCounters:
    """Per-turn and per-session token counters."""

    # Per-turn (reset at the start of each turn).
    turn_prompt_tokens: int = 0
    turn_completion_tokens: int = 0
    turn_total_tokens: int = 0
    turn_cached_tokens: int = 0
    turn_reasoning_tokens: int = 0
    turn_cache_creation_tokens: int = 0
    turn_duration_ms: float = 0.0
    turn_latency_ms: int = 0

    # Per-session (accumulate across turns).
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_total_tokens: int = 0
    session_cached_tokens: int = 0
    session_reasoning_tokens: int = 0
    session_cache_creation_tokens: int = 0
    session_duration_ms: float = 0.0
    session_latency_ms: int = 0
    session_token_ledger: list[dict[str, Any]] = field(default_factory=list)
    model_token_totals: dict[str, dict[str, int]] = field(default_factory=dict)
    mode_token_totals: dict[str, dict[str, int]] = field(default_factory=dict)
    stage_token_totals: dict[str, dict[str, int]] = field(default_factory=dict)

    # Session metadata.
    session_started_utc: datetime | None = None
    session_turn_count: int = 0
    session_success_count: int = 0
    session_error_count: int = 0
    session_cancelled_count: int = 0
    session_total_latency_ms: float = 0.0
    session_mode_counts: dict[str, int] = field(default_factory=dict)

    def reset_turn(self) -> None:
        """Reset per-turn counters at the start of a new turn."""
        self.turn_prompt_tokens = 0
        self.turn_completion_tokens = 0
        self.turn_total_tokens = 0
        self.turn_cached_tokens = 0
        self.turn_reasoning_tokens = 0
        self.turn_cache_creation_tokens = 0
        self.turn_duration_ms = 0.0
        self.turn_latency_ms = 0


@dataclass(slots=True)
class TurnContext:
    """Mutable per-turn diagnostic context.

    Passed through ``handle_message`` and updated by ``on_turn_complete``.
    Transport hosts read this via ``runtime.session.last_turn_context`` for
    ``/debug context`` rendering.
    """

    timestamp_utc: str = ""
    raw_message: str = ""
    mode_id: str = "default"
    mode_kind: str = ""
    query: str = ""
    preferred_model_id: str = ""
    resolved_model: dict[str, Any] = field(default_factory=dict)
    handoff: str = ""
    turn_input: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None


# ── SessionState ────────────────────────────────────────────────────────


class SessionState:
    """Mutable routing, telemetry, and diagnostics state for one session.

    Everything here changes over the session's life: the active route mode,
    the current workflow stage, the operator signal, the last slice/context
    snapshot, token counters, the persisted state snapshot, and the restart
    handshake. ``HostRuntime.session`` holds the single instance.
    """

    def __init__(self) -> None:
        # ── Route state ────────────────────────────────────────────────
        self.current_mode_id: str = ""
        self.current_stage_id: str = "-"
        self.next_model_id: str = "-"
        self.next_model_provider: str = "-"

        # ── Operator signal (pause/stop/terminate/interrupt/eof/suspend) ──
        # The transport sets this when the operator presses the corresponding
        # key binding. The turn loop polls it via ``TurnHooks.operator_signal``
        # once per iteration. The host clears it at the start of each turn.
        self.operator_signal_value: OperatorSignal = OperatorSignal.NONE

        # ── Token counters ─────────────────────────────────────────────
        self.tokens = TokenCounters()
        if self.tokens.session_started_utc is None:
            self.tokens.session_started_utc = datetime.now(UTC)

        # ── Manifest cache ──────────────────────────────────────────────
        # Mutable: setup_framework seeds it, /models reload swaps it.
        self.manifest: Any = None

        # ── State snapshot ─────────────────────────────────────────────
        self.state_snapshot: dict[str, Any] | None = None

        # ── Slice ─────────────────────────────────────────────────────
        self.last_slice_finished_at: float | None = None
        # Latest ContextBudgetSnapshot produced by the context-mapper pass
        # on this turn. Read by utilization-aware slice providers via
        # ``TurnHooks.last_context_snapshot``.
        self.last_context_snapshot: Any = None

        # ── Last-turn diagnostics ─────────────────────────────────────
        self.last_turn_context: TurnContext | None = None

        # ── Restart state ─────────────────────────────────────────────
        self.restart_required: bool = False
        self.restart_reason: str = ""

        # Per-turn count of policy auto-approvals; the approval flow reads
        # and increments it, and the dispatcher resets it at each turn start.
        self.auto_approved_this_turn: int = 0

        # Pre-turn snapshot of durable approval state for the recorded
        # blockers named in the thread tail (resolved in the async turn
        # path; consumed by the synchronous thread digest). ``None`` until
        # the first turn resolves it.
        self.live_workflow_state: Any | None = None

    def set_last_turn_context(self, ctx: TurnContext) -> None:
        self.last_turn_context = ctx

    def request_restart(self, reason: str) -> None:
        self.restart_required = True
        self.restart_reason = reason


# ── RuntimeResources — the MCP pool async lifecycle ─────────────────────


class RuntimeResources:
    """Owns MCP and provider-proxy resources and their async lifecycle.

    This is the one async resource the runtime manages. The lifecycle is
    subtle (anyio cancel-scope ownership, deferred start, cancellation-safe
    teardown) — see ``runtime.instructions.md`` and ``debugging.md``. All of it
    lives here so the mechanics are co-located and the orchestration reads
    top to bottom. ``HostRuntime.resources`` holds the single instance.

    The methods reference the owning ``HostRuntime`` (passed at construction)
    for the in-process MCP resolver choice and post-start ambient resolver
    registration that needs current runtime identity.
    """

    def __init__(self, runtime: HostRuntime) -> None:
        self._runtime = runtime
        self._in_process_mcp: bool = runtime._in_process_mcp
        self._mcp_tool_pool: Any = None
        self._mcp_warm_configs: list[Any] = []
        self._mcp_cold_configs: list[Any] = []
        self._proxy_configs: list[Any] = []
        self._proxy_pool: Any = None
        self._proxy_routes: dict[str, Any] = {}
        self._mcp_started: bool = False
        self._mcp_start_task: asyncio.Task[None] | None = None
        """Background task started by start_pool_deferred(); awaited by
        ensure_pool_ready() before any pool read. When None, the pool
        was either started synchronously (start()) or never started."""
        self._mcp_start_exc: BaseException | None = None
        """Cached exception from a failed deferred start, consumed once by
        ensure_pool_ready() so it doesn't re-raise on every subsequent call."""
        self._mcp_started_event: asyncio.Event | None = None
        """Set by the deferred-start task once the pool is constructed, so
        ensure_pool_ready() can return without waiting for the (long-lived)
        start task to finish. The deferred task stays alive after this point
        so it can tear down the pool in the same task that entered the MCP
        anyio cancel scopes (anyio forbids exiting a cancel scope from a
        different task than the one that entered it — see debugging.md)."""
        self._mcp_close_signal: asyncio.Event | None = None
        """Set by aclose() to tell the deferred-start task to tear down the
        pool. The task then closes the pool's child AsyncExitStacks in its
        own task, where the anyio cancel scopes were entered."""

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def mcp_tool_pool(self) -> Any:
        return self._mcp_tool_pool

    @property
    def proxy_pool(self) -> Any:
        """The live provider-proxy pool (``None`` before start / after close)."""
        return self._proxy_pool

    @property
    def proxy_routes(self) -> dict[str, Any]:
        """Ready provider routes keyed by provider id (copy)."""
        return dict(self._proxy_routes)

    @property
    def mcp_warm_configs(self) -> list[Any]:
        return self._mcp_warm_configs

    @property
    def mcp_cold_configs(self) -> list[Any]:
        return self._mcp_cold_configs

    @property
    def mcp_started(self) -> bool:
        return self._mcp_started

    def set_mcp_configs(self, warm: list[Any], cold: list[Any]) -> None:
        """Set compiled MCP server configs (sync — no pool creation)."""
        self._mcp_warm_configs = warm
        self._mcp_cold_configs = cold

    def set_proxy_configs(self, configs: list[Any]) -> None:
        """Set compiled provider proxy configs (sync — no process creation)."""
        self._proxy_configs = configs

    def proxy_route_for(self, provider: str, protocol: str) -> Any | None:
        """Return the ready compatible proxy route for a provider gateway.

        Liveness-gated: when the route's owning proxy process exited
        mid-session, the stale route is dropped and ``None`` is returned so
        the provider gateway falls back to the direct endpoint. A dead
        proxy must degrade to "no compression savings", never hang or fail
        a model call (the 2026-10-26 headroom death: the cached Ollama
        client kept pointing at the dead loopback port for the rest of the
        process because nothing re-checked route liveness). A bounded
        in-loop respawn is attempted opportunistically by
        :meth:`ensure_proxy_route` (the async path); this sync accessor is
        safe to call from the gateway client factories.
        """
        route = self._proxy_routes.get(provider.strip().lower())
        if route is None or route.protocol != protocol:
            return None
        pool = self._proxy_pool
        if pool is not None and not pool.route_alive(provider):
            pool.drop_route(provider)
            self._proxy_routes = pool.routes
            return None
        return route

    async def ensure_proxy_route(self, provider: str, protocol: str) -> Any | None:
        """Return a live compatible proxy route, respawning a dead proxy once.

        The async counterpart of :meth:`proxy_route_for`, for callers that
        can await (REPL transport loops between turns). A dead host-owned
        proxy gets one bounded restart attempt; ``None`` when the proxy
        stays down so the caller falls back to the direct endpoint.
        """
        route = self.proxy_route_for(provider, protocol)
        if route is not None:
            return route
        pool = self._proxy_pool
        if pool is None:
            return None
        if await pool.respawn(provider):
            self._proxy_routes = pool.routes
        return self._proxy_routes.get(provider.strip().lower())

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the MCP tool pool synchronously inside the host's loop.

        Awaits the full warm-server spawn before returning. Transport hosts
        that need the prompt up immediately should call
        :meth:`start_pool_deferred` instead and :meth:`ensure_pool_ready`
        before the first turn.
        """
        await self._start_pool_impl(deferred=False)

    def start_pool_deferred(self) -> asyncio.Task[None]:
        """Start the MCP pool as a background task; return immediately.

        Schedules :meth:`_start_pool_impl` as an :class:`asyncio.Task` so
        the host can render the prompt while warm servers spawn. Callers
        that read :attr:`mcp_tool_pool` must ``await``
        :meth:`ensure_pool_ready` first. Idempotent: if a deferred start is
        already in flight, returns the existing task.
        """
        if self._mcp_start_task is not None and not self._mcp_start_task.done():
            return self._mcp_start_task
        if self._mcp_started and self._mcp_start_task is None:
            # Already started synchronously; return a completed no-op task.
            loop = asyncio.get_running_loop()
            self._mcp_start_task = loop.create_task(asyncio.sleep(0))
            return self._mcp_start_task
        # The deferred task is long-lived: it builds the pool, signals
        # _mcp_started_event, then blocks on _mcp_close_signal so it can
        # tear the pool down in the SAME task that entered the MCP anyio
        # cancel scopes. aclose() sets the close signal and awaits the task.
        # See debugging.md (anyio cross-loop/task rule).
        self._mcp_started_event = asyncio.Event()
        self._mcp_close_signal = asyncio.Event()
        self._mcp_start_task = asyncio.create_task(self._start_pool_impl(deferred=True))
        return self._mcp_start_task

    async def ensure_pool_ready(self) -> None:
        """Await the deferred pool start if one is in flight.

        Call this before reading :attr:`mcp_tool_pool` when the host used
        :meth:`start_pool_deferred`. If the pool was started synchronously
        via :meth:`start`, this is a no-op. If no start was ever scheduled,
        runs :meth:`start` inline (back-compat for hosts that never deferred).

        If the deferred start failed, the exception is raised once and
        cached — subsequent calls re-raise the cached exception so the
        caller fails loudly but doesn't re-await a completed task.
        """
        if self._mcp_start_exc is not None:
            raise self._mcp_start_exc
        if self._mcp_started_event is not None:
            # Deferred start: wait for the pool to be constructed (not for
            # the task to finish — the task is long-lived and only ends at
            # teardown). Cache the exception once if the build phase failed.
            try:
                await self._mcp_started_event.wait()
            except BaseException as exc:  # pylint: disable=broad-except
                self._mcp_start_exc = exc
                raise
            if self._mcp_start_exc is not None:
                raise self._mcp_start_exc
            # If the build phase failed, the task ended with an exception
            # (the finally still set the event). Cache + re-raise once.
            if (
                self._mcp_start_task is not None
                and self._mcp_start_task.done()
                and not self._mcp_started
            ):
                exc = self._mcp_start_task.exception()
                if exc is not None:
                    self._mcp_start_exc = exc
                    raise exc
            return
        if self._mcp_start_task is not None:
            # Sync-start fallback path: the task is a completed no-op or
            # an already-finished start task. Awaiting it is safe.
            try:
                await self._mcp_start_task
            except BaseException as exc:
                self._mcp_start_exc = exc
                raise
            return
        if not self._mcp_started:
            await self.start()

    async def _start_pool_impl(self, *, deferred: bool) -> None:
        """Shared pool-start implementation for start() and deferred start.

        For the sync path (``deferred=False``), this builds the pool and
        returns; ``aclose()`` later closes it in the caller's task (the
        same task, so the anyio cross-task rule is satisfied).

        For the deferred path (``deferred=True``), this builds the pool,
        signals ``_mcp_started_event`` so ``ensure_pool_ready`` can return,
        then stays alive until ``aclose`` sets ``_mcp_close_signal``. The
        teardown runs in THIS task because the MCP ``stdio_client`` contexts
        entered during ``pool.start()`` bind anyio cancel scopes to this
        task; anyio forbids exiting them from a different task (see
        debugging.md). Signaling the started event is wrapped in try/finally
        so a build-phase failure or early return never leaves
        ``ensure_pool_ready`` blocked on the event.
        """
        try:
            await self._build_proxy_pool()
            await self._build_pool()
        except BaseException:
            await self._close_proxy_pool()
            raise
        finally:
            # On the deferred path the started event MUST be set whether or
            # not the build succeeded — otherwise ensure_pool_ready blocks
            # forever. The cached exception is recorded by ensure_pool_ready
            # when it re-raises via the event-wait path, and by the finally
            # below for the failure case.
            if deferred and self._mcp_started_event is not None:
                self._mcp_started_event.set()
        if not deferred:
            return
        # Deferred: only the successful-build path reaches here. If the
        # build failed, _mcp_started is still False and there is no live
        # pool to keep alive — exit so aclose's task-await returns.
        if not self._mcp_started:
            return
        # Stay alive so THIS task owns the teardown. aclose() sets the
        # close signal and awaits this task. The pool's anyio cancel scopes
        # were entered above (in this task) and can only be exited here.
        assert self._mcp_close_signal is not None
        # Cancellation may arrive at two points: while waiting on the
        # close signal, or during pool.aclose(). Both must be handled so
        # teardown completes (kill MCP subprocesses, exit anyio cancel
        # scopes in THIS owning task) and the task ends cleanly instead
        # of propagating a CancelledError that asyncio.shield in aclose()
        # would re-raise past its `except Exception` (CancelledError is
        # BaseException, not Exception, since 3.8+). This happens on
        # asyncio.run shutdown (Ctrl+C): the main task is cancelled, then
        # asyncio.run cancels all remaining tasks including this one.
        # We cannot shield pool.aclose() (would run it in a different task
        # and trip anyio's cross-task cancel-scope guard — the original
        # bug, see deferred-start-owns-teardown.md). AsyncExitStack.aclose
        # closes remaining contexts best-effort even when interrupted, so
        # subprocesses are still killed. After catching, finish
        # synchronously so the task completes normally.
        try:
            await self._mcp_close_signal.wait()
        except asyncio.CancelledError:
            logger.debug(
                "RuntimeResources: close-signal wait cancelled during "
                "deferred teardown; proceeding to pool cleanup"
            )
        if self._mcp_tool_pool is not None:
            try:
                await self._mcp_tool_pool.aclose()
            except asyncio.CancelledError:
                logger.debug(
                    "RuntimeResources: pool aclose cancelled during deferred "
                    "teardown; subprocess cleanup best-effort"
                )
            self._mcp_tool_pool = None
        await self._close_proxy_pool()
        self._mcp_started = False
        logger.debug("RuntimeResources: MCP pool closed (deferred teardown)")

    async def _build_proxy_pool(self) -> None:
        """Start provider proxies before lazy gateway construction begins."""
        if not self._proxy_configs:
            self._proxy_routes = {}
            return
        from kernel.proxy_launch.pool import (  # lazy: defers proxy launch startup cost
            ProxyPool,
        )

        self._proxy_pool = ProxyPool(self._proxy_configs)
        try:
            await self._proxy_pool.start()
        except BaseException:
            await self._proxy_pool.aclose()
            self._proxy_pool = None
            raise
        self._proxy_routes = self._proxy_pool.routes
        logger.info(
            "RuntimeResources: provider proxies ready for %d provider(s)",
            len(self._proxy_routes),
        )

    async def _close_proxy_pool(self) -> None:
        """Close host-owned proxy processes and discard their provider routes."""
        if self._proxy_pool is not None:
            await self._proxy_pool.aclose()
            self._proxy_pool = None
        self._proxy_routes = {}

    async def _build_pool(self) -> None:
        """Construct and start the MCP pool (shared by sync and deferred)."""
        if self._mcp_started:
            return

        if not self._mcp_warm_configs:
            logger.debug("RuntimeResources.start: no warm configs; pool will be None")
            self._mcp_started = True
            return

        from kernel.tools.mcp_client import MCPToolPool

        # The pool MUST carry both warm AND cold configs. Cold servers (e.g.
        # mem0, lisa-guard) are lazily connected on first scoped dispatch
        # (memory.* aliases route to the default backend, which may be cold).
        # If cold configs are omitted from the pool, server-scoped dispatch
        # (_call_tool_scoped) cannot resolve the server_id and returns
        # "Unknown MCP server '<pack>.mem0'" — breaking all memory operations
        # when the default backend is a cold HTTP server. The `warm` flag on
        # each config controls eager-vs-lazy start; cold servers stay down
        # at start() and connect best-effort on first call.
        pool_kwargs: dict[str, Any] = {}
        if self._in_process_mcp:
            pool_kwargs["in_process_resolver"] = _make_in_process_resolver()
        configs = await resolve_mcp_binary_configs(
            [*self._mcp_warm_configs, *self._mcp_cold_configs]
        )
        self._mcp_tool_pool = MCPToolPool(
            configs,
            **pool_kwargs,
        )
        try:
            await self._mcp_tool_pool.start()
        except BaseException:
            # start() failed — the pool may have partially-started warm
            # servers with open subprocesses. Close the pool to kill them,
            # then re-raise so the caller knows startup failed.
            try:
                await self._mcp_tool_pool.aclose()
            except Exception:  # pylint: disable=broad-except
                logger.debug("RuntimeResources: pool aclose after failed start", exc_info=True)
            self._mcp_tool_pool = None
            raise
        # Ambient values must be in place before the first turn so MCP
        # proxy policies can enforce runtime scope independently of which
        # capability implementation was selected.
        rt = self._runtime
        from runtime.ambient_resolvers import register_ambient_resolvers

        register_ambient_resolvers(
            self._mcp_tool_pool,
            user_slug=rt.config.user_slug,
            current_mode_id=rt.session.current_mode_id,
            app_id="lisa-host",
        )
        # Register generic aliases declared in pack MCP manifests so the
        # model sees stable tool names (e.g. ``cloudflare.search``) instead
        # of filesystem-derived namespaced ids.
        from runtime.mcp_pool import register_mcp_aliases

        register_mcp_aliases(self._mcp_tool_pool)
        self._mcp_started = True
        logger.info(
            "RuntimeResources: MCP pool started with %d warm server(s)",
            len(self._mcp_warm_configs),
        )

    async def aclose(self) -> None:
        """Close the MCP tool pool in the same event loop that started it.

        Must be called from the same event loop that called ``start()`` or
        ``start_pool_deferred()``. If a deferred start is still in flight,
        it is awaited first so the pool is fully constructed before we close
        it. After this, the runtime is no longer usable — construct a new
        one for a fresh session.
        """
        # Deferred-start path: the long-lived start task owns the anyio
        # cancel scopes (it entered the MCP stdio_client contexts). It must
        # close the pool in its own task. Signal it to tear down and await
        # the task (shielded so a cancellation of the caller doesn't leave
        # the teardown half-done and orphan MCP subprocesses).
        if (
            self._mcp_start_task is not None
            and not self._mcp_start_task.done()
            and self._mcp_close_signal is not None
        ):
            # If the build phase is still in flight, wait for it first so
            # the pool is fully constructed before we ask for teardown.
            if self._mcp_started_event is not None and not self._mcp_started_event.is_set():
                try:
                    await self._mcp_started_event.wait()
                except Exception:  # pylint: disable=broad-except
                    logger.debug(
                        "RuntimeResources: deferred build failed during aclose",
                        exc_info=True,
                    )
            self._mcp_close_signal.set()
            try:
                await asyncio.shield(self._mcp_start_task)
            except asyncio.CancelledError:
                # Cancellation of the aclose() caller (Ctrl+C /
                # asyncio.run shutdown) propagates here. The start_task
                # owns the real teardown and swallows CancelledError around
                # both close_signal.wait() and pool.aclose() so subprocesses
                # are killed best-effort and the task ends normally. We MUST
                # NOT re-raise here: propagating CancelledError out of
                # aclose() during asyncio.run shutdown surfaces as the ugly
                # "Cancelled via cancel scope ... by <Task _start_pool_impl>"
                # traceback. Suppress so aclose returns cleanly; asyncio.run
                # will then cancel + await the start task, whose own handlers
                # finish teardown. See deferred-start-owns-teardown.md.
                logger.debug(
                    "RuntimeResources: aclose cancelled while awaiting "
                    "deferred start task teardown; suppressing (start task "
                    "finishes teardown best-effort)"
                )
            except Exception:  # pylint: disable=broad-except
                # The task may raise the cached build-phase exception or a
                # teardown error; we're tearing down either way.
                logger.debug(
                    "RuntimeResources: deferred start task ended during aclose",
                    exc_info=True,
                )
            self._mcp_start_task = None
            self._mcp_start_exc = None
            self._mcp_started_event = None
            self._mcp_close_signal = None
            logger.debug("RuntimeResources: MCP pool closed (deferred)")
            return
        # Sync-start path (start() was called directly, or the deferred
        # task already finished): close the pool inline in this task, which
        # is the same task that entered the MCP anyio cancel scopes.
        # Teardown mirrors the build order in reverse (proxy pool was
        # started BEFORE the MCP pool): kill proxy processes first. Closing
        # the MCP pool first then the proxy pool hangs the loop (2026-09-21
        # headroom incident: the mcp command's runtime.aclose() never
        # returned after cold stdio sessions + warm proxies both ran in the
        # same task — the loop sat idle in select with no pending callbacks).
        # Proxy-first teardown is verified to complete cleanly.
        await self._close_proxy_pool()
        if self._mcp_tool_pool is not None:
            await self._mcp_tool_pool.aclose()
            self._mcp_tool_pool = None
        self._mcp_started = False
        self._mcp_start_task = None
        self._mcp_start_exc = None
        self._mcp_started_event = None
        self._mcp_close_signal = None
        logger.debug("RuntimeResources: MCP pool closed")


# ── HostRuntime — thin composition root ─────────────────────────────────


class HostRuntime:
    """Typed, mutable host runtime — the replacement for ``_RUNTIME_STATE``.

    Constructed from a ``HostConfig``. Composed of five focused sub-objects
    (see module docstring). ``config``, ``session_plan``, ``execution_policy``,
    and the lazily-discovered Copilot-customizations cache are the only state
    owned directly; everything else lives on ``paths``, ``providers``,
    ``resources``, ``session``, or ``services``.
    """

    def __init__(
        self,
        config: HostConfig,
        *,
        session_plan: SessionPlan | None = None,
        in_process_mcp: bool = False,
    ) -> None:
        self.session_plan = session_plan or compile_selected_session_plan(
            session_context=config.session_context,
            data_dir=config.data_dir,
        )
        source_matches_user_slug = (
            bool(config.session_context) and config.user_slug == config.session_context
        )
        effective_user_slug = (
            config.user_slug
            if config.user_slug and not source_matches_user_slug
            else self.session_plan.manifest.name
        )
        surface = self.session_plan.manifest.surface
        self.config = replace(
            config,
            user_slug=effective_user_slug,
            session_context=self.session_plan.source_uri,
            surface_type=config.surface_type or surface.type,
            surface_id=config.surface_id or surface.id,
            memory_scope=config.memory_scope or surface.memory_scope,
        )
        self.execution_policy = resolve_execution_policy(
            default_policy=self.config.execution_policy,
            session_request=self.session_plan.manifest.execution,
        )

        # The session-plan-declared model manifest path (context.models.path)
        # when present; otherwise falls back to config.resolved_models_path.
        self._models_path_override: Path | None = self._resolve_models_path()

        # Whether first-party MCP servers run in-process via the MCP SDK's
        # in-memory transport instead of subprocesses. Read by
        # ``RuntimeResources`` so the host never imports ``mcp_server``.
        self._in_process_mcp: bool = in_process_mcp

        # ── Composed sub-objects ──────────────────────────────────────
        self.paths = HostPaths.from_config(
            self.config, models_path_override=self._models_path_override
        )
        self.providers = ProviderRegistry()
        self.services = HostServices.build(self.paths.data_dir)
        self.session = SessionState()
        # A host started per message must resume the routing state the
        # previous process left; re-entering the start mode re-routed every
        # turn and missed the owner mode's provider cache prefix.
        self.session.current_mode_id = (
            latest_turn_owner_mode(thread_path=self.paths.thread_path)
            or self.session_plan.manifest.routing.start_mode
        )
        self.resources = RuntimeResources(self)

        # ── Copilot customizations cache ──────────────────────────────
        # Lazily discovered always-on + applyTo-scoped instructions, plus the
        # skill index, from .github/{instructions,skills} and the .claude/
        # equivalents. None means "not yet discovered"; "" means "discovered,
        # nothing found or discovery failed gracefully."
        self._copilot_customizations_text: str | None = None
        self._copilot_bundle: Any = None
        # Per-turn dedupe of scoped instruction bodies already appended to a
        # tool result, keyed by the kernel's dispatch token (one per turn and
        # per fork/delegate child) then by instruction path. Bounded by the
        # number of scoped instructions, so it cannot grow without limit.
        self._scoped_instructions_injected: dict[str, set[str]] = {}

        # Per-turn tally of policy auto-approvals; reset by handle_message
        # before consulting a turn's pending approval. ``runtime.approval``
        # increments it; capping here prevents a runaway auto-approve loop.
        self._auto_approved_this_turn: int = 0

    # ── Plan / config projections used across the host ────────────────

    @property
    def start_mode_id(self) -> str:
        """Return the plan-selected initial route mode."""
        return self.session_plan.manifest.routing.start_mode

    @property
    def copilot_customizations_text(self) -> str:
        """Unconditional customization guidance for the system prompt.

        Carries always-on files (``copilot-instructions.md``, ``AGENTS.md``,
        ``CLAUDE.md``), instructions that declare no ``applyTo``, and the skill
        index. Those apply to everything, so they belong in every request.

        Two things are held back because the system prompt is assembled before
        any file or task is known, and paying for them on every turn is waste:

        - Instructions that declare ``applyTo``, appended to a tool result the
          first time the model touches a matching path — see
          :meth:`scoped_instruction_text_for_path`.
        - Skill *bodies*, replaced by a name/description/path index the model
          reads from when a task matches — see ``_skill_index_block``.

        Returns the formatted Markdown block, or ``""`` if discovery fails or
        no customizations are found (graceful degradation). Cached after
        first access — files don't change mid-session.
        """
        if self._copilot_customizations_text is not None:
            return self._copilot_customizations_text

        try:
            bundle = self._copilot_customization_bundle()
            self._copilot_customizations_text = "\n\n".join(
                _unconditional_customization_blocks(bundle)
            )
        except Exception:
            logger.warning(
                "Failed to discover Copilot customizations; continuing without them.",
                exc_info=True,
            )
            self._copilot_customizations_text = ""

        return self._copilot_customizations_text

    def _copilot_customization_bundle(self) -> Any:
        """Discover and cache the workspace customization bundle."""
        if self._copilot_bundle is None:
            from sdk.content import discover_copilot_customizations

            self._copilot_bundle = discover_copilot_customizations(
                self._resolve_copilot_workspace_root()
            )
        return self._copilot_bundle

    def scoped_instruction_text_for_path(self, file_path: str) -> str:
        """Return the scoped instruction bodies whose ``applyTo`` matches ``file_path``.

        ``file_path`` is workspace-relative and POSIX-style, the same shape the
        SDK matcher expects. It is normalized first: backslashes and a leading
        ``./`` are removed, and an empty or traversal-escaping value matches
        nothing — injecting guidance for an unmatched path would resurrect
        exactly the eager injection this defers.

        An absolute path is read as workspace-relative (the leading slash is
        stripped) rather than rejected. That can only *miss* a
        directory-anchored glob, never attach a wrong body, which is the safe
        direction; coding tools document relative paths anyway.
        """
        return "\n\n".join(
            _scoped_instruction_block(inst)
            for inst in self._matching_scoped_instructions(file_path)
        )

    def _matching_scoped_instructions(self, file_path: str) -> tuple[Any, ...]:
        """Return the ``applyTo``-scoped instructions matching a workspace path."""
        return self._matching_scoped_instructions_for_paths((file_path,))

    def _matching_scoped_instructions_for_paths(
        self, file_paths: tuple[str, ...]
    ) -> tuple[Any, ...]:
        """Return the scoped instructions matching any of the given paths.

        Each path is normalized first; a path that escapes the workspace
        contributes nothing. Results are deduped by instruction path and keep
        discovery order, so a batched edit of five Python files yields one copy
        of the Python guidance.
        """
        relatives = [
            relative
            for relative in (_workspace_relative_posix_path(p) for p in file_paths)
            if relative
        ]
        if not relatives:
            return ()
        try:
            from sdk.content import matches_apply_to
        except Exception:  # pragma: no cover - the matcher ships with the SDK
            return ()
        matched: list[Any] = []
        seen: set[str] = set()
        for instruction in self._scoped_instructions():
            if instruction.path in seen:
                continue
            globs = list(instruction.apply_to)
            if any(matches_apply_to(globs, relative) for relative in relatives):
                matched.append(instruction)
                seen.add(instruction.path)
        return tuple(matched)

    def _scoped_instructions(self) -> tuple[Any, ...]:
        """Return the ``applyTo``-scoped instructions in the discovered bundle."""
        try:
            return tuple(
                inst
                for inst in self._copilot_customization_bundle().instructions
                if inst.apply_to
            )
        except Exception:
            logger.debug("Copilot customization discovery failed", exc_info=True)
            return ()

    def decorate_pack_tool_result(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        rendered: str,
        turn_token: str,
    ) -> str:
        """Append matching scoped instructions to a pack tool's rendered output.

        Wired as ``ToolHooks.post_dispatch_result``. Paths come from the tool's
        own arguments — the kernel does not know which argument holds one, and
        tools that touch no identifiable file (commit, run_check, a
        workspace-wide listing) carry none, so they pass through untouched.

        Each instruction body is appended at most once per dispatch token,
        which the kernel issues per turn *and* per fork or delegate child. The
        key is the instruction's path, so two instructions that happen to share
        a body are still injected separately.
        """
        paths = _tool_call_paths(arguments)
        if not paths:
            return rendered
        try:
            matched = self._matching_scoped_instructions_for_paths(paths)
        except Exception:  # pylint: disable=broad-except
            logger.debug("Scoped instruction lookup failed for %s", paths, exc_info=True)
            return rendered
        if not matched:
            return rendered

        seen = self._scoped_instruction_seen(turn_token)
        fresh = [inst for inst in matched if inst.path not in seen]
        if not fresh:
            return rendered
        seen.update(inst.path for inst in fresh)
        blocks = "\n\n".join(_scoped_instruction_block(inst) for inst in fresh)
        return f"{rendered}\n\n{blocks}"

    def _scoped_instruction_seen(self, turn_token: str) -> set[str]:
        """Return this dispatch token's already-injected path set.

        Bounded to the most recent tokens. A token is created per turn and per
        fork/delegate child, so a long-lived process would otherwise retain one
        entry per turn forever; no reader is interested in a token older than
        the turns still running.
        """
        seen = self._scoped_instructions_injected.get(turn_token)
        if seen is None:
            seen = self._scoped_instructions_injected[turn_token] = set()
            while len(self._scoped_instructions_injected) > _MAX_TRACKED_TURNS:
                oldest = next(iter(self._scoped_instructions_injected))
                if oldest == turn_token:
                    break
                del self._scoped_instructions_injected[oldest]
        return seen

    def _resolve_copilot_workspace_root(self) -> Path:
        """Resolve the workspace root for customization discovery.

        Resolution order, naming no specific pack:

        1. The session plan's authored ``context.workspace_root``.
        2. Any pack in ``resources.packs.configuration`` that declares a
           ``workspace_root`` key, taken in sorted pack-name order so the
           result is deterministic when several packs declare one.
        3. The process working directory.

        Relative paths resolve against the process working directory, which
        is the behavior the pack-configuration lookup has always had.
        """
        raw = self._authored_workspace_root() or self._pack_declared_workspace_root()
        if raw:
            return Path(raw).expanduser().resolve()
        return Path.cwd()

    def _authored_workspace_root(self) -> str | None:
        """Return the session plan's authored ``context.workspace_root``, if any."""
        try:
            return self.session_plan.manifest.context.workspace_root
        except Exception:
            return None

    def _pack_declared_workspace_root(self) -> str | None:
        """Return a ``workspace_root`` declared by any configured pack.

        Scans every pack's configuration block instead of naming one pack, so
        the runtime carries no dependency on a specific distribution. Sorted
        by pack name for determinism.
        """
        try:
            pack_config = self.session_plan.manifest.resources.packs.configuration
        except Exception:
            return None
        for pack_name in sorted(pack_config):
            block = pack_config[pack_name]
            if not isinstance(block, Mapping):
                continue
            raw = block.get("workspace_root")
            if isinstance(raw, str) and raw:
                return raw
        return None

    def _resolve_models_path(self) -> Path | None:
        """Resolve the model-manifest path override from the session plan.

        Returns the resolved absolute path when ``context.models.path`` is
        set on the session plan, or ``None`` to fall back to
        ``config.resolved_models_path``. A session-relative path requires
        ``session_plan.source_dir``; a missing file or wrong type raises a
        startup ``ValueError``.
        """
        from runtime.model_runtime import resolve_session_manifest_path

        session_source_dir: Path | None = None
        if self.session_plan.source_dir:
            session_source_dir = Path(self.session_plan.source_dir)
        return resolve_session_manifest_path(
            session_plan=self.session_plan,
            session_source_dir=session_source_dir,
            data_dir=self.config.data_dir,
        )

    # ── Async lifecycle delegates ─────────────────────────────────────

    async def start(self) -> None:
        await self.resources.start()

    def start_pool_deferred(self) -> asyncio.Task[None]:
        return self.resources.start_pool_deferred()

    async def ensure_pool_ready(self) -> None:
        await self.resources.ensure_pool_ready()

    async def aclose(self) -> None:
        await self.resources.aclose()


__all__ = [
    "HostPaths",
    "HostRuntime",
    "HostServices",
    "ProviderRegistry",
    "RuntimeResources",
    "SessionState",
    "TokenCounters",
    "TurnContext",
]
