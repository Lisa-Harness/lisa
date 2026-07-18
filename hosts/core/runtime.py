"""HostRuntime — the typed runtime object that replaces ``_RUNTIME_STATE``.

This is the mutable, stateful counterpart to the frozen ``HostConfig``. It
owns:
- Cached provider clients (Azure OpenAI, OpenAI, Anthropic, Ollama, Google).
- The model manifest cache.
- The MCP tool pool (warm + cold configs), with async ``start()``/``aclose()``
  that **must** run inside the host's real event loop.
- Session/token counters.
- Thread/log paths.
- Last-turn diagnostics.

Transport hosts construct a ``HostRuntime`` from a ``HostConfig``, call
``await runtime.start()`` inside their event loop, execute turns via
``handle_message``, and call ``await runtime.aclose()`` at shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hosts.core.config import HostConfig
from hosts.core.session_context import compile_selected_session_plan
from kernel.session_plan import SessionPlan

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TokenCounters:
    """Per-turn and per-session token counters."""

    # Per-turn (reset at the start of each turn).
    turn_prompt_tokens: int = 0
    turn_completion_tokens: int = 0
    turn_total_tokens: int = 0
    turn_cached_tokens: int = 0
    turn_reasoning_tokens: int = 0
    turn_duration_ms: float = 0.0
    turn_latency_ms: int = 0

    # Per-session (accumulate across turns).
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_total_tokens: int = 0
    session_cached_tokens: int = 0
    session_reasoning_tokens: int = 0
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
        self.turn_duration_ms = 0.0
        self.turn_latency_ms = 0


@dataclass(slots=True)
class TurnContext:
    """Mutable per-turn diagnostic context.

    Passed through ``handle_message`` and updated by ``on_turn_complete``.
    Transport hosts read this via ``runtime.last_turn_context`` for
    ``/debug context`` rendering.
    """

    timestamp_utc: str = ""
    raw_message: str = ""
    mode_id: str = "default"
    mode_kind: str = ""
    query: str = ""
    deep_route: bool = False
    preferred_model_id: str = ""
    resolved_model: dict[str, Any] = field(default_factory=dict)
    handoff: str = ""
    turn_input: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    router_debug: dict[str, Any] | None = None


class HostRuntime:
    """Typed, mutable host runtime — the replacement for ``_RUNTIME_STATE``.

    Constructed from a ``HostConfig``. Owns all mutable state that was
    previously scattered across the ``_RUNTIME_STATE`` dict: cached provider
    clients, manifest cache, MCP pool handles, token counters, thread/log
    paths, and last-turn diagnostics.

    Async lifecycle::

        runtime = HostRuntime(config)
        await runtime.start()     # starts MCP pool in host's event loop
        result = await handle_message(runtime, "/search test", transport)
        await runtime.aclose()    # closes MCP pool in same loop
    """

    def __init__(self, config: HostConfig, *, session_plan: SessionPlan | None = None) -> None:
        self.session_plan = session_plan or compile_selected_session_plan(
            session_context=config.session_context,
            data_dir=config.data_dir,
            force_refresh=config.session_context_refresh,
        )
        source_matches_user_slug = (
            bool(config.session_context)
            and config.user_slug == config.session_context
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

        # ── Cached provider clients ───────────────────────────────────
        self._azure_client: Any = None
        self._openai_client: Any = None
        self._anthropic_http_client: Any = None
        self._ollama_client: Any = None
        self._google_client: Any = None

        # ── Manifest cache ────────────────────────────────────────────
        self._manifest: Any = None

        # ── MCP pool ──────────────────────────────────────────────────
        self._mcp_tool_pool: Any = None
        self._mcp_warm_configs: list[Any] = []
        self._mcp_cold_configs: list[Any] = []
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

        # ── Memory config (selected by capability provider policy) ───
        self.memory_config: Any = None
        self.memory_config_pack_slug: str | None = None

        # ── Token counters ────────────────────────────────────────────
        self.tokens = TokenCounters()
        if self.tokens.session_started_utc is None:
            self.tokens.session_started_utc = datetime.now(UTC)

        # ── Runtime route state ───────────────────────────────────────
        self.current_mode_id: str = self.session_plan.manifest.routing.start_mode
        self.current_stage_id: str = "-"
        self.next_model_id: str = "-"
        self.next_model_provider: str = "-"

        # ── State snapshot ────────────────────────────────────────────
        self._state_snapshot: dict[str, Any] | None = None

        # ── Slice ─────────────────────────────────────────────────────
        self.last_slice_finished_at: float | None = None
        # Latest ContextBudgetSnapshot produced by the context-mapper pass
        # on this turn. Mutated by the host's ``assemble_user_context``
        # path; read by utilization-aware slice providers (e.g. the
        # compaction pack) via ``TurnHooks.last_context_snapshot``.
        self.last_context_snapshot: Any = None

        # ── Last-turn diagnostics ─────────────────────────────────────
        self.last_turn_context: TurnContext | None = None
        self.last_router_debug: dict[str, Any] | None = None

        # ── Restart state ─────────────────────────────────────────────
        self.restart_required: bool = False
        self.restart_reason: str = ""
        # Stashed by the CLI host from the LISA_RESTART_RESUME_MESSAGE env
        # var on startup; the REPL replays it as the first turn, then clears
        # it. ``None`` means no resume pending.
        self.pending_resume_message: str | None = None

        # ── Approval persistence ──────────────────────────────────────
        from hosts.core.approvals import FileApprovalService
        from hosts.core.code_artifacts import FileCodeChangeArtifactStore

        self.approval_service = FileApprovalService(self.data_dir / "approvals.json")
        self.code_change_artifact_store = FileCodeChangeArtifactStore(
            self.data_dir / "code-change-artifacts.json"
        )

        # ── Token usage handler ───────────────────────────────────────
        self._token_usage_handler: Any = None

    @property
    def start_mode_id(self) -> str:
        """Return the plan-selected initial route mode."""

        return self.session_plan.manifest.routing.start_mode

    # ── Path accessors ────────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    @property
    def thread_dir(self) -> Path:
        return self.config.resolved_thread_dir

    @property
    def thread_path(self) -> Path | None:
        if self.config.no_thread:
            return None
        user = self.config.user_slug
        if not user:
            return None
        return self.thread_dir / f"{user}.jsonl"

    @property
    def log_path(self) -> Path | None:
        return self.config.resolved_log_path

    @property
    def host_log_path(self) -> Path:
        return self.config.resolved_host_log_path

    @property
    def token_usage_path(self) -> Path:
        return self.data_dir / "token-usage.jsonl"

    @property
    def models_path(self) -> Path:
        return self.config.resolved_models_path

    @property
    def models_proposal_path(self) -> Path:
        return self.config.resolved_models_proposal_path

    # ── Provider client accessors (lazy) ──────────────────────────────

    @property
    def azure_client(self) -> Any:
        return self._azure_client

    @azure_client.setter
    def azure_client(self, value: Any) -> None:
        self._azure_client = value

    @property
    def openai_client(self) -> Any:
        return self._openai_client

    @openai_client.setter
    def openai_client(self, value: Any) -> None:
        self._openai_client = value

    @property
    def anthropic_http_client(self) -> Any:
        return self._anthropic_http_client

    @anthropic_http_client.setter
    def anthropic_http_client(self, value: Any) -> None:
        self._anthropic_http_client = value

    @property
    def ollama_client(self) -> Any:
        return self._ollama_client

    @ollama_client.setter
    def ollama_client(self, value: Any) -> None:
        self._ollama_client = value

    @property
    def google_client(self) -> Any:
        return self._google_client

    @google_client.setter
    def google_client(self, value: Any) -> None:
        self._google_client = value

    # ── Manifest cache ────────────────────────────────────────────────

    @property
    def manifest(self) -> Any:
        return self._manifest

    @manifest.setter
    def manifest(self, value: Any) -> None:
        self._manifest = value

    # ── MCP pool ──────────────────────────────────────────────────────

    @property
    def mcp_tool_pool(self) -> Any:
        return self._mcp_tool_pool

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
        that read :attr:`mcp_tool_pool` must ``await`` :meth:`ensure_pool_ready`
        first. Idempotent: if a deferred start is already in flight,
        returns the existing task.
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
        self._mcp_start_task = asyncio.create_task(
            self._start_pool_impl(deferred=True)
        )
        return self._mcp_start_task

    async def ensure_pool_ready(self) -> None:
        """Await the deferred pool start if one is in flight.

        Call this before reading :attr:`mcp_tool_pool` when the host used
        :meth:`start_pool_deferred`. If the pool was started synchronously
        via :meth:`start`, this is a no-op. If no start was ever scheduled,
        runs :meth:`start` inline (back-compat for hosts that never
        deferred).

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
            await self._build_pool()
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
                "HostRuntime: close-signal wait cancelled during deferred "
                "teardown; proceeding to pool cleanup"
            )
        if self._mcp_tool_pool is not None:
            try:
                await self._mcp_tool_pool.aclose()
            except asyncio.CancelledError:
                logger.debug(
                    "HostRuntime: pool aclose cancelled during deferred "
                    "teardown; subprocess cleanup best-effort"
                )
            self._mcp_tool_pool = None
        self._mcp_started = False
        logger.debug("HostRuntime: MCP pool closed (deferred teardown)")

    async def _build_pool(self) -> None:
        """Construct and start the MCP pool (shared by sync and deferred)."""
        if self._mcp_started:
            return

        if not self._mcp_warm_configs:
            logger.debug("HostRuntime.start: no warm configs; pool will be None")
            self._mcp_started = True
            return

        from kernel.tools.mcp_client import MCPToolPool  # noqa: PLC0415

        # The pool MUST carry both warm AND cold configs. Cold servers (e.g.
        # mem0, lisa-guard) are lazily connected on first scoped dispatch
        # (memory.* aliases route to the default backend, which may be cold).
        # If cold configs are omitted from the pool, server-scoped dispatch
        # (_call_tool_scoped) cannot resolve the server_id and returns
        # "Unknown MCP server '<pack>.mem0'" — breaking all memory operations
        # when the default backend is a cold HTTP server. The `warm` flag on
        # each config controls eager-vs-lazy start; cold servers stay down
        # at start() and connect best-effort on first call.
        self._mcp_tool_pool = MCPToolPool([*self._mcp_warm_configs, *self._mcp_cold_configs])
        try:
            await self._mcp_tool_pool.start()
        except BaseException:
            # start() failed — the pool may have partially-started warm
            # servers with open subprocesses. Close the pool to kill them,
            # then re-raise so the caller knows startup failed.
            try:
                await self._mcp_tool_pool.aclose()
            except Exception:  # pylint: disable=broad-except
                logger.debug("HostRuntime: pool aclose after failed start", exc_info=True)
            self._mcp_tool_pool = None
            raise
        # Register ambient resolvers + the generic ``memory.*`` synthetic
        # aliases on the now-live pool. This MUST run after ``start()`` so
        # the alias handlers forward to live server sessions and the
        # proxy's ambient resolvers are in place before the first turn.
        # Without this, transport hosts that call ``runtime.start()``
        # directly (cli, web, vscode) would never register
        # ``memory.add`` / ``memory.search`` / ``memory.list`` — the
        # learning-pack operations (recall_memory,
        # synthesize_and_store_learning) reach them via
        # ``ctx.execute_tool_fn`` and fail with "Tool 'memory.list' is not
        # registered on any MCP server". The ``hosts/little_lisa`` host
        # has its own ``astart_warm_mcp_pool`` that calls this too; the
        # registration is idempotent at the pool level (synthetic tools
        # replace by name), so double registration is harmless.
        from hosts.core.memory_aliases import register_memory_aliases  # noqa: PLC0415

        register_memory_aliases(
            self._mcp_tool_pool,
            memory_config=self.memory_config,
            memory_config_pack_slug=self.memory_config_pack_slug,
            user_slug=self.config.user_slug,
            current_mode_id=self.current_mode_id,
            app_id="lisa-host",
        )
        # Register generic aliases declared in pack MCP manifests so the
        # model sees stable tool names (e.g. ``cloudflare.search``) instead
        # of filesystem-derived namespaced ids.
        from hosts.core.mcp_pool import register_mcp_aliases  # noqa: PLC0415

        register_mcp_aliases(self._mcp_tool_pool)
        self._mcp_started = True
        logger.info(
            "HostRuntime: MCP pool started with %d warm server(s)",
            len(self._mcp_warm_configs),
        )

    async def aclose(self) -> None:
        """Close the MCP tool pool in the same event loop that started it.

        Must be called from the same event loop that called ``start()``
        or ``start_pool_deferred()``. If a deferred start is still in
        flight, it is awaited first so the pool is fully constructed
        before we close it. After this, the runtime is no longer usable
        — construct a new one for a fresh session.
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
                        "HostRuntime: deferred build failed during aclose",
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
                    "HostRuntime: aclose cancelled while awaiting "
                    "deferred start task teardown; suppressing (start task "
                    "finishes teardown best-effort)"
                )
            except Exception:  # pylint: disable=broad-except
                # The task may raise the cached build-phase exception or a
                # teardown error; we're tearing down either way.
                logger.debug(
                    "HostRuntime: deferred start task ended during aclose",
                    exc_info=True,
                )
            self._mcp_start_task = None
            self._mcp_start_exc = None
            self._mcp_started_event = None
            self._mcp_close_signal = None
            logger.debug("HostRuntime: MCP pool closed (deferred)")
            return
        # Sync-start path (start() was called directly, or the deferred
        # task already finished): close the pool inline in this task, which
        # is the same task that entered the MCP anyio cancel scopes.
        if self._mcp_tool_pool is not None:
            await self._mcp_tool_pool.aclose()
            self._mcp_tool_pool = None
        self._mcp_started = False
        self._mcp_start_task = None
        self._mcp_start_exc = None
        self._mcp_started_event = None
        self._mcp_close_signal = None
        logger.debug("HostRuntime: MCP pool closed")

    # ── State snapshot ────────────────────────────────────────────────

    @property
    def state_snapshot(self) -> dict[str, Any] | None:
        return self._state_snapshot

    @state_snapshot.setter
    def state_snapshot(self, value: dict[str, Any] | None) -> None:
        self._state_snapshot = value

    # ── Diagnostics ───────────────────────────────────────────────────

    def set_last_turn_context(self, ctx: TurnContext) -> None:
        self.last_turn_context = ctx

    def set_last_router_debug(self, payload: dict[str, Any]) -> None:
        self.last_router_debug = payload

    # ── Restart ───────────────────────────────────────────────────────

    def request_restart(self, reason: str) -> None:
        self.restart_required = True
        self.restart_reason = reason