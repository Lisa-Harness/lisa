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
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hosts.core.config import HostConfig

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
    frontdoor_debug: dict[str, Any] | None = None


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

    def __init__(self, config: HostConfig) -> None:
        self.config = config

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

        # ── Memory config (selected by capability provider policy) ───
        self.memory_config: Any = None
        self.memory_config_pack_slug: str | None = None

        # ── Token counters ────────────────────────────────────────────
        self.tokens = TokenCounters()
        if self.tokens.session_started_utc is None:
            self.tokens.session_started_utc = datetime.now(UTC)

        # ── Runtime route state ───────────────────────────────────────
        self.current_mode_id: str = config.start_mode or "default"
        self.current_stage_id: str = "-"
        self.next_model_id: str = "-"
        self.next_model_provider: str = "-"

        # ── State snapshot ────────────────────────────────────────────
        self._state_snapshot: dict[str, Any] | None = None

        # ── Slice ─────────────────────────────────────────────────────
        self.last_slice_finished_at: float | None = None

        # ── Last-turn diagnostics ─────────────────────────────────────
        self.last_turn_context: TurnContext | None = None
        self.last_frontdoor_debug: dict[str, Any] | None = None

        # ── Restart state ─────────────────────────────────────────────
        self.restart_required: bool = False
        self.restart_reason: str = ""
        # Stashed by the CLI host from the LISA_RESTART_RESUME_MESSAGE env
        # var on startup; the REPL replays it as the first turn, then clears
        # it. ``None`` means no resume pending.
        self.pending_resume_message: str | None = None

        # ── Token usage handler ───────────────────────────────────────
        self._token_usage_handler: Any = None

    # ── Path accessors ────────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    @property
    def memory_db_path(self) -> Path:
        return self.config.resolved_memory_db_path

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
        if self._mcp_start_task is not None:
            try:
                await self._mcp_start_task
            except BaseException as exc:
                # Cache the exception so subsequent callers don't re-await
                # a completed task (which would re-raise). The task object
                # is consumed; future calls hit the cached exc path above.
                self._mcp_start_exc = exc
                raise
            return
        if not self._mcp_started:
            await self.start()

    async def _start_pool_impl(self, *, deferred: bool) -> None:
        """Shared pool-start implementation for start() and deferred start."""
        if self._mcp_started:
            return

        if not self._mcp_warm_configs:
            logger.debug("HostRuntime.start: no warm configs; pool will be None")
            self._mcp_started = True
            return

        from kernel.tools.mcp_client import MCPToolPool  # noqa: PLC0415

        self._mcp_tool_pool = MCPToolPool(self._mcp_warm_configs)
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
        self._mcp_started = True
        logger.info(
            "HostRuntime: MCP pool started with %d warm server(s)%s",
            len(self._mcp_warm_configs),
            " (deferred)" if deferred else "",
        )

    async def aclose(self) -> None:
        """Close the MCP tool pool in the same event loop that started it.

        Must be called from the same event loop that called ``start()``
        or ``start_pool_deferred()``. If a deferred start is still in
        flight, it is awaited first so the pool is fully constructed
        before we close it. After this, the runtime is no longer usable
        — construct a new one for a fresh session.
        """
        # If a deferred start is still in flight, await it first so the
        # pool is fully constructed before we close it. On failure, the
        # task stored the exception; swallow it — we're tearing down.
        if self._mcp_start_task is not None and not self._mcp_start_task.done():
            try:
                await self._mcp_start_task
            except Exception:  # pylint: disable=broad-except
                logger.debug(
                    "HostRuntime: deferred start failed during aclose",
                    exc_info=True,
                )
        if self._mcp_tool_pool is not None:
            await self._mcp_tool_pool.aclose()
            self._mcp_tool_pool = None
        self._mcp_started = False
        self._mcp_start_task = None
        self._mcp_start_exc = None
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

    def set_last_frontdoor_debug(self, payload: dict[str, Any]) -> None:
        self.last_frontdoor_debug = payload

    # ── Restart ───────────────────────────────────────────────────────

    def request_restart(self, reason: str) -> None:
        self.restart_required = True
        self.restart_reason = reason