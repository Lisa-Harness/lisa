"""Transport-neutral message dispatcher for Lisa Harness hosts.

This is the core ``handle_message`` entry point that all transport hosts
(cli, web) call to execute one user turn. It:

1. **Host command dispatch** — delegates to ``runtime.commands`` for
   ``/help``, ``/settings``, ``/tokens``, ``/debug``, ``/consolidate``,
    ``/intake``, ``/pack``, ``/models``, ``/modes``, ``/operations``,
    ``/tools``, and to ``runtime.approval`` for ``/approve`` / ``/deny``.
    Pack-registered commands are discovered via
    ``lisa.extensions.host_commands`` and dispatched after built-in host
    commands.
2. **Approval resume** — a short affirmative/negative reply while exactly
   one approval is pending is treated as an implicit ``/approve`` /
   ``/deny`` via :func:`runtime.approval.try_affirmative_approval_resume`.
3. **Model turn routing** — for non-command messages: parses slash-command
   routes, then routes natural language directly to the current mode and
   executes the turn via ``kernel.turn_runner.run_turn``. Mode handoff
   happens mid-turn through the ``transition_stage`` tool — there is no
   separate router planner.
4. **Turn execution** — builds ``TurnHooks`` via
   ``runtime.turn_hooks_builder.build_turn_hooks``, calls ``run_turn``,
   persists thread entries, updates token counters, and emits transport
   callbacks at each pipeline stage.

The approval subsystem (identity rebuild, affirmative resume, policy
auto-continue, post-approval continuation, paired-recovery settlement)
lives in :mod:`runtime.approval`. This file owns only dispatch and the
model-turn pipeline; it injects its own :func:`_route_model_turn` into the
approval entry points so the post-approval synthesis turn reuses the
single-turn machinery without a circular import.

It does NOT import Rich, Typer, aiohttp, or any transport-specific code.
Rendering is the transport host's job.

Public surface
--------------
- :func:`handle_message`
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kernel.ipsar import parse_route
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import OperationExecutionError
from kernel.seams.protocols import ExtensionItemBuilder
from kernel.turn_continuation import TurnContinuation, resume_after_pause
from kernel.turn_phases._routing import TurnDecision, evaluate_turn_decision
from kernel.turn_runner import (
    TurnResult,
    run_turn,
)
from sdk.logging_setup import emit_structured_event

from runtime.approval import (
    build_approval_data,
    handle_approval_command,
    is_affirmative_resume,
    maybe_auto_continue_approval,
    try_affirmative_approval_resume,
)
from runtime.commands import (
    format_operation_error,
    format_token_stats,
    format_unsupported_mode,
    handle_debug_command,
    handle_help_command,
    handle_intake_command,
    handle_last_command,
    handle_mcp_command,
    handle_models_command,
    handle_pack_command,
    handle_policy_command,
    handle_proxy_command,
    handle_simple_command,
    handle_workflow_command,
    mode_support_gap,
    preferred_model_id_for_route,
    render_modes_list,
    render_operations_list,
    render_runtime_settings,
    render_tools_list,
    run_explicit_slice_command,
    set_runtime_route_preview,
)
from runtime.context_assembly import build_turn_context
from runtime.execution_policy import resolve_execution_policy
from runtime.provider_endpoints import proxy_route_view
from runtime.thread import append_thread_entry, load_recent_thread_entries, paused_turn_owner_mode
from runtime.token_accounting import apply_turn_usage, coerce_non_negative_int
from runtime.turn_continuation import (
    load_continuation,
    persist_continuation,
    usage_since_resume,
)
from runtime.turn_hooks_builder import build_turn_hooks
from runtime.turn_logging import build_turn_completed_payload
from runtime.types import HostCommandResult

# ── Pack-command discovery (lazy) ────────────────────────────────────

_pack_command_cache: list[Any] | None = None


def _pack_host_commands() -> list[Any]:
    """Return cached pack-registered host command entries.

    Discovers ``lisa.extensions.host_commands`` entry points on first call
    and caches the result. Pack commands are consulted after built-in host
    commands in ``_dispatch_host_command``.
    """
    global _pack_command_cache  # noqa: PLW0603 — module-level cache seam
    if _pack_command_cache is not None:
        return _pack_command_cache
    from kernel.host_command_discovery import discover_pack_host_commands

    _pack_command_cache = discover_pack_host_commands()
    return _pack_command_cache


# ── Token-accounting adapter ──────────────────────────────────────────


def _token_state_dict(runtime: HostRuntime) -> dict[str, Any]:
    """Build a dict view of the runtime's token counters for apply_turn_usage.

    ``apply_turn_usage`` expects a dict-based state (the legacy
    ``_RUNTIME_STATE`` shape). This helper builds a compatible dict from
    the typed ``TokenCounters`` dataclass, calls the accounting function,
    then syncs the results back.
    """
    t = runtime.session.tokens
    return {
        "turn_prompt_tokens": t.turn_prompt_tokens,
        "turn_completion_tokens": t.turn_completion_tokens,
        "turn_total_tokens": t.turn_total_tokens,
        "turn_cached_tokens": t.turn_cached_tokens,
        "turn_reasoning_tokens": t.turn_reasoning_tokens,
        "turn_cache_creation_tokens": t.turn_cache_creation_tokens,
        "turn_duration_ms": t.turn_duration_ms,
        "turn_latency_ms": t.turn_latency_ms,
        "session_prompt_tokens": t.session_prompt_tokens,
        "session_completion_tokens": t.session_completion_tokens,
        "session_total_tokens": t.session_total_tokens,
        "session_cached_tokens": t.session_cached_tokens,
        "session_reasoning_tokens": t.session_reasoning_tokens,
        "session_cache_creation_tokens": t.session_cache_creation_tokens,
        "session_duration_ms": t.session_duration_ms,
        "session_latency_ms": t.session_latency_ms,
        "session_token_ledger": t.session_token_ledger,
        "model_token_totals": t.model_token_totals,
        "mode_token_totals": t.mode_token_totals,
        "stage_token_totals": t.stage_token_totals,
        "session_turn_count": t.session_turn_count,
        "session_success_count": t.session_success_count,
        "session_error_count": t.session_error_count,
        "session_cancelled_count": t.session_cancelled_count,
        "session_total_latency_ms": t.session_total_latency_ms,
        "session_mode_counts": t.session_mode_counts,
    }


def _sync_token_state(runtime: HostRuntime, state: dict[str, Any]) -> None:
    """Sync the dict-based token state back into the typed TokenCounters."""
    t = runtime.session.tokens
    t.turn_prompt_tokens = coerce_non_negative_int(state.get("turn_prompt_tokens", 0))
    t.turn_completion_tokens = coerce_non_negative_int(state.get("turn_completion_tokens", 0))
    t.turn_total_tokens = coerce_non_negative_int(state.get("turn_total_tokens", 0))
    t.turn_cached_tokens = coerce_non_negative_int(state.get("turn_cached_tokens", 0))
    t.turn_reasoning_tokens = coerce_non_negative_int(state.get("turn_reasoning_tokens", 0))
    t.turn_cache_creation_tokens = coerce_non_negative_int(
        state.get("turn_cache_creation_tokens", 0)
    )
    t.turn_duration_ms = float(state.get("turn_duration_ms", 0.0) or 0.0)
    t.turn_latency_ms = coerce_non_negative_int(state.get("turn_latency_ms", 0))
    t.session_prompt_tokens = coerce_non_negative_int(state.get("session_prompt_tokens", 0))
    t.session_completion_tokens = coerce_non_negative_int(state.get("session_completion_tokens", 0))
    t.session_total_tokens = coerce_non_negative_int(state.get("session_total_tokens", 0))
    t.session_cached_tokens = coerce_non_negative_int(state.get("session_cached_tokens", 0))
    t.session_reasoning_tokens = coerce_non_negative_int(state.get("session_reasoning_tokens", 0))
    t.session_cache_creation_tokens = coerce_non_negative_int(
        state.get("session_cache_creation_tokens", 0)
    )
    t.session_duration_ms = float(state.get("session_duration_ms", 0.0) or 0.0)
    t.session_latency_ms = coerce_non_negative_int(state.get("session_latency_ms", 0))
    t.session_token_ledger = state.get("session_token_ledger", t.session_token_ledger)
    t.model_token_totals = state.get("model_token_totals", t.model_token_totals)
    t.mode_token_totals = state.get("mode_token_totals", t.mode_token_totals)
    t.stage_token_totals = state.get("stage_token_totals", t.stage_token_totals)
    t.session_turn_count = coerce_non_negative_int(state.get("session_turn_count", 0))
    t.session_success_count = coerce_non_negative_int(state.get("session_success_count", 0))
    t.session_error_count = coerce_non_negative_int(state.get("session_error_count", 0))
    t.session_cancelled_count = coerce_non_negative_int(state.get("session_cancelled_count", 0))
    t.session_total_latency_ms = float(state.get("session_total_latency_ms", 0.0) or 0.0)
    t.session_mode_counts = state.get("session_mode_counts", t.session_mode_counts)


def _apply_turn_usage(
    runtime: HostRuntime,
    result: Any,
    *,
    model_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
) -> None:
    """Apply turn usage via the dict-based token_accounting module."""
    state = _token_state_dict(runtime)
    per_model = getattr(result, "per_model_usage", None)
    per_mode = getattr(result, "per_mode_usage", None)
    apply_turn_usage(
        state,
        result,
        model_id=model_id,
        mode_id=mode_id,
        stage_id=stage_id,
        per_model_usage=per_model if isinstance(per_model, list) else None,
        per_mode_usage=per_mode if isinstance(per_mode, list) else None,
    )
    _sync_token_state(runtime, state)


if TYPE_CHECKING:
    from runtime.callbacks import TransportCallbacks
    from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)


async def decide_pending_approval(
    runtime: HostRuntime,
    request_id: str,
    *,
    approve: bool,
    transport: TransportCallbacks,
    remember: str | None = None,
    remember_default: bool = False,
) -> Any:
    """Decide one approval by value through the typed approval core.

    Public typed seam for hosts that speak values (not the slash grammar):
    injects this module's ``_route_model_turn`` as the continuation router,
    so the post-approval synthesis turn reuses the single-turn machinery
    while the caller never imports a private symbol.
    """
    from runtime.approval import decide_approval

    return await decide_approval(
        runtime,
        request_id,
        approve=approve,
        transport=transport,
        route_model_turn=_route_model_turn,
        remember=remember,
        remember_default=remember_default,
    )


async def handle_message(
    runtime: HostRuntime,
    message: str,
    transport: TransportCallbacks | None = None,
    *,
    extra_item_builders: list[ExtensionItemBuilder] | None = None,
) -> HostCommandResult:
    """Route one user message through the host runtime.

    Args:
        runtime: The typed host runtime with config, MCP pool, and state.
        message: The raw user message (may be a slash command or natural
            language).
        transport: Optional transport callbacks for streaming, debug
            rendering, and error delivery. When ``None``, a
            ``NullTransport`` is used (headless / test mode).
        extra_item_builders: Optional host-supplied ``ExtensionItemBuilder``
            instances (e.g. the VS Code sidecar's editor-context bridge).
            Injected through to ``build_turn_hooks`` so host-contributed
            context items reach the session-owned context mapper. ``None``
            (default) behaves exactly as before.

    Returns:
        A ``HostCommandResult`` with the response text and optional
        structured data for transport rendering.
    """
    from runtime.callbacks import NullTransport

    tport: TransportCallbacks = transport or NullTransport()
    runtime.session.auto_approved_this_turn = 0
    stripped = message.strip()
    lowered = stripped.lower()

    logger.debug("handle_message: %s", stripped[:80])

    # ── Host command dispatch ─────────────────────────────────────────
    cmd_result = await _dispatch_host_command(runtime, stripped, lowered, message, tport)
    if cmd_result is not None:
        return cmd_result

    # ── Affirmative-resume of a paused approval ─────────────────────
    # When the user replies with a short affirmative ("proceed", "continue",
    # "approved", "ok", "yes", …) while exactly one approval is pending for
    # this session, treat it as an implicit ``/approve <id>`` rather than
    # routing a fresh model turn. Without this, the model re-issues the same
    # effectful tool, the registry mints a new ``request_id``, and the user
    # is asked to approve a different id on every retry — the approval loop.
    # The kernel-side idempotency (``_find_pending_for_action``) prevents the
    # duplicate mint even if this check is bypassed; this host check keeps
    # the user-facing experience stable for conversational resumes.
    resume_result = await try_affirmative_approval_resume(
        runtime,
        stripped,
        lowered,
        tport,
        route_model_turn=_route_model_turn,
    )
    if resume_result is not None:
        return resume_result

    # ── Model turn (mode dispatch + run_turn) ────────────────────────
    # ``_route_model_turn`` returns a ``HostCommandResult`` on the normal
    # turn path (so ``operator_exit`` survives) but a bare ``str`` on the
    # early-exit paths (budget gate, unsupported mode). Normalize to a
    # ``HostCommandResult`` so the transport always receives a string for
    # ``on_response`` and the caller preserves ``operator_exit``.
    try:
        model_result = await _route_model_turn(
            runtime, message, tport, extra_item_builders=extra_item_builders
        )
    except OperationExecutionError as exc:
        await tport.on_error(f"Operation error: {exc}", exc)
        response_text = format_operation_error(exc)
        await tport.on_response(response_text)
        return HostCommandResult(text=response_text, is_error=True)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "handle_message: turn failed: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        await tport.on_error(f"Turn failed: {type(exc).__name__}: {exc}", exc)
        response_text = f"Turn failed: {type(exc).__name__}: {exc}"
        await tport.on_response(response_text)
        return HostCommandResult(text=response_text, is_error=True)

    if isinstance(model_result, HostCommandResult):
        await tport.on_response(model_result.text, model_result.data)
        return model_result
    await tport.on_response(model_result)
    return HostCommandResult(text=model_result)


# ── Host command dispatch ──────────────────────────────────────────────

# Host-command vocabulary registration is owned by ``runtime.bootstrap`` and
# runs exactly once per process — see
# :func:`runtime.host_commands.register_builtin_host_commands`. Registering at
# import time here was a side effect that broke the kernel rule
# "registration and operation discovery must be side-effect-free on
# import" and coupled every test that imported this module to a globally
# mutated registry.


def _dispatch_simple_command(
    runtime: HostRuntime,
    lowered: str,
    message: str,
) -> str | None:
    """Match the simple static-render built-in commands.

    These commands take no argument handling beyond the message itself —
    each renders a fixed runtime view (tokens, settings, modes, operations,
    tools). Returns the response text, or ``None`` when the lowered message
    matches no simple command.
    """
    if lowered in {"/tokens", "tokens"}:
        return handle_simple_command(runtime, message, "tokens", format_token_stats(runtime))
    if lowered in {"/settings", "settings"}:
        return handle_simple_command(runtime, message, "settings", render_runtime_settings(runtime))
    if lowered in {"/modes", "modes"}:
        return handle_simple_command(runtime, message, "modes", render_modes_list(runtime))
    if lowered in {"/operations", "operations"}:
        return handle_simple_command(
            runtime, message, "operations", render_operations_list(runtime)
        )
    if lowered in {"/tools", "tools"}:
        return handle_simple_command(runtime, message, "tools", render_tools_list(runtime))
    return None


async def _dispatch_approval_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    transport: TransportCallbacks,
) -> tuple[str | None, HostCommandResult | None]:
    """Match the /approve and /deny approval commands.

    Returns:
        A ``(response_text, continuation_result)`` tuple; both ``None``
        when the message is not an approval command. ``continuation_result``
        is set when the approve path ran a post-approval continuation turn
        and returned a structured result (preserving ``operator_exit``)
        rather than plain text.
    """
    response_text: str | None = None
    continuation_result: HostCommandResult | None = None
    if lowered == "/approve" or lowered.startswith("/approve "):
        approved = await handle_approval_command(
            runtime,
            stripped,
            approve=True,
            transport=transport,
            route_model_turn=_route_model_turn,
        )
        if isinstance(approved, HostCommandResult):
            continuation_result = approved
        else:
            response_text = approved
    elif lowered == "/deny" or lowered.startswith("/deny "):
        response_text = await handle_approval_command(
            runtime,
            stripped,
            approve=False,
            transport=transport,
            route_model_turn=_route_model_turn,
        )
    return response_text, continuation_result


_PREFIX_ASYNC_COMMANDS: tuple[
    tuple[str, Callable[[HostRuntime, str, str], str | Awaitable[str]]], ...
] = (
    # (prefix, handler) pairs the dispatcher calls with
    # (runtime, stripped, message). First matching prefix wins. Handlers
    # may be sync or async — awaitables are awaited.
    ("/pack", handle_pack_command),
    ("/models", handle_models_command),
    ("/mcp", handle_mcp_command),
    ("/proxy", handle_proxy_command),
    ("/last", handle_last_command),
    ("/workflow", handle_workflow_command),
    ("/debug", handle_debug_command),
)


async def _dispatch_async_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
) -> str | None:
    """Match the built-in commands with async handlers.

    Covers /help, /policy, /consolidate (alias /slice), /intake, /pack,
    /models, /mcp, /proxy, /last, and /debug. Returns the response text,
    or ``None`` when the message matches none of these commands.
    """
    response: str | None = None
    if lowered.startswith("/help") or lowered == "help":
        response = await handle_help_command(runtime, stripped, lowered, message)
    elif lowered == "/policy" or lowered.startswith("/policy "):
        response = handle_simple_command(
            runtime, message, "policy", handle_policy_command(runtime, stripped, message)
        )
    elif lowered in {"/consolidate", "consolidate", "/slice", "slice"}:
        response = await run_explicit_slice_command(runtime, message=message)
    elif lowered in {"/intake", "intake"}:
        response = await handle_intake_command(runtime, message=message)
    else:
        for prefix, handler in _PREFIX_ASYNC_COMMANDS:
            if lowered.startswith(prefix):
                result = handler(runtime, stripped, message)
                response = await result if inspect.isawaitable(result) else result
                break
    return response


async def _dispatch_builtin_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
    transport: TransportCallbacks,
) -> tuple[str | None, HostCommandResult | None]:
    """Match one built-in host shortcut command.

    Returns:
        A ``(response_text, continuation_result)`` tuple. Exactly one
        element is set when a command matched; both are ``None`` when no
        built-in command matched (the caller continues to pack commands).
        ``continuation_result`` is set when the approve path ran a
        post-approval continuation turn and returned a structured result
        rather than plain text.
    """
    # Simple static-render commands first (no awaitable handlers).
    response_text = _dispatch_simple_command(runtime, lowered, message)
    continuation_result: HostCommandResult | None = None
    if response_text is not None:
        return response_text, continuation_result

    # Commands with async handlers.
    response_text = await _dispatch_async_command(runtime, stripped, lowered, message)
    if response_text is not None:
        return response_text, continuation_result

    # Approval commands (structured continuation handling).
    return await _dispatch_approval_command(runtime, stripped, lowered, transport)


async def _dispatch_pack_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
) -> str | None:
    """Match the first pack-registered command whose prefix fits ``lowered``.

    Packs register slash commands via lisa.extensions.host_commands. The
    host owns only the dispatch mechanism and host-intrinsic commands;
    pack-owned commands live in the pack that owns the operations they
    call. Returns the handler's response text, or ``None`` when no pack
    command matched (or every matching handler returned ``None`` — the
    scan continues past a handler that declines with ``None``).
    """
    for entry in _pack_host_commands():
        for spec in entry.specs:
            if lowered.startswith(spec.prefix):
                try:
                    response_text: str | None = await spec.handler(runtime, stripped, message)
                except Exception as exc:  # pylint: disable=broad-except
                    response_text = f"Command {spec.prefix} failed: {exc}"
                if response_text is not None:
                    return response_text
    return None


_STATE_SETTLING_COMMAND_PREFIXES = (
    "/cancel",
    "/skip",
    "/retry",
    "/deny",
)


def _is_state_settling_command(lowered: str) -> bool:
    """True when the operator command settles durable workflow state.

    These commands decide pending work (recovery items, approvals) — the
    operator's answer to a paused/stalled turn. Their outcome must land in
    the thread the same way ``/approve`` lands one, or the recorded tail
    keeps replaying the question the operator already answered
    (the 2026-10-05 stale-yield loop: ``/cancel-all`` cleaned up the
    recovery items at 02:57:51Z but the thread tail still ended at
    "paused awaiting your decision", so every later turn re-yielded the
    resolved question).
    """
    return lowered.startswith(_STATE_SETTLING_COMMAND_PREFIXES)


def _record_state_settling_command(
    runtime: HostRuntime,
    *,
    message: str,
    response_text: str,
    state_change: str,
) -> None:
    """Append the thread entry for a state-settling operator command.

    Read-only commands (``/stalls``, ``/stall``, ``/help``) keep today's
    behavior — no entry, no noise. State-settling commands record the
    decision with ``operator-host-command`` provenance so context
    assembly (and the model reading the thread) sees the operator's
    answer as history, not a gap.
    """
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id=runtime.session.current_mode_id or "cmd",
        query=message,
        response_text=response_text,
        provenance={
            "source": "operator-host-command",
            "state_change": state_change,
        },
    )


async def _dispatch_host_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
    transport: TransportCallbacks,
) -> HostCommandResult | None:
    """Try each host shortcut command; return result or ``None`` to continue.

    Delegates to the individual command handlers in ``runtime.commands``.
    Each handler returns a plain text string; this function wraps it in a
    ``HostCommandResult`` and emits the transport callback.

    State-settling operator commands (``/cancel``/``/skip``/``/retry``/
    ``/deny`` and their ``-all`` variants) additionally append a thread
    entry recording the decision — the durable answer to whatever
    paused/stalled turn prompted it.
    """
    # Ensure the (possibly deferred) MCP pool is ready before any slash
    # command touches runtime.resources.mcp_tool_pool (/settings, /debug, /pack).
    # Cheap commands (/help, /quit) don't need the pool, but awaiting a
    # completed no-op task is free.
    await runtime.ensure_pool_ready()
    response_text, continuation_result = await _dispatch_builtin_command(
        runtime, stripped, lowered, message, transport
    )

    if continuation_result is not None:
        await transport.on_response(continuation_result.text)
        return continuation_result

    if response_text is not None:
        if _is_state_settling_command(lowered):
            _record_state_settling_command(
                runtime,
                message=message,
                response_text=response_text,
                state_change=(
                    "approval_denied" if lowered.startswith("/deny") else "recovery_settled"
                ),
            )
        await transport.on_response(response_text)
        return HostCommandResult(text=response_text)

    # ── Pack-registered commands ────────────────────────────────────
    pack_response = await _dispatch_pack_command(runtime, stripped, lowered, message)
    if pack_response is not None:
        if _is_state_settling_command(lowered):
            _record_state_settling_command(
                runtime,
                message=message,
                response_text=pack_response,
                state_change="recovery_settled",
            )
        await transport.on_response(pack_response)
        return HostCommandResult(text=pack_response)

    return None


# ── Approval flow (moved to ``runtime.approval``) ────────────────────────
# The identity rebuild, affirmative-resume detection, policy auto-continue,
# post-approval continuation, and paired-recovery settlement that were
# embedded here now live in :mod:`runtime.approval`. Only the dispatch calls
# above and the model-turn pending-approval intercept below remain; they
# wire the approval entry points by injecting :func:`_route_model_turn`.


# ── Model turn routing ────────────────────────────────────────────────


async def _route_model_turn(
    runtime: HostRuntime,
    message: str,
    transport: TransportCallbacks,
    *,
    task_id: str = "",
    continuation: TurnContinuation | None = None,
    extra_item_builders: list[ExtensionItemBuilder] | None = None,
) -> str | HostCommandResult:
    """Handle model-backed routing: mode dispatch and run_turn execution.

    Routing order (unified FSM — see docs/ROUTING_RFC_HANDOFF.md):
    1. Slash commands → ``kernel.ipsar.parse_route``.
    2. Natural language → check ``start_mode`` override from session
       context front matter. When set, bypass and route directly to
       the declared mode.
    3. Otherwise → go directly to ``current_mode_id`` (defaults to
       ``default``). The model hands off between modes mid-turn via
       ``transition_stage`` — no separate planner call.
    4. Execute the resolved mode turn via ``kernel.turn_runner.run_turn``.

    ``extra_item_builders`` flows through to ``build_turn_hooks`` so a
    host-supplied item builder (the editor-context bridge) reaches the
    turn's context assembly on every resume path below.
    """
    # Ensure the (possibly deferred) MCP pool is ready before building
    # turn hooks, which snapshot runtime.resources.mcp_tool_pool.
    await runtime.ensure_pool_ready()
    if continuation is not None:
        return await _execute_mode_turn(
            runtime,
            continuation.query,
            continuation.mode_id,
            continuation.query,
            transport,
            task_id=continuation.identity.task_id,
            continuation=continuation,
            extra_item_builders=extra_item_builders,
        )
    if not message.lstrip().startswith("/") and _is_task_resume_request(message):
        paused = _paused_turn_continuation(runtime.paths.thread_path)
        if paused is not None:
            return await _route_model_turn(
                runtime,
                message,
                transport,
                continuation=resume_after_pause(paused, message),
                extra_item_builders=extra_item_builders,
            )
    if message.lstrip().startswith("/"):
        mode_id, query = parse_route(message)
    else:
        resumed_mode, resumed_task_id = _resumable_task_context(runtime.paths.thread_path, message)
        if not task_id:
            task_id = resumed_task_id
        # Check for start_mode override from session-context front matter.
        # When set to a real owning mode (learn, coding, etc.), bypass and
        # go directly to the declared mode — this keeps the user pinned
        # in that mode unless they explicitly /switch. `default` is NOT a
        # real owning mode: pinning it would silently disable handoff
        # routing on every turn. Exempt `default` so that
        # start_mode: default means "let the model decide via handoff."
        start_mode = runtime.start_mode_id
        if start_mode and start_mode != "default" and runtime.session.current_mode_id == start_mode:
            mode_id = start_mode
            query = message
            logger.debug(
                "start_mode bypass: routing directly to mode %r (handoff skipped)",
                start_mode,
            )
        else:
            # Natural language: go directly to current_mode_id. The model
            # can hand off to a better-suited mode mid-turn by calling
            # transition_stage. No separate planner call needed.
            mode_id = runtime.session.current_mode_id or "default"
            # current_mode_id resets per process; a resume after a paused turn must reach its owner.
            if _is_task_resume_request(message):
                mode_id = (
                    resumed_mode
                    or paused_turn_owner_mode(thread_path=runtime.paths.thread_path)
                    or mode_id
                )
            query = message
            logger.debug(
                "route_model_turn: natural language -> mode=%s (start_mode=%s)",
                mode_id,
                runtime.start_mode_id,
            )

    return await _execute_mode_turn(
        runtime,
        message,
        mode_id,
        query,
        transport,
        task_id=task_id,
        extra_item_builders=extra_item_builders,
    )


# ── Turn execution ────────────────────────────────────────────────────


def _session_budget_blocked(runtime: HostRuntime) -> str | None:
    """Return the budget-exhausted block message, or ``None`` to proceed.

    Pre-turn session budget gate. When the session has declared a
    ``max_session_tokens`` cap and the accumulated processed tokens (uncached
    input plus output across prior turns) already meet or exceed it, block the
    turn before any model call — the user must start a fresh session or
    raise the cap rather than continue burning tokens. Cached prefix reads are
    excluded so retaining useful context does not consume the work budget.
    """
    max_session_tokens = runtime.session_plan.manifest.execution.max_session_tokens
    tokens = runtime.session.tokens
    used = max(0, tokens.session_total_tokens - tokens.session_cached_tokens)
    if max_session_tokens and used >= max_session_tokens:
        message_text = (
            f"Session token budget exhausted ({used:,} / {max_session_tokens:,} tokens). "
            "Start a new session (or raise `max_session_tokens` in your session context) "
            "to continue."
        )
        logger.warning(
            "session token budget exhausted: used=%d cap=%d; blocking turn",
            used,
            max_session_tokens,
        )
        return message_text
    return None


def _log_operator_exit(result: TurnResult, mode_id: str) -> None:
    """Log the operator exit path (pause/stop/terminate) for one turn.

    The kernel produces ``operator_exit`` when an operator signal or a cap
    path produced a PauseAction / StopAction / TerminateAction:

    - ``paused``: the transport holds the turn open; the next input re-enters
      the same turn against the preserved WorkflowPosition. The transport
      decides how to surface this (e.g. a "paused" indicator).
    - ``stopped``: the transport returns the prompt. The next message is a
      fresh turn against the preserved position.
    - ``terminated``: the transport returns the prompt. The workflow is
      marked complete; the next message is a fresh turn with no workflow.

    The kernel handles WorkflowPosition marking; the host only needs to
    surface the exit to the transport.
    """
    _operator_exit = getattr(result, "operator_exit", "")
    if not _operator_exit:
        return
    if _operator_exit == "paused":
        logger.info("handle_message: turn paused by operator (mode=%s)", mode_id)
    elif _operator_exit == "stopped":
        logger.info("handle_message: turn stopped by operator (mode=%s)", mode_id)
    elif _operator_exit == "terminated":
        logger.info("handle_message: turn terminated by operator (mode=%s)", mode_id)


def _build_thread_provenance(resolved_model: Any, *, operator_exit: str = "") -> dict[str, Any]:
    """Build the thread-entry provenance dict for one completed turn."""
    provenance: dict[str, Any] = {
        "source": "observed",
        "finish_reason": "unknown",
        "model_id": resolved_model.model_id,
    }
    if operator_exit:
        provenance["operator_exit"] = operator_exit
    return provenance


def _resumable_task_id(thread_path: Any, message: str) -> str:
    """Reuse a task only for a recorded unfinished turn and a matching reply."""
    return _resumable_task_context(thread_path, message)[1]


#: Operator opt-in that declares a task diagnostic at creation. The ad-hoc
#: "why do the tests fail?" turn inside a checked-commit mode would otherwise
#: end in a blocked completion, and an operator who then waves it through by
#: hand produces an unaudited exception. A declared, stamped exception is the
#: better trade. This is deliberately an operator surface only: the model has
#: no path to this value, so the exception can never be self-granted.
#:
#: The syntax is ``[task_goal: <value>]``, parsed by ``_TASK_GOAL_MARKER_RE``
#: below — not by the literal string that used to live here. Comparing against
#: a literal made detection stricter than the stripping regex, so a
#: whitespace variant was silently removed from the message while the turn
#: still ran as ``implementation`` and blocked.

#: Matches any ``[task_goal: <value>]`` declaration, capturing the value.
#: One pattern serves both detection and stripping so the two can never
#: disagree: a message whose marker is removed has necessarily been read.
#: ``docs/CODING_WORKFLOW.md`` documents whitespace tolerance, so the value
#: group is trimmed rather than matched literally.
_TASK_GOAL_MARKER_RE = re.compile(r"\[\s*task_goal\s*:\s*([^\]]*?)\s*\]", re.IGNORECASE)


def _declared_task_goal(message: str) -> str:
    """Return the goal named by an operator ``[task_goal: …]`` marker, or ``""``.

    The marker is whitespace-tolerant and case-insensitive, matching the
    documented syntax: ``[task_goal: diagnostic]``, ``[task_goal:diagnostic]``,
    ``[ task_goal : Diagnostic ]`` and interior runs of spaces all resolve to
    ``diagnostic``. An unrecognized value returns that value so the caller can
    fail closed on it.
    """
    match = _TASK_GOAL_MARKER_RE.search(message)
    if match is None:
        return ""
    return " ".join(match.group(1).split()).lower()


def _operator_declared_diagnostic(message: str) -> bool:
    """Return True when the operator declared this task diagnostic at creation."""
    return _declared_task_goal(message) == "diagnostic"


def _strip_task_goal_marker(message: str) -> str:
    """Remove the operator's ``[task_goal: …]`` declaration from the message.

    The marker is an operator directive, not content. Leaving it in would
    replay it to the model every turn, persist it in the thread entry's
    ``query``, and make it look like part of the request in a later resume.

    Args:
        message: The raw operator message.

    Returns:
        The message with any ``[task_goal: <value>]`` declaration removed and
        surrounding whitespace collapsed.
    """
    stripped = _TASK_GOAL_MARKER_RE.sub(" ", message)
    return " ".join(stripped.split())


def _resolve_task_goal(message: str) -> str:
    """Resolve the task goal at creation: ``diagnostic`` or ``implementation``.

    The operator escape hatch (F1 Option B, v1): the ad-hoc "why do the tests
    fail?" turn inside a checked-commit mode would otherwise end in a blocked
    completion, and an operator who then waves it through by hand produces an
    unaudited exception. This is an OPERATOR declaration, never a model one,
    so the exception can never be self-granted.

    A delegate child declares its own goal through ``DelegationSpec``, so this
    operator surface is the parent-turn path only.

    An unrecognized value fails closed to ``implementation``: the marker is
    stripped either way, so accepting an unknown goal would give the operator a
    silent no-op that looks like a declaration. Only a value this function
    understands changes the contract, and the block message names the accepted
    syntax when the turn then blocks.
    """
    declared = _declared_task_goal(message)
    if not declared:
        return "implementation"
    if declared != "diagnostic":
        logger.warning(
            "unrecognized task_goal declaration %r; only 'diagnostic' changes the "
            "completion contract, so this turn keeps the implementation default",
            declared,
        )
    return "diagnostic" if declared == "diagnostic" else "implementation"


def _proxy_protocol_for_provider(runtime: HostRuntime, provider: str) -> str | None:
    """Return the proxy wire protocol serving ``provider``, or ``None``.

    Asks the provider-gateway seam rather than holding a host-side provider
    list: the gateway implementation is the only component that knows whether
    it reads a proxy route at all. A provider with no proxy protocol cannot be
    served by a proxy, so route resolution for it is a no-op.
    """
    from runtime.mcp_pool import _proxy_protocol_resolver

    return _proxy_protocol_resolver(runtime)(provider)


async def _respawn_dead_proxy_for_provider(runtime: HostRuntime, provider: str) -> None:
    """Give a dead proxy serving ``provider`` one bounded restart attempt.

    The sync route accessor the gateway factories use cannot await, so it can
    only drop a dead route and let the call go direct. The turn boundary is
    the place that can await, which makes it the production caller of
    :meth:`RuntimeResources.ensure_proxy_route` — and therefore of the pool's
    supervised restart budget, single-flight guard, and permanent-failure
    ceiling.

    Best-effort by design: a proxy that cannot be restarted, has exhausted its
    budget, or serves no provider here leaves the turn on the direct endpoint.
    Proxy routing is an optimization, never a correctness requirement.

    Args:
        runtime: The host runtime owning the proxy pool.
        provider: Provider gateway id resolved for this turn.
    """
    protocol = _proxy_protocol_for_provider(runtime, provider)
    if protocol is None:
        return
    if runtime.resources.proxy_pool is None:
        return
    try:
        await runtime.resources.ensure_proxy_route(provider, protocol)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "proxy restart attempt for provider %r failed; continuing on the direct endpoint",
            provider,
            exc_info=True,
        )


async def _emit_model_call_stage(
    runtime: HostRuntime,
    transport: TransportCallbacks,
    mode: Any,
    resolved_model: Any,
    turn_index: int,
) -> None:
    """Announce the turn's model call, restoring its proxy route first.

    ``turn_index`` rides the payload so transports can label their output with
    the same turn number the model briefing and the thread entry carry — one
    source of truth across the model's self-address, persistence, and
    operator-facing rendering. ``proxy_route`` rides it so a transport can show
    whether this provider's requests flow through a proxy pack or go direct;
    that view is informational only, because the authoritative routing
    decision happens in the gateway client factories.

    A proxy that died between turns is restored before the gateway client is
    built for this turn, which is the only async seam that can await the
    bounded restart (see :func:`_respawn_dead_proxy_for_provider`).

    Args:
        runtime: The host runtime.
        transport: The turn's transport callbacks.
        mode: The mode config owning this turn.
        resolved_model: The model resolved for this turn.
        turn_index: This turn's global index.
    """
    await _respawn_dead_proxy_for_provider(runtime, resolved_model.provider)
    await transport.on_stage(
        "model_call",
        {
            "mode_id": mode.mode_id,
            "model_id": resolved_model.model_id,
            "provider": resolved_model.provider,
            "turn_index": turn_index,
            "proxy_route": proxy_route_view(runtime, resolved_model.provider),
        },
    )


def _is_task_resume_request(message: str) -> bool:
    """Recognize explicit task continuation without treating it as approval."""
    lowered = " ".join(message.strip().lower().rstrip(".!?").split()).replace("'", "")
    return is_affirmative_resume(lowered) or any(
        lowered.startswith(phrase)
        for phrase in (
            "pick up where we left off",
            "lets pick up where we left off",
            "let us pick up where we left off",
            "continue where we left off",
            "resume where we left off",
        )
    )


def _kernel_loop_provenance(
    runtime: HostRuntime, result: TurnResult, resumed: TurnContinuation | None
) -> dict[str, str]:
    """Link resumed loops and keep a paused loop resumable in place, not re-routed."""
    paused_at = result.continuation
    if paused_at is None or paused_at.requires_approval or runtime.paths.thread_path is None:
        return {"kernel_turn_id": resumed.turn_id if resumed is not None else ""}
    return {
        "kernel_turn_id": paused_at.turn_id,
        "turn_continuation_ref": persist_continuation(runtime.paths.thread_path, paused_at),
    }


def _paused_turn_continuation(thread_path: Any) -> TurnContinuation | None:
    """Return the latest turn's saved loop when it paused at an iteration boundary."""
    entries = load_recent_thread_entries(thread_path=thread_path, limit=1)
    if not entries:
        return None
    provenance = entries[-1].get("provenance") or {}
    if provenance.get("operator_exit") != "paused":
        return None
    try:
        saved = load_continuation(thread_path, provenance)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("paused turn is not resumable in place: %s", exc)
        return None
    return saved if saved is not None and not saved.requires_approval else None


def _resumable_task_context(thread_path: Any, message: str) -> tuple[str, str]:
    """Recover the owner and task ID across repeated explicit resume turns."""
    if not _is_task_resume_request(message):
        return "", ""
    closed_task_ids: set[str] = set()
    for entry in reversed(load_recent_thread_entries(thread_path=thread_path, limit=50)):
        provenance = entry.get("provenance") or {}
        source = provenance.get("source")
        task_id = str(provenance.get("task_id") or "")
        if (
            entry.get("yielded")
            or source in {"host-approved", "host-approved-failed"}
            or provenance.get("operator_exit") == "paused"
            or provenance.get("pending_approval_request_id")
            or provenance.get("block_origin") == "task_completion"
        ):
            if task_id in closed_task_ids:
                break
            owner_mode = (
                str(entry.get("mode_id") or "")
                if source in {"host-approved", "host-approved-failed"}
                else str(entry.get("handoff_to") or entry.get("mode_id") or "")
            )
            return owner_mode, task_id
        if task_id:
            closed_task_ids.add(task_id)
        if not _is_task_resume_request(str(entry.get("query") or "")):
            break
    return "", ""


async def _handle_pending_approval(
    runtime: HostRuntime,
    result: TurnResult,
    *,
    mode_id: str,
    query: str,
    resolved_model: Any,
    turn_index: int,
    task_id: str,
    transport: TransportCallbacks,
) -> HostCommandResult | None:
    """Persist a turn paused for approval, then auto-continue or return the approval payload."""
    pending_approval = getattr(result, "pending_approval", None)
    if pending_approval is None:
        return None
    # A turn can pause for approval after a handoff to a different mode.
    # Update the live session mode before returning so the completion and
    # approval-continuation flows keep the owning mode instead of reverting
    # to the entry mode on the next turn.
    if result.handoff_to:
        runtime.session.current_mode_id = result.handoff_to
    # Persist the paused turn so the post-approval continuation can recover its request.
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id=mode_id,
        query=query,
        response_text=result.response_text,
        provenance={
            **_build_thread_provenance(resolved_model),
            "task_id": task_id,
            "pending_approval_request_id": pending_approval.request_id,
            "pending_approval_tool_call_id": result.pending_approval_tool_call_id,
            "pending_approval_recovery_item_id": result.pending_approval_recovery_item_id,
            "kernel_turn_id": result.continuation.turn_id if result.continuation else "",
            "turn_continuation_ref": persist_continuation(
                runtime.paths.thread_path, result.continuation
            )
            if result.continuation is not None
            else "",
        },
        effect_receipts=_serialize_effect_receipts(result),
        handoff_to=result.handoff_to or "",
        turn=turn_index,
        turn_input={"calls": result.model_requests} if result.model_requests else None,
        transcript=result.transcript,
    )
    approved_result = (
        None
        if result.continuation is not None
        else await maybe_auto_continue_approval(
            runtime,
            pending_approval,
            transport,
            route_model_turn=_route_model_turn,
        )
    )
    if approved_result is not None:
        return approved_result
    approval_data = build_approval_data(runtime, pending_approval)
    response_text = result.response_text
    if result.continuation is not None and result.continuation.reason == "auto_approval_cap":
        approval_data["suspension_reason"] = "auto_approval_cap"
        response_text = (
            "Automatic approval allowance reached. The turn and pending tool batch are preserved; "
            "approval resumes this action and grants another allowance.\n\n"
            f"{response_text}"
        )
    # Panel transports render a keystroke panel from ``data["approval"]``.
    # Plain transports only get the text — append the installed command
    # hints so the operator still has a copy-pasteable escape. Runtime owns
    # the command vocabulary; the kernel's neutral prompt never mentions it.
    request_id = str(approval_data.get("request_id") or "")
    hints = [
        f"/approve {request_id}" if request_id else "/approve",
        f"/deny {request_id}" if request_id else "/deny",
    ]
    return HostCommandResult(
        text=f"{response_text}\n\nRun {' or '.join(hints)}.",
        data=approval_data,
    )


def _serialize_effect_receipts(result: TurnResult) -> list[dict[str, str]]:
    """Serialize the effect-receipt ledger for the thread entry.

    The thread entry is the canonical source of receipts; the
    ``turn.completed`` event links to it by turn number rather than
    duplicating the receipt list.
    """
    return [
        {
            "capability_id": str(getattr(r, "capability_id", "")),
            "effect": str(getattr(r, "effect", "")),
            "artifact_id": str(getattr(r, "artifact_id", "") or ""),
            "mode_id": str(getattr(r, "mode_id", "") or ""),
        }
        for r in (result.effect_receipts or [])
    ]


def _record_operation_error_thread_entry(
    runtime: HostRuntime,
    run_context: dict[str, Any],
    mode: Any,
    query: str,
    *,
    message: str,
    exc: OperationExecutionError,
    response_text: str,
    turn_index: int,
) -> None:
    """Persist the thread entry for a turn that failed with an operation error."""
    run_context["operation_error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id=mode.mode_id,
        query=query or message,
        response_text=response_text,
        provenance={
            "source": "system-verified",
            "finish_reason": "error",
            "task_id": run_context.get("task_id", ""),
        },
        turn=turn_index,
    )


def _update_completion_context(
    run_context: dict[str, Any],
    mode: Any,
    query: str,
    *,
    message: str,
    result: TurnResult,
    resolved_model: Any,
) -> None:
    """Update the run context's completion block with the turn's outcome."""
    completion_context = run_context.get("completion")
    if not isinstance(completion_context, dict):
        completion_context = {}
    completion_context.update(
        {
            "mode_id": mode.mode_id,
            "query": query or message,
            "response_preview": result.response_text[:240],
            "response_chars": len(result.response_text),
            # How the response was produced: "" (normal answer) or
            # "reasoning_tail" (no visible answer; the surfaced text is a
            # labeled quote of the tail of the model's reasoning). The full
            # reasoning rides the thread entry's ``reasoning_text`` so the
            # exhaustive log keeps it; playback surfaces bound/redact it.
            "content_source": str(getattr(result, "content_source", "") or ""),
            "tool_call_count": coerce_non_negative_int(result.tool_call_count),
            "iteration_count": coerce_non_negative_int(result.iteration_count),
            "prompt_tokens": coerce_non_negative_int(result.prompt_tokens),
            "completion_tokens": coerce_non_negative_int(result.completion_tokens),
            "total_tokens": coerce_non_negative_int(result.total_tokens),
            "resolved_model": {
                "model_id": resolved_model.model_id,
                "provider": resolved_model.provider,
                "reasoning_model": resolved_model.reasoning_model,
            },
        }
    )
    run_context["completion"] = completion_context


def _emit_turn_completed_event(
    mode_id: str,
    result: TurnResult,
    *,
    turn_index: int,
    task_id: str,
) -> None:
    """Emit the kernel turn's completion event for one finished turn.

    Emitted only after the thread entry has been appended, so observing this
    event is evidence that the turn is durable: a consumer can address the
    entry by ``turn_index`` and find it. Emitting before the append let a
    consumer treat completion as proof of persistence when the entry did not
    exist yet.

    ``turn_index`` and ``task_id`` are the locators that make the claim
    checkable. The effect-receipt ledger still lives only in the thread entry
    (the canonical, turn-addressable source) and is not duplicated here.
    """
    emit_structured_event(
        logger,
        level=logging.INFO,
        event="turn.completed",
        message=(f"turn.completed: mode={mode_id} handoff={result.handoff_to or ''}"),
        **build_turn_completed_payload(
            mode_id=mode_id,
            result=result,
            coerce_int=coerce_non_negative_int,
            turn_index=turn_index,
            task_id=task_id,
        ),
    )


def _apply_turn_decision(runtime: HostRuntime, mode_id: str, result: TurnResult) -> None:
    """Classify the turn outcome and persist a mid-turn mode handoff.

    Centralized loop-decision classification: the host does not iterate
    the loop itself — ``run_turn`` does that internally — but this
    classification ensures the host's post-turn handoff and persistence
    logic is consistent with the same contract the kernel uses (the
    uniform three-outcome CONTINUE / HANDOFF / FINISH contract).

    Unified FSM routing: when the model called ``transition_stage`` with
    a mode_id target, update ``current_mode_id`` so the next turn starts
    in the new mode (see docs/ROUTING_RFC_HANDOFF.md).
    """
    _decision = evaluate_turn_decision(
        has_tool_calls=bool(result.tool_call_count),
        has_transition_request=bool(result.handoff_to),
        transition_target=result.handoff_to or None,
        is_terminal=not bool(result.handoff_to),
        budget_exhausted=False,
        has_content=bool(result.response_text),
    )
    logger.debug(
        "turn decision: %s (target=%s reason=%s)",
        _decision.decision.value,
        _decision.target,
        _decision.reason,
    )
    if _decision.decision is TurnDecision.HANDOFF and _decision.target:
        runtime.session.current_mode_id = _decision.target
        logger.debug(
            "mode handoff: %s -> %s",
            mode_id,
            _decision.target,
        )


def _log_turn_start(mode: Any, resolved_model: Any, query: str) -> None:
    """Emit the turn-start log pair (info + debug) for the mode turn."""
    logger.info(
        "turn started: mode=%s model=%s query=%s",
        mode.mode_id,
        resolved_model.model_id,
        query[:80],
    )
    logger.debug(
        "turn execution starting: mode=%s model=%s provider=%s",
        mode.mode_id,
        resolved_model.model_id,
        resolved_model.provider,
    )


async def _resolve_live_workflow_state_for_turn(runtime: HostRuntime) -> Any | None:
    """Snapshot durable approval state for the recorded blockers in the tail.

    Async seam for the synchronous thread digest: collect the approval
    request ids the recorded thread tail names, then read each from the
    live approval store so the digest can annotate blockers the store
    already settled as superseded (the 2026-10-05 stale-yield loop: a
    recorded "awaiting your decision" persisted after the continuation
    turn died, and the model re-yielded the resolved question every turn).
    """
    from runtime.thread_digest import recorded_approval_request_ids, resolve_live_workflow_state

    return await resolve_live_workflow_state(
        runtime,
        request_ids=recorded_approval_request_ids(runtime.paths.thread_path),
    )


def _consume_task_goal_declaration(message: str, query: str) -> tuple[str, str, str]:
    """Resolve this turn's declared goal and strip the declaration.

    The operator's ``[task_goal: …]`` marker is a directive, not content: it
    selects the goal for this turn and is then removed, so it is never
    replayed to the model as part of the request nor persisted in the thread
    entry's ``query``.

    Args:
        message: The raw operator message for this turn.
        query: The routed query text (may be empty).

    Returns:
        ``(task_goal, message, query)`` with the declaration removed from both
        the message and the query.
    """
    task_goal = _resolve_task_goal(message)
    if task_goal == "implementation" and not _TASK_GOAL_MARKER_RE.search(message):
        return task_goal, message, query
    return (
        task_goal,
        _strip_task_goal_marker(message),
        _strip_task_goal_marker(query) if query else query,
    )


async def _execute_mode_turn(
    runtime: HostRuntime,
    message: str,
    mode_id: str,
    query: str,
    transport: TransportCallbacks,
    *,
    task_id: str = "",
    continuation: TurnContinuation | None = None,
    extra_item_builders: list[ExtensionItemBuilder] | None = None,
) -> HostCommandResult | str:
    """Resolve mode, build context, and run the turn through the kernel.

    This is the main turn-execution path: it resolves the mode config from
    the workflow catalog, resolves the model, builds the turn context,
    constructs ``TurnHooks`` via ``build_turn_hooks``, calls ``run_turn``,
    persists the result, and updates token counters.

    ``extra_item_builders`` — optional host-supplied ``ExtensionItemBuilder``
    instances (the VS Code sidecar's editor-context bridge) composed into
    the turn's context assembly. Keyword-only so existing positional
    callers and test stubs are unaffected.
    """
    catalog = get_workflow_catalog()
    mode = catalog.get(mode_id) or catalog["default"]
    task_goal, message, query = _consume_task_goal_declaration(message, query)
    execution_policy = resolve_execution_policy(
        default_policy=runtime.config.execution_policy,
        session_request=runtime.session_plan.manifest.execution,
        scope_ids=tuple(
            scope_id
            for scope_id in (
                str(getattr(mode, "mode_provider_id", "") or ""),
                (
                    f"{getattr(mode, 'mode_provider_id', '')}:{mode.mode_id}"
                    if getattr(mode, "mode_provider_id", "")
                    else mode.mode_id
                ),
            )
            if scope_id
        ),
    )

    preferred_model_id = preferred_model_id_for_route(
        raw_message=message,
        mode_id=mode.mode_id,
        manifest=runtime.session.manifest,
    )
    set_runtime_route_preview(
        runtime,
        mode_id=mode.mode_id,
        preferred_model_id=preferred_model_id,
    )

    # Check for unsupported mode gap.
    if mode_support_gap(mode) is not None:
        return _handle_unsupported_mode(runtime, message, mode, query)

    # Pre-turn session budget gate.
    budget_block = _session_budget_blocked(runtime)
    if budget_block is not None:
        return budget_block

    # Resolve the model for this turn.
    from runtime.commands import _resolve_model as _resolve_model_cmd

    resolved_model = _resolve_model_cmd(runtime, preferred_model_id)

    # Compute the global turn index at turn start (before the turn runs)
    # so the model can be annotated with its turn number and the
    # pre-computed value can be threaded through to ``append_thread_entry``.
    # One source of truth: the number the model sees in the briefing is the
    # number stamped on the thread entry — no off-by-one between display and
    # persistence.
    from runtime.thread import next_turn_index

    turn_index = next_turn_index(thread_path=runtime.paths.thread_path)
    task_id = task_id or _resumable_task_id(runtime.paths.thread_path, message) or uuid4().hex

    # Resolve the live approval state for recorded blockers before the
    # turn starts. The digest builders are synchronous; this is the async
    # seam where the durable stores are readable. A recorded blocker the
    # store already settled (consumed / denied / cancelled) is then
    # annotated as superseded instead of replayed as a live prompt —
    # the 2026-10-05 stale-yield loop.
    runtime.session.live_workflow_state = await _resolve_live_workflow_state_for_turn(runtime)

    # Build the turn context dict.
    run_context = build_turn_context(
        raw_message=message,
        mode_id=mode.mode_id,
        query=query or message,
        preferred_model_id=preferred_model_id,
        resolved_model=resolved_model,
        mode=mode,
    )
    run_context.update(turn_index=turn_index, task_id=task_id, task_goal=task_goal)

    _set_turn_context(runtime, run_context)

    # Emit stage callback.
    await _emit_model_call_stage(runtime, transport, mode, resolved_model, turn_index)

    # Build turn hooks and execute. Host-supplied item builders (the editor
    # context bridge) are composed in alongside the installed-pack builder.
    hooks = build_turn_hooks(
        runtime,
        context=run_context,
        transport=transport,
        extra_item_builders=extra_item_builders,
    )
    _log_turn_start(mode, resolved_model, query or message)
    try:
        result: TurnResult = await run_turn(
            mode_id=mode.mode_id,
            query=query or message,
            hooks=hooks,
            preferred_model_id=preferred_model_id,
            execution_policy=execution_policy,
            continuation=continuation,
        )
    except OperationExecutionError as exc:
        response_text = format_operation_error(exc)
        _record_operation_error_thread_entry(
            runtime,
            run_context,
            mode,
            query,
            message=message,
            exc=exc,
            response_text=response_text,
            turn_index=turn_index,
        )
        await transport.on_error(f"Operation error: {exc}", exc)
        runtime.session.operator_signal_value = ""
        return response_text

    # Clear the operator signal after the turn ends so the next turn starts
    # clean (the signal is one-shot; the transport re-sets it on the next
    # key press).
    runtime.session.operator_signal_value = ""

    if result.transitioned_to:
        runtime.session.current_stage_id = result.transitioned_to
    _apply_turn_usage(
        runtime,
        usage_since_resume(result, continuation),
        model_id=resolved_model.model_id,
        mode_id=mode.mode_id,
        stage_id=runtime.session.current_stage_id or "-",
    )

    # ── Operator exit handling (pause/stop/terminate) ──────────────────
    # The kernel produces ``operator_exit`` when an operator signal or a cap
    # path produced a PauseAction / StopAction / TerminateAction. The
    # kernel handles WorkflowPosition marking; the host only needs to
    # surface the exit to the transport.
    _log_operator_exit(result, mode.mode_id)

    # ── Pending-approval intercept (policy-driven auto-approve + payload) ──
    # ``run_turn`` pauses the turn before the side effect and exposes the
    # canonical invocation on ``result.pending_approval``. The policy service
    # has already pre-decided it: a policy ``allow`` minted a receipt (record
    # state = approved), while ``ask`` left it pending. Drive the two paths:
    #
    # - already-approved -> run the normal approve machinery in-process (up to
    #   a per-turn cap) so a policy-contained action completes unattended,
    #   WITHOUT waiting on the operator;
    # - still-pending     -> attach a structured ``data["approval"]`` payload
    #   so a panel-capable transport (the CLI) renders a keystroke panel
    #   instead of the bare "run /approve <uuid>" text.
    pending_approval_result = await _handle_pending_approval(
        runtime,
        result,
        mode_id=mode.mode_id,
        query=query or message,
        resolved_model=resolved_model,
        turn_index=turn_index,
        task_id=task_id,
        transport=transport,
    )
    if pending_approval_result is not None:
        return pending_approval_result

    # Update completion context.
    _update_completion_context(
        run_context,
        mode,
        query,
        message=message,
        result=result,
        resolved_model=resolved_model,
    )
    _set_turn_context(runtime, run_context)

    # Audit: emit the kernel turn's completion event. This runs after the
    # thread entry is appended (below) so the event is evidence of durability
    # rather than a claim made before persistence.

    # Persist the thread entry.
    provenance: dict[str, Any] = _build_thread_provenance(
        resolved_model, operator_exit=str(getattr(result, "operator_exit", "") or "")
    )
    provenance["task_id"] = task_id
    provenance.update(_kernel_loop_provenance(runtime, result, continuation))
    if result.block_origin:
        provenance["block_origin"] = result.block_origin

    # Serialize the effect-receipt ledger and mode/stage transitions into
    # the thread entry so the workflow view can be derived losslessly
    # from the thread alone — no separate mutable position store needed.
    # The thread entry is the canonical source of receipts; the
    # ``turn.completed`` event links to it by turn number rather than
    # duplicating the receipt list.
    receipts = _serialize_effect_receipts(result)

    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id=mode.mode_id,
        query=query or message,
        response_text=result.response_text,
        content_source=str(getattr(result, "content_source", "") or ""),
        reasoning_text=str(getattr(result, "reasoning_text", "") or ""),
        provenance=provenance,
        effect_receipts=receipts,
        handoff_to=result.handoff_to or "",
        transitioned_to=result.transitioned_to or "",
        turn=turn_index,
        yielded=result.yielded is not None,
        yield_question=result.yielded.question if result.yielded else "",
        turn_input={"calls": result.model_requests}
        if result.model_requests
        else run_context.get("turn_input")
        if isinstance(run_context.get("turn_input"), dict)
        else None,
        # Thread → memory provenance: memory records minted by this turn's
        # auto-capture (stamped onto run_context by the turn-complete hook).
        memory_ids=run_context.get("memory_ids")
        if isinstance(run_context.get("memory_ids"), list)
        else None,
        transcript=result.transcript,
    )

    # Audit: emit the kernel turn's completion event, now that the entry the
    # event points at exists.
    _emit_turn_completed_event(
        mode.mode_id,
        result,
        turn_index=turn_index,
        task_id=task_id,
    )

    # ── Centralized loop decision classification ────────────────────
    # Classify the turn's outcome using the uniform three-outcome contract
    # (CONTINUE / HANDOFF / FINISH) and persist a mid-turn handoff when the
    # model called transition_stage with a mode_id target.
    _apply_turn_decision(runtime, mode.mode_id, result)

    # Emit debug context if debug mode is on.
    if runtime.config.debug:
        await transport.on_debug_context(run_context)

    # ── Structured yield remediation → operator command hints ────────
    # The kernel emits verbs + target ids on a yield (see
    # ``kernel.models.YieldRemediation``); the runtime renders the installed
    # pack command syntax from the discovered host-command registry. A verb
    # with no matching installed command renders nothing — the pack that
    # would own it is not installed.
    remediation_payload = _render_yield_remediation(runtime, result.yielded)

    return HostCommandResult(
        text=_with_yield_remediation(result, remediation_payload["remediation"]),
        operator_exit=getattr(result, "operator_exit", ""),
        # Surface the model's ask-the-user terminal action structurally: a
        # non-empty question means the turn ended in a yield — the model
        # asked the operator something instead of finishing. The question
        # rides both the typed field (one-shot hosts exit non-zero) and
        # ``data["yielded"]`` so panel transports can render it inline. The
        # context and target mode ride data so hosts can render a formatted
        # yield panel (why the turn stopped, where a reply resumes) without
        # parsing prose.
        yielded_question=(result.yielded.question if result.yielded else ""),
        data=_yield_payload(result.yielded, remediation_lines=remediation_payload["remediation"])
        if result.yielded
        else None,
    )


def _with_yield_remediation(result: Any, commands: list[str]) -> str:
    """Return the response text, footnoted with yield escape commands.

    Extracted from the turn pipeline to keep that function within its
    statement budget; the behavior lives in :func:`_append_yield_remediation`.
    """
    if not result.yielded:
        return result.response_text
    return _append_yield_remediation(result.response_text, commands)


def _render_yield_remediation(runtime: HostRuntime, yielded: Any) -> dict[str, Any]:
    """Render kernel remediation verbs as installed command lines.

    Returns ``{"remediation": [line, ...]}`` (empty list when there is
    nothing to render — no remediation on the yield, or no installed
    command matches a verb). Carried in ``data`` so panel transports can
    render buttons/links instead of parsing prose, and appended to the
    response text (see :func:`_append_yield_remediation`) so a plain-text
    transport surfaces it too.
    """
    remediation = getattr(yielded, "remediation", None)
    if remediation is None:
        return {"remediation": []}
    commands: list[str] = []
    for action in remediation.actions:
        prefix = _command_prefix_for_verb(runtime, str(action.verb))
        if not prefix:
            continue
        commands.append(f"{prefix} {action.target_id}")
    return {"remediation": commands}


def _append_yield_remediation(text: str, commands: list[str]) -> str:
    """Append the operator's escape commands below a yielded turn's question.

    A stall yield can only be cleared by a slash command. The remediation
    lines existed only in the ``data`` payload, so a transport that renders
    text alone showed "A previous step stalled and needs your input to
    continue" with no way to act — and a plain-sentence reply cannot clear a
    stall, because the affirmative-resume path only fires when an approval is
    pending.

    Observed cost: turns 486-488 of the recorded session yielded the same
    stall three times while the operator answered "approved", "proceed", and
    "let's pick up where we left off" — every reply correct, none able to
    escape, because nothing told them `/skip <id>` was the verb.

    Follows the completion-block convention: appended as a footnote so the
    model's own prose is never replaced, and idempotent so re-reading a
    thread entry that already carries the footnote does not double it.
    """
    if not commands:
        return text
    block = "To continue, use: " + ", ".join(f"`{line}`" for line in commands)
    if block in text:
        return text
    return f"{text}\n\n{block}" if text.strip() else block


def _yield_payload(yielded: Any, *, remediation_lines: list[str]) -> dict[str, Any]:
    """Build the ``data`` payload for a yielded turn.

    The question mirrors the typed ``yielded_question`` field for transports
    that read ``data``, and the context / target mode ride beside it so hosts
    can render a formatted yield — why the turn stopped and where a reply
    resumes — without parsing prose.
    """
    return {
        "yielded": yielded.question,
        "yield_context": yielded.context,
        "yield_mode_id": yielded.target_mode_id,
        "remediation": remediation_lines,
    }


def _command_prefix_for_verb(runtime: HostRuntime, verb: str) -> str:
    """Return the installed slash prefix for a remediation verb, or "".

    Resolved from pack-registered host commands (the recovery pack owns
    ``/stalls``/``/stall``/``/retry``/``/skip``/``/cancel``); host-intrinsic
    verbs (approve/deny) fall back to the host vocabulary. A verb the
    installed packs do not expose renders nothing.
    """
    verb_to_prefix: dict[str, str] = {}
    for entry in _pack_host_commands():
        for spec in entry.specs:
            verb_to_prefix.setdefault(spec.prefix.lstrip("/").lower(), spec.prefix)
    verb_to_prefix.setdefault("approve", "/approve")
    verb_to_prefix.setdefault("deny", "/deny")
    # ``inspect`` maps to the one-item detail command; the list form would
    # not accept the target id this renderer appends.
    verb_to_prefix.setdefault("inspect", verb_to_prefix.get("stall", ""))
    return verb_to_prefix.get(verb, "")


def _handle_unsupported_mode(
    runtime: HostRuntime,
    message: str,
    mode: Any,
    query: str,
) -> str:
    """Handle a mode that the host cannot run."""
    logger.info(
        "unsupported mode %s (gap=%s)",
        mode.mode_id,
        mode_support_gap(mode),
    )
    unsupported_context: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_message": message,
        "mode_id": mode.mode_id,
        "mode_kind": mode.mode_kind,
        "query": query,
        "handoff": "unsupported mode",
        "unsupported_requirement": mode_support_gap(mode),
    }
    _set_turn_context(runtime, unsupported_context)
    response_text = format_unsupported_mode(mode)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id=mode.mode_id,
        query=query,
        response_text=response_text,
    )
    return response_text


# ── Helpers ───────────────────────────────────────────────────────────


def _set_turn_context(runtime: HostRuntime, context: dict[str, Any]) -> None:
    """Set the runtime's last-turn context from a dict.

    Converts the dict to a ``TurnContext`` dataclass. Only known fields are
    mapped; extra fields are silently dropped.
    """
    from runtime.runtime import TurnContext

    runtime.session.last_turn_context = TurnContext(
        timestamp_utc=str(context.get("timestamp_utc") or ""),
        raw_message=str(context.get("raw_message") or ""),
        mode_id=str(context.get("mode_id") or "default"),
        mode_kind=str(context.get("mode_kind") or ""),
        query=str(context.get("query") or ""),
        preferred_model_id=str(context.get("preferred_model_id") or ""),
        resolved_model=context.get("resolved_model")  # type: ignore[arg-type]
        if isinstance(context.get("resolved_model"), dict)
        else {},
        handoff=str(context.get("handoff") or ""),
        turn_input=context.get("turn_input")
        if isinstance(context.get("turn_input"), dict)
        else None,
        completion=context.get("completion")
        if isinstance(context.get("completion"), dict)
        else None,
    )
