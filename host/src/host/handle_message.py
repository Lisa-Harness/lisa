"""Transport-neutral message dispatcher for Lisa framework hosts.

This is the core ``handle_message`` entry point that all transport hosts
(cli, web) call to execute one user turn. It:

1. **Host command dispatch** — delegates to ``host.commands`` for
   ``/help``, ``/settings``, ``/tokens``, ``/debug``, ``/consolidate``,
    ``/intake``, ``/pack``, ``/models``, ``/modes``, ``/operations``,
    ``/tools``, and to ``host.approval`` for ``/approve`` / ``/deny``.
    Pack-registered commands are discovered via
    ``lisa.extensions.host_commands`` and dispatched after built-in host
    commands.
2. **Approval resume** — a short affirmative/negative reply while exactly
   one approval is pending is treated as an implicit ``/approve`` /
   ``/deny`` via :func:`host.approval.try_affirmative_approval_resume`.
3. **Model turn routing** — for non-command messages: parses slash-command
   routes, then routes natural language directly to the current mode and
   executes the turn via ``kernel.turn_runner.run_turn``. Mode handoff
   happens mid-turn through the ``transition_stage`` tool — there is no
   separate router planner.
4. **Turn execution** — builds ``TurnHooks`` via
   ``host.turn_hooks_builder.build_turn_hooks``, calls ``run_turn``,
   persists thread entries, updates token counters, and emits transport
   callbacks at each pipeline stage.

The approval subsystem (identity rebuild, affirmative resume, policy
auto-continue, post-approval continuation, paired-recovery settlement)
lives in :mod:`host.approval`. This file owns only dispatch and the
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

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kernel.ipsar import parse_route
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import OperationExecutionError
from kernel.turn_phases._routing import TurnDecision, evaluate_turn_decision
from kernel.turn_runner import (
    TurnResult,
    run_turn,
)
from sdk.logging_setup import emit_structured_event

from host.approval import (
    build_approval_data,
    handle_approval_command,
    maybe_auto_continue_approval,
    try_affirmative_approval_resume,
)
from host.commands import (
    format_operation_error,
    format_token_stats,
    format_unsupported_mode,
    handle_debug_command,
    handle_help_command,
    handle_intake_command,
    handle_mcp_command,
    handle_models_command,
    handle_pack_command,
    handle_simple_command,
    mode_support_gap,
    preferred_model_id_for_route,
    render_modes_list,
    render_operations_list,
    render_runtime_settings,
    render_tools_list,
    run_explicit_slice_command,
    set_runtime_route_preview,
)
from host.context_assembly import build_turn_context
from host.thread import append_thread_entry
from host.token_accounting import apply_turn_usage, coerce_non_negative_int
from host.turn_hooks_builder import build_turn_hooks
from host.turn_logging import build_turn_completed_payload
from host.types import HostCommandResult

# ── Pack-command discovery (lazy) ────────────────────────────────────

_pack_command_cache: list[Any] | None = None


def _pack_host_commands() -> list[Any]:
    """Return cached pack-registered host command entries.

    Discovers ``lisa.extensions.host_commands`` entry points on first call
    and caches the result. Pack commands are consulted after built-in host
    commands in ``_dispatch_host_command``.
    """
    global _pack_command_cache
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
    from host.callbacks import TransportCallbacks
    from host.runtime import HostRuntime

logger = logging.getLogger(__name__)


async def handle_message(
    runtime: HostRuntime,
    message: str,
    transport: TransportCallbacks | None = None,
) -> HostCommandResult:
    """Route one user message through the host runtime.

    Args:
        runtime: The typed host runtime with config, MCP pool, and state.
        message: The raw user message (may be a slash command or natural
            language).
        transport: Optional transport callbacks for streaming, debug
            rendering, and error delivery. When ``None``, a
            ``NullTransport`` is used (headless / test mode).

    Returns:
        A ``HostCommandResult`` with the response text and optional
        structured data for transport rendering.
    """
    from host.callbacks import NullTransport

    tport: TransportCallbacks = transport or NullTransport()
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
        model_result = await _route_model_turn(runtime, message, tport)
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

# Host-command vocabulary registration is owned by ``host.bootstrap`` and
# runs exactly once per process — see
# :func:`host.host_commands.register_builtin_host_commands`. Registering at
# import time here was a side effect that broke the kernel rule
# "registration and operation discovery must be side-effect-free on
# import" and coupled every test that imported this module to a globally
# mutated registry.


async def _dispatch_host_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
    transport: TransportCallbacks,
) -> HostCommandResult | None:
    """Try each host shortcut command; return result or ``None`` to continue.

    Delegates to the individual command handlers in ``host.commands``.
    Each handler returns a plain text string; this function wraps it in a
    ``HostCommandResult`` and emits the transport callback.
    """
    # Ensure the (possibly deferred) MCP pool is ready before any slash
    # command touches runtime.resources.mcp_tool_pool (/settings, /debug, /pack).
    # Cheap commands (/help, /quit) don't need the pool, but awaiting a
    # completed no-op task is free.
    await runtime.ensure_pool_ready()
    response_text: str | None = None
    # The approve path may run a post-approval continuation turn and return
    # a structured result (preserving ``operator_exit``) rather than plain
    # text; when set it short-circuits the string wrapping below.
    continuation_result: HostCommandResult | None = None

    if lowered.startswith("/help") or lowered == "help":
        response_text = await handle_help_command(runtime, stripped, lowered, message)
    elif lowered in {"/tokens", "tokens"}:
        response_text = handle_simple_command(
            runtime, message, "tokens", format_token_stats(runtime)
        )
    elif lowered in {"/settings", "settings"}:
        response_text = handle_simple_command(
            runtime, message, "settings", render_runtime_settings(runtime)
        )
    elif lowered in {"/consolidate", "consolidate", "/slice", "slice"}:
        response_text = await run_explicit_slice_command(runtime, message=message)
    elif lowered in {"/intake", "intake"}:
        response_text = await handle_intake_command(runtime, message=message)
    elif lowered.startswith("/pack"):
        response_text = await handle_pack_command(runtime, stripped, message)
    elif lowered.startswith("/models"):
        response_text = await handle_models_command(runtime, stripped, message)
    elif lowered in {"/modes", "modes"}:
        response_text = handle_simple_command(runtime, message, "modes", render_modes_list(runtime))
    elif lowered in {"/operations", "operations"}:
        response_text = handle_simple_command(
            runtime, message, "operations", render_operations_list(runtime)
        )
    elif lowered in {"/tools", "tools"}:
        response_text = handle_simple_command(runtime, message, "tools", render_tools_list(runtime))
    elif lowered.startswith("/mcp"):
        response_text = await handle_mcp_command(runtime, stripped, message)
    elif lowered == "/approve" or lowered.startswith("/approve "):
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
    elif lowered.startswith("/debug"):
        response_text = handle_debug_command(runtime, stripped, message)

    if continuation_result is not None:
        await transport.on_response(continuation_result.text)
        return continuation_result

    if response_text is not None:
        await transport.on_response(response_text)
        return HostCommandResult(text=response_text)

    # ── Pack-registered commands ────────────────────────────────────
    # Packs register slash commands via lisa.extensions.host_commands.
    # The host owns only the dispatch mechanism and host-intrinsic
    # commands; pack-owned commands live in the pack that owns the
    # operations they call.
    for entry in _pack_host_commands():
        for spec in entry.specs:
            if lowered.startswith(spec.prefix):
                try:
                    response_text = await spec.handler(runtime, stripped, message)
                except Exception as exc:  # pylint: disable=broad-except
                    response_text = f"Command {spec.prefix} failed: {exc}"
                if response_text is not None:
                    await transport.on_response(response_text)
                    return HostCommandResult(text=response_text)

    return None


# ── Approval flow (moved to ``host.approval``) ────────────────────────
# The identity rebuild, affirmative-resume detection, policy auto-continue,
# post-approval continuation, and paired-recovery settlement that were
# embedded here now live in :mod:`host.approval`. Only the dispatch calls
# above and the model-turn pending-approval intercept below remain; they
# wire the approval entry points by injecting :func:`_route_model_turn`.


# ── Model turn routing ────────────────────────────────────────────────


async def _route_model_turn(
    runtime: HostRuntime,
    message: str,
    transport: TransportCallbacks,
) -> str:
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
    """
    # Ensure the (possibly deferred) MCP pool is ready before building
    # turn hooks, which snapshot runtime.resources.mcp_tool_pool.
    await runtime.ensure_pool_ready()
    if message.lstrip().startswith("/"):
        mode_id, query = parse_route(message)
    else:
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
            query = message
            logger.debug(
                "route_model_turn: natural language -> mode=%s (start_mode=%s)",
                mode_id,
                runtime.start_mode_id,
            )

    return await _execute_mode_turn(runtime, message, mode_id, query, transport)


# ── Turn execution ────────────────────────────────────────────────────


async def _execute_mode_turn(
    runtime: HostRuntime,
    message: str,
    mode_id: str,
    query: str,
    transport: TransportCallbacks,
) -> HostCommandResult | str:
    """Resolve mode, build context, and run the turn through the kernel.

    This is the main turn-execution path: it resolves the mode config from
    the workflow catalog, resolves the model, builds the turn context,
    constructs ``TurnHooks`` via ``build_turn_hooks``, calls ``run_turn``,
    persists the result, and updates token counters.
    """
    catalog = get_workflow_catalog()
    mode = catalog.get(mode_id) or catalog["default"]

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

    # Pre-turn session budget gate. When the session has declared a
    # ``max_session_tokens`` cap and the accumulated session total (across
    # prior turns in this session) already meets or exceeds it, block the
    # turn before any model call — the user must start a fresh session or
    # raise the cap rather than continue burning tokens.
    max_session_tokens = runtime.session_plan.manifest.execution.max_session_tokens
    if max_session_tokens and runtime.session.tokens.session_total_tokens >= max_session_tokens:
        used = runtime.session.tokens.session_total_tokens
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

    # Resolve the model for this turn.
    from host.commands import _resolve_model as _resolve_model_cmd

    resolved_model = _resolve_model_cmd(runtime, preferred_model_id)

    # Compute the global turn index at turn start (before the turn runs)
    # so the model can be annotated with its turn number and the
    # pre-computed value can be threaded through to ``append_thread_entry``.
    # One source of truth: the number the model sees in the briefing is the
    # number stamped on the thread entry — no off-by-one between display and
    # persistence.
    from host.thread import next_turn_index  # noqa: PLC0415

    turn_index = next_turn_index(thread_path=runtime.paths.thread_path)

    # Build the turn context dict.
    run_context = build_turn_context(
        raw_message=message,
        mode_id=mode.mode_id,
        query=query or message,
        preferred_model_id=preferred_model_id,
        resolved_model=resolved_model,
        mode=mode,
    )
    run_context["turn_index"] = turn_index

    _set_turn_context(runtime, run_context)

    # Emit stage callback.
    await transport.on_stage(
        "model_call",
        {
            "mode_id": mode.mode_id,
            "model_id": resolved_model.model_id,
            "provider": resolved_model.provider,
        },
    )

    # Build turn hooks and execute.
    hooks = build_turn_hooks(runtime, context=run_context, transport=transport)
    logger.info(
        "turn started: mode=%s model=%s query=%s",
        mode.mode_id,
        resolved_model.model_id,
        (query or message)[:80],
    )
    logger.debug(
        "turn execution starting: mode=%s model=%s provider=%s",
        mode.mode_id,
        resolved_model.model_id,
        resolved_model.provider,
    )
    try:
        result: TurnResult = await run_turn(
            mode_id=mode.mode_id,
            query=query or message,
            hooks=hooks,
            preferred_model_id=preferred_model_id,
            execution_policy=runtime.execution_policy,
        )
    except OperationExecutionError as exc:
        run_context["operation_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        response_text = format_operation_error(exc)
        append_thread_entry(
            thread_path=runtime.paths.thread_path,
            user_slug=runtime.config.user_slug,
            mode_id=mode.mode_id,
            query=query or message,
            response_text=response_text,
            provenance={"source": "system-verified", "finish_reason": "error"},
            turn=turn_index,
        )
        await transport.on_error(f"Operation error: {exc}", exc)
        runtime.session.operator_signal_value = ""
        return response_text

    # Reset the per-turn auto-approval cap counter before we consult
    # ``pending_approval`` — this turn's budget of unattended auto-continues
    # starts here (see ``host.approval.maybe_auto_continue_approval``).
    runtime.session.auto_approved_this_turn = 0

    # Clear the operator signal after the turn ends so the next turn starts
    # clean (the signal is one-shot; the transport re-sets it on the next
    # key press).
    runtime.session.operator_signal_value = ""

    # ── Operator exit handling (pause/stop/terminate) ──────────────────
    # The kernel produces ``operator_exit`` when an operator signal or a cap
    # path produced a PauseAction / StopAction / TerminateAction.
    # - ``paused``: the transport holds the turn open; the next input re-enters
    #   the same turn against the preserved WorkflowPosition. The transport
    #   decides how to surface this (e.g. a "paused" indicator).
    # - ``stopped``: the transport returns the prompt. The next message is a
    #   fresh turn against the preserved position.
    # - ``terminated``: the transport returns the prompt. The workflow is
    #   marked complete; the next message is a fresh turn with no workflow.
    # The kernel handles WorkflowPosition marking; the host only needs to
    # surface the exit to the transport.
    _operator_exit = getattr(result, "operator_exit", "")
    if _operator_exit:
        if _operator_exit == "paused":
            logger.info("handle_message: turn paused by operator (mode=%s)", mode.mode_id)
        elif _operator_exit == "stopped":
            logger.info("handle_message: turn stopped by operator (mode=%s)", mode.mode_id)
        elif _operator_exit == "terminated":
            logger.info("handle_message: turn terminated by operator (mode=%s)", mode.mode_id)

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
    pending_approval = getattr(result, "pending_approval", None)
    if pending_approval is not None:
        approved_result = await maybe_auto_continue_approval(
            runtime,
            pending_approval,
            transport,
            route_model_turn=_route_model_turn,
        )
        if approved_result is not None:
            return approved_result
        approval_data = build_approval_data(runtime, pending_approval)
        payload_text = result.response_text
        return HostCommandResult(text=payload_text, data=approval_data)

    # Apply token usage accounting.
    # Sync the runtime's stage from the result's terminal transition so the
    # per-stage breakdown reflects where the turn actually ended (workflow
    # FSM advancements) rather than the entry stage set by
    # set_runtime_route_preview at dispatch. Without this, every turn's
    # stage_token_totals row is the entry stage regardless of how many
    # stages the model advanced through.
    if result.transitioned_to:
        runtime.session.current_stage_id = result.transitioned_to
        logger.debug("stage synced: %s", result.transitioned_to)
    _apply_turn_usage(
        runtime,
        result,
        model_id=resolved_model.model_id,
        mode_id=mode.mode_id,
        stage_id=runtime.session.current_stage_id or "-",
    )

    # Update completion context.
    completion_context = run_context.get("completion")
    if not isinstance(completion_context, dict):
        completion_context = {}
    completion_context.update(
        {
            "mode_id": mode.mode_id,
            "query": query or message,
            "response_preview": result.response_text[:240],
            "response_chars": len(result.response_text),
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
    _set_turn_context(runtime, run_context)

    # Audit: emit the kernel turn's completion event. The event carries
    # mode/handoff/blocked metadata; the effect-receipt ledger lives in
    # the thread entry (the canonical, turn-addressable source) and is
    # not duplicated in the event payload.
    emit_structured_event(
        logger,
        level=logging.INFO,
        event="turn.completed",
        message=(f"turn.completed: mode={mode.mode_id} handoff={result.handoff_to or ''}"),
        **build_turn_completed_payload(
            mode_id=mode.mode_id,
            result=result,
            coerce_int=coerce_non_negative_int,
        ),
    )

    # Persist the thread entry.
    provenance: dict[str, Any] = {
        "source": "observed",
        "finish_reason": "unknown",
        "model_id": resolved_model.model_id,
    }

    # Serialize the effect-receipt ledger and mode/stage transitions into
    # the thread entry so the workflow view can be derived losslessly
    # from the thread alone — no separate mutable position store needed.
    # The thread entry is the canonical source of receipts; the
    # ``turn.completed`` event links to it by turn number rather than
    # duplicating the receipt list.
    receipts = [
        {
            "capability_id": str(getattr(r, "capability_id", "")),
            "effect": str(getattr(r, "effect", "")),
            "artifact_id": str(getattr(r, "artifact_id", "") or ""),
        }
        for r in (result.effect_receipts or [])
    ]

    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id=mode.mode_id,
        query=query or message,
        response_text=result.response_text,
        provenance=provenance,
        effect_receipts=receipts,
        handoff_to=result.handoff_to or "",
        transitioned_to=result.transitioned_to or "",
        turn=turn_index,
        yielded=result.yielded is not None,
        yield_question=result.yielded.question if result.yielded else "",
        turn_input=run_context.get("turn_input")
        if isinstance(run_context.get("turn_input"), dict)
        else None,
        # Thread → memory provenance: memory records minted by this turn's
        # auto-capture (stamped onto run_context by the turn-complete hook).
        memory_ids=run_context.get("memory_ids")
        if isinstance(run_context.get("memory_ids"), list)
        else None,
    )

    # ── Centralized loop decision classification ────────────────────
    # Classify the turn's outcome using the uniform three-outcome contract
    # (CONTINUE / HANDOFF / FINISH). The host does not iterate the loop
    # itself — ``run_turn`` does that internally — but this classification
    # ensures the host's post-turn handoff and persistence logic is
    # consistent with the same contract the kernel uses. It also gives
    # the transport a structured outcome for rendering.
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

    # Unified FSM routing: persist mid-turn mode handoff. When the model
    # called transition_stage with a mode_id target, update current_mode_id
    # so the next turn starts in the new mode. See docs/ROUTING_RFC_HANDOFF.md.
    if _decision.decision is TurnDecision.HANDOFF and _decision.target:
        runtime.session.current_mode_id = _decision.target
        logger.debug(
            "mode handoff: %s -> %s",
            mode.mode_id,
            _decision.target,
        )

    # Emit debug context if debug mode is on.
    if runtime.config.debug:
        await transport.on_debug_context(run_context)

    return HostCommandResult(
        text=result.response_text,
        operator_exit=getattr(result, "operator_exit", ""),
        # Surface the model's ask-the-user terminal action structurally: a
        # non-empty question means the turn ended in a yield — the model
        # asked the operator something instead of finishing. The question
        # rides both the typed field (one-shot hosts exit non-zero) and
        # ``data["yielded"]`` so panel transports can render it inline.
        yielded_question=(result.yielded.question if result.yielded else ""),
        data={"yielded": result.yielded.question} if result.yielded else None,
    )


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
    from host.runtime import TurnContext

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
