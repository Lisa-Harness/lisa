"""Transport-neutral message dispatcher for Lisa framework hosts.

This is the core ``handle_message`` entry point that all transport hosts
(cli, web) call to execute one user turn. It:

1. **Host command dispatch** — delegates to ``hosts.core.commands`` for
   ``/help``, ``/settings``, ``/tokens``, ``/debug``, ``/consolidate``,
    ``/intake``, ``/pack``, ``/models``, ``/modes``, ``/operations``,
    ``/tools``. Pack-registered commands (``/checkpoint``, ``/restore``,
    ``/checkpoints``, ``/summarize-history``) are discovered via
    ``lisa.extensions.host_commands`` and dispatched after built-in host
    commands.
2. **Model turn routing** — for non-command messages: parses slash-command
   routes, then routes natural language directly to the current mode and
   executes the turn via ``kernel.turn_runner.run_turn``. Mode handoff
   happens mid-turn through the ``transition_stage`` tool — there is no
   separate router planner.
3. **Turn execution** — builds ``TurnHooks`` via
   ``hosts.core.turn_hooks_builder.build_turn_hooks``, calls ``run_turn``,
   persists thread entries, updates token counters, and emits transport
   callbacks at each pipeline stage.

It does NOT import Rich, Typer, aiohttp, or any transport-specific code.
Rendering is the transport host's job.

Public surface
--------------
- :func:`handle_message`
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kernel.approval import ApprovalIdentity, ApprovalRequired, render_approval_request
from kernel.ipsar import parse_route
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import OperationExecutionError
from kernel.turn_runner import TurnResult, execute_approved_invocation, run_turn

from hosts.core.commands import (
    format_operation_error,
    format_token_stats,
    format_unsupported_mode,
    handle_debug_command,
    handle_help_command,
    handle_intake_command,
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
from hosts.core.context_assembly import build_turn_context
from hosts.core.host_commands import (
    register_host_command_exacts,
    register_host_command_prefix,
    register_host_command_prefixes,
)
from hosts.core.thread import append_thread_entry
from hosts.core.token_accounting import apply_turn_usage, coerce_non_negative_int
from hosts.core.turn_hooks_builder import build_turn_hooks
from hosts.core.types import HostCommandResult

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
    t = runtime.tokens
    return {
        "turn_prompt_tokens": t.turn_prompt_tokens,
        "turn_completion_tokens": t.turn_completion_tokens,
        "turn_total_tokens": t.turn_total_tokens,
        "turn_cached_tokens": t.turn_cached_tokens,
        "turn_reasoning_tokens": t.turn_reasoning_tokens,
        "turn_duration_ms": t.turn_duration_ms,
        "turn_latency_ms": t.turn_latency_ms,
        "session_prompt_tokens": t.session_prompt_tokens,
        "session_completion_tokens": t.session_completion_tokens,
        "session_total_tokens": t.session_total_tokens,
        "session_cached_tokens": t.session_cached_tokens,
        "session_reasoning_tokens": t.session_reasoning_tokens,
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
    t = runtime.tokens
    t.turn_prompt_tokens = coerce_non_negative_int(state.get("turn_prompt_tokens", 0))
    t.turn_completion_tokens = coerce_non_negative_int(state.get("turn_completion_tokens", 0))
    t.turn_total_tokens = coerce_non_negative_int(state.get("turn_total_tokens", 0))
    t.turn_cached_tokens = coerce_non_negative_int(state.get("turn_cached_tokens", 0))
    t.turn_reasoning_tokens = coerce_non_negative_int(state.get("turn_reasoning_tokens", 0))
    t.turn_duration_ms = float(state.get("turn_duration_ms", 0.0) or 0.0)
    t.turn_latency_ms = coerce_non_negative_int(state.get("turn_latency_ms", 0))
    t.session_prompt_tokens = coerce_non_negative_int(state.get("session_prompt_tokens", 0))
    t.session_completion_tokens = coerce_non_negative_int(state.get("session_completion_tokens", 0))
    t.session_total_tokens = coerce_non_negative_int(state.get("session_total_tokens", 0))
    t.session_cached_tokens = coerce_non_negative_int(state.get("session_cached_tokens", 0))
    t.session_reasoning_tokens = coerce_non_negative_int(state.get("session_reasoning_tokens", 0))
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
    apply_turn_usage(
        state,
        result,
        model_id=model_id,
        mode_id=mode_id,
        stage_id=stage_id,
        per_model_usage=per_model if isinstance(per_model, list) else None,
    )
    _sync_token_state(runtime, state)


if TYPE_CHECKING:
    from hosts.core.callbacks import TransportCallbacks
    from hosts.core.runtime import HostRuntime

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
    from hosts.core.callbacks import NullTransport

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
    resume_result = await _try_affirmative_approval_resume(
        runtime, stripped, lowered, tport
    )
    if resume_result is not None:
        return resume_result

    # ── Model turn (mode dispatch + run_turn) ────────────────────────
    try:
        response_text = await _route_model_turn(runtime, message, tport)
    except OperationExecutionError as exc:
        await tport.on_error(f"Operation error: {exc}", exc)
        response_text = format_operation_error(exc)
        await tport.on_response(response_text)
        return HostCommandResult(text=response_text, is_error=True)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("handle_message: turn failed: %s: %s", type(exc).__name__, exc)
        await tport.on_error(f"Turn failed: {type(exc).__name__}: {exc}", exc)
        response_text = f"Turn failed: {type(exc).__name__}: {exc}"
        await tport.on_response(response_text)
        return HostCommandResult(text=response_text, is_error=True)

    await tport.on_response(response_text)
    return HostCommandResult(text=response_text)


# ── Host command dispatch ──────────────────────────────────────────────

# Register the host-command prefixes and bare-word aliases with the
# dynamic registry in ``hosts.core.host_commands``. This is the single
# source of truth: the dispatch table below is the authority on what gets
# intercepted, and the registry feeds the transcript filter (``thread.py``
# and ``state_provider.py``) so the persistence layer never owns command
# vocabulary.
#
# ``/quit`` is CLI-only and never appended to the thread.
register_host_command_prefixes(
    (
        "/help",
        "/tokens",
        "/settings",
        "/consolidate",
        "/slice",
        "/intake",
        "/pack",
        "/models",
        "/modes",
        "/operations",
        "/tools",
        "/approve",
        "/deny",
        "/debug",
    )
)
register_host_command_exacts(
    (
        "help",
        "tokens",
        "settings",
        "consolidate",
        "slice",
        "modes",
        "operations",
        "tools",
    )
)

# Register pack-discovered command prefixes for transcript filtering.
# Pack commands are discovered via lisa.extensions.host_commands and their
# prefixes must be registered so the thread excerpt builder strips them.
for _entry in _pack_host_commands():
    for _spec in _entry.specs:
        register_host_command_prefix(_spec.prefix)


async def _dispatch_host_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
    transport: TransportCallbacks,
) -> HostCommandResult | None:
    """Try each host shortcut command; return result or ``None`` to continue.

    Delegates to the individual command handlers in ``hosts.core.commands``.
    Each handler returns a plain text string; this function wraps it in a
    ``HostCommandResult`` and emits the transport callback.
    """
    # Ensure the (possibly deferred) MCP pool is ready before any slash
    # command touches runtime.mcp_tool_pool (/settings, /debug, /checkpoint,
    # /restore, /pack). Cheap commands (/help, /quit) don't need the pool,
    # but awaiting a completed no-op task is free.
    await runtime.ensure_pool_ready()
    response_text: str | None = None

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
    elif lowered == "/approve" or lowered.startswith("/approve "):
        response_text = await _handle_approval_command(
            runtime,
            stripped,
            approve=True,
            transport=transport,
        )
    elif lowered == "/deny" or lowered.startswith("/deny "):
        response_text = await _handle_approval_command(
            runtime,
            stripped,
            approve=False,
            transport=transport,
        )
    elif lowered.startswith("/debug"):
        response_text = handle_debug_command(runtime, stripped, message)

    if response_text is not None:
        await transport.on_response(response_text)
        return HostCommandResult(text=response_text)

    # ── Pack-registered commands ────────────────────────────────────
    # Packs register slash commands via lisa.extensions.host_commands.
    # The host owns only the dispatch mechanism and host-intrinsic
    # commands; pack-owned commands (e.g. /checkpoint, /restore) live in
    # the pack that owns the operations they call.
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


def _approval_identity(runtime: HostRuntime) -> ApprovalIdentity:
    """Build the current host identity used to validate approval ownership."""

    return ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id=runtime.current_mode_id,
        session_plan_digest=runtime.session_plan.plan_digest,
    )


# Short affirmative phrases that, when a single approval is pending, are
# treated as an implicit ``/approve <id>`` rather than a fresh model turn. Kept
# deliberately small and unambiguous — anything longer or question-like is
# routed to the model so the user can still hold a normal conversation
# while an approval is pending.
_AFFIRMATIVE_RESUME_PHRASES = frozenset(
    {
        "proceed",
        "continue",
        "approved",
        "approve",
        "approve it",
        "go ahead",
        "go for it",
        "do it",
        "ok",
        "okay",
        "yes",
        "y",
        "yep",
        "yeah",
        "sure",
        "confirm",
        "confirmed",
        "continue anyway",
        "proceed anyway",
    }
)


def _is_affirmative_resume(lowered: str) -> bool:
    """Return True when ``lowered`` is a short affirmative resume phrase."""

    return lowered in _AFFIRMATIVE_RESUME_PHRASES


async def _try_affirmative_approval_resume(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    transport: TransportCallbacks,
) -> HostCommandResult | None:
    """Treat a short affirmative as ``/approve <single-pending-id>``.

    Returns a ``HostCommandResult`` when the message was consumed as a
    resume, or ``None`` to let normal model-turn routing proceed. Only fires
    when the message is an affirmative phrase AND exactly one non-expired
    approval is pending for the current session identity. When multiple
    approvals are pending the user must disambiguate with an explicit
    ``/approve <id>`` — guessing would silently execute the wrong action.
    """

    if not _is_affirmative_resume(lowered):
        return None
    service = runtime.approval_service
    if service is None or not hasattr(service, "list_pending"):
        return None
    try:
        pending = await service.list_pending(_approval_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("affirmative-resume list_pending failed: %s", exc)
        return None
    if len(pending) != 1:
        return None
    request_id = pending[0].invocation.request_id
    logger.info(
        "affirmative-resume: treating %r as /approve %s",
        stripped,
        request_id,
    )
    result_text = await _handle_approval_command(
        runtime,
        f"/approve {request_id}",
        approve=True,
        transport=transport,
    )
    await transport.on_response(result_text)
    return HostCommandResult(text=result_text)


def _approval_result_text(result: Any) -> str:
    """Render a direct action result without running a model synthesis pass."""

    output_text = str(getattr(result, "output_text", "") or "").strip()
    if output_text:
        return output_text
    output_json = getattr(result, "output_json", {}) or {}
    if output_json:
        return json.dumps(output_json, ensure_ascii=True, sort_keys=True)
    return "Approved action completed."


async def _handle_approval_command(
    runtime: HostRuntime,
    command: str,
    *,
    approve: bool,
    transport: TransportCallbacks,
) -> str:
    """Approve or deny one persisted invocation without asking the model."""

    _, _, request_id = command.partition(" ")
    request_id = request_id.strip()
    verb = "approve" if approve else "deny"
    if not request_id:
        return f"Usage: /{verb} <request_id>"

    record = await runtime.approval_service.get(request_id)
    if record is None:
        return f"No pending approval request named '{request_id}'."

    identity = _approval_identity(runtime)
    if not approve:
        if await runtime.approval_service.deny(request_id, identity):
            return f"Denied approval request {request_id}."
        return f"Approval request {request_id} is not pending for this session."

    receipt = await runtime.approval_service.approve(request_id, identity)
    if receipt is None:
        return f"Approval request {request_id} is not pending for this session."

    hooks = build_turn_hooks(runtime, context={}, transport=transport)
    try:
        result = await execute_approved_invocation(invocation=record.invocation, hooks=hooks)
    except ApprovalRequired as required:
        return render_approval_request(required.invocation)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("approved action %s failed: %s", request_id, exc)
        return f"Approved action failed: {type(exc).__name__}: {exc}"
    return _approval_result_text(result)


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
    # turn hooks, which snapshot runtime.mcp_tool_pool.
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
        if start_mode and start_mode != "default" and runtime.current_mode_id == start_mode:
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
            mode_id = runtime.current_mode_id or "default"
            query = message

    return await _execute_mode_turn(runtime, message, mode_id, query, transport)


# ── Turn execution ────────────────────────────────────────────────────


async def _execute_mode_turn(
    runtime: HostRuntime,
    message: str,
    mode_id: str,
    query: str,
    transport: TransportCallbacks,
) -> str:
    """Resolve mode, build context, and run the turn through the kernel.

    This is the main turn-execution path: it resolves the mode config from
    the workflow catalog, resolves the model, builds the turn context,
    constructs ``TurnHooks`` via ``build_turn_hooks``, calls ``run_turn``,
    persists the result, and updates token counters.
    """
    catalog = get_workflow_catalog()
    mode = catalog.get(mode_id) or catalog["default"]

    # Ensure the model manifest is loaded so mode_models overrides are honored.
    if runtime.manifest is None:
        from hosts.core.model_runtime import load_manifest as _load_manifest

        runtime.manifest = _load_manifest(
            {}, data_dir=runtime.data_dir, path=runtime.models_path
        )

    preferred_model_id = preferred_model_id_for_route(
        raw_message=message,
        mode_id=mode.mode_id,
        manifest=runtime.manifest,
    )
    set_runtime_route_preview(
        runtime,
        mode_id=mode.mode_id,
        preferred_model_id=preferred_model_id,
    )

    # Check for unsupported mode gap.
    if mode_support_gap(mode) is not None:
        return _handle_unsupported_mode(runtime, message, mode, query)

    # Resolve the model for this turn.
    from hosts.core.commands import _resolve_model as _resolve_model_cmd

    resolved_model = _resolve_model_cmd(runtime, preferred_model_id)

    # Build the turn context dict.
    run_context = build_turn_context(
        raw_message=message,
        mode_id=mode.mode_id,
        query=query or message,
        preferred_model_id=preferred_model_id,
        resolved_model=resolved_model,
        mode=mode,
    )

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
            thread_path=runtime.thread_path,
            user_slug=runtime.config.user_slug,
            mode_id=mode.mode_id,
            query=query or message,
            response_text=response_text,
            provenance={"source": "system-verified", "finish_reason": "error"},
        )
        await transport.on_error(f"Operation error: {exc}", exc)
        return response_text

    # Apply token usage accounting.
    _apply_turn_usage(
        runtime,
        result,
        model_id=resolved_model.model_id,
        mode_id=mode.mode_id,
        stage_id=runtime.current_stage_id or "-",
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

    # Persist the thread entry.
    provenance: dict[str, Any] = {
        "source": "observed",
        "finish_reason": "unknown",
        "model_id": resolved_model.model_id,
    }

    append_thread_entry(
        thread_path=runtime.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id=mode.mode_id,
        query=query or message,
        response_text=result.response_text,
        provenance=provenance,
    )

    # Unified FSM routing: persist mid-turn mode handoff. When the model
    # called transition_stage with a mode_id target, update current_mode_id
    # so the next turn starts in the new mode. See docs/ROUTING_RFC_HANDOFF.md.
    if result.handoff_to and result.handoff_to != mode.mode_id:
        runtime.current_mode_id = result.handoff_to
        logger.debug(
            "mode handoff: %s -> %s",
            mode.mode_id,
            result.handoff_to,
        )

    # Emit debug context if debug mode is on.
    if runtime.config.debug:
        await transport.on_debug_context(run_context)

    return result.response_text


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
        thread_path=runtime.thread_path,
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
    from hosts.core.runtime import TurnContext

    runtime.last_turn_context = TurnContext(
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
