"""Transport-neutral message dispatcher for Lisa framework hosts.

This is the core ``handle_message`` entry point that all transport hosts
(cli, web) call to execute one user turn. It:

1. **Host command dispatch** — delegates to ``host.commands`` for
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
   ``host.turn_hooks_builder.build_turn_hooks``, calls ``run_turn``,
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

from kernel.approval import (
    ApprovalIdentity,
    ApprovalInvocation,
    ApprovalRequired,
    render_approval_request,
)
from kernel.ipsar import parse_route
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import OperationExecutionError
from kernel.turn_runner import TurnResult, execute_approved_invocation, run_turn
from sdk.logging_setup import emit_structured_event

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
from host.host_commands import (
    register_host_command_exacts,
    register_host_command_prefix,
    register_host_command_prefixes,
)
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
    resume_result = await _try_affirmative_approval_resume(runtime, stripped, lowered, tport)
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
# dynamic registry in ``host.host_commands``. This is the single
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
        "/mcp",
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
        "mcp",
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

    Delegates to the individual command handlers in ``host.commands``.
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
    elif lowered.startswith("/mcp"):
        response_text = await handle_mcp_command(runtime, stripped, message)
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
#
# This set has two tiers:
# - ``_AFFIRMATIVE_EXACT``: conversational affirmatives (``"ok"``, ``"yes"``)
#   that must match the *whole* message. They would false-positive as
#   prefixes (``"ok but…"`` is not a resume).
# - Imperative resume phrases (``"proceed"``, ``"continue"``) live in the
#   shared registry (:func:`kernel.routing_vocabulary.get_resume_phrases`)
#   and are matched as **prefixes** so ``"proceed with the current step in
#   the plan"`` matches ``"proceed"``. The registry is session-authored via
#   ``routing.resume_phrases`` in ``session_context.md``; the historical
#   default is used when no session vocabulary is wired.
_AFFIRMATIVE_EXACT = frozenset(
    {
        "approved",
        "approve",
        "approve it",
        "ok",
        "okay",
        "yes",
        "y",
        "yep",
        "yeah",
        "sure",
        "confirm",
        "confirmed",
    }
)


def _is_affirmative_resume(lowered: str) -> bool:
    """Return True when ``lowered`` is a short affirmative resume phrase.

    Matches the whole message against the exact-affirmative set, or the
    message's start against any registered resume phrase (prefix match).
    The resume-phrase vocabulary is session-authored and shared with the
    kernel's continuation-query detector via
    :func:`kernel.routing_vocabulary.get_resume_phrases`.
    """
    if lowered in _AFFIRMATIVE_EXACT:
        return True
    # Prefix match against the shared resume-phrase registry. ``"proceed
    # anyway"`` and ``"continue anyway"`` are historically affirmative —
    # the registry's default list includes ``"proceed"`` and ``"continue"``,
    # so the prefix match covers them without a dedicated entry.
    from kernel.routing_vocabulary import get_resume_phrases  # noqa: PLC0415

    return any(lowered.startswith(phrase) for phrase in get_resume_phrases())


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
            # Denying the approval also unblocks the paired recovery record:
            # the original model-turn stall was ``blocks_workflow=True`` and
            # would otherwise re-yield on every subsequent turn. Settle it
            # as cancelled so the workflow can advance.
            await _settle_paired_recovery_record(
                runtime,
                record.invocation,
                state="cancelled",
                last_error="denied by operator",
            )
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
        await _settle_paired_recovery_record(
            runtime, record.invocation, state="stalled", last_error=f"{type(exc).__name__}: {exc}"
        )
        return f"Approved action failed: {type(exc).__name__}: {exc}"
    # The approved action executed against the real registry, which consumes
    # the receipt. But the *model-turn* path that originally raised
    # ``ApprovalRequired`` also settled a paired recovery record as
    # ``stalled`` with ``blocks_workflow=True`` (turn_runner.py). The approval
    # and recovery stores are independent with no shared id, so approving the
    # action left the stall permanently blocking every subsequent turn. Settle
    # the paired record now so the workflow unblocks.
    await _settle_paired_recovery_record(
        runtime,
        record.invocation,
        state="succeeded",
        result_preview=_approval_result_text(result)[:240],
    )
    return _approval_result_text(result)


async def _settle_paired_recovery_record(
    runtime: HostRuntime,
    invocation: ApprovalInvocation,
    *,
    state: str,
    last_error: str = "",
    result_preview: str = "",
) -> None:
    """Settle the recovery record paired to an approved invocation.

    The model-turn path raises ``ApprovalRequired`` *after* the recovery
    service has already ``begin``-ed a record for the same target. Approving
    via ``/approve`` re-executes through a fresh registry path that does not
    touch the recovery service, so the original record stays ``stalled`` and
    ``blocks_workflow=True`` — the zombie stall that re-yields on every
    subsequent turn. This correlates by ``target_id`` + identity
    (``session_plan_digest`` is part of the recovery identity) and settles
    the first open stalled match. Correlation is deliberately by target id,
    not approval request id: the two stores never share an id.
    """

    recovery_service = getattr(runtime, "recovery_service", None)
    if recovery_service is None or not hasattr(recovery_service, "list_open"):
        return
    recovery_identity = _recovery_identity_for_invocation(runtime, invocation)
    if recovery_identity is None:
        return
    try:
        open_records = await recovery_service.list_open(recovery_identity)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("paired-recovery list_open failed for %s: %s", invocation.request_id, exc)
        return
    for open_record in open_records:
        if open_record.target_id != invocation.target_id:
            continue
        if getattr(open_record, "state", "") != "stalled":
            continue
        try:
            await recovery_service.settle(
                open_record.work_item_id,
                recovery_identity,
                state=state,  # type: ignore[arg-type]
                last_error=last_error,
                result_preview=result_preview,
                blocks_workflow=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "paired-recovery settle failed for %s (work item %s): %s",
                invocation.request_id,
                open_record.work_item_id,
                exc,
            )
        return


def _recovery_identity_for_invocation(
    runtime: HostRuntime,
    invocation: ApprovalInvocation,
) -> Any | None:
    """Build the recovery identity scoped to an approval invocation.

    ``ApprovalIdentity`` and ``RecoveryIdentity`` carry the same fields with
    the same scoping; ``session_plan_digest`` is part of both, so a record
    minted mid-turn matches the invocation that paused for the same turn.
    Returns ``None`` when the digests diverge (stale approval from a prior
    session plan) so the caller can skip settlement safely.
    """

    from kernel.recovery import RecoveryIdentity  # noqa: PLC0415

    if invocation.session_plan_digest != runtime.session_plan.plan_digest:
        return None
    identity = invocation.identity
    return RecoveryIdentity(
        workspace_id=identity.workspace_id,
        user_id=identity.user_id,
        user_slug=identity.user_slug,
        surface_type=identity.surface_type,  # type: ignore[arg-type]
        surface_id=identity.surface_id,
        memory_scope=identity.memory_scope,  # type: ignore[arg-type]
        session_plan_digest=identity.session_plan_digest,
    )


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

    # Pre-turn session budget gate. When the session has declared a
    # ``max_session_tokens`` cap and the accumulated session total (across
    # prior turns in this session) already meets or exceeds it, block the
    # turn before any model call — the user must start a fresh session or
    # raise the cap rather than continue burning tokens.
    max_session_tokens = runtime.session_plan.manifest.execution.max_session_tokens
    if max_session_tokens and runtime.tokens.session_total_tokens >= max_session_tokens:
        used = runtime.tokens.session_total_tokens
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
    # Sync the runtime's stage from the result's terminal transition so the
    # per-stage breakdown reflects where the turn actually ended (workflow
    # FSM advancements) rather than the entry stage set by
    # set_runtime_route_preview at dispatch. Without this, every turn's
    # stage_token_totals row is the entry stage regardless of how many
    # stages the model advanced through.
    if result.transitioned_to:
        runtime.current_stage_id = result.transitioned_to
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

    # Audit: emit the kernel turn's completion event carrying the effect
    # receipt ledger (Phase 8: one verification primitive) so
    # hallucination signals are queryable from the JSONL stream without
    # reconstructing individual dispatch entries.
    emit_structured_event(
        logger,
        level=logging.INFO,
        event="turn.completed",
        message=(
            f"turn.completed: mode={mode.mode_id} "
            f"handoff={result.handoff_to or ''}"
        ),
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
    from host.runtime import TurnContext

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
