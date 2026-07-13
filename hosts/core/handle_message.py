"""Transport-neutral message dispatcher for Lisa framework hosts.

This is the core ``handle_message`` entry point that all transport hosts
(cli, web, vscode) call to execute one user turn. It:

1. **Host command dispatch** — delegates to ``hosts.core.commands`` for
   ``/help``, ``/settings``, ``/tokens``, ``/debug``, ``/consolidate``,
   ``/checkpoint``, ``/restore``, ``/checkpoints``, ``/pack``.
2. **Model turn routing** — for non-command messages: checks for ``/deep``
   prefix, parses slash-command routes, plans front-door routes for natural
   language, tries the front-door direct-response shortcut, tries the
   front-door clarification, then executes the mode turn via
   ``kernel.turn_runner.run_turn``.
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

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kernel.ipsar import parse_route
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import RouteDecision as FrontDoorRouteDecision
from kernel.operations.registry import OperationExecutionError
from kernel.turn_runner import TurnResult, run_turn

from hosts.core.commands import (
    explicit_deep_query,
    format_unsupported_mode,
    format_operation_error,
    frontdoor_model_alias,
    handle_checkpoint_command,
    handle_debug_command,
    handle_help_command,
    handle_list_checkpoints_command,
    handle_pack_command,
    handle_restore_command,
    handle_simple_command,
    model_alias_for_mode,
    mode_support_gap,
    preferred_model_id_for_route,
    render_runtime_settings,
    format_token_stats,
    run_explicit_slice_command,
    set_runtime_route_preview,
)
from hosts.core.context_assembly import build_turn_context
from hosts.core.frontdoor_routing import plan_frontdoor_route
from hosts.core.thread import append_thread_entry, build_recent_thread_excerpt
from hosts.core.token_accounting import apply_turn_usage, coerce_non_negative_int
from hosts.core.turn_hooks_builder import build_turn_hooks
from hosts.core.types import HostCommandResult


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
    t.session_total_latency_ms = float(state.get("session_total_latency_ms", 0.0) or 0.0)
    t.session_mode_counts = state.get("session_mode_counts", t.session_mode_counts)


def _apply_turn_usage(
    runtime: HostRuntime,
    result: Any,
    *,
    model_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
    frontdoor_routing: dict[str, Any] | None = None,
) -> None:
    """Apply turn usage via the dict-based token_accounting module."""
    state = _token_state_dict(runtime)
    apply_turn_usage(
        state,
        result,
        model_id=model_id,
        mode_id=mode_id,
        stage_id=stage_id,
        frontdoor_routing=frontdoor_routing,
    )
    _sync_token_state(runtime, state)

if TYPE_CHECKING:
    from hosts.core.callbacks import NullTransport, TransportCallbacks
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

    # ── Model turn (front-door + run_turn) ────────────────────────────
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
    response_text: str | None = None

    if lowered.startswith("/help") or lowered == "help":
        response_text = await handle_help_command(runtime, stripped, lowered, message)
    elif lowered in {"/tokens", "tokens"}:
        response_text = handle_simple_command(runtime, message, "tokens", format_token_stats(runtime))
    elif lowered in {"/settings", "settings"}:
        response_text = handle_simple_command(runtime, message, "settings", render_runtime_settings(runtime))
    elif lowered in {"/consolidate", "consolidate", "/slice", "slice"}:
        response_text = await run_explicit_slice_command(runtime, message=message)
    elif lowered.startswith("/checkpoint"):
        response_text = await handle_checkpoint_command(runtime, stripped, message)
    elif lowered.startswith("/restore"):
        response_text = await handle_restore_command(runtime, stripped, message)
    elif lowered in {"/checkpoints", "checkpoints"}:
        response_text = await handle_list_checkpoints_command(runtime, message)
    elif lowered.startswith("/pack"):
        response_text = await handle_pack_command(runtime, stripped, message)
    elif lowered.startswith("/debug"):
        response_text = handle_debug_command(runtime, stripped, message)

    if response_text is not None:
        await transport.on_response(response_text)
        return HostCommandResult(text=response_text)

    return None


# ── Model turn routing ────────────────────────────────────────────────


async def _route_model_turn(
    runtime: HostRuntime,
    message: str,
    transport: TransportCallbacks,
) -> str:
    """Handle model-backed routing: front-door planning and run_turn dispatch.

    Routing order:
    1. ``/deep`` prefix → default mode with deep-designated model.
    2. Other slash commands → ``kernel.ipsar.parse_route``.
    3. Natural language → check ``start_mode`` override from session
       context front matter. When set, bypass front-door routing entirely
       and route directly to the declared mode.
    4. Otherwise → front-door routing with direct-response shortcut and
       clarification attempts.
    5. Execute the resolved mode turn via ``kernel.turn_runner.run_turn``.
    """
    deep_query = explicit_deep_query(message)
    deep_route = deep_query is not None
    if deep_route:
        mode_id = "default"
        query = deep_query or ""
    elif message.lstrip().startswith("/"):
        mode_id, query = parse_route(message)
    else:
        # Check for start_mode override from session-context front matter.
        # When set, bypass front-door routing and go directly to the
        # declared mode (unless the user has already switched modes via
        # an explicit /mode command during this session).
        start_mode = runtime.config.start_mode
        if start_mode and runtime.current_mode_id == start_mode:
            mode_id = start_mode
            query = message
            logger.debug(
                "start_mode bypass: routing directly to mode %r "
                "(front-door skipped)", start_mode,
            )
        else:
            # Try front-door direct-response shortcut.
            direct = await _try_frontdoor_shortcut(runtime, message, transport)
            if direct is not None:
                return direct

            # Plan the full front-door route.
            route_decision = await _plan_frontdoor_route(runtime, message)
            clarification = _try_frontdoor_clarification(
                runtime, message, route_decision, transport,
            )
            if clarification is not None:
                return clarification

            mode_id = route_decision.selected_mode_id
            query = route_decision.normalized_query or message

    return await _execute_mode_turn(runtime, message, mode_id, query, deep_route, transport)


async def _try_frontdoor_shortcut(
    runtime: HostRuntime,
    message: str,
    transport: TransportCallbacks,
) -> str | None:
    """Try the front-door direct-response optimization; return response or ``None``.

    When the front-door planner provides a ``direct_response`` and the
    selected mode is ``default``, skip the full model turn and return the
    planner's response directly.
    """
    route_decision = await _plan_frontdoor_route(runtime, message)
    if (
        route_decision.clarification_needed
        or not route_decision.direct_response
        or route_decision.selected_mode_id != "default"
    ):
        return None

    response_text = route_decision.direct_response
    planner_debug = runtime.last_frontdoor_debug or {}

    # Build turn context for the shortcut.
    from hosts.core.commands import _resolve_model as _resolve_model_cmd

    resolved_default = _resolve_model_cmd(runtime, "default")
    direct_context: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_message": message,
        "mode_id": "default",
        "handoff": "front-door optimization (direct response)",
        "frontdoor_route": route_decision.model_dump(mode="json"),
        "frontdoor_debug": planner_debug or None,
    }
    _set_turn_context(runtime, direct_context)

    # Persist thread and token usage.
    append_thread_entry(
        thread_path=runtime.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query=route_decision.normalized_query or message,
        response_text=response_text,
    )
    _apply_turn_usage(
        runtime,
        TurnResult(
            response_text=response_text,
            prompt_tokens=0,
            completion_tokens=0,
        ),
        model_id=resolved_default.model_id,
        mode_id="default",
        stage_id="-",
        frontdoor_routing=planner_debug or None,
    )

    await transport.on_stage("frontdoor", {"shortcut": "direct_response"})
    return response_text


def _try_frontdoor_clarification(
    runtime: HostRuntime,
    message: str,
    route_decision: FrontDoorRouteDecision,
    _transport: TransportCallbacks,
) -> str | None:
    """Return a clarification question if the front-door requests one."""
    _ = _transport  # reserved for future stage callbacks
    if not (route_decision.clarification_needed and route_decision.clarification_question):
        return None

    response_text = route_decision.clarification_question
    planner_debug = runtime.last_frontdoor_debug or {}
    clarification_context: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_message": message,
        "mode_id": "front-door",
        "handoff": "front-door clarification",
        "frontdoor_route": route_decision.model_dump(mode="json"),
        "frontdoor_debug": planner_debug or None,
    }
    _set_turn_context(runtime, clarification_context)

    append_thread_entry(
        thread_path=runtime.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="front-door",
        query=message,
        response_text=response_text,
    )
    return response_text


async def _plan_frontdoor_route(
    runtime: HostRuntime,
    message: str,
) -> FrontDoorRouteDecision:
    """Plan the front-door route for a natural-language message.

    Delegates to ``hosts.core.frontdoor_routing.plan_frontdoor_route`` with
    all callables resolved from the runtime.
    """
    from hosts.core.commands import _resolve_model as _resolve_model_cmd
    from hosts.core.context_assembly import (
        build_frontdoor_route_catalog_markdown as _build_catalog,
        build_frontdoor_user_prompt,
        load_frontdoor_route_prompt,
        load_frontdoor_fast_path_verifier_prompt,
    )
    from hosts.core.frontdoor_routing import last_debug_dict
    from hosts.core.providers import complete_live
    from hosts.core.turn_hooks_builder import content_loader

    loader = content_loader(runtime)
    conversation_excerpt = build_recent_thread_excerpt(
        thread_path=runtime.thread_path,
        limit=3,
    )

    def _resolve_model(model_id: str) -> Any:
        return _resolve_model_cmd(runtime, model_id)

    async def _complete_live(**kwargs: Any) -> Any:
        return await complete_live(
            runtime=runtime,
            transport=None,
            **kwargs,
        )

    def _coerce_int(value: Any) -> int:
        return coerce_non_negative_int(value)

    def _preview(text: str, limit: int) -> str:
        return text[:limit]

    def _load_route_prompt() -> str:
        return load_frontdoor_route_prompt(loader)

    def _load_fast_path_verifier_prompt() -> str:
        return load_frontdoor_fast_path_verifier_prompt(loader)

    def _mode_supported(_mode_id: str, mode: Any) -> bool:
        return mode_support_gap(mode) is None

    def _build_user_prompt(**kwargs: Any) -> str:
        # plan_frontdoor_route doesn't pass user_slug — inject it from runtime.
        kwargs.setdefault("user_slug", runtime.config.user_slug)
        return build_frontdoor_user_prompt(**kwargs)

    def _get_catalog() -> dict[str, Any]:
        return get_workflow_catalog()

    def _set_last_fd_debug(payload: dict[str, Any]) -> None:
        runtime.set_last_frontdoor_debug(payload)

    decision = await plan_frontdoor_route(
        raw_message=message,
        conversation_excerpt_markdown=conversation_excerpt,
        frontdoor_alias=frontdoor_model_alias(),
        resolve_model=_resolve_model,
        complete_live=_complete_live,
        model_alias_for_mode=model_alias_for_mode,
        coerce_int=_coerce_int,
        preview_fn=_preview,
        load_route_prompt=_load_route_prompt,
        load_fast_path_verifier_prompt=_load_fast_path_verifier_prompt,
        mode_supported=_mode_supported,
        build_user_prompt=_build_user_prompt,
        get_workflow_catalog=_get_catalog,
        set_last_frontdoor_debug=_set_last_fd_debug,
    )
    # The module-level last_debug_dict is also updated by plan_frontdoor_route.
    _ = last_debug_dict()
    return decision


# ── Turn execution ────────────────────────────────────────────────────


async def _execute_mode_turn(
    runtime: HostRuntime,
    message: str,
    mode_id: str,
    query: str,
    deep_route: bool,
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
        deep_route=deep_route,
    )
    set_runtime_route_preview(
        runtime,
        mode_id=mode.mode_id,
        preferred_model_id=preferred_model_id,
    )

    # Check for unsupported mode gap.
    if mode_support_gap(mode) is not None:
        return _handle_unsupported_mode(runtime, message, mode, query, deep_route)

    # Resolve the model for this turn.
    from hosts.core.commands import _resolve_model as _resolve_model_cmd

    resolved_model = _resolve_model_cmd(runtime, preferred_model_id)

    # Build the turn context dict.
    run_context = build_turn_context(
        raw_message=message,
        mode_id=mode.mode_id,
        query=query or message,
        deep_route=deep_route,
        preferred_model_id=preferred_model_id,
        resolved_model=resolved_model,
        mode=mode,
    )

    # Inject front-door routing debug info for natural-language turns.
    if not message.lstrip().startswith("/"):
        planner_debug = runtime.last_frontdoor_debug or {}
        planner_model_payload = planner_debug.get("resolved_model")
        planner_model_id = ""
        planner_model_provider = ""
        if isinstance(planner_model_payload, dict):
            planner_model_id = str(planner_model_payload.get("model_id") or "")
            planner_model_provider = str(planner_model_payload.get("provider") or "")

        run_context["frontdoor_route"] = {
            "selected_mode_id": mode.mode_id,
            "normalized_query": query or message,
            "model_alias": frontdoor_model_alias(),
            "resolved_model_id": planner_model_id,
            "resolved_model_provider": planner_model_provider,
            "reasoning": planner_debug.get("planner_reasoning", ""),
            "confidence_score": planner_debug.get("confidence_score", 0.0),
            "fallback_used": planner_debug.get("fallback_used", False),
            "fallback_reason": planner_debug.get("fallback_reason", ""),
            "prompt_tokens": coerce_non_negative_int(planner_debug.get("planner_prompt_tokens")),
            "completion_tokens": coerce_non_negative_int(
                planner_debug.get("planner_completion_tokens")
            ),
            "total_tokens": coerce_non_negative_int(planner_debug.get("planner_total_tokens")),
        }
        if planner_debug:
            run_context["frontdoor_debug"] = planner_debug

    _set_turn_context(runtime, run_context)

    # Emit stage callback.
    await transport.on_stage("model_call", {
        "mode_id": mode.mode_id,
        "model_id": resolved_model.model_id,
        "provider": resolved_model.provider,
    })

    # Build turn hooks and execute.
    hooks = build_turn_hooks(runtime, context=run_context, transport=transport)
    try:
        result: TurnResult = await run_turn(
            mode_id=mode.mode_id,
            query=query or message,
            hooks=hooks,
            preferred_model_id=preferred_model_id,
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
        frontdoor_routing=runtime.last_frontdoor_debug,
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
    append_thread_entry(
        thread_path=runtime.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id=mode.mode_id,
        query=query or message,
        response_text=result.response_text,
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
    deep_route: bool,
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
        "deep_route": deep_route,
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
        deep_route=bool(context.get("deep_route") or False),
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
        frontdoor_debug=context.get("frontdoor_debug")
        if isinstance(context.get("frontdoor_debug"), dict)
        else None,
    )