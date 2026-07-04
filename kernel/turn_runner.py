"""Kernel turn runner — the single model+tool loop shared by lightweight hosts and the full app.

This module provides `run_turn()` and `TurnHooks`, the core conversational turn
loop that executes one mode's pipeline: pre-turn operations, model completion with
tool dispatch, and post-turn side effects.

DESIGN — Extension via TurnHooks, not replacement:
    Little Lisa and the production app both call `run_turn()`. They differ only
    in which `TurnHooks` they provide. Every hook defaults to a sensible no-op or
    stub, so a minimal host participates with zero overrides.

    The full app (`app/agent.py`) can eventually delegate its chat and workflow
    completion paths to `run_turn()` by registering `AppTurnRunnerHooks` that
    override model resolution, context assembly, tool dispatch, and post-turn
    persistence — without duplicating the tool loop.

Gaps (use the full `app/agent.py` turn loop when you need):
    - FSM stage transitions (transition_stage, session state machines)
    - Mode delegation (delegate_mode tool dispatch)
    - Live system state, passive intelligence, reflection context
    - Self-model, architecture atlas, capability vibes
    - Thread persistence, handoff artifacts, conversation digest refresh
    - Model restrict/unavailable gating, provider fallback
    - Proactive/autopilot surface reflection
    - Deterministic intent handling (notes, reminders, memory recall/disputes)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from kernel.ipsar_fsm import (
    get_mode_system_intro,
    get_workflow_catalog,
    run_mode_entry_operations,
    run_mode_pre_turn_operations,
)
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    ToolExecutionContext,
    ToolResult,
)
from kernel.operations.registry import (
    build_openai_operation_schema,
    execute_operation,
    get_operation_catalog,
)
from kernel.slice import (
    SliceMode,
    SliceOutcome,
    SlicePolicy,
    SliceTuning,
    build_policy_for_trigger,
    resolve_mode_for_trigger,
)

logger = logging.getLogger(__name__)

# ── Public types ───────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Result of one call to `run_turn()`."""

    response_text: str
    tool_call_count: int = 0
    iteration_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    duration_ms: float = 0.0
    latency_ms: int = 0
    slice_outcomes: list[SliceOutcome] = field(default_factory=list)
    """SliceOutcome records for every opportunistic slice that ran
    during this turn. Populated only when the host has wired the
    corresponding slice slot. Empty list when opportunistic
    house-keeping is off (``slice_tuning=None`` or all triggers set
    to ``never``)."""


@dataclass
class TurnHooks:
    """Extension points for the kernel turn runner.

    The host (lightweight host or production app) provides concrete implementations.
    Every callable defaults to a no-op or sensible stub, so a minimal host can
    participate with zero overrides.

    DESIGN: Hooks are named for *what they do*, not *why the app needs them*.
    This keeps the kernel free of app-specific concepts like "passive
    intelligence," "capability vibes," or "self-model context."
    """

    # ── Model resolution ────────────────────────────────────────────────
    resolve_model: Callable[[str], Any] | None = None
    """Map a model_id string to a model-definition object.
    Default: returns a simple dataclass with model_id and reasoning_model=False.
    """

    # ── Completion ───────────────────────────────────────────────────────
    complete: Callable[..., Awaitable[Any]] | None = None
    """Core completion function.  Must accept messages, model_def, max_tokens,
    temperature, tools (optional), and response_format (optional).  Returns
    an object with `.content` (str) and `.tool_calls` (list[dict]).

    A host MUST provide this.  If None, `run_turn` raises NotImplementedError.
    """

    # ── Pre-turn context assembly ────────────────────────────────────────
    assemble_system_prompt: Callable[[str, str | None], str] | None = None
    """Given mode_instruction: str and optional stage_instruction: str|None,
    return the full system prompt.  Default: mode_instruction only.

    The app injects SYSTEM_PROMPT, turn-input contract, and planner block here.
    """

    assemble_user_context: Callable[[str, str, str], str] | None = None
    """Given mode_id: str, query: str, and pre_turn_briefing: str, return
    additional user-context content injected before the query in the final
    user message.  Default: empty string.

    The app injects surface blocks, user blocks, live context, memory recall,
    self-model, architecture atlas, etc. here.
    """

    # ── Tool schemas ─────────────────────────────────────────────────────
    build_extra_tool_schemas: Callable[[str], list[dict[str, Any]]] | None = None
    """Return extra tool schemas to merge with operation-sourced schemas.
    Default: empty list.  The app injects its tool catalog, transition tools,
    and delegation tools here.
    """

    # ── Tool/operation dispatch ──────────────────────────────────────────
    execute_tool: Callable[..., Awaitable[ToolResult]] | None = None
    """Execute a named tool with a ToolExecutionContext. Default: returns a
    ToolResult with 'tool unavailable' text.
    """

    execute_operation: Callable[..., Awaitable[OperationResult]] | None = None
    """Execute a named operation with an OperationExecutionContext. Default:
    delegates to `kernel.operations.registry.execute_operation`.
    """

    # ── Memory (lite) ────────────────────────────────────────────────────
    recall_memory: Callable[[str, str, int], Awaitable[str]] | None = None
    """Return a memory snippet for the given query (str), scoped to a
    memory_scope (str), within a token budget (int). Default: empty string.

    The app overrides with build_auto_recall_snippet.
    """

    store_memory: Callable[[str, str, str], Awaitable[None]] | None = None
    """Persist a (query: str, response: str, memory_scope: str) tuple as a
    memory. Default: no-op.

    The app overrides with memory_store.record_memory.
    """

    # ── Post-turn side effects ───────────────────────────────────────────
    on_turn_complete: Callable[[str, str, str, list[dict[str, Any]]], Awaitable[None]] | None = None
    """Called after every successful turn with mode_id, query, response_text,
    and the full message list. Default: no-op.

    The app hooks thread persistence, conversation digest refresh, provenance,
    and context-usage events here.
    """

    # ── Opportunistic house-keeping (turn-anchored slices) ───────────────
    # See kernel/slice.py for the full contract. Each slice slot is
    # independently optional. A host that wants only post-turn slices
    # leaves pre_turn_slice and explicit_slice as None. A host that
    # wants nothing (purposeful neglect) leaves all three as None.
    slice_tuning: Any = None
    """SliceTuning instance (from kernel.slice) that controls budget
    shape. Default: ``None`` means "use SliceTuning() defaults and
    disable all three triggers." A host picks the trigger modes by
    setting the appropriate enabled flags.
    """

    pre_turn_slice: Callable[[Any], Awaitable[Any]] | None = None
    """Opportunistic slice fired before the model call. Receives a
    ``SlicePolicy`` (mode=PRE_TURN) and returns a ``SliceOutcome``.

    The framework short-circuits the call if the budget is zero (e.g.
    the pressure curve has collapsed to nothing on this turn).
    """

    post_turn_slice: Callable[[Any], Awaitable[Any]] | None = None
    """Opportunistic slice fired after the response is committed.
    Receives a ``SlicePolicy`` (mode=POST_TURN) and returns a
    ``SliceOutcome``.
    """

    explicit_slice: Callable[[Any], Awaitable[Any]] | None = None
    """Slice fired only when an operation or CLI command invokes it.
    Receives a ``SlicePolicy`` (mode=EXPLICIT). The caller's budget
    is honored verbatim — pressure does not apply."""

    last_slice_finished_at: float | None = None
    """Monotonic timestamp of the last slice that completed on this
    host. Used by the framework's pressure math. The host can also
    read/write this to coordinate multiple slices."""


# ── Default hook implementations ───────────────────────────────────────────


async def _noop_complete(**kwargs: Any) -> Any:
    raise NotImplementedError(
        "TurnHooks.complete is required. Provide a completion function that "
        "returns an object with .content (str) and .tool_calls (list[dict])."
    )


def _default_resolve_model(model_id: str) -> Any:
    """Return a minimal stub model definition."""
    return _StubModelDef(model_id=model_id or "default")


def _default_assemble_system_prompt(
    mode_instruction: str,
    _stage_instruction: str | None,
) -> str:
    return mode_instruction


def _default_assemble_user_context(
    _mode_id: str,
    _query: str,
    _pre_turn_briefing: str,
) -> str:
    return ""


def _default_extra_tool_schemas(_mode_id: str) -> list[dict[str, Any]]:
    return []


async def _default_execute_tool(
    tool_id: str,
    _ctx: ToolExecutionContext,
) -> ToolResult:
    return ToolResult(
        tool_id=tool_id,
        output_text=f"Tool '{tool_id}' is not wired in this host.",
    )


async def _default_execute_operation(
    operation_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    return await execute_operation(operation_id, ctx)


async def _default_recall_memory(
    _query: str,
    _memory_scope: str,
    _budget_tokens: int,
) -> str:
    return ""


async def _default_store_memory(
    _query: str,
    _response: str,
    _memory_scope: str,
) -> None:
    return


async def _default_on_turn_complete(
    _mode_id: str,
    _query: str,
    _response_text: str,
    _messages: list[dict[str, Any]],
) -> None:
    return


async def _default_slice(_policy: SlicePolicy) -> SliceOutcome:
    return SliceOutcome()


# ── Internal helpers ────────────────────────────────────────────────────────


async def _fire_slice(
    *,
    hooks: "TurnHooks",
    trigger: SliceMode,
    slice_outcomes: list[SliceOutcome],
) -> SliceOutcome | None:
    """Dispatch one opportunistic house-keeping slice.

    Returns the outcome if the slice actually ran, ``None`` otherwise.
    The hook is invoked only when the host has both wired the
    corresponding slot AND enabled the trigger in its ``SliceTuning``.

    The caller is responsible for updating
    ``hooks.last_slice_finished_at`` after the call returns; that
    field is on the host's ``TurnHooks`` instance so it survives across
    turns of the same session.
    """
    slot = {
        SliceMode.PRE_TURN: hooks.pre_turn_slice,
        SliceMode.POST_TURN: hooks.post_turn_slice,
        SliceMode.EXPLICIT: hooks.explicit_slice,
    }.get(trigger)
    if slot is None:
        return None

    tuning = hooks.slice_tuning
    if tuning is None:
        return None
    if resolve_mode_for_trigger(tuning=tuning, trigger=trigger) == SliceMode.NEVER:
        return None

    has_pre_turn = (
        hooks.pre_turn_slice is not None
        and resolve_mode_for_trigger(tuning=tuning, trigger=SliceMode.PRE_TURN) != SliceMode.NEVER
    )
    has_post_turn = (
        hooks.post_turn_slice is not None
        and resolve_mode_for_trigger(tuning=tuning, trigger=SliceMode.POST_TURN) != SliceMode.NEVER
    )
    policy = build_policy_for_trigger(
        trigger=trigger,
        tuning=tuning,
        last_slice_finished_at=hooks.last_slice_finished_at,
        has_pre_turn=has_pre_turn,
        has_post_turn=has_post_turn,
    )
    if policy.budget_ms <= 0:
        return None

    started = time.monotonic()
    try:
        outcome = await slot(policy)
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = (time.monotonic() - started) * 1000.0
        logger.warning(
            "%s slice failed: %s: %s",
            trigger.value,
            type(exc).__name__,
            exc,
        )
        outcome = SliceOutcome(
            work_items_completed=0,
            elapsed_ms=elapsed_ms,
            last_item="<error>",
            notes=(f"{type(exc).__name__}: {exc}",),
        )
    if not isinstance(outcome, SliceOutcome):
        # Defensive: a host that returns a non-SliceOutcome should not
        # break the turn. Coerce silently and log once.
        logger.debug(
            "%s slice returned %s instead of SliceOutcome; coercing",
            trigger.value,
            type(outcome).__name__,
        )
        outcome = SliceOutcome(
            work_items_completed=0,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )
    else:
        # Backfill elapsed_ms if the host didn't measure.
        if outcome.elapsed_ms <= 0:
            outcome = SliceOutcome(
                work_items_completed=outcome.work_items_completed,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item=outcome.last_item,
                notes=outcome.notes,
            )
    slice_outcomes.append(outcome)
    return outcome


@dataclass(frozen=True, slots=True)
class _StubModelDef:
    model_id: str
    reasoning_model: bool = False


def _extract_tool_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, dict):
        return str(function.get("name", ""))
    return str(tool_call.get("name", ""))


def _extract_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    import json as _json

    raw = None
    function = tool_call.get("function")
    if isinstance(function, dict):
        raw = function.get("arguments", "{}")
    else:
        raw = tool_call.get("arguments", "{}")
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except _json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _extract_pre_turn_briefing(result: OperationResult) -> str:
    """Return the briefing text that should be injected into the user payload.

    Declarative operations may fall back to rendering their full structured payload
    as JSON when `output_text` is empty. That is useful for direct operation
    callers, but pre-turn prompt assembly should only inject the authored briefing
    text field itself. When the structured payload explicitly says `output_text`
    is empty, treat the pre-turn briefing as absent.
    """

    payload = result.output_json
    if isinstance(payload, dict):
        for field_name in ("output_text", "body_markdown", "markdown", "answer_summary", "summary"):
            candidate = payload.get(field_name)
            if isinstance(candidate, str):
                return candidate.strip()

    return result.output_text.strip()


# ── Core turn runner ────────────────────────────────────────────────────────


async def run_turn(
    *,
    mode_id: str,
    query: str,
    hooks: TurnHooks,
    operation_catalog: dict[str, Any] | None = None,
    max_tool_iterations: int = 5,
    preferred_model_id: str | None = None,
    logger_override: logging.Logger | None = None,
) -> TurnResult:
    """Execute one conversational turn with the full model+tool loop.

    This is the single turn loop shared by lightweight hosts and the production app.
    Both call the same function — they differ only in which `TurnHooks` they
    provide.

    Phases:
    1. **PRE-TURN**: Resolve mode config, run pre-turn/entry operations,
       assemble system prompt and user context, recall memory snippet.
    2. **TOOL LOOP**: Call hooks.complete → dispatch tool/operation calls
       via hooks → feed results back → repeat until the model returns
       content without tools or max_tool_iterations is reached.
    3. **POST-TURN**: Call hooks.on_turn_complete and hooks.store_memory.

    Args:
        mode_id: The route mode string (e.g. 'default', 'discuss', 'search').
        query: The user's message text after command-prefix stripping.
        hooks: Extension-point callbacks provided by the host.
        operation_catalog: Pre-loaded operation catalog. Defaults to
            `get_operation_catalog()`.
        max_tool_iterations: Maximum number of tool-calling loop iterations.
        preferred_model_id: Optional per-turn model alias override supplied by
            the host when routing needs an explicit front-door model.
        logger_override: Optional logger for test capture.

    Returns:
        TurnResult with the final response text and loop statistics.

    Raises:
        NotImplementedError: If hooks.complete is None.
    """
    _log = logger_override or logger
    catalog = operation_catalog or get_operation_catalog()
    mode_config = get_workflow_catalog().get(mode_id)

    # Resolve hook callables with defaults
    resolve_model = hooks.resolve_model or _default_resolve_model
    complete = hooks.complete or _noop_complete
    assemble_system = hooks.assemble_system_prompt or _default_assemble_system_prompt
    assemble_user = hooks.assemble_user_context or _default_assemble_user_context
    extra_tool_schemas_fn = hooks.build_extra_tool_schemas or _default_extra_tool_schemas
    exec_tool = hooks.execute_tool or _default_execute_tool
    exec_op = hooks.execute_operation or _default_execute_operation
    recall_mem = hooks.recall_memory or _default_recall_memory
    store_mem = hooks.store_memory or _default_store_memory
    on_complete = hooks.on_turn_complete or _default_on_turn_complete

    # ── Phase 1: Pre-turn ─────────────────────────────────────────────
    pre_briefing = ""
    if (
        mode_config is not None
        and mode_config.mode_kind == "action"
        and mode_config.entry_operation_ids
    ):
        # Action modes run entry operations and return immediately
        ctx = _build_minimal_context(
            mode_id=mode_id,
            query=query,
            resolve_model_fn=resolve_model,
            complete_fn=complete,
            execute_tool_fn=exec_tool,
            execute_operation_fn=exec_op,
        )
        result = await run_mode_entry_operations(mode_id, ctx)
        await store_mem(query, result.output_text, "personal")
        return TurnResult(
            response_text=result.output_text.strip() or "Done",
            tool_call_count=0,
            iteration_count=1,
        )

    # Chat/workflow modes: run pre-turn operations
    ctx = _build_minimal_context(
        mode_id=mode_id,
        query=query,
        resolve_model_fn=resolve_model,
        complete_fn=complete,
        execute_tool_fn=exec_tool,
        execute_operation_fn=exec_op,
    )
    pre_result = await run_mode_pre_turn_operations(mode_id, ctx)
    pre_briefing = _extract_pre_turn_briefing(pre_result)

    # ── Phase 1.5: Pre-turn opportunistic slice ────────────────────────
    # Fires before the model call so any state the recall/assemble
    # hooks will read is fresh. Bounded by the pressure curve.
    slice_outcomes: list[SliceOutcome] = []
    pre_turn_outcome = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=slice_outcomes,
    )
    if pre_turn_outcome is not None:
        hooks.last_slice_finished_at = time.monotonic()

    # Assemble prompts
    mode_instruction = get_mode_system_intro(mode_id)
    system_prompt = assemble_system(mode_instruction, None)
    user_context = assemble_user(mode_id, query, pre_briefing)
    memory_snippet = await recall_mem(query, "personal", 2000)

    # ── Phase 2: Tool loop ────────────────────────────────────────────
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Build user message
    user_parts: list[str] = []
    if pre_briefing:
        user_parts.append(f"## Pre-turn Briefing\n\n{pre_briefing}")
    if memory_snippet:
        user_parts.append(f"## Memory Recall\n\n{memory_snippet}")
    if user_context:
        user_parts.append(user_context)
    user_parts.append(f"## User Query\n\n{query}")
    user_content = "\n\n".join(user_parts)
    messages.append({"role": "user", "content": user_content})

    # Build tool schemas for this turn
    tool_schemas: list[dict[str, Any]] = []
    tool_schemas.extend(build_openai_operation_schema(mode_id))
    tool_schemas.extend(extra_tool_schemas_fn(mode_id))

    # Resolve preferred model
    preferred_model_id = (
        preferred_model_id.strip()
        if isinstance(preferred_model_id, str) and preferred_model_id.strip()
        else mode_config.default_model
        if mode_config and mode_config.default_model
        else "default"
    )
    model_def = resolve_model(preferred_model_id)

    content = ""
    total_tool_calls = 0
    iteration = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    total_duration_ms = 0.0
    total_latency_ms = 0

    while iteration < max_tool_iterations:
        iteration += 1
        completion = await complete(
            messages=messages,
            model_def=model_def,
            max_tokens=4096,
            temperature=0.4,
            tools=tool_schemas if tool_schemas else None,
        )
        (
            prompt_tokens,
            completion_tokens,
            completion_total_tokens,
            cached_tokens,
            reasoning_tokens,
            duration_ms,
            latency_ms,
        ) = _extract_completion_usage(completion)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += completion_total_tokens
        total_cached_tokens += cached_tokens
        total_reasoning_tokens += reasoning_tokens
        total_duration_ms += duration_ms
        total_latency_ms += latency_ms

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if getattr(completion, "content", None):
            assistant_msg["content"] = completion.content
        tool_calls = getattr(completion, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            content = getattr(completion, "content", "") or ""
            content = content.strip()
            break

        # Dispatch tool/operation calls
        standard_tools_dispatched = False
        for tool_call in tool_calls:
            tool_name = _extract_tool_name(tool_call)
            tool_call_id = str(tool_call.get("id") or "")
            tool_arguments = _extract_tool_arguments(tool_call)

            tool_output = "Tool call budget exhausted."
            execution_result: OperationResult | ToolResult | None = None

            if tool_name in catalog:
                standard_tools_dispatched = True
                try:
                    op_ctx = _build_minimal_context(
                        mode_id=mode_id,
                        query=query,
                        resolve_model_fn=resolve_model,
                        complete_fn=complete,
                        execute_tool_fn=exec_tool,
                        execute_operation_fn=exec_op,
                    )
                    op_ctx.params = tool_arguments
                    execution_result = await exec_op(tool_name, op_ctx)
                    tool_output = _stringify_result(execution_result)
                    total_tool_calls += 1
                except Exception as exc:  # pylint: disable=broad-except
                    _log.warning("operation '%s' failed in turn runner: %s", tool_name, exc)
                    tool_output = f"Operation execution failed: {type(exc).__name__}: {exc}"
            else:
                standard_tools_dispatched = True
                try:
                    t_ctx = ToolExecutionContext(
                        tool_id=tool_name,
                        workspace_id="lite-host",
                        user_id="lite-user",
                        user_slug="lite-user",
                        mode_id=mode_id,
                        query=query,
                        parameters=tool_arguments,
                    )
                    execution_result = await exec_tool(tool_name, t_ctx)
                    tool_output = _stringify_result(execution_result)
                    total_tool_calls += 1
                except Exception as exc:  # pylint: disable=broad-except
                    _log.warning("tool '%s' failed in turn runner: %s", tool_name, exc)
                    tool_output = f"Tool execution failed: {type(exc).__name__}: {exc}"

            if not tool_output.strip():
                tool_output = "Done"

            tool_result_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": str(tool_output),
            }
            # Preserve thought_signature for Gemini thinking mode round-trips.
            # Google requires it inside the function object ("functionCall parts").
            thought_sig = tool_call.get("thought_signature")
            if thought_sig:
                tool_result_msg["function"] = {"thought_signature": thought_sig}
            messages.append(tool_result_msg)

        if not standard_tools_dispatched:
            # No non-transition tools were dispatched — model returned text
            content = getattr(completion, "content", "") or ""
            content = content.strip()
            break

    if not content:
        content = "The model did not produce a final response."

    # ── Phase 3: Post-turn ────────────────────────────────────────────
    await on_complete(mode_id, query, content, messages)
    await store_mem(query, content, "personal")

    # ── Phase 3.5: Post-turn opportunistic slice ──────────────────────
    # Fires after the response is committed. The user has already seen
    # the answer; this slice consumes the deferred pressure budget.
    post_turn_outcome = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.POST_TURN,
        slice_outcomes=slice_outcomes,
    )
    if post_turn_outcome is not None:
        hooks.last_slice_finished_at = time.monotonic()

    return TurnResult(
        response_text=content,
        tool_call_count=total_tool_calls,
        iteration_count=iteration,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=total_cached_tokens,
        reasoning_tokens=total_reasoning_tokens,
        duration_ms=total_duration_ms,
        latency_ms=total_latency_ms,
        slice_outcomes=slice_outcomes,
    )


def _build_minimal_context(
    *,
    mode_id: str,
    query: str,
    resolve_model_fn: Any,
    complete_fn: Any,
    execute_tool_fn: Any,
    execute_operation_fn: Any,
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext for the turn runner."""
    return OperationExecutionContext(
        workspace_id="lite-host",
        user_id="lite-user",
        user_slug="lite-user",
        user_name="Lite User",
        conversation_id="lite-host-console",
        surface_type="personal_dm",
        surface_id="lite-host-console",
        memory_scope="personal",
        mode_id=mode_id,
        query=query,
        model_id="default",
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
    )


def _stringify_result(result: OperationResult | ToolResult) -> str:
    """Convert a tool or operation result into plain text for model consumption."""
    output_text = getattr(result, "output_text", "") or ""
    output_json = getattr(result, "output_json", None) or {}
    if output_text.strip():
        return output_text.strip()
    if output_json:
        import json as _json

        return _json.dumps(output_json)
    return "Done"


def _coerce_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _extract_completion_usage(
    completion: Any,
) -> tuple[int, int, int, int, int, float, int]:
    """Extract token, sub-counter, duration, and latency counts from one completion.

    Returns a 7-tuple ``(prompt_tokens, completion_tokens, total_tokens,
    cached_tokens, reasoning_tokens, duration_ms, latency_ms)``.

    The function reads both the flat attributes some hosts set on the
    completion object (``prompt_tokens`` etc.) and the nested ``usage``
    object the kernel gateways populate. Sub-counters (``cached_tokens``,
    ``reasoning_tokens``) and ``duration_ms`` are read defensively so a
    missing field never raises; they default to 0 when the provider did
    not surface them.
    """

    prompt_tokens = _coerce_non_negative_int(getattr(completion, "prompt_tokens", 0))
    completion_tokens = _coerce_non_negative_int(getattr(completion, "completion_tokens", 0))
    total_tokens = _coerce_non_negative_int(getattr(completion, "total_tokens", 0))
    cached_tokens = _coerce_non_negative_int(getattr(completion, "cached_tokens", 0))
    reasoning_tokens = _coerce_non_negative_int(getattr(completion, "reasoning_tokens", 0))
    duration_ms = _coerce_non_negative_float(getattr(completion, "duration_ms", 0.0))
    latency_ms = _coerce_non_negative_int(getattr(completion, "latency_ms", 0))

    usage = getattr(completion, "usage", None)
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    usage_cached = 0
    usage_reasoning = 0
    if isinstance(usage, dict):
        usage_prompt = _coerce_non_negative_int(
            usage.get("prompt_tokens") or usage.get("input_tokens")
        )
        usage_completion = _coerce_non_negative_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )
        usage_total = _coerce_non_negative_int(usage.get("total_tokens"))
        usage_cached = _coerce_non_negative_int(usage.get("cached_tokens"))
        usage_reasoning = _coerce_non_negative_int(usage.get("reasoning_tokens"))
    elif usage is not None:
        usage_prompt = _coerce_non_negative_int(
            getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        )
        usage_completion = _coerce_non_negative_int(
            getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        )
        usage_total = _coerce_non_negative_int(getattr(usage, "total_tokens", None))
        usage_cached = _coerce_non_negative_int(
            getattr(usage, "cached_tokens", None)
        )
        usage_reasoning = _coerce_non_negative_int(
            getattr(usage, "reasoning_tokens", None)
        )

    prompt = prompt_tokens or usage_prompt
    completion_value = completion_tokens or usage_completion
    total_value = total_tokens or usage_total
    if total_value <= 0:
        total_value = prompt + completion_value
    cached = cached_tokens or usage_cached
    reasoning = reasoning_tokens or usage_reasoning

    return prompt, completion_value, total_value, cached, reasoning, duration_ms, latency_ms
