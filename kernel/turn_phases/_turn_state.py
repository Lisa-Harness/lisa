"""Mutable turn-state container and closure-free helpers for ``_run_turn_inner``.

This module owns the per-turn mutable state that was previously captured by
nested closures inside ``kernel.turn_runner._run_turn_inner``.  The five
closures (``_emit_stage``, ``_mcp_tool_executor``, ``_record_model_usage``,
``_budget_forces_final``, ``_resolve_iteration_model``) have been promoted to
module-level functions that accept the state explicitly as a ``_TurnState``
dataclass.  This makes them independently testable and eliminates the closure
capture of mutable parent state.

DESIGN — explicit state passing, not global:
    ``_TurnState`` is instantiated once per turn and threaded through every
    helper call.  No module-level mutable singletons.  The turn runner
    mutates fields on the instance (e.g. ``state.total_tool_calls += 1``) and
    reads them back in the loop.  The state object is never shared across
    turns or across concurrent fork children.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kernel.models import ToolResult
from kernel.seams.protocols import RouterModelRef
from kernel.tools.registry import execute_tool

if TYPE_CHECKING:
    from kernel.models import TurnBudget
    from kernel.tools.mcp_client import MCPToolPool

logger = logging.getLogger(__name__)


@dataclass
class _TurnState:
    """Mutable per-turn counters and accumulators.

    Instantiated once at the top of ``_run_turn_inner`` and threaded through
    every helper call.  Fields are intentionally mutable (no ``frozen=True``)
    — the turn loop updates them in place.
    """

    # ── Token / usage accumulators ───────────────────────────────────
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_duration_ms: float = 0.0
    total_latency_ms: int = 0
    total_tool_calls: int = 0
    per_model_usage: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ── Router-relevant state ────────────────────────────────────────
    last_tool_result_text: str | None = None
    required_capabilities: tuple[str, ...] = ()
    # Metadata of the model effective in the preceding iteration. Set after
    # the first completion; on iteration 1 the router receives the static
    # preferred model via RouterRequest.effective_model.
    effective_model: Any = None
    # When the router returns a terminal action (finish / transition_to),
    # resolve_iteration_model stores it here so the turn loop can act on
    # it after the model-selection path. Cleared at the start of each
    # iteration.
    pending_router_decision: Any = None

    # ── Content / outcome accumulators ───────────────────────────────
    content: str = ""
    transitioned_to: str | None = None
    handoff_to: str | None = None
    fork_results: list[Any] | None = None
    workflow_terminated: bool = False
    iteration: int = 0
    # ── Per-step iteration sub-budget ────────────────────────────────
    # Iterations and tool calls consumed by the *current* workflow step
    # only, reset to 0 on every successful ``transition_stage`` (the same
    # site as the durable instance advancement). The global ``iteration``
    # counter is NOT reset on handoffs — it bounds the whole turn — but
    # ``step_iteration`` bounds a single step so a long chained workflow
    # (default→plan→search→ponder→coding→markdown) does not starve its
    # final step on the shared ``max_tool_iterations`` budget. When the
    # per-step cap binds before the global cap, the kernel synthesizes a
    # handoff to the next remaining step instead of running the generic
    # empty-content final completion (see Phase 3 of the durable-workflow
    # plan, 2026-07-29). When no ``WorkflowInstance`` is wired, these
    # counters are unused and the global cap alone governs the loop.
    step_iteration: int = 0
    """Model-completion iterations consumed by the current workflow step.
    Reset to 0 on every ``transition_stage``; bounded by
    ``per_step_max_iterations`` from ``kernel.config.Settings``."""
    step_tool_calls: int = 0
    """Tool calls dispatched during the current workflow step. Reset to 0
    on every ``transition_stage``; bounded by ``per_step_max_tool_calls``
    as a secondary guard against a model stuck in a tool-calling rut."""

    # ── Chain routing accumulator ────────────────────────────────────
    chain_satisfied: set[str] = field(default_factory=set)

    # ── Hard-failure / loop-guard state ───────────────────────────────
    entry_failed: bool = False
    """True when an entry operation for a handoff target failed this turn
    (e.g. ``web_search`` returned no content). Once set, the turn must not
    continue in the data-less target mode — the model would drift into an
    action mode and hallucinate a solution (the 2026-07-25 ``crew/``
    incident). The handoff handler sets this and breaks the loop with a
    deterministic user-facing failure instead of ``continue``-ing."""
    entry_failed_mode: str = ""
    """The mode_id whose entry operation failed, for the user-facing
    failure message (e.g. ``"search"``)."""
    entry_failed_reason: str = ""
    """Short human-readable failure reason from the caught exception,
    surfaced to the user alongside the mode name."""
    handoff_count: int = 0
    """Number of mid-turn mode handoffs executed this turn. Bounded by
    ``_MAX_HANDOFFS_PER_TURN`` as a hard backstop against routing-hint
    loops that the ``chain_satisfied`` accumulator fails to break (e.g.
    when a pack omits ``routing_intent_tags`` or ``_detect_query_intents``
    over-detects)."""
    last_hint_text: str = ""
    """The most recently injected routing-hint text. Used by the
    duplicate-hint dedup guard: when the recomputed post-handoff hint is
    byte-identical to the previous one for >=2 consecutive handoffs, the
    injection is suppressed and the max-hops guard takes over."""
    duplicate_hint_count: int = 0
    """Consecutive count of identical routing hints. Resets to 0 on a
    distinct hint; when it reaches ``_DUPLICATE_HINT_LIMIT`` the hint is
    suppressed."""
    forced_fork_target: str = ""
    """The deterministic ``fork:mode,mode,...`` target the kernel will
    execute itself if the model refuses to call ``transition_stage``.
    Populated when a fork routing hint is active (see
    ``_compute_fork_target_string``); consumed by the forced-fork
    enforcer in the turn loop. Empty when the routing decision is a
    single handoff, a sequential chain, or no gap at all."""
    pending_planned_handoff: str = ""
    """A ``target_stage_id`` produced by a planner action mode's entry
    operation (e.g. ``plan_workflow``) during a mid-turn handoff. When
    set, the turn loop synthesizes a ``transition_stage`` tool call for
    this target at the top of the next iteration — before calling the
    model — so the kernel drives the next handoff deterministically
    instead of begging the model to call ``transition_stage`` (the
    2026-07-28 stall: the model in ``plan`` mode read the plan result
    and stopped). Cleared after the synthesized dispatch."""
    non_fork_tool_dispatches: int = 0
    """Count of tool dispatches this turn that were NOT
    ``transition_stage`` with the expected ``fork:`` target, while a
    ``forced_fork_target`` is set. The enforcer fires when this reaches
    ``_FORCED_FORK_NON_FORK_TOOL_CAP``."""
    called_tool_names: set[str] = field(default_factory=set)
    """Names of every tool actually dispatched this turn (operations,
    pack tools, MCP tools). Used by the fork-child validator to confirm
    that an action mode's child invoked at least one of its authored
    action tools — not just read-only fetches. A child that did 4
    ``fetch_page`` calls but never called ``markdown_forge_document``
    hallucinated its success (the 2026-07-27 follow-on incident)."""


# ── Closure-free helper functions ──────────────────────────────────────────


async def emit_stage(
    state: _TurnState,
    on_stage_hook: Any,
    log: logging.Logger,
    label: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Wrap ``hooks.on_stage`` with None-check and try/except.

    Failures are logged and swallowed — a transport error never breaks the
    turn.
    """
    if on_stage_hook is None:
        return
    payload = dict(data or {})
    payload.setdefault("iteration", state.iteration)
    payload.setdefault("_state.iteration", state.iteration)
    try:
        await on_stage_hook(label, payload)
    except Exception as stage_exc:  # pylint: disable=broad-except
        log.debug("on_stage(%s) hook failed: %s", label, stage_exc)


async def mcp_tool_executor(
    tool_id: str,
    _ctx: Any,
    pool: MCPToolPool,
) -> ToolResult:
    """Bridge ``OperationExecutionContext.execute_tool_fn`` to in-process tools.

    Catalog-driven: if the tool_id is in the internal tool catalog (discovered
    from packs), dispatch in-process via ``tools.registry.execute_tool``.
    Otherwise fall through to the MCP pool.  Declarative operations that need
    to call tools route through here.
    """
    from kernel.tools.registry import get_tool_catalog as _get_tools  # noqa: PLC0415
    from kernel.turn_phases._post_turn import _to_tool_context  # noqa: PLC0415

    tool_catalog = _get_tools()
    definition = tool_catalog.get(tool_id)
    if definition is not None and definition.implementation_module is not None:
        tool_ctx = _to_tool_context(tool_id, _ctx)
        return await execute_tool(tool_id, tool_ctx)

    params = getattr(_ctx, "parameters", {}) or {}
    mcp_result = await pool.call_tool(tool_id, params)
    return ToolResult(
        tool_id=tool_id,
        output_text=mcp_result.output_text,
        error=mcp_result.output_text if mcp_result.is_error else "",
    )


def record_model_usage(
    state: _TurnState,
    model_def: Any,
    *,
    p: int,
    c: int,
    t: int,
    ct: int,
    rt: int,
    d: float,
    lat: int,
) -> None:
    """Merge one completion's counters into the per-model accumulator."""
    mid = (getattr(model_def, "model_id", "") or "-").strip()
    entry = state.per_model_usage.setdefault(mid, {"model_id": mid})
    entry["prompt_tokens"] = entry.get("prompt_tokens", 0) + p
    entry["completion_tokens"] = entry.get("completion_tokens", 0) + c
    entry["total_tokens"] = entry.get("total_tokens", 0) + t
    entry["cached_tokens"] = entry.get("cached_tokens", 0) + ct
    entry["reasoning_tokens"] = entry.get("reasoning_tokens", 0) + rt
    entry["duration_ms"] = entry.get("duration_ms", 0.0) + d
    entry["latency_ms"] = entry.get("latency_ms", 0) + lat


def budget_forces_final(
    state: _TurnState,
    turn_budget: TurnBudget | None,
    turn_start_mono: float,
) -> bool:
    """Return True when the turn budget is exhausted.

    A budget field set to 0 means unlimited and never forces.  When any
    nonzero field is exhausted, the model gets one more call — but with no
    tools advertised so it must produce a final text response.
    """
    if turn_budget is None:
        return False
    if turn_budget.max_tool_calls and state.total_tool_calls >= turn_budget.max_tool_calls:
        return True
    if turn_budget.max_total_tokens and state.total_tokens >= turn_budget.max_total_tokens:
        return True
    if turn_budget.max_wall_ms:
        elapsed_ms = int((time.monotonic() - turn_start_mono) * 1000)
        if elapsed_ms >= turn_budget.max_wall_ms:
            return True
    return False


def resolve_iteration_model(
    state: _TurnState,
    iteration_num: int,
    hooks: Any,
    preferred_model_id: str,
    resolve_model: Any,
    model_def: Any,
    query: str,
    mode_id: str,
    turn_budget: TurnBudget | None,
    log: logging.Logger,
) -> Any:
    """Three-tier per-iteration model selection.

    1. ``ContextSeams.model_router`` — the formal ``ModelRouter`` Protocol
       with a widened ``RouterRequest`` (now including ``effective_model``).
    2. ``ModelPolicy.choose`` — the legacy duck-typed optional method.
    3. Static ``model_def`` — the once-resolved ``preferred_model_id``.

    When the router returns a terminal action (``finish`` or
    ``transition_to``), the decision is stored on
    ``state.pending_router_decision`` and the static ``model_def`` is
    returned — the turn loop inspects the pending decision after this call
    and forces a final completion or a workflow transition accordingly.
    """
    from kernel.seams.protocols import (  # noqa: PLC0415
        RouterDecision,
        RouterRequest,
    )

    state.pending_router_decision = None

    router = hooks.context_seams.model_router
    if router is not None:
        effective_ref = _effective_model_ref(state, model_def, preferred_model_id)
        req = RouterRequest(
            default_model_id=preferred_model_id,
            iteration=iteration_num,
            tool_calls_so_far=state.total_tool_calls,
            budget_remaining=(
                None
                if turn_budget is None
                else max(0, turn_budget.max_tool_calls - state.total_tool_calls)
                if turn_budget.max_tool_calls
                else None
            ),
            query=query,
            last_tool_result=state.last_tool_result_text,
            required_capabilities=state.required_capabilities,
            mode_id=mode_id,
            effective_model=effective_ref,
        )
        try:
            decision = router.choose(req)
        except Exception:  # pylint: disable=broad-except
            log.warning(
                "model_router.choose failed (iteration=%d)",
                iteration_num,
                exc_info=True,
            )
            decision = None
        if isinstance(decision, RouterDecision):
            if decision.finish.strip():
                state.pending_router_decision = decision
                return model_def
            if decision.transition_to.strip():
                state.pending_router_decision = decision
                return model_def
            if decision.model_id.strip():
                resolved = resolve_model(decision.model_id.strip())
                if resolved is not None:
                    state.effective_model = resolved
                    return resolved
                log.warning(
                    "model_router returned unknown model_id %r; falling back",
                    decision.model_id,
                )
        elif isinstance(decision, str) and decision.strip():
            # Tolerate routers that return a bare model_id string.
            resolved = resolve_model(decision.strip())
            if resolved is not None:
                state.effective_model = resolved
                return resolved
    # Legacy duck-typed ``choose`` on ModelPolicy.
    policy = hooks.context_seams.model_policy
    choose_fn = getattr(policy, "choose", None)
    if callable(choose_fn):
        chosen_id = policy.choose(
            default_model_id=preferred_model_id,
            iteration=iteration_num,
            tool_calls_so_far=state.total_tool_calls,
            budget_remaining=(
                None
                if turn_budget is None
                else max(0, turn_budget.max_tool_calls - state.total_tool_calls)
                if turn_budget.max_tool_calls
                else None
            ),
        )
        if isinstance(chosen_id, str) and chosen_id.strip():
            resolved = resolve_model(chosen_id.strip())
            if resolved is not None:
                state.effective_model = resolved
                return resolved
    state.effective_model = model_def
    return model_def


def _effective_model_ref(
    state: _TurnState,
    model_def: Any,
    preferred_model_id: str,
) -> RouterModelRef:
    """Build a RouterModelRef from the effective/preceding model.

    On iteration 1, ``state.effective_model`` is None (nothing has run yet),
    so we use ``model_def`` — the statically-resolved preferred model. On
    later iterations, ``state.effective_model`` carries the model actually
    selected in the preceding iteration.
    """
    source = state.effective_model if state.effective_model is not None else model_def
    if source is None:
        return RouterModelRef(model_id=preferred_model_id)
    return RouterModelRef(
        model_id=getattr(source, "model_id", "") or preferred_model_id,
        provider=getattr(source, "provider", "") or "",
        cost_tier=getattr(source, "cost_tier", "") or "",
        latency_tier=getattr(source, "latency_tier", "") or "",
        capabilities=tuple(getattr(source, "capabilities", ()) or ()),
        reasoning_model=bool(getattr(source, "reasoning_model", False)),
        thinking_model=bool(getattr(source, "thinking_model", False)),
    )