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
    roles_needed: tuple[str, ...] = ()

    # ── Content / outcome accumulators ───────────────────────────────
    content: str = ""
    transitioned_to: str | None = None
    handoff_to: str | None = None
    fork_results: list[Any] | None = None
    workflow_terminated: bool = False
    iteration: int = 0

    # ── Chain routing accumulator ────────────────────────────────────
    chain_satisfied: set[str] = field(default_factory=set)


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
    try:
        await on_stage_hook(label, data or {})
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
       with a widened ``RouterRequest``.
    2. ``ModelPolicy.choose`` — the legacy duck-typed optional method.
    3. Static ``model_def`` — the once-resolved ``preferred_model_id``.
    """
    from kernel.seams.protocols import RouterDecision, RouterRequest  # noqa: PLC0415

    router = hooks.context_seams.model_router
    if router is not None:
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
            roles_needed=state.roles_needed,
        )
        try:
            decision = router.choose(req)
        except Exception:  # pylint: disable=broad-except
            log.warning(
                "model_router.choose failed (iteration=%d): %s",
                iteration_num,
                exc_info=True,
            )
            decision = None
        if isinstance(decision, RouterDecision) and decision.model_id.strip():
            resolved = resolve_model(decision.model_id.strip())
            if resolved is not None:
                return resolved
            log.warning(
                "model_router returned unknown model_id %r; falling back",
                decision.model_id,
            )
        elif isinstance(decision, str) and decision.strip():
            # Tolerate routers that return a bare model_id string.
            resolved = resolve_model(decision.strip())
            if resolved is not None:
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
            return resolve_model(chosen_id.strip())
    return model_def