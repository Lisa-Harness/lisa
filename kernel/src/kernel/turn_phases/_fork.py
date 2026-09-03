"""Fork fan-out orchestration for the turn runner.

Extracted from ``kernel.turn_runner`` to reduce file complexity. Owns the
parallel fan-out to heterogeneous modes: budget division, child turn
execution, status mapping (``ok``/``no_tools``/``timeout``/``error``/
``cancelled``), and stage-event emission. Also owns
``_build_minimal_context``, the shared ``OperationExecutionContext``
builder used by both the fork path and the main turn loop.

DESIGN — Re-exported by ``kernel.turn_runner``:
    Existing tests import ``_fork_mode_requires_tools`` and
    ``_build_minimal_context`` directly from ``kernel.turn_runner``. That
    module re-exports every symbol below so no test file needs editing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from kernel.config import settings as _kernel_settings
from kernel.execution_policy import ExecutionPolicy
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import (
    ConversationSurface,
    ForkChildResult,
    ForkTarget,
    MemoryScope,
    OperationExecutionContext,
    OperationResult,
    TurnBudget,
    YieldPolicy,
)
from kernel.seams.host_adapter import build_callable_host
from kernel.tools.mcp_client import MCPToolPool

if TYPE_CHECKING:
    from kernel.turn_runner import TurnHooks

__all__ = [
    "_build_minimal_context",
    "_execute_fork_fanout",
    "_fork_child_budget",
    "_fork_mode_requires_tools",
    "_run_fork_child",
    "_run_fork_child_turn",
]


def _fork_mode_requires_tools(mode_id: str) -> bool:
    """Return True when a fork child's mode declares side-effecting action tools.

    Delegates to :func:`kernel.operations.registry.mode_completion_effects` —
    the single canonical resolver shared with the child-turn classifier — so
    the "does this mode require tools?" check and the "which tools?" check
    agree. A mode with no side-effecting completion contract (chat / reasoning)
    returns False; its output is the response text, not a side effect.
    """
    from kernel.operations.registry import (  # noqa: PLC0415  # lazy: breaks cycle with registry
        mode_completion_effects,
    )

    try:
        return bool(mode_completion_effects(mode_id))
    except Exception:  # pylint: disable=broad-except
        return False


async def _run_fork_child_turn(
    *,
    target: ForkTarget,
    hooks: TurnHooks,
    child_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    started: float,
    logger_override: logging.Logger | None,
    max_tool_iterations: int = 0,
) -> tuple[ForkChildResult, str, int]:
    """Run one fork child turn and map the outcome to a ``ForkChildResult``.

    Returns ``(result, status, elapsed_ms)`` so the caller can emit a
    ``fork_child_complete`` stage event without re-deriving the status.
    Extracted from ``_execute_fork_fanout`` to keep that function under
    the McCabe complexity cap.

    A child whose mode declares action operations/tools (``remember``,
    ``markdown``, etc.) but completes with **zero tool calls** is marked
    ``status="no_tools"`` rather than ``"ok"``. Such a child typically
    produced a hallucinated "Done — wrote it to disk" text response
    without invoking its write/remember tool, leaving nothing on disk or
    in the memory store. The parent must not report these as succeeded.
    """
    _flog = logger_override or logging.getLogger(__name__)
    _flog.debug(
        "fork child starting: mode=%s stage=%s",
        target.mode_id,
        target.stage_id,
    )
    try:
        # Lazy import to avoid a circular dependency: ``kernel.turn_runner``
        # re-exports this module, so importing it at module top would create
        # an import cycle. The function-level import is cheap (module is
        # already cached in ``sys.modules`` after the first turn).
        from kernel.turn_runner import run_turn as _run_turn  # noqa: PLC0415

        child_result = await _run_turn(
            mode_id=target.mode_id,
            query=target.query,
            hooks=hooks,
            max_tool_iterations=(
                max_tool_iterations or _kernel_settings.fork_child_max_tool_iterations
            ),
            turn_budget=child_budget,
            execution_policy=execution_policy,
            logger_override=_flog,
            # Fork children must NOT re-route. A hint telling a forked
            # markdown/search child to hand off again creates a loop and,
            # with strict tool schemas, invalid-content 500s from the
            # provider. The child already runs in its target mode; let it
            # run to completion without re-injecting the original query's
            # multi-intent routing signal.
            suppress_routing_hint=True,
            # Fork children never get the ``yield`` tool. A fork target
            # declares ``fork_allowed: true`` — it commits to completing or
            # failing without asking the user. The unconditional
            # ``YieldPolicy(allowed=False)`` ensures the ``yield`` tool is
            # not advertised and the lifecycle fallback cannot fire.
            yield_policy_override=YieldPolicy(allowed=False),
            # Mark this turn as a fork child so the budget briefing emits
            # a ``## Sub-Turn Contract`` block (framing the iteration
            # budget as a write-deadline and naming the action tools the
            # child must invoke) and the turn loop injects a one-shot
            # midpoint write-nudge. Without this, a fork child sees only
            # ``iters: 49/50`` and read-loops to the cap without writing
            # (the 2026-08-03 incident). The ``action_tool_ids`` are
            # resolved from the mode's declared completion effects inside
            # ``_run_turn_inner`` via ``mode_completion_effects``.
            sub_turn_kind="fork",
            # Pin the child to the target's stage so a forked workflow
            # stage runs its own stage instruction rather than the mode's
            # default (or the parent session's current stage). Previously
            # ``ForkTarget.stage_id`` was parsed and telemetry-only but
            # never applied to the child turn.
            stage_id_override=target.stage_id or None,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        # ── Shared child-turn classification ───────────────────────────
        # Delegate to ``classify_child_turn`` so the fork and delegate
        # paths share one classifier, one status union, and one completion
        # contract. This fixes the 2026-08-04 audit findings:
        # 3.1 — ``TurnResult.blocked`` is now checked first, so a
        # capability-gated child surfaces ``blocked`` (router declined to
        # escalate) instead of ``no_tools`` ("model too weak").
        # 4.1 — ``mode_completion_effects`` replaces ``_fork_mode_action_tool_ids``
        # so operations admitted via their own ``mode_allowlist`` are
        # included in the action-tool set.
        # 7.1 — the duplicated classifier is removed; both paths call the
        # same function.
        from kernel.operations.registry import (  # noqa: PLC0415  # lazy: breaks cycle with registry
            mode_completion_effects,
        )
        from kernel.turn_phases._child_turn import (  # noqa: PLC0415  # lazy: breaks cycle with turn_runner
            classify_child_turn,
        )

        action_tool_ids = mode_completion_effects(
            target.mode_id,
            mcp_pool=getattr(hooks, "mcp_tool_pool", None),
        )
        outcome = classify_child_turn(
            child_result=child_result,
            mode_id=target.mode_id,
            elapsed_ms=elapsed_ms,
            action_tool_ids=action_tool_ids,
        )
        # Map the canonical ``ChildTurnOutcome`` to the host-facing
        # ``ForkChildResult``. The ``status`` mapping is 1:1 except
        # ``complete`` → ``ok`` (the legacy fork status) so existing
        # callers and tests that inspect ``r.status == "ok"`` are preserved.
        _fork_status = "ok" if outcome.status == "complete" else outcome.status
        _flog.debug(
            "fork child classified: mode=%s status=%s tool_calls=%d duration_ms=%d",
            target.mode_id,
            _fork_status,
            child_result.tool_call_count,
            elapsed_ms,
        )
        return (
            ForkChildResult(
                target=target,
                status=_fork_status,
                output_text=outcome.output_text,
                error_message=outcome.error_message,
                called_tool_names=list(outcome.called_tool_names),
                yield_action=outcome.yield_action,
                duration_ms=outcome.duration_ms,
                per_model_usage=[dict(e) for e in outcome.per_model_usage],
                per_mode_usage=[dict(e) for e in outcome.per_mode_usage],
            ),
            _fork_status,
            elapsed_ms,
        )
    except TimeoutError:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _flog.debug("fork child timed out: mode=%s", target.mode_id)
        return (
            ForkChildResult(
                target=target,
                status="timeout",
                error_message="Child turn timed out.",
                duration_ms=elapsed_ms,
            ),
            "timeout",
            elapsed_ms,
        )
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _flog.warning("fork child %s failed: %s: %s", target.mode_id, type(exc).__name__, exc)
        return (
            ForkChildResult(
                target=target,
                status="error",
                error_message=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms,
            ),
            "error",
            elapsed_ms,
        )


def _fork_child_budget(turn_budget: TurnBudget | None, n: int) -> TurnBudget | None:
    """Divide the parent turn budget equally across ``n`` children.

    A field set to ``0`` means unlimited and is passed through as ``0``.
    Returns ``None`` when the parent has no budget.
    """
    if turn_budget is None:
        return None
    return TurnBudget(
        max_tool_calls=(turn_budget.max_tool_calls // n if turn_budget.max_tool_calls else 0),
        max_total_tokens=(turn_budget.max_total_tokens // n if turn_budget.max_total_tokens else 0),
        max_completion_tokens=(
            turn_budget.max_completion_tokens // n if turn_budget.max_completion_tokens else 0
        ),
        max_wall_ms=(turn_budget.max_wall_ms // n if turn_budget.max_wall_ms else 0),
    )


async def _run_fork_child(
    *,
    idx: int,
    target: ForkTarget,
    results: list[ForkChildResult | None],
    turn_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    n: int,
    hooks: TurnHooks,
    emit_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None,
    logger_override: logging.Logger | None,
    max_tool_iterations: int = 0,
) -> None:
    """Run one fork child: emit start, run the child turn, store result, emit complete.

    Extracted from ``_execute_fork_fanout`` to keep that orchestrator under
    the McCabe complexity cap. ``results`` is mutated in place at ``idx``.
    """
    _flog = logger_override or logging.getLogger(__name__)
    started = time.monotonic()
    child_budget = _fork_child_budget(turn_budget, n)
    if emit_stage is not None:
        await emit_stage(
            "fork_child_start",
            {
                "child_index": idx,
                "mode_id": target.mode_id,
                "stage_id": target.stage_id,
                "query_preview": (target.query or "")[:80],
            },
        )
    result, status, elapsed_ms = await _run_fork_child_turn(
        target=target,
        hooks=hooks,
        child_budget=child_budget,
        execution_policy=execution_policy,
        started=started,
        logger_override=_flog,
        max_tool_iterations=max_tool_iterations,
    )
    results[idx] = result
    if emit_stage is not None:
        await emit_stage(
            "fork_child_complete",
            {
                "child_index": idx,
                "mode_id": target.mode_id,
                "stage_id": target.stage_id,
                "status": status,
                "duration_ms": elapsed_ms,
            },
        )


async def _execute_fork_fanout(
    *,
    fork_targets: list[ForkTarget],
    hooks: TurnHooks,
    pool: MCPToolPool,
    resolve_model: Callable[[str], Any],
    complete: Callable[..., Awaitable[Any]],
    exec_op: Callable[..., Awaitable[OperationResult]],
    turn_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    turn_start_mono: float,
    logger_override: logging.Logger | None,
    emit_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    max_tool_iterations: int = 0,
) -> list[ForkChildResult]:
    """Execute a parallel fork fan-out to heterogeneous modes.

    Each child runs in an isolated session with its own mode's system prompt
    and tool surface. Children are fanned out via ``asyncio.TaskGroup`` and
    collected into a list of ``ForkChildResult``.

    The parent turn's ``TurnBudget`` is divided equally across N children.
    Each child gets a sub-budget. If a child exhausts its sub-budget, it
    returns a partial result.

    ``emit_stage`` is the parent turn's ``_emit_stage`` helper. When set,
    the fan-out emits ``fork_start`` / ``fork_child_start`` (one per
    child) / ``fork_child_complete`` (one per child) / ``fork_complete``
    so a transport can render one line per active child. Child turns
    run with ``on_stage`` suppressed on their hooks so their internal
    stage events do not mix into the parent's stage line.
    """
    _flog = logger_override or logging.getLogger(__name__)
    n = len(fork_targets)
    if n == 0:
        return []

    _flog.debug(
        "fork fan-out starting: children=%d targets=%s",
        n,
        [t.mode_id for t in fork_targets],
    )

    if emit_stage is not None:
        await emit_stage(
            "fork_start",
            {
                "children": n,
                "targets": [{"mode_id": t.mode_id, "stage_id": t.stage_id} for t in fork_targets],
            },
        )

    results: list[ForkChildResult | None] = [None] * n

    async def _run_child(idx: int, target: ForkTarget) -> None:
        await _run_fork_child(
            idx=idx,
            target=target,
            results=results,
            turn_budget=turn_budget,
            execution_policy=execution_policy,
            n=n,
            hooks=hooks,
            emit_stage=emit_stage,
            logger_override=_flog,
            max_tool_iterations=max_tool_iterations,
        )

    try:
        async with asyncio.TaskGroup() as tg:
            for idx, target in enumerate(fork_targets):
                tg.create_task(_run_child(idx, target))
    except ExceptionGroup as eg:  # pylint: disable=broad-except
        _flog.warning("fork fan-out TaskGroup had errors: %s", eg.exceptions)

    _flog.debug(
        "fork fan-out complete: statuses=%s",
        [r.status if r else "cancelled" for r in results],
    )

    # Fill in any None slots as cancelled
    final_results: list[ForkChildResult] = []
    for idx, r in enumerate(results):
        if r is not None:
            final_results.append(r)
        else:
            final_results.append(
                ForkChildResult(
                    target=fork_targets[idx],
                    status="cancelled",
                    error_message="Child turn was cancelled.",
                )
            )
    if emit_stage is not None:
        await emit_stage(
            "fork_complete",
            {
                "children": n,
                "ok": sum(1 for r in final_results if r.status == "ok"),
                "failed": sum(1 for r in final_results if r.status != "ok"),
            },
        )
    return final_results


def _build_minimal_context(
    *,
    mode_id: str,
    query: str,
    resolve_model_fn: Any,
    complete_fn: Any,
    read_authored_text_fn: Any,
    execute_tool_fn: Any,
    execute_operation_fn: Any,
    state_provider: Any = None,
    pack_configuration_resolver: Any = None,
    approval_service: Any = None,
    code_change_artifact_store: Any = None,
    workspace_id: str | None = None,
    user_id: str | None = None,
    user_slug: str | None = None,
    user_name: str | None = None,
    principal_key: str | None = None,
    conversation_id: str | None = None,
    surface_type: str | None = None,
    surface_id: str | None = None,
    memory_scope: str | None = None,
    session_plan_digest: str = "",
    data_dir: str = "",
    delegate_fn: Any = None,
    fork_fn: Any = None,
    conversation_excerpt_markdown: str = "",
    response: str = "",
    gateway_provider_fn: Any = None,
    model_router: Any = None,
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext for the turn runner.

    ``gateway_provider_fn`` is forwarded to ``build_callable_host`` so the
    host's ``complete`` dispatches through the provider gateway (the live
    completion path) instead of the removed legacy callable. Callers pass a
    zero-arg callable that resolves ``hooks.context_seams.provider_gateways``.
    """
    host = build_callable_host(
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=read_authored_text_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
        delegate_fn=delegate_fn,
        fork_fn=fork_fn,
        gateway_provider_fn=gateway_provider_fn,
    )
    # Identity defaults mirror the ``OperationExecutionContext`` model
    # defaults (single-user personal-DM console). Production hosts wire
    # their own identity; bare/test contexts get the canonical defaults.
    _ws_id = workspace_id if workspace_id is not None else "default"
    _u_id = user_id if user_id is not None else "default"
    _u_slug = user_slug if user_slug is not None else "default"
    _u_name = user_name if user_name is not None else ""
    _pk = principal_key if principal_key is not None else "default"
    _conv_id = conversation_id if conversation_id is not None else "default"
    _surf_type = surface_type if surface_type is not None else "personal_dm"
    _surf_id = surface_id if surface_id is not None else "default"
    _mem_scope = memory_scope if memory_scope is not None else "personal"
    return OperationExecutionContext(
        workspace_id=_ws_id,
        user_id=_u_id,
        user_slug=_u_slug,
        user_name=_u_name,
        principal_key=_pk,
        conversation_id=_conv_id,
        surface_type=cast(ConversationSurface, _surf_type),
        surface_id=_surf_id,
        memory_scope=cast(MemoryScope, _mem_scope),
        mode_id=mode_id,
        query=query,
        model_id="default",
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        response=response,
        state_provider=state_provider,
        pack_configuration_resolver=pack_configuration_resolver,
        approval_service=approval_service,
        session_plan_digest=session_plan_digest,
        code_change_artifact_store=code_change_artifact_store,
        data_dir=data_dir,
        host=host,
        model_router=model_router,
    )
