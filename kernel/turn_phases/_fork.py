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

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import (
    ConversationSurface,
    ForkChildResult,
    ForkTarget,
    MemoryScope,
    OperationExecutionContext,
    OperationResult,
    TurnBudget,
)
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
    """Return True when a fork child's mode declares action operations/tools.

    Modes like ``learn`` and ``markdown`` exist to *take action* (distill +
    persist memories, write documents). Their ``_mode.md`` front matter
    declares non-empty ``allowed_operation_ids`` or ``allowed_tool_ids``.
    A fork child running such a mode that completes with **zero tool calls**
    almost certainly produced a hallucinated "Done — wrote it to disk" text
    response without invoking its write/learn tool, leaving nothing on disk
    or in memory. The caller marks these ``status="no_tools"`` instead of
    ``"ok"`` so the parent does not report a false success.

    Chat-only modes (``default``, ``ponder``) with empty allowlists
    legitimately complete without tools, so this returns False for them.
    """
    mode_config = get_workflow_catalog().get(mode_id)
    if mode_config is None:
        return False
    # Use the *authored* allowlists, not the merged (effective) ones, so a
    # chat-only mode that inherits ambient capabilities (e.g. recall_memory
    # via default_available) is not misflagged as requiring tools to complete.
    return bool(mode_config.authored_allowed_operation_ids) or bool(
        mode_config.authored_allowed_tool_ids
    )


async def _run_fork_child_turn(
    *,
    target: ForkTarget,
    hooks: TurnHooks,
    child_budget: TurnBudget | None,
    started: float,
    logger_override: logging.Logger | None,
) -> tuple[ForkChildResult, str, int]:
    """Run one fork child turn and map the outcome to a ``ForkChildResult``.

    Returns ``(result, status, elapsed_ms)`` so the caller can emit a
    ``fork_child_complete`` stage event without re-deriving the status.
    Extracted from ``_execute_fork_fanout`` to keep that function under
    the McCabe complexity cap.

    A child whose mode declares action operations/tools (``learn``,
    ``markdown``, etc.) but completes with **zero tool calls** is marked
    ``status="no_tools"`` rather than ``"ok"``. Such a child typically
    produced a hallucinated "Done — wrote it to disk" text response
    without invoking its write/learn tool, leaving nothing on disk or
    in the memory store. The parent must not report these as succeeded.
    """
    _flog = logger_override or logging.getLogger(__name__)
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
            max_tool_iterations=5,
            turn_budget=child_budget,
            logger_override=_flog,
            # Fork children must NOT re-route. A hint telling a forked
            # markdown/search child to hand off again creates a loop and,
            # with strict tool schemas, invalid-content 500s from the
            # provider. The child already runs in its target mode; let it
            # run to completion without re-injecting the original query's
            # multi-intent routing signal.
            suppress_routing_hint=True,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if child_result.tool_call_count == 0 and _fork_mode_requires_tools(target.mode_id):
            _flog.warning(
                "fork child %s completed with 0 tool calls but its mode declares "
                "action operations/tools; marking no_tools (likely hallucinated success)",
                target.mode_id,
            )
            return (
                ForkChildResult(
                    target=target,
                    status="no_tools",
                    output_text=child_result.response_text,
                    error_message=(
                        f"fork child {target.mode_id} produced no tool calls; its mode "
                        "declares action operations/tools, so its reported success is "
                        "unverified (nothing written to disk or memory)."
                    ),
                    duration_ms=elapsed_ms,
                ),
                "no_tools",
                elapsed_ms,
            )
        return (
            ForkChildResult(
                target=target,
                status="ok",
                output_text=child_result.response_text,
                duration_ms=elapsed_ms,
            ),
            "ok",
            elapsed_ms,
        )
    except TimeoutError:
        elapsed_ms = int((time.monotonic() - started) * 1000)
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
        max_wall_ms=(turn_budget.max_wall_ms // n if turn_budget.max_wall_ms else 0),
    )


async def _run_fork_child(
    *,
    idx: int,
    target: ForkTarget,
    results: list[ForkChildResult | None],
    turn_budget: TurnBudget | None,
    n: int,
    hooks: TurnHooks,
    emit_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None,
    logger_override: logging.Logger | None,
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
        started=started,
        logger_override=_flog,
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
    turn_start_mono: float,
    logger_override: logging.Logger | None,
    emit_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
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
            n=n,
            hooks=hooks,
            emit_stage=emit_stage,
            logger_override=_flog,
        )

    try:
        async with asyncio.TaskGroup() as tg:
            for idx, target in enumerate(fork_targets):
                tg.create_task(_run_child(idx, target))
    except ExceptionGroup as eg:  # pylint: disable=broad-except
        _flog.warning("fork fan-out TaskGroup had errors: %s", eg.exceptions)

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
    workspace_id: str = "lite-host",
    user_id: str = "lite-user",
    user_slug: str = "lite-user",
    user_name: str = "Lite User",
    principal_key: str = "lite-user",
    conversation_id: str = "lite-host-console",
    surface_type: str = "personal_dm",
    surface_id: str = "lite-host-console",
    memory_scope: str = "personal",
    session_plan_digest: str = "",
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext for the turn runner."""
    return OperationExecutionContext(
        workspace_id=workspace_id,
        user_id=user_id,
        user_slug=user_slug,
        user_name=user_name,
        principal_key=principal_key,
        conversation_id=conversation_id,
        surface_type=cast(ConversationSurface, surface_type),
        surface_id=surface_id,
        memory_scope=cast(MemoryScope, memory_scope),
        mode_id=mode_id,
        query=query,
        model_id="default",
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=read_authored_text_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
        state_provider=state_provider,
        pack_configuration_resolver=pack_configuration_resolver,
        approval_service=approval_service,
        session_plan_digest=session_plan_digest,
        code_change_artifact_store=code_change_artifact_store,
    )
