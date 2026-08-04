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
    "_fork_yield_policy_override",
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


def _fork_mode_action_tool_ids(mode_id: str) -> set[str]:
    """Return the authored *side-effecting* action tool ids a fork child must invoke.

    The subset of the mode's ``allowed_operation_ids`` /
    ``allowed_tool_ids`` whose ``effect`` is ``write`` or
    ``external_write`` — the tools that actually *land* the mode's work
    on disk or in memory (e.g. ``markdown_forge_document``, ``write_file``,
    ``synthesize_and_store_learning``, ``coding_apply_patch``). A fork
    child that completes without invoking any of these hallucinated its
    success: it did read-only fetches / searches but never wrote, learned,
    or applied anything.

    Read-effect authored tools (e.g. ``ponder``'s ``web_search``) are
    intentionally excluded. A reasoning mode's *output* is its response
    text (the opinion/analysis), not a side-effect; its authored
    ``web_search`` is one research aid among several (``fetch_page`` is
    another), so demanding it specifically would false-positive on a child
    that gathered evidence via ``fetch_page`` and produced a real opinion.
    The 2026-07-27 third run hit exactly this: ``ponder`` called
    ``fetch_page`` 3 times and was wrongly flagged ``no_tools`` because it
    skipped ``web_search``. Only side-effecting tools are load-bearing for
    "did the child do its job?"
    """
    mode_config = get_workflow_catalog().get(mode_id)
    if mode_config is None:
        return set()
    authored_ops = set(mode_config.authored_allowed_operation_ids)
    authored_tools = set(mode_config.authored_allowed_tool_ids)
    if not authored_ops and not authored_tools:
        return set()
    # Resolve operation effects via the in-process operation catalog. Tools
    # are loaded from the catalog MCP server (a separate process) and are
    # not reliably present in the in-process tool catalog during a fork
    # child turn, so a tool id not found in the operation catalog is
    # treated as a candidate action tool (conservative: include it). This
    # correctly excludes ``ponder``'s ``web_search`` (an operation,
    # read-effect) while including ``markdown``'s ``write_file`` /
    # ``markdown_generate`` (tools, not in the op catalog → included).
    from kernel.operations.registry import get_operation_catalog as _get_ops  # noqa: PLC0415

    op_catalog = _get_ops()
    result: set[str] = set()
    for op_id in authored_ops:
        op_def = op_catalog.get(op_id)
        if op_def is None:
            # Unknown operation — include it (can't prove it's read-only).
            result.add(op_id)
            continue
        effect = str(getattr(op_def, "effect", "read") or "read")
        if effect in ("write", "external_write"):
            result.add(op_id)
    # Tools not resolvable in-process: include them. A mode that authors a
    # tool in its allowlist did so because the tool does its work; we only
    # exclude operations we can prove are read-effect.
    result.update(authored_tools)
    return result


def _fork_yield_policy_override(mode_id: str) -> YieldPolicy | None:
    """Return the mode's ``fork_yield_policy`` to apply when forked, or ``None``.

    A mode may author a *second* ``YieldPolicy`` in ``_mode.md`` front matter
    under ``fork_yield_policy`` to govern how it behaves when invoked as a
    fork child (parallel fan-out). Fork children run with no interactive
    channel — the user cannot answer a mid-turn question — so a mode that
    allows inline asks in the foreground may forbid them in the background
    (``fork_yield_policy: { allowed: false }``) and the ``yield`` tool is not
    advertised to the child at all.

    Returns ``None`` when the key is absent OR the mode is unknown to the
    catalog. ``None`` is the sentinel ``run_turn``'s ``yield_policy_override``
    param interprets as "use the mode's foreground ``yield_policy``" — so
    fork children inherit the prior behavior (backward compatible).
    """
    mode_config = get_workflow_catalog().get(mode_id)
    if mode_config is None:
        return None
    policy = getattr(mode_config, "fork_yield_policy", None)
    # ``fork_yield_policy`` is ``YieldPolicy | None``; ``None`` means
    # "no fork override". A ``YieldPolicy()`` default (empty) is a real
    # override — it means "allow yield in the background" — so we return it.
    return policy


async def _run_fork_child_turn(
    *,
    target: ForkTarget,
    hooks: TurnHooks,
    child_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    started: float,
    logger_override: logging.Logger | None,
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
            max_tool_iterations=_kernel_settings.fork_child_max_tool_iterations,
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
            # Apply the mode's ``fork_yield_policy`` (authored in
            # ``_mode.md`` front matter) so a fork child's ``yield`` tool
            # surface follows the background policy. A mode that allows
            # inline asks in the foreground may forbid them in the
            # background (``fork_yield_policy: { allowed: false }``) and
            # the ``yield`` tool is not advertised to the child at all.
            # ``None`` (no ``fork_yield_policy`` authored) falls back to the
            # mode's foreground ``yield_policy`` — backward compatible.
            yield_policy_override=_fork_yield_policy_override(target.mode_id),
            # Mark this turn as a fork child so the budget briefing emits
            # a ``## Sub-Turn Contract`` block (framing the iteration
            # budget as a write-deadline and naming the action tools the
            # child must invoke) and the turn loop injects a one-shot
            # midpoint write-nudge. Without this, a fork child sees only
            # ``iters: 49/50`` and read-loops to the cap without writing
            # (the 2026-08-03 incident). The ``action_tool_ids`` are
            # resolved from the mode's authored allowlist inside
            # ``_run_turn_inner`` via ``_fork_mode_action_tool_ids``.
            sub_turn_kind="fork",
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

        action_tool_ids = mode_completion_effects(target.mode_id)
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
        return (
            ForkChildResult(
                target=target,
                status=_fork_status,
                output_text=outcome.output_text,
                error_message=outcome.error_message,
                called_tool_names=list(outcome.called_tool_names),
                yield_action=outcome.yield_action,
                duration_ms=outcome.duration_ms,
            ),
            _fork_status,
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
            execution_policy=execution_policy,
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
    data_dir: str = "",
    delegate_fn: Any = None,
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext for the turn runner."""
    host = build_callable_host(
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=read_authored_text_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
        delegate_fn=delegate_fn,
    )
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
        state_provider=state_provider,
        pack_configuration_resolver=pack_configuration_resolver,
        approval_service=approval_service,
        session_plan_digest=session_plan_digest,
        code_change_artifact_store=code_change_artifact_store,
        data_dir=data_dir,
        host=host,
    )
