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
from kernel.tools.mcp_client import MCPToolPool, bare_tool_name

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

    Modes like ``remember`` and ``markdown`` exist to *take action* (distill +
    persist memories, write documents). Their ``_mode.md`` front matter
    declares non-empty ``allowed_operation_ids`` or ``allowed_tool_ids``
    that include at least one write/external_write-effect tool. A fork
    child running such a mode that completes with **zero tool calls**
    almost certainly produced a hallucinated "Done — wrote it to disk" text
    response without invoking its write/remember tool, leaving nothing on disk
    or in memory. The caller marks these ``status="no_tools"`` instead of
    ``"ok"`` so the parent does not report a false success.

    Modes whose authored tools are all read-effect (e.g. ``ponder``, whose
    only authored operation is ``web_search``) legitimately complete with
    zero tool calls — their output is the response text (the reasoning),
    not a side-effect — so this returns False for them. Chat-only modes
    (``default``) with empty allowlists also return False.
    """
    return bool(_fork_mode_action_tool_ids(mode_id))


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
        # A child the harness force-yielded (iteration cap, stuck-in-stage,
        # or budget exhaustion) was *interrupted*, not hallucinating. Its
        # tool-call pattern is irrelevant — it never got to finish, so we
        # cannot conclude anything about whether it would have invoked its
        # action tools. Classify it as ``interrupted`` before the zero-tool /
        # action-tool checks so a coding child that gathered evidence via
        # ``lisa-storage.read_file`` then hit the iteration cap is not
        # reported as "hallucinated success — nothing written" (the
        # 2026-08-02 fan-out regression). The parent treats ``interrupted``
        # like ``no_tools`` for the deterministic abort but with honest
        # "interrupted by harness" messaging.
        #
        # A child the *model* yielded from (it called the ``yield`` tool to
        # ask the user a question) is NOT an interruption — it is an honest
        # punt. Classify it as ``model_yield`` and carry the ``YieldAction``
        # so the parent can propagate the question to the user. The
        # discriminator is ``child_result.yielded.context``: the in-loop
        # ``yield`` tool dispatch (``turn_runner.py``) sets
        # ``context="model-emitted yield"``; the lifecycle resolver's
        # cap/budget/stuck paths set ``context="iteration cap …"`` etc.
        if child_result.yielded is not None:
            _ya = child_result.yielded
            _is_model_yield = _ya.context == "model-emitted yield"
            if _is_model_yield:
                _flog.info(
                    "fork child %s model-emitted yield (asking the user: %s); "
                    "propagating question to parent",
                    target.mode_id,
                    (_ya.question or "")[:120],
                )
                return (
                    ForkChildResult(
                        target=target,
                        status="model_yield",
                        output_text=child_result.response_text,
                        called_tool_names=sorted(child_result.called_tool_names),
                        yield_action=_ya,
                        duration_ms=elapsed_ms,
                    ),
                    "model_yield",
                    elapsed_ms,
                )
            _flog.warning(
                "fork child %s was force-yielded by the harness (iteration "
                "cap / stuck-in-stage / budget); marking interrupted (not "
                "hallucinated) — its tool-call pattern is inconclusive",
                target.mode_id,
            )
            return (
                ForkChildResult(
                    target=target,
                    status="interrupted",
                    output_text=child_result.response_text,
                    error_message=(
                        f"fork child {target.mode_id} was interrupted by the "
                        f"turn lifecycle (force-yielded) before completing its "
                        f"action tools; its reported output is partial. "
                        f"Re-issue the request, or raise "
                        f"FORK_CHILD_MAX_TOOL_ITERATIONS (currently "
                        f"{_kernel_settings.fork_child_max_tool_iterations})."
                    ),
                    called_tool_names=sorted(child_result.called_tool_names),
                    yield_action=_ya,
                    duration_ms=elapsed_ms,
                ),
                "interrupted",
                elapsed_ms,
            )
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
                    called_tool_names=sorted(child_result.called_tool_names),
                    duration_ms=elapsed_ms,
                ),
                "no_tools",
                elapsed_ms,
            )
        # A child that made tool calls but never invoked any of its mode's
        # authored *action* tools (only read-only fetches / searches) also
        # hallucinated its success: it gathered evidence but never wrote,
        # learned, or applied anything. The 2026-07-27 follow-on: the
        # ``markdown`` child did 4 ``fetch_page`` calls then exited empty
        # without ever calling ``markdown_forge_document``, so nothing
        # landed on disk yet the ``tool_call_count == 0`` guard passed it
        # as ``ok``. Catch it here.
        action_tool_ids = _fork_mode_action_tool_ids(target.mode_id)
        # Namespace-aware match: the authored ``action_tool_ids`` are bare
        # front-matter names (``read_file``, ``write_file``, …) while the
        # dispatched ``called_tool_names`` are namespaced MCP ids
        # (``lisa-storage.read_file``) when the tool was merged into the
        # mode's allowlist via the ambient ``default_available_tools``
        # contract. A plain ``set & set`` intersection is empty in that
        # case, false-flagging every fork child that legitimately used its
        # ambient storage tools as ``no_tools`` (the 2026-08-02 fan-out
        # regression). Strip the ``server_id.`` prefix from each called
        # name before intersecting so a namespaced call matches its bare
        # authored action tool. Bare dispatched names (e.g.
        # ``synthesize_and_store_learning``, ``recall_memory``) are
        # unchanged by ``bare_tool_name``.
        called_bare = {bare_tool_name(n) for n in child_result.called_tool_names}
        if action_tool_ids and not (called_bare & action_tool_ids):
            _flog.warning(
                "fork child %s made %d tool call(s) but none were its "
                "mode's authored action tools %s (called: %s); marking "
                "no_tools (hallucinated success — nothing written/learned)",
                target.mode_id,
                child_result.tool_call_count,
                sorted(action_tool_ids),
                sorted(child_result.called_tool_names),
            )
            return (
                ForkChildResult(
                    target=target,
                    status="no_tools",
                    output_text=child_result.response_text,
                    error_message=(
                        f"fork child {target.mode_id} made "
                        f"{child_result.tool_call_count} tool call(s) but "
                        f"none were its mode's authored action tools "
                        f"({', '.join(sorted(action_tool_ids))}); its "
                        "reported success is unverified — nothing was "
                        "written to disk or memory."
                    ),
                    called_tool_names=sorted(child_result.called_tool_names),
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
                called_tool_names=sorted(child_result.called_tool_names),
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
