"""Delegate operation — sequential sub-turn delegation to another mode.

A thin pack-level handler that reads authored content, resolves the target
mode from the installed catalog, builds a :class:`DelegationSpec` from
LLM-supplied parameters, calls the injected ``ctx.host.delegate``, and
returns the :class:`DelegationResult` as the operation's structured
output.

The handler owns no routing or framework state. The kernel wires
``delegate_fn`` into the :class:`OperationHost` (``ctx.host``) at turn
start; the default implementation (``_default_delegate_fn``) runs
``run_turn`` recursively as a ``sub_turn_kind="delegate"`` child. A host
may override ``TurnHooks.delegate_fn`` to inject a custom delegation
backend without touching this handler.

Per the pack-content rule, the model-facing instruction prose lives in
authored Markdown (``operations/delegation/delegate.md``) loaded through
``ctx.host.read_authored_text``. This module contains only path
constants, the structured ``Parameters`` model, and substitution glue.
"""

from __future__ import annotations

import logging

from kernel.delegation import DelegationResult, DelegationSpec
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import OperationExecutionContext, OperationResult
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

_DELEGATE_INSTRUCTION_PATH = "operations/delegation/delegate.md"


class Parameters(BaseModel):
    """User-supplied inputs for the delegate operation."""

    target_mode_id: str = Field(
        ...,
        description=(
            "The routable mode_id to delegate to. Must be one of the "
            "Available Modes listed in the operation instruction."
        ),
    )
    query: str = Field(
        ...,
        description=(
            "The scoped input text for the delegated child. This is the "
            "child turn's user query — not the parent's full query."
        ),
    )
    stage_id: str = Field(
        default="",
        description=(
            "Optional stage_id for the target mode. Defaults to the "
            "mode's start_stage_id when empty."
        ),
    )
    max_tool_iterations: int = Field(
        default=0,
        description=(
            "Optional per-child iteration cap override. 0 means use the "
            "kernel default (DELEGATE_CHILD_MAX_TOOL_ITERATIONS)."
        ),
        ge=0,
    )
    model_alias: str = Field(
        default="",
        description=(
            "Optional model alias override for the child turn. Empty "
            "means use the target mode's resolved model."
        ),
    )


def _resolve_parameters(ctx: OperationExecutionContext) -> Parameters:
    """Build a :class:`Parameters` from the operation context.

    Validates the LLM-supplied ``ctx.params`` dict against the schema. A
    ``ValidationError`` surfaces a clear error to the model via the
    returned ``OperationResult`` rather than crashing the turn.
    """
    raw = ctx.params or {}
    # Allow the caller to pass the query via ctx.query when the operation
    # is invoked as an entry operation (no explicit ``query`` param).
    if not raw.get("query") and ctx.query:
        raw = {**raw, "query": ctx.query}
    return Parameters.model_validate(raw)


def _render_result_text(result: DelegationResult) -> str:
    """Render a :class:`DelegationResult` as a human-readable text block.

    The parent mode reads this from ``output_text``. The structured
    payload (including ``status``, ``tool_call_count``, and
    ``called_tool_names``) is in ``output_json``.
    """
    lines: list[str] = [
        f"## Delegation to `{result.target_mode_id}`",
        "",
        f"- **Status**: `{result.status}`",
        f"- **Duration**: {result.duration_ms} ms",
        f"- **Tool calls**: {result.tool_call_count}",
    ]
    if result.called_tool_names:
        lines.append(f"- **Tools called**: {', '.join(result.called_tool_names)}")
    # Effect receipts surface the child's completed side effects — most
    # importantly write receipts with artifact ids (worktree / document),
    # so the parent can resume interrupted work instead of re-delegating
    # from scratch (the 2026-10-21 delegate incident).
    _write_receipts = [
        r for r in result.effect_receipts if str(r.get("effect", "")) in ("write", "external_write")
    ]
    if _write_receipts:
        _artifact_bits = [
            f"`{r.get('capability_id', '')}` → `{r.get('artifact_id', '')}`"
            for r in _write_receipts
            if r.get("artifact_id")
        ]
        if _artifact_bits:
            lines.append(f"- **Completed writes**: {'; '.join(_artifact_bits)}")
    if result.error_message:
        lines.append("")
        lines.append(f"**Error**: {result.error_message}")
    if result.output_text:
        lines.append("")
        lines.append("### Child output")
        lines.append("")
        lines.append(result.output_text)
    return "\n".join(lines)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Delegate a sub-turn to another mode and return the structured result.

    Builds a :class:`DelegationSpec` from the validated parameters, calls
    the injected ``ctx.host.delegate``, and returns the
    :class:`DelegationResult` as both ``output_text`` (rendered) and
    ``output_json`` (structured).

    Fails loudly when ``delegate`` is not wired on the host (the host has
    not enabled delegation) or the target mode is not in the installed
    catalog — per the pack-content rule, no silent fallback.
    """
    try:
        params = _resolve_parameters(ctx)
    except ValidationError as exc:
        return OperationResult(
            output_text=f"delegate parameter validation failed: {exc}",
            output_json={"status": "error", "error_message": str(exc)},
        )

    # Resolve the target mode against the live catalog so the operation
    # fails loudly on an unknown mode_id rather than handing the kernel
    # an invalid target.
    catalog = get_workflow_catalog()
    target_config = catalog.get(params.target_mode_id)
    if target_config is None:
        return OperationResult(
            output_text=(
                f"delegate target mode '{params.target_mode_id}' is not in "
                "the installed catalog; cannot delegate."
            ),
            output_json={
                "target_mode_id": params.target_mode_id,
                "status": "error",
                "error_message": f"unknown target mode: {params.target_mode_id}",
            },
        )

    host = getattr(ctx, "host", None)
    if host is None or getattr(host, "delegate", None) is None:
        return OperationResult(
            output_text=(
                "ctx.host.delegate is not wired; the host has not enabled "
                "delegation. Cannot delegate."
            ),
            output_json={
                "target_mode_id": params.target_mode_id,
                "status": "error",
                "error_message": "host.delegate not wired",
            },
        )

    spec = DelegationSpec(
        target_mode_id=params.target_mode_id,
        query=params.query,
        stage_id=params.stage_id,
        max_tool_iterations=params.max_tool_iterations,
        model_alias=params.model_alias,
        # Thread the operation's own wall-clock ceiling so the child
        # force-completes gracefully before the registry's hard cancel.
        timeout_seconds=int(getattr(ctx, "operation_timeout_seconds", 0) or 0),
    )

    result = await host.delegate(spec)
    if not isinstance(result, DelegationResult):
        return OperationResult(
            output_text=(
                "runtime.delegate returned an unexpected result type "
                f"({type(result).__name__}); expected DelegationResult."
            ),
            output_json={
                "target_mode_id": params.target_mode_id,
                "status": "error",
                "error_message": f"unexpected result type: {type(result).__name__}",
            },
        )

    return OperationResult(
        output_text=_render_result_text(result),
        output_json=result.model_dump(mode="json"),
    )
