"""Fork operation — parallel sub-turn fan-out to multiple modes.

A thin pack-level handler that reads authored content, resolves each target
mode from the installed catalog, builds a :class:`ForkSpec` from LLM-supplied
parameters, calls the injected ``ctx.host.fork``, and returns the
:class:`ForkResult` as the operation's structured output.

The handler owns no routing or framework state. The kernel wires
``fork_fn`` into the :class:`OperationHost` (``ctx.host``) at turn
start; the default implementation (``_default_fork_fn``) wraps
``_execute_fork_fanout`` from :mod:`kernel.turn_phases._fork`, fanning
the targets out via an ``asyncio.TaskGroup`` and classifying each child
via the shared ``classify_child_turn``. A host may override
``TurnHooks.fork_fn`` to inject a custom fan-out backend without
touching this handler.

Per the pack-content rule, the model-facing instruction prose lives in
authored Markdown (``operations/delegation/fork.md``) loaded through
``ctx.host.read_authored_text``. This module contains only path
constants, the structured ``Parameters`` model, and substitution glue.
"""

from __future__ import annotations

import logging

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import (
    ForkResult,
    ForkSpec,
    ForkTarget,
    OperationExecutionContext,
    OperationResult,
)
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

_FORK_INSTRUCTION_PATH = "operations/delegation/fork.md"


class TargetParam(BaseModel):
    """One fork target as supplied by the LLM."""

    mode_id: str = Field(
        ...,
        description=(
            "The routable mode_id to fan out to. Must be one of the "
            "Available Modes listed in the operation instruction."
        ),
    )
    stage_id: str = Field(
        default="",
        description=(
            "Optional stage_id for the target mode. Defaults to the "
            "mode's start_stage_id when empty."
        ),
    )
    query: str = Field(
        default="",
        description=(
            "The scoped input text for this child. This is the child "
            "turn's user query — not the parent's full query. When "
            "empty, the parent turn's query is used."
        ),
    )


class Parameters(BaseModel):
    """User-supplied inputs for the fork operation."""

    targets: list[TargetParam] = Field(
        ...,
        description=(
            "The routable modes to fan out to in parallel. At least one "
            "target is required; each ``mode_id`` must be a routable "
            "mode from the installed catalog."
        ),
    )
    max_tool_iterations: int = Field(
        default=0,
        description=(
            "Optional per-child iteration cap override applied to every "
            "child. 0 means use the kernel default "
            "(FORK_CHILD_MAX_TOOL_ITERATIONS)."
        ),
        ge=0,
    )
    model_alias: str = Field(
        default="",
        description=(
            "Optional model alias override applied to every child turn. "
            "Empty means use each target mode's resolved model."
        ),
    )


def _resolve_parameters(ctx: OperationExecutionContext) -> Parameters:
    """Build a :class:`Parameters` from the operation context.

    Validates the LLM-supplied ``ctx.params`` dict against the schema. A
    ``ValidationError`` surfaces a clear error to the model via the
    returned ``OperationResult`` rather than crashing the turn.
    """
    raw = ctx.params or {}
    # Allow the caller to pass targets implicitly when the operation is
    # invoked as an entry operation (no explicit ``targets`` param). In
    # that shape the single target is the current mode with ctx.query.
    if not raw.get("targets") and ctx.query:
        raw = {
            **raw,
            "targets": [{"mode_id": ctx.mode_id, "query": ctx.query}],
        }
    return Parameters.model_validate(raw)


def _render_result_text(result: ForkResult) -> str:
    """Render a :class:`ForkResult` as a human-readable text block.

    The parent mode reads this from ``output_text``. The structured
    payload (including ``succeeded`` and per-child ``status`` /
    ``called_tool_names``) is in ``output_json``.
    """
    lines: list[str] = [
        f"## Fork fan-out: {result.succeeded}/{len(result.children)} children succeeded",
        "",
    ]
    for child in result.children:
        lines.append(f"### {child.target.mode_id}")
        lines.append("")
        lines.append(f"- **Status**: `{child.status}`")
        lines.append(f"- **Duration**: {child.duration_ms} ms")
        if child.called_tool_names:
            lines.append(f"- **Tools called**: {', '.join(child.called_tool_names)}")
        # Effect receipts surface the child's completed side effects —
        # write receipts carry the artifact id the parent can resume when
        # a child ended interrupted (the 2026-10-21 delegate incident).
        _write_receipts = [
            r
            for r in child.effect_receipts
            if str(r.get("effect", "")) in ("write", "external_write")
        ]
        if _write_receipts:
            _artifact_bits = [
                f"`{r.get('capability_id', '')}` → `{r.get('artifact_id', '')}`"
                for r in _write_receipts
                if r.get("artifact_id")
            ]
            if _artifact_bits:
                lines.append(f"- **Completed writes**: {'; '.join(_artifact_bits)}")
        if child.error_message:
            lines.append(f"- **Error**: {child.error_message}")
        if child.output_text:
            lines.append("")
            lines.append(child.output_text)
        lines.append("")
    return "\n".join(lines).rstrip()


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Fan out a parallel sub-turn to multiple modes and return the result.

    Builds a :class:`ForkSpec` from the validated parameters, calls the
    injected ``ctx.host.fork``, and returns the :class:`ForkResult` as
    both ``output_text`` (rendered) and ``output_json`` (structured).

    Fails loudly when ``fork`` is not wired on the host (the host has
    not enabled fan-out) or any target mode is not in the installed
    catalog — per the pack-content rule, no silent fallback.
    """
    try:
        params = _resolve_parameters(ctx)
    except ValidationError as exc:
        return OperationResult(
            output_text=f"fork parameter validation failed: {exc}",
            output_json={"status": "error", "error_message": str(exc)},
        )

    if not params.targets:
        return OperationResult(
            output_text="fork requires at least one target.",
            output_json={"status": "error", "error_message": "no targets provided"},
        )

    # Resolve each target mode against the live catalog so the operation
    # fails loudly on an unknown mode_id rather than handing the kernel
    # an invalid target.
    catalog = get_workflow_catalog()
    fork_targets: list[ForkTarget] = []
    for tp in params.targets:
        target_config = catalog.get(tp.mode_id)
        if target_config is None:
            return OperationResult(
                output_text=(
                    f"fork target mode '{tp.mode_id}' is not in the installed catalog; cannot fork."
                ),
                output_json={
                    "target_mode_id": tp.mode_id,
                    "status": "error",
                    "error_message": f"unknown target mode: {tp.mode_id}",
                },
            )
        resolved_stage = tp.stage_id or target_config.start_stage_id
        fork_targets.append(
            ForkTarget(
                mode_id=tp.mode_id,
                stage_id=resolved_stage,
                query=tp.query or ctx.query,
            )
        )

    host = getattr(ctx, "host", None)
    if host is None or getattr(host, "fork", None) is None:
        return OperationResult(
            output_text=(
                "ctx.host.fork is not wired; the host has not enabled fork fan-out. Cannot fork."
            ),
            output_json={
                "status": "error",
                "error_message": "host.fork not wired",
            },
        )

    spec = ForkSpec(
        targets=fork_targets,
        max_tool_iterations=params.max_tool_iterations,
        model_alias=params.model_alias,
        # Thread the operation's own wall-clock ceiling so the children
        # force-complete gracefully before the registry's hard cancel.
        timeout_seconds=int(getattr(ctx, "operation_timeout_seconds", 0) or 0),
    )

    result = await host.fork(spec)
    if not isinstance(result, ForkResult):
        return OperationResult(
            output_text=(
                "runtime.fork returned an unexpected result type "
                f"({type(result).__name__}); expected ForkResult."
            ),
            output_json={
                "status": "error",
                "error_message": f"unexpected result type: {type(result).__name__}",
            },
        )

    return OperationResult(
        output_text=_render_result_text(result),
        output_json=result.model_dump(mode="json"),
    )
