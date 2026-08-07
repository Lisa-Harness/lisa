"""Checkpoint restore operation.

Loads a saved ``CheckpointBundle`` from ``lisa-storage`` by ID and returns
the full bundle contents — including turn history, enrichment fields, and
state — for the host to restore.

The operation itself does NOT restore state into the running session.
Restoration is a host concern: the host reads the ``OperationResult`` and
feeds the ``unified_state`` / ``workflow_position`` dicts back into
its runtime state and the turn runner's hooks. The operation is a pure
load-and-return; the host is the restore executor.

This separation keeps the operation reusable across hosts (each host
restores differently) while the storage and enrichment logic stays in
the pack.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]


class Parameters(BaseModel):
    """User-supplied inputs for the restore operation."""

    checkpoint_id: str = Field(
        ...,
        description="The checkpoint ID to restore.",
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Load a checkpoint from lisa-storage and return its full contents."""
    params = ctx.params or {}
    checkpoint_id = str(params.get("checkpoint_id") or "").strip()
    if not checkpoint_id:
        return OperationResult(output_text="Checkpoint restore failed: no checkpoint_id provided.")

    if ctx.host is None:
        return OperationResult(
            output_text="Checkpoint restore failed: tool execution not injected.",
        )

    tool_result = await ctx.host.execute_tool(
        "restore_checkpoint",
        ToolExecutionContext(
            tool_id="restore_checkpoint",
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"checkpoint_id": checkpoint_id},
        ),
    )

    output_text = getattr(tool_result, "output_text", "") or str(tool_result)
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        return OperationResult(output_text=f"Checkpoint restore failed: {error_text}")

    # Parse the bundle JSON returned by lisa-storage.
    try:
        bundle_data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError) as exc:
        return OperationResult(output_text=f"Checkpoint restore failed: invalid bundle JSON: {exc}")

    if isinstance(bundle_data, dict) and "error" in bundle_data:
        return OperationResult(output_text=f"Checkpoint restore failed: {bundle_data['error']}")

    # Build a human-readable summary for the user.
    label = str(bundle_data.get("label") or "")
    stage = str(bundle_data.get("stage_id") or "")
    turn_count = int(bundle_data.get("turn_count") or 0)
    mode = str(bundle_data.get("mode_id") or "")
    overview = str(bundle_data.get("overview") or "")
    work_done = str(bundle_data.get("work_done") or "")
    important_files = bundle_data.get("important_files") or []
    next_steps = bundle_data.get("next_steps") or []

    summary_lines = [f"Restored checkpoint {checkpoint_id}"]
    if label:
        summary_lines.append(f"Label: {label}")
    if mode:
        summary_lines.append(f"Mode: {mode}")
    if stage:
        summary_lines.append(f"Stage: {stage}")
    summary_lines.append(f"Turns: {turn_count}")
    if overview:
        summary_lines.append(f"\nOverview: {overview}")
    if work_done:
        summary_lines.append(f"Work done: {work_done}")
    if important_files:
        summary_lines.append(f"Important files: {', '.join(important_files)}")
    if next_steps:
        summary_lines.append("Next steps:")
        for step in next_steps:
            summary_lines.append(f"  - {step}")

    return OperationResult(
        output_text="\n".join(summary_lines),
        output_json=bundle_data,
        extra={
            "checkpoint_id": checkpoint_id,
            "label": label,
            "turn_count": turn_count,
            "has_overview": bool(overview),
            "has_next_steps": bool(next_steps),
        },
    )
