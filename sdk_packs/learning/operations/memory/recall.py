"""Memory recall operation for workflow modes that need fuzzy historical lookup.

Called by coach and other modes when the model wants to search prior notes,
decisions, or vague historical threads without relying on exact keyword
filters.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from kernel.models import MemoryRecord, OperationExecutionContext, OperationResult, ToolExecutionContext
from sdk.config import settings
from sdk_packs.learning.utils.memory_recall import build_memory_recall_answer


class Parameters(BaseModel):
    """User-supplied inputs for the memory recall operation."""

    query: str = Field(..., description="The memory, note, or prior thread to recall.")
    deep: bool = Field(
        default=False,
        description="Force a deep recall synthesis pass even when confidence is high.",
    )
    limit: int = Field(
        default_factory=lambda: settings.memory_recall_final_limit,
        description="Maximum number of ranked evidence memories to include.",
        ge=1,
        le=20,
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the fused memory recall flow and return text plus structured evidence.

    An empty query returns a no-op result so action modes can expose the
    operation directly without wrapper logic.
    """
    raw_query = str(ctx.params.get("query") or ctx.query or "").strip()
    if not raw_query:
        return OperationResult()

    if not callable(ctx.execute_tool_fn):
        return OperationResult(output_text="Memory recall unavailable: tool execution not injected.")

    default_limit = max(1, ctx.memory_recall_default_limit or settings.memory_recall_final_limit)
    limit = int(ctx.params.get("limit", default_limit) or 1)

    tool_result = await ctx.execute_tool_fn(
        "memory.list",
        ToolExecutionContext(
            tool_id="memory.list",
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            query=raw_query,
            parameters={
                "user_slug": ctx.user_slug,
                "limit": max(1, min(limit, 20)),
            },
        ),
    )
    if getattr(tool_result, "error", None):
        return OperationResult(output_text=f"Memory recall unavailable: {tool_result.error}")

    records: list[MemoryRecord] = []
    try:
        data = json.loads(tool_result.output_text or "[]")
    except (json.JSONDecodeError, TypeError):
        data = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                records.append(MemoryRecord.model_validate(item))
            except (ValidationError, TypeError):
                continue

    recall = build_memory_recall_answer(
        query=raw_query,
        records=records,
        limit=limit,
    )

    if not recall.found:
        return OperationResult(
            output_text=recall.answer,
            output_json=recall.model_dump(mode="json"),
            extra={"confidence": recall.confidence, "auto_deep_used": recall.auto_deep_used},
        )

    lines = [recall.answer.strip()]
    if recall.uncertainty_note:
        lines.append(f"Uncertainty: {recall.uncertainty_note}")
    if recall.evidence:
        lines.append("")
        lines.append("Relevant memories:")
        for evidence in recall.evidence[:3]:
            lines.append(f"- {evidence.created_at[:10]} — {evidence.content}")

    return OperationResult(
        output_text="\n".join(lines),
        output_json=recall.model_dump(mode="json"),
        extra={"confidence": recall.confidence, "auto_deep_used": recall.auto_deep_used},
    )

