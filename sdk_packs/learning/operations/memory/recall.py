"""Memory recall operation for workflow modes that need fuzzy historical lookup.

Called by coach and other modes when the model wants to search prior notes,
decisions, or vague historical threads without relying on exact keyword
filters.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult
from sdk.config import settings


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

    if not callable(ctx.answer_memory_recall_fn):
        return OperationResult(output_text="Memory recall unavailable: recall hook not injected.")

    default_limit = max(1, ctx.memory_recall_default_limit or settings.memory_recall_final_limit)

    recall = await ctx.answer_memory_recall_fn(
        query=raw_query,
        user_slug=ctx.user_slug,
        workspace_id=ctx.workspace_id,
        memory_scope=ctx.memory_scope,
        surface_id=ctx.surface_id,
        complete_fn=ctx.complete_fn,
        force_deep=bool(ctx.params.get("deep", False)),
        limit=int(ctx.params.get("limit", default_limit) or 1),
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

