"""Memory recall operation for workflow modes that need fuzzy historical lookup.

Called by coach and other modes when the model wants to search prior notes,
decisions, or vague historical threads without relying on exact keyword
filters.
"""

from __future__ import annotations

import json
import logging

from kernel.models import (
    MemoryRecallAnswer,
    MemoryRecord,
    OperationExecutionContext,
    OperationResult,
    ToolExecutionContext,
    ToolResult,
)
from pydantic import BaseModel, Field, ValidationError
from sdk.config import settings
from sdk.packs.memory.utils.memory_recall import build_memory_recall_answer

logger = logging.getLogger(__name__)


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


def _resolve_recall_limit(ctx: OperationExecutionContext) -> int:
    """Resolve the effective evidence limit from params with sane clamping."""
    default_limit = max(1, ctx.memory_recall_default_limit or settings.memory_recall_final_limit)
    return int(ctx.params.get("limit", default_limit) or 1)


def _build_search_context(
    ctx: OperationExecutionContext, raw_query: str, limit: int
) -> ToolExecutionContext:
    """Build the ``memory.search`` ToolExecutionContext mirroring the synthesize path.

    Scope fields (user_slug / workspace_id / surface_id / memory_scope) must
    ride in ``parameters`` — the memory.search synthetic alias forwards only
    the parameters dict to the native search backend (see
    kernel.tools.synthetic.mcp_alias_tool), so setting them on the context
    alone leaves them at backend defaults and the scoped WHERE clause
    returns zero rows.
    """
    return ToolExecutionContext(
        tool_id="memory.search",
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        surface_id=ctx.surface_id,
        memory_scope=ctx.memory_scope,
        mode_id=ctx.mode_id,
        query=raw_query,
        parameters={
            "query": raw_query,
            "user_slug": ctx.user_slug,
            "workspace_id": ctx.workspace_id,
            "surface_id": ctx.surface_id,
            "memory_scope": ctx.memory_scope,
            "limit": max(1, min(limit, 20)),
        },
    )


def _parse_memory_records(tool_result: ToolResult) -> list[MemoryRecord]:
    """Parse the memory.search tool result into validated ``MemoryRecord`` items.

    Tolerates malformed payloads: JSON-decode failures, dict envelopes
    (``{"results": [...]}``), and per-item validation errors all degrade to
    fewer records (or none) rather than raising.
    """
    records: list[MemoryRecord] = []
    try:
        data = json.loads(tool_result.output_text or "[]")
    except (json.JSONDecodeError, TypeError):
        data = []
    # Defensive envelope unwrap: some backends (e.g. Mem0) return a
    # paginated dict envelope {"results": [...]} instead of a bare list.
    # The alias result_map should already normalize this, but this
    # belt-and-suspenders fallback catches any backend that slips through.
    if isinstance(data, dict) and "results" in data:
        results = data.get("results")
        if isinstance(results, list):
            data = results
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                records.append(MemoryRecord.model_validate(item))
            except (ValidationError, TypeError) as exc:
                logger.debug(
                    "recall_memory: dropped record on validation: %s",
                    exc,
                    exc_info=False,
                )
                continue
    return records


def _render_recall_output(recall: MemoryRecallAnswer) -> OperationResult:
    """Build the final ``OperationResult`` from a synthesized recall answer.

    The not-found path returns the bare answer; the found path appends the
    uncertainty note and the top-3 evidence lines with thread-turn pointers.
    """
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
            # Render the thread-turn pointer when present so the model can
            # verify the claim against ground truth via thread_read_range.
            turn_part = (
                f" (learned around turn {evidence.source_turn})"
                if evidence.source_turn is not None
                else ""
            )
            lines.append(f"- {evidence.created_at[:10]}{turn_part} — {evidence.content}")

    return OperationResult(
        output_text="\n".join(lines),
        output_json=recall.model_dump(mode="json"),
        extra={"confidence": recall.confidence, "auto_deep_used": recall.auto_deep_used},
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the fused memory recall flow and return text plus structured evidence.

    An empty query returns a no-op result so action modes can expose the
    operation directly without wrapper logic.
    """
    raw_query = str(ctx.params.get("query") or ctx.query or "").strip()
    if not raw_query:
        return OperationResult()

    if ctx.host is None:
        return OperationResult(
            output_text="Memory recall unavailable: tool execution not injected."
        )

    limit = _resolve_recall_limit(ctx)

    # Use memory.search (semantic/lexical search with the query), not
    # memory.list (recent listing without query). The query must reach the
    # backend's search engine — setting it only on ToolExecutionContext.query
    # is not enough; it must also be in parameters (which is what
    # _mcp_tool_executor forwards to the pool).
    #
    # The same applies to the scope fields: the memory.search synthetic alias
    # forwards ONLY the `parameters` dict to the native search_memory backend
    # (see kernel.tools.synthetic.mcp_alias_tool — its handler closes over
    # `arguments`, never the ToolExecutionContext). Setting surface_id /
    # workspace_id / memory_scope on the ToolExecutionContext alone leaves
    # them at the native tool's defaults (surface_id="", workspace_id="default"),
    # so the store's scoped WHERE clause (user_slug, workspace_id,
    # memory_scope, surface_id) returns zero rows — recall reports "No memory
    # matches found" even when the store is full. Mirror the synthesize path,
    # which passes all three scope fields explicitly in parameters.
    try:
        search_ctx = _build_search_context(ctx, raw_query, limit)
        tool_result = await ctx.host.execute_tool("memory.search", search_ctx)
    except TypeError as exc:
        if "not wired" in str(exc):
            return OperationResult(
                output_text="Memory recall unavailable: tool execution not injected."
            )
        raise
    if getattr(tool_result, "error", None):
        return OperationResult(output_text=f"Memory recall unavailable: {tool_result.error}")

    records = _parse_memory_records(tool_result)

    deep = bool(ctx.params.get("deep", False))

    recall = build_memory_recall_answer(
        query=raw_query,
        records=records,
        limit=limit,
        deep=deep,
    )

    return _render_recall_output(recall)
