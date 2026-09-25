"""Dispute memory operation — correct or supersede an existing memory record.

No-LLM write-through: the model provides the correction text directly and
the operation writes a superseding record with boosted salience. The old
record is preserved (audit trail) but its salience is demoted so the
correction outranks it in future recall.
"""

from __future__ import annotations

import contextlib
import json
import logging

from kernel.models import (
    MemoryRecord,
    OperationExecutionContext,
    OperationResult,
    ToolExecutionContext,
)
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class Parameters(BaseModel):
    """User-supplied inputs for the dispute memory operation."""

    memory_id: str = Field(
        default="",
        description="The ID of the memory record to dispute. Provide this OR search_query.",
    )
    search_query: str = Field(
        default="",
        description="Find the record to dispute by searching memory content. Provide this OR memory_id.",
    )
    correction: str = Field(
        ...,
        description="The corrected content that supersedes the disputed record.",
    )
    reason: str = Field(
        default="user_correction",
        description="Why the record is being disputed (e.g. 'user_correction', 'outdated', 'inaccurate').",
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Correct or supersede an existing memory record.

    Finds the target record by id or search, then writes a superseding
    record with boosted salience so the correction outranks the original
    in future recall. The old record is preserved for audit.
    """
    params = ctx.params or {}
    memory_id = str(params.get("memory_id", "") or "").strip()
    search_query = str(params.get("search_query", "") or "").strip()
    correction = str(params.get("correction", "") or "").strip()
    reason = str(params.get("reason", "user_correction") or "user_correction").strip()

    if not correction:
        return OperationResult(output_text="⛔ No correction provided.")

    if ctx.host is None:
        return OperationResult(
            output_text="Memory dispute unavailable: tool execution not injected."
        )

    # ── Find the target record ──────────────────────────────────────
    disputed_record: MemoryRecord | None = None

    if memory_id:
        # Fetch the target record directly by id — surface-scoped on the
        # server side, so a wrong-scope id returns "not found" rather than
        # a cross-surface record.
        get_result = await ctx.host.execute_tool(
            "memory.get",
            ToolExecutionContext(
                tool_id="memory.get",
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                parameters={
                    "memory_id": int(memory_id),
                    "user_slug": ctx.user_slug,
                },
            ),
        )
        if getattr(get_result, "error", None):
            return OperationResult(
                output_text=f"Memory dispute: could not fetch memory: {get_result.error}"
            )
        raw = (get_result.output_text or "").strip()
        if raw and raw != "No memory found.":
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data = None
            if isinstance(data, dict):
                with contextlib.suppress(ValidationError, TypeError):
                    disputed_record = MemoryRecord.model_validate(data)

    elif search_query:
        # Search by content.
        search_result = await ctx.host.execute_tool(
            "memory.search",
            ToolExecutionContext(
                tool_id="memory.search",
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                query=search_query,
                parameters={
                    "query": search_query,
                    "user_slug": ctx.user_slug,
                    "limit": 1,
                },
            ),
        )
        if getattr(search_result, "error", None):
            return OperationResult(
                output_text=f"Memory dispute: search failed: {search_result.error}"
            )
        try:
            data = json.loads(search_result.output_text or "[]")
        except (json.JSONDecodeError, TypeError):
            data = []
        if isinstance(data, dict) and "results" in data:
            results = data.get("results")
            if isinstance(results, list):
                data = results
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                with contextlib.suppress(ValidationError, TypeError):
                    disputed_record = MemoryRecord.model_validate(first)

    if disputed_record is None:
        return OperationResult(
            output_text=(
                "⛔ Could not find a memory record to dispute. "
                "Provide a valid memory_id or a search_query that matches an existing record."
            )
        )

    # ── Write the superseding record ─────────────────────────────────
    # Boost salience so the correction outranks the original in future
    # recall. The old record stays with its original salience (audit trail).
    boosted_salience = max(0.8, min(1.0, disputed_record.salience + 0.2))

    add_result = await ctx.host.execute_tool(
        "memory.add",
        ToolExecutionContext(
            tool_id="memory.add",
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={
                "content": correction,
                "user_slug": ctx.user_slug,
                "memory_scope": disputed_record.memory_scope,
                "memory_tier": disputed_record.memory_tier or "long_term",
                "operation_id": "dispute_memory",
                "salience": boosted_salience,
                "content_json": {
                    "dispute_reason": reason,
                    "supersedes_id": str(disputed_record.id),
                    "original_content": disputed_record.content[:200],
                },
            },
        ),
    )

    if getattr(add_result, "error", None):
        return OperationResult(
            output_text=f"Memory dispute: failed to write correction: {add_result.error}"
        )

    return OperationResult(
        output_text=(
            f"✅ Memory record {disputed_record.id} disputed and superseded.\n"
            f"Correction stored with salience {boosted_salience:.1f} "
            f"(original was {disputed_record.salience:.1f}).\n"
            f"Reason: {reason}"
        ),
        output_json={
            "superseded_id": str(disputed_record.id),
            "correction": correction,
            "reason": reason,
            "salience": boosted_salience,
        },
    )
