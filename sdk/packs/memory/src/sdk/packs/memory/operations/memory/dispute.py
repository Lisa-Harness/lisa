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
from typing import Any

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
        description=(
            "Numeric ID of the memory record to dispute, exactly as shown "
            "in memory.search or memory-stored results (e.g. `227`). Never "
            "invent an id; if you only know the content, use search_query. "
            "Provide this OR search_query."
        ),
    )
    search_query: str = Field(
        default="",
        description=(
            "Find the record to dispute by searching memory content. Provide this OR memory_id."
        ),
    )
    correction: str = Field(
        ...,
        description="The corrected content that supersedes the disputed record.",
    )
    reason: str = Field(
        default="user_correction",
        description=(
            "Why the record is being disputed (e.g. 'user_correction', 'outdated', 'inaccurate')."
        ),
    )


def _scoped_parameters(
    ctx: OperationExecutionContext,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build alias-bound parameters carrying the full identity scope.

    The ``memory.get`` / ``memory.search`` synthetic aliases forward only the
    ``parameters`` dict to the native backend (see
    ``kernel.tools.synthetic.mcp_alias_tool``), and the native tools scope
    their WHERE clause on all four identity fields: ``user_slug``,
    ``workspace_id``, ``surface_id``, ``memory_scope``. Mirrors
    ``recall_memory``'s ``_build_search_context`` so dispute reads exactly
    the slice recall surfaces — a scope mismatch here made records recall
    surfaced undisputable (three failed dispute lookups on the 2026-10-05
    thread).
    """
    return {
        "user_slug": ctx.user_slug,
        "workspace_id": ctx.workspace_id,
        "surface_id": ctx.surface_id,
        "memory_scope": ctx.memory_scope,
        **extra,
    }


async def _find_disputed_record_by_id(
    ctx: OperationExecutionContext,
    memory_id: str,
) -> MemoryRecord | None:
    """Fetch the target record directly by id (surface-scoped server-side).

    A wrong-scope id returns ``None`` ("not found") rather than a
    cross-surface record. A NON-NUMERIC id also returns ``None`` — models
    sometimes guess a slug-style id (``mem_<date>_<slug>``) that no
    backend ever issued (first-party lisa_memory ids are integers;
    external backends use UUID strings). Treating an unparseable id as
    "not found" routes the caller to the existing friendly
    not-found/search guidance instead of an unclassified ValueError
    escape (the 2026-10-18 production incident).
    """
    try:
        native_id = int(memory_id)
    except (TypeError, ValueError):
        logger.warning(
            "dispute_memory: non-numeric memory_id %r is not a native id; "
            "treat as not found (caller should retry with search_query)",
            memory_id,
        )
        return None
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
            parameters=_scoped_parameters(ctx, {"memory_id": native_id}),
        ),
    )
    if getattr(get_result, "error", None):
        raise DisputeLookupError(f"could not fetch memory: {get_result.error}")
    raw = (get_result.output_text or "").strip()
    if raw and raw != "No memory found.":
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict):
            with contextlib.suppress(ValidationError, TypeError):
                return MemoryRecord.model_validate(data)
    return None


async def _find_disputed_record_by_search(
    ctx: OperationExecutionContext,
    search_query: str,
) -> MemoryRecord | None:
    """Find the target record by content search (first hit wins)."""
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
            parameters=_scoped_parameters(ctx, {"query": search_query, "limit": 1}),
        ),
    )
    if getattr(search_result, "error", None):
        raise DisputeLookupError(f"search failed: {search_result.error}")
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
                return MemoryRecord.model_validate(first)
    return None


class DisputeLookupError(RuntimeError):
    """Raised when the memory.get / memory.search lookup call itself fails."""


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

    try:
        if memory_id:
            disputed_record = await _find_disputed_record_by_id(ctx, memory_id)
        elif search_query:
            disputed_record = await _find_disputed_record_by_search(ctx, search_query)
    except DisputeLookupError as exc:
        return OperationResult(output_text=f"Memory dispute: {exc}")

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
