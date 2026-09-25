"""Consolidate a cluster of episodic memories into one durable semantic record.

The memory-consolidation slice (``sdk/packs/memory/slice_provider.py``)
assembles a batch of related episodic records and invokes this operation
via ``turn_hooks.execute_operation``. This operation distills the batch
into a single self-contained semantic fact using the same structured-output
loop as ``synthesize_learning``, writes the semantic record via
``memory.add`` with ``memory_kind="semantic"``, a boosted salience, and
``supersedes_id``/``item_ids`` linking it to its sources, then retires
each episodic source via ``memory.retire`` so the default
``status="active"`` recall pool stops surfacing the originals.

Reuses the ``synthesize_learning`` distillation seam (``ctx.host.complete``
+ ``json_only_retry_messages`` + ``parse_structured_payload``) and the
authored-prompt loader (no inline fallbacks per the pack-content rule).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext
from kernel.operations.structured_output import (
    completion_content,
    json_only_retry_messages,
    parse_structured_payload,
)
from pydantic import BaseModel, Field, ValidationError
from sdk.content import require_authored_text

logger = logging.getLogger(__name__)

# Authored-text path — loaded via ctx.host.read_authored_text (SDK loader).
_SYSTEM_PROMPT_PATH = "operations/memory/prompts/consolidate_semantic_system.md"

# Fallback default for the consolidation LLM token budget. The authoritative
# source is the session manifest's
# ``resources.packs.configuration.lisa-pack-memory`` block, surfaced to
# the op at runtime as ``ctx.pack_config``.
_CONSOLIDATE_MAX_TOKENS = 768

#: Max chars of the summary shown in the user-facing output preview.
_PREVIEW_CHARS = 80
#: Max retired ids shown inline in the output before switching to "+N more".
_PREVIEW_IDS = 8
#: Salience floor and boost for the consolidated semantic record.
_SEMANTIC_SALIENCE_FLOOR = 0.8
_SEMANTIC_SALIENCE_BOOST = 0.1


@dataclass(frozen=True)
class _ResolvedSettings:
    """Resolved operation budgets: ``ctx.pack_config`` overrides module fallbacks."""

    consolidate_max_tokens: int


def _cfg_int(pack_config: Any, key: str, default: int) -> int:
    """Read an int from ``ctx.pack_config`` with a module-constant fallback."""
    if pack_config is None:
        return default
    raw = pack_config.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "consolidate_semantic: pack_config[%r]=%r is not an int; using default %d",
            key,
            raw,
            default,
        )
        return default


def _resolve_settings(ctx: OperationExecutionContext) -> _ResolvedSettings:
    """Resolve operation budgets from ``ctx.pack_config`` with fallbacks."""
    return _ResolvedSettings(
        consolidate_max_tokens=_cfg_int(
            ctx.pack_config, "memory_consolidate_max_tokens", _CONSOLIDATE_MAX_TOKENS
        ),
    )


def _load_system_prompt(ctx: OperationExecutionContext) -> str:
    """Load the consolidation system prompt via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """
    return require_authored_text(
        ctx.host.read_authored_text if ctx.host is not None else None,
        _SYSTEM_PROMPT_PATH,
        context="consolidate_semantic operation",
    )


class Parameters(BaseModel):
    """User-/slice-supplied inputs for the consolidation operation."""

    episodic_records: list[dict[str, Any]] = Field(
        ...,
        description="The episodic memory records to consolidate, each with "
        "at least id, content, and created_at.",
        min_length=1,
    )
    memory_scope: str = Field("personal", description="Privacy scope for the semantic record.")
    surface_id: str = Field("", description="Surface isolation key.")
    user_slug: str = Field("default", description="The user identity to scope to.")
    workspace_id: str = Field("default", description="Workspace isolation key.")


class ConsolidatedSemantic(BaseModel):
    """Strict bounded output from the consolidation LLM call."""

    summary: str = Field(
        ...,
        description="One concise, self-contained semantic sentence preserving "
        "the durable knowledge across the cluster.",
        min_length=1,
        max_length=500,
    )
    confidence: float = Field(
        0.8,
        description="Confidence in the consolidated summary (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )
    item_ids: list[int] = Field(
        ...,
        description="The input record ids the summary was derived from.",
        min_length=1,
    )


def _render_batch(records: list[dict[str, Any]]) -> str:
    """Render the episodic batch as a numbered list for the LLM user message."""
    lines: list[str] = []
    for idx, rec in enumerate(records, start=1):
        rid = rec.get("id", "?")
        content = str(rec.get("content", "")).strip()
        created = str(rec.get("created_at", "")).strip()
        lines.append(f"{idx}. [id={rid}, created={created}] {content}")
    return "\n".join(lines)


def _parse_consolidated(content: str) -> ConsolidatedSemantic:
    """Parse structured output with a guard against JSON-fragment fallbacks."""
    normalized = (content or "").strip()
    if not normalized:
        raise ValueError("empty consolidation output")
    try:
        return cast(
            ConsolidatedSemantic, parse_structured_payload(ConsolidatedSemantic, normalized)
        )
    except ValidationError as exc:
        # Mirror synthesize_learning's guard: if the model emitted JSON that
        # failed validation, refuse to fall back to prose parsing — surface
        # the failure instead of persisting garbage.
        if _looks_like_json(normalized):
            raise RuntimeError(
                "consolidate_semantic: model returned JSON-shaped output that "
                f"failed validation; refusing to fall back. Raw: {normalized[:200]!r}"
            ) from exc
        raise


def _looks_like_json(text: str) -> bool:
    """Return True when the response looks like JSON (fenced or bare object)."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("```") or stripped.startswith("{") or stripped.startswith("["):
        return True
    return bool("{" in stripped and "}" in stripped)


# pylint: disable=protected-access — reusing the cast helper pattern from synthesize_learning


async def _distill_batch(
    ctx: OperationExecutionContext, system_prompt: str, batch_text: str, max_tokens: int
) -> ConsolidatedSemantic:
    """Run the 2-attempt LLM distillation loop over the episodic batch."""
    model_id = ctx.preferred_model_id
    model_def = ctx.host.resolve_model(model_id) if ctx.host is not None else None
    if model_def is None and ctx.host is not None:
        model_def = ctx.host.resolve_model("default")
    if model_def is None:
        raise RuntimeError("consolidate_semantic could not resolve a model")
    first_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": batch_text},
    ]
    last_exc: Exception | None = None
    for attempt_idx, attempt_messages in enumerate(
        (first_messages, json_only_retry_messages(first_messages, ConsolidatedSemantic))
    ):
        try:
            raw = await ctx.host.complete(  # type: ignore[union-attr]
                messages=attempt_messages,
                model_def=model_def,
                max_tokens=max_tokens,
                temperature=0.1,
                response_format=ConsolidatedSemantic if attempt_idx == 0 else None,
            )
            return _parse_consolidated(completion_content(raw))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if attempt_idx == 0:
                continue
            break
    raise RuntimeError(f"consolidate_semantic distillation failed: {last_exc}") from last_exc


async def _persist_semantic(
    ctx: OperationExecutionContext,
    parsed: Parameters,
    consolidated: ConsolidatedSemantic,
    item_ids: list[int],
    source_count: int,
) -> int | str | None:
    """Write the consolidated semantic record via ``memory.add`` and return its id."""
    assert ctx.host is not None
    max_source_salience = max(
        (float(rec.get("salience", 0.0)) for rec in parsed.episodic_records), default=0.0
    )
    boosted_salience = max(
        _SEMANTIC_SALIENCE_FLOOR, min(1.0, max_source_salience + _SEMANTIC_SALIENCE_BOOST)
    )
    supersedes_id = item_ids[0] if item_ids else None
    add_result = await ctx.host.execute_tool(
        "memory.add",
        ToolExecutionContext(
            tool_id="memory.add",
            workspace_id=parsed.workspace_id,
            user_id=ctx.user_id,
            user_slug=parsed.user_slug,
            surface_id=parsed.surface_id,
            memory_scope=parsed.memory_scope,  # type: ignore[arg-type]
            mode_id=ctx.mode_id,
            parameters={
                "content": consolidated.summary,
                "user_slug": parsed.user_slug,
                "workspace_id": parsed.workspace_id,
                "surface_id": parsed.surface_id,
                "memory_scope": parsed.memory_scope,
                "memory_tier": "long_term",
                "memory_kind": "semantic",
                "operation_id": "consolidate_semantic",
                "salience": boosted_salience,
                "supersedes_id": supersedes_id,
                "item_ids": item_ids,
                "confidence": consolidated.confidence,
                "content_json": {
                    "consolidated_from": item_ids,
                    "source_count": source_count,
                },
            },
        ),
    )
    if getattr(add_result, "error", None):
        raise RuntimeError(f"semantic record storage failed: {add_result.error}")
    add_text = getattr(add_result, "output_text", "") or ""
    if "ID " in add_text:
        try:
            return int(add_text.split("ID ", 1)[1].rstrip("."))
        except (ValueError, IndexError):
            return None
    return None


async def _retire_sources(
    ctx: OperationExecutionContext, parsed: Parameters, item_ids: list[int]
) -> list[int]:
    """Retire each episodic source via ``memory.retire``; return the ids retired."""
    assert ctx.host is not None
    retired_ids: list[int] = []
    for eid in item_ids:
        retire_result = await ctx.host.execute_tool(
            "memory.retire",
            ToolExecutionContext(
                tool_id="memory.retire",
                workspace_id=parsed.workspace_id,
                user_id=ctx.user_id,
                user_slug=parsed.user_slug,
                surface_id=parsed.surface_id,
                memory_scope=parsed.memory_scope,
                mode_id=ctx.mode_id,
                parameters={
                    "memory_id": eid,
                    "user_slug": parsed.user_slug,
                    "workspace_id": parsed.workspace_id,
                    "surface_id": parsed.surface_id,
                    "memory_scope": parsed.memory_scope,
                },
            ),
        )
        if not getattr(retire_result, "error", None):
            retired_ids.append(eid)
    return retired_ids


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Distill an episodic batch into one semantic record and retire the sources."""
    params = ctx.params or {}
    try:
        parsed = Parameters.model_validate(params)
    except ValidationError as exc:
        return OperationResult(output_text=f"⛔ Invalid consolidation parameters: {exc}")

    records = parsed.episodic_records
    if not records:
        return OperationResult(output_text="⛔ No episodic records provided to consolidate.")

    if ctx.host is None:
        raise RuntimeError("consolidate_semantic requires the OperationHost seam")

    # Step 1: Distill the batch with the LLM (delegates to _distill_batch).
    system_prompt = _load_system_prompt(ctx)
    batch_text = _render_batch(records)
    consolidate_max_tokens = _resolve_settings(ctx).consolidate_max_tokens
    try:
        consolidated = await _distill_batch(ctx, system_prompt, batch_text, consolidate_max_tokens)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"consolidate_semantic distillation failed: {exc}") from exc

    # Validate item_ids are a subset of the input ids (guard against
    # the model hallucinating ids that were not in the batch).
    input_ids = {int(rec.get("id")) for rec in records if rec.get("id") is not None}
    item_ids = [eid for eid in consolidated.item_ids if eid in input_ids]
    if not item_ids:
        # Fall back to the full input set if the model returned no valid ids.
        item_ids = sorted(input_ids)

    # Step 2: Persist the semantic record via memory.add with boosted salience.
    try:
        semantic_id = await _persist_semantic(ctx, parsed, consolidated, item_ids, len(records))
    except RuntimeError as exc:
        return OperationResult(output_text=f"⛔ {exc}")

    # Step 3: Retire the episodic sources via memory.retire so the default
    # status='active' recall pool stops surfacing them. The originals stay
    # queryable with status=None (audit trail preserved).
    retired_ids = await _retire_sources(ctx, parsed, item_ids)

    preview = consolidated.summary[:_PREVIEW_CHARS]
    return OperationResult(
        output_text=(
            f"✅ Consolidated {len(records)} episodic records into one semantic "
            f"record (id={semantic_id}); retired {len(retired_ids)} sources: "
            f"{_retired_ids_preview(retired_ids)}"
            f"'{preview}{'...' if len(consolidated.summary) > _PREVIEW_CHARS else ''}'"
        ),
        extra={
            "consolidated": True,
            "semantic_id": semantic_id,
            "retired_ids": retired_ids,
            "item_ids": item_ids,
            "summary": consolidated.summary,
            "confidence": consolidated.confidence,
        },
    )


def _retired_ids_preview(ids: list[int]) -> str:
    """Format the retired-ids list for the user-facing output line."""
    if not ids:
        return ""
    shown = ", ".join(str(i) for i in ids[:_PREVIEW_IDS])
    more = f", +{len(ids) - _PREVIEW_IDS} more" if len(ids) > _PREVIEW_IDS else ""
    return f"[{shown}{more}]. "
