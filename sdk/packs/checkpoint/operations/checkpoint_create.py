"""Checkpoint create operation.

Builds an enriched ``CheckpointBundle`` from the current conversation and
host state, then persists it via the ``lisa-storage`` MCP server's
``save_checkpoint`` tool.

Enrichment: the operation calls the model to produce a structured summary
(overview, work_done, important_files, next_steps) from the conversation
excerpt. The enrichment system prompt is authored content loaded via
``ctx.read_authored_text_fn`` (``operations/prompts/checkpoint_create_enrichment.md``);
per the pack-content rule, a missing/empty loader raises rather than silently
falling back to a bare bundle. A caller may override the prompt via
``params["prompt"]``.

The operation declares ``save_checkpoint`` in its ``tool_ids`` so it can
call ``lisa-storage`` through ``ctx.execute_tool_fn``. The host's tool
executor routes this to the MCP tool pool with server-scoped dispatch
to ``lisa-storage``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext
from kernel.operations.structured_output import parse_structured_payload

logger = logging.getLogger(__name__)

__all__ = ["CheckpointEnrichment", "Parameters", "execute"]


class Parameters(BaseModel):
    """User-supplied inputs for the checkpoint operation."""

    label: str = Field(
        default="",
        description="Optional human-readable label for this checkpoint.",
    )
    turn_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation turns to include in the bundle. Each entry "
        "should have 'query' and 'response_text' keys at minimum.",
    )
    unified_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized UnifiedState to include. When empty, the "
        "operation reads from ctx.state_provider if available.",
    )
    workflow_session_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized WorkflowSessionState to include. When empty, "
        "the operation omits it.",
    )
    prompt: str = Field(
        default="",
        description="Optional natural-language prompt override for checkpoint enrichment.",
    )


class CheckpointEnrichment(BaseModel):
    """Structured output from the checkpoint enrichment LLM call."""

    overview: str = Field(
        default="",
        description="2-3 sentence narrative summary of what has happened in "
        "this session so far.",
    )
    work_done: str = Field(
        default="",
        description="What was accomplished or decided. Focus on outcomes, "
        "not process.",
    )
    important_files: list[str] = Field(
        default_factory=list,
        description="File paths that were read, written, or are otherwise "
        "relevant to the current work.",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="2-5 concrete next steps the user or agent should take "
        "after restoring from this checkpoint.",
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _checkpoint_id() -> str:
    return uuid.uuid4().hex[:16]


def _format_conversation_for_model(turn_history: list[dict[str, Any]]) -> str:
    """Render turn history as a readable transcript for the enrichment model."""
    if not turn_history:
        return ""
    lines: list[str] = []
    for i, turn in enumerate(turn_history, start=1):
        query = str(turn.get("query") or turn.get("user_message") or "").strip()
        response = str(turn.get("response_text") or turn.get("assistant_response") or "").strip()
        if query:
            lines.append(f"Turn {i} — User: {query}")
        if response:
            # Truncate long responses to keep the enrichment prompt bounded.
            preview = response[:500] + ("..." if len(response) > 500 else "")
            lines.append(f"Turn {i} — Lisa: {preview}")
    return "\n".join(lines)


async def _enrich_checkpoint(
    ctx: OperationExecutionContext,
    turn_history: list[dict[str, Any]],
    prompt_override: str = "",
) -> CheckpointEnrichment:
    """Call the model to produce structured enrichment for the checkpoint.

    Requires a wired completion function, a resolvable model, and an authored
    enrichment prompt (loaded via ``ctx.read_authored_text_fn`` or overridden
    via ``prompt_override``). Raises ``RuntimeError`` when any required input is
    missing — no silent fallback to an unenriched bundle.
    """
    transcript = _format_conversation_for_model(turn_history)
    if not transcript:
        raise RuntimeError("checkpoint enrichment requires non-empty turn history")
    if ctx.complete_fn is None or not callable(ctx.resolve_model_fn):
        raise RuntimeError("checkpoint enrichment requires complete_fn and resolve_model_fn")

    model_id = ctx.preferred_model_id or "deep"
    model_def = ctx.resolve_model_fn(model_id)
    if model_def is None:
        model_def = ctx.resolve_model_fn("default")
    if model_def is None:
        raise RuntimeError("checkpoint enrichment could not resolve a model")

    system_prompt = prompt_override.strip()
    if not system_prompt:
        if not callable(ctx.read_authored_text_fn):
            raise RuntimeError(
                "read_authored_text_fn is not wired; cannot load checkpoint enrichment prompt"
            )
        try:
            loaded = str(
                ctx.read_authored_text_fn("operations/prompts/checkpoint_create_enrichment.md")
                or ""
            ).strip()
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"read_authored_text_fn failed for checkpoint enrichment prompt: {exc}"
            ) from exc
        if not loaded:
            raise RuntimeError(
                "Authored checkpoint enrichment prompt is empty or missing "
                "(operations/prompts/checkpoint_create_enrichment.md)"
            )
        system_prompt = loaded

    try:
        raw = await ctx.complete_fn(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            model_def=model_def,
            max_tokens=500,
            temperature=0.1,
            response_format=CheckpointEnrichment,
        )
        content = getattr(raw, "content", "") or str(raw)
        return cast(CheckpointEnrichment, parse_structured_payload(CheckpointEnrichment, content))
    except ValidationError as exc:
        raise RuntimeError(f"checkpoint enrichment structured output parse failed: {exc}") from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"checkpoint enrichment model call failed: {exc}") from exc


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Build an enriched checkpoint and persist it to lisa-storage."""
    params = ctx.params or {}
    parsed = Parameters.model_validate(params)

    label = parsed.label or str(params.get("label") or "").strip()
    turn_history = list(parsed.turn_history or params.get("turn_history") or [])

    # Read UnifiedState from params or fall back to state_provider.
    unified_state: dict[str, Any] = {}
    if parsed.unified_state:
        unified_state = dict(parsed.unified_state)
    elif ctx.state_provider is not None:
        state = ctx.state_provider()
        if state is not None:
            try:
                unified_state = state.model_dump(mode="json")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("checkpoint: state_provider dump failed: %s", exc)

    workflow_session_state = dict(parsed.workflow_session_state or {})

    # Enrichment: call the model to summarize the conversation.
    enrichment = await _enrich_checkpoint(ctx, turn_history, parsed.prompt)

    bundle = {
        "checkpoint_id": _checkpoint_id(),
        "created_at": _now_iso(),
        "label": label,
        "mode_id": ctx.mode_id,
        "stage_id": str(ctx.stage_id or ""),
        "turn_count": len(turn_history),
        "unified_state": unified_state,
        "workflow_session_state": workflow_session_state,
        "turn_history": turn_history,
        "overview": enrichment.overview,
        "work_done": enrichment.work_done,
        "important_files": enrichment.important_files,
        "next_steps": enrichment.next_steps,
    }

    # Persist via lisa-storage's save_checkpoint tool.
    if not callable(ctx.execute_tool_fn):
        return OperationResult(
            output_text="Checkpoint failed: tool execution not injected.",
        )

    tool_result = await ctx.execute_tool_fn(
        "save_checkpoint",
        ToolExecutionContext(
            tool_id="save_checkpoint",
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"bundle": bundle},
        ),
    )

    output_text = getattr(tool_result, "output_text", "") or str(tool_result)
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        return OperationResult(
            output_text=f"Checkpoint failed: {error_text}",
        )

    # Parse the save_checkpoint response to extract the checkpoint_id.
    import json

    try:
        data = json.loads(output_text)
        cid = data.get("checkpoint_id", "")
    except (json.JSONDecodeError, TypeError):
        cid = bundle["checkpoint_id"]

    summary_parts = [f"Checkpoint saved: {cid}"]
    if label:
        summary_parts.append(f"label: {label}")
    if enrichment.overview:
        summary_parts.append(f"overview: {enrichment.overview[:120]}")
    if enrichment.next_steps:
        summary_parts.append(f"next_steps: {len(enrichment.next_steps)} item(s)")

    return OperationResult(
        output_text=" | ".join(summary_parts),
        output_json={"checkpoint_id": cid, "label": label},
        extra={
            "enriched": bool(enrichment.overview),
            "turn_count": len(turn_history),
            "important_files": enrichment.important_files,
            "next_steps": enrichment.next_steps,
        },
    )