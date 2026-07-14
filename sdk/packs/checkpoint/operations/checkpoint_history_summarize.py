"""Checkpoint history summarization operation.

Condenses older turns into one structured summary record while keeping the
most recent turns verbatim. This is designed for checkpoint flows that want
compact restorable history without replaying full transcripts.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from kernel.models import OperationExecutionContext, OperationResult
from kernel.operations.structured_output import parse_structured_payload

logger = logging.getLogger(__name__)

__all__ = ["HistorySummary", "Parameters", "execute"]


class Parameters(BaseModel):
    """User-supplied inputs for history summarization."""

    turn_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation turns to summarize.",
    )
    retention_count: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Number of most-recent turns to keep verbatim.",
    )
    prompt: str = Field(
        default="",
        description="Optional natural-language prompt override for history summarization.",
    )


class HistorySummary(BaseModel):
    """Structured summary returned by the model for old turns."""

    condensed_narrative: str = Field(
        default="",
        description="Short narrative summary of older turns.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Grounded bullet points worth retaining.",
    )


def _format_turns(turn_history: list[dict[str, Any]]) -> str:
    """Render turns into compact transcript text for summarization."""
    if not turn_history:
        return ""
    lines: list[str] = []
    for i, turn in enumerate(turn_history, start=1):
        query = str(turn.get("query") or turn.get("user_message") or "").strip()
        response = str(turn.get("response_text") or turn.get("assistant_response") or "").strip()
        if query:
            lines.append(f"Turn {i} - User: {query[:300]}")
        if response:
            lines.append(f"Turn {i} - Lisa: {response[:450]}")
    return "\n".join(lines)


def _fallback_summary(old_turns: list[dict[str, Any]]) -> HistorySummary:
    """Create deterministic fallback summary when model summarization is unavailable."""
    first_query = ""
    last_query = ""
    for turn in old_turns:
        query = str(turn.get("query") or turn.get("user_message") or "").strip()
        if query and not first_query:
            first_query = query
        if query:
            last_query = query
    points: list[str] = []
    if first_query:
        points.append(f"Started with: {first_query[:140]}")
    if last_query and last_query != first_query:
        points.append(f"Most recent earlier query: {last_query[:140]}")
    if not points:
        points.append("Older turns condensed without model enrichment.")
    return HistorySummary(
        condensed_narrative=f"Condensed {len(old_turns)} earlier turns.",
        key_points=points,
    )


async def _summarize_old_turns(
    ctx: OperationExecutionContext,
    old_turns: list[dict[str, Any]],
    prompt_override: str = "",
) -> HistorySummary:
    """Call the model to summarize older turns, with graceful fallback."""
    transcript = _format_turns(old_turns)
    if not transcript or not ctx.complete_fn or not callable(ctx.resolve_model_fn):
        return _fallback_summary(old_turns)

    model_id = ctx.preferred_model_id or "deep"
    model_def = ctx.resolve_model_fn(model_id)
    if model_def is None:
        model_def = ctx.resolve_model_fn("default")
    if model_def is None:
        return _fallback_summary(old_turns)

    system_prompt = prompt_override.strip()
    if not system_prompt and callable(ctx.read_authored_text_fn):
        system_prompt = str(
            ctx.read_authored_text_fn("operations/prompts/checkpoint_history_summarize.md")
            or ""
        ).strip()
    if not system_prompt:
        return _fallback_summary(old_turns)

    try:
        raw = await ctx.complete_fn(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            model_def=model_def,
            max_tokens=350,
            temperature=0.1,
            response_format=HistorySummary,
        )
        content = getattr(raw, "content", "") or str(raw)
        return cast(HistorySummary, parse_structured_payload(HistorySummary, content))
    except (ValidationError, Exception) as exc:  # pylint: disable=broad-except
        logger.warning("checkpoint history summarize failed, using fallback: %s", exc)
        return _fallback_summary(old_turns)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Condense old turns into one summary record and keep recent turns verbatim."""
    params = ctx.params or {}
    parsed = Parameters.model_validate(params)

    turn_history = list(parsed.turn_history)
    retention_count = int(parsed.retention_count)
    if len(turn_history) <= retention_count:
        return OperationResult(
            output_text="History already within retention window; no summarization needed.",
            output_json={
                "turn_history": turn_history,
                "old_turn_count": 0,
                "retained_turn_count": len(turn_history),
                "was_summarized": False,
            },
            extra={"was_summarized": False, "turn_count": len(turn_history)},
        )

    old_turns = turn_history[:-retention_count]
    recent_turns = turn_history[-retention_count:]
    summary = await _summarize_old_turns(ctx, old_turns, parsed.prompt)

    summary_turn = {
        "role": "system",
        "source": "checkpoint_history_summarize",
        "summarized_turn_count": len(old_turns),
        "summary": summary.condensed_narrative,
        "summary_bullets": list(summary.key_points),
    }
    new_turn_history = [summary_turn, *recent_turns]

    output_json = {
        "turn_history": new_turn_history,
        "summary_turn": summary_turn,
        "old_turn_count": len(old_turns),
        "retained_turn_count": len(recent_turns),
        "was_summarized": True,
    }

    return OperationResult(
        output_text=(
            f"Condensed {len(old_turns)} turn(s); retained {len(recent_turns)} latest turn(s)."
        ),
        output_json=output_json,
        extra={
            "was_summarized": True,
            "old_turn_count": len(old_turns),
            "retained_turn_count": len(recent_turns),
            "summary_preview": summary.condensed_narrative[:180],
            "summary_bullets": list(summary.key_points),
        },
    )
