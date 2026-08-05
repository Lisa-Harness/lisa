"""Utilization-aware context compaction operation.

Thin wrapper around ``checkpoint_history_summarize`` that surfaces the
utilization pressure framing. The compaction pack's slice provider
defaults to reusing ``checkpoint_history_summarize`` directly; this
operation is only invoked when a session plan overrides
``CompactionTuning.summarize_operation`` to ``context_compact`` to get
the more aggressive, pressure-aware prompt.

The implementation deliberately reuses the existing structured-output
contract (``HistorySummary``) and the same model-call path as
``checkpoint_history_summarize``. One compaction primitive, two prompts,
two triggers — no forking of the compaction logic itself. The
``HistorySummary`` model is imported from the checkpoint pack rather than
redeclared here, so the two operations can never drift apart.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from kernel.models import OperationExecutionContext, OperationResult
from kernel.operations.structured_output import completion_content, parse_structured_payload
from sdk.packs.checkpoint.operations.checkpoint_history_summarize import HistorySummary

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]


class Parameters(BaseModel):
    """User-supplied inputs for context compaction."""

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
        description="Optional natural-language prompt override.",
    )
    utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current context-window utilization fraction (0.0-1.0).",
    )


def _format_turns(turn_history: list[dict[str, Any]]) -> str:
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


def _load_prompt(ctx: OperationExecutionContext) -> str:
    if ctx.host is None:
        raise RuntimeError("OperationHost seam is not wired; cannot load context_compact prompt")
    try:
        loaded = str(
            ctx.host.read_authored_text("operations/prompts/context_compact.md") or ""
        ).strip()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"read_authored_text failed for context_compact prompt: {exc}") from exc
    if not loaded:
        raise RuntimeError(
            "Authored context_compact prompt is empty or missing "
            "(operations/prompts/context_compact.md)"
        )
    return loaded


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Condense older turns with utilization-aware framing; keep recent turns verbatim."""
    params = ctx.params or {}
    parsed = Parameters.model_validate(params)

    turn_history = list(parsed.turn_history)
    retention_count = int(parsed.retention_count)
    if len(turn_history) <= retention_count:
        return OperationResult(
            output_text="History already within retention window; no compaction needed.",
            output_json={
                "turn_history": turn_history,
                "old_turn_count": 0,
                "retained_turn_count": len(turn_history),
                "was_summarized": False,
                "utilization": parsed.utilization,
            },
            extra={"was_summarized": False, "turn_count": len(turn_history)},
        )

    old_turns = turn_history[:-retention_count]
    recent_turns = turn_history[-retention_count:]

    transcript = _format_turns(old_turns)
    if not transcript:
        raise RuntimeError("context_compact requires non-empty old turns")
    if ctx.host is None:
        raise RuntimeError("context_compact requires the OperationHost seam")

    model_id = ctx.preferred_model_id or "default"
    model_def = ctx.host.resolve_model(model_id)
    if model_def is None:
        model_def = ctx.host.resolve_model("default")
    if model_def is None:
        raise RuntimeError("context_compact could not resolve a model")

    system_prompt = parsed.prompt.strip() or _load_prompt(ctx)
    pressure_line = (
        f"\n\nCurrent context-window utilization: {parsed.utilization:.1%}."
        if parsed.utilization > 0.0
        else ""
    )

    try:
        raw = await ctx.host.complete(
            messages=[
                {"role": "system", "content": system_prompt + pressure_line},
                {"role": "user", "content": transcript},
            ],
            model_def=model_def,
            max_tokens=350,
            temperature=0.1,
            response_format=HistorySummary,
        )
        content = completion_content(raw)
        summary = cast(HistorySummary, parse_structured_payload(HistorySummary, content))
    except ValidationError as exc:
        raise RuntimeError(f"context_compact structured output parse failed: {exc}") from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"context_compact model call failed: {exc}") from exc

    summary_turn = {
        "role": "system",
        "source": "context_compact",
        "summarized_turn_count": len(old_turns),
        "summary": summary.condensed_narrative,
        "summary_bullets": list(summary.key_points),
        "utilization": parsed.utilization,
    }
    new_turn_history = [summary_turn, *recent_turns]

    output_json = {
        "turn_history": new_turn_history,
        "summary_turn": summary_turn,
        "old_turn_count": len(old_turns),
        "retained_turn_count": len(recent_turns),
        "was_summarized": True,
        "utilization": parsed.utilization,
    }
    return OperationResult(
        output_text=(
            f"Compacted {len(old_turns)} turn(s) at {parsed.utilization:.1%} utilization; "
            f"retained {len(recent_turns)} latest turn(s)."
        ),
        output_json=output_json,
        extra={
            "was_summarized": True,
            "old_turn_count": len(old_turns),
            "retained_turn_count": len(recent_turns),
            "utilization": parsed.utilization,
            "summary_preview": summary.condensed_narrative[:180],
            "summary_bullets": list(summary.key_points),
        },
    )
