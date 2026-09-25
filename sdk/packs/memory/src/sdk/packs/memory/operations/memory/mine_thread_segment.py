"""Mine a sealed thread segment into durable semantic memories.

The thread-mining slice (``sdk/packs/memory/slice_provider.py``) tracks
sealed thread segments via a ``mine.marker`` sidecar and invokes this
operation with each newly sealed segment's turns. The operation distills
the turns into 1-3 durable facts using the same structured-output loop as
``synthesize_learning`` / ``consolidate_semantic``, then writes each fact
via ``memory.add`` with ``memory_kind="semantic"``, boosted salience, and
``content_json.thread_turn_lo/hi`` provenance pointing back at the mined
turn range.

The thread itself is never mutated — mining is a pure read over the
lossless store. The segment IS the archive; there is nothing to retire.
"""

from __future__ import annotations

import logging
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
_SYSTEM_PROMPT_PATH = "operations/memory/prompts/mine_thread_segment_system.md"

# Fallback default for the mining LLM token budget. The authoritative source
# is the session manifest's ``resources.packs.configuration.lisa-pack-memory``
# block (key ``memory_mine_max_tokens``), surfaced at runtime as
# ``ctx.pack_config``.
_MINE_MAX_TOKENS = 640

#: Salience floor for mined semantic records (sealed-segment evidence is
#: cross-session verified by construction).
_MINED_SALIENCE_FLOOR = 0.7

#: Max turns of a segment rendered into the LLM user message. 256-turn
#: segments render to a large prompt; cap the input window to keep the
#: distillation call bounded (the slice budget bounds the rest).
_MAX_RENDERED_TURNS = 60


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
            "mine_thread_segment: pack_config[%r]=%r is not an int; using default %d",
            key,
            raw,
            default,
        )
        return default


def _resolve_settings(ctx: OperationExecutionContext) -> int:
    """Resolve the mining token budget from ``ctx.pack_config`` with a fallback."""
    return _cfg_int(ctx.pack_config, "memory_mine_max_tokens", _MINE_MAX_TOKENS)


def _load_system_prompt(ctx: OperationExecutionContext) -> str:
    """Load the mining system prompt via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """
    return require_authored_text(
        ctx.host.read_authored_text if ctx.host is not None else None,
        _SYSTEM_PROMPT_PATH,
        context="mine_thread_segment operation",
    )


class Parameters(BaseModel):
    """Slice-supplied inputs for the thread-mining operation."""

    turns: list[dict[str, Any]] = Field(
        ...,
        description="The sealed segment's thread entries, each with at least "
        "turn, query, and response_text.",
        min_length=1,
    )
    memory_scope: str = Field("personal", description="Privacy scope for mined records.")
    surface_id: str = Field("", description="Surface isolation key.")
    user_slug: str = Field("default", description="The user identity to scope to.")
    workspace_id: str = Field("default", description="Workspace isolation key.")


class MinedFacts(BaseModel):
    """Strict bounded output from the mining LLM call."""

    facts: list[str] = Field(
        ...,
        description="1 to 3 concise durable facts the turns proved; empty when "
        "nothing durable was found.",
        max_length=3,
    )


def _render_turns(turns: list[dict[str, Any]]) -> str:
    """Render the segment's turns as a numbered list for the LLM user message."""
    lines: list[str] = []
    for rec in turns[-_MAX_RENDERED_TURNS:]:
        turn = rec.get("turn", "?")
        query = str(rec.get("query", "")).strip()
        response = str(rec.get("response_text", "")).strip()
        preview = (response or query)[:400]
        if not query and not response:
            continue
        lines.append(f"[turn={turn}] {query[:200]}\n  -> {preview}")
    return "\n".join(lines)


def _parse_mined(content: str) -> MinedFacts:
    """Parse structured output; raise on JSON-shaped garbage (no prose fallback)."""
    normalized = (content or "").strip()
    if not normalized:
        raise ValueError("empty mining output")
    try:
        return cast(MinedFacts, parse_structured_payload(MinedFacts, normalized))
    except ValidationError as exc:
        # Mirror synthesize_learning's guard: JSON-shaped output that failed
        # validation must not be prose-parsed into garbage facts.
        if _looks_like_json(normalized):
            raise RuntimeError(
                "mine_thread_segment: model returned JSON-shaped output that "
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


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Distill a sealed segment's turns into durable semantic memories."""
    params = ctx.params or {}
    try:
        parsed = Parameters.model_validate(params)
    except ValidationError as exc:
        return OperationResult(output_text=f"⛔ Invalid mining parameters: {exc}")

    turns = parsed.turns
    if not turns:
        return OperationResult(output_text="⛔ No turns provided to mine.")

    if ctx.host is None:
        raise RuntimeError("mine_thread_segment requires the OperationHost seam")

    system_prompt = _load_system_prompt(ctx)
    turns_text = _render_turns(turns)
    if not turns_text.strip():
        return OperationResult(
            output_text="No mineable turns in segment (all entries empty).",
            extra={"mined": False, "memory_ids": []},
        )

    # Thread-turn provenance bounds for the mined records.
    turn_indices = [
        int(rec["turn"]) for rec in turns if isinstance(rec.get("turn"), int)
    ]
    turn_lo = min(turn_indices) if turn_indices else None
    turn_hi = max(turn_indices) if turn_indices else None

    # Distillation loop: 2 attempts (structured first, JSON-only retry).
    model_id = ctx.preferred_model_id
    model_def = ctx.host.resolve_model(model_id)
    if model_def is None:
        model_def = ctx.host.resolve_model("default")
    if model_def is None:
        raise RuntimeError("mine_thread_segment could not resolve a model")

    first_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": turns_text},
    ]
    mine_max_tokens = _resolve_settings(ctx)
    mined: MinedFacts | None = None
    last_exc: Exception | None = None
    for attempt_idx, attempt_messages in enumerate(
        (first_messages, json_only_retry_messages(first_messages, MinedFacts))
    ):
        try:
            raw = await ctx.host.complete(
                messages=attempt_messages,
                model_def=model_def,
                max_tokens=mine_max_tokens,
                temperature=0.1,
                response_format=MinedFacts if attempt_idx == 0 else None,
            )
            mined = _parse_mined(completion_content(raw))
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if attempt_idx == 0:
                continue
            break
    if mined is None:
        raise RuntimeError(f"mine_thread_segment distillation failed: {last_exc}")

    facts = [f.strip() for f in mined.facts if f.strip()]
    if not facts:
        return OperationResult(
            output_text="Nothing durable found in segment; nothing stored.",
            extra={"mined": False, "memory_ids": []},
        )

    # Persist each durable fact as a semantic record with thread-turn
    # provenance. The thread is never mutated — the segment is the archive.
    content_json: dict[str, object] = {
        "mined_from_segment": True,
    }
    if turn_lo is not None:
        content_json["thread_turn_lo"] = turn_lo
    if turn_hi is not None:
        content_json["thread_turn_hi"] = turn_hi

    memory_ids: list[int] = []
    for fact in facts:
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
                    "content": fact,
                    "user_slug": parsed.user_slug,
                    "workspace_id": parsed.workspace_id,
                    "surface_id": parsed.surface_id,
                    "memory_scope": parsed.memory_scope,
                    "memory_tier": "long_term",
                    "memory_kind": "semantic",
                    "operation_id": "mine_thread_segment",
                    "salience": _MINED_SALIENCE_FLOOR,
                    "confidence": 0.85,
                    "content_json": content_json,
                },
            ),
        )
        if getattr(add_result, "error", None):
            logger.warning(
                "mine_thread_segment: fact storage failed: %s", add_result.error
            )
            continue
        add_text = getattr(add_result, "output_text", "") or ""
        if "ID " in add_text:
            try:
                memory_ids.append(int(add_text.split("ID ", 1)[1].rstrip(".")))
            except (ValueError, IndexError):
                pass

    return OperationResult(
        output_text=(
            f"✅ Mined {len(facts)} durable fact(s) from turns "
            f"{turn_lo}–{turn_hi}: {len(memory_ids)} stored."
        ),
        extra={
            "mined": True,
            "memory_ids": memory_ids,
            "thread_turn_lo": turn_lo,
            "thread_turn_hi": turn_hi,
            "facts": facts,
        },
    )