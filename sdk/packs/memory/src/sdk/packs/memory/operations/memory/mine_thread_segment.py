"""Mine a sealed thread segment into durable semantic memories.

The thread-mining slice (``sdk/packs/memory/mining_slice.py``) tracks
sealed thread segments via a ``mine.marker`` sidecar and invokes this
operation with each newly sealed segment's turns. The operation distills
the turns into 1-3 durable facts using the same structured-output loop as
``synthesize_learning``, with the same write-time dedup: similar existing
records are fetched and handed to the distiller, which decides per fact
whether to create, update, or skip. Created/updated records carry
``memory_kind="semantic"``, boosted salience, and
``content_json.thread_turn_lo/hi`` provenance pointing back at the mined
turn range.

The thread itself is never mutated — mining is a pure read over the
lossless store. The segment IS the archive; updates retire only the
derived memory record they supersede.

Per-segment outcomes — the mining slice keys its durable ``mine.marker``
checkpoint on these:

- **success** — at least one fact stored: a normal result with
  ``extra["mined"] = True``.
- **intentional no-durable-knowledge** — the distiller skipped every fact
  as already stored: ``extra["mined"] = False`` plus a ``skipped`` count.
- **retryable storage failure** — any ``memory.add`` error:
  ``status="error"`` with ``error_detail``. The slice leaves the segment
  pending so the next pass retries it.

A partial storage failure (some facts stored, then an error) is retryable:
the operation stops at the first storage error and its error result does
not enumerate partial writes — the store's write-time dedup makes the
re-mine idempotent, so the retry only fills what is missing.
"""

from __future__ import annotations

import contextlib
import json
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

#: Existing records fetched for the dedup decision, per mining call.
_DEDUP_CANDIDATE_LIMIT = 8

#: Max turn queries joined into the dedup-candidate search text.
_DEDUP_QUERY_TURN_CAP = 10


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


class MinedFact(BaseModel):
    """One dedup-decided fact from the mining call."""

    action: str = Field(
        ...,
        description="'create' (new durable knowledge), 'update' (refines or corrects "
        "the existing record with id ``supersedes_id``), or 'skip' (already stored).",
        pattern="^(create|update|skip)$",
    )
    fact: str = Field(
        ...,
        description="The concise, self-contained durable fact. Empty only when action is 'skip'.",
        min_length=0,
        max_length=400,
    )
    supersedes_id: int | str | None = Field(
        default=None,
        description="Required when action is 'update': the id of the existing record "
        "this fact supersedes. Must be one of the ids listed in the input.",
    )
    reason: str = Field(
        default="",
        description="One short clause: why create/update/skip.",
        max_length=200,
    )


class MinedFacts(BaseModel):
    """Strict bounded output from the mining LLM call."""

    facts: list[MinedFact] = Field(
        ...,
        description="0 to 3 dedup-decided durable facts the turns proved; empty "
        "when nothing durable was found.",
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
    if stripped.startswith(("```", "{", "[")):
        return True
    return bool("{" in stripped and "}" in stripped)


def _dedup_query(turns: list[dict[str, Any]]) -> str:
    """Join the segment's user queries into a search text for dedup candidates."""
    queries: list[str] = []
    for rec in turns:
        query = str(rec.get("query", "")).strip()
        if query:
            queries.append(query)
        if len(queries) >= _DEDUP_QUERY_TURN_CAP:
            break
    return "\n".join(queries)


async def _dedup_candidates(
    ctx: OperationExecutionContext, parsed: Parameters, query: str
) -> list[dict[str, Any]]:
    """Fetch scoped existing records similar to the segment for dedup.

    A search failure degrades to an empty candidate list: mining proceeds
    and creates records (it never fails because dedup candidates could not
    be loaded).
    """
    assert ctx.host is not None
    search_ctx = ToolExecutionContext(
        tool_id="memory.search",
        workspace_id=parsed.workspace_id,
        user_id=ctx.user_id,
        user_slug=parsed.user_slug,
        surface_id=parsed.surface_id,
        memory_scope=parsed.memory_scope,  # type: ignore[arg-type]
        mode_id=ctx.mode_id,
        query=query,
        parameters={
            "query": query,
            "user_slug": parsed.user_slug,
            "workspace_id": parsed.workspace_id,
            "surface_id": parsed.surface_id,
            "memory_scope": parsed.memory_scope,
            "limit": _DEDUP_CANDIDATE_LIMIT,
        },
    )
    try:
        result = await ctx.host.execute_tool("memory.search", search_ctx)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "mine_thread_segment: dedup candidate search failed (%s); "
            "mining without existing records",
            exc,
        )
        return []
    if getattr(result, "error", None):
        logger.warning(
            "mine_thread_segment: dedup candidate search error: %s; "
            "mining without existing records",
            result.error,
        )
        return []
    try:
        data = json.loads(getattr(result, "output_text", "") or "[]")
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]
    return [item for item in data if isinstance(item, dict)]


def _render_dedup_candidates(candidates: list[dict[str, Any]]) -> str:
    """Render existing records for the distiller's user message."""
    if not candidates:
        return "Existing memories: (none)"
    lines = ["Existing memories that may already cover this segment:"]
    for rec in candidates:
        rid = rec.get("id", "?")
        content = str(rec.get("content", "")).strip()
        created = str(rec.get("created_at", ""))[:10]
        lines.append(f"- [id={rid}, created={created}] {content}")
    return "\n".join(lines)


async def _distill_facts(
    ctx: OperationExecutionContext,
    *,
    system_prompt: str,
    turns_text: str,
    candidates: list[dict[str, Any]],
    mine_max_tokens: int,
) -> MinedFacts:
    """Distill a segment's rendered turns into durable facts via the model.

    Two attempts: structured output first, then a JSON-only retry with a
    schema-echoing user message. Raises when both attempts fail.
    """
    model_id = ctx.preferred_model_id
    model_def = ctx.host.resolve_model(model_id)
    if model_def is None:
        model_def = ctx.host.resolve_model("default")
    if model_def is None:
        raise RuntimeError("mine_thread_segment could not resolve a model")

    user_content = f"{turns_text}\n\n{_render_dedup_candidates(candidates)}"
    first_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
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
    return mined


async def _retire_record(
    ctx: OperationExecutionContext, parsed: Parameters, record_id: int | str
) -> bool:
    """Retire one superseded record; returns True when the backend confirmed."""
    assert ctx.host is not None
    retire_ctx = ToolExecutionContext(
        tool_id="memory.retire",
        workspace_id=parsed.workspace_id,
        user_id=ctx.user_id,
        user_slug=parsed.user_slug,
        surface_id=parsed.surface_id,
        memory_scope=parsed.memory_scope,  # type: ignore[arg-type]
        mode_id=ctx.mode_id,
        parameters={
            "memory_id": record_id,
            "user_slug": parsed.user_slug,
            "workspace_id": parsed.workspace_id,
            "surface_id": parsed.surface_id,
            "memory_scope": parsed.memory_scope,
        },
    )
    try:
        result = await ctx.host.execute_tool("memory.retire", retire_ctx)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "mine_thread_segment: retire of superseded record %s failed: %s",
            record_id,
            exc,
        )
        return False
    if getattr(result, "error", None):
        logger.warning(
            "mine_thread_segment: retire of superseded record %s errored: %s",
            record_id,
            result.error,
        )
        return False
    return True


def _parse_memory_id(add_text: str) -> int | None:
    """Extract the minted record id from a ``Memory stored with ID <n>`` reply."""
    if "ID " not in add_text:
        return None
    with contextlib.suppress(ValueError, IndexError):
        return int(add_text.split("ID ", 1)[1].rstrip("."))
    return None


def _validate_mining_params(
    ctx: OperationExecutionContext,
) -> tuple[Parameters, str, OperationResult | None]:
    """Validate the operation's params and mining preconditions.

    Returns:
        A ``(parsed, turns_text, error_result)`` tuple. ``error_result``
        is ``None`` on success, or an OperationResult the caller should
        return immediately (invalid params, no turns, or no mineable
        turn content). ``turns_text`` is the rendered turn content (only
        meaningful on success).
    """
    params = ctx.params or {}
    try:
        parsed = Parameters.model_validate(params)
    except ValidationError as exc:
        return (
            Parameters.model_construct(),
            "",
            OperationResult(output_text=f"⛔ Invalid mining parameters: {exc}"),
        )

    turns = parsed.turns
    if not turns:
        return parsed, "", OperationResult(output_text="⛔ No turns provided to mine.")

    if ctx.host is None:
        raise RuntimeError("mine_thread_segment requires the OperationHost seam")

    turns_text = _render_turns(turns)
    if not turns_text.strip():
        return (
            parsed,
            turns_text,
            OperationResult(
                output_text="No mineable turns in segment (all entries empty).",
                extra={"mined": False, "memory_ids": []},
            ),
        )
    return parsed, turns_text, None


class _MiningStorageFailure(RuntimeError):
    """A memory.add call errored while persisting one mined fact."""


async def _persist_mined_fact(
    ctx: OperationExecutionContext,
    parsed: Parameters,
    *,
    item: MinedFact,
    candidate_ids: set[int | str],
    turn_lo: int | None,
    turn_hi: int | None,
) -> tuple[str, int | None, int | str | None]:
    """Persist one mined fact; return ``(kind, minted_id, supersedes_id)``.

    Raises :class:`_MiningStorageFailure` when the storage call errors —
    the caller surfaces a typed retryable failure instead of skipping the
    fact silently.
    """
    fact = item.fact.strip()
    supersedes = item.supersedes_id if item.supersedes_id in candidate_ids else None
    if item.action == "update" and supersedes is None:
        logger.warning(
            "mine_thread_segment: update without a valid supersedes_id (got %r); "
            "storing as a new record",
            item.supersedes_id,
        )

    content_json: dict[str, object] = {"mined_from_segment": True}
    if turn_lo is not None:
        content_json["thread_turn_lo"] = turn_lo
    if turn_hi is not None:
        content_json["thread_turn_hi"] = turn_hi
    parameters: dict[str, Any] = {
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
    }
    if supersedes is not None:
        content_json["supersedes_id"] = supersedes
        parameters["supersedes_id"] = supersedes

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
            parameters=parameters,
        ),
    )
    if getattr(add_result, "error", None):
        raise _MiningStorageFailure(str(add_result.error))
    minted = _parse_memory_id(getattr(add_result, "output_text", "") or "")
    return ("update" if supersedes is not None else "create"), minted, supersedes


async def _apply_mining_decisions(
    ctx: OperationExecutionContext,
    parsed: Parameters,
    *,
    facts: list[MinedFact],
    candidate_ids: set[int | str],
    turn_lo: int | None,
    turn_hi: int | None,
) -> tuple[list[int], list[int], list[int | str]]:
    """Persist each mined fact's decision; returns (created, updated, retired).

    Raises :class:`_MiningStorageFailure` on the first storage error —
    facts after it are not attempted this pass.
    """
    created_ids: list[int] = []
    updated_ids: list[int] = []
    retired_ids: list[int | str] = []
    for item in facts:
        result = await _persist_mined_fact(
            ctx,
            parsed,
            item=item,
            candidate_ids=candidate_ids,
            turn_lo=turn_lo,
            turn_hi=turn_hi,
        )
        kind, minted, supersedes = result
        if kind == "update":
            if minted is not None:
                updated_ids.append(minted)
            assert supersedes is not None
            if await _retire_record(ctx, parsed, supersedes):
                retired_ids.append(supersedes)
        elif minted is not None:
            created_ids.append(minted)
    return created_ids, updated_ids, retired_ids


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Distill a sealed segment's turns into dedup-decided semantic memories."""
    parsed, turns_text, invalid = _validate_mining_params(ctx)
    if invalid is not None:
        return invalid
    turns = parsed.turns

    system_prompt = _load_system_prompt(ctx)

    # Thread-turn provenance bounds for the mined records.
    turn_indices = [int(rec["turn"]) for rec in turns if isinstance(rec.get("turn"), int)]
    turn_lo = min(turn_indices) if turn_indices else None
    turn_hi = max(turn_indices) if turn_indices else None

    # Write-time dedup: fetch similar existing records for the distiller.
    candidates = await _dedup_candidates(ctx, parsed, _dedup_query(turns))
    candidate_ids = {rec.get("id") for rec in candidates if rec.get("id") is not None}

    # Distillation loop: 2 attempts (structured first, JSON-only retry).
    mined = await _distill_facts(
        ctx,
        system_prompt=system_prompt,
        turns_text=turns_text,
        candidates=candidates,
        mine_max_tokens=_resolve_settings(ctx),
    )

    actionable = [item for item in mined.facts if item.action != "skip" and item.fact.strip()]
    if not actionable:
        return OperationResult(
            output_text="Nothing new mined from segment; existing memories already cover it.",
            extra={"mined": False, "memory_ids": [], "skipped": len(mined.facts)},
        )

    try:
        created_ids, updated_ids, retired_ids = await _apply_mining_decisions(
            ctx,
            parsed,
            facts=actionable,
            candidate_ids=candidate_ids,
            turn_lo=turn_lo,
            turn_hi=turn_hi,
        )
    except _MiningStorageFailure as exc:
        # Typed retryable failure: the mining slice keys its checkpoint on
        # status="error" and leaves this segment pending. extra keeps the
        # public shape (mined False, no ids) — the store's write-time dedup
        # makes the re-mine idempotent rather than this result enumerating
        # partial writes.
        logger.warning("mine_thread_segment: segment storage failed: %s", exc)
        return OperationResult(
            output_text=f"⛔ Mining storage failed: {exc}",
            status="error",
            error_detail=str(exc),
            extra={"mined": False, "memory_ids": []},
        )

    stored_count = len(created_ids) + len(updated_ids)
    return OperationResult(
        output_text=(
            f"✅ Mined {stored_count} durable fact(s) from turns "
            f"{turn_lo}–{turn_hi} ({len(retired_ids)} superseded, "
            f"{len(mined.facts) - len(actionable)} skipped as duplicates)."
        ),
        extra={
            "mined": True,
            "memory_ids": created_ids + updated_ids,
            "created_ids": created_ids,
            "updated_ids": updated_ids,
            "retired_ids": retired_ids,
            "thread_turn_lo": turn_lo,
            "thread_turn_hi": turn_hi,
            "facts": [item.fact for item in actionable],
        },
    )
