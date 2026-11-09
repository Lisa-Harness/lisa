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

An ``update`` decision is one atomic supersession: ``memory.add`` carries
``supersedes_id`` and a scoped active original retires in the same store
transaction — the operation performs no separate best-effort retire call.

A partial storage failure (some facts stored, then an error) is retryable:
the operation stops at the first storage error and its error result does
not enumerate partial writes — the store's write-time dedup makes the
re-mine idempotent, so the retry only fills what is missing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext
from kernel.operations.structured_output import (
    assistant_bootstrap_messages,
    completion_content,
    json_only_retry_messages,
    parse_structured_payload,
)
from pydantic import BaseModel, Field, ValidationError
from sdk.content import require_authored_text
from sdk.memory.write_result import parse_memory_add_id, parse_supersession_retired

logger = logging.getLogger(__name__)

# Authored-text path — loaded via ctx.host.read_authored_text (SDK loader).
_SYSTEM_PROMPT_PATH = "operations/memory/prompts/mine_thread_segment_system.md"

# Fallback default for the mining LLM token budget. The authoritative source
# is the session manifest's ``resources.packs.configuration.lisa-pack-memory``
# block (key ``memory_mine_max_tokens``), surfaced at runtime as
# ``ctx.pack_config``.
#
# Mirrors synthesize_learning's 16384 default (its documented lesson: local
# OpenAI-compatible providers truncate the JSON envelope mid-array when the
# budget is too small, and reasoning models — glm-5.3 — spend the completion
# budget thinking first). The 2026-11-08..09 crash-loop was exactly that
# shape: 21 consecutive ``mine_thread_segment distillation failed`` errors
# with the model's reasoning text ('Let me analyze this conv…') failing
# JSON validation, and one ``facts.0.fact: string_too_long`` truncation.
_MINE_MAX_TOKENS = 16384

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
    except json.JSONDecodeError as exc:
        logger.warning(
            "mine_thread_segment: dedup candidate search returned invalid JSON (%s); "
            "mining without existing records",
            exc,
        )
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

    Three attempts: structured output, a JSON-only retry with a
    schema-echoing user message, then an assistant-bootstrap prefill
    (``{"facts": [``) that forces the model to continue inside the JSON
    envelope — the poison shape the 2026-11-08..09 crash-loop logged was
    reasoning prose before any JSON, which a re-ask alone reproduces.
    Raises when all attempts fail.
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
    bootstrap_messages = assistant_bootstrap_messages(first_messages, MinedFacts, top_key="facts")
    bootstrap_prefix = str(bootstrap_messages[-1]["content"])
    attempts: tuple[tuple[int, list[dict[str, Any]], bool], ...] = (
        (1, first_messages, False),
        (2, json_only_retry_messages(first_messages, MinedFacts), False),
        (2, bootstrap_messages, True),
    )
    for multiplier, attempt_messages, is_bootstrap in attempts:
        try:
            raw = await ctx.host.complete(
                messages=attempt_messages,
                model_def=model_def,
                # Escalate the budget on the retries, mirroring
                # synthesize_learning (the kernel's truncation-rescue
                # pattern): on reasoning models the visible JSON shares the
                # completion budget with reasoning tokens, so re-issuing the
                # SAME budget with a JSON-only nudge just reproduces the
                # truncation (the 2026-11-08..09 mining crash-loop: 21 failed
                # segments, zero mined, because a 640-token budget starved
                # every attempt identically).
                max_tokens=mine_max_tokens * multiplier,
                temperature=0.1,
                response_format=MinedFacts if multiplier == 1 else None,
            )
            raw_text = completion_content(raw)
            if is_bootstrap:
                # The completion continues the prefilled envelope; parse the
                # concatenation, never the continuation alone.
                raw_text = bootstrap_prefix + raw_text
            mined = _parse_mined(raw_text)
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            continue
    if mined is None:
        raise RuntimeError(f"mine_thread_segment distillation failed: {last_exc}")
    return mined


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
) -> tuple[str, int | None, int | str | None, bool | None]:
    """Persist one fact and return its kind, IDs, and retirement outcome.

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
    minted = parse_memory_add_id(add_result)
    if minted is None:
        logger.warning("mine_thread_segment: provider did not return a numeric memory ID")
    retired = parse_supersession_retired(add_result) if supersedes is not None else None
    return ("update" if supersedes is not None else "create"), minted, supersedes, retired


class _SupersessionOutcome(TypedDict):
    memory_id: int | None
    supersedes_id: int | str
    retired: bool | None


@dataclass(slots=True)
class _MiningOutcome:
    created_ids: list[int] = field(default_factory=list)
    updated_ids: list[int] = field(default_factory=list)
    stored_unparsed: int = 0
    retired_ids: list[int | str] = field(default_factory=list)
    not_retired_by_write_ids: list[int | str] = field(default_factory=list)
    retirement_unknown_ids: list[int | str] = field(default_factory=list)
    supersession_outcomes: list[_SupersessionOutcome] = field(default_factory=list)


async def _apply_mining_decisions(
    ctx: OperationExecutionContext,
    parsed: Parameters,
    *,
    facts: list[MinedFact],
    candidate_ids: set[int | str],
    turn_lo: int | None,
    turn_hi: int | None,
) -> _MiningOutcome:
    """Persist each mined fact's decision and record confirmed outcomes.

    Raises :class:`_MiningStorageFailure` on the first storage error —
    facts after it are not attempted this pass.

    An ``update`` is one atomic supersession: ``memory.add`` carries
    ``supersedes_id`` and reports whether the scoped original was retired
    in the same transaction.
    """
    outcome = _MiningOutcome()
    for item in facts:
        result = await _persist_mined_fact(
            ctx,
            parsed,
            item=item,
            candidate_ids=candidate_ids,
            turn_lo=turn_lo,
            turn_hi=turn_hi,
        )
        kind, minted, supersedes, retired = result
        if minted is None:
            outcome.stored_unparsed += 1
        if kind == "update":
            if minted is not None:
                outcome.updated_ids.append(minted)
            assert supersedes is not None
            outcome.supersession_outcomes.append(
                {"memory_id": minted, "supersedes_id": supersedes, "retired": retired}
            )
            if retired is True:
                outcome.retired_ids.append(supersedes)
            elif retired is False:
                outcome.not_retired_by_write_ids.append(supersedes)
            else:
                logger.warning(
                    "mine_thread_segment: provider did not report retirement outcome for %s",
                    supersedes,
                )
                outcome.retirement_unknown_ids.append(supersedes)
        elif minted is not None:
            outcome.created_ids.append(minted)
    return outcome


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
    candidate_ids: set[int | str] = {
        rec_id for rec in candidates if (rec_id := rec.get("id")) is not None
    }

    # Distillation loop: 3 attempts (structured, JSON-only retry,
    # assistant-bootstrap prefill).
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
        outcome = await _apply_mining_decisions(
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

    stored_count = len(outcome.created_ids) + len(outcome.updated_ids) + outcome.stored_unparsed
    return OperationResult(
        output_text=(
            f"✅ Mined {stored_count} durable fact(s) from turns "
            f"{turn_lo}–{turn_hi} ({len(outcome.retired_ids)} retired, "
            f"{len(outcome.not_retired_by_write_ids)} not retired by this write, "
            f"{len(outcome.retirement_unknown_ids)} retirement outcomes unknown, "
            f"{len(mined.facts) - len(actionable)} skipped as duplicates)."
        ),
        extra={
            "mined": True,
            "memory_ids": outcome.created_ids + outcome.updated_ids,
            "created_ids": outcome.created_ids,
            "updated_ids": outcome.updated_ids,
            "stored_unparsed": outcome.stored_unparsed,
            "retired_ids": outcome.retired_ids,
            "not_retired_by_write_ids": outcome.not_retired_by_write_ids,
            "retirement_unknown_ids": outcome.retirement_unknown_ids,
            "supersession_outcomes": outcome.supersession_outcomes,
            "thread_turn_lo": turn_lo,
            "thread_turn_hi": turn_hi,
            "facts": [item.fact for item in actionable],
        },
    )
