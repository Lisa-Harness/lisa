"""Synthesize and store learning operation.

Distills raw conversation content into 1-3 durable facts using the operation's
preferred model, then persists those facts to the memory store — with
write-time dedup: before distilling, the operation searches the scoped store
for similar existing records and hands them to the distiller, which must
decide per fact whether to ``create`` a new record, ``update`` an existing
one (supersede + retire), or ``skip`` because the knowledge is already
stored. Unconditional inserts are the failure mode this replaces: repeated
captures of the same preference accumulate near-duplicates that later
outnumber and outvote each other in recall.

Captures are stored as ``memory_kind="semantic"`` — they are distilled,
self-contained facts by construction, not raw conversation events.
"""

from __future__ import annotations

import json
import logging
import re
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

#: Facts kept per distilled learning (bullet/sentence fallback cap).
_MAX_DISTILLED_FACTS = 3
#: Existing records fetched for the dedup decision, per distillation call.
_DEDUP_CANDIDATE_LIMIT = 8

# Authored-text path — loaded via ctx.host.read_authored_text (SDK loader).
_SYSTEM_PROMPT_PATH = "operations/memory/prompts/synthesize_learning_system.md"

# Fallback default for the distillation LLM token budget. The authoritative
# source is the session manifest's
# ``resources.packs.configuration.lisa-pack-memory.<key>`` block, surfaced to
# the op at runtime as ``ctx.pack_config``. This constant is only the
# last-resort fallback when no manifest config is supplied so the op still
# works in a bare host.
#
# Manifest key (under ``resources.packs.configuration.lisa-pack-memory``):
#   memory_distill_max_tokens.
#
# Local OpenAI-compatible providers (Ollama) frequently truncate the JSON
# envelope + 3 facts mid-array when the budget is too small, which the
# ``_looks_like_json`` guard correctly refuses to fall back to prose.
# Reasoning models (glm-5.3) also spend this budget thinking first.
_DISTILL_MAX_TOKENS = 16384


@dataclass(frozen=True)
class _ResolvedSettings:
    """Resolved operation budgets: ``ctx.pack_config`` overrides module fallbacks.

    The authoritative source is the session manifest's
    ``resources.packs.configuration.lisa-pack-memory`` block, surfaced at
    runtime as ``ctx.pack_config`` (an immutable MappingProxyType populated
    by ``kernel/operations/registry._resolve_execution_configuration`` before
    ``execute`` runs). The module ``_*`` constant is the last-resort fallback
    used only when the manifest supplies no value.
    """

    distill_max_tokens: int


def _cfg_int(pack_config: Any, key: str, default: int) -> int:
    """Read an int from ``ctx.pack_config`` with a module-constant fallback.

    Coerces non-int values to the default rather than raising, since pack
    config is session-authored and a malformed value should not crash the
    op mid-turn.
    """
    if pack_config is None:
        return default
    raw = pack_config.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "synthesize_learning: pack_config[%r]=%r is not an int; using default %d",
            key,
            raw,
            default,
        )
        return default


def _resolve_settings(ctx: OperationExecutionContext) -> _ResolvedSettings:
    """Resolve operation budgets from ``ctx.pack_config`` with fallbacks."""
    return _ResolvedSettings(
        distill_max_tokens=_cfg_int(
            ctx.pack_config, "memory_distill_max_tokens", _DISTILL_MAX_TOKENS
        ),
    )


def _load_system_prompt(ctx: OperationExecutionContext) -> str:
    """Load the distillation system prompt via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """

    return require_authored_text(
        ctx.host.read_authored_text if ctx.host is not None else None,
        _SYSTEM_PROMPT_PATH,
        context="synthesize_learning operation",
    )


class Parameters(BaseModel):
    content: str = Field(
        ...,
        description="The raw learning, conclusion, or strategic decision to distill and store.",
    )
    salience: float = Field(
        0.5,
        description="Importance from 0.0 (trivial context) to 1.0 (critical strategic alignment).",
    )
    memory_tier: str = Field(
        "daily",
        description=(
            "Memory tier: 'daily', 'weekly', or 'long_term'. Defaults to "
            "'daily' so model-called captures (which often omit the tier) "
            "land as short-lived workflow state rather than polluting "
            "long-term recall. The auto-capture path overrides this with "
            "the salience-graded tier."
        ),
    )


class DistilledFact(BaseModel):
    """One dedup-decided fact from the distillation call."""

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
        description="One short clause: why create/update/skip (e.g. 'new preference', "
        "'refines existing', 'duplicate of existing').",
        max_length=200,
    )


class DistilledLearning(BaseModel):
    """Strict bounded output from the distillation LLM call."""

    facts: list[DistilledFact] = Field(
        ...,
        description="1 to 3 dedup-decided facts distilled from the raw content.",
        min_length=0,
        max_length=3,
    )


def _parse_bullet_facts(normalized: str) -> list[str]:
    """Extract up to three facts from bullet/list lines (non-JSON providers)."""
    bullet_pattern = re.compile(r"^(?:[-*\u2022]|\d+[\.)\]])\s+(?P<fact>.+)$")
    facts: list[str] = []
    for line in normalized.splitlines():
        match = bullet_pattern.match(line.strip())
        if not match:
            continue
        fact = match.group("fact").strip()
        if fact:
            facts.append(fact)
        if len(facts) == _MAX_DISTILLED_FACTS:
            break
    return facts


def _parse_sentence_facts(normalized: str) -> list[str]:
    """Extract up to three facts from plain prose (no JSON, no bullets)."""
    sentence_facts: list[str] = []
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", normalized):
        candidate = chunk.strip().strip('"').strip("'")
        if not candidate:
            continue
        sentence_facts.append(candidate)
        if len(sentence_facts) == _MAX_DISTILLED_FACTS:
            break
    return sentence_facts


def _fallback_learning(normalized: str, source: str) -> DistilledLearning:
    """Wrap prose-parsed facts as creates when the provider cannot emit JSON.

    Dedup is impossible on this path — the provider returned prose even
    after the JSON-only retry — so every fallback fact becomes a
    ``create``. That is the honest degradation: the knowledge is kept (the
    capture still succeeds) at the cost of one duplicate-creation
    opportunity, and the warning makes the degradation visible.
    """
    facts = _parse_bullet_facts(normalized) or _parse_sentence_facts(normalized)
    logger.warning(
        "synthesize_and_store_learning: provider returned %s after the JSON-only "
        "retry; storing %d fact(s) without a dedup decision",
        source,
        len(facts),
    )
    return DistilledLearning(
        facts=[
            DistilledFact(action="create", fact=fact, reason=f"prose fallback ({source})")
            for fact in facts
        ]
    )


def _parse_distilled_learning(content: str) -> DistilledLearning:
    """Parse structured output with a bullet-list fallback for non-JSON providers."""
    normalized = (content or "").strip()
    if not normalized:
        raise ValueError("empty distillation output")

    try:
        return cast(DistilledLearning, parse_structured_payload(DistilledLearning, normalized))
    except ValidationError as direct_exc:
        # Guard: if the response looks like JSON (fenced or bare object), the
        # prose/bullet fallbacks would misinterpret JSON syntax lines as
        # facts (e.g. ``json``, ``{``, ``"facts": [``). That produced garbage
        # memory records like ``json | { | facts": [``. When the model tried
        # to return JSON but it failed to validate, raise rather than persist
        # JSON fragments as facts. The caller surfaces the failure instead.
        if _looks_like_json(normalized):
            raise RuntimeError(
                "synthesize_and_store_learning: model returned JSON-shaped "
                "output that failed validation; refusing to fall back to prose "
                f"parsing. Validation: {_validation_summary(direct_exc)}. "
                f"Raw output: {normalized[:200]!r}"
            ) from direct_exc
        bullet_facts = _parse_bullet_facts(normalized)
        if bullet_facts:
            return _fallback_learning(normalized, "bullet-list prose")

        # Fallback for providers that return plain prose instead of JSON or bullets.
        sentence_facts = _parse_sentence_facts(normalized)
        if sentence_facts:
            return _fallback_learning(normalized, "plain prose")

        raise direct_exc


def _validation_summary(exc: ValidationError) -> str:
    """One-line ``loc: type`` list, e.g. ``facts: too_long`` or ``json_invalid``."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['type']}" for err in exc.errors()
    )


def _looks_like_json(text: str) -> bool:
    """Return True when the response looks like JSON (fenced or bare object).

    Used to gate the prose fallbacks in ``_parse_distilled_learning``. When
    the model attempted JSON but it failed to validate, persisting JSON
    syntax fragments as facts produces garbage memory records.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Fenced JSON (```json ... ``` or ``` ... ```) or a bare object/array.
    if stripped.startswith(("```", "{", "[")):
        return True
    # Inner brace after any leading whitespace/prose — a strong JSON signal.
    return bool("{" in stripped and "}" in stripped)


def _default_learning_content(ctx: OperationExecutionContext) -> str:
    """Build learning input from the completed exchange when params omit content."""
    query = (ctx.query or "").strip()
    response = (ctx.response or "").strip()
    if query and response:
        return f"User asked:\n{query}\n\nLisa answered:\n{response}"
    return query or response


async def _dedup_candidates(
    ctx: OperationExecutionContext, raw_content: str
) -> list[dict[str, Any]]:
    """Fetch scoped existing records similar to ``raw_content`` for dedup.

    Uses ``memory.search`` (semantic/lexical with the raw content as the
    query). A search failure is logged and degraded to an empty candidate
    list — the distillation then has nothing to dedup against and creates
    new records (the capture never fails because dedup candidates could
    not be loaded).
    """
    assert ctx.host is not None
    search_ctx = ToolExecutionContext(
        tool_id="memory.search",
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        surface_id=ctx.surface_id,
        memory_scope=ctx.memory_scope,
        mode_id=ctx.mode_id,
        query=raw_content,
        parameters={
            "query": raw_content,
            "user_slug": ctx.user_slug,
            "workspace_id": ctx.workspace_id,
            "surface_id": ctx.surface_id,
            "memory_scope": ctx.memory_scope,
            "limit": _DEDUP_CANDIDATE_LIMIT,
        },
    )
    try:
        result = await ctx.host.execute_tool("memory.search", search_ctx)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "synthesize_and_store_learning: dedup candidate search failed (%s); "
            "distilling without existing records",
            exc,
        )
        return []
    if getattr(result, "error", None):
        logger.warning(
            "synthesize_and_store_learning: dedup candidate search error: %s; "
            "distilling without existing records",
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
    lines = ["Existing memories that may already cover this content:"]
    for rec in candidates:
        rid = rec.get("id", "?")
        content = str(rec.get("content", "")).strip()
        created = str(rec.get("created_at", ""))[:10]
        lines.append(f"- [id={rid}, created={created}] {content}")
    return "\n".join(lines)


async def _distill_facts(
    ctx: OperationExecutionContext,
    *,
    model_def: Any,
    system_prompt: str,
    raw_content: str,
    candidates: list[dict[str, Any]],
    distill_max_tokens: int,
) -> DistilledLearning:
    """Run the distillation loop and return the dedup-decided facts.

    The distillation loop. Local OpenAI-compatible models (Ollama) frequently
    ignore ``response_format`` and return empty content, or emit JSON that
    truncates mid-array (``max_tokens`` too small for the envelope + 3 facts).
    Both produce unparseable output. Per the structured-output contract
    (/memories/repo/structured-output-contract.md), retry once with an
    explicit JSON-only instruction before surfacing the failure. The retry
    covers ALL first-attempt failures — empty output, JSON-truncation
    (RuntimeError from the _looks_like_json guard), and ValidationError.
    """
    if ctx.host is None:
        raise RuntimeError("synthesize_and_store_learning requires the OperationHost seam")
    user_content = f"{raw_content}\n\n{_render_dedup_candidates(candidates)}"
    first_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    distilled: DistilledLearning | None = None
    last_exc: Exception | None = None
    for attempt_idx, attempt_messages in enumerate(
        (first_messages, json_only_retry_messages(first_messages, DistilledLearning))
    ):
        try:
            raw = await ctx.host.complete(
                messages=attempt_messages,
                model_def=model_def,
                # Escalate the budget on the retry, mirroring the kernel's
                # truncation-rescue pattern (finish_reason=length → one retry
                # at 2×): on reasoning models served via Ollama the visible
                # JSON shares the completion budget with reasoning tokens,
                # so the first attempt's budget can truncate the array
                # mid-fact (the 2026-09-23 finances-profile incidents) —
                # re-issuing the SAME budget with a JSON-only nudge just
                # reproduces the truncation.
                max_tokens=distill_max_tokens * (2 if attempt_idx else 1),
                temperature=0.1,
                # First attempt asks for native structured output; the retry
                # clears response_format so strict hosts don't re-reject and
                # the model emits free-form JSON per the appended instruction.
                response_format=DistilledLearning if attempt_idx == 0 else None,
            )
            distilled = _parse_distilled_learning(completion_content(raw))
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if attempt_idx == 0:
                continue
            break
    if distilled is None:
        raise RuntimeError(
            f"synthesize_and_store_learning distillation failed: {last_exc}"
        ) from last_exc
    return distilled


def _build_store_content_json(ctx: OperationExecutionContext) -> dict[str, object]:
    """Build the stored record's content_json provenance payload.

    Memory → thread linkage: when the host stamps the global turn index into
    the capture context, carry it into the stored record so recall can point
    back to the exact turn the learning came from.
    """
    content_json: dict[str, object] = {"query": ctx.query, "model": ctx.model_id}
    if "had_briefing" in ctx.extra:
        content_json["had_chase_briefing"] = ctx.extra["had_briefing"]
    thread_turn = ctx.extra.get("thread_turn")
    if isinstance(thread_turn, int):
        content_json["thread_turn"] = thread_turn
    return content_json


def _parse_memory_id(add_text: str) -> int | None:
    """Extract the minted record id from a ``Memory stored with ID <n>`` reply."""
    if "ID " not in add_text:
        return None
    try:
        return int(add_text.split("ID ", 1)[1].rstrip("."))
    except (ValueError, IndexError):
        return None


async def _retire_record(ctx: OperationExecutionContext, record_id: int | str) -> bool:
    """Retire one superseded record; returns True when the backend confirmed."""
    assert ctx.host is not None
    retire_ctx = ToolExecutionContext(
        tool_id="memory.retire",
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        surface_id=ctx.surface_id,
        memory_scope=ctx.memory_scope,
        mode_id=ctx.mode_id,
        parameters={
            "memory_id": record_id,
            "user_slug": ctx.user_slug,
            "workspace_id": ctx.workspace_id,
            "surface_id": ctx.surface_id,
            "memory_scope": ctx.memory_scope,
        },
    )
    try:
        result = await ctx.host.execute_tool("memory.retire", retire_ctx)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "synthesize_and_store_learning: retire of superseded record %s failed: %s "
            "(superseding record was written; the original stays active)",
            record_id,
            exc,
        )
        return False
    if getattr(result, "error", None):
        logger.warning(
            "synthesize_and_store_learning: retire of superseded record %s errored: %s "
            "(superseding record was written; the original stays active)",
            record_id,
            result.error,
        )
        return False
    return True


@dataclass(slots=True)
class _UpsertOutcome:
    """Accounting of one distillation's applied fact decisions."""

    created_ids: list[int]
    updated_ids: list[int]
    retired_ids: list[int | str]
    #: Adds that succeeded but whose id the backend did not echo — stored
    #: work without thread linkage.
    stored_unparsed: int
    skipped: int
    error: str = ""


class _FactStorageFailure(RuntimeError):
    """A memory.add call errored while persisting one fact's decision."""


async def _apply_one_fact(
    ctx: OperationExecutionContext,
    *,
    item: DistilledFact,
    candidate_ids: set[int | str],
    content_json_base: dict[str, object],
    memory_tier: str,
    salience: float,
    memory_operation_id: str,
) -> tuple[str, int | None, int | str | None]:
    """Persist one fact's decision; return ``(kind, minted_id, supersedes_id)``.

    ``kind`` is ``"create"``, ``"update"``, or ``"skip"``. Raises
    ``_FactStorageFailure`` when the storage call errors (the caller turns
    it into the typed error result).
    """
    assert ctx.host is not None
    if item.action == "skip" or not item.fact.strip():
        return "skip", None, None

    supersedes = item.supersedes_id if item.supersedes_id in candidate_ids else None
    if item.action == "update" and supersedes is None:
        # Update without a valid target id degrades to a create — the
        # knowledge is real, only the dedup linkage failed.
        logger.warning(
            "synthesize_and_store_learning: update without a valid supersedes_id "
            "(got %r); storing as a new record",
            item.supersedes_id,
        )

    content_json: dict[str, object] = dict(content_json_base)
    # Captures are distilled, self-contained facts — semantic, not episodic.
    # Raw conversation events never reach this path (the distiller refuses
    # to persist them), so episodic had no honest writer left.
    parameters: dict[str, Any] = {
        "content": item.fact.strip(),
        "user_slug": ctx.user_slug,
        "workspace_id": ctx.workspace_id,
        "surface_id": ctx.surface_id,
        "memory_scope": ctx.memory_scope,
        "memory_tier": memory_tier,
        "memory_kind": "semantic",
        "salience": salience,
        "operation_id": memory_operation_id,
        "confidence": 1.0,
        "content_json": content_json,
    }
    if supersedes is not None:
        content_json["supersedes_id"] = supersedes
        parameters["supersedes_id"] = supersedes

    tool_result = await ctx.host.execute_tool(
        "memory.add",
        ToolExecutionContext(
            tool_id="memory.add",
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters=parameters,
        ),
    )
    if getattr(tool_result, "error", None):
        raise _FactStorageFailure(str(tool_result.error))

    minted = _parse_memory_id(getattr(tool_result, "output_text", "") or "")
    return ("update" if supersedes is not None else "create"), minted, supersedes


async def _apply_fact_decisions(
    ctx: OperationExecutionContext,
    *,
    distilled: DistilledLearning,
    candidate_ids: set[int | str],
    content_json_base: dict[str, object],
    memory_tier: str,
    salience: float,
    memory_operation_id: str,
) -> _UpsertOutcome:
    """Apply each fact's dedup decision, retiring superseded originals."""
    outcome = _UpsertOutcome(
        created_ids=[], updated_ids=[], retired_ids=[], stored_unparsed=0, skipped=0
    )
    for item in distilled.facts:
        kind, minted, supersedes = await _apply_one_fact(
            ctx,
            item=item,
            candidate_ids=candidate_ids,
            content_json_base=content_json_base,
            memory_tier=memory_tier,
            salience=salience,
            memory_operation_id=memory_operation_id,
        )
        if kind == "skip":
            outcome.skipped += 1
            continue
        if kind == "update":
            if minted is not None:
                outcome.updated_ids.append(minted)
            else:
                outcome.stored_unparsed += 1
            assert supersedes is not None
            if await _retire_record(ctx, supersedes):
                outcome.retired_ids.append(supersedes)
        elif minted is not None:
            outcome.created_ids.append(minted)
        else:
            outcome.stored_unparsed += 1
    return outcome


@dataclass(frozen=True)
class _CaptureRequest:
    """One capture's resolved inputs: content, tier, salience, model."""

    raw_content: str
    memory_tier: str
    salience: float
    model_def: Any


def _resolve_capture_request(ctx: OperationExecutionContext) -> _CaptureRequest | OperationResult:
    """Resolve the capture's inputs; an OperationResult short-circuits execute.

    Content defaults to the completed exchange when params omit it. The tier
    defaults to 'daily' so model-called captures that omit the tier land as
    short-lived workflow state; the auto-capture path always supplies an
    explicit, salience-graded tier, so this default only binds the
    model-invoked path (see Parameters.memory_tier).
    """
    params = ctx.params or {}
    raw_content = str(params.get("content", "") or "").strip() or _default_learning_content(ctx)
    if not raw_content:
        return OperationResult(output_text="⛔ No content provided to store.")

    if ctx.host is None:
        raise RuntimeError("synthesize_and_store_learning requires the OperationHost seam")
    model_id = ctx.preferred_model_id
    model_def = ctx.host.resolve_model(model_id)
    if model_def is None:
        model_def = ctx.host.resolve_model("default")
    if model_def is None:
        raise RuntimeError("synthesize_and_store_learning could not resolve a model")

    return _CaptureRequest(
        raw_content=raw_content,
        memory_tier=params.get("memory_tier", "daily"),
        salience=float(params.get("salience", 0.5)),
        model_def=model_def,
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Distill raw content into dedup-decided facts, then upsert to memory."""
    # Keep learning records under one canonical operation id so recall queries stay stable.
    memory_operation_id = "save_learning"

    resolved = _resolve_capture_request(ctx)
    if isinstance(resolved, OperationResult):
        return resolved
    raw_content = resolved.raw_content
    memory_tier = resolved.memory_tier
    salience = resolved.salience
    model_def = resolved.model_def

    # Step 1: Distill with the LLM. Distillation is required (not optional) —
    # a missing completion function, unresolved model, or authored prompt
    # raises rather than silently persisting the raw conversation text.

    system_prompt = _load_system_prompt(ctx)
    # Token budget for the distillation call, resolved from ``ctx.pack_config``
    # (key ``memory_distill_max_tokens`` under
    # ``resources.packs.configuration.lisa-pack-memory``) with a module
    # fallback. See ``_DISTILL_MAX_TOKENS`` for the truncation history.
    distill_max_tokens = _resolve_settings(ctx).distill_max_tokens

    # Write-time dedup: fetch similar existing records so the distiller can
    # decide create/update/skip per fact.
    candidates = await _dedup_candidates(ctx, raw_content)
    candidate_ids = {rec.get("id") for rec in candidates if rec.get("id") is not None}

    distilled = await _distill_facts(
        ctx,
        model_def=model_def,
        system_prompt=system_prompt,
        raw_content=raw_content,
        candidates=candidates,
        distill_max_tokens=distill_max_tokens,
    )

    # Step 2: Apply each fact's decision.
    if ctx.host is None:
        return OperationResult(
            output_text="Learning storage unavailable: tool execution not injected."
        )

    content_json_base = _build_store_content_json(ctx)
    try:
        applied = await _apply_fact_decisions(
            ctx,
            distilled=distilled,
            candidate_ids=candidate_ids,
            content_json_base=content_json_base,
            memory_tier=memory_tier,
            salience=salience,
            memory_operation_id=memory_operation_id,
        )
    except _FactStorageFailure as exc:
        # status="error" is the typed "operation failed" contract: the
        # registry warns, effect receipts are skipped, and the dispatch
        # layer counts a consecutive failure. A bare success envelope with
        # a ⛔ message hid memory-persistence outages from telemetry and
        # recovery (2026-11-02 WiDWebsite profile: two "success" records,
        # zero receipts, memory.db untouched).
        return OperationResult(
            output_text=f"⛔ Learning storage failed: {exc}",
            status="error",
            error_detail=str(exc),
        )

    stored_any = (
        applied.created_ids or applied.updated_ids or applied.retired_ids or applied.stored_unparsed
    )
    if not stored_any:
        return OperationResult(
            output_text="Nothing new to store: the durable knowledge is already in memory.",
            extra={"recorded": False, "created_ids": [], "updated_ids": [], "retired_ids": []},
        )

    actions: list[str] = []
    if applied.created_ids or applied.stored_unparsed:
        created_count = len(applied.created_ids) + max(
            0, applied.stored_unparsed - len(applied.updated_ids)
        )
        actions.append(f"created {created_count}")
    if applied.updated_ids:
        actions.append(f"updated {len(applied.updated_ids)}")
    if applied.skipped:
        actions.append(f"skipped {applied.skipped} duplicate(s)")

    # Thread → memory linkage: surface the minted record ids so the host can
    # stamp them onto the thread entry's provenance. The lisa-memory backend
    # replies "Memory stored with ID <n>" per record.
    minted_ids = applied.created_ids + applied.updated_ids
    extra: dict[str, object] = {
        "recorded": True,
        "created_ids": applied.created_ids,
        "updated_ids": applied.updated_ids,
        "retired_ids": applied.retired_ids,
        "skipped": applied.skipped,
    }
    if minted_ids:
        extra["memory_ids"] = minted_ids
    return OperationResult(
        output_text=f"✅ Learning stored: {', '.join(actions)}.",
        extra=extra,
    )
