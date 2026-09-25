"""Synthesize and store learning operation.

Distills raw conversation content into 1-3 durable facts using the operation's
preferred model, then persists those facts to the memory store.

This replaces the old pattern of recording raw conversation text verbatim.
The LLM distillation step ensures each memory record is compact, self-contained,
and useful in future recall without needing the original conversation context.
"""

from __future__ import annotations

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
_DISTILL_MAX_TOKENS = 512


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


class DistilledLearning(BaseModel):
    """Strict bounded output from the distillation LLM call."""

    facts: list[str] = Field(
        ...,
        description="1 to 3 concise, durable facts distilled from the raw content.",
        min_length=1,
        max_length=3,
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
                f"parsing. Raw output: {normalized[:200]!r}"
            ) from direct_exc
        bullet_pattern = re.compile(r"^(?:[-*\u2022]|\d+[\.)\]])\s+(?P<fact>.+)$")
        facts: list[str] = []
        for line in normalized.splitlines():
            match = bullet_pattern.match(line.strip())
            if not match:
                continue
            fact = match.group("fact").strip()
            if fact:
                facts.append(fact)
            if len(facts) == 3:
                break
        if facts:
            return DistilledLearning(facts=facts)

        # Fallback for providers that return plain prose instead of JSON or bullets.
        sentence_facts: list[str] = []
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", normalized):
            candidate = chunk.strip().strip('"').strip("'")
            if not candidate:
                continue
            sentence_facts.append(candidate)
            if len(sentence_facts) == 3:
                break
        if sentence_facts:
            return DistilledLearning(facts=sentence_facts)

        raise direct_exc


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
    if stripped.startswith("```"):
        return True
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    # Inner brace after any leading whitespace/prose — a strong JSON signal.
    if "{" in stripped and "}" in stripped:
        return True
    return False


def _default_learning_content(ctx: OperationExecutionContext) -> str:
    """Build learning input from the completed exchange when params omit content."""
    query = (ctx.query or "").strip()
    response = (ctx.response or "").strip()
    if query and response:
        return f"User asked:\n{query}\n\nLisa answered:\n{response}"
    return query or response


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Distill raw content into durable facts, then persist to memory."""
    params = ctx.params or {}
    raw_content = str(params.get("content", "") or "").strip() or _default_learning_content(ctx)
    if not raw_content:
        return OperationResult(output_text="⛔ No content provided to store.")

    salience = float(params.get("salience", 0.5))
    # Default to 'daily' so model-called captures that omit the tier land as
    # short-lived workflow state. The auto-capture path always supplies an
    # explicit, salience-graded tier, so this default only binds the
    # model-invoked path (see Parameters.memory_tier).
    memory_tier = params.get("memory_tier", "daily")
    # Keep learning records under one canonical operation id so recall queries stay stable.
    memory_operation_id = "save_learning"

    # Step 1: Distill with the LLM. Distillation is required (not optional) —
    # a missing completion function, unresolved model, or authored prompt
    # raises rather than silently persisting the raw conversation text.
    if ctx.host is None:
        raise RuntimeError("synthesize_and_store_learning requires the OperationHost seam")
    model_id = ctx.preferred_model_id
    model_def = ctx.host.resolve_model(model_id)
    if model_def is None:
        model_def = ctx.host.resolve_model("default")
    if model_def is None:
        raise RuntimeError("synthesize_and_store_learning could not resolve a model")

    system_prompt = _load_system_prompt(ctx)
    # The distillation loop. Local OpenAI-compatible models (Ollama) frequently
    # ignore ``response_format`` and return empty content, or emit JSON that
    # truncates mid-array (``max_tokens`` too small for the envelope + 3 facts).
    # Both produce unparseable output. Per the structured-output contract
    # (/memories/repo/structured-output-contract.md), retry once with an
    # explicit JSON-only instruction before surfacing the failure. The retry
    # covers ALL first-attempt failures — empty output, JSON-truncation
    # (RuntimeError from the _looks_like_json guard), and ValidationError.
    first_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_content},
    ]
    # Token budget for the distillation call, resolved from ``ctx.pack_config``
    # (key ``memory_distill_max_tokens`` under
    # ``resources.packs.configuration.lisa-pack-memory``) with a module
    # fallback of 512. 512 leaves headroom for the JSON envelope + 3 durable
    # facts; 300 truncated the array mid-fact on 2026-07-18, producing invalid
    # JSON that the _looks_like_json guard correctly refused to fall back to
    # prose.
    distill_max_tokens = _resolve_settings(ctx).distill_max_tokens
    distilled_text: str | None = None
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
            distilled_text = " | ".join(distilled.facts)
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if attempt_idx == 0:
                continue
            break
    if distilled_text is None:
        raise RuntimeError(
            f"synthesize_and_store_learning distillation failed: {last_exc}"
        ) from last_exc

    # Step 2: Persist via the memory.add MCP tool (default memory backend).
    content_json: dict[str, object] = {"query": ctx.query, "model": ctx.model_id}
    if "had_briefing" in ctx.extra:
        content_json["had_chase_briefing"] = ctx.extra["had_briefing"]
    # Memory → thread linkage: when the host stamps the global turn index
    # into the capture context, carry it into the stored record so recall
    # can point back to the exact turn the learning came from.
    thread_turn = ctx.extra.get("thread_turn")
    if isinstance(thread_turn, int):
        content_json["thread_turn"] = thread_turn

    if ctx.host is None:
        return OperationResult(
            output_text="Learning storage unavailable: tool execution not injected."
        )

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
            parameters={
                "content": distilled_text,
                "user_slug": ctx.user_slug,
                "workspace_id": ctx.workspace_id,
                "surface_id": ctx.surface_id,
                "memory_scope": ctx.memory_scope,
                "memory_tier": memory_tier,
                "salience": salience,
                "operation_id": memory_operation_id,
                "confidence": 1.0,
                "content_json": content_json,
            },
        ),
    )
    if getattr(tool_result, "error", None):
        return OperationResult(output_text=f"⛔ Learning storage failed: {tool_result.error}")

    # Thread → memory linkage: surface the minted record id so the host can
    # stamp it onto the thread entry's provenance. The lisa-memory backend
    # replies "Memory stored with ID <n>".
    memory_id: int | None = None
    add_text = getattr(tool_result, "output_text", "") or ""
    if "ID " in add_text:
        try:
            memory_id = int(add_text.split("ID ", 1)[1].rstrip("."))
        except (ValueError, IndexError):
            memory_id = None

    preview = distilled_text[:80]
    extra: dict[str, object] = {"recorded": True, "distilled": distilled_text}
    if memory_id is not None:
        extra["memory_ids"] = [memory_id]
    return OperationResult(
        output_text=f"✅ Learning stored: '{preview}{'...' if len(distilled_text) > 80 else ''}'",
        extra=extra,
    )
