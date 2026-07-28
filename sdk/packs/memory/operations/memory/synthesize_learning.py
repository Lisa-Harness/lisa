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
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext
from kernel.operations.structured_output import (
    json_only_retry_messages,
    parse_structured_payload,
)
from sdk.content import require_authored_text

logger = logging.getLogger(__name__)

# Authored-text path — loaded via ctx.read_authored_text_fn (SDK loader).
_SYSTEM_PROMPT_PATH = "operations/memory/prompts/synthesize_learning_system.md"


def _load_system_prompt(ctx: OperationExecutionContext) -> str:
    """Load the distillation system prompt via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """

    return require_authored_text(
        ctx.read_authored_text_fn,
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
        "long_term",
        description="Memory tier: 'long_term', 'daily', or 'weekly'.",
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
    memory_tier = params.get("memory_tier", "long_term")
    # Keep learning records under one canonical operation id so recall queries stay stable.
    memory_operation_id = "save_learning"

    # Step 1: Distill with the LLM. Distillation is required (not optional) —
    # a missing completion function, unresolved model, or authored prompt
    # raises rather than silently persisting the raw conversation text.
    if ctx.complete_fn is None or not callable(ctx.resolve_model_fn):
        raise RuntimeError(
            "synthesize_and_store_learning requires complete_fn and resolve_model_fn"
        )
    model_id = ctx.preferred_model_id
    model_def = ctx.resolve_model_fn(model_id)
    if model_def is None:
        model_def = ctx.resolve_model_fn("default")
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
    # 512 tokens leaves headroom for the JSON envelope + 3 durable facts; 300
    # truncated the array mid-fact on 2026-07-18, producing invalid JSON that
    # the _looks_like_json guard correctly refused to fall back to prose.
    distill_max_tokens = 512
    distilled_text: str | None = None
    last_exc: Exception | None = None
    for attempt_idx, attempt_messages in enumerate(
        (first_messages, json_only_retry_messages(first_messages, DistilledLearning))
    ):
        try:
            raw = await ctx.complete_fn(
                messages=attempt_messages,
                model_def=model_def,
                max_tokens=distill_max_tokens,
                temperature=0.1,
                # First attempt asks for native structured output; the retry
                # clears response_format so strict hosts don't re-reject and
                # the model emits free-form JSON per the appended instruction.
                response_format=DistilledLearning if attempt_idx == 0 else None,
            )
            distilled = _parse_distilled_learning(raw.content)
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

    if not callable(ctx.execute_tool_fn):
        return OperationResult(
            output_text="Learning storage unavailable: tool execution not injected."
        )

    tool_result = await ctx.execute_tool_fn(
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
            },
        ),
    )
    if getattr(tool_result, "error", None):
        return OperationResult(output_text=f"⛔ Learning storage failed: {tool_result.error}")

    preview = distilled_text[:80]
    return OperationResult(
        output_text=f"✅ Learning stored: '{preview}{'...' if len(distilled_text) > 80 else ''}'",
        extra={"recorded": True, "distilled": distilled_text},
    )
