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
from typing import cast

from pydantic import BaseModel, Field, ValidationError

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext
from kernel.operations.structured_output import parse_structured_payload

logger = logging.getLogger(__name__)

# Authored-text path — loaded via ctx.read_authored_text_fn (SDK loader).
_SYSTEM_PROMPT_PATH = "operations/memory/prompts/synthesize_learning_system.md"


def _load_system_prompt(ctx: OperationExecutionContext) -> str:
    """Load the distillation system prompt via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """

    if not callable(ctx.read_authored_text_fn):
        raise RuntimeError(
            f"read_authored_text_fn is not wired; cannot load authored content at {_SYSTEM_PROMPT_PATH}"
        )
    try:
        loaded = str(ctx.read_authored_text_fn(_SYSTEM_PROMPT_PATH) or "").strip()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"read_authored_text_fn failed for {_SYSTEM_PROMPT_PATH}: {exc}") from exc
    if not loaded:
        raise RuntimeError(f"Authored text at {_SYSTEM_PROMPT_PATH} is empty or missing")
    return loaded


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
        model_def = ctx.resolve_model_fn("deep")
    if model_def is None:
        raise RuntimeError("synthesize_and_store_learning could not resolve a model")

    system_prompt = _load_system_prompt(ctx)
    try:
        raw = await ctx.complete_fn(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_content},
            ],
            model_def=model_def,
            max_tokens=300,
            temperature=0.1,
            response_format=DistilledLearning,
        )
        distilled = _parse_distilled_learning(raw.content)
        distilled_text = " | ".join(distilled.facts)
    except RuntimeError:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            f"synthesize_and_store_learning distillation failed: {exc}"
        ) from exc

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
        return OperationResult(
            output_text=f"⛔ Learning storage failed: {tool_result.error}"
        )

    preview = distilled_text[:80]
    return OperationResult(
        output_text=f"✅ Learning stored: '{preview}{'...' if len(distilled_text) > 80 else ''}'",
        extra={"recorded": True, "distilled": distilled_text},
    )
