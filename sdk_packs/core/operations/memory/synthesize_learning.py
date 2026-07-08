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

from kernel.models import MemoryWriteRequest, OperationExecutionContext, OperationResult
from kernel.operations.structured_output import parse_structured_payload

logger = logging.getLogger(__name__)


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

    # Step 1: Distill with LLM if complete_fn is available; otherwise store raw
    distilled_text = raw_content
    if ctx.complete_fn and callable(ctx.resolve_model_fn):
        model_id = ctx.preferred_model_id
        model_def = ctx.resolve_model_fn(model_id)
        if model_def is None:
            model_def = ctx.resolve_model_fn("deep")

        if model_def is not None:
            try:
                raw = await ctx.complete_fn(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a memory curator. Extract 1-3 concise, self-contained, "
                                "durable facts from the input. Each fact must make sense without "
                                "the original conversation context. Be specific and operational. "
                                "Return the distilled facts as a JSON object with a facts array."
                            ),
                        },
                        {"role": "user", "content": raw_content},
                    ],
                    model_def=model_def,
                    max_tokens=300,
                    temperature=0.1,
                    response_format=DistilledLearning,
                )
                distilled = _parse_distilled_learning(raw.content)
                distilled_text = " | ".join(distilled.facts)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "synthesize_and_store_learning distillation failed, storing raw: %s", exc
                )
                distilled_text = raw_content

    # Step 2: Persist to memory store
    content_json: dict[str, object] = {"query": ctx.query, "model": ctx.model_id}
    if "had_briefing" in ctx.extra:
        content_json["had_chase_briefing"] = ctx.extra["had_briefing"]

    record_memory_fn = getattr(ctx.memory_store, "record_memory", None)
    if not callable(record_memory_fn):
        return OperationResult(
            output_text="Learning storage unavailable: memory store not injected."
        )

    await record_memory_fn(
        MemoryWriteRequest(
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            memory_scope=ctx.memory_scope,
            surface_id=ctx.surface_id,
            source_user_id=ctx.user_id,
            source_user_slug=ctx.user_slug,
            agent_id=f"{memory_operation_id}-agent",
            operation_id=memory_operation_id,
            instruction_id=f"{memory_operation_id}.v1",
            memory_tier=memory_tier,
            content=distilled_text,
            content_json=content_json,
            salience=salience,
        )
    )

    preview = distilled_text[:80]
    return OperationResult(
        output_text=f"✅ Learning stored: '{preview}{'...' if len(distilled_text) > 80 else ''}'",
        extra={"recorded": True, "distilled": distilled_text},
    )
