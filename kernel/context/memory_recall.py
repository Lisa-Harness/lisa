"""Memory recall ranking and synthesis for vague or explicit recall requests.

Called by the default chat intent router and the reusable `recall_memory` operation.
This module fuses recent-memory signals, lexical overlap, and optional semantic
embeddings before asking a model to synthesize the best answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from kernel.models import (
    MemoryRecallAnswer,
    MemoryRecallContext,
    MemoryRecallEvidence,
    MemoryRecord,
    MemoryScope,
)
from pydantic import BaseModel, Field

import app.model_policy as model_policy
import app.providers as providers
from app.config import settings
from app.content_registry import read_authored_text
from app.memory_store import get_memory_store

__all__ = [
    "MemoryRecallAnswer",
    "MemoryRecallContext",
    "MemoryRecallEvidence",
    "model_policy",
    "providers",
    "settings",
]

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "anything",
    "did",
    "do",
    "ever",
    "for",
    "have",
    "i",
    "it",
    "like",
    "me",
    "my",
    "of",
    "remember",
    "said",
    "something",
    "that",
    "the",
    "thing",
    "to",
    "what",
    "was",
    "we",
    "were",
    "you",
}
_VAGUE_RECALL_PHRASES = (
    "do you remember",
    "what was that",
    "what were we saying",
    "that thing",
    "something about",
    "sort of remember",
)


class _MemoryRecallSynthesis(BaseModel):
    """Structured synthesis output returned by the recall answering model."""

    found: bool = True
    answer: str = Field(default="")
    uncertainty_note: str = Field(default="")


@lru_cache(maxsize=1)
def _load_synthesis_system_prompt() -> str:
    """Load the authored system prompt for memory-recall synthesis."""
    return read_authored_text("prompts/memory_recall/synthesis_system.md").strip()


@lru_cache(maxsize=1)
def _load_synthesis_user_prompt_template() -> str:
    """Load the authored user-prompt template for memory-recall synthesis."""
    return read_authored_text("prompts/memory_recall/synthesis_user.md").strip()


def _build_synthesis_user_prompt(
    context: MemoryRecallContext,
    *,
    draft_answer: str = "",
) -> str:
    """Render the authored synthesis user prompt with bounded dynamic evidence."""
    draft_answer_block = ""
    if draft_answer.strip():
        draft_answer_block = f"\n\nCheap draft answer:\n{draft_answer.strip()}"

    return _load_synthesis_user_prompt_template().format(
        recall_query=context.query,
        ranked_evidence=_format_evidence_payload(context),
        draft_answer_block=draft_answer_block,
    )


def _tokenize(text: str) -> set[str]:
    """Tokenize text into normalized content words for lexical matching."""
    tokens = {match.group(0).lower() for match in _TOKEN_RE.finditer(text.lower())}
    return {token for token in tokens if token and token not in _STOPWORDS}


def _lexical_score(query_tokens: set[str], content: str, raw_query: str) -> float:
    """Score token overlap between the query and one memory content string."""
    if not query_tokens:
        return 0.0

    content_lower = content.lower()
    content_tokens = _tokenize(content)
    overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)

    normalized_query = raw_query.strip().lower()
    if normalized_query and normalized_query in content_lower:
        overlap = max(overlap, 0.9)

    return min(1.0, overlap)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(l_value * r_value for l_value, r_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot_product / (left_norm * right_norm)))


def _temporal_decay(created_at: str) -> float:
    """Return a [0, 1] freshness multiplier that decays with age.

    Uses an exponential half-life model: a memory at the half-life age
    scores 0.5, a memory twice the half-life scores 0.25, and so on. The
    result is clamped to ``[memory_decay_floor, 1.0]`` so a very old
    memory never collapses the ranker to zero. The model is a function
    of ``created_at`` only (not ``last_accessed_at``) so a memory that
    is "old but important" still has a chance to surface; access-driven
    promotion is handled in the consolidation job, not the ranker.
    """
    if not settings.memory_decay_enabled:
        return 1.0
    half_life_days = float(settings.memory_decay_half_life_days or 0.0)
    if half_life_days <= 0:
        return 1.0
    try:
        created_dt = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return 1.0
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=UTC)
    age_seconds = max(0.0, (datetime.now(UTC) - created_dt).total_seconds())
    age_days = age_seconds / 86400.0
    raw = math.pow(0.5, age_days / half_life_days)
    floor_value = float(settings.memory_decay_floor or 0.0)
    return float(max(floor_value, min(1.0, raw)))


def _query_is_vague(query: str) -> bool:
    """Return True when the recall request is underspecified or fuzzy."""
    query_lower = query.lower()
    token_count = len(_tokenize(query))
    return token_count <= 3 or any(phrase in query_lower for phrase in _VAGUE_RECALL_PHRASES)


def _ensure_candidate(
    candidates: dict[int, MemoryRecallEvidence],
    record: MemoryRecord,
) -> MemoryRecallEvidence:
    """Return one mutable candidate object for the given memory record."""
    existing = candidates.get(record.id)
    if existing is not None:
        return existing

    candidate = MemoryRecallEvidence(
        memory_id=record.id,
        created_at=record.created_at,
        content=record.content,
        memory_scope=record.memory_scope,
        surface_id=record.surface_id,
        source_user_slug=record.source_user_slug,
        operation_id=record.operation_id,
        memory_tier=record.memory_tier,
        salience=record.salience,
    )
    candidates[record.id] = candidate
    return candidate


def _estimate_confidence(candidates: list[MemoryRecallEvidence]) -> float:
    """Estimate recall confidence from top-candidate strength and separation."""
    if not candidates:
        return 0.0

    top_score = candidates[0].combined_score
    second_score = candidates[1].combined_score if len(candidates) > 1 else 0.0
    separation = max(0.0, top_score - second_score)
    confidence = min(1.0, (top_score * 0.75) + (separation * 0.5))
    return round(confidence, 3)


def _resolve_model(model_key: str, fallback_key: str) -> providers.ModelDef:
    """Resolve one chat model, falling back to a known-safe default."""
    model_def = providers.resolve_model(model_key)
    if model_def is not None:
        return model_def

    fallback = providers.resolve_model(fallback_key)
    if fallback is not None:
        return fallback
    return providers.MODEL_REGISTRY["gpt-4o"]


def _format_evidence_payload(context: MemoryRecallContext) -> str:
    """Render compact ranked evidence as JSON for the synthesis model."""
    payload = {
        "query": context.query,
        "confidence": context.confidence,
        "deep_hint": context.deep_hint,
        "used_semantic_search": context.used_semantic_search,
        "candidates": [candidate.model_dump(mode="json") for candidate in context.candidates],
    }
    return json.dumps(payload, indent=2)


async def build_memory_recall_context(
    *,
    query: str,
    user_slug: str,
    workspace_id: str,
    memory_scope: MemoryScope = "personal",
    surface_id: str = "",
    limit: int | None = None,
    embed_fn: Any = providers.embed_texts,
) -> MemoryRecallContext:
    """Build ranked recall context from recent, lexical, and semantic signals."""
    memory_store = get_memory_store()
    candidate_pool_size = max(1, settings.memory_recall_candidate_pool_size)
    records = await memory_store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=candidate_pool_size,
    )
    if not records:
        return MemoryRecallContext(
            query=query,
            memory_scope=memory_scope,
            surface_id=surface_id,
        )

    candidates_by_id: dict[int, MemoryRecallEvidence] = {}
    recent_limit = min(len(records), max(1, settings.memory_recall_recent_limit))
    for index, record in enumerate(records[:recent_limit]):
        candidate = _ensure_candidate(candidates_by_id, record)
        candidate.recency_score = max(
            candidate.recency_score,
            round(1.0 - (index / max(recent_limit, 1)), 3),
        )
        if "recent memory" not in candidate.match_reasons:
            candidate.match_reasons.append("recent memory")

    query_tokens = _tokenize(query)
    lexical_ranked: list[tuple[float, MemoryRecord]] = []
    for record in records:
        lexical_score = _lexical_score(query_tokens, record.content, query)
        if lexical_score <= 0.0:
            continue
        lexical_ranked.append((lexical_score, record))

    lexical_ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
    for lexical_score, record in lexical_ranked[: max(1, settings.memory_recall_lexical_limit)]:
        candidate = _ensure_candidate(candidates_by_id, record)
        candidate.lexical_score = max(candidate.lexical_score, round(lexical_score, 3))
        if "lexical overlap" not in candidate.match_reasons:
            candidate.match_reasons.append("lexical overlap")

    used_semantic_search = False
    semantic_backfill_count = 0
    if settings.memory_embedding_enabled and settings.memory_embedding_model.strip():
        memory_ids = [record.id for record in records]
        existing_embeddings = await memory_store.get_memory_embeddings(memory_ids)
        indexed_embeddings = await memory_store.ensure_embeddings_for_records(records)
        semantic_backfill_count = max(0, len(indexed_embeddings) - len(existing_embeddings))

        if indexed_embeddings:
            try:
                embedding_result = await embed_fn(
                    [query],
                    provider=settings.memory_embedding_provider.strip() or "azure_openai",
                    model_id=settings.memory_embedding_model.strip(),
                    dimensions=settings.memory_embedding_dimensions,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("query embedding generation failed: %s", exc)
            else:
                query_vector = embedding_result.embeddings[0] if embedding_result.embeddings else []
                if query_vector:
                    semantic_ranked: list[tuple[float, MemoryRecord]] = []
                    for record in records:
                        embedding = indexed_embeddings.get(record.id)
                        if embedding is None:
                            continue
                        similarity = _cosine_similarity(query_vector, embedding.vector)
                        if similarity <= 0.0:
                            continue
                        semantic_ranked.append((similarity, record))

                    semantic_ranked.sort(
                        key=lambda item: (item[0], item[1].id),
                        reverse=True,
                    )
                    semantic_limit = max(1, settings.memory_recall_semantic_limit)
                    for similarity, record in semantic_ranked[:semantic_limit]:
                        candidate = _ensure_candidate(candidates_by_id, record)
                        candidate.semantic_score = max(
                            candidate.semantic_score,
                            round(similarity, 3),
                        )
                        if "semantic similarity" not in candidate.match_reasons:
                            candidate.match_reasons.append("semantic similarity")
                    used_semantic_search = bool(semantic_ranked)

    candidates = list(candidates_by_id.values())
    semantic_available = used_semantic_search and any(
        candidate.semantic_score > 0 for candidate in candidates
    )
    decay_enabled = bool(getattr(settings, "memory_decay_enabled", True))
    for candidate in candidates:
        salience_term = max(0.0, min(candidate.salience, 1.0)) * 0.1
        decay_term = (
            _temporal_decay(candidate.created_at)
            if decay_enabled
            else 1.0
        )
        if semantic_available:
            candidate.combined_score = round(
                (candidate.semantic_score * 0.55)
                + (candidate.lexical_score * 0.20)
                + (candidate.recency_score * 0.05)
                + (salience_term)
                + (decay_term * 0.10),
                3,
            )
        else:
            candidate.combined_score = round(
                (candidate.lexical_score * 0.50)
                + (candidate.recency_score * 0.15)
                + (salience_term * 2.0)  # scale back up; no semantic slot
                + (decay_term * 0.15),
                3,
            )

    candidates.sort(key=lambda item: (item.combined_score, item.created_at), reverse=True)
    final_limit = max(1, limit or settings.memory_recall_final_limit)
    candidates = candidates[:final_limit]
    confidence = _estimate_confidence(candidates)
    top_score = candidates[0].combined_score if candidates else 0.0
    ambiguous_top_band = [
        candidate
        for candidate in candidates
        if candidate.combined_score >= max(0.15, top_score - 0.12)
    ]
    deep_hint = (
        _query_is_vague(query)
        or confidence < settings.memory_recall_auto_deep_threshold
        or len(ambiguous_top_band) >= 3
    )

    return MemoryRecallContext(
        query=query,
        memory_scope=memory_scope,
        surface_id=surface_id,
        confidence=confidence,
        deep_hint=deep_hint,
        used_semantic_search=used_semantic_search,
        semantic_backfill_count=semantic_backfill_count,
        candidate_pool_size=len(records),
        candidates=candidates,
    )


async def _synthesize_answer(
    *,
    context: MemoryRecallContext,
    complete_fn: Any,
    model_def: providers.ModelDef,
    draft_answer: str = "",
) -> _MemoryRecallSynthesis:
    """Use one model pass to turn ranked evidence into a recall answer."""
    system_prompt = _load_synthesis_system_prompt()
    user_prompt = _build_synthesis_user_prompt(context, draft_answer=draft_answer)

    raw = await complete_fn(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_def=model_def,
        max_tokens=1200 if model_def.reasoning_model else 900,
        temperature=0.1,
        response_format=_MemoryRecallSynthesis,
    )
    return _MemoryRecallSynthesis.model_validate_json(raw.content)


def _fallback_answer(context: MemoryRecallContext) -> MemoryRecallAnswer:
    """Return a deterministic answer when no model synthesis function is available."""
    if not context.candidates:
        return MemoryRecallAnswer(
            query=context.query,
            found=False,
            answer="I could not find anything relevant in memory for that.",
            confidence=0.0,
        )

    lines = ["Here are the closest memories I found:"]
    for candidate in context.candidates[:3]:
        lines.append(f"- {candidate.created_at[:16]}: {candidate.content}")
    if context.deep_hint:
        lines.append("The match is fuzzy enough that a deeper pass could help.")

    return MemoryRecallAnswer(
        query=context.query,
        found=True,
        answer="\n".join(lines),
        confidence=context.confidence,
        used_semantic_search=context.used_semantic_search,
        auto_deep_used=False,
        evidence=context.candidates,
    )


async def answer_memory_recall(
    *,
    query: str,
    user_slug: str,
    workspace_id: str,
    complete_fn: Any,
    memory_scope: MemoryScope = "personal",
    surface_id: str = "",
    force_deep: bool = False,
    limit: int | None = None,
    embed_fn: Any = providers.embed_texts,
) -> MemoryRecallAnswer:
    """Return a synthesized recall answer for one fuzzy or explicit memory query."""
    context = await build_memory_recall_context(
        query=query,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit,
        embed_fn=embed_fn,
    )
    if not context.candidates:
        return MemoryRecallAnswer(
            query=query,
            found=False,
            answer="I could not find anything relevant in memory for that.",
            confidence=0.0,
            used_semantic_search=context.used_semantic_search,
        )

    if complete_fn is None:
        return _fallback_answer(context)

    cheap_model = _resolve_model(settings.memory_recall_cheap_model, "gpt-4o")
    try:
        synthesis = await _synthesize_answer(
            context=context,
            complete_fn=complete_fn,
            model_def=cheap_model,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("cheap recall synthesis failed: %s", exc)
        return _fallback_answer(context)

    auto_deep_used = False
    final_synthesis = synthesis
    if force_deep or context.deep_hint:
        escalation_model = model_policy.resolve_model_role("deep_reasoning")
        if (
            escalation_model.provider != cheap_model.provider
            or escalation_model.model_id != cheap_model.model_id
        ):
            try:
                final_synthesis = await _synthesize_answer(
                    context=context,
                    complete_fn=complete_fn,
                    model_def=escalation_model,
                    draft_answer=synthesis.answer,
                )
                auto_deep_used = True
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("deep recall synthesis failed: %s", exc)

    # Phase 2 uplift: update access_count + last_accessed_at for recalled memories
    _recalled_ids = [c.memory_id for c in context.candidates if c.memory_id]
    if _recalled_ids:
        try:
            await asyncio.to_thread(
                get_memory_store()._update_memory_access,
                _recalled_ids,  # noqa: SLF001
            )
        except Exception:  # pylint: disable=broad-except
            pass

    return MemoryRecallAnswer(
        query=query,
        found=final_synthesis.found,
        answer=final_synthesis.answer,
        confidence=context.confidence,
        used_semantic_search=context.used_semantic_search,
        auto_deep_used=auto_deep_used,
        uncertainty_note=final_synthesis.uncertainty_note,
        evidence=context.candidates,
    )


_RECALL_INTENT_PHRASES = (
    "remember",
    "what did i say",
    "what did we discuss",
    "last time",
    "do you recall",
    "remind me",
    "what was that",
    "what were we",
    "i mentioned",
    "i told you",
    "i said",
    "you noted",
    "you remember",
    "we talked about",
    "what did you note",
    "show me my notes",
    "pull up my",
    "find my note",
)


def _classify_recall_intent(query: str) -> str:
    """Return 'recall' when the query looks like a memory retrieval request.

    Returns 'recall' or 'other'. Used to gate Phase 3 auto-recall injection.
    """
    query_lower = query.lower().strip()
    return "recall" if any(phrase in query_lower for phrase in _RECALL_INTENT_PHRASES) else "other"


async def build_auto_recall_snippet(
    *,
    query: str,
    user_slug: str,
    workspace_id: str,
    memory_scope: MemoryScope = "personal",
    surface_id: str = "",
    budget_tokens: int = 600,
) -> str:
    """Return a compact recall context block for auto-injection into the chat prompt.

    Builds recall context from top-scoring candidates and formats them as a
    brief Markdown block. Truncates to *budget_tokens* approximate tokens
    (1 token ≈ 4 chars) to stay within the prompt budget.
    """
    ctx = await build_memory_recall_context(
        query=query,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
    )
    if not ctx.candidates:
        return ""

    budget_chars = budget_tokens * 4
    lines: list[str] = ["## Relevant Memory Notes"]
    used = len(lines[0])
    top = sorted(
        ctx.candidates,
        key=lambda c: c.recency_score + c.lexical_score,
        reverse=True,
    )
    for candidate in top:
        line = f"- {candidate.content}"
        if used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)
