"""Shared memory-recall ranking logic for the learning pack."""

from __future__ import annotations

import json
import re

from kernel.models import MemoryRecallAnswer, MemoryRecallEvidence, MemoryRecord

_RANKED_MIN_TOKEN_LEN = 3


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase tokens of at least 3 characters."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= _RANKED_MIN_TOKEN_LEN
    }


def build_memory_recall_answer(
    *,
    query: str,
    records: list[MemoryRecord],
    limit: int,
) -> MemoryRecallAnswer:
    """Rank records by lexical overlap plus salience and build the recall answer."""
    if not records:
        return MemoryRecallAnswer(
            query=query,
            found=False,
            answer=f"No memory matches found for: {query}",
        )

    query_tokens = _tokenize(query)
    ranked: list[tuple[float, MemoryRecord]] = []
    for record in records:
        haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
        overlap = len(query_tokens.intersection(_tokenize(haystack)))
        if overlap <= 0 and query_tokens:
            continue
        lexical_score = overlap / max(len(query_tokens), 1) if query_tokens else 1.0
        combined = min(1.0, lexical_score + max(0.0, record.salience) * 0.2)
        ranked.append((combined, record))

    ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
    top = ranked[: max(1, min(int(limit or 1), 8))]
    if not top:
        return MemoryRecallAnswer(
            query=query,
            found=False,
            answer=f"No memory matches found for: {query}",
        )

    evidence: list[MemoryRecallEvidence] = []
    for score, record in top:
        evidence.append(
            MemoryRecallEvidence(
                memory_id=record.id,
                created_at=record.created_at,
                content=record.content,
                memory_scope=record.memory_scope,
                surface_id=record.surface_id,
                source_user_slug=record.source_user_slug,
                operation_id=record.operation_id,
                memory_tier=record.memory_tier,
                salience=record.salience,
                combined_score=round(score, 3),
                lexical_score=round(score, 3),
                recency_score=1.0,
                match_reasons=["lexical overlap"],
            )
        )

    summary = evidence[0].content
    confidence = evidence[0].combined_score if evidence else 0.0
    return MemoryRecallAnswer(
        query=query,
        found=True,
        answer=summary,
        confidence=round(confidence, 3),
        used_semantic_search=False,
        auto_deep_used=False,
        evidence=evidence,
    )
