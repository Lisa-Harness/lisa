"""Shared memory-recall ranking logic for the learning pack."""

from __future__ import annotations

import json
import re

from kernel.models import MemoryRecallAnswer, MemoryRecallEvidence, MemoryRecord

_RANKED_MIN_TOKEN_LEN = 3
#: Minimum lexical relevance a record must clear to lead the ranked block
#: (2026-10-14 self-review F1). Records below the floor still surface — the
#: backend vouched for them — but only after every record that clears it,
#: regardless of salience. Salience breaks ties among relevant records only.
_RELEVANCE_FLOOR = 0.2


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
    deep: bool = False,
) -> MemoryRecallAnswer:
    """Rank records by relevance first, then salience, and build the recall answer.

    The backend (``memory.search``) already filtered for semantic relevance
    before we get here — the records it returned are candidates, not raw
    noise. But the backend's vouch is not a ranking: a high-salience record
    with zero query overlap must not lead the block over a lexically
    relevant one (the 2026-10-14 self-review F1 finding — live, a harness
    self-assessment query surfaced an off-topic product-strategy memory
    first). Ranking is therefore relevance-first:

    - A record clearing ``_RELEVANCE_FLOOR`` lexical overlap leads the block;
      salience breaks ties only among relevant records.
    - Records below the floor (including zero-overlap ones) still surface —
      semantically-relevant but lexically-disjoint memories (the common case
      when the model paraphrases a stored fact) must not be dropped — but
      they rank after every relevant record, by salience.
    - ``deep`` no longer flips salience above relevance; it only tightens
      the tie-break weight salience carries among relevant records.
    """
    if not records:
        return MemoryRecallAnswer(
            query=query,
            found=False,
            answer=f"No memory matches found for: {query}",
        )

    query_tokens = _tokenize(query)
    ranked: list[tuple[float, bool, MemoryRecord]] = []
    for record in records:
        haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
        overlap = len(query_tokens.intersection(_tokenize(haystack)))
        lexical_score = overlap / max(len(query_tokens), 1) if query_tokens else 0.0
        # Relevance-first ranking (2026-10-14 self-review F1): a record must
        # clear the relevance floor to lead the block; salience only breaks
        # ties among relevant records. Below-floor records still surface
        # (the backend vouched for them) but rank after every relevant one.
        # ``deep`` tightens the salience tiebreak weight; it never lets
        # salience outrank relevance.
        salience = max(0.0, min(1.0, record.salience))
        relevant = lexical_score >= _RELEVANCE_FLOOR
        tiebreak_weight = 0.5 if deep else 0.2
        if relevant:
            combined = min(1.0, lexical_score + salience * tiebreak_weight)
        else:
            combined = min(1.0, salience * tiebreak_weight)
        ranked.append((combined, relevant, record))

    # Relevant records lead; ties break by combined score, then record id.
    ranked.sort(key=lambda item: (item[1], item[0], item[2].id), reverse=True)
    top = ranked[: max(1, min(int(limit or 1), 8))]
    if not top:
        return MemoryRecallAnswer(
            query=query,
            found=False,
            answer=f"No memory matches found for: {query}",
        )

    evidence: list[MemoryRecallEvidence] = []
    for score, relevant, record in top:
        reasons = ["lexical overlap"] if relevant else ["salience match"]
        # Memory → thread provenance: surface the turn index the learning
        # was distilled from (when the record carries one) so recall output
        # renders a verifiable pointer instead of an unanchored claim.
        source_turn = record.content_json.get("thread_turn")
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
                source_turn=source_turn if isinstance(source_turn, int) else None,
                match_reasons=reasons,
            )
        )

    summary = evidence[0].content
    confidence = evidence[0].combined_score if evidence else 0.0
    if query_tokens and confidence <= 0.0:
        return MemoryRecallAnswer(
            query=query,
            found=False,
            answer=f"No memory matches found for: {query}",
        )

    # Render the answer as a pointer when the top hit carries a thread-turn
    # provenance: "learned around turn N" is verifiable via thread_read_range.
    lead_turn = evidence[0].source_turn
    if lead_turn is not None:
        summary = f"{summary} (learned around turn {lead_turn})"

    return MemoryRecallAnswer(
        query=query,
        found=True,
        answer=summary,
        confidence=round(confidence, 3),
        used_semantic_search=False,
        auto_deep_used=False,
        evidence=evidence,
    )
