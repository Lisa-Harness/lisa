"""Shared memory-recall ranking logic for the memory pack.

Relevance-first ranking over the candidates the backend surfaced. The
backend (``memory.search``) already filtered for semantic/lexical relevance
before we get here — but its vouch is not a ranking. A high-salience record
with zero query overlap must not lead the block over a lexically relevant
one. Ranking is therefore:

- Records clearing ``_RELEVANCE_FLOOR`` lexical overlap lead the block;
  salience breaks ties among relevant records only.
- Below-floor records still surface (the backend vouched for them, and
  a paraphrase of a stored fact shares no tokens with the query), but
  rank after every relevant one, by salience.

Scoring uses the shared ``sdk.memory.lexical`` tokenizer — the same token
definition the store's recall pools use — so a record the backend surfaced
as a match can never look disjoint to this ranker.
"""

from __future__ import annotations

import json

from kernel.models import MemoryRecallAnswer, MemoryRecallEvidence, MemoryRecord
from sdk.memory.lexical import tokenize

#: Minimum lexical relevance a record must clear to lead the ranked block.
#: Records below the floor still surface — the backend vouched for them —
#: but only after every record that clears it, regardless of salience.
#: Salience breaks ties among relevant records only.
_RELEVANCE_FLOOR = 0.2

#: Salience tiebreak weight among relevant records.
_TIEBREAK_WEIGHT = 0.2


def build_memory_recall_answer(
    *,
    query: str,
    records: list[MemoryRecord],
    limit: int,
) -> MemoryRecallAnswer:
    """Rank records by relevance first, then salience, and build the recall answer."""
    if not records:
        return MemoryRecallAnswer(
            query=query,
            found=False,
            answer=f"No memory matches found for: {query}",
        )

    query_tokens = tokenize(query)
    ranked: list[tuple[float, bool, MemoryRecord]] = []
    for record in records:
        haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
        overlap = len(query_tokens.intersection(tokenize(haystack)))
        lexical_score = overlap / len(query_tokens) if query_tokens else 0.0
        # Relevance-first: a record must clear the relevance floor to lead
        # the block; salience only breaks ties among relevant records.
        # Below-floor records still surface (the backend vouched for them)
        # but rank after every relevant one.
        salience = max(0.0, min(1.0, record.salience))
        relevant = lexical_score >= _RELEVANCE_FLOOR
        combined = (
            min(1.0, lexical_score + salience * _TIEBREAK_WEIGHT)
            if relevant
            else salience * _TIEBREAK_WEIGHT
        )
        ranked.append((combined, relevant, record))

    # Relevant records lead; ties break by combined score, then record id.
    ranked.sort(key=lambda item: (item[1], item[0], str(item[2].id)), reverse=True)
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
                lexical_score=round(score - record.salience * _TIEBREAK_WEIGHT, 3)
                if relevant
                else 0.0,
                source_turn=source_turn if isinstance(source_turn, int) else None,
                match_reasons=reasons,
            )
        )

    summary = evidence[0].content
    confidence = evidence[0].combined_score if evidence else 0.0

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
        evidence=evidence,
    )
