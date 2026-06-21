"""Deterministic passive-intelligence lane built from scoped memories and turns.

This module captures the missing middle between raw episodic memories and the
heavier reflection / strategy synthesis pipeline. It is intentionally cheap:
all scoring is local, deterministic, and privacy-scoped.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from kernel.models import (
    MemoryRecord,
    MemoryScope,
    PassiveIntelligenceItem,
    PassiveIntelligenceSnapshot,
)

from app import blob_client
from app.config import settings
from app.memory_store import get_memory_store
from app.thread_store import get_thread_store

logger = logging.getLogger(__name__)

__all__ = [
    "blob_client",
    "settings",
]

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "up",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}
_NEGATION_TOKENS = frozenset(
    {"blocked", "broken", "cant", "can't", "failing", "issue", "stuck", "wont", "won't"}
)
_PREFERENCE_TOKENS = frozenset({"avoid", "hate", "like", "love", "prefer", "want"})
_RESOLVED_TOKENS = frozenset(
    {"closed", "completed", "deployed", "done", "fixed", "merged", "resolved", "shipped"}
)
_DIRTY_SCOPES: set[tuple[str, str, MemoryScope, str]] = set()
_DIRTY_LOCK = asyncio.Lock()


def _tokenize(text: str) -> set[str]:
    tokens = {match.group(0).lower() for match in _TOKEN_RE.finditer(text.lower())}
    return {token for token in tokens if token and token not in _STOPWORDS and len(token) >= 3}


def _recency_decay(timestamp_iso: str, half_life_hours: float = 24.0) -> float:
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_hours = max(0.0, (datetime.now(UTC) - ts).total_seconds() / 3600.0)
        return 2 ** (-age_hours / max(half_life_hours, 1.0))
    except Exception:  # pylint: disable=broad-except
        return 0.0


def _insight_id(kind: str, label: str) -> str:
    return hashlib.sha256(f"{kind}:{label}".encode()).hexdigest()[:16]


def _classify_kind(all_tokens: set[str]) -> str:
    if all_tokens & _PREFERENCE_TOKENS:
        return "preference"
    if (all_tokens & _NEGATION_TOKENS) and not (all_tokens & _RESOLVED_TOKENS):
        return "open_loop"
    if any(any(ch.isdigit() for ch in token) for token in all_tokens):
        return "entity"
    return "theme"


def _composite_score(evidence_count: int, recency_score: float, salience_score: float) -> float:
    return round(
        0.45 * min(1.0, evidence_count / 8)
        + 0.30 * recency_score
        + 0.25 * salience_score,
        4,
    )


def _build_label(token: str, entries: list[dict[str, object]]) -> tuple[str, list[str]]:
    companions: dict[str, int] = defaultdict(int)
    for entry in entries:
        for companion in entry["tokens"]:  # type: ignore[index]
            if companion == token:
                continue
            companions[str(companion)] += 1
    top_companions = [
        companion
        for companion, count in sorted(companions.items(), key=lambda item: (-item[1], item[0]))
        if count >= settings.passive_intelligence_min_recurrence
    ][:1]
    lexical_tokens = [token, *top_companions]
    if top_companions:
        return f"{token} {top_companions[0]}", lexical_tokens
    return token, lexical_tokens


def build_passive_intelligence_snapshot(
    *,
    workspace_id: str,
    user_slug: str,
    memory_scope: MemoryScope,
    surface_id: str,
    memory_records: list[MemoryRecord],
    turn_messages: list[dict[str, str]],
) -> PassiveIntelligenceSnapshot:
    """Build one passive-intelligence snapshot from scoped memories and thread turns."""
    now = datetime.now(UTC).isoformat()
    if not settings.passive_intelligence_enabled:
        return PassiveIntelligenceSnapshot(
            generated_at=now,
            workspace_id=workspace_id,
            user_slug=user_slug,
            memory_scope=memory_scope,
            surface_id=surface_id,
            status="skipped",
        )

    corpus: list[dict[str, object]] = []
    for record in memory_records:
        if not record.content.strip():
            continue
        tokens = _tokenize(record.content)
        if not tokens:
            continue
        corpus.append(
            {
                "tokens": tokens,
                "salience": max(0.0, float(record.salience or 0.0)),
                "ts": record.created_at,
                "content": record.content[:300],
                "memory_id": record.id,
            }
        )

    for turn in turn_messages:
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        tokens = _tokenize(content)
        if not tokens:
            continue
        corpus.append(
            {
                "tokens": tokens,
                "salience": 0.4,
                "ts": str(turn.get("created_at", "") or ""),
                "content": content[:300],
                "memory_id": None,
            }
        )

    if not corpus:
        return PassiveIntelligenceSnapshot(
            generated_at=now,
            workspace_id=workspace_id,
            user_slug=user_slug,
            memory_scope=memory_scope,
            surface_id=surface_id,
            source_memory_count=len(memory_records),
            item_count=0,
            summary_markdown="",
            insights=[],
            status="ok",
        )

    token_entries: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in corpus:
        for token in entry["tokens"]:  # type: ignore[index]
            token_entries[str(token)].append(entry)

    scored_items: list[tuple[float, PassiveIntelligenceItem]] = []
    seen_labels: set[str] = set()
    for token, entries in token_entries.items():
        if len(entries) < settings.passive_intelligence_min_recurrence:
            continue

        avg_salience = sum(float(entry["salience"]) for entry in entries) / len(entries)
        if avg_salience < settings.passive_intelligence_min_salience:
            continue

        recency_samples = [
            _recency_decay(str(entry.get("ts", "")))
            for entry in entries
            if entry.get("ts")
        ]
        recency_score = (
            sum(recency_samples) / len(recency_samples)
            if recency_samples
            else 0.5
        )

        all_tokens: set[str] = set()
        snippets: list[str] = []
        memory_ids: list[int] = []
        for entry in entries:
            all_tokens.update(entry["tokens"])  # type: ignore[arg-type]
            raw = str(entry.get("content", "") or "").strip()
            if raw and raw not in snippets:
                snippets.append(raw)
            memory_id = entry.get("memory_id")
            if isinstance(memory_id, int):
                memory_ids.append(memory_id)

        label, lexical_tokens = _build_label(token, entries)
        if label in seen_labels:
            continue
        seen_labels.add(label)

        kind = _classify_kind(all_tokens)
        score = _composite_score(len(entries), recency_score, avg_salience)
        if score < settings.passive_intelligence_min_score:
            continue

        summary = (
            f"Open loop around {label}."
            if kind == "open_loop"
            else f"Recurring preference around {label}."
            if kind == "preference"
            else f"Recurring theme around {label}."
        )

        scored_items.append(
            (
                score,
                PassiveIntelligenceItem(
                    insight_id=_insight_id(kind, label),
                    label=label,
                    kind=kind,  # type: ignore[arg-type]
                    summary=summary,
                    evidence_count=len(entries),
                    recency_score=round(recency_score, 4),
                    salience_score=round(avg_salience, 4),
                    last_seen_at=max(
                        (str(entry.get("ts", "") or "") for entry in entries),
                        default="",
                    ),
                    source_memory_ids=sorted(set(memory_ids)),
                    evidence_snippets=snippets[: settings.passive_intelligence_evidence_limit],
                    lexical_tokens=lexical_tokens,
                ),
            )
        )

    scored_items.sort(key=lambda item: item[0], reverse=True)
    insights = [item for _, item in scored_items[: settings.passive_intelligence_max_items]]
    summary_lines = ["## Passive Intelligence Notes"]
    for item in insights:
        summary_lines.append(f"- {item.summary}")

    return PassiveIntelligenceSnapshot(
        generated_at=now,
        workspace_id=workspace_id,
        user_slug=user_slug,
        memory_scope=memory_scope,
        surface_id=surface_id,
        source_memory_count=len(memory_records),
        item_count=len(insights),
        summary_markdown="\n".join(summary_lines) if insights else "",
        insights=insights,
        status="ok",
    )


async def mark_passive_intelligence_dirty(
    *,
    workspace_id: str,
    user_slug: str,
    memory_scope: MemoryScope,
    surface_id: str,
) -> None:
    """Mark one privacy-scoped passive-intelligence snapshot as needing refresh."""
    async with _DIRTY_LOCK:
        _DIRTY_SCOPES.add((workspace_id, user_slug, memory_scope, surface_id))


async def drain_passive_intelligence_scopes() -> list[tuple[str, str, MemoryScope, str]]:
    """Return and clear the current dirty-scope queue."""
    async with _DIRTY_LOCK:
        scopes = list(_DIRTY_SCOPES)
        _DIRTY_SCOPES.clear()
    return scopes


def _iter_scoped_turn_messages(
    *,
    user_slug: str,
    surface_id: str,
    memory_scope: MemoryScope,
) -> list[dict[str, str]]:
    thread_store = get_thread_store()
    messages: list[dict[str, str]] = []
    for state in thread_store.iter_cached_states():
        if memory_scope == "shared":
            if getattr(state, "surface_id", "") != surface_id:
                continue
        else:
            if getattr(state, "user_slug", "") != user_slug:
                continue
            if surface_id and getattr(state, "surface_id", "") != surface_id:
                continue

        for turn in state.turns[-settings.passive_intelligence_window_turns :]:
            messages.append(
                {
                    "content": turn.content,
                    "created_at": getattr(turn, "created_at", ""),
                }
            )
    return messages


async def refresh_passive_intelligence_snapshot(
    *,
    workspace_id: str,
    user_slug: str,
    memory_scope: MemoryScope,
    surface_id: str,
) -> PassiveIntelligenceSnapshot:
    """Refresh one scoped passive-intelligence snapshot from local stores."""
    memory_store = get_memory_store()
    memory_records = await memory_store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=settings.passive_intelligence_window_memories,
    )
    turn_messages = _iter_scoped_turn_messages(
        user_slug=user_slug,
        surface_id=surface_id,
        memory_scope=memory_scope,
    )
    snapshot = build_passive_intelligence_snapshot(
        workspace_id=workspace_id,
        user_slug=user_slug,
        memory_scope=memory_scope,
        surface_id=surface_id,
        memory_records=memory_records,
        turn_messages=turn_messages,
    )
    blob_client.write_passive_intelligence_snapshot(snapshot)
    return snapshot


async def refresh_dirty_passive_intelligence_snapshots() -> int:
    """Refresh all currently dirty passive-intelligence scopes."""
    if not settings.passive_intelligence_enabled:
        return 0

    refreshed = 0
    dirty_scopes = await drain_passive_intelligence_scopes()
    for workspace_id, user_slug, memory_scope, surface_id in dirty_scopes:
        try:
            await refresh_passive_intelligence_snapshot(
                workspace_id=workspace_id,
                user_slug=user_slug,
                memory_scope=memory_scope,
                surface_id=surface_id,
            )
            refreshed += 1
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "passive-intelligence refresh failed (%s/%s/%s): %s",
                workspace_id,
                memory_scope,
                user_slug or surface_id,
                exc,
            )
    return refreshed


def passive_intelligence_is_stale(snapshot: PassiveIntelligenceSnapshot | None) -> bool:
    """Return whether one snapshot is missing or older than the configured TTL."""
    if snapshot is None or not snapshot.generated_at:
        return True
    try:
        generated_at = datetime.fromisoformat(snapshot.generated_at)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)
    except ValueError:
        return True
    expiry = generated_at + timedelta(minutes=settings.passive_intelligence_stale_after_minutes)
    return datetime.now(UTC) >= expiry


def _score_item_overlap(query_tokens: set[str], item: PassiveIntelligenceItem) -> float:
    if not query_tokens:
        return 0.0
    item_tokens = set(item.lexical_tokens) or _tokenize(item.label)
    overlap = len(query_tokens & item_tokens) / max(len(query_tokens), 1)
    return round(overlap, 4)


async def build_passive_intelligence_snippet(
    *,
    query: str,
    user_slug: str,
    workspace_id: str,
    memory_scope: MemoryScope,
    surface_id: str,
    budget_tokens: int,
) -> str:
    """Return a compact scoped passive-intelligence block for prompt injection."""
    if (
        not settings.passive_intelligence_enabled
        or not settings.passive_intelligence_default_chat_enabled
    ):
        return ""

    snapshot = blob_client.read_passive_intelligence_snapshot(
        workspace_id=workspace_id,
        user_slug=user_slug,
        memory_scope=memory_scope,
        surface_id=surface_id,
    )
    if passive_intelligence_is_stale(snapshot):
        snapshot = await refresh_passive_intelligence_snapshot(
            workspace_id=workspace_id,
            user_slug=user_slug,
            memory_scope=memory_scope,
            surface_id=surface_id,
        )

    if snapshot is None or not snapshot.insights:
        return ""

    query_tokens = _tokenize(query)
    scored = [(_score_item_overlap(query_tokens, item), item) for item in snapshot.insights]
    selected = [
        item
        for overlap, item in sorted(
            scored,
            key=lambda value: (
                value[0],
                value[1].recency_score,
                value[1].salience_score,
            ),
            reverse=True,
        )
        if overlap >= settings.passive_intelligence_default_chat_min_overlap
    ]
    if not selected:
        return ""

    budget_chars = budget_tokens * 4
    lines = ["## Passive Intelligence Notes"]
    used = len(lines[0])
    for item in selected:
        line = f"- {item.summary}"
        if used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1

    return "\n".join(lines) if len(lines) > 1 else ""
