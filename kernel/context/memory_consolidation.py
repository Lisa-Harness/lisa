"""Memory consolidation job for the Lisa memory architecture.

Runs periodically to:

1. Promote high-access episodic memories to ``semantic`` kind so durable
   facts accumulate while ephemeral notes age out.
2. Evict low-value memories under the configured caps and floors
   (age TTL, salience floor, per-scope cap, per-kind cap).

The job is lightweight by design: it only issues UPDATE statements and
emits one sync-queue event per batch. It never calls external APIs.

Registered by ``heartbeat.create_scheduler()`` when
``settings.memory_consolidation_enabled`` is True.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

from app.config import settings
from app.memory_store import get_memory_store

logger = logging.getLogger(__name__)


async def run_memory_consolidation() -> dict[str, int]:
    """Run the full consolidation pass: promotion + eviction.

    Returns a small dict of counts for observability. Heartbeat and the
    /debug consolidation consumer both read this shape.
    """
    store = get_memory_store()
    result: dict[str, int] = {
        "promoted": 0,
        "evicted_age": 0,
        "evicted_salience": 0,
        "evicted_scope_cap": 0,
        "evicted_kind_cap": 0,
    }
    try:
        promoted = await asyncio.to_thread(
            _promote_episodic_memories, str(store._db_path)
        )
        result["promoted"] = promoted
    except Exception:  # pylint: disable=broad-except
        logger.exception("memory consolidation promotion step failed")

    # Eviction runs in the same job to keep the policy surface small.
    # Personal scope is the high-traffic path, so it is run per known user
    # slug we can find in the memories table; surface-scoped memories
    # are evicted separately per known surface.
    try:
        for scope_target in await asyncio.to_thread(_list_eviction_targets):
            stats = await asyncio.to_thread(
                store.evict_memories_for_consolidation,
                memory_scope=scope_target["memory_scope"],
                surface_id=scope_target["surface_id"],
                user_slug=scope_target["user_slug"],
            )
            logger.debug(
                "memory consolidation eviction: scope=%s surface=%s user=%s stats=%s",
                scope_target["memory_scope"],
                scope_target["surface_id"],
                scope_target["user_slug"],
                stats,
            )
            result["evicted_age"] += int(stats.get("age_retired", 0))
            result["evicted_salience"] += int(stats.get("salience_retired", 0))
            result["evicted_scope_cap"] += int(stats.get("scope_cap_retired", 0))
            result["evicted_kind_cap"] += int(stats.get("kind_cap_retired", 0))
    except Exception:  # pylint: disable=broad-except
        logger.exception("memory consolidation eviction step failed")

    total_evicted = sum(
        int(v)
        for k, v in result.items()
        if k.startswith("evicted_")
    )
    if result["promoted"] or total_evicted:
        logger.info(
            "memory consolidation: promoted=%d evicted=%d (age=%d salience=%d scope=%d kind=%d)",
            result["promoted"],
            total_evicted,
            result["evicted_age"],
            result["evicted_salience"],
            result["evicted_scope_cap"],
            result["evicted_kind_cap"],
        )
    return result


def _promote_episodic_memories(db_path: str) -> int:
    """Promote eligible episodic memories to ``semantic`` kind.

    Promotion criteria (all must hold):

    - ``memory_kind == 'episodic'``
    - ``access_count >= settings.memory_promotion_threshold``
    - ``status == 'active'``
    """
    threshold = int(settings.memory_promotion_threshold)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE memories"
            " SET memory_kind = 'semantic'"
            " WHERE memory_kind = 'episodic'"
            " AND status = 'active'"
            " AND access_count >= ?",
            (threshold,),
        )
        conn.commit()
        return int(cursor.rowcount or 0)


def _list_eviction_targets() -> list[dict[str, str]]:
    """Return one (memory_scope, surface_id, user_slug) tuple per populated scope.

    The eviction policy is per-scope, so we discover scopes that actually
    have rows and run one eviction pass per scope. This avoids the cost
    of a global eviction pass when most scopes are empty.
    """
    store = get_memory_store()
    targets = store.list_active_scopes()
    if not targets:
        # Always run at least one personal pass on a fresh db.
        targets.append({"memory_scope": "personal", "surface_id": "", "user_slug": ""})
    return targets
