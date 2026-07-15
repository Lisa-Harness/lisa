"""Output-normalization adapters for Mem0 MCP tool responses.

These functions are declared in ``mcp_servers.yaml`` under
``memory.backends[].result_maps`` and resolved by
:func:`hosts.core.memory_aliases._backend_result_maps` at alias-registration
time. Each one receives the raw ``output_text`` from the Mem0 MCP tool and
returns a normalized string in the **canonical memory shape** — a JSON array
of ``MemoryRecord``-compatible dicts — so the learning-pack operations
(``recall_memory``, ``synthesize_and_store_learning``) consume a uniform
shape regardless of which backend is active.

Mem0 response shapes (v3):
    - ``get_memories`` (list): ``{"count": N, "next": ..., "previous": ..., "results": [...]}``
    - ``search_memories`` (search): ``{"results": [...]}`` — items include a ``score`` field
    - ``add_memory`` (add): ``{"results": [{"id": ..., "memory": ..., "event_id": ...}]}``

Each result item has:
    - ``id``: UUID string (not int)
    - ``memory``: the memory text (not ``content``)
    - ``created_at``: ISO datetime string
    - ``score`` (search only): float 0..1
    - ``user_id``: the scoped user
    - ``metadata``: dict
    - ``categories``: list[str]
    - ``updated_at``: ISO datetime or null
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _unwrap_results(raw_text: str) -> list[dict[str, Any]]:
    """Parse the raw MCP tool output and extract the results list.

    Handles three shapes:
    1. Mem0 paginated envelope: ``{"results": [...], "count": ...}``
    2. Bare list of items (already canonical from some other backend)
    3. Empty or invalid → returns ``[]``
    """
    if not raw_text or not raw_text.strip():
        return []
    try:
        data: Any = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        logger.debug("result_map: failed to parse JSON, returning empty list")
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    return []


def _mem0_item_to_record(item: dict[str, Any]) -> dict[str, Any]:
    """Map one Mem0 result item to a canonical ``MemoryRecord``-compatible dict.

    Mem0 fields → canonical fields:
        id (UUID str) → id (kept as str; MemoryRecord.id accepts int|str)
        memory → content
        created_at → created_at
        user_id → user_slug
        score → salience (search responses only)
        metadata → content_json (merged as-is)
        categories → stored in content_json for preservation
    """
    content_json: dict[str, Any] = {}
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        content_json["metadata"] = metadata
    categories = item.get("categories")
    if isinstance(categories, list):
        content_json["categories"] = categories

    return {
        "id": str(item.get("id", "")),
        "content": str(item.get("memory", "")),
        "created_at": str(item.get("created_at", "")),
        "user_slug": str(item.get("user_id", "")),
        "user_id": str(item.get("user_id", "")),
        "salience": float(item.get("score", 0.0) or 0.0),
        "content_json": content_json,
        "memory_scope": "personal",
        "memory_tier": "long_term",
        "surface_id": "",
        "source_user_slug": str(item.get("user_id", "")),
        "operation_id": "mem0.add_memory",
        "agent_id": "lisa",
        "workspace_id": "default",
        "memory_kind": "episodic",
        "confidence": 1.0,
        "evidence_ids": [],
        "supersedes_id": None,
        "valid_until": None,
        "last_accessed_at": None,
        "access_count": 0,
        "status": "active",
    }


def mem0_list_result_map(raw_text: str) -> str:
    """Normalize ``get_memories`` output into a canonical JSON array of records.

    Mem0's ``get_memories`` returns a paginated envelope:
    ``{"count": N, "next": ..., "previous": ..., "results": [...]}``

    This unwraps ``results``, maps each item to the ``MemoryRecord`` shape,
    and returns a flat JSON array string.
    """
    items = _unwrap_results(raw_text)
    records = [_mem0_item_to_record(item) for item in items]
    return json.dumps(records, default=str)


def mem0_search_result_map(raw_text: str) -> str:
    """Normalize ``search_memories`` output into a canonical JSON array of records.

    Same envelope as ``get_memories`` (``{"results": [...]}``), but each
    search item includes a ``score`` field (float 0..1) that maps to
    ``salience`` in the canonical record.
    """
    items = _unwrap_results(raw_text)
    records = [_mem0_item_to_record(item) for item in items]
    return json.dumps(records, default=str)


def mem0_add_result_map(raw_text: str) -> str:
    """Normalize ``add_memory`` output into a simple confirmation string.

    Mem0's ``add_memory`` returns ``{"results": [{"id": ..., "memory": ..., "event_id": ...}]}``
    (an async job accepted response). Normalize to a simple confirmation
    so operations checking ``tool_result.error`` see a clean success.
    """
    items = _unwrap_results(raw_text)
    if not items:
        return "Memory stored."
    first = items[0]
    memory_id = first.get("id", "")
    if memory_id:
        return f"Memory stored with ID {memory_id}"
    return "Memory stored."