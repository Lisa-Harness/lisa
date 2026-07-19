"""Normalization adapters from Letta response shapes to canonical Lisa memory JSON.

These functions are referenced by name (dotted import path) in
``mcp_servers.yaml`` ``result_maps`` and resolved lazily by the host
when wiring the generic ``memory.*`` aliases. Each takes the adapter
MCP tool's ``output_text`` (a JSON string) and returns a normalized
JSON string that the existing Lisa operations
(``recall_memory``, ``synthesize_and_store_learning``) can consume via
:class:`kernel.models.MemoryRecord`.

Letta archival memory passages carry ``id`` and ``text``. Lisa
``MemoryRecord`` requires ``id`` and ``created_at``; the rest is
optional. We synthesize stable string ids (Letta ids are already
strings), echo the passage text as ``content``, and stash Letta-specific
fields under ``content_json.letta`` so no new top-level schema field is
needed.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["normalize_add", "normalize_search", "normalize_list"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _record_from_passage(
    passage: dict[str, Any],
    *,
    user_slug: str = "default",
    workspace_id: str = "default",
    memory_scope: str = "personal",
    surface_id: str = "",
    agent_id: str = "letta",
    operation_id: str = "letta.add",
    instruction_id: str = "letta.v1",
    memory_tier: str = "long_term",
    salience: float = 0.0,
) -> dict[str, Any]:
    """Project one Letta passage into a canonical Lisa ``MemoryRecord`` dict."""
    pid = str(passage.get("id") or "")
    text = str(passage.get("text") or passage.get("content") or "")
    created = str(passage.get("created_at") or passage.get("date") or _now_iso())
    content_json: dict[str, Any] = {"source": "letta", "letta": {"passage_id": pid}}
    # Preserve any extra metadata the adapter may have attached.
    extra = passage.get("metadata")
    if isinstance(extra, dict):
        content_json["letta"]["metadata"] = extra
    return {
        "id": pid or f"letta-{_now_iso()}",
        "workspace_id": workspace_id,
        "user_id": user_slug,
        "user_slug": user_slug,
        "memory_scope": memory_scope,
        "surface_id": surface_id,
        "source_user_id": user_slug,
        "source_user_slug": user_slug,
        "agent_id": agent_id,
        "operation_id": operation_id,
        "instruction_id": instruction_id,
        "memory_tier": memory_tier,
        "content": text,
        "content_json": content_json,
        "salience": salience,
        "created_at": created,
    }


def normalize_add(output_text: str) -> str:
    """Normalize an ``add_memory`` adapter response.

    The adapter returns a JSON object describing the created passage (or
    an error envelope). We return a single ``MemoryRecord``-shaped dict
    so downstream operations can validate it directly.
    """
    data = _parse(output_text)
    if not isinstance(data, dict):
        return json.dumps({"error": "letta add returned non-object"})
    if "error" in data and not data.get("id"):
        return json.dumps(data)
    return json.dumps(_record_from_passage(data))


def normalize_search(output_text: str) -> str:
    """Normalize a ``search_memory`` adapter response to a bare JSON array.

    Letta search returns a list of passages (possibly wrapped in a
    ``results`` envelope). The memory recall operation expects a bare
    array; the envelope unwrap here is the symmetric counterpart of the
    belt-and-suspenders fallback already in ``recall.py``.
    """
    data = _parse(output_text)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        data = data["results"]
    if not isinstance(data, list):
        return json.dumps([])
    records = [_record_from_passage(p) for p in data if isinstance(p, dict)]
    return json.dumps(records)


def normalize_list(output_text: str) -> str:
    """Normalize a ``list_memories`` adapter response to a bare JSON array."""
    data = _parse(output_text)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        data = data["results"]
    if not isinstance(data, list):
        return json.dumps([])
    records = [_record_from_passage(p) for p in data if isinstance(p, dict)]
    return json.dumps(records)