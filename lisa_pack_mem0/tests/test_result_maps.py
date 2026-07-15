"""Tests for lisa_pack_mem0.result_maps — Mem0 response normalization adapters.

Covers:
- mem0_list_result_map unwraps the paginated envelope into a flat JSON array.
- mem0_search_result_map maps score → salience.
- mem0_add_result_map normalizes the add response to a confirmation string.
- Edge cases: empty input, bare list, invalid JSON, missing fields.
"""

from __future__ import annotations

import json

from lisa_pack_mem0.result_maps import (
    mem0_add_result_map,
    mem0_list_result_map,
    mem0_search_result_map,
)


# ── Realistic Mem0 response payloads ──────────────────────────────────────


_MEM0_LIST_RESPONSE = json.dumps({
    "count": 2,
    "next": None,
    "previous": None,
    "results": [
        {
            "id": "f4cbdb08-7062-4f3e-8eb2-9f5c80dfe64c",
            "memory": "User moved to San Francisco from New York in January 2026",
            "metadata": {"category": "location"},
            "categories": ["location"],
            "created_at": "2026-01-15T10:30:00Z",
            "updated_at": "2026-01-15T10:30:00Z",
        },
        {
            "id": "a2b8c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
            "memory": "Alex prefers vegetarian restaurants",
            "metadata": {},
            "categories": [],
            "created_at": "2026-07-05T15:30:00Z",
            "updated_at": "2026-07-05T15:30:00Z",
        },
    ],
})

_MEM0_SEARCH_RESPONSE = json.dumps({
    "results": [
        {
            "id": "ea925981-272f-40dd-b576-be64e4871429",
            "memory": "Likes to play cricket and plays cricket on weekends.",
            "user_id": "alice",
            "metadata": {"category": "hobbies"},
            "categories": ["hobbies"],
            "score": 0.82,
            "expiration_date": None,
            "created_at": "2024-07-26T10:29:36.630547-07:00",
            "updated_at": None,
        },
    ],
})

_MEM0_ADD_RESPONSE = json.dumps({
    "results": [
        {
            "id": "new-mem-uuid-1234",
            "memory": "The user likes hiking",
            "event_id": "evt-5678",
        },
    ],
})


# ── Tests ─────────────────────────────────────────────────────────────────


def test_mem0_list_result_map_unwraps_envelope() -> None:
    """The paginated dict envelope is unwrapped to a flat JSON array."""
    raw = mem0_list_result_map(_MEM0_LIST_RESPONSE)
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["content"] == "User moved to San Francisco from New York in January 2026"
    assert data[0]["id"] == "f4cbdb08-7062-4f3e-8eb2-9f5c80dfe64c"
    assert data[0]["created_at"] == "2026-01-15T10:30:00Z"
    # metadata preserved in content_json
    assert data[0]["content_json"]["metadata"] == {"category": "location"}
    assert data[0]["content_json"]["categories"] == ["location"]


def test_mem0_search_result_map_maps_score_to_salience() -> None:
    """The search score field maps to salience in the canonical record."""
    raw = mem0_search_result_map(_MEM0_SEARCH_RESPONSE)
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["content"] == "Likes to play cricket and plays cricket on weekends."
    assert data[0]["salience"] == 0.82
    assert data[0]["user_slug"] == "alice"


def test_mem0_add_result_map_returns_confirmation() -> None:
    """The add response normalizes to a simple confirmation string."""
    raw = mem0_add_result_map(_MEM0_ADD_RESPONSE)
    assert "Memory stored with ID new-mem-uuid-1234" == raw


def test_mem0_list_result_map_empty_input() -> None:
    """Empty input returns an empty JSON array."""
    assert json.loads(mem0_list_result_map("")) == []


def test_mem0_list_result_map_bare_list() -> None:
    """A bare list (already canonical) passes through after normalization."""
    bare = json.dumps([{"id": 1, "memory": "test", "created_at": "2026-01-01"}])
    data = json.loads(mem0_list_result_map(bare))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["content"] == "test"


def test_mem0_list_result_map_invalid_json() -> None:
    """Invalid JSON returns an empty array."""
    assert json.loads(mem0_list_result_map("not json")) == []


def test_mem0_search_result_map_missing_score() -> None:
    """Items without a score field default salience to 0.0."""
    raw = json.dumps({"results": [{"id": "x", "memory": "no score", "created_at": "2026-01-01"}]})
    data = json.loads(mem0_search_result_map(raw))
    assert data[0]["salience"] == 0.0


def test_mem0_add_result_map_empty() -> None:
    """Empty add response returns a generic confirmation."""
    assert mem0_add_result_map("") == "Memory stored."
    assert mem0_add_result_map("{}") == "Memory stored."