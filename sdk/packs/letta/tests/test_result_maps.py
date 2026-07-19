"""Tests for Letta result map normalizers."""

from __future__ import annotations

import json

import pytest

from sdk.packs.letta.utils.result_maps import normalize_add, normalize_list, normalize_search


def test_normalize_add_projects_passage_to_record() -> None:
    raw = json.dumps({"id": "pass-1", "text": "durable fact", "created_at": "2026-07-18T00:00:00Z"})
    out = json.loads(normalize_add(raw))
    assert out["id"] == "pass-1"
    assert out["content"] == "durable fact"
    assert out["created_at"] == "2026-07-18T00:00:00Z"
    assert out["content_json"]["source"] == "letta"
    assert out["content_json"]["letta"]["passage_id"] == "pass-1"


def test_normalize_add_passes_error_envelope_through() -> None:
    raw = json.dumps({"error": "boom"})
    out = json.loads(normalize_add(raw))
    assert out["error"] == "boom"


def test_normalize_add_tolerates_non_object() -> None:
    out = json.loads(normalize_add("not json"))
    assert "error" in out


def test_normalize_search_returns_bare_array() -> None:
    raw = json.dumps([{"id": "p1", "text": "a"}, {"id": "p2", "text": "b"}])
    out = json.loads(normalize_search(raw))
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["content"] == "a"
    assert out[1]["id"] == "p2"


def test_normalize_search_unwraps_results_envelope() -> None:
    raw = json.dumps({"results": [{"id": "p1", "text": "x"}]})
    out = json.loads(normalize_search(raw))
    assert isinstance(out, list)
    assert len(out) == 1


def test_normalize_search_empty_and_malformed() -> None:
    assert json.loads(normalize_search("")) == []
    assert json.loads(normalize_search("not json")) == []
    assert json.loads(normalize_search("{}")) == []


def test_normalize_list_returns_bare_array() -> None:
    raw = json.dumps([{"id": "p1", "text": "recent"}])
    out = json.loads(normalize_list(raw))
    assert isinstance(out, list)
    assert out[0]["content"] == "recent"


def test_normalize_list_unwraps_envelope() -> None:
    raw = json.dumps({"results": [{"id": "p1", "text": "r"}]})
    out = json.loads(normalize_list(raw))
    assert len(out) == 1


def test_record_carries_letta_metadata() -> None:
    raw = json.dumps({"id": "p1", "text": "t", "metadata": {"salience": 0.9}})
    out = json.loads(normalize_add(raw))
    assert out["content_json"]["letta"]["metadata"] == {"salience": 0.9}