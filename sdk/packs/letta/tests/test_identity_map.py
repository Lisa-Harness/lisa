"""Tests for the Lisa-to-Letta identity map sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdk.packs.letta.utils.identity_map import IdentityEntry, IdentityMap, identity_key


def test_identity_key_is_stable(tmp_path: Path) -> None:
    k1 = identity_key(workspace_id="ws", user_slug="u", surface_id="s", conversation_id="c")
    k2 = identity_key(workspace_id="ws", user_slug="u", surface_id="s", conversation_id="c")
    assert k1 == k2
    assert "ws" in k1 and "u" in k1


def test_identity_key_defaults_missing_parts() -> None:
    k = identity_key(workspace_id="", user_slug="", surface_id="", conversation_id="")
    assert k.startswith("default|default|")


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    imap = IdentityMap(tmp_path)
    assert imap.load() == {}
    assert imap.get("any") is None


def test_upsert_and_get_roundtrip(tmp_path: Path) -> None:
    imap = IdentityMap(tmp_path)
    entry = IdentityEntry(agent_id="agent-1", agent_name="lisa-agent")
    imap.upsert("k", entry)
    loaded = imap.get("k")
    assert loaded is not None
    assert loaded.agent_id == "agent-1"
    assert loaded.agent_name == "lisa-agent"


def test_update_patches_fields(tmp_path: Path) -> None:
    imap = IdentityMap(tmp_path)
    imap.upsert("k", IdentityEntry(agent_id="a1"))
    updated = imap.update("k", last_sync_turn=5, last_sync_at="2026-07-18")
    assert updated is not None
    assert updated.last_sync_turn == 5
    assert updated.last_sync_at == "2026-07-18"
    # Persisted
    reloaded = imap.get("k")
    assert reloaded is not None
    assert reloaded.last_sync_turn == 5


def test_update_returns_none_for_missing_key(tmp_path: Path) -> None:
    imap = IdentityMap(tmp_path)
    assert imap.update("missing", agent_id="x") is None


def test_all_entries_parses_all(tmp_path: Path) -> None:
    imap = IdentityMap(tmp_path)
    imap.upsert("k1", IdentityEntry(agent_id="a1"))
    imap.upsert("k2", IdentityEntry(agent_id="a2"))
    entries = imap.all_entries()
    assert len(entries) == 2
    assert {e.agent_id for e in entries.values()} == {"a1", "a2"}


def test_corrupt_file_returns_empty(tmp_path: Path) -> None:
    imap = IdentityMap(tmp_path)
    imap.upsert("k", IdentityEntry(agent_id="a1"))
    # Corrupt the file
    imap.path.write_text("{not valid json", encoding="utf-8")
    assert imap.load() == {}
    assert imap.get("k") is None