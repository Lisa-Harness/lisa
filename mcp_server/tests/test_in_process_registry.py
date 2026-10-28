"""Tests for the in-process MCP server registry.

Covers:
- ``in_process_server_ids`` returns the expected first-party server ids.
- ``get_in_process_server`` lazily imports and caches server entries.
- Unknown server_ids return ``None``.
"""

from __future__ import annotations

from mcp_server.registry import get_in_process_server, in_process_server_ids


def test_in_process_server_ids_contains_first_party() -> None:
    """All first-party server ids are available for in-process mode."""
    ids = in_process_server_ids()
    assert "lisa-catalog" in ids
    assert "lisa-content" in ids
    assert "lisa-fetch" in ids
    assert "lisa-guard" in ids
    assert "lisa-memory" in ids
    assert "lisa-storage" in ids
    assert "lisa-thread" in ids


def test_get_in_process_server_returns_entry() -> None:
    """Resolving a known server_id returns an entry with mcp + init."""
    entry = get_in_process_server("lisa-content")
    assert entry is not None
    assert entry.server_id == "lisa-content"
    assert entry.mcp is not None
    assert callable(entry.init)


def test_get_in_process_server_caches() -> None:
    """Repeated lookups return the same cached entry object."""
    entry1 = get_in_process_server("lisa-catalog")
    entry2 = get_in_process_server("lisa-catalog")
    assert entry1 is entry2


def test_get_in_process_server_unknown_returns_none() -> None:
    """Unknown server_ids return None (not a first-party server)."""
    entry = get_in_process_server("nonexistent-server")
    assert entry is None
