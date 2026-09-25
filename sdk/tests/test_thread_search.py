"""Tests for ``ThreadStore.search`` — lexical-overlap ranking over thread entries.

Covers:

- Exact query token match ranks first.
- Partial overlap scores lower than full overlap.
- Empty query returns most recent entries (recency fallback).
- Entries with no query and no response are skipped.
- ``limit`` caps results and is clamped to 20.
- Host-command entries are stripped when ``is_host_command`` is wired.
- Search spans both sealed and active segments when ``zstandard`` is
  available (skipped otherwise — the SDK's decompression path requires
  the optional ``zstandard`` dependency, which is a host-only dep).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sdk.thread import ThreadStore
from sdk.thread._backend import _read_index, _sealed_segment_path, _segment_paths

# ── Helpers ───────────────────────────────────────────────────────────


def _write_active_entry(
    thread_path: Path,
    *,
    query: str = "",
    response_text: str = "",
    mode_id: str = "default",
    turn: int | None = None,
    handoff_to: str = "",
) -> None:
    """Append one entry to the active segment (``.current.jsonl``)."""
    paths = _segment_paths(thread_path)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
        "handoff_to": handoff_to,
    }
    if turn is not None:
        entry["turn"] = turn
    with paths.current.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _zstd_available() -> bool:
    try:
        import zstandard

        return bool(zstandard.ZstdCompressor)
    except ImportError:
        return False


def _seal_active(thread_path: Path) -> None:
    """Seal the active segment using zstd compression (host mirrors this)."""
    import zstandard
    from sdk.thread._backend import SegmentRecord

    paths = _segment_paths(thread_path)
    if not paths.current.exists():
        return
    raw_bytes = paths.current.read_bytes()
    if not raw_bytes.strip():
        return

    # Compress without a trained dictionary (test simplicity).
    cctx = zstandard.ZstdCompressor(level=5)
    compressed = cctx.compress(raw_bytes)

    records = _read_index(paths.index_path)
    seg_id = f"{len(records) + 1:04d}"
    sealed_path = _sealed_segment_path(paths.base, seg_id)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_bytes(compressed)

    # Parse entries to count turns and derive the turn range.
    entries = [
        json.loads(line)
        for line in raw_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    turn_lo = sum(r.turn_hi - r.turn_lo + 1 for r in records)
    turn_hi = turn_lo + len(entries) - 1
    record = SegmentRecord(
        seg_id=seg_id,
        turn_lo=turn_lo,
        turn_hi=turn_hi,
        ts_lo="",
        ts_hi="",
        bytes_raw=len(raw_bytes),
        bytes_zst=len(compressed),
    )
    with paths.index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    paths.current.unlink()


def _is_host_cmd(query: str) -> bool:
    """Trivial host-command filter for testing."""
    return query.startswith("/")


# ── search() — basic ranking ─────────────────────────────────────────


class TestThreadStoreSearch:
    def test_exact_token_match_ranks_first(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="unrelated topic",
            response_text="something else entirely",
        )
        _write_active_entry(
            thread_path,
            query="fix the router bug",
            response_text="patched the routing logic",
        )
        store = ThreadStore(thread_path)
        results = store.search(query="router bug")
        assert len(results) == 2
        # The entry containing both "router" and "bug" should rank first.
        assert results[0]["query"] == "fix the router bug"
        assert results[0]["score"] > results[1]["score"]

    def test_partial_overlap_scores_lower(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="router configuration",
            response_text="set up the router",
        )
        _write_active_entry(
            thread_path,
            query="router bug fix",
            response_text="router bug was in the routing module",
        )
        store = ThreadStore(thread_path)
        results = store.search(query="router bug")
        # Both match "router", but only the second matches "bug" too.
        assert results[0]["query"] == "router bug fix"
        assert results[0]["score"] > results[1]["score"]

    def test_empty_query_returns_most_recent(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="first", response_text="a1")
        _write_active_entry(thread_path, query="second", response_text="a2")
        _write_active_entry(thread_path, query="third", response_text="a3")
        store = ThreadStore(thread_path)
        results = store.search(query="", limit=10)
        assert len(results) == 3
        # Empty query → all score 1.0 → sort by turn desc → most recent first.
        assert results[0]["query"] == "third"
        assert results[1]["query"] == "second"
        assert results[2]["query"] == "first"

    def test_empty_entries_are_skipped(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="", response_text="")
        _write_active_entry(thread_path, query="real query", response_text="real answer")
        store = ThreadStore(thread_path)
        results = store.search(query="real")
        assert len(results) == 1
        assert results[0]["query"] == "real query"

    def test_limit_caps_results(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        for i in range(10):
            _write_active_entry(
                thread_path,
                query=f"topic {i}",
                response_text=f"answer {i}",
            )
        store = ThreadStore(thread_path)
        results = store.search(query="topic", limit=3)
        assert len(results) == 3

    def test_limit_clamped_to_20(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        for i in range(25):
            _write_active_entry(
                thread_path,
                query=f"shared topic {i}",
                response_text=f"answer {i}",
            )
        store = ThreadStore(thread_path)
        results = store.search(query="shared topic", limit=100)
        assert len(results) == 20

    def test_empty_store_returns_empty_list(self, tmp_path: Path) -> None:
        store = ThreadStore(tmp_path / "nonexistent")
        assert store.search(query="anything") == []

    def test_result_shape(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="test query",
            response_text="a" * 300,
            mode_id="planner",
            handoff_to="executor",
        )
        store = ThreadStore(thread_path)
        results = store.search(query="test")
        assert len(results) == 1
        result = results[0]
        assert result["query"] == "test query"
        assert result["mode_id"] == "planner"
        assert result["handoff_to"] == "executor"
        # response_preview is capped at 200 chars.
        assert len(result["response_preview"]) == 200
        assert isinstance(result["score"], float)
        assert isinstance(result["turn"], int)

    def test_host_command_filter(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="/pack list", response_text="output")
        _write_active_entry(
            thread_path,
            query="real question",
            response_text="real answer",
        )
        store = ThreadStore(thread_path)
        results = store.search(query="real", is_host_command=_is_host_cmd)
        assert len(results) == 1
        assert results[0]["query"] == "real question"

    def test_host_command_filter_not_passed_keeps_all(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="/pack list", response_text="output")
        _write_active_entry(
            thread_path,
            query="real question",
            response_text="real answer",
        )
        store = ThreadStore(thread_path)
        results = store.search(query="pack")
        # Without the filter, the /pack entry is included.
        assert any(r["query"] == "/pack list" for r in results)

    def test_response_text_is_searched(self, tmp_path: Path) -> None:
        """The response body is part of the haystack, not just the query."""
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="what happened",
            response_text="the database migration failed",
        )
        store = ThreadStore(thread_path)
        results = store.search(query="database migration")
        assert len(results) == 1
        assert results[0]["score"] > 0.0

    def test_mode_id_is_searched(self, tmp_path: Path) -> None:
        """mode_id is part of the haystack for mode-based lookups."""
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="do the thing",
            response_text="done",
            mode_id="executor",
        )
        store = ThreadStore(thread_path)
        results = store.search(query="executor")
        assert len(results) == 1

    def test_global_turn_indices_are_correct(self, tmp_path: Path) -> None:
        """Turn indices in results match the 0-based global turn index."""
        thread_path = tmp_path / "thread"
        for i in range(5):
            _write_active_entry(
                thread_path,
                query=f"query {i}",
                response_text=f"answer {i}",
            )
        store = ThreadStore(thread_path)
        results = store.search(query="query", limit=5)
        turns = [r["turn"] for r in results]
        # Turn indices should be 0–4, ordered most-recent-first (4, 3, 2, 1, 0).
        assert sorted(turns) == [0, 1, 2, 3, 4]
        assert turns[0] == 4  # most recent


# ── search() — sealed + active segments ──────────────────────────────


@pytest.mark.skipif(not _zstd_available(), reason="zstandard not installed")
class TestThreadStoreSearchSealedSegments:
    """Search must span both sealed and active segments."""

    def test_search_finds_entries_in_sealed_segment(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        # Write 3 entries, then seal them into a sealed segment.
        _write_active_entry(
            thread_path,
            query="zstd compression design",
            response_text="decided to use zstd for thread sealing",
        )
        _write_active_entry(thread_path, query="other topic", response_text="other answer")
        _write_active_entry(thread_path, query="another topic", response_text="another answer")
        _seal_active(thread_path)

        # Write 2 more entries to the new active segment.
        _write_active_entry(
            thread_path,
            query="recent question",
            response_text="recent answer",
        )
        _write_active_entry(thread_path, query="latest question", response_text="latest answer")

        store = ThreadStore(thread_path)
        results = store.search(query="zstd compression")
        assert len(results) >= 1
        # The sealed-segment entry should be found and rank first.
        assert results[0]["query"] == "zstd compression design"
        # Its global turn index should be 0 (first entry, now sealed).
        assert results[0]["turn"] == 0

    def test_search_spans_all_sealed_and_active(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        # Seal two segments.
        _write_active_entry(thread_path, query="alpha sealed", response_text="alpha answer")
        _write_active_entry(thread_path, query="beta sealed", response_text="beta answer")
        _seal_active(thread_path)
        _write_active_entry(thread_path, query="gamma sealed", response_text="gamma answer")
        _write_active_entry(thread_path, query="delta sealed", response_text="delta answer")
        _seal_active(thread_path)

        # Active tail.
        _write_active_entry(thread_path, query="epsilon active", response_text="epsilon answer")

        store = ThreadStore(thread_path)
        results = store.search(query="sealed")
        # Should find all 4 sealed entries (alpha, beta, gamma, delta).
        sealed_queries = {r["query"] for r in results if "sealed" in r["query"]}
        assert sealed_queries == {"alpha sealed", "beta sealed", "gamma sealed", "delta sealed"}

    def test_sealed_turn_indices_are_global(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="first sealed", response_text="a1")
        _write_active_entry(thread_path, query="second sealed", response_text="a2")
        _seal_active(thread_path)
        _write_active_entry(thread_path, query="third active", response_text="a3")

        store = ThreadStore(thread_path)
        results = store.search(query="sealed active", limit=10)
        turns_by_query = {r["query"]: r["turn"] for r in results}
        # Sealed entries get turns 0 and 1; active entry gets turn 2.
        assert turns_by_query["first sealed"] == 0
        assert turns_by_query["second sealed"] == 1
        assert turns_by_query["third active"] == 2