"""Tests for ``host.thread`` segmented store + excerpt filtering.

Covers:

- The host-command stripping behaviour so that operator UI gestures
  (``/pack disable github``, ``/help``, ``/tokens``, ...) are not replayed
  to the model as "recent conversation" or accumulated in the digest.
- The segmented + zstd-sealed store: sealing at the segment threshold,
  lossless round-trip, recent-N spanning sealed + active segments, range
  queries, destructive write-back, and the one-time legacy rename.
"""

from __future__ import annotations

import json
from pathlib import Path

# Importing handle_message triggers the registration of host-command
# prefixes in the dynamic registry (host_commands.py). The tests below
# verify both the registry and the transcript filter that depends on it.
import host.handle_message  # noqa: F401
from host.host_commands import (
    host_command_exact,
    host_command_prefixes,
    is_host_command_query,
    register_host_command_exact,
    register_host_command_prefix,
)
from host.thread import (
    SegmentRecord,
    _read_index,
    _segment_paths,
    append_thread_entry,
    build_recent_thread_excerpt,
    build_thread_manifest,
    load_recent_thread_entries,
    load_thread_range,
    load_thread_digest,
    next_turn_index,
    write_turn_history,
)

# ── is_host_command_query ────────────────────────────────────────────


class TestIsHostCommandQuery:
    def test_slash_prefix_commands_match(self) -> None:
        assert is_host_command_query("/pack disable github")
        assert is_host_command_query("/pack list")
        assert is_host_command_query("/help")
        assert is_host_command_query("/tokens")
        assert is_host_command_query("/settings")
        assert is_host_command_query("/debug context")
        assert is_host_command_query("/consolidate")
        assert is_host_command_query("/intake")
        assert is_host_command_query("/models show")

    def test_bare_word_aliases_match(self) -> None:
        assert is_host_command_query("help")
        assert is_host_command_query("tokens")
        assert is_host_command_query("settings")
        assert is_host_command_query("consolidate")

    def test_case_insensitive(self) -> None:
        assert is_host_command_query("/PACK DISABLE GITHUB")
        assert is_host_command_query("/Help")
        assert is_host_command_query("  /pack list  ")

    def test_whitespace_stripped(self) -> None:
        assert is_host_command_query("   /pack list   ")

    def test_normal_dialogue_does_not_match(self) -> None:
        assert not is_host_command_query("tell me about your context")
        assert not is_host_command_query("what is /pack?")
        assert not is_host_command_query("explain /help to me")
        assert not is_host_command_query("/quit")
        assert not is_host_command_query("")

    def test_empty_or_whitespace_only(self) -> None:
        assert not is_host_command_query("")
        assert not is_host_command_query("   ")

    def test_registry_populated_by_handle_message_import(self) -> None:
        """The dispatch layer must register its prefixes on import."""
        prefixes = host_command_prefixes()
        # All dispatch-table prefixes should be registered.
        assert "/help" in prefixes
        assert "/tokens" in prefixes
        assert "/settings" in prefixes
        assert "/consolidate" in prefixes
        assert "/slice" in prefixes
        assert "/intake" in prefixes
        assert "/pack" in prefixes
        assert "/models" in prefixes
        assert "/debug" in prefixes
        # Pack-registered commands (checkpoint pack) should also be registered.
        assert "/checkpoint" in prefixes
        assert "/restore" in prefixes
        assert "/checkpoints" in prefixes

    def test_registry_exact_populated_by_handle_message_import(self) -> None:
        exact = host_command_exact()
        assert "help" in exact
        assert "tokens" in exact
        assert "settings" in exact
        assert "consolidate" in exact
        assert "slice" in exact

    def test_dynamic_registration_works(self) -> None:
        """A newly registered prefix is immediately recognised."""
        register_host_command_prefix("/custom-cmd")
        assert is_host_command_query("/custom-cmd do something")
        assert not is_host_command_query("/custom-other thing")

    def test_dynamic_registration_exact_works(self) -> None:
        register_host_command_exact("custom-alias")
        assert is_host_command_query("custom-alias")
        assert not is_host_command_query("custom-other")


# ── build_recent_thread_excerpt ──────────────────────────────────────


def _write_entry(
    thread_path: Path,
    *,
    user_slug: str = "tester",
    mode_id: str = "default",
    query: str,
    response_text: str = "",
    segment_turns: int = 256,
) -> None:
    append_thread_entry(
        thread_path=thread_path,
        user_slug=user_slug,
        mode_id=mode_id,
        query=query,
        response_text=response_text,
        segment_turns=segment_turns,
    )


class TestBuildRecentThreadExcerpt:
    def test_returns_empty_when_no_thread(self, tmp_path: Path) -> None:
        assert build_recent_thread_excerpt(thread_path=None, limit=3) == ""
        assert (
            build_recent_thread_excerpt(thread_path=tmp_path / "missing", limit=3) == ""
        )

    def test_returns_empty_when_all_entries_are_host_commands(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_entry(thread_path, query="/pack list", response_text="packs: ...")
        _write_entry(thread_path, query="/help", response_text="help text")
        _write_entry(thread_path, query="/tokens", response_text="tokens: ...")
        assert build_recent_thread_excerpt(thread_path=thread_path, limit=3) == ""

    def test_strips_host_commands_keeps_dialogue(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        # Sequence mirroring the bug report: a /pack command followed by
        # a real question.
        _write_entry(thread_path, query="/pack disable github", response_text="ok")
        _write_entry(
            thread_path,
            query="tell me what you think about your current context",
            response_text="here is my reflection",
        )
        excerpt = build_recent_thread_excerpt(thread_path=thread_path, limit=3)
        assert "Recent thread context:" in excerpt
        # The /pack command must NOT appear in the excerpt.
        assert "/pack disable github" not in excerpt
        assert "ok" not in excerpt
        # The real question must appear.
        assert "tell me what you think about your current context" in excerpt
        assert "here is my reflection" in excerpt

    def test_respects_limit_among_dialogue_only(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        for i in range(5):
            _write_entry(
                thread_path,
                query=f"question {i}",
                response_text=f"answer {i}",
            )
        excerpt = build_recent_thread_excerpt(thread_path=thread_path, limit=3)
        # Header + exactly 3 dialogue lines.
        body_lines = [line for line in excerpt.splitlines() if line.startswith("- ")]
        assert len(body_lines) == 3
        # The 3 most recent dialogue turns should be q2, q3, q4.
        assert "question 4" in excerpt
        assert "question 3" in excerpt
        assert "question 2" in excerpt
        assert "question 1" not in excerpt
        assert "question 0" not in excerpt

    def test_overread_fills_limit_after_host_command_run(self, tmp_path: Path) -> None:
        """A run of host commands should not starve the excerpt of dialogue."""
        thread_path = tmp_path / "thread"
        # Four host commands, then two real questions.
        _write_entry(thread_path, query="/pack list", response_text="...")
        _write_entry(thread_path, query="/pack disable github", response_text="ok")
        _write_entry(thread_path, query="/help", response_text="help")
        _write_entry(thread_path, query="/tokens", response_text="tokens")
        _write_entry(thread_path, query="real question 1", response_text="real answer 1")
        _write_entry(thread_path, query="real question 2", response_text="real answer 2")
        excerpt = build_recent_thread_excerpt(thread_path=thread_path, limit=2)
        body_lines = [line for line in excerpt.splitlines() if line.startswith("- ")]
        assert len(body_lines) == 2
        assert "real question 1" in excerpt
        assert "real question 2" in excerpt
        assert "/pack" not in excerpt
        assert "/help" not in excerpt
        assert "/tokens" not in excerpt

    def test_chronological_order_oldest_first(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_entry(thread_path, query="older", response_text="older answer")
        _write_entry(thread_path, query="newer", response_text="newer answer")
        excerpt = build_recent_thread_excerpt(thread_path=thread_path, limit=5)
        assert excerpt.index("older") < excerpt.index("newer")


# ── build_thread_manifest ───────────────────────────────────────────


class TestBuildThreadManifest:
    def test_marks_most_recent_row_as_current(self, tmp_path: Path) -> None:
        """The final (most recent) row is annotated ``← current``.

        Regression for the 2026-08-14 "bringing back old ideas" symptom:
        the manifest was a flat table of identical-looking rows, so the
        model re-raised stale queries. The current marker anchors the
        model on the latest exchange.
        """
        thread_path = tmp_path / "thread"
        _write_entry(thread_path, query="older question", response_text="older answer")
        _write_entry(thread_path, query="newer question", response_text="newer answer")
        manifest = build_thread_manifest(thread_path=thread_path, limit=30)
        assert "## Thread Manifest" in manifest
        rows = [line for line in manifest.splitlines() if line.startswith("| ")]
        # Header + separator + 2 data rows.
        assert len(rows) == 4
        # The most recent (last) data row carries the current marker; the
        # older row does not.
        assert "← current" not in rows[2]
        assert "← current" in rows[3]
        assert "newer question" in rows[3]

    def test_single_entry_is_current(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_entry(thread_path, query="only question", response_text="only answer")
        manifest = build_thread_manifest(thread_path=thread_path, limit=30)
        rows = [line for line in manifest.splitlines() if line.startswith("| ")]
        # Header + separator + 1 data row.
        assert len(rows) == 3
        assert "← current" in rows[2]

    def test_empty_thread_returns_empty_string(self, tmp_path: Path) -> None:
        assert build_thread_manifest(thread_path=tmp_path / "missing", limit=30) == ""


# ── load_recent_thread_entries (smoke) ───────────────────────────────


class TestLoadRecentThreadEntries:
    def test_returns_last_n_entries(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        for i in range(5):
            _write_entry(thread_path, query=f"q{i}", response_text=f"a{i}")
        entries = load_recent_thread_entries(thread_path=thread_path, limit=3)
        assert len(entries) == 3
        assert entries[0]["query"] == "q2"
        assert entries[-1]["query"] == "q4"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        # Write directly to the active segment to test malformed-line skip.
        thread_path = tmp_path / "thread"
        paths = _segment_paths(thread_path)
        paths.current.parent.mkdir(parents=True, exist_ok=True)
        paths.current.write_text(
            '{"query": "good", "response_text": "ok"}\n'
            "not json\n"
            '{"query": "also good", "response_text": "ok2"}\n',
            encoding="utf-8",
        )
        entries = load_recent_thread_entries(thread_path=thread_path, limit=10)
        assert len(entries) == 2
        assert entries[0]["query"] == "good"
        assert entries[1]["query"] == "also good"


# ── next_turn_index ──────────────────────────────────────────────────


class TestNextTurnIndex:
    def test_empty_store_returns_zero(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        assert next_turn_index(thread_path=thread_path) == 0

    def test_none_path_returns_zero(self) -> None:
        assert next_turn_index(thread_path=None) == 0

    def test_active_only_matches_next_appended_turn(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        for _ in range(5):
            _write_entry(thread_path, query="q", response_text="a")
        # The next entry should be turn 5 (0-based, 5 entries already in).
        assert next_turn_index(thread_path=thread_path) == 5
        _write_entry(thread_path, query="q5", response_text="a5")
        entries = load_thread_range(
            thread_path=thread_path, start_turn=5, end_turn=5
        )
        assert len(entries) == 1
        assert entries[0]["turn"] == 5

    def test_sealed_plus_active_matches(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        # Seal at 3 entries per segment so we get sealed + active.
        for _ in range(3):
            _write_entry(thread_path, query="q", response_text="a", segment_turns=3)
        # After 3 entries the segment sealed; active is now empty.
        # Next turn should be 3.
        assert next_turn_index(thread_path=thread_path) == 3
        _write_entry(thread_path, query="q3", response_text="a3", segment_turns=3)
        entries = load_thread_range(
            thread_path=thread_path, start_turn=3, end_turn=3
        )
        assert len(entries) == 1
        assert entries[0]["turn"] == 3


# ── Segmented store: seal, range, write-back, legacy rename ──────────


class TestSegmentedStore:
    def test_seal_triggers_at_segment_threshold(self, tmp_path: Path) -> None:
        """Appending past the threshold creates a sealed segment + fresh active."""
        thread_path = tmp_path / "thread"
        for i in range(4):
            _write_entry(thread_path, query=f"q{i}", response_text=f"a{i}", segment_turns=3)
        paths = _segment_paths(thread_path)
        # One sealed segment should exist.
        sealed = list(thread_path.parent.glob("thread.*.jsonl.zst"))
        assert len(sealed) == 1
        # Index should have one record.
        records = _read_index(paths.index_path)
        assert len(records) == 1
        assert records[0].turn_lo == 0
        assert records[0].turn_hi == 2
        # Active segment should have the 1 entry written after the seal.
        active = load_recent_thread_entries(thread_path=thread_path, limit=10)
        assert len(active) == 4  # 3 sealed + 1 active

    def test_sealed_segment_decompresses_losslessly(self, tmp_path: Path) -> None:
        """A sealed segment round-trips losslessly through zstd."""
        thread_path = tmp_path / "thread"
        for i in range(3):
            _write_entry(
                thread_path,
                query=f"query-{i}",
                response_text=f"response-{i}",
                segment_turns=3,
            )
        # The seal happens on the 4th append (at 3 == threshold), so append one more.
        _write_entry(thread_path, query="q3", response_text="r3", segment_turns=3)
        entries = load_recent_thread_entries(thread_path=thread_path, limit=10)
        assert len(entries) == 4
        queries = [e["query"] for e in entries]
        assert queries == ["query-0", "query-1", "query-2", "q3"]

    def test_load_recent_spans_sealed_and_active(self, tmp_path: Path) -> None:
        """load_recent returns entries from both sealed and active segments."""
        thread_path = tmp_path / "thread"
        # Seal a segment at 3, then add 2 more to active.
        for i in range(5):
            _write_entry(thread_path, query=f"q{i}", response_text=f"a{i}", segment_turns=3)
        entries = load_recent_thread_entries(thread_path=thread_path, limit=10)
        assert len(entries) == 5
        assert [e["query"] for e in entries] == ["q0", "q1", "q2", "q3", "q4"]

    def test_load_thread_range(self, tmp_path: Path) -> None:
        """Range query returns the correct slice across sealed + active."""
        thread_path = tmp_path / "thread"
        for i in range(7):
            _write_entry(thread_path, query=f"q{i}", response_text=f"a{i}", segment_turns=3)
        # Turns 0-2 in seg 1, 3-5 in seg 2, 6 in active.
        # Range [1, 4] should return q1, q2, q3, q4.
        entries = load_thread_range(thread_path=thread_path, start_turn=1, end_turn=4)
        assert [e["query"] for e in entries] == ["q1", "q2", "q3", "q4"]
        # Range [0, 6] returns everything.
        all_entries = load_thread_range(thread_path=thread_path, start_turn=0, end_turn=6)
        assert len(all_entries) == 7

    def test_write_turn_history_clears_sealed_segments(self, tmp_path: Path) -> None:
        """Destructive write-back removes sealed segments + index, replaces active."""
        thread_path = tmp_path / "thread"
        for i in range(5):
            _write_entry(thread_path, query=f"q{i}", response_text=f"a{i}", segment_turns=3)
        # Verify we have sealed segments.
        sealed = list(thread_path.parent.glob("thread.*.jsonl.zst"))
        assert len(sealed) >= 1
        # Destructive write-back.
        write_turn_history(
            thread_path=thread_path,
            new_history=[{"query": "new", "response_text": "new response"}],
        )
        # Sealed segments + index should be gone.
        sealed_after = list(thread_path.parent.glob("thread.*.jsonl.zst"))
        assert len(sealed_after) == 0
        paths = _segment_paths(thread_path)
        assert not paths.index_path.exists()
        # Active segment should have the new history.
        entries = load_recent_thread_entries(thread_path=thread_path, limit=10)
        assert len(entries) == 1
        assert entries[0]["query"] == "new"

    def test_one_time_rename_of_old_jsonl(self, tmp_path: Path) -> None:
        """A legacy ``<base>.jsonl`` is renamed to the active segment on first append."""
        thread_path = tmp_path / "thread"
        thread_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path = Path(f"{thread_path}.jsonl")
        legacy_path.write_text(
            json.dumps({"query": "old", "response_text": "old resp"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Append a new entry — the rename guard should move the legacy file.
        _write_entry(thread_path, query="new", response_text="new resp")
        # Legacy path should be gone.
        assert not legacy_path.exists()
        # Active segment should have both entries.
        entries = load_recent_thread_entries(thread_path=thread_path, limit=10)
        queries = [e["query"] for e in entries]
        assert "old" in queries
        assert "new" in queries

    def test_load_thread_digest_returns_empty_when_absent(self, tmp_path: Path) -> None:
        """Digest sidecar returns empty when absent."""
        thread_path = tmp_path / "thread"
        assert load_thread_digest(thread_path=thread_path) == ""
        assert load_thread_digest(thread_path=None) == ""