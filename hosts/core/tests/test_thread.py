"""Tests for ``hosts.core.thread`` transcript excerpt + digest filtering.

Covers the host-command stripping behaviour introduced so that operator
UI gestures (``/pack disable github``, ``/help``, ``/tokens``, ...) are
not replayed to the model as "recent conversation" or accumulated in the
``thread_digest.md`` sidecar.
"""

from __future__ import annotations

from pathlib import Path

from hosts.core.thread import (
    _is_host_command_query,
    append_thread_entry,
    build_recent_thread_excerpt,
    load_recent_thread_entries,
)

# ── _is_host_command_query ───────────────────────────────────────────


class TestIsHostCommandQuery:
    def test_slash_prefix_commands_match(self) -> None:
        assert _is_host_command_query("/pack disable github")
        assert _is_host_command_query("/pack list")
        assert _is_command_query_returns_true("/help")
        assert _is_host_command_query("/tokens")
        assert _is_host_command_query("/settings")
        assert _is_host_command_query("/debug context")
        assert _is_host_command_query("/checkpoint")
        assert _is_host_command_query("/consolidate")
        assert _is_host_command_query("/checkpoints")
        assert _is_host_command_query("/restore my-snapshot")

    def test_bare_word_aliases_match(self) -> None:
        assert _is_host_command_query("help")
        assert _is_host_command_query("tokens")
        assert _is_host_command_query("settings")
        assert _is_host_command_query("consolidate")
        assert _is_host_command_query("checkpoints")

    def test_case_insensitive(self) -> None:
        assert _is_host_command_query("/PACK DISABLE GITHUB")
        assert _is_host_command_query("/Help")
        assert _is_host_command_query("  /pack list  ")

    def test_whitespace_stripped(self) -> None:
        assert _is_host_command_query("   /pack list   ")

    def test_normal_dialogue_does_not_match(self) -> None:
        assert not _is_host_command_query("tell me about your context")
        assert not _is_host_command_query("what is /pack?")
        assert not _is_host_command_query("explain /help to me")
        # ``/deep`` is deliberately excluded — it routes to the model and
        # the query body IS the conversation.
        assert not _is_host_command_query("/deep design a retry strategy")
        assert not _is_host_command_query("/quit")
        assert not _is_host_command_query("deep")
        assert not _is_host_command_query("")

    def test_empty_or_whitespace_only(self) -> None:
        assert not _is_host_command_query("")
        assert not _is_host_command_query("   ")


def _is_command_query_returns_true(query: str) -> bool:
    """Helper so the test list above reads symmetrically."""
    return _is_host_command_query(query)


# ── build_recent_thread_excerpt ──────────────────────────────────────


def _write_entry(
    thread_path: Path,
    *,
    user_slug: str = "tester",
    mode_id: str = "default",
    query: str,
    response_text: str = "",
) -> None:
    append_thread_entry(
        thread_path=thread_path,
        user_slug=user_slug,
        mode_id=mode_id,
        query=query,
        response_text=response_text,
    )


class TestBuildRecentThreadExcerpt:
    def test_returns_empty_when_no_thread(self, tmp_path: Path) -> None:
        assert build_recent_thread_excerpt(thread_path=None, limit=3) == ""
        assert (
            build_recent_thread_excerpt(thread_path=tmp_path / "missing.jsonl", limit=3)
            == ""
        )

    def test_returns_empty_when_all_entries_are_host_commands(
        self, tmp_path: Path
    ) -> None:
        thread_path = tmp_path / "thread.jsonl"
        _write_entry(thread_path, query="/pack list", response_text="packs: ...")
        _write_entry(thread_path, query="/help", response_text="help text")
        _write_entry(thread_path, query="/tokens", response_text="tokens: ...")
        assert build_recent_thread_excerpt(thread_path=thread_path, limit=3) == ""

    def test_strips_host_commands_keeps_dialogue(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread.jsonl"
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
        thread_path = tmp_path / "thread.jsonl"
        for i in range(5):
            _write_entry(
                thread_path,
                query=f"question {i}",
                response_text=f"answer {i}",
            )
        excerpt = build_recent_thread_excerpt(thread_path=thread_path, limit=3)
        # Header + exactly 3 dialogue lines.
        body_lines = [
            line for line in excerpt.splitlines() if line.startswith("- ")
        ]
        assert len(body_lines) == 3
        # The 3 most recent dialogue turns should be q2, q3, q4.
        assert "question 4" in excerpt
        assert "question 3" in excerpt
        assert "question 2" in excerpt
        assert "question 1" not in excerpt
        assert "question 0" not in excerpt

    def test_overread_fills_limit_after_host_command_run(
        self, tmp_path: Path
    ) -> None:
        """A run of host commands should not starve the excerpt of dialogue."""
        thread_path = tmp_path / "thread.jsonl"
        # Four host commands, then two real questions.
        _write_entry(thread_path, query="/pack list", response_text="...")
        _write_entry(thread_path, query="/pack disable github", response_text="ok")
        _write_entry(thread_path, query="/help", response_text="help")
        _write_entry(thread_path, query="/tokens", response_text="tokens")
        _write_entry(thread_path, query="real question 1", response_text="real answer 1")
        _write_entry(thread_path, query="real question 2", response_text="real answer 2")
        excerpt = build_recent_thread_excerpt(thread_path=thread_path, limit=2)
        body_lines = [
            line for line in excerpt.splitlines() if line.startswith("- ")
        ]
        assert len(body_lines) == 2
        assert "real question 1" in excerpt
        assert "real question 2" in excerpt
        assert "/pack" not in excerpt
        assert "/help" not in excerpt
        assert "/tokens" not in excerpt

    def test_chronological_order_oldest_first(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread.jsonl"
        _write_entry(thread_path, query="older", response_text="older answer")
        _write_entry(thread_path, query="newer", response_text="newer answer")
        excerpt = build_recent_thread_excerpt(thread_path=thread_path, limit=5)
        assert excerpt.index("older") < excerpt.index("newer")


# ── load_recent_thread_entries (smoke) ───────────────────────────────


class TestLoadRecentThreadEntries:
    def test_returns_last_n_entries(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread.jsonl"
        for i in range(5):
            _write_entry(thread_path, query=f"q{i}", response_text=f"a{i}")
        entries = load_recent_thread_entries(thread_path=thread_path, limit=3)
        assert len(entries) == 3
        assert entries[0]["query"] == "q2"
        assert entries[-1]["query"] == "q4"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread.jsonl"
        thread_path.write_text(
            '{"query": "good", "response_text": "ok"}\n'
            "not json\n"
            '{"query": "also good", "response_text": "ok2"}\n',
            encoding="utf-8",
        )
        entries = load_recent_thread_entries(thread_path=thread_path, limit=10)
        assert len(entries) == 2
        assert entries[0]["query"] == "good"
        assert entries[1]["query"] == "also good"