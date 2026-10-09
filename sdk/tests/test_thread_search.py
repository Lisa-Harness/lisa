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

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from sdk.thread import (
    DEFAULT_READOUT_BYTES,
    MAX_READOUT_BYTES,
    ThreadReader,
    ThreadRedactionPolicy,
    ThreadStore,
    result_object_path,
)
from sdk.thread._backend import _read_index, _sealed_segment_path, _segment_paths
from sdk.thread._results import _TOOL_CONTEXT_BYTES

# ── Helpers ───────────────────────────────────────────────────────────


def _write_active_entry(
    thread_path: Path,
    *,
    query: str = "",
    response_text: str = "",
    mode_id: str = "default",
    timestamp_utc: str | None = None,
    turn: int | None = None,
    handoff_to: str = "",
    transcript: list[dict[str, Any]] | None = None,
    task_id: str = "",
    provenance: dict[str, Any] | None = None,
    effect_receipts: list[dict[str, Any]] | None = None,
    yielded: bool = False,
    yield_question: str = "",
) -> None:
    """Append one entry to the active segment (``.current.jsonl``)."""
    paths = _segment_paths(thread_path)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "mode_id": mode_id,
        "timestamp_utc": timestamp_utc,
        "query": query,
        "response_text": response_text,
        "handoff_to": handoff_to,
    }
    if turn is not None:
        entry["turn"] = turn
    if transcript is not None:
        entry["transcript"] = transcript
    merged_provenance = dict(provenance or {})
    if task_id:
        merged_provenance["task_id"] = task_id
    if merged_provenance:
        entry["provenance"] = merged_provenance
    if effect_receipts is not None:
        entry["effect_receipts"] = effect_receipts
    if yielded:
        entry["yielded"] = True
        if yield_question:
            entry["yield_question"] = yield_question
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
    entries = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
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


class TestThreadReader:
    def test_stored_result_refs_do_not_infer_origin_time(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        original = "sensitive tool output " * 100
        payload = json.dumps(
            {"content": original, "task_id": "older", "tool_name": "read"}
        ).encode()
        ref = "r" + hashlib.sha256(payload).hexdigest()
        path = result_object_path(thread_path, ref)
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        policy = ThreadRedactionPolicy.from_patterns(["sensitive"])
        reader = ThreadReader(ThreadStore(thread_path), policy)
        page = json.loads(reader.read_content(ref, limit_chars=100))
        found = json.loads(reader.find_content(ref, "tool"))
        assert page["source"] == found["source"]
        assert page["source"]["scope"] == "stored_tool_result"
        assert page["source"]["timestamp_utc"] is None
        assert "active request are unknown" in page["source"]["interpretation"]
        assert page["content"] == policy.redact(original)[:100]
        assert found["matches"][0]["match_offset_chars"] == policy.redact(original).index("tool")
        assert path.read_bytes() == payload

    def test_reference_metadata_is_redacted_and_expandable(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            response_text="sensitive historical claim",
            timestamp_utc="sensitive timestamp " * 200,
            provenance={"source": "host-approved-sensitive"},
        )
        policy = ThreadRedactionPolicy.from_patterns(["sensitive"])
        reader = ThreadReader(ThreadStore(thread_path), policy)
        text = reader.read_content("t0.response_text")
        page = json.loads(text)
        assert "sensitive" not in text
        assert page["source"]["source_kind"] == "host"
        stamp = page["source"]["timestamp_utc"]
        assert stamp["abbreviated"] is True
        restored = json.loads(reader.read_content(stamp["ref"]))
        assert restored["content"] == policy.redact("sensitive timestamp " * 200)
        assert json.loads(reader.read_content("t0.provenance_source"))["content"] == policy.redact(
            "host-approved-sensitive"
        )

    def test_text_history_labels_latest_recorded_exchange(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="first request", response_text="first reply")
        timestamp = "2026-10-02T16:39:06+00:00"
        _write_active_entry(
            thread_path,
            query="second request",
            response_text="second reply",
            timestamp_utc=timestamp,
        )
        store = ThreadStore(thread_path)
        excerpt = store.build_recent_excerpt(limit=2)
        assert excerpt.index("first request") < excerpt.index("second request")
        assert "recorded at unknown" in excerpt
        assert timestamp in excerpt
        assert excerpt.endswith("they do not establish current task status.")
        assert "resolved past exchanges" not in excerpt
        manifest = store.build_manifest(limit=2)
        assert "Recorded History" in manifest
        assert "Recorded At (UTC)" in manifest
        assert timestamp in manifest
        assert manifest.count("latest recorded") == 1
        assert "← current" not in manifest

    @pytest.mark.parametrize("timestamp", ["2026-10-02T16:39:06+00:00", None])
    def test_page_cache_serves_repeat_reads_as_cached_hits(
        self, tmp_path: Path, timestamp: str | None
    ) -> None:
        """Thread refs are immutable; a bounded cache serves repeat reads cheaply."""
        from sdk.thread import PageCache

        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="cached request",
            response_text="cached body " * 400,
            timestamp_utc=timestamp,
        )
        reader = ThreadReader(ThreadStore(thread_path), page_cache=PageCache(max_pages=8))
        first = json.loads(reader.read_content("t0.response_text", 0, 400))
        repeat = json.loads(reader.read_content("t0.response_text", 0, 400))
        other_limit = json.loads(reader.read_content("t0.response_text", 0, 200))

        assert "cached" not in first
        assert repeat["cached"] is True
        assert repeat["content"] == first["content"]
        assert repeat["source"] == first["source"]
        assert other_limit["content"] != first["content"]
        found_first = json.loads(reader.find_content("t0.response_text", "cached"))
        found_repeat = json.loads(reader.find_content("t0.response_text", "cached"))
        assert found_repeat["cached"] is True
        assert found_repeat["matches"] == found_first["matches"]

    @pytest.mark.parametrize("timestamp", ["2026-10-02T16:39:06+00:00", None])
    def test_references_preserve_historical_source(
        self, tmp_path: Path, timestamp: str | None
    ) -> None:
        thread_path = tmp_path / "thread"
        content = "Historical ownership claim: no owned worktree. " * 10
        _write_active_entry(
            thread_path,
            query="past request",
            response_text=content,
            timestamp_utc=timestamp,
            provenance={"source": "observed"},
            transcript=[{"role": "tool", "content": "recorded tool output"}],
        )
        reader = ThreadReader(ThreadStore(thread_path))
        offset = 0
        chunks = []
        while True:
            page = json.loads(reader.read_content("t0.response_text", offset, 45))
            assert page["source"] == {
                "scope": "recorded_history",
                "interpretation": (
                    "Recorded exchanges, not the active request; "
                    "earlier instructions and diagnoses are historical."
                ),
                "turn": 0,
                "timestamp_utc": timestamp,
                "provenance_source": "observed",
                "source_kind": "assistant",
            }
            chunks.append(page["content"])
            if not page["has_more"]:
                break
            offset = page["next_offset_chars"]
        assert "".join(chunks) == content
        found = json.loads(reader.find_content("t0.response_text", "ownership"))
        assert found["source"] == page["source"]
        assert found["matches"][0]["match_offset_chars"] == content.index("ownership")
        assert json.loads(reader.read_content("t0.query"))["source"]["source_kind"] == "user"
        assert json.loads(reader.read_content("t0.m0.content"))["source"]["source_kind"] == "tool"
        assert (
            json.loads(reader.read_content("t0.timestamp_utc"))["source"]["source_kind"]
            == "entry_metadata"
        )

    @pytest.mark.parametrize("budget", [1024, 2048, 8192])
    def test_task_view_labels_recorded_selection(self, tmp_path: Path, budget: int) -> None:
        thread_path = tmp_path / "thread"
        old_time = "2026-10-01T00:00:00+00:00"
        latest_time = "2026-10-02T00:00:00+00:00"
        _write_active_entry(thread_path, query="old", task_id="old", timestamp_utc=old_time)
        _write_active_entry(
            thread_path,
            query="previous request " * 2000,
            task_id="previous",
            timestamp_utc=latest_time,
        )
        reader = ThreadReader(ThreadStore(thread_path))
        for task_id, selection, timestamp in (
            ("", "latest_recorded", latest_time),
            ("old", "explicit_task_id", old_time),
            ("live-not-yet-recorded", "explicit_task_id", None),
        ):
            text = reader.task_view(task_id, budget_bytes=budget)
            assert len(text.encode("utf-8")) <= budget
            view = json.loads(text)
            assert view["scope"] == "recorded_history"
            assert view["selection"] == selection
            assert view["status_basis"] == "recorded_observation_not_live_state"
            assert view["as_of_utc"] == timestamp
            assert "not the active request" in view["interpretation"]

    @pytest.mark.parametrize("timestamp", ["2026-10-02T16:39:06+00:00", None])
    def test_archive_views_label_recorded_entries(
        self, tmp_path: Path, timestamp: str | None
    ) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="historical needle",
            timestamp_utc=timestamp,
            provenance={"source": "observed"},
        )
        reader = ThreadReader(ThreadStore(thread_path))
        for text in (reader.read_range(0, 0), reader.read_recent(), reader.search("needle")):
            page = json.loads(text)
            assert page["scope"] == "recorded_history"
            assert "not the active request" in page["interpretation"]
            assert page["entries"][0]["timestamp_utc"] == timestamp
            assert page["entries"][0]["provenance_source"] == "observed"

        policy = ThreadRedactionPolicy.from_patterns(["historical", "2026", "observed"])
        redacted = json.loads(ThreadReader(ThreadStore(thread_path), policy).read_recent())
        assert "observed" not in redacted["entries"][0]["provenance_source"]
        assert "2026" not in str(redacted["entries"][0]["timestamp_utc"])
        assert ThreadStore(thread_path).load_recent(limit=1)[0]["timestamp_utc"] == timestamp

    @pytest.mark.parametrize("nested", [True, False])
    def test_search_does_not_match_an_excluded_call_in_a_batch(
        self, tmp_path: Path, nested: bool
    ) -> None:
        thread_path = tmp_path / "thread"
        functions = [
            {"name": "read", "arguments": '{"path":"src/needle.py"}'},
            {"name": "exec", "arguments": '{"command":"forbidden unique evidence"}'},
        ]
        calls = [
            {"id": str(index), **({"function": function} if nested else function)}
            for index, function in enumerate(functions)
        ]
        _write_active_entry(
            thread_path,
            query="batch",
            transcript=[
                {
                    "role": "assistant",
                    "tool_calls": calls,
                }
            ],
        )
        reader = ThreadReader(ThreadStore(thread_path))
        assert (
            json.loads(reader.search("forbidden unique evidence", tool_name="read"))["entries"]
            == []
        )
        match = json.loads(reader.search("needle", tool_name="read"))["entries"][0]["matches"][0]
        assert match["ref"] == "t0.m0.c0.arguments"
        assert json.loads(reader.read_content(match["ref"]))["content"] == functions[0]["arguments"]

    def test_search_filters_actual_task_identity_and_turn_window(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        for task in ("task-a", "task-b", "task-a"):
            _write_active_entry(thread_path, query="shared needle", task_id=task)
        reader = ThreadReader(ThreadStore(thread_path))
        entries = json.loads(reader.search("needle", task_id="task-b", start_turn=1, end_turn=1))[
            "entries"
        ]
        assert [entry["turn"] for entry in entries] == [1]
        assert entries[0]["task_id"] == "task-b"
        assert json.loads(reader.read_content("t1.task_id"))["content"] == "task-b"
        assert json.loads(reader.search("needle", task_id="not-a-task"))["entries"] == []

    def test_find_content_pages_literal_matches_after_redaction(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        original = "secret " + ("\U0001f642 prefix a.*b suffix " * 5)
        _write_active_entry(thread_path, query=original)
        policy = ThreadRedactionPolicy.from_patterns(("secret",))
        reader = ThreadReader(ThreadStore(thread_path), policy)
        offset = 0
        positions = []
        while True:
            text = reader.find_content("t0.query", "a.*b", offset, 2)
            assert len(text.encode("utf-8")) <= DEFAULT_READOUT_BYTES
            page = json.loads(text)
            for match in page["matches"]:
                positions.append(match["match_offset_chars"])
                assert "a.*b" in match["excerpt"]
                expanded = json.loads(reader.read_content("t0.query", match["offset_chars"]))
                assert "a.*b" in expanded["content"]
            if not page["has_more"]:
                break
            assert page["next_offset_chars"] > offset
            offset = page["next_offset_chars"]
        assert len(positions) == len(set(positions)) == 5
        permitted = policy.redact(original)
        assert all(permitted[position : position + 4] == "a.*b" for position in positions)
        assert json.loads(reader.find_content("t0.query", "secret"))["matches"] == []

    def test_task_view_summarizes_scoped_state_with_expandable_references(
        self, tmp_path: Path
    ) -> None:
        thread_path = tmp_path / "thread"
        ref = "r" + "a" * 64
        _write_active_entry(thread_path, query="unrelated", task_id="task-other")
        _write_active_entry(
            thread_path,
            query="/approve xyz",
            response_text="ignored host command",
            task_id="task-1",
        )
        _write_active_entry(
            thread_path,
            query="refactor the parser",
            response_text="started work",
            mode_id="default",
            handoff_to="coding",
            task_id="task-1",
            effect_receipts=[
                {"capability_id": "files.write", "effect": "write", "artifact_id": "parser"},
                {"capability_id": "files.write", "effect": "write", "artifact_id": "lexer"},
            ],
            transcript=[
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": json.dumps(
                        {
                            "ref": ref,
                            "abbreviated": True,
                        }
                    ),
                },
            ],
        )
        _write_active_entry(
            thread_path,
            query="keep going",
            response_text="paused for approval",
            mode_id="coding",
            task_id="task-1",
            provenance={"pending_approval_request_id": "req-9"},
        )
        reader = ThreadReader(ThreadStore(thread_path))
        view = json.loads(reader.task_view("task-1"))
        assert view["task_id"] == "task-1"
        assert view["turns"] == 3
        assert view["status"] == "awaiting_approval"
        assert view["blocker"] == {
            "kind": "approval",
            "request_id": "req-9",
            "request_id_ref": "t3.pending_approval_request_id",
        }
        assert view["objective"]["text"] == "refactor the parser"
        assert view["objective_ref"] == "t2.query"
        assert view["modes"] == ["default", "coding"]
        assert view["effects"] == [
            {
                "capability_id": "files.write",
                "effect": "write",
                "count": 2,
                "artifact_ids": ["parser", "lexer"],
            }
        ]
        assert view["evidence"] == ["t2.m0.content"]
        assert json.loads(reader.read_content(view["evidence"][0]))["content"] == json.dumps(
            {"ref": ref, "abbreviated": True}
        )
        assert view["latest"]["text"] == "paused for approval"
        assert (
            json.loads(reader.read_content(view["objective_ref"]))["content"]
            == "refactor the parser"
        )

    def test_task_view_defaults_to_latest_task_and_bounds_output(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        refs = ["r" + f"{index:064x}" for index in range(40)]
        _write_active_entry(thread_path, query="old task", task_id="task-old")
        _write_active_entry(
            thread_path,
            query="current objective " + "x" * 2000,
            response_text="y" * 2000,
            task_id="task-new",
            transcript=[
                {
                    "role": "tool",
                    "tool_call_id": f"c{i}",
                    "content": json.dumps(
                        {
                            "ref": ref,
                            "abbreviated": True,
                        }
                    ),
                }
                for i, ref in enumerate(refs)
            ],
        )
        view_text = ThreadReader(ThreadStore(thread_path)).task_view(budget_bytes=2048)
        assert len(view_text.encode("utf-8")) <= 2048
        view = json.loads(view_text)
        assert view["task_id"] == "task-new"
        assert view["objective"]["abbreviated"] is True
        assert view["evidence_truncated"] is True
        assert len(view["evidence"]) < len(refs)

    @pytest.mark.parametrize("budget", [1024, 2048, 8192])
    @pytest.mark.parametrize("blocker", ["question", "approval"])
    @pytest.mark.parametrize(
        "large_text",
        ["x" * 20000, "\x01" * 20000, "\U0001f642" * 20000],
        ids=["ascii", "escaped", "unicode"],
    )
    def test_task_view_bounds_all_metadata(
        self, tmp_path: Path, budget: int, large_text: str, blocker: str
    ) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query=large_text,
            response_text=large_text,
            mode_id=large_text,
            task_id=large_text,
            provenance={"pending_approval_request_id": large_text} if blocker == "approval" else {},
            effect_receipts=[
                {"capability_id": large_text, "effect": "write", "artifact_id": large_text}
            ],
            yielded=blocker == "question",
            yield_question=large_text,
        )
        reader = ThreadReader(ThreadStore(thread_path))
        text = reader.task_view(budget_bytes=budget)
        assert len(text.encode("utf-8")) <= budget
        view = json.loads(text)
        assert view["status"] == ("awaiting_user" if blocker == "question" else "awaiting_approval")
        assert view["blocker"]["kind"] == blocker
        field = "question" if blocker == "question" else "request_id"
        assert json.loads(reader.read_content(view["blocker"][f"{field}_ref"]))[
            "total_chars"
        ] == len(large_text)
        assert view["task_id"]["abbreviated"] is True
        assert view["task_id_ref"] == "t0.task_id"
        assert view["objective_ref"] == "t0.query"
        assert view["latest_ref"] == "t0.response_text"
        assert json.loads(reader.read_content(view["objective_ref"]))["total_chars"] == len(
            large_text
        )

    def test_task_view_redacts_all_metadata_without_changing_storage(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="secret objective",
            response_text="secret response",
            mode_id="secret mode",
            task_id="secret task",
            provenance={"pending_approval_request_id": "secret approval"},
            effect_receipts=[
                {
                    "capability_id": "secret tool",
                    "effect": "secret effect",
                    "artifact_id": "secret artifact",
                }
            ],
        )
        policy = ThreadRedactionPolicy.from_patterns(("secret",))
        store = ThreadStore(thread_path)
        reader = ThreadReader(store, policy)
        assert "secret" not in reader.task_view("secret task")
        assert "secret" not in reader.task_view("secret missing task")
        _write_active_entry(
            thread_path,
            query="continue",
            task_id="secret task",
            yielded=True,
            yield_question="secret question",
        )
        assert "secret" not in reader.task_view("secret task")
        assert store.load_range(start_turn=0, end_turn=0)[0]["query"] == "secret objective"

    def test_task_view_distinguishes_recorded_effects_for_one_capability(
        self, tmp_path: Path
    ) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="update",
            task_id="task-1",
            effect_receipts=[
                {"capability_id": "files.tool", "effect": "read", "artifact_id": ""},
                {"capability_id": "files.tool", "effect": "write", "artifact_id": "change-1"},
                {"capability_id": "files.tool", "effect": "write", "artifact_id": "change-1"},
            ],
        )
        view = json.loads(ThreadReader(ThreadStore(thread_path)).task_view())
        assert view["effects"] == [
            {"capability_id": "files.tool", "effect": "read", "count": 1, "artifact_ids": []},
            {
                "capability_id": "files.tool",
                "effect": "write",
                "count": 2,
                "artifact_ids": ["change-1"],
            },
        ]

    def test_task_view_keeps_latest_task_across_host_commands_and_reports_window(
        self, tmp_path: Path
    ) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="objective", task_id="task-1")
        _write_active_entry(thread_path, query="continue", task_id="task-1")
        _write_active_entry(thread_path, query="/debug")
        reader = ThreadReader(ThreadStore(thread_path))
        complete = json.loads(reader.task_view())
        assert complete["task_id"] == "task-1"
        assert complete["objective"]["text"] == "objective"
        assert complete["history_may_be_incomplete"] is False
        partial = json.loads(reader.task_view(scan_limit=2))
        assert partial["history_may_be_incomplete"] is True
        assert partial["objective"]["text"] == "continue"

    @pytest.mark.parametrize("existing", [False, True])
    def test_task_view_missing_task_stays_bounded(self, tmp_path: Path, existing: bool) -> None:
        thread_path = tmp_path / "thread"
        if existing:
            _write_active_entry(thread_path, query="other task", task_id="other")
        reader = ThreadReader(ThreadStore(thread_path))
        text = reader.task_view("unknown-" * 5000, budget_bytes=1024)
        assert len(text.encode("utf-8")) <= 1024
        assert json.loads(text)["status"] == "empty"
        with pytest.raises(ValueError, match="positive"):
            reader.task_view(budget_bytes=0)

    def test_search_finds_filtered_tool_result_with_expandable_match(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        content = "padding " * 1000 + "valid check_ids: unit, lint" + " tail" * 1000
        _write_active_entry(
            thread_path,
            query="run checks",
            transcript=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "check",
                            "function": {
                                "name": "coding_run_check",
                                "arguments": '{"path":"src/main.py"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "check", "content": content},
            ],
        )
        reader = ThreadReader(ThreadStore(thread_path))
        text = reader.search("valid check_ids", tool_name="coding_run_check", path="src/main.py")
        page = json.loads(text)
        assert len(text.encode("utf-8")) <= DEFAULT_READOUT_BYTES
        match = page["entries"][0]["matches"][0]
        assert match["ref"] == "t0.m1.content"
        assert "valid check_ids" in match["excerpt"]
        expanded = json.loads(reader.read_content(match["ref"], match["offset_chars"], 400))
        assert "valid check_ids" in expanded["content"]

    @pytest.mark.parametrize("kind", ["range", "recent", "search"])
    def test_pages_are_bounded_and_snapshot_stable(self, tmp_path: Path, kind: str) -> None:
        thread_path = tmp_path / "thread"
        control_text = "\x01" * 10000
        for turn in range(30):
            _write_active_entry(
                thread_path,
                query=control_text,
                response_text=control_text,
                mode_id=control_text,
                handoff_to=control_text,
                turn=turn + 999,
            )
        reader = ThreadReader(ThreadStore(thread_path))
        seen: list[int] = []
        cursor = None
        cursors: set[str] = set()
        while True:
            if kind == "range":
                text = reader.read_range(0, 999, cursor=cursor)
            elif kind == "recent":
                text = reader.read_recent(30, cursor=cursor)
            else:
                text = reader.search("", 20, cursor=cursor)
            assert len(text.encode("utf-8")) <= DEFAULT_READOUT_BYTES
            page = json.loads(text)
            assert page["entries"]
            seen.extend(entry["turn"] for entry in page["entries"])
            if cursor is None:
                assert page["has_more"]
                _write_active_entry(thread_path, query="later append", response_text="new")
            if not page["has_more"]:
                break
            cursor = page["next_cursor"]
            assert cursor not in cursors
            cursors.add(cursor)
        expected = list(range(29, 9, -1)) if kind == "search" else list(range(30))
        assert seen == expected

    @pytest.mark.parametrize("kind", ["range", "recent"])
    def test_transcript_can_continue_inside_one_turn(self, tmp_path: Path, kind: str) -> None:
        thread_path = tmp_path / "thread"
        transcript = [
            {"role": "tool", "tool_call_id": f"call_{index}", "content": str(index) * 1000}
            for index in range(200)
        ]
        _write_active_entry(thread_path, query="tools", transcript=transcript)
        reader = ThreadReader(ThreadStore(thread_path))
        refs: list[str] = []
        cursor = None
        positions: set[str] = set()
        while True:
            if kind == "range":
                text = reader.read_range(0, 0, include_transcript=True, cursor=cursor)
            else:
                text = reader.read_recent(include_transcript=True, cursor=cursor)
            assert len(text.encode("utf-8")) <= DEFAULT_READOUT_BYTES
            page = json.loads(text)
            for entry in page["entries"]:
                refs.extend(message["ref"] for message in entry["transcript"])
            if not page["has_more"]:
                break
            cursor = page["next_cursor"]
            assert cursor not in positions
            positions.add(cursor)
        assert positions
        assert refs == [f"t0.m{index}" for index in range(len(transcript))]

    @pytest.mark.parametrize("fragment", ["\U0001f642", "\x00", '"\\', "e\u0301"])
    def test_expansion_reconstructs_unicode_and_escaped_content(
        self, tmp_path: Path, fragment: str
    ) -> None:
        thread_path = tmp_path / "thread"
        original = fragment * 10000
        _write_active_entry(
            thread_path,
            query="read",
            transcript=[{"role": "tool", "content": original}],
        )
        paths = _segment_paths(thread_path)
        raw_before = paths.current.read_bytes()
        reader = ThreadReader(ThreadStore(thread_path))
        view = json.loads(reader.read_range(0, 0, include_transcript=True))
        stub = view["entries"][0]["transcript"][0]["content"]
        assert stub["abbreviated"]
        assert stub["total_chars"] == len(original)
        assert stub["omitted_chars"] >= 0
        offset = 0
        pieces: list[str] = []
        while True:
            text = reader.read_content(stub["ref"], offset, 1000000, budget_bytes=1000000)
            assert len(text.encode("utf-8")) <= MAX_READOUT_BYTES
            page = json.loads(text)
            assert page["offset_chars"] == offset
            assert page["content"]
            pieces.append(page["content"])
            if not page["has_more"]:
                break
            assert page["next_offset_chars"] > offset
            offset = page["next_offset_chars"]
        assert "".join(pieces) == original
        assert paths.current.read_bytes() == raw_before

    def test_redaction_precedes_paging_and_covers_both_tool_call_shapes(
        self, tmp_path: Path
    ) -> None:
        thread_path = tmp_path / "thread"
        sensitive = "before secret-value after"
        calls = [
            {"id": "nested", "function": {"name": "read", "arguments": sensitive}},
            {"id": "flat", "name": "read", "arguments": sensitive},
        ]
        transcript = [
            {"role": "tool", "content": sensitive},
            {"role": "assistant", "tool_calls": calls},
        ]
        _write_active_entry(
            thread_path,
            query=sensitive,
            response_text=sensitive,
            transcript=transcript,
        )
        policy = ThreadRedactionPolicy.from_patterns(("secret-value",))
        reader = ThreadReader(ThreadStore(thread_path), policy)
        assert "secret-value" not in reader.read_range(0, 0, include_transcript=True)
        assert "secret-value" not in reader.search("secret-value")
        refs = ["t0.query", "t0.response_text", "t0.m0.content", "t0.m1.tool_calls"]
        for ref in refs:
            offset = 0
            restored: list[str] = []
            while True:
                page = json.loads(reader.read_content(ref, offset, 5))
                restored.append(page["content"])
                if not page["has_more"]:
                    break
                offset = page["next_offset_chars"]
            text = "".join(restored)
            assert "secret-value" not in text
            if ref.endswith("tool_calls"):
                decoded = json.loads(text)
                assert decoded[0]["function"]["arguments"] == policy.redact(sensitive)
                assert decoded[1]["arguments"] == policy.redact(sensitive)
            else:
                assert text == policy.redact(sensitive)
        raw = ThreadStore(thread_path).load_range(start_turn=0, end_turn=0)
        assert raw[0]["query"] == sensitive

    @pytest.mark.skipif(not _zstd_available(), reason="zstandard not installed")
    def test_references_and_cursors_survive_sealing_and_reader_restart(
        self, tmp_path: Path
    ) -> None:
        thread_path = tmp_path / "thread"
        original = "result" * 1000
        transcript = [{"role": "tool", "content": original} for _ in range(100)]
        _write_active_entry(thread_path, query="old", response_text=original, transcript=transcript)
        reader = ThreadReader(ThreadStore(thread_path))
        first = json.loads(reader.read_range(0, 999, include_transcript=True))
        assert first["has_more"]
        first_refs = [message["ref"] for message in first["entries"][0]["transcript"]]
        _seal_active(thread_path)
        _write_active_entry(thread_path, query="new", response_text="active tail")
        restarted = ThreadReader(ThreadStore(thread_path))
        content = json.loads(restarted.read_content("t0.response_text", limit_chars=10000))
        assert content["content"] == original
        assert json.loads(restarted.read_content("t1.query"))["content"] == "new"
        cursor = first["next_cursor"]
        refs = first_refs
        while cursor is not None:
            page = json.loads(restarted.read_range(0, 999, include_transcript=True, cursor=cursor))
            assert all(entry["turn"] == 0 for entry in page["entries"])
            refs.extend(
                message["ref"] for entry in page["entries"] for message in entry["transcript"]
            )
            cursor = page["next_cursor"]
        assert refs == [f"t0.m{index}" for index in range(100)]

    @pytest.mark.parametrize(
        "ref",
        [
            "../other",
            "t999.query",
            "t0.m999.content",
            "t0.reasoning_text",
            "t0.turn_input",
            "t0.m0.reasoning_content",
            "t0",
            "t0.query/../../other",
        ],
    )
    def test_invalid_or_private_references_are_rejected(self, tmp_path: Path, ref: str) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(
            thread_path,
            query="visible",
            transcript=[
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "private",
                }
            ],
        )
        reader = ThreadReader(ThreadStore(thread_path))
        with pytest.raises(ValueError):
            reader.read_content(ref)
        message = json.loads(json.loads(reader.read_content("t0.m0"))["content"])
        assert message == {"role": "assistant", "content": "answer"}

    def test_cursor_and_offset_errors_are_explicit(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        _write_active_entry(thread_path, query="visible")
        reader = ThreadReader(ThreadStore(thread_path))
        with pytest.raises(ValueError, match="cursor"):
            reader.read_range(0, 0, cursor="invalid")
        with pytest.raises(ValueError, match="Transcript cursor"):
            reader.read_range(0, 0, cursor="range.0.0.0.1")
        with pytest.raises(ValueError, match="this search"):
            reader.search("other", cursor="search.0.0.0000000000000000")
        with pytest.raises(ValueError, match="content length"):
            reader.read_content("t0.query", offset_chars=999)
        with pytest.raises(ValueError, match="Invalid reference"):
            reader.read_content("t0.query", offset_chars=-1)
        with pytest.raises(ValueError, match="Invalid reference"):
            reader.read_content("t0.query", limit_chars=0)
        with pytest.raises(ValueError, match="budget_bytes"):
            reader.read_range(0, 0, budget_bytes=0)

    def test_empty_views_and_terminal_content_offset(self, tmp_path: Path) -> None:
        thread_path = tmp_path / "thread"
        reader = ThreadReader(ThreadStore(thread_path))
        empty = {
            "scope": "recorded_history",
            "interpretation": (
                "Recorded exchanges, not the active request; "
                "earlier instructions and diagnoses are historical."
            ),
            "entries": [],
            "next_cursor": None,
            "has_more": False,
        }
        assert json.loads(reader.read_range(0, 999)) == empty
        assert json.loads(reader.read_recent()) == empty
        assert json.loads(reader.search("")) == empty
        _write_active_entry(thread_path, query="done")
        terminal = json.loads(reader.read_content("t0.query", offset_chars=4))
        assert terminal["content"] == ""
        assert terminal["next_offset_chars"] is None
        assert terminal["has_more"] is False


# ── project_tool_messages() — retrieval pages under pressure ─────────


class TestProjectionRetrievalPages:
    @staticmethod
    def _envelope(ref: str, content: str) -> str:
        return json.dumps(
            {"ref": ref, "content": content, "has_more": False},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def test_over_threshold_page_pointers_without_pressure(self) -> None:
        """An already-stored page too large for inline residency pointers
        even without pressure (Lisa-Harness/lisa#5).

        The page's ``ref`` addresses stored content; persisting a copy of a
        page of an already-stored value creates a second address for the
        same bytes and a stub-of-a-page the model must expand twice. The
        window that bit: pages over ``_MAX_RETRIEVAL_BYTES`` (32KB) but
        under the turn ceiling (64KB) — too big to stay inline, not enough
        to trigger pressure — fell through to persist+preview. The pointer
        must not depend on turn-level pressure: whenever a complete page
        (``has_more`` false) would not be kept inline, it pointers.
        """
        from sdk.thread._results import project_tool_messages

        original = "stored worktree state " * 2100  # ~44KB page: in the window
        stored_ref = "r" + f"{3:064x}"

        def load(ref: str) -> dict[str, Any]:
            return {"content": original, "task_id": "task", "tool_name": "coding_status"}

        persisted: dict[str, dict[str, Any]] = {}

        def persist(result: dict[str, Any]) -> str:
            digest = hashlib.sha256(json.dumps(result).encode()).hexdigest()
            ref = "r" + digest
            persisted.setdefault(ref, result)
            return ref

        page = self._envelope(stored_ref, original)
        page_bytes = len(page.encode("utf-8"))
        assert 32768 < page_bytes < 65536, "page must sit in the no-pointer window"
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "p", "function": {"name": "thread_read_content"}}],
            },
            {"role": "tool", "tool_call_id": "p", "name": "thread_read_content", "content": page},
        ]
        projected = project_tool_messages(messages, "task", persist, None, load)
        pointer = json.loads(projected[1]["content"])
        assert pointer["ref"] == stored_ref
        assert pointer["page_omitted"] is True
        assert not persisted, "no fresh result object may be persisted for a stored page"

    def test_page_without_room_pointers_without_persisting(self, tmp_path: Path) -> None:
        """An older page that newer results crowd out pointers to its stored value."""
        from sdk.thread._results import project_tool_messages, result_object_path

        original = "stored worktree state " * 400
        stored_ref = "r" + f"{1:064x}"
        path = result_object_path(tmp_path, stored_ref)
        path.parent.mkdir(parents=True)
        path.write_bytes(
            json.dumps(
                {"content": original, "task_id": "task", "tool_name": "coding_status"}
            ).encode()
        )
        persisted: dict[str, dict[str, Any]] = {}

        def persist(result: dict[str, Any]) -> str:
            digest = hashlib.sha256(json.dumps(result).encode()).hexdigest()
            ref = "r" + digest
            persisted.setdefault(ref, result)
            return ref

        def load(ref: str) -> dict[str, Any]:
            if ref == stored_ref:
                return {"content": original, "task_id": "task", "tool_name": "coding_status"}
            return persisted[ref]

        page = self._envelope(stored_ref, original[:7500])
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "p", "function": {"name": "thread_read_content"}}],
            },
            {"role": "tool", "tool_call_id": "p", "name": "thread_read_content", "content": page},
        ]
        for index in range(8):  # newer work results claim the resident budget
            messages += [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": f"f{index}", "function": {"name": "big_tool"}}],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"f{index}",
                    "name": "big_tool",
                    "content": "x" * 8000,
                },
            ]
        projected = project_tool_messages(messages, "task", persist, None, load)
        pointer = json.loads(projected[1]["content"])
        assert pointer["ref"] == stored_ref
        assert pointer["page_omitted"] is True
        assert "expand" in pointer
        assert not any(
            result.get("tool_name") == "thread_read_content" for result in persisted.values()
        )

    def test_pointer_projection_is_idempotent(self, tmp_path: Path) -> None:
        from sdk.thread._results import project_tool_messages

        original = "stored worktree state " * 200
        stored_ref = "r" + f"{2:064x}"

        def load(ref: str) -> dict[str, Any]:
            return {"content": original, "task_id": "task", "tool_name": "coding_status"}

        persisted: dict[str, dict[str, Any]] = {}

        def persist(result: dict[str, Any]) -> str:
            digest = hashlib.sha256(json.dumps(result).encode()).hexdigest()
            ref = "r" + digest
            persisted.setdefault(ref, result)
            return ref

        page = self._envelope(stored_ref, original[:50])
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "p", "function": {"name": "thread_read_content"}}],
            },
            {"role": "tool", "tool_call_id": "p", "name": "thread_read_content", "content": page},
        ]
        first = project_tool_messages(messages, "task", persist, None, load)
        second = project_tool_messages(first, "task", persist, None, load)
        assert first[1]["content"] == second[1]["content"]
        assert not persisted

    def test_no_pressure_projection_is_unchanged(self, tmp_path: Path) -> None:
        from sdk.thread._results import project_tool_messages

        original = "stored worktree state " * 200
        stored_ref = "r" + f"{3:064x}"

        def load(ref: str) -> dict[str, Any]:
            return {"content": original, "task_id": "task", "tool_name": "coding_status"}

        page = self._envelope(stored_ref, original[:50])
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "p", "function": {"name": "thread_read_content"}}],
            },
            {"role": "tool", "tool_call_id": "p", "name": "thread_read_content", "content": page},
        ]
        projected = project_tool_messages(
            messages, "task", lambda result: "r" + "0" * 64, None, load
        )
        assert projected[1]["content"] == page


# ── project_tool_messages() — newest-first resident allocation ────────


class TestProjectionNewestFirstAllocation:
    """Newest results stay verbatim; only what cannot stay resident collapses.

    Regression shape for the 2026-10-08 turn-618 non-convergence: once raw
    turn history passed the 64KB budget, projection treated the whole turn
    as "under pressure" forever — even after eviction left ~12KB resident —
    so every new read became a ~400-char preview and every
    ``thread_read_content`` expansion was pointered back to its own ref.
    The model re-read the same file in shrinking windows (500→5 lines) and
    exhausted the 20-iteration cap without writing anything.
    """

    @staticmethod
    def _call(call_id: str, name: str, content: str) -> list[dict[str, Any]]:
        return [
            {"role": "assistant", "tool_calls": [{"id": call_id, "function": {"name": name}}]},
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": content},
        ]

    @staticmethod
    def _store() -> tuple[dict[str, dict[str, Any]], Any]:
        persisted: dict[str, dict[str, Any]] = {}

        def persist(result: dict[str, Any]) -> str:
            ref = "r" + hashlib.sha256(json.dumps(result).encode()).hexdigest()
            persisted.setdefault(ref, result)
            return ref

        return persisted, persist

    def _turn(self, stored_ref: str, stored: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for index in range(40):  # ~280KB of earlier medium reads
            messages += self._call(
                f"read-{index}", "lisa-coding.coding_read_file", f"read {index}: " + "x" * 7000
            )
        messages += self._call("small", "lisa-coding.coding_read_file", "func main() {}\n" * 40)
        page = json.dumps(
            {"ref": stored_ref, "content": stored[:700], "has_more": False},
            separators=(",", ":"),
        )
        messages += self._call("page", "lisa-thread.thread_read_content", page)
        return messages

    @staticmethod
    def _tool_bytes(projected: list[dict[str, Any]]) -> int:
        return sum(len(m["content"].encode()) for m in projected if m.get("role") == "tool")

    def test_new_results_stay_verbatim_after_history_exceeds_budget(self) -> None:
        from sdk.thread._results import project_tool_messages

        persisted, persist = self._store()
        stored = "package main\n" * 100
        stored_ref = persist({"content": stored, "task_id": "task", "tool_name": "read"})
        messages = self._turn(stored_ref, stored)
        projected = project_tool_messages(messages, "task", persist, load=persisted.__getitem__)

        by_id = {m.get("tool_call_id"): m["content"] for m in projected if m.get("role") == "tool"}
        assert by_id["small"] == messages[-3]["content"], "fresh small read was abbreviated"
        assert by_id["page"] == messages[-1]["content"], (
            "latest expansion page was pointered back to its own ref"
        )
        verbatim_reads = [
            call_id
            for call_id, content in by_id.items()
            if str(call_id).startswith("read-") and content.startswith("read ")
        ]
        assert len(verbatim_reads) >= 6, (
            f"only {len(verbatim_reads)} recent reads kept verbatim; resident budget is wasted"
        )
        assert self._tool_bytes(projected) <= _TOOL_CONTEXT_BYTES
        assert any("inventory_ref" in m.get("content", "") for m in projected)

    def test_newest_first_projection_is_idempotent_and_lossless(self) -> None:
        from sdk.thread._results import project_tool_messages

        persisted, persist = self._store()
        stored = "package main\n" * 100
        stored_ref = persist({"content": stored, "task_id": "task", "tool_name": "read"})
        messages = self._turn(stored_ref, stored)
        projected = project_tool_messages(messages, "task", persist, load=persisted.__getitem__)
        again = project_tool_messages(projected, "task", persist, load=persisted.__getitem__)
        assert again == projected
        marker = next(m for m in projected if "inventory_ref" in m.get("content", ""))
        inventory = json.loads(persisted[json.loads(marker["content"])["inventory_ref"]]["content"])
        first_ref = inventory[0]["results"][0]["ref"]
        assert persisted[first_ref]["content"] == messages[1]["content"]


# ── project_tool_messages() — fixed preview scale ────────────────────


class TestProjectionPreviewScale:
    """Previews keep one 600+300 scale; overage evicts oldest, never shrinks.

    Regression shape for the 2026-10-05 turn-524 thrash: a turn-wide
    overage used to shrink every preview toward 120+60 chars, so
    re-expanding evidence re-added bytes and deepened the collapse.
    """

    @staticmethod
    def _messages(tool_bytes: int) -> list[dict[str, Any]]:
        big = "x" * (tool_bytes // 2)
        return [
            {
                "role": "assistant",
                "tool_calls": [{"id": "a", "function": {"name": "tool_a"}}],
            },
            {"role": "tool", "tool_call_id": "a", "name": "tool_a", "content": big},
            {
                "role": "assistant",
                "tool_calls": [{"id": "b", "function": {"name": "tool_b"}}],
            },
            {"role": "tool", "tool_call_id": "b", "name": "tool_b", "content": big},
        ]

    @staticmethod
    def _preview_heads(projected: list[dict[str, Any]]) -> list[int]:
        heads: list[int] = []
        for message in projected:
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str):
                continue
            try:
                preview = json.loads(content)
            except (TypeError, ValueError):
                continue
            if preview.get("abbreviated") is True:
                heads.append(len(preview["head"]))
        return heads

    @pytest.mark.parametrize("overage", [0, 2, 200_000])
    def test_previews_keep_full_scale_at_any_overage(self, overage: int) -> None:
        from sdk.thread._results import project_tool_messages

        projected = project_tool_messages(
            self._messages(_TOOL_CONTEXT_BYTES + overage), "task", lambda r: "r" + "0" * 64
        )
        heads = self._preview_heads(projected)
        assert heads and all(head == 600 for head in heads)


# ── project_tool_messages() — small work results inline under pressure ───


class TestProjectionSmallWorkResultInline:
    """Abbreviation must not fire when it would omit nothing.

    Regression shape for the 2026-10-06 turn-557/558 micro-window loop:
    under pressure every work-tool result was persisted and previewed,
    including results smaller than the active head+tail scale (whose
    ``omitted_chars`` was zero). The preview envelope plus JSON escaping
    cost more wire bytes than the content it replaced, so the projector
    inflated the context it bounds and coerced the model into re-reading
    files through ever-smaller line windows (31 of 50 reads at 2-3 lines,
    ~7.5s each, before the 50-read cap).
    """

    @staticmethod
    def _pressure_messages(small_result: str) -> list[dict[str, Any]]:
        filler = "x" * (_TOOL_CONTEXT_BYTES + 4096)
        return [
            {
                "role": "assistant",
                "tool_calls": [{"id": "big", "function": {"name": "coding_read_file"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "big",
                "name": "coding_read_file",
                "content": filler,
            },
            {
                "role": "assistant",
                "tool_calls": [{"id": "small", "function": {"name": "coding_read_file"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "small",
                "name": "coding_read_file",
                "content": small_result,
            },
        ]

    @staticmethod
    def _persisted(result: dict[str, Any]) -> str:
        return "r" + hashlib.sha256(json.dumps(result).encode()).hexdigest()

    def test_result_fitting_the_pressure_scale_stays_inline(self) -> None:
        """A work result no larger than head+tail is visible whole under pressure."""
        from sdk.thread._results import project_tool_messages

        # At this overage the graduated scale is 576+288 visible chars
        # (fraction 4736/98304): a result that fits must not be previewed.
        small = "line of evidence " * 40  # 640 chars < 864
        projected = project_tool_messages(self._pressure_messages(small), "task", self._persisted)
        small_message = next(m for m in projected if m.get("tool_call_id") == "small")
        assert small_message["content"] == small, (
            "work result fitting the preview scale was abbreviated without omission"
        )
        big_message = next(m for m in projected if m.get("tool_call_id") == "big")
        assert json.loads(big_message["content"])["abbreviated"] is True

    def test_result_over_the_inline_cap_is_abbreviated_with_omission(self) -> None:
        """A work result larger than the inline cap still previews (and omits)."""
        from sdk.thread._results import project_tool_messages

        large = "line of evidence " * 600  # 10200 chars > 8192
        projected = project_tool_messages(self._pressure_messages(large), "task", self._persisted)
        small_message = next(m for m in projected if m.get("tool_call_id") == "small")
        preview = json.loads(small_message["content"])
        assert preview["abbreviated"] is True
        assert preview["omitted_chars"] > 0

    def test_inline_small_results_remain_inventory_eligible(self) -> None:
        """Over-budget turns still evict inline results through the inventory."""
        from sdk.thread._results import _TOOL_CONTEXT_BYTES, project_tool_messages

        persisted: dict[str, dict[str, Any]] = {}

        def persist(result: dict[str, Any]) -> str:
            ref = "r" + hashlib.sha256(json.dumps(result).encode()).hexdigest()
            persisted.setdefault(ref, result)
            return ref

        small = "y" * 600
        # Many results that fit the pressure scale individually but exceed the
        # absolute budget together: _compact_groups must evict them to the
        # inventory (persisting on demand), keeping the wire bound.
        messages: list[dict[str, Any]] = []
        for index in range(120):
            call_id = f"s{index}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": call_id, "function": {"name": "coding_read_file"}}],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": "coding_read_file",
                        "content": small,
                    },
                ]
            )
        projected = project_tool_messages(messages, "task", persist)
        projected_bytes = sum(
            len(m["content"].encode()) for m in projected if m.get("role") == "tool"
        )
        assert projected_bytes <= _TOOL_CONTEXT_BYTES
        assert any("inventory_ref" in m.get("content", "") for m in projected), (
            "absolute projection bound lost its inventory eviction path"
        )
        assert any(
            result.get("tool_name") == "coding_read_file" for result in persisted.values()
        ), "evicted inline results were not persisted for lossless recovery"

    def test_inline_projection_is_idempotent(self) -> None:
        """Re-projecting the projected form does not re-abbreviate inline results."""
        from sdk.thread._results import project_tool_messages

        small = "line of evidence " * 40
        first = project_tool_messages(self._pressure_messages(small), "task", self._persisted)
        second = project_tool_messages(first, "task", self._persisted)
        assert first == second

    def test_result_without_room_is_not_inflated_into_a_preview(self) -> None:
        """No room left must not swap a tiny result for a larger preview envelope."""
        from sdk.thread._results import _preview, project_tool_messages

        tiny = "line of evidence " * 50  # 850 chars <= 600+300 excerpts
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "tiny", "function": {"name": "read"}}]},
            {"role": "tool", "tool_call_id": "tiny", "name": "read", "content": tiny},
            {"role": "assistant", "tool_calls": [{"id": "new", "function": {"name": "read"}}]},
            {"role": "tool", "tool_call_id": "new", "name": "read", "content": "z" * 3000},
        ]
        # budget 4000 - reserve 250 leaves 750 bytes after the newest result.
        projected = project_tool_messages(messages, "task", self._persisted, budget_bytes=4000)
        assert projected[1]["content"] == tiny
        assert len(_preview("r" + "0" * 64, tiny).encode()) > len(tiny.encode())


# ── project_tool_messages() — retrieval expansions excluded from pressure ──


class TestProjectionExpansionExclusion:
    """Readout recall is not new evidence; its bytes cannot deepen collapse.

    Regression shape for the 2026-10-05 turn-524 loop: every
    ``thread_read_content`` page the model fetched was itself counted in
    ``tool_bytes``, so recalling already-stored evidence re-triggered and
    deepened the very collapse that required the recall — and the fetched
    page was pointer-collapsed back to a teaser of itself.
    """

    @staticmethod
    def _persist(_: dict[str, Any]) -> str:
        return "r" + "0" * 64

    @staticmethod
    def _big_tool_result(call_id: str, name: str, chars: int) -> list[dict[str, Any]]:
        return [
            {
                "role": "assistant",
                "tool_calls": [{"id": call_id, "function": {"name": name}}],
            },
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": "x" * chars},
        ]

    @staticmethod
    def _expansion_message(chars: int) -> list[dict[str, Any]]:
        envelope = json.dumps(
            {"ref": "r" + "a" * 64, "content": "evidence " * (chars // 9), "has_more": False},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return [
            {
                "role": "assistant",
                "tool_calls": [{"id": "ex", "function": {"name": "thread_read_content"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "ex",
                "name": "thread_read_content",
                "content": envelope,
            },
        ]

    @staticmethod
    def _head_sizes(projected: list[dict[str, Any]]) -> list[int]:
        heads: list[int] = []
        for message in projected:
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str):
                continue
            try:
                preview = json.loads(content)
            except (TypeError, ValueError):
                continue
            if preview.get("abbreviated") is True:
                heads.append(len(preview["head"]))
        return heads

    def test_expansion_bytes_do_not_trigger_pressure(self) -> None:
        """A recall page must not collapse previews when work bytes fit the budget."""
        from sdk.thread._results import _TOOL_CONTEXT_BYTES, project_tool_messages

        work_chars = _TOOL_CONTEXT_BYTES - 8192  # under the threshold before the page
        messages = (
            self._big_tool_result("wa", "coding_read_file", work_chars // 2)
            + self._big_tool_result("wb", "coding_read_file", work_chars // 2)
            + self._expansion_message(8192)
        )
        projected = project_tool_messages(messages, "task", self._persist)
        page = next(m for m in projected if m.get("tool_call_id") == "ex")
        assert "evidence" in str(page.get("content")), (
            "expansion page was collapsed into a teaser of itself"
        )
        assert self._head_sizes(projected) and all(
            head == 600 for head in self._head_sizes(projected)
        ), "work-result previews collapsed without real overage"

    def test_expansion_does_not_deepen_existing_collapse(self) -> None:
        """Over-budget previews scale identically with and without a recall page."""
        from sdk.thread._results import _TOOL_CONTEXT_BYTES, project_tool_messages

        work_chars = _TOOL_CONTEXT_BYTES + 16384  # overage already in effect
        base = self._big_tool_result("wa", "coding_read_file", work_chars // 2) + (
            self._big_tool_result("wb", "coding_read_file", work_chars // 2)
        )
        without_page = project_tool_messages(base, "task", self._persist)
        with_page = project_tool_messages(
            [*base, *self._expansion_message(16_384)], "task", self._persist
        )
        heads_without = self._head_sizes(without_page)
        heads_with = [
            len(json.loads(m["content"])["head"])
            for m in with_page
            if m.get("tool_call_id") != "ex"
            and isinstance(m.get("content"), str)
            and json.loads(m["content"]).get("abbreviated") is True
        ]
        assert heads_without and heads_without == heads_with, (
            f"previews {heads_without} deepened to {heads_with} by a recall page"
        )

    @pytest.mark.parametrize("page_count", [30, 140])
    def test_many_retrieval_pages_stay_bounded_and_keep_latest_readable(
        self, page_count: int
    ) -> None:
        from sdk.thread._results import _TOOL_CONTEXT_BYTES, project_tool_messages

        persisted: dict[str, dict[str, Any]] = {}

        def persist(result: dict[str, Any]) -> str:
            ref = "r" + hashlib.sha256(json.dumps(result).encode()).hexdigest()
            persisted[ref] = result
            return ref

        messages: list[dict[str, Any]] = []
        for index in range(page_count):
            call_id = f"recall-{index}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": call_id, "function": {"name": "lisa-thread.thread_search"}}
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": "lisa-thread.thread_search",
                        "content": f"evidence-{index}: " + "x" * 4000,
                    },
                ]
            )

        projected = project_tool_messages(messages, "task", persist)
        tool_bytes = sum(
            len(message["content"].encode()) for message in projected if message["role"] == "tool"
        )
        assert tool_bytes <= _TOOL_CONTEXT_BYTES
        latest = next(
            message
            for message in projected
            if message.get("tool_call_id") == f"recall-{page_count - 1}"
        )
        assert f"evidence-{page_count - 1}" in latest["content"]
        assert any(
            "inventory_ref" in message.get("content", "")
            or '"abbreviated":true' in message.get("content", "")
            for message in projected
        )
        assert persisted
        assert (
            project_tool_messages(projected, "task", persist, load=persisted.__getitem__)
            == projected
        )

    def test_reused_call_id_does_not_reclassify_earlier_work_as_recall(self) -> None:
        from sdk.thread._results import project_tool_messages

        # 20000 chars stays verbatim as a recall page but must preview as work.
        messages = self._big_tool_result(
            "same", "coding_read_file", 20_000
        ) + self._big_tool_result("same", "thread_search", 4000)
        messages[1].pop("name")
        projected = project_tool_messages(messages, "task", self._persist)
        work_preview = json.loads(projected[1]["content"])
        assert work_preview["abbreviated"] is True
        assert projected[3]["content"] == "x" * 4000


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
