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

    def test_pressure_pointer_replaces_page_without_persisting(self, tmp_path: Path) -> None:
        from sdk.thread._results import project_tool_messages, result_object_path

        original = "stored worktree state " * 200
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

        page = self._envelope(stored_ref, original[:50])
        filler = "x" * (_TOOL_CONTEXT_BYTES + 1)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "p", "function": {"name": "thread_read_content"}}],
            },
            {"role": "tool", "tool_call_id": "p", "name": "thread_read_content", "content": page},
            {"role": "assistant", "tool_calls": [{"id": "f", "function": {"name": "big_tool"}}]},
            {"role": "tool", "tool_call_id": "f", "name": "big_tool", "content": filler},
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
