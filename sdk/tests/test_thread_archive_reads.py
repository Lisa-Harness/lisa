"""Bounded-read regressions for the thread archive: tail, single-turn, search, result pages."""

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
    _backend,
    load_result_object,
)
from sdk.thread._backend import _segment_paths
from sdk.thread._results import read_result_window, result_object_path


def _zstd_available() -> bool:
    try:
        import zstandard
    except ImportError:
        return False
    return bool(zstandard.ZstdCompressor)


def _write_active_entry(thread: Path, *, query: str, response_text: str) -> None:
    paths = _segment_paths(thread)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    entry = {"mode_id": "default", "query": query, "response_text": response_text}
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _seal_active(thread: Path) -> None:
    import zstandard

    paths = _segment_paths(thread)
    raw = paths.current.read_bytes()
    count = sum(1 for line in raw.splitlines() if line.strip())
    prior = _backend._read_index(paths.index_path)
    seg_id = f"{len(prior) + 1:04d}"
    sealed = _backend._sealed_segment_path(paths.base, seg_id)
    sealed.write_bytes(zstandard.ZstdCompressor().compress(raw))
    lo = sum(r.turn_hi - r.turn_lo + 1 for r in prior)
    record = _backend.SegmentRecord(seg_id, lo, lo + count - 1, "", "", len(raw), 0)
    with paths.index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict()) + "\n")
    paths.current.unlink()


needs_zstd = pytest.mark.skipif(not _zstd_available(), reason="zstandard not installed")


def _fill(thread: Path, count: int, *, start: int = 0, pad: int = 200) -> None:
    for index in range(start, start + count):
        _write_active_entry(thread, query=f"q{index} " + "x" * pad, response_text=f"r{index}")


class _ParseSpy:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sizes: list[int] = []
        original = _backend._parse_line

        def spy(line: bytes) -> dict[str, Any] | None:
            self.sizes.append(len(line))
            return original(line)

        monkeypatch.setattr(_backend, "_parse_line", spy)


def _store_result(base: Path, content: str) -> str:
    payload = json.dumps(
        {"content": content, "task_id": "t", "tool_name": "x"},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    ref = "r" + hashlib.sha256(payload).hexdigest()
    path = result_object_path(base, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ref


def _seal_raw(thread: Path, lines: list[str], valid: int) -> None:
    """Seal raw JSONL lines, possibly malformed, as a legacy segment with ``valid`` entries."""
    import zstandard

    paths = _segment_paths(thread)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    raw = ("\n".join(lines) + "\n").encode()
    sealed = _backend._sealed_segment_path(paths.base, "0001")
    sealed.write_bytes(zstandard.ZstdCompressor().compress(raw))
    record = _backend.SegmentRecord("0001", 0, valid - 1, "", "", len(raw), 0)
    paths.index_path.write_text(json.dumps(record.to_dict()) + "\n")


class TestActiveSegmentReads:
    def test_recent_parses_only_the_tail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        thread = tmp_path / "u"
        _fill(thread, 200)
        spy = _ParseSpy(monkeypatch)
        recent = ThreadStore(thread).load_recent(limit=3)
        assert [e["response_text"] for e in recent] == ["r197", "r198", "r199"]
        assert len(spy.sizes) == 3

    def test_recent_handles_lines_longer_than_a_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        thread = tmp_path / "u"
        _fill(thread, 3)
        _write_active_entry(thread, query="big " + "y" * 200_000, response_text="huge")
        _fill(thread, 2, start=4)
        recent = ThreadStore(thread).load_recent(limit=4)
        assert [e["response_text"] for e in recent] == ["r2", "huge", "r4", "r5"]

    def test_recent_skips_malformed_tail_lines(self, tmp_path: Path):
        thread = tmp_path / "u"
        _fill(thread, 4)
        with _segment_paths(thread).current.open("a") as fh:
            fh.write("{broken\n\n")
        assert [e["response_text"] for e in ThreadStore(thread).load_recent(limit=2)] == [
            "r2",
            "r3",
        ]

    def test_single_turn_parses_one_line(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        thread = tmp_path / "u"
        _fill(thread, 100)
        ThreadStore(thread).next_turn_index()  # warm the malformed-line map
        spy = _ParseSpy(monkeypatch)
        reader = ThreadReader(ThreadStore(thread))
        assert "q42" in reader.read_content("t42.query")
        assert len(spy.sizes) == 1

    def test_load_turns_parses_only_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        thread = tmp_path / "u"
        _fill(thread, 50)
        ThreadStore(thread).next_turn_index()
        spy = _ParseSpy(monkeypatch)
        loaded = ThreadStore(thread).load_turns([3, 40, 999])
        assert sorted(loaded) == [3, 40]
        assert len(spy.sizes) == 2

    def test_next_turn_index_parses_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        thread = tmp_path / "u"
        _fill(thread, 30)
        ThreadStore(thread).next_turn_index()
        spy = _ParseSpy(monkeypatch)
        assert ThreadStore(thread).next_turn_index() == 30
        assert spy.sizes == []


@needs_zstd
class TestSealedReads:
    def test_single_sealed_turn_and_resident_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        thread = tmp_path / "u"
        _fill(thread, 60)
        _seal_active(thread)
        _fill(thread, 5, start=60)
        ThreadStore(thread).next_turn_index()
        spy = _ParseSpy(monkeypatch)
        entries = ThreadStore(thread).load_range(start_turn=10, end_turn=10)
        assert [e["response_text"] for e in entries] == ["r10"]
        # Streaming decode stops at the window end: lines after turn 10 never parse.
        assert len(spy.sizes) == 11

    def test_recent_crosses_into_sealed_tail(self, tmp_path: Path):
        thread = tmp_path / "u"
        _fill(thread, 40)
        _seal_active(thread)
        _fill(thread, 2, start=40)
        recent = ThreadStore(thread).load_recent(limit=5)
        assert [e["response_text"] for e in recent] == ["r37", "r38", "r39", "r40", "r41"]

    def test_sealed_tail_retries_when_tail_has_malformed_lines(self, tmp_path: Path):
        good = [json.dumps({"query": f"q{i}", "response_text": f"r{i}"}) for i in range(5)]
        _seal_raw(tmp_path / "u", [*good, *["{bad"] * 60], valid=5)
        recent = ThreadStore(tmp_path / "u").load_recent(limit=3)
        assert [e["response_text"] for e in recent] == ["r2", "r3", "r4"]

    def test_search_keeps_only_top_k_resident(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import weakref

        class Tracked(dict):  # type: ignore[type-arg]
            pass

        thread = tmp_path / "u"
        _fill(thread, 100, pad=5)
        _seal_active(thread)
        _fill(thread, 100, start=100, pad=5)
        store = ThreadStore(thread)
        alive: set[int] = set()
        peak = 0
        original = ThreadStore._score_entry

        def tracked(self: ThreadStore, *args: Any) -> Any:
            nonlocal peak
            scored = original(self, *args)
            if scored is None:
                return None
            score, turn, result = scored
            wrapped = Tracked(result)
            key = id(wrapped)
            alive.add(key)
            weakref.finalize(wrapped, alive.discard, key)
            peak = max(peak, len(alive))
            return score, turn, wrapped

        monkeypatch.setattr(ThreadStore, "_score_entry", tracked)
        results = store.search(query="q150", limit=3)
        assert results[0]["turn"] == 150
        assert peak <= 3 + 3

    def test_search_matches_exhaustive_ranking(self, tmp_path: Path):
        thread = tmp_path / "u"
        for index in range(30):
            _write_active_entry(
                thread, query="alpha beta" if index % 3 else "alpha", response_text=f"w{index}"
            )
        _seal_active(thread)
        _fill(thread, 5, start=30, pad=3)
        store = ThreadStore(thread)
        got = store.search(query="alpha beta", limit=7)
        expected = sorted(
            (
                s
                for turn, entry in store.iter_range(start_turn=0, end_turn=10**9)
                if (s := store._score_entry(entry, turn, {"alpha", "beta"}, None))
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )[:7]
        assert [r["turn"] for r in got] == [t for _, t, _ in expected]

    def test_legacy_malformed_sealed_lines_stay_readable(self, tmp_path: Path):
        _seal_raw(tmp_path / "u", ['{"query":"a"}', "{oops", "", '{"query":"b"}'], valid=2)
        loaded = ThreadStore(tmp_path / "u").load_range(start_turn=0, end_turn=9)
        assert [e["query"] for e in loaded] == ["a", "b"]


CONTENTS = [
    "",
    "plain ascii " * 5000,
    'quote " backslash \\ newline \n tab \t \u0001 ' * 4000,
    "unicode é漢字 😀 mixed 🧪 " * 9000,
    "😀" * 70_000,
]


class TestResultWindow:
    @pytest.mark.parametrize("content", CONTENTS)
    @pytest.mark.parametrize("offset", [0, 1, 65_535, 65_536, 100_000])
    def test_window_matches_full_load(self, tmp_path: Path, content: str, offset: int):
        base = tmp_path / "u"
        ref = _store_result(base, content)
        offset = min(offset, len(content))
        window, total = read_result_window(base, ref, offset_chars=offset, limit_chars=777)
        assert total == len(content)
        assert window == content[offset : offset + 777]
        assert load_result_object(base, ref)["content"] == content

    def test_large_value_pages_without_materializing(self, tmp_path: Path):
        base = tmp_path / "u"
        content = "0123456789é😀\n" * 400_000
        ref = _store_result(base, content)
        window, total = read_result_window(base, ref, offset_chars=2_000_000, limit_chars=50)
        assert total == len(content)
        assert window == content[2_000_000:2_000_050]

    def test_tampered_object_fails_integrity(self, tmp_path: Path):
        base = tmp_path / "u"
        ref = _store_result(base, "abc" * 50_000)
        path = result_object_path(base, ref)
        data = bytearray(path.read_bytes())
        data[100_000] ^= 1
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="integrity"):
            read_result_window(base, ref, offset_chars=0, limit_chars=10)

    def test_missing_object(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not found"):
            read_result_window(tmp_path / "u", "r" + "0" * 64, offset_chars=0, limit_chars=1)

    def test_non_canonical_layout_falls_back(self, tmp_path: Path):
        base = tmp_path / "u"
        payload = b'{"task_id":"t","content":"h\\u00e9llo"}'
        ref = "r" + hashlib.sha256(payload).hexdigest()
        path = result_object_path(base, ref)
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        assert read_result_window(base, ref, offset_chars=1, limit_chars=3) == ("éll", 5)

    def test_reader_pages_identical_with_and_without_empty_redaction(self, tmp_path: Path):
        thread = tmp_path / "u"
        content = "line é😀\n" * 30_000
        ref = _store_result(thread, content)
        plain = ThreadReader(ThreadStore(thread))
        empty = ThreadReader(
            ThreadStore(thread, redaction=ThreadRedactionPolicy()),
            redaction=ThreadRedactionPolicy(),
        )
        first = json.loads(plain.read_content(ref, 1234, 5000))
        assert first == json.loads(empty.read_content(ref, 1234, 5000))
        assert first["total_chars"] == len(content)
        assert first["content"] == content[1234 : 1234 + len(first["content"])]
        assert first["has_more"] is True


def _tool_exchange(index: int, size: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{index}",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"c{index}",
            "name": "read_file",
            "content": f"{index}" * size,
        },
    ]


class TestProjectionIrreversible:
    def _project(self, messages: list[dict[str, Any]], store: dict[str, Any], budget: int):
        from sdk.thread import project_tool_messages

        def persist(result: dict[str, Any]) -> str:
            payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            ref = "r" + hashlib.sha256(payload.encode()).hexdigest()
            store[ref] = result
            return ref

        return project_tool_messages(
            messages, "task", persist, None, lambda ref: store[ref], budget_bytes=budget
        )

    def test_reprojecting_a_projection_never_re_expands(self):
        store: dict[str, Any] = {}
        raw = [m for i in range(3) for m in _tool_exchange(i, 6000)]
        first = self._project(raw, store, 8000)
        assert len(json.dumps(first)) < len(json.dumps(raw))
        tail = _tool_exchange(3, 100)
        # A much larger budget later must keep earlier reductions as they were.
        again = self._project([*first, *tail], store, 10_000_000)
        assert again[: len(first)] == first
        assert again[len(first) :] == tail
        # And the same budget is a fixed point.
        assert self._project(first, store, 8000) == first


class TestReadCacheBudget:
    def test_cached_page_never_exceeds_a_later_smaller_budget(self, tmp_path: Path):
        from sdk.thread import PageCache

        thread = tmp_path / "u"
        ref = _store_result(thread, "word é " * 20_000)
        reader = ThreadReader(ThreadStore(thread), page_cache=PageCache())
        big = reader.read_content(ref, 0, 100_000, budget_bytes=MAX_READOUT_BYTES)
        small = reader.read_content(ref, 0, 100_000, budget_bytes=DEFAULT_READOUT_BYTES)
        assert len(big.encode()) > DEFAULT_READOUT_BYTES
        assert len(small.encode()) <= DEFAULT_READOUT_BYTES
        reader.read_content(ref, 0, 1000, budget_bytes=MAX_READOUT_BYTES)
        again = reader.read_content(ref, 0, 1000, budget_bytes=MAX_READOUT_BYTES)
        assert json.loads(again)["cached"]
        found_big = reader.find_content(ref, "word", 0, 20, budget_bytes=MAX_READOUT_BYTES)
        found_small = reader.find_content(ref, "word", 0, 20, budget_bytes=DEFAULT_READOUT_BYTES)
        assert len(found_small.encode()) <= DEFAULT_READOUT_BYTES
        assert "cached" not in json.loads(found_small) and "cached" not in json.loads(found_big)


class TestWrappedResultObjects:
    def test_object_without_content_key_loads_wrapped(self, tmp_path: Path):
        base = tmp_path / "u"
        payload = b'{"output_text":"{\\"a\\":1}"}'
        ref = "r" + hashlib.sha256(payload).hexdigest()
        path = result_object_path(base, ref)
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        assert load_result_object(base, ref) == {"content": {"output_text": '{"a":1}'}}
        window, total = read_result_window(base, ref, offset_chars=0, limit_chars=1000)
        assert json.loads(window) == {"output_text": '{"a":1}'} and total == len(window)

    def test_non_string_content_is_invalid(self, tmp_path: Path):
        base = tmp_path / "u"
        payload = b'{"content":{"x":1}}'
        ref = "r" + hashlib.sha256(payload).hexdigest()
        path = result_object_path(base, ref)
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="Invalid result-object content"):
            load_result_object(base, ref)


class TestProjectionFullFidelityUntilPressure:
    def test_large_results_stay_verbatim_when_total_fits(self):
        store: dict[str, Any] = {}
        raw = [
            *_tool_exchange(0, 100_000),
            *_tool_exchange(1, 100_000),
            *_tool_exchange(2, 50_000),
        ]
        projected = TestProjectionIrreversible()._project(raw, store, 400_000)
        assert projected == raw
        assert store == {}

    def test_only_oldest_reduce_under_total_pressure(self):
        store: dict[str, Any] = {}
        raw = [m for i in range(3) for m in _tool_exchange(i, 100_000)]
        projected = TestProjectionIrreversible()._project(raw, store, 250_000)
        assert projected[-1]["content"] == raw[-1]["content"]
        assert projected[1]["content"] != raw[1]["content"]


class TestActiveOrdinalsWithMalformedLine:
    def _thread(self, tmp_path: Path) -> Path:
        thread = tmp_path / "u"
        _fill(thread, 3)
        with _segment_paths(thread).current.open("a") as handle:
            handle.write("{broken\n")
        _fill(thread, 3, start=3)
        return thread

    def test_ordinals_count_valid_entries_only(self, tmp_path: Path):
        store = ThreadStore(self._thread(tmp_path))
        assert store.next_turn_index() == 6
        assert [(t, e["response_text"]) for t, e in store.iter_range(start_turn=0, end_turn=9)] == [
            (i, f"r{i}") for i in range(6)
        ]
        assert [e["response_text"] for e in store.load_recent(limit=2)] == ["r4", "r5"]
        assert store.load_range(start_turn=3, end_turn=3)[0]["response_text"] == "r3"

    def test_refs_and_appends_stay_consistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        thread = self._thread(tmp_path)
        store = ThreadStore(thread)
        assert store.next_turn_index() == 6  # builds the map
        spy = _ParseSpy(monkeypatch)
        assert "q4" in ThreadReader(store).read_content("t4.query")
        assert len(spy.sizes) == 1
        _fill(thread, 1, start=6)
        assert store.next_turn_index() == 7
        assert store.load_range(start_turn=6, end_turn=6)[0]["response_text"] == "r6"

    def test_replaced_active_file_resets_the_map(self, tmp_path: Path):
        thread = self._thread(tmp_path)
        store = ThreadStore(thread)
        assert store.next_turn_index() == 6
        current = _segment_paths(thread).current
        current.unlink()
        _fill(thread, 8, start=100)
        assert store.next_turn_index() == 8
        assert store.load_range(start_turn=0, end_turn=0)[0]["response_text"] == "r100"
