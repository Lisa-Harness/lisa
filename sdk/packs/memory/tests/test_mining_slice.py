"""Tests for the thread-mining slice provider.

Mocks the ``turn_hooks`` view (``execute_operation``, ``execute_tool``,
model seams) to verify segment discovery, marker persistence, the
positional-marker recovery, budget short-circuit, and the factory's
tuning resolution — no live model or SQLite store involved.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import zstandard
from kernel.slice import SliceMode, SlicePolicy
from sdk.packs.memory.mining_slice import (
    _LAST_MINED_SEGMENT_FILE_NAME,
    MemoryMiningSlice,
    MemoryMiningSliceConfig,
    get_slice_provider_factory,
)
from sdk.thread import SegmentRecord


def _config(
    *,
    memory_data_dir: Path | None = None,
    user_slug: str = "adam",
    max_segments_per_slice: int = 2,
) -> MemoryMiningSliceConfig:
    return MemoryMiningSliceConfig(
        enabled=True,
        preferred_model_id="default",
        mine_operation="mine_thread_segment",
        max_segments_per_slice=max_segments_per_slice,
        memory_data_dir=memory_data_dir,
        user_slug=user_slug,
    )


def _policy(budget_ms: int = 500) -> SlicePolicy:
    return SlicePolicy(
        mode=SliceMode.POST_TURN,
        budget_ms=budget_ms,
        base_budget_ms=10,
        max_deferred_ms=200,
        hard_cap_ms=500,
    )


def _write_segment_fixture(
    base: Path, *, seg_id: str = "0001", turns: tuple[int, int] = (0, 2)
) -> None:
    entries = [
        {
            "turn": i,
            "user_slug": "adam",
            "mode_id": "default",
            "query": f"q{i}",
            "response_text": f"response {i}",
        }
        for i in range(turns[0], turns[1] + 1)
    ]
    raw = "".join(json.dumps(entry) + "\n" for entry in entries).encode()
    compressed = zstandard.ZstdCompressor().compress(raw)
    Path(f"{base}.{seg_id}.jsonl.zst").write_bytes(compressed)
    record = SegmentRecord(seg_id, turns[0], turns[1], "", "", len(raw), len(compressed))
    index_path = Path(f"{base}.index.jsonl")
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict()) + "\n")
    Path(f"{base}.current.jsonl").write_text("", encoding="utf-8")


def _thread_base(tmp_path: Path, user_slug: str = "adam") -> Path:
    base = tmp_path / "threads" / user_slug
    base.parent.mkdir(parents=True, exist_ok=True)
    return base


def test_missing_marker_root_is_a_noop() -> None:
    slc = MemoryMiningSlice(config=_config(), turn_hooks=_FakeHooks())

    assert slc._mine_marker_path() is None
    assert slc._load_last_mined_seg_id() == ""
    slc._persist_mine_marker("0001")


def test_marker_io_failures_are_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).mkdir()
    slc = MemoryMiningSlice(config=_config(memory_data_dir=tmp_path), turn_hooks=_FakeHooks())

    assert slc._load_last_mined_seg_id() == ""
    slc._persist_mine_marker("0001")

    assert "failed to read mine.marker" in caplog.text
    assert "failed to persist mine.marker" in caplog.text


def test_segment_index_reports_read_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = _thread_base(tmp_path)
    Path(f"{base}.index.jsonl").mkdir()
    slc = MemoryMiningSlice(config=_config(memory_data_dir=tmp_path), turn_hooks=_FakeHooks())

    assert slc._segment_index(base) == []
    assert "failed to read segment index" in caplog.text


def test_segment_index_ignores_invalid_records(tmp_path: Path) -> None:
    base = _thread_base(tmp_path)
    record = {"seg_id": "0001", "turn_lo": 0, "turn_hi": 2}
    Path(f"{base}.index.jsonl").write_text(
        "\n{bad json}\n[]\n{}\n" + json.dumps(record) + "\n", encoding="utf-8"
    )
    slc = MemoryMiningSlice(config=_config(memory_data_dir=tmp_path), turn_hooks=_FakeHooks())

    assert slc._segment_index(base) == [record]


class _FakeHooks:
    """Minimal hooks view with execute_tool + execute_operation + model seams."""

    def __init__(
        self,
        operation_extra: dict[str, Any] | None = None,
        *,
        operation_status: str = "success",
    ) -> None:
        self._operation_extra = operation_extra or {
            "mined": True,
            "memory_ids": [4, 5],
        }
        self._operation_status = operation_status
        self.op_calls: list[tuple[str, dict[str, Any]]] = []
        self.user_slug = "adam"
        self.memory_data_dir = None

    def resolve_model(self, model_id: str) -> dict[str, Any]:
        return {"model_id": "test-model"}

    async def complete(self, **kwargs: Any) -> str:
        return json.dumps({"facts": [{"action": "create", "fact": "f"}]})

    def read_authored_text(self, path: str) -> str:
        return "Mining system prompt (stubbed)."

    async def execute_operation(self, op_id: str, ctx: Any) -> Any:
        self.op_calls.append((op_id, dict(getattr(ctx, "params", {}) or {})))
        return type(
            "_OpResult",
            (),
            {
                "output_text": "mined",
                "extra": self._operation_extra,
                "status": self._operation_status,
            },
        )()


@pytest.mark.asyncio
async def test_mines_new_segment_and_advances_marker(tmp_path: Path) -> None:
    hooks = _FakeHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 1
    assert outcome.last_item == "mined:0001(0-2)->2"
    assert hooks.op_calls[0][0] == "mine_thread_segment"
    assert len(hooks.op_calls[0][1]["turns"]) == 3
    assert (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).read_text().strip() == "0001"


@pytest.mark.asyncio
async def test_marker_present_skips_mined_segment(tmp_path: Path) -> None:
    hooks = _FakeHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).write_text("0001", encoding="utf-8")
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 0
    assert hooks.op_calls == []


@pytest.mark.asyncio
async def test_marker_id_missing_from_index_remines_from_start(tmp_path: Path) -> None:
    """Positional marker recovery: a marker id absent from the index (rebuilt
    store) must not strand the slice mid-scan — it re-mines from the start,
    which is safe because the operation's write-time dedup skips records
    that already exist."""
    hooks = _FakeHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).write_text("zzzz", encoding="utf-8")
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 1


@pytest.mark.asyncio
async def test_no_thread_base_no_ops(tmp_path: Path) -> None:
    """No memory_data_dir → no thread base → empty outcome, no crash."""
    hooks = _FakeHooks()
    slc = MemoryMiningSlice(config=_config(memory_data_dir=None), turn_hooks=hooks)

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 0
    assert hooks.op_calls == []


@pytest.mark.asyncio
async def test_unreadable_segment_advances_marker_without_op(tmp_path: Path) -> None:
    """A segment listed in the index whose turn range cannot be loaded is
    skipped permanently (marker advances) so it is not retried forever."""
    hooks = _FakeHooks()
    base = _thread_base(tmp_path)
    # Index lists a segment whose compressed file does not exist.
    record = SegmentRecord("0001", 0, 2, "", "", 10, 5)
    Path(f"{base}.index.jsonl").write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 0
    assert hooks.op_calls == []
    assert (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).read_text().strip() == "0001"


@pytest.mark.asyncio
async def test_operation_failure_does_not_advance_marker(tmp_path: Path) -> None:
    """A failed operation leaves the segment pending — the next slice retries."""

    class _FailingHooks(_FakeHooks):
        async def execute_operation(self, op_id: str, ctx: Any) -> Any:
            raise RuntimeError("distillation down")

    hooks = _FailingHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 0
    assert not (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_operation_exception_stops_pass_and_retries_in_order(tmp_path: Path) -> None:
    class _FailOnceHooks(_FakeHooks):
        attempts = 0

        async def execute_operation(self, op_id: str, ctx: Any) -> Any:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("distillation down")
            return await super().execute_operation(op_id, ctx)

    hooks = _FailOnceHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base, seg_id="0001", turns=(0, 1))
    _write_segment_fixture(base, seg_id="0002", turns=(2, 3))
    slc = MemoryMiningSlice(config=_config(memory_data_dir=tmp_path), turn_hooks=hooks)

    failed = await slc.slice(None, _policy())

    assert failed.work_items_completed == 0
    assert hooks.attempts == 1
    assert not (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).exists()

    retried = await slc.slice(None, _policy())

    assert retried.work_items_completed == 2
    assert [params["turns"][0]["turn"] for _, params in hooks.op_calls] == [0, 2]
    assert (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).read_text().strip() == "0002"


class _StatusSwitchHooks(_FakeHooks):
    """First call succeeds, later calls report storage failures."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    async def execute_operation(self, op_id: str, ctx: Any) -> Any:
        self._calls += 1
        if self._calls > 1:
            self.op_calls.append((op_id, dict(getattr(ctx, "params", {}) or {})))
            return type(
                "_OpResult",
                (),
                {
                    "output_text": "mining failed",
                    "extra": {"mined": False, "memory_ids": []},
                    "status": "error",
                },
            )()
        return await super().execute_operation(op_id, ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation_extra", "operation_status"),
    [
        ({"mined": False, "memory_ids": [], "error": "backend offline"}, "error"),
        ({"mined": False, "memory_ids": [4]}, "error"),
    ],
    ids=["zero_storage", "partial_storage"],
)
async def test_error_result_leaves_segment_pending(
    tmp_path: Path,
    operation_extra: dict[str, Any],
    operation_status: str,
) -> None:
    """A returned status='error' result is a retryable storage failure:
    the marker stays un-advanced so the next pass retries the segment."""
    hooks = _FakeHooks(operation_extra=operation_extra, operation_status=operation_status)
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 0
    assert not (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_error_result_retries_next_pass(tmp_path: Path) -> None:
    """After a storage-failure pass the pending segment mines on retry."""
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    failing = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"),
        turn_hooks=_FakeHooks(
            operation_extra={"mined": False, "memory_ids": []},
            operation_status="error",
        ),
    )
    failed = await failing.slice(None, _policy())

    assert failed.work_items_completed == 0
    assert not (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).exists()

    success = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=_FakeHooks()
    )
    outcome = await success.slice(None, _policy())

    assert outcome.work_items_completed == 1
    assert (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).read_text().strip() == "0001"


@pytest.mark.asyncio
async def test_all_skip_result_advances_marker(tmp_path: Path) -> None:
    """An intentional all-skip result (mined False, no error) is not a
    failure: the segment is done, so the marker advances."""
    hooks = _FakeHooks(operation_extra={"mined": False, "memory_ids": [], "skipped": 1})
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 1
    assert (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).read_text().strip() == "0001"


@pytest.mark.asyncio
async def test_storage_failure_stops_the_pass_after_prior_success(tmp_path: Path) -> None:
    """Partial-pass failure: earlier segments stay mined, the failing one
    stays pending, and later segments wait for the next pass."""
    hooks = _StatusSwitchHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base, seg_id="0001", turns=(0, 1))
    _write_segment_fixture(base, seg_id="0002", turns=(2, 3))
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 1
    assert (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).read_text().strip() == "0001"
    assert len(hooks.op_calls) == 2


@pytest.mark.asyncio
async def test_segments_capped_per_slice(tmp_path: Path) -> None:
    """Only max_segments_per_slice segments are mined per pass."""
    hooks = _FakeHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base, seg_id="0001", turns=(0, 1))
    _write_segment_fixture(base, seg_id="0002", turns=(2, 3))
    _write_segment_fixture(base, seg_id="0003", turns=(4, 5))
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam", max_segments_per_slice=2),
        turn_hooks=hooks,
    )

    outcome = await slc.slice(None, _policy())

    assert outcome.work_items_completed == 2
    assert (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).read_text().strip() == "0002"


@pytest.mark.asyncio
async def test_budget_zero_means_unlimited_mines_segment(tmp_path: Path) -> None:
    """A zero budget disables the short-circuit (``SlicePolicy``: the
    provider short-circuits only when ``elapsed_ms`` exceeds a positive
    budget), so the segment still mines."""
    hooks = _FakeHooks()
    base = _thread_base(tmp_path)
    _write_segment_fixture(base)
    slc = MemoryMiningSlice(
        config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks
    )

    outcome = await slc.slice(None, _policy(budget_ms=0))

    assert outcome.work_items_completed == 1


@pytest.mark.asyncio
async def test_expired_budget_does_not_advance_the_marker(tmp_path: Path) -> None:
    hooks = _FakeHooks()
    _write_segment_fixture(_thread_base(tmp_path))
    slc = MemoryMiningSlice(config=_config(memory_data_dir=tmp_path), turn_hooks=hooks)

    completed, last_item = await slc._mine_new_segments(_policy(budget_ms=1), time.monotonic() - 1)

    assert (completed, last_item) == (0, "")
    assert hooks.op_calls == []
    assert not (tmp_path / _LAST_MINED_SEGMENT_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_missing_operation_dispatcher_does_no_work(tmp_path: Path) -> None:
    slc = MemoryMiningSlice(config=_config(memory_data_dir=tmp_path), turn_hooks=object())

    assert (await slc.slice(None, _policy())).work_items_completed == 0


# ── Factory ───────────────────────────────────────────────────────────


class _Plan:
    """Minimal session-plan stand-in exposing context.memory.mining."""

    def __init__(self, mining: Any) -> None:
        memory = type("Mem", (), {"mining": mining})()
        context = type("C", (), {"memory": memory})()
        self.manifest = type("M", (), {"context": context})()


@pytest.mark.asyncio
async def test_factory_returns_none_when_tuning_missing() -> None:
    factory = get_slice_provider_factory()
    assert factory(_FakeHooks(), _Plan(None)) is None


@pytest.mark.parametrize(
    "plan",
    [
        None,
        SimpleNamespace(manifest=SimpleNamespace(context=None)),
        SimpleNamespace(manifest=SimpleNamespace(context=SimpleNamespace(memory=None))),
    ],
)
def test_factory_returns_none_for_incomplete_plan(plan: Any) -> None:
    assert get_slice_provider_factory()(_FakeHooks(), plan) is None


def test_factory_builds_slice_from_view() -> None:
    class _Tuning:
        enabled = True
        preferred_model_id = "default"
        mine_operation = "mine_thread_segment"
        max_segments_per_slice = 3

    class _View(_FakeHooks):
        def __init__(self) -> None:
            super().__init__()
            self.memory_data_dir = Path("/tmp/lisa-data")
            self.user_slug = "adam"
            self.workspace_id = "ws"
            self.memory_scope = "personal"
            self.surface_id = "console"

    slc = get_slice_provider_factory()(_View(), _Plan(_Tuning()))

    assert isinstance(slc, MemoryMiningSlice)
    assert slc._config.mine_operation == "mine_thread_segment"
    assert slc._config.max_segments_per_slice == 3
    assert slc._config.memory_data_dir == Path("/tmp/lisa-data")
    assert slc._config.user_slug == "adam"


def test_factory_returns_none_when_disabled() -> None:
    class _Tuning:
        enabled = False

    assert get_slice_provider_factory()(_FakeHooks(), _Plan(_Tuning())) is None
