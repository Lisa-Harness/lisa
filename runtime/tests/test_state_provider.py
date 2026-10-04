"""Tests for ``runtime.state_provider`` snapshot compilation and slice helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import ContextItem, UnifiedState
from runtime.state_provider import (
    FileBackedStateProvider,
    _slice_budget_expired,
    build_unified_state,
    load_state_snapshot,
)

from runtime import state_provider


def _item_dict(item_id: str = "state:x:1", body: str = "payload") -> dict[str, Any]:
    return {
        "item_id": item_id,
        "namespace": "signals",
        "source": "unit",
        "source_type": "intake",
        "evidence_type": "snapshot",
        "title": "t",
        "body": body,
        "collected_at": "2026-10-04T00:00:00+00:00",
    }


# ── build_unified_state ──────────────────────────────────────────────


def test_build_unified_state_none_snapshot_returns_empty() -> None:
    state = build_unified_state(state_snapshot=None)

    assert isinstance(state, UnifiedState)
    assert state.context_items == []
    assert state.compiled_at


def test_build_unified_state_parses_snapshot_items() -> None:
    snapshot = {
        "compiled_at": "2026-10-01T12:00:00+00:00",
        "context_items": [_item_dict("a"), _item_dict("b")],
    }

    state = build_unified_state(state_snapshot=snapshot)

    assert state.compiled_at == "2026-10-01T12:00:00+00:00"
    assert [item.item_id for item in state.context_items] == ["a", "b"]
    assert all(isinstance(item, ContextItem) for item in state.context_items)


def test_build_unified_state_generated_at_fallback() -> None:
    snapshot = {"generated_at": "2026-09-30T00:00:00+00:00", "context_items": []}

    state = build_unified_state(state_snapshot=snapshot)

    assert state.compiled_at == "2026-09-30T00:00:00+00:00"


def test_build_unified_state_invalid_items_coerced_to_empty() -> None:
    snapshot = {"context_items": [{"totally": "invalid"}]}

    state = build_unified_state(state_snapshot=snapshot)

    assert state.context_items == []


def test_build_unified_state_non_list_items_ignored() -> None:
    assert build_unified_state(state_snapshot={"context_items": "nope"}).context_items == []
    assert build_unified_state(state_snapshot={"context_items": None}).context_items == []


def test_build_unified_state_item_builder_contributes_and_dedupes() -> None:
    snapshot = {"context_items": [_item_dict("existing")]}

    class _Builder:
        def build_extension_context_items(self, runtime_context: Any, state: Any) -> list[Any]:
            assert state.compiled_at  # receives the compiled state
            return [
                ContextItem.model_validate(_item_dict("existing")),  # duplicate → skipped
                ContextItem.model_validate(_item_dict("fresh")),
                "not-a-context-item",  # non-item → skipped
            ]

    state = build_unified_state(state_snapshot=snapshot, item_builder=_Builder())

    assert [item.item_id for item in state.context_items] == ["existing", "fresh"]


def test_build_unified_state_builder_exception_keeps_intake_items() -> None:
    snapshot = {"context_items": [_item_dict("kept")]}

    class _Boom:
        def build_extension_context_items(self, runtime_context: Any, state: Any) -> list[Any]:
            raise RuntimeError("builder blew up")

    state = build_unified_state(state_snapshot=snapshot, item_builder=_Boom())

    assert [item.item_id for item in state.context_items] == ["kept"]


def test_build_unified_state_builder_empty_contribution() -> None:
    class _Empty:
        def build_extension_context_items(self, runtime_context: Any, state: Any) -> list[Any]:
            return []

    state = build_unified_state(state_snapshot=None, item_builder=_Empty())

    assert state.context_items == []


def test_build_unified_state_merges_thread_digest_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    digest_items = [
        ContextItem.model_validate(_item_dict("digest-1")),
        ContextItem.model_validate(_item_dict("existing")),
    ]
    monkeypatch.setattr(
        "runtime.thread_digest.build_thread_context_items",
        lambda path, redaction=None, settings=None: iter(digest_items),
    )
    snapshot = {"context_items": [_item_dict("existing")]}

    state = build_unified_state(
        state_snapshot=snapshot,
        thread_path=tmp_path,
        thread_redaction=object(),
    )

    assert [item.item_id for item in state.context_items] == ["existing", "digest-1"]


# ── load_state_snapshot ──────────────────────────────────────────────


def test_load_state_snapshot_explicit_path(tmp_path: Path) -> None:
    snapshot_file = tmp_path / "custom.json"
    snapshot_file.write_text(json.dumps({"compiled_at": "x", "keys": 2}), encoding="utf-8")

    loaded = load_state_snapshot(explicit_path=str(snapshot_file), data_dir=tmp_path)

    assert loaded == {"compiled_at": "x", "keys": 2}


def test_load_state_snapshot_explicit_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="State snapshot file not found"):
        load_state_snapshot(explicit_path=str(tmp_path / "absent.json"), data_dir=tmp_path)


def test_load_state_snapshot_default_missing_returns_none(tmp_path: Path) -> None:
    assert load_state_snapshot(explicit_path="", data_dir=tmp_path) is None
    assert load_state_snapshot(explicit_path="  ", data_dir=tmp_path) is None


def test_load_state_snapshot_default_path_found(tmp_path: Path) -> None:
    from sdk.layout import STATE_DIR

    state_dir = tmp_path / STATE_DIR
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(json.dumps({"a": 1}), encoding="utf-8")

    assert load_state_snapshot(explicit_path=None, data_dir=tmp_path) == {"a": 1}


def test_load_state_snapshot_unreadable_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="is unreadable"):
        load_state_snapshot(explicit_path=str(bad), data_dir=tmp_path)


def test_load_state_snapshot_non_object_raises(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_state_snapshot(explicit_path=str(bad), data_dir=tmp_path)


# ── _slice_budget_expired ────────────────────────────────────────────


def test_slice_budget_expired() -> None:
    assert _slice_budget_expired(SimpleNamespace(budget_ms=0), time.monotonic()) is False
    assert _slice_budget_expired(SimpleNamespace(budget_ms=-5), time.monotonic()) is False
    assert _slice_budget_expired(object(), time.monotonic()) is False  # missing attr → no budget

    now = time.monotonic()
    assert _slice_budget_expired(SimpleNamespace(budget_ms=60_000), now) is False
    assert _slice_budget_expired(SimpleNamespace(budget_ms=1), now - 0.5) is True


# ── FileBackedStateProvider ──────────────────────────────────────────


def test_file_backed_state_provider_call_returns_none(tmp_path: Path) -> None:
    provider = FileBackedStateProvider(thread_path=tmp_path)

    assert provider() is None


@pytest.mark.asyncio
async def test_file_backed_state_provider_slice_is_no_op(tmp_path: Path) -> None:
    provider = FileBackedStateProvider(thread_path=None)

    outcome = await provider.slice(object(), object())

    assert outcome.work_items_completed == 0
    assert outcome.elapsed_ms == 0.0
    assert outcome.last_item == ""
    assert outcome.notes == ("no_op_no_compaction",)


# ── private helpers ──────────────────────────────────────────────────


def test_coerce_items_passthrough_and_invalid() -> None:
    item = ContextItem.model_validate(_item_dict("direct"))

    result = state_provider._coerce_items([item, _item_dict("parsed")])

    assert result[0] is item  # existing ContextItem passes through untouched
    assert result[1].item_id == "parsed"
    assert state_provider._coerce_items(None) == []
    assert state_provider._coerce_items("string") == []
