"""Tests for the self-model slice's context item lifecycle.

The slice publishes a self-model item only while the runtime is degraded. A
later healthy (or failed) run must retract it, or the model keeps reading a
degradation that has already cleared.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from runtime.turn_hooks_builder import _run_self_model_slice_impl

from runtime import turn_hooks_builder

_ITEM_ID = "state:self_model:user"


def _runtime(tmp_path: Path) -> Any:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, user_slug="user")


def _item_ids(runtime: Any) -> list[str]:
    snapshot = runtime.session.state_snapshot or {}
    return [item.get("item_id") for item in snapshot.get("context_items", [])]


def _returning(health: str, text: str = "## Self Model\nvibe"):  # type: ignore[no-untyped-def]
    async def _execute(_operation_id: str, _ctx: Any) -> Any:
        return SimpleNamespace(output_json={"operational_health": health}, output_text=text)

    return _execute


async def _failing(_operation_id: str, _ctx: Any) -> Any:
    raise RuntimeError("introspection broke")


async def _run(runtime: Any) -> tuple[int, str, str]:
    return await _run_self_model_slice_impl(runtime, lambda _name: None, None)


@pytest.mark.asyncio
async def test_degraded_self_model_publishes_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(turn_hooks_builder, "execute_operation", _returning("degraded"))

    assert await _run(runtime) == (1, "self_model", "self_model health=degraded")
    assert _item_ids(runtime) == [_ITEM_ID]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "execute",
    [_returning("healthy"), _returning("degraded", text="  "), _failing],
    ids=["healthy", "empty-vibe", "failed"],
)
async def test_non_degraded_run_retracts_previous_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: Any
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(turn_hooks_builder, "execute_operation", _returning("degraded"))
    await _run(runtime)
    runtime.session.state_snapshot["context_items"].append({"item_id": "other"})

    monkeypatch.setattr(turn_hooks_builder, "execute_operation", execute)

    assert await _run(runtime) == (0, "", "")
    assert _item_ids(runtime) == ["other"]
