"""Tests for the file-backed ``FileWorkflowPositionStore``.

Moved from ``kernel/tests/test_workflow_instance.py``: the store is a
host-owned persistence implementation (``host/workflow_position_store.py``),
so its tests belong alongside the other ``host`` service tests, not
in the kernel suite. Covers save/load/load_active/clear semantics and
restart survival.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.recovery import RecoveryIdentity
from kernel.workflow_position import WorkflowPosition, WorkflowStep

from host.workflow_position_store import FileWorkflowPositionStore


def _identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        memory_scope="personal",
        session_plan_digest="digest",
    )


def _other_identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="other",
        user_id="other",
        user_slug="other",
        surface_type="personal_dm",
        surface_id="other",
        memory_scope="personal",
        session_plan_digest="digest",
    )


# ── FileWorkflowPositionStore ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_save_load_round_trip(tmp_path: Path) -> None:
    """An instance saved with an identity loads back for that identity."""
    store = FileWorkflowPositionStore(tmp_path / "workflow-positions.json")
    instance = WorkflowPosition(
        declared_steps=(WorkflowStep(step_id="search_0", mode_id="search"),),
        current_step_id="search_0",
        mode_id="search",
    )
    identity = _identity()
    await store.save(instance, identity)
    loaded = await store.load(instance.position_id, identity)
    assert loaded is not None
    assert loaded.position_id == instance.position_id
    assert loaded.mode_id == "search"


@pytest.mark.asyncio
async def test_store_load_returns_none_for_wrong_identity(tmp_path: Path) -> None:
    """An instance does not load for a different identity."""
    store = FileWorkflowPositionStore(tmp_path / "workflow-positions.json")
    instance = WorkflowPosition(mode_id="search")
    await store.save(instance, _identity())
    assert await store.load(instance.position_id, _other_identity()) is None


@pytest.mark.asyncio
async def test_store_load_active_returns_non_complete_instance(tmp_path: Path) -> None:
    """load_active returns the single non-complete instance for an identity."""
    store = FileWorkflowPositionStore(tmp_path / "workflow-positions.json")
    active = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b", is_terminal=True),
        ),
        current_step_id="b_1",
        completed_steps=("a_0",),
    )
    complete = WorkflowPosition(
        declared_steps=(WorkflowStep(step_id="x_0", mode_id="x"),),
        completed_steps=("x_0",),
    )
    identity = _identity()
    await store.save(active, identity)
    await store.save(complete, identity)
    loaded = await store.load_active(identity)
    assert loaded is not None
    assert loaded.position_id == active.position_id


@pytest.mark.asyncio
async def test_store_load_active_returns_none_when_all_complete(tmp_path: Path) -> None:
    """load_active returns None when every instance for the identity is complete."""
    store = FileWorkflowPositionStore(tmp_path / "workflow-positions.json")
    complete = WorkflowPosition(
        declared_steps=(WorkflowStep(step_id="x_0", mode_id="x"),),
        completed_steps=("x_0",),
    )
    identity = _identity()
    await store.save(complete, identity)
    assert await store.load_active(identity) is None


@pytest.mark.asyncio
async def test_store_clear_removes_instance(tmp_path: Path) -> None:
    """clear removes an instance for its identity and returns True."""
    store = FileWorkflowPositionStore(tmp_path / "workflow-positions.json")
    instance = WorkflowPosition(mode_id="search")
    identity = _identity()
    await store.save(instance, identity)
    assert await store.clear(instance.position_id, identity) is True
    assert await store.load(instance.position_id, identity) is None
    # Clearing again is False (already gone).
    assert await store.clear(instance.position_id, identity) is False


@pytest.mark.asyncio
async def test_store_clear_rejects_wrong_identity(tmp_path: Path) -> None:
    """clear returns False for a non-matching identity."""
    store = FileWorkflowPositionStore(tmp_path / "workflow-positions.json")
    instance = WorkflowPosition(mode_id="search")
    await store.save(instance, _identity())
    assert await store.clear(instance.position_id, _other_identity()) is False


@pytest.mark.asyncio
async def test_store_survives_restart(tmp_path: Path) -> None:
    """A new store instance at the same path loads persisted instances."""
    path = tmp_path / "workflow-positions.json"
    store1 = FileWorkflowPositionStore(path)
    instance = WorkflowPosition(mode_id="search")
    identity = _identity()
    await store1.save(instance, identity)
    # Simulate restart: new store object, same file.
    store2 = FileWorkflowPositionStore(path)
    loaded = await store2.load(instance.position_id, identity)
    assert loaded is not None
    assert loaded.mode_id == "search"
