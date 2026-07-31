"""Tests for the file-backed ``FileWorkflowInstanceStore``.

Moved from ``kernel/tests/test_workflow_instance.py``: the store is a
host-owned persistence implementation (``hosts/core/workflow_instance_store.py``),
so its tests belong alongside the other ``hosts/core`` service tests, not
in the kernel suite. Covers save/load/load_active/clear semantics and
restart survival.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.recovery import RecoveryIdentity
from kernel.workflow_instance import WorkflowInstance, WorkflowStep

from hosts.core.workflow_instance_store import FileWorkflowInstanceStore


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


# ── FileWorkflowInstanceStore ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_save_load_round_trip(tmp_path: Path) -> None:
    """An instance saved with an identity loads back for that identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="search_0", mode_id="search"),),
        current_step_id="search_0",
        current_mode_id="search",
    )
    identity = _identity()
    await store.save(instance, identity)
    loaded = await store.load(instance.instance_id, identity)
    assert loaded is not None
    assert loaded.instance_id == instance.instance_id
    assert loaded.current_mode_id == "search"


@pytest.mark.asyncio
async def test_store_load_returns_none_for_wrong_identity(tmp_path: Path) -> None:
    """An instance does not load for a different identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(current_mode_id="search")
    await store.save(instance, _identity())
    assert await store.load(instance.instance_id, _other_identity()) is None


@pytest.mark.asyncio
async def test_store_load_active_returns_non_complete_instance(tmp_path: Path) -> None:
    """load_active returns the single non-complete instance for an identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    active = WorkflowInstance(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b", is_terminal=True),
        ),
        current_step_id="b_1",
        completed_steps=("a_0",),
    )
    complete = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="x_0", mode_id="x"),),
        completed_steps=("x_0",),
    )
    identity = _identity()
    await store.save(active, identity)
    await store.save(complete, identity)
    loaded = await store.load_active(identity)
    assert loaded is not None
    assert loaded.instance_id == active.instance_id


@pytest.mark.asyncio
async def test_store_load_active_returns_none_when_all_complete(tmp_path: Path) -> None:
    """load_active returns None when every instance for the identity is complete."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    complete = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="x_0", mode_id="x"),),
        completed_steps=("x_0",),
    )
    identity = _identity()
    await store.save(complete, identity)
    assert await store.load_active(identity) is None


@pytest.mark.asyncio
async def test_store_clear_removes_instance(tmp_path: Path) -> None:
    """clear removes an instance for its identity and returns True."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(current_mode_id="search")
    identity = _identity()
    await store.save(instance, identity)
    assert await store.clear(instance.instance_id, identity) is True
    assert await store.load(instance.instance_id, identity) is None
    # Clearing again is False (already gone).
    assert await store.clear(instance.instance_id, identity) is False


@pytest.mark.asyncio
async def test_store_clear_rejects_wrong_identity(tmp_path: Path) -> None:
    """clear returns False for a non-matching identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(current_mode_id="search")
    await store.save(instance, _identity())
    assert await store.clear(instance.instance_id, _other_identity()) is False


@pytest.mark.asyncio
async def test_store_survives_restart(tmp_path: Path) -> None:
    """A new store instance at the same path loads persisted instances."""
    path = tmp_path / "workflow-instances.json"
    store1 = FileWorkflowInstanceStore(path)
    instance = WorkflowInstance(current_mode_id="search")
    identity = _identity()
    await store1.save(instance, identity)
    # Simulate restart: new store object, same file.
    store2 = FileWorkflowInstanceStore(path)
    loaded = await store2.load(instance.instance_id, identity)
    assert loaded is not None
    assert loaded.current_mode_id == "search"
