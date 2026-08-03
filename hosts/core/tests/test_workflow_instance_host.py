"""Phase 5 tests: wire the canonical host (``HostRuntime`` + ``turn_hooks_builder``).

Covers the boot-load resume of an active ``WorkflowInstance`` from the
durable store and the wiring of the five workflow/checkpoint seams in
``build_turn_hooks``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.recovery import RecoveryIdentity
from kernel.workflow_instance import WorkflowInstance, WorkflowStep

from hosts.core import HostConfig, HostRuntime
from hosts.core.turn_hooks_builder import build_turn_hooks
from hosts.core.workflow_instance_store import FileWorkflowInstanceStore


def _identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="lisa",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        memory_scope="personal",
        session_plan_digest="digest",
    )


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """A HostRuntime with a minimal models.yaml, mirroring test_recovery_service."""
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(
        tmp_path,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )


def _active_instance() -> WorkflowInstance:
    """A non-complete instance mid-flight in coding mode."""
    return WorkflowInstance(
        plan_digest="abc",
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search", satisfies_intents=("research",)),
            WorkflowStep(
                step_id="coding_1",
                mode_id="coding",
                satisfies_intents=("coding",),
                depends_on=("search_0",),
            ),
            WorkflowStep(
                step_id="markdown_2",
                mode_id="markdown",
                satisfies_intents=("write",),
                depends_on=("coding_1",),
                is_terminal=True,
            ),
        ),
        completed_steps=("search_0",),
        current_step_id="coding_1",
        current_mode_id="coding",
        spawned_from="adhoc",
    )


@pytest.mark.asyncio
async def test_boot_load_resumes_active_instance(tmp_path: Path) -> None:
    """A non-complete instance in the store is resumed on host boot.

    Pre-populate the store, construct a fresh HostRuntime, and verify
    ``_active_workflow_instance`` is restored and ``current_mode_id``
    is overridden from the instance's current step's mode (not the
    plan's static ``start_mode``).
    """
    # Build the runtime first to get the real identity the boot-load will use.
    runtime = _make_runtime(tmp_path)
    identity = runtime._workflow_identity()

    # Pre-populate the store with an active instance using that identity.
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = _active_instance()
    await store.save(instance, identity)

    # Construct a fresh runtime — boot-load should find the instance.
    fresh = _make_runtime(tmp_path)
    fresh._boot_load_active_workflow_instance()  # pylint: disable=protected-access

    assert fresh._active_workflow_instance is not None
    resumed = fresh._active_workflow_instance
    assert resumed.instance_id == instance.instance_id
    assert resumed.current_step_id == "coding_1"
    # current_mode_id is overridden from the instance, not the plan's start_mode.
    assert fresh.current_mode_id == "coding"


def test_boot_load_no_active_instance_keeps_start_mode(tmp_path: Path) -> None:
    """When no instance is in the store, ``current_mode_id`` stays at start_mode."""
    runtime = _make_runtime(tmp_path)
    runtime._boot_load_active_workflow_instance()  # pylint: disable=protected-access
    assert runtime._active_workflow_instance is None
    # The plan's default start_mode (default) is preserved.
    assert runtime.current_mode_id == "default"


@pytest.mark.asyncio
async def test_set_workflow_instance_persists_and_clears_on_complete(tmp_path: Path) -> None:
    """Advancing to the last step clears the active ref and removes the store entry."""
    runtime = _make_runtime(tmp_path)
    identity = runtime._workflow_identity()
    instance = _active_instance()
    # Seed the active ref and store.
    await runtime.workflow_instance_store.save(instance, identity)
    runtime._active_workflow_instance = instance

    # Advance to the terminal step (markdown_2), then complete it.
    from kernel.workflow_instance import advance_step

    advanced = advance_step(instance, "coding_1")  # now current=markdown_2
    runtime._set_workflow_instance(advanced)
    assert runtime._active_workflow_instance is advanced  # not yet complete

    completed = advance_step(advanced, "markdown_2")  # now complete
    runtime._set_workflow_instance(completed)
    # Complete → active ref cleared.
    assert runtime._active_workflow_instance is None
    # Store entry removed.
    loaded = await runtime.workflow_instance_store.load(completed.instance_id, identity)
    assert loaded is None


def test_build_turn_hooks_wires_workflow_seams(tmp_path: Path) -> None:
    """``build_turn_hooks`` wires all five workflow/checkpoint seams.

    Before Phase 5 these were all ``None`` (dormant). This test pins the
    wiring so a future refactor doesn't silently un-wire them.
    """
    runtime = _make_runtime(tmp_path)
    hooks = build_turn_hooks(runtime)

    # The five seams the kernel defines but the host left dormant.
    assert hooks.workflow_instance is not None
    assert hooks.workflow_instance_updater is not None
    assert hooks.workflow_session is not None
    assert hooks.workflow_session_updater is not None
    assert hooks.on_checkpoint is not None

    # workflow_instance is a callable returning the active ref (None here).
    assert hooks.workflow_instance() is None
    # workflow_instance_updater is the host's _set_workflow_instance method.
    assert hooks.workflow_instance_updater == runtime._set_workflow_instance
    # on_checkpoint is the host's _on_checkpoint method.
    assert hooks.on_checkpoint == runtime._on_checkpoint
