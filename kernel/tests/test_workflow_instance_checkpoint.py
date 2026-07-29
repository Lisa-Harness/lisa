"""Phase 6 tests: checkpoint + recovery reference the workflow instance.

Covers:
- ``CheckpointBundle`` carries ``workflow_instance`` and round-trips it.
- ``_build_checkpoint_bundle`` populates it from ``hooks.workflow_instance``.
- ``RecoveryRecord`` carries ``workflow_instance_id`` / ``step_id``.
- ``step_blocks_instance`` helper composes the record's step with the instance graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kernel.models import CheckpointBundle, UnifiedState
from kernel.recovery import RecoveryIdentity, RecoveryRecord, step_blocks_instance
from kernel.turn_runner import TurnHooks, _build_checkpoint_bundle
from kernel.workflow_instance import WorkflowInstance, WorkflowStep


@dataclass(frozen=True, slots=True)
class _StubState:
    """Minimal state for _build_checkpoint_bundle."""

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"compiled_at": "2026-07-29", "context_markdown": "hello"}


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


def _instance() -> WorkflowInstance:
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


# ── CheckpointBundle ─────────────────────────────────────────────────


def test_checkpoint_bundle_carries_workflow_instance() -> None:
    """CheckpointBundle round-trips the workflow_instance field."""
    instance = _instance()
    bundle = CheckpointBundle(
        checkpoint_id="abc",
        created_at="2026-07-29T00:00:00Z",
        mode_id="coding",
        workflow_instance=instance.model_dump(mode="json"),
    )
    dumped = bundle.model_dump(mode="json")
    restored = CheckpointBundle.model_validate(dumped)
    assert restored.workflow_instance["instance_id"] == instance.instance_id
    assert restored.workflow_instance["current_step_id"] == "coding_1"


def test_checkpoint_bundle_defaults_empty_workflow_instance() -> None:
    """Default workflow_instance is an empty dict (no workflow in flight)."""
    bundle = CheckpointBundle(checkpoint_id="x", created_at="2026-01-01T00:00:00Z")
    assert bundle.workflow_instance == {}


def test_build_checkpoint_bundle_populates_workflow_instance() -> None:
    """_build_checkpoint_bundle reads hooks.workflow_instance into the bundle."""
    instance = _instance()
    hooks = TurnHooks(
        complete=None,  # not used by _build_checkpoint_bundle
        state_provider=lambda: _StubState(),
        workflow_instance=lambda: instance,
    )
    bundle = _build_checkpoint_bundle(
        mode_id="coding",
        transitioned_to=None,
        is_workflow=False,
        hooks=hooks,
    )
    assert bundle is not None
    assert bundle.workflow_instance["instance_id"] == instance.instance_id
    assert bundle.workflow_instance["current_step_id"] == "coding_1"


def test_build_checkpoint_bundle_no_instance_yields_empty_field() -> None:
    """When no workflow_instance is wired, the field is empty (not omitted)."""
    hooks = TurnHooks(
        complete=None,
        state_provider=lambda: _StubState(),
    )
    bundle = _build_checkpoint_bundle(
        mode_id="default",
        transitioned_to=None,
        is_workflow=False,
        hooks=hooks,
    )
    assert bundle is not None
    assert bundle.workflow_instance == {}


# ── RecoveryRecord ───────────────────────────────────────────────────


def test_recovery_record_carries_workflow_instance_id_and_step() -> None:
    """RecoveryRecord round-trips workflow_instance_id and step_id."""
    record = RecoveryRecord(
        identity=_identity(),
        mode_id="coding",
        target_kind="operation",
        target_id="read_file",
        workflow_instance_id="inst_123",
        step_id="coding_1",
    )
    dumped = record.model_dump(mode="json")
    restored = RecoveryRecord.model_validate(dumped)
    assert restored.workflow_instance_id == "inst_123"
    assert restored.step_id == "coding_1"


def test_recovery_record_defaults_empty_workflow_fields() -> None:
    """Default workflow_instance_id/step_id are empty (no workflow in flight)."""
    record = RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id="read_file",
    )
    assert record.workflow_instance_id == ""
    assert record.step_id == ""


# ── step_blocks_instance ─────────────────────────────────────────────


def test_step_blocks_instance_when_step_is_open_and_uncompleted() -> None:
    """A stalled record for an uncompleted declared step blocks the instance."""
    instance = _instance()  # current_step_id=coding_1, completed=search_0
    record = RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id="read_file",
        state="stalled",
        workflow_instance_id=instance.instance_id,
        step_id="coding_1",  # declared, not completed
    )
    assert step_blocks_instance(record, instance) is True


def test_step_blocks_instance_false_for_completed_step() -> None:
    """A stalled record for an already-completed step does not block (stale)."""
    instance = _instance()  # completed_steps=("search_0",)
    record = RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id="web_search",
        state="stalled",
        step_id="search_0",  # already completed
    )
    assert step_blocks_instance(record, instance) is False


def test_step_blocks_instance_false_when_no_step_id() -> None:
    """A record with no step_id never blocks (no multi-stage workflow)."""
    instance = _instance()
    record = RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id="read_file",
        state="stalled",
    )
    assert step_blocks_instance(record, instance) is False


def test_step_blocks_instance_false_when_no_instance() -> None:
    """When no instance is active, no record blocks."""
    record = RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id="read_file",
        state="stalled",
        step_id="coding_1",
    )
    assert step_blocks_instance(record, None) is False


def test_step_blocks_instance_false_for_succeeded_record() -> None:
    """A succeeded (non-open) record never blocks."""
    instance = _instance()
    record = RecoveryRecord(
        identity=_identity(),
        target_kind="operation",
        target_id="read_file",
        state="succeeded",
        step_id="coding_1",
    )
    assert step_blocks_instance(record, instance) is False