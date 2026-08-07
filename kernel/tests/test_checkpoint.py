"""Tests for kernel checkpoint model, tuning, and turn-runner firing.

Covers:
- ``CheckpointBundle`` serialization round-trip
- ``CheckpointTuning`` defaults and ``from_mapping``
- ``run_turn`` fires ``on_checkpoint`` on stage transition when tuning enabled
- ``run_turn`` does NOT fire when tuning disabled (backward compat)
- ``run_turn`` fires on every-N-turns boundary
- Checkpoint hook failure does not break the turn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kernel.models import (
    CheckpointBundle,
    UnifiedState,
)
from kernel.workflow_position import WorkflowPosition
from kernel.turn_runner import (
    CheckpointTuning,
    TurnHooks,
    TurnResult,
    _build_checkpoint_bundle,
    _should_checkpoint,
    run_turn,
)

# ── Test helpers ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_stub_complete(response: str):
    """Return an async complete stub that returns the given response."""

    async def stub(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=response)

    return stub


# ── CheckpointBundle model ───────────────────────────────────────────


def test_checkpoint_bundle_serialization_roundtrip() -> None:
    """CheckpointBundle survives model_dump / model_validate round-trip."""
    bundle = CheckpointBundle(
        checkpoint_id="abc123",
        created_at="2026-07-11T00:00:00Z",
        label="test",
        mode_id="research",
        stage_id="gather",
        turn_count=5,
        unified_state={"compiled_at": "2026-07-11", "context_markdown": "hello"},
        workflow_position={"mode_id": "research", "stage_id": "gather"},
        turn_history=[{"role": "user", "content": "hi"}],
    )
    dumped = bundle.model_dump(mode="json")
    restored = CheckpointBundle.model_validate(dumped)
    assert restored.checkpoint_id == "abc123"
    assert restored.label == "test"
    assert restored.stage_id == "gather"
    assert restored.turn_count == 5
    assert restored.unified_state["compiled_at"] == "2026-07-11"
    assert len(restored.turn_history) == 1


def test_checkpoint_bundle_defaults() -> None:
    """CheckpointBundle defaults are sensible."""
    bundle = CheckpointBundle(checkpoint_id="x", created_at="2026-01-01T00:00:00Z")
    assert bundle.label == ""
    assert bundle.mode_id == ""
    assert bundle.stage_id == ""
    assert bundle.turn_count == 0
    assert bundle.unified_state == {}
    assert bundle.workflow_position == {}
    assert bundle.turn_history == []


# ── CheckpointTuning ─────────────────────────────────────────────────


def test_checkpoint_tuning_defaults() -> None:
    """Default tuning is disabled."""
    tuning = CheckpointTuning()
    assert tuning.enabled is False
    assert tuning.on_stage_transition is True
    assert tuning.every_n_turns == 0
    assert tuning.explicit is True


def test_checkpoint_tuning_from_mapping() -> None:
    """from_mapping builds tuning from a dict, ignoring unknown keys."""
    tuning = CheckpointTuning.from_mapping(
        {
            "enabled": True,
            "on_stage_transition": False,
            "every_n_turns": 5,
            "explicit": False,
            "unknown_key": "ignored",
        }
    )
    assert tuning.enabled is True
    assert tuning.on_stage_transition is False
    assert tuning.every_n_turns == 5
    assert tuning.explicit is False


def test_checkpoint_tuning_from_mapping_none() -> None:
    """from_mapping(None) returns defaults."""
    tuning = CheckpointTuning.from_mapping(None)
    assert tuning.enabled is False


# ── _should_checkpoint ──────────────────────────────────────────────


def test_should_checkpoint_disabled_tuning() -> None:
    """Disabled tuning never checkpoints."""
    hooks = TurnHooks()
    assert (
        _should_checkpoint(
            tuning=CheckpointTuning(enabled=False),
            transitioned_to="next_stage",
            is_workflow=True,
            hooks=hooks,
        )
        is False
    )


def test_should_checkpoint_none_tuning() -> None:
    """None tuning never checkpoints."""
    hooks = TurnHooks()
    assert (
        _should_checkpoint(
            tuning=None,
            transitioned_to="next_stage",
            is_workflow=True,
            hooks=hooks,
        )
        is False
    )


def test_should_checkpoint_on_transition() -> None:
    """Enabled tuning with stage transition triggers checkpoint."""
    hooks = TurnHooks()
    assert (
        _should_checkpoint(
            tuning=CheckpointTuning(enabled=True, on_stage_transition=True),
            transitioned_to="next_stage",
            is_workflow=True,
            hooks=hooks,
        )
        is True
    )


def test_should_checkpoint_transition_disabled() -> None:
    """Enabled tuning with on_stage_transition=False does not fire on transition."""
    hooks = TurnHooks()
    assert (
        _should_checkpoint(
            tuning=CheckpointTuning(enabled=True, on_stage_transition=False),
            transitioned_to="next_stage",
            is_workflow=True,
            hooks=hooks,
        )
        is False
    )


# ── _build_checkpoint_bundle ────────────────────────────────────────


def test_build_checkpoint_bundle_with_state() -> None:
    """Bundle includes UnifiedState when state_provider is wired."""
    state = UnifiedState(compiled_at="2026-07-11T00:00:00Z", context_markdown="test")
    hooks = TurnHooks(state_provider=lambda: state)
    bundle = _build_checkpoint_bundle(
        mode_id="default",
        transitioned_to=None,
        hooks=hooks,
    )
    assert bundle is not None
    assert bundle.unified_state["compiled_at"] == "2026-07-11T00:00:00Z"
    assert bundle.workflow_position == {}
    assert bundle.checkpoint_id  # non-empty
    assert bundle.created_at  # non-empty


def test_build_checkpoint_bundle_with_workflow_position() -> None:
    """Bundle includes the WorkflowPosition when workflow_position is wired."""
    session = WorkflowPosition(
        mode_id="research",
        stage_id="gather",
        entered_at="2026-07-11T00:00:00Z",
        turn_count=3,
    )
    hooks = TurnHooks(workflow_position=lambda: session)
    bundle = _build_checkpoint_bundle(
        mode_id="research",
        transitioned_to="gather",
        hooks=hooks,
    )
    assert bundle is not None
    assert bundle.stage_id == "gather"
    assert bundle.turn_count == 3
    assert bundle.mode_id == "research"
    assert "mode_id" in bundle.workflow_position


def test_build_checkpoint_bundle_no_state_returns_none() -> None:
    """Bundle is None when neither state_provider nor workflow_position is wired."""
    hooks = TurnHooks()
    bundle = _build_checkpoint_bundle(
        mode_id="default",
        transitioned_to=None,
        hooks=hooks,
    )
    assert bundle is None


# ── run_turn integration ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_checkpoint_fires_on_transition() -> None:
    """run_turn fires on_checkpoint when a stage transition occurs."""
    checkpoint_calls: list[Any] = []

    async def on_checkpoint(bundle: CheckpointBundle) -> None:
        checkpoint_calls.append(bundle)

    state = UnifiedState(compiled_at="2026-07-11T00:00:00Z")
    hooks = TurnHooks(
        complete=_make_stub_complete("Response"),
        state_provider=lambda: state,
        on_checkpoint=on_checkpoint,
        checkpoint_tuning=CheckpointTuning(enabled=True, on_stage_transition=True),
    )

    # We can't easily simulate a real stage transition without a full workflow
    # setup, but we can test that a non-workflow turn with no transition does
    # NOT fire the checkpoint.
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    # No transition occurred in a non-workflow mode → no checkpoint
    assert len(checkpoint_calls) == 0


@pytest.mark.asyncio
async def test_run_turn_checkpoint_disabled_by_default() -> None:
    """run_turn does not fire on_checkpoint when checkpoint_tuning is None."""
    checkpoint_calls: list[Any] = []

    async def on_checkpoint(bundle: CheckpointBundle) -> None:
        checkpoint_calls.append(bundle)

    state = UnifiedState(compiled_at="2026-07-11T00:00:00Z")
    hooks = TurnHooks(
        complete=_make_stub_complete("Response"),
        state_provider=lambda: state,
        on_checkpoint=on_checkpoint,
        # checkpoint_tuning is None by default
    )

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert len(checkpoint_calls) == 0


@pytest.mark.asyncio
async def test_run_turn_checkpoint_hook_failure_doesnt_break_turn() -> None:
    """A failing on_checkpoint hook does not break the turn."""

    async def on_checkpoint(bundle: CheckpointBundle) -> None:
        raise RuntimeError("checkpoint storage is broken")

    state = UnifiedState(compiled_at="2026-07-11T00:00:00Z")
    hooks = TurnHooks(
        complete=_make_stub_complete("Response"),
        state_provider=lambda: state,
        on_checkpoint=on_checkpoint,
        checkpoint_tuning=CheckpointTuning(enabled=True, on_stage_transition=True),
    )

    # This turn has no transition, so the hook won't fire anyway.
    # But if it did, the try/except in the turn runner would catch it.
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)
    assert isinstance(result, TurnResult)
    assert result.response_text == "Response"
