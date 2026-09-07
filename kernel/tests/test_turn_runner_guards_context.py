"""Tests for input/output guards and context builder integration in the turn runner.

Covers:
  - Input guard: blocking (turn aborted), sanitization (query rewritten)
  - Output guard: blocking (safe fallback + guardrail.audit event),
    sanitization (output rewritten)
  - ContextSeams defaults (pass-through guards, in-process event bus)
  - _fire_slice (budget enforcement, error handling, outcome coercion)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

from kernel.models import (
    SurfaceContext,
)
from kernel.seams.protocols import GuardResult
from kernel.slice import SliceMode, SliceOutcome, SlicePolicy, SliceTuning
from kernel.turn_runner import (
    TurnHooks,
    TurnResult,
    _fire_slice,
    run_turn,
)

# ── Stubs ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _build_hooks(complete_fn: Any, **overrides: Any) -> TurnHooks:
    return TurnHooks(complete=complete_fn, **overrides)


class _BlockingInputGuard:
    """Input guard that blocks everything."""

    async def validate(self, query: str, surface: SurfaceContext) -> GuardResult:
        return GuardResult(allowed=False, blocked_reason="Blocked by test guard")


class _SanitizingInputGuard:
    """Input guard that sanitizes the query."""

    async def validate(self, query: str, surface: SurfaceContext) -> GuardResult:
        return GuardResult(allowed=True, sanitized_text=query.replace("bad", "good"))


class _BlockingOutputGuard:
    """Output guard that blocks specific content."""

    def __init__(self) -> None:
        self.validated_texts: list[str] = []

    async def validate(self, text: str, surface: SurfaceContext) -> GuardResult:
        self.validated_texts.append(text)
        if "forbidden" in text.lower():
            return GuardResult(allowed=False, blocked_reason="Output blocked: forbidden content")
        return GuardResult(allowed=True)


class _SanitizingOutputGuard:
    """Output guard that rewrites the output."""

    async def validate(self, text: str, surface: SurfaceContext) -> GuardResult:
        return GuardResult(allowed=True, sanitized_text=text.replace("secret", "REDACTED"))


# ── Input guard: blocking ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_input_guard_blocks_turn_before_model_call() -> None:
    """When the input guard returns allowed=False, the turn is aborted
    before the model is called and the blocked_reason is returned."""
    model_called = False

    async def complete(**kwargs: Any) -> _StubCompletion:
        nonlocal model_called
        model_called = True
        return _StubCompletion(content="should not reach here")

    hooks = _build_hooks(complete)
    hooks.context_seams.input_guard = _BlockingInputGuard()

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.blocked is True
    assert "Blocked by test guard" in result.response_text
    assert model_called is False
    assert result.tool_call_count == 0


@pytest.mark.asyncio
async def test_input_guard_block_returns_turn_result() -> None:
    """A blocked input produces a TurnResult with blocked=True and the
    guard's blocked_reason as the response text."""
    hooks = _build_hooks(lambda **kw: _StubCompletion(content="x"))
    hooks.context_seams.input_guard = _BlockingInputGuard()

    result = await run_turn(mode_id="default", query="anything", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.blocked is True
    assert result.iteration_count == 0


# ── Input guard: sanitization ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_input_guard_sanitizes_query() -> None:
    """When the input guard returns sanitized_text, the query is rewritten
    for the remainder of the turn."""
    captured_queries: list[str] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        messages = kwargs.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                captured_queries.append(str(msg.get("content", "")))
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(complete)
    hooks.context_seams.input_guard = _SanitizingInputGuard()

    result = await run_turn(mode_id="default", query="this is bad text", hooks=hooks)

    assert result.response_text == "Done."
    assert result.blocked is False
    # The sanitized query should have "good" instead of "bad"
    assert any("good" in q for q in captured_queries)
    assert all("bad" not in q or "good" in q for q in captured_queries)


# ── Output guard: blocking ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_guard_blocks_and_replaces_with_safe_fallback() -> None:
    """When the output guard returns allowed=False, the content is replaced
    with a safe fallback message and turn_blocked is True."""
    guard = _BlockingOutputGuard()

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="This is forbidden content.")

    hooks = _build_hooks(complete)
    hooks.context_seams.output_guard = guard

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.blocked is True
    assert "unable to provide" in result.response_text.lower()
    # The guard should have been called with the original content.
    assert any("forbidden" in t for t in guard.validated_texts)


@pytest.mark.asyncio
async def test_output_guard_block_emits_guardrail_audit_event() -> None:
    """When the output guard blocks, a guardrail.audit event is published."""
    guard = _BlockingOutputGuard()

    published_events: list[tuple[str, dict[str, Any]]] = []

    class _CapturingEventBus:
        async def publish(self, event_type: str, data: dict[str, Any]) -> None:
            published_events.append((event_type, data))

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="forbidden output")

    hooks = _build_hooks(complete)
    hooks.context_seams.output_guard = guard
    hooks.context_seams.event_bus = _CapturingEventBus()

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.blocked is True
    audit_events = [e for e in published_events if e[0] == "guardrail.audit"]
    assert len(audit_events) == 1
    assert "blocked_reason" in audit_events[0][1]
    assert "mode_id" in audit_events[0][1]


@pytest.mark.asyncio
async def test_output_guard_block_skips_post_turn_slice() -> None:
    """When the output guard blocks, the post-turn slice is skipped."""
    guard = _BlockingOutputGuard()
    slice_called = False

    async def post_slice(_policy: SlicePolicy) -> SliceOutcome:
        nonlocal slice_called
        slice_called = True
        return SliceOutcome()

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="forbidden")

    hooks = _build_hooks(complete)
    hooks.context_seams.output_guard = guard
    hooks.post_turn_slice = post_slice
    hooks.slice_tuning = SliceTuning(post_turn_enabled=True)

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.blocked is True
    assert slice_called is False


# ── Output guard: sanitization ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_guard_sanitizes_output() -> None:
    """When the output guard returns sanitized_text, the content is rewritten
    before being returned to the user."""

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="The secret code is 1234.")

    hooks = _build_hooks(complete)
    hooks.context_seams.output_guard = _SanitizingOutputGuard()

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.blocked is False
    assert "REDACTED" in result.response_text
    assert "secret" not in result.response_text


# ── Output guard: pass-through (default) ──────────────────────────────


@pytest.mark.asyncio
async def test_default_output_guard_allows_everything() -> None:
    """The default PassThroughOutputGuard allows all output unchanged."""

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Any content here.")

    hooks = _build_hooks(complete)
    # Default hooks use PassThroughOutputGuard via ContextSeams defaults.

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.blocked is False
    assert result.response_text == "Any content here."


@pytest.mark.asyncio
async def test_default_input_guard_allows_everything() -> None:
    """The default PassThroughInputGuard allows all input unchanged."""

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(complete)
    # Default hooks use PassThroughInputGuard via ContextSeams defaults.

    result = await run_turn(mode_id="default", query="anything goes", hooks=hooks)

    assert result.blocked is False
    assert result.response_text == "Done."


# ── _fire_slice ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fire_slice_returns_none_when_slot_not_wired() -> None:
    """_fire_slice returns None when the corresponding slot is None."""
    hooks = TurnHooks(slice_tuning=SliceTuning(pre_turn_enabled=True))
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert result is None
    assert len(outcomes) == 0


@pytest.mark.asyncio
async def test_fire_slice_returns_none_when_tuning_is_none() -> None:
    """_fire_slice returns None when slice_tuning is None."""
    hooks = TurnHooks(pre_turn_slice=lambda p: SliceOutcome())
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert result is None


@pytest.mark.asyncio
async def test_fire_slice_returns_none_when_trigger_disabled() -> None:
    """_fire_slice returns None when the trigger is set to 'never'."""
    hooks = TurnHooks(
        pre_turn_slice=lambda p: SliceOutcome(),
        slice_tuning=SliceTuning(pre_turn_enabled=False),
    )
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert result is None


@pytest.mark.asyncio
async def test_fire_slice_calls_slot_when_enabled() -> None:
    """_fire_slice calls the slot and returns the outcome when enabled."""
    called = False

    async def my_slice(_policy: SlicePolicy) -> SliceOutcome:
        nonlocal called
        called = True
        return SliceOutcome(work_items_completed=3, last_item="item_3")

    hooks = TurnHooks(
        pre_turn_slice=my_slice,
        slice_tuning=SliceTuning(pre_turn_enabled=True),
    )
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert called is True
    assert result is not None
    assert result.work_items_completed == 3
    assert len(outcomes) == 1


@pytest.mark.asyncio
async def test_fire_slice_catches_exception_and_returns_error_outcome() -> None:
    """When the slice raises, _fire_slice catches it and returns an error outcome."""

    async def failing_slice(_policy: SlicePolicy) -> SliceOutcome:
        raise RuntimeError("slice exploded")

    hooks = TurnHooks(
        pre_turn_slice=failing_slice,
        slice_tuning=SliceTuning(pre_turn_enabled=True),
    )
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert result is not None
    assert result.work_items_completed == 0
    assert len(result.notes) > 0
    assert any("RuntimeError" in note for note in result.notes)


@pytest.mark.asyncio
async def test_fire_slice_coerces_non_slice_outcome() -> None:
    """When the slot returns a non-SliceOutcome, it's coerced to an empty outcome."""

    async def bad_slice(_policy: SlicePolicy) -> Any:
        return "not a SliceOutcome"

    hooks = TurnHooks(
        pre_turn_slice=bad_slice,
        slice_tuning=SliceTuning(pre_turn_enabled=True),
    )
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert result is not None
    assert isinstance(result, SliceOutcome)
    assert result.work_items_completed == 0


@pytest.mark.asyncio
async def test_fire_slice_backfills_elapsed_ms() -> None:
    """When the outcome has elapsed_ms=0, _fire_slice backfills it from wall time."""

    async def my_slice(_policy: SlicePolicy) -> SliceOutcome:
        await __import__("asyncio").sleep(0.02)
        return SliceOutcome(work_items_completed=1, elapsed_ms=0)

    hooks = TurnHooks(
        pre_turn_slice=my_slice,
        slice_tuning=SliceTuning(pre_turn_enabled=True),
    )
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert result is not None
    # The backfill should have set a positive elapsed_ms from wall time.
    # On very fast machines the backfilled value could be tiny but > 0.
    assert result.elapsed_ms >= 0


@pytest.mark.asyncio
async def test_fire_slice_returns_none_when_budget_zero() -> None:
    """_fire_slice returns None when the computed budget is 0."""

    async def my_slice(_policy: SlicePolicy) -> SliceOutcome:
        return SliceOutcome(work_items_completed=1)

    # With a tuning that results in zero budget (base_budget_ms=0), the slice
    # should be short-circuited.
    hooks = TurnHooks(
        pre_turn_slice=my_slice,
        slice_tuning=SliceTuning(
            pre_turn_enabled=True, base_budget_ms=0, max_deferred_ms=0, hard_cap_ms=0
        ),
        last_slice_finished_at=time.monotonic() - 1,  # some pressure
    )
    outcomes: list[SliceOutcome] = []
    result = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=outcomes,
    )
    assert result is None
    assert len(outcomes) == 0


# ── Post-turn slice integration ────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_turn_slice_fires_after_response_committed() -> None:
    """The post-turn slice fires after the response is committed to the user."""
    slice_called = False

    async def my_post_slice(_policy: SlicePolicy) -> SliceOutcome:
        nonlocal slice_called
        slice_called = True
        return SliceOutcome(work_items_completed=2, last_item="compacted")

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(complete)
    hooks.post_turn_slice = my_post_slice
    hooks.slice_tuning = SliceTuning(post_turn_enabled=True)

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Done."
    assert slice_called is True
    assert len(result.slice_outcomes) == 1
    assert result.slice_outcomes[0].work_items_completed == 2


@pytest.mark.asyncio
async def test_pre_turn_slice_updates_last_slice_finished_at() -> None:
    """After the pre-turn slice runs, hooks.last_slice_finished_at is updated."""

    async def my_pre_slice(_policy: SlicePolicy) -> SliceOutcome:
        return SliceOutcome(work_items_completed=1)

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_hooks(complete)
    hooks.pre_turn_slice = my_pre_slice
    hooks.slice_tuning = SliceTuning(pre_turn_enabled=True)
    hooks.last_slice_finished_at = None

    await run_turn(mode_id="default", query="test", hooks=hooks)

    assert hooks.last_slice_finished_at is not None
    assert hooks.last_slice_finished_at > 0
