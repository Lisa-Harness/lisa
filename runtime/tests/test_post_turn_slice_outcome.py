"""Tests for the post-turn slice outcome composition in ``_build_slice_callables``.

The 2026-10-03 regression: the refactor that extracted
``_compose_self_model_outcome`` to module level left ``await`` on its call
site in ``_post_turn_slice``. ``_compose_self_model_outcome`` is a plain
synchronous function returning a ``SliceOutcome`` — the ``await`` raised
``TypeError: object SliceOutcome can't be used in 'await' expression`` on
every post-turn slice, so the kernel's ``_fire_slice`` swallowed the failure
and discarded both the self-model outcome and the chained memory
consolidation results every turn.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from kernel.slice import SliceMode, SliceOutcome, SlicePolicy
from runtime.state_provider import FileBackedStateProvider
from runtime.turn_hooks_builder import _build_slice_callables, _compose_self_model_outcome


def _make_runtime(tmp_path: Path) -> Any:
    from runtime.testing import build_test_runtime

    return build_test_runtime(
        tmp_path,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )


def _policy() -> SlicePolicy:
    return SlicePolicy(mode=SliceMode.POST_TURN, budget_ms=50)


@pytest.mark.asyncio
async def test_post_turn_slice_returns_outcome(tmp_path: Path) -> None:
    """The post-turn slice callable must return a SliceOutcome, not raise.

    Before the fix, ``await`` on the synchronous ``_compose_self_model_outcome``
    raised ``TypeError`` before the outcome could be returned.
    """
    runtime = _make_runtime(tmp_path)
    provider = FileBackedStateProvider(thread_path=None)
    (
        _introspect_fn,
        _run_self_model_slice,
        _pre_turn_slice,
        post_turn_slice,
        _explicit_slice,
    ) = _build_slice_callables(runtime, provider, lambda model_id: None)

    assert inspect.iscoroutinefunction(post_turn_slice)

    outcome = await post_turn_slice(_policy())

    assert isinstance(outcome, SliceOutcome)
    # The file provider reports zero work (no-op compaction) and the
    # self-model returns zero work on a healthy stubbed runtime, so the
    # base outcome passes through unchanged.
    assert outcome.work_items_completed == 0


def test_compose_self_model_outcome_is_synchronous() -> None:
    """``_compose_self_model_outcome`` is a plain sync function.

    Guards against reintroducing ``await`` at the call site: awaiting its
    return value raises ``TypeError`` because it is not awaitable.
    """
    assert not inspect.iscoroutinefunction(_compose_self_model_outcome)

    base = SliceOutcome(work_items_completed=2, elapsed_ms=10.0, last_item="base", notes=("n",))
    composed = _compose_self_model_outcome(base, (3, "self_model", "health=degraded"))

    assert composed.work_items_completed == 5
    assert composed.last_item == "self_model"
    assert composed.notes == ("n", "health=degraded")

    with pytest.raises(TypeError):
        # Demonstrate the failure mode the call site must not reintroduce:
        # awaiting the plain SliceOutcome.
        import asyncio

        async def _await_it() -> Any:
            return await _compose_self_model_outcome(base, (0, "", ""))

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_await_it())
