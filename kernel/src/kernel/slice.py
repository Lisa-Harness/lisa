"""Opportunistic house-keeping for the kernel turn runner.

The kernel exposes three turn-anchored slice slots (pre_turn_slice,
post_turn_slice, explicit_slice) and one pressure-driven budget model
that determines how much work the slots can do on a given turn.

DESIGN — pressure, not a clock:

    The slice budget is a function of how long it has been since the last
    slice finished, not a fixed per-turn value. A user who idles for an
    hour and then runs one turn pays the deferred cost on that turn
    (clamped by a hard cap). A user hammering the REPL pays the minimum
    per turn. The curve is monotonic, so the user can predict the
    behavior from the knobs alone.

    A slice is bounded by ``budget_ms`` and the framework short-circuits
    the call if the deadline has passed. A provider's ``slice()`` drains
    as many work items as it can within the budget and returns. There is
    no background scheduler, no wall-clock tick, no cron.

DESIGN — trigger modes are orthogonal to pressure:

    The pressure curve is the mechanism. The trigger mode is the
    policy. A host can wire any subset of the three slice slots:

      - ``pre_turn``: slice runs before the model call. Useful for
        warming caches so the model's first response reflects fresh
        state.
      - ``post_turn``: slice runs after the response is committed.
        Default for hosts that want zero latency overhead on the user
        response.
      - ``explicit``: slice only runs when an operation or CLI command
        invokes it. Useful for "I want to control when this happens."

    ``never`` (or simply not wiring the hook) is a fourth valid state
    that means "purposeful neglect" — no opportunistic work at all.

The slice is owned by a ``StateProvider`` (see ``kernel/io/protocols.py``).
This module only owns the *policy*: the budget math and the trigger
wiring. Providers own the work items.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Any


class SliceMode(enum.StrEnum):
    """When a slice runs within a turn."""

    NEVER = "never"
    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class SlicePolicy:
    """Bounded-budget policy for one opportunistic house-keeping call.

    Attributes:
        mode: Which trigger the slice is firing under.
        budget_ms: Total wall-clock budget for this slice. The provider
            MUST short-circuit when ``elapsed_ms`` exceeds this value.
        base_budget_ms: Minimum slice size per turn. Even at zero
            pressure the slice gets this much time so micro-jobs
            (e.g. one row eviction) can complete.
        max_deferred_ms: Maximum pressure contribution. The pressure
            curve asymptotically approaches this value but is clamped
            by ``hard_cap_ms``.
        hard_cap_ms: Absolute ceiling on the per-turn slice cost. The
            user response latency is the upper bound on this number.
        pressure_half_life_seconds: Time after which the pressure
            contribution is halfway to its asymptote. Smaller = the
            system catches up faster.
        pre_turn_share: When both pre_turn and post_turn are wired, the
            fraction of the budget that the pre_turn slice consumes
            before the model call. The remainder goes to post_turn.
    """

    mode: SliceMode
    budget_ms: int
    base_budget_ms: int = 10
    max_deferred_ms: int = 200
    hard_cap_ms: int = 500
    pressure_half_life_seconds: float = 900.0
    pre_turn_share: float = 0.3
    metadata: dict[str, Any] = field(default_factory=dict)

    def split_for_trigger(self, *, has_pre_turn: bool, has_post_turn: bool) -> SlicePolicy:
        """Return a new policy with ``budget_ms`` adjusted for the trigger pair.

        When both pre_turn and post_turn are wired, the pre_turn slice
        gets ``budget_ms * pre_turn_share`` and the post_turn slice
        gets the remainder (``budget_ms * (1 - pre_turn_share)``). If
        only one is wired, the budget is passed through unchanged.
        The ``mode`` field of the returned policy is preserved; this
        method only adjusts the per-trigger allocation.
        """

        if not (has_pre_turn and has_post_turn):
            return self
        if self.mode == SliceMode.PRE_TURN:
            share = self.pre_turn_share
        elif self.mode == SliceMode.POST_TURN:
            share = 1.0 - self.pre_turn_share
        else:
            return self
        return SlicePolicy(
            mode=self.mode,
            budget_ms=int(self.budget_ms * share),
            base_budget_ms=self.base_budget_ms,
            max_deferred_ms=self.max_deferred_ms,
            hard_cap_ms=self.hard_cap_ms,
            pressure_half_life_seconds=self.pressure_half_life_seconds,
            pre_turn_share=self.pre_turn_share,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class SliceOutcome:
    """What one opportunistic slice did.

    Returned by a provider's ``slice()`` and recorded in the turn
    telemetry so the user can see how much opportunistic work is
    actually happening.
    """

    work_items_completed: int = 0
    elapsed_ms: float = 0.0
    last_item: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SliceTuning:
    """Host-level knobs for the slice policy.

    All fields have sensible defaults so a host can construct an empty
    ``SliceTuning()`` and let the framework fill in the rest. The
    default curve is "asymptotic with a 15-minute half-life, capped at
    500 ms, minimum 10 ms per turn."
    """

    base_budget_ms: int = 10
    max_deferred_ms: int = 200
    hard_cap_ms: int = 500
    pressure_half_life_seconds: float = 900.0
    pre_turn_share: float = 0.3
    pre_turn_enabled: bool = False
    post_turn_enabled: bool = True
    explicit_enabled: bool = False

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> SliceTuning:
        """Build a tuning from a host config dict. Unknown keys are ignored."""
        if not mapping:
            return cls()
        return cls(
            base_budget_ms=int(mapping.get("base_budget_ms", 10) or 10),
            max_deferred_ms=int(mapping.get("max_deferred_ms", 200) or 200),
            hard_cap_ms=int(mapping.get("hard_cap_ms", 500) or 500),
            pressure_half_life_seconds=float(
                mapping.get("pressure_half_life_seconds", 900.0) or 900.0
            ),
            pre_turn_share=float(mapping.get("pre_turn_share", 0.3) or 0.3),
            pre_turn_enabled=bool(mapping.get("pre_turn_enabled", False)),
            post_turn_enabled=bool(mapping.get("post_turn_enabled", True)),
            explicit_enabled=bool(mapping.get("explicit_enabled", False)),
        )


def compute_budget_ms(
    *,
    now_monotonic: float,
    last_slice_finished_at: float | None,
    tuning: SliceTuning,
) -> int:
    """Return the wall-clock budget (in ms) for the next opportunistic slice.

    Pressure model::

        pressure = max_deferred_ms * (1 - 0.5 ^ (gap / half_life))
        budget   = clamp(base + pressure, 0, hard_cap)

    When ``last_slice_finished_at`` is None (first turn of a session),
    the pressure is treated as fully decayed and the budget collapses
    to ``base_budget_ms``. This is the right starting point: nothing
    has been deferred yet, so a tiny slice is enough.

    When the gap is larger than the half-life, the pressure approaches
    ``max_deferred_ms`` but is clamped by ``hard_cap_ms``. The user
    always sees a bounded response time.
    """
    base = max(0, int(tuning.base_budget_ms))
    cap = max(base, int(tuning.hard_cap_ms))
    max_deferred = max(0, int(tuning.max_deferred_ms))
    half_life = max(1.0, float(tuning.pressure_half_life_seconds))
    if last_slice_finished_at is None:
        pressure = 0.0
    else:
        gap = max(0.0, float(now_monotonic) - float(last_slice_finished_at))
        # 0.5 ** (gap / half_life) decays from 1 (fresh) to 0 (long ago).
        # 1 - that is the accumulated pressure fraction.
        pressure = max_deferred * (1.0 - math.pow(0.5, gap / half_life))
    budget = base + pressure
    return int(min(cap, max(0, budget)))


def resolve_mode_for_trigger(
    *,
    tuning: SliceTuning,
    trigger: SliceMode,
) -> SliceMode:
    """Return the effective mode for a trigger, or SliceMode.NEVER if disabled.

    The kernel calls this for each of the three trigger slots before
    computing the per-turn budget. A host that has not enabled
    ``pre_turn`` gets ``NEVER`` back and the framework skips the call
    entirely.
    """
    if trigger == SliceMode.PRE_TURN and not tuning.pre_turn_enabled:
        return SliceMode.NEVER
    if trigger == SliceMode.POST_TURN and not tuning.post_turn_enabled:
        return SliceMode.NEVER
    if trigger == SliceMode.EXPLICIT and not tuning.explicit_enabled:
        return SliceMode.NEVER
    return trigger


def build_policy_for_trigger(
    *,
    trigger: SliceMode,
    tuning: SliceTuning,
    last_slice_finished_at: float | None,
    has_pre_turn: bool = False,
    has_post_turn: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SlicePolicy:
    """Compose a full ``SlicePolicy`` for one trigger slot on this turn.

    The framework calls this twice per turn (once for pre_turn, once
    for post_turn) and any number of times for explicit calls. The
    caller passes ``has_pre_turn`` and ``has_post_turn`` so the
    ``pre_turn_share`` split is applied correctly.
    """
    effective_mode = resolve_mode_for_trigger(tuning=tuning, trigger=trigger)
    if effective_mode == SliceMode.NEVER:
        return SlicePolicy(
            mode=SliceMode.NEVER,
            budget_ms=0,
            base_budget_ms=tuning.base_budget_ms,
            max_deferred_ms=tuning.max_deferred_ms,
            hard_cap_ms=tuning.hard_cap_ms,
            pressure_half_life_seconds=tuning.pressure_half_life_seconds,
            pre_turn_share=tuning.pre_turn_share,
            metadata=metadata or {},
        )
    budget_ms = compute_budget_ms(
        now_monotonic=time.monotonic(),
        last_slice_finished_at=last_slice_finished_at,
        tuning=tuning,
    )
    policy = SlicePolicy(
        mode=trigger,
        budget_ms=budget_ms,
        base_budget_ms=tuning.base_budget_ms,
        max_deferred_ms=tuning.max_deferred_ms,
        hard_cap_ms=tuning.hard_cap_ms,
        pressure_half_life_seconds=tuning.pressure_half_life_seconds,
        pre_turn_share=tuning.pre_turn_share,
        metadata=metadata or {},
    )
    return policy.split_for_trigger(
        # The wiring shape is the same for both pre_turn and
        # post_turn calls; ``split_for_trigger`` looks at
        # ``self.mode`` to decide which share to apply.
        has_pre_turn=has_pre_turn,
        has_post_turn=has_post_turn,
    )


__all__ = [
    "SliceMode",
    "SliceOutcome",
    "SlicePolicy",
    "SliceTuning",
    "build_policy_for_trigger",
    "compute_budget_ms",
    "resolve_mode_for_trigger",
]
