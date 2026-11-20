"""Shared remaining-wall arithmetic for delegate and fork child turns.

Both child kinds must honor the parent turn's wall budget: the parent's
``turn_start_mono`` anchor plus ``TurnBudget.max_wall_ms`` bounds how much
wall time any child may consume. The delegate path already enforced this;
the fork path did not (the 2026-11-18 review finding: fork children could
outlive the parent's wall budget because only the operation timeout was
armed). This module owns the one calculation both paths share.

This module must not import ``kernel.turn_runner`` or
``kernel.turn_phases._defaults`` (which import ``_fork``): it is the shared
leaf both call.
"""

from __future__ import annotations

import time

from kernel.models import TurnBudget


def remaining_wall_ms(
    *,
    turn_budget: TurnBudget | None,
    turn_start_mono: float | None,
    now: float | None = None,
) -> int | None:
    """Return the parent's unspent wall budget in ms, or ``None``.

    ``None`` means no bound applies: no budget, an unlimited wall field
    (``max_wall_ms == 0``), or an unanchored start. Real turns always
    capture a positive ``time.monotonic()`` anchor; ``turn_start_mono <= 0``
    is the documented unanchored sentinel for non-turn callers and must
    stay unbounded, never spent.

    Otherwise returns ``max(0, max_wall_ms - elapsed_ms)`` using
    ``time.monotonic()`` unless the test-only ``now`` is supplied.
    """
    if turn_budget is None or turn_budget.max_wall_ms <= 0:
        return None
    if turn_start_mono is None or turn_start_mono <= 0:
        return None
    current = time.monotonic() if now is None else now
    elapsed_ms = int((current - turn_start_mono) * 1000)
    return max(0, turn_budget.max_wall_ms - elapsed_ms)


def child_wall_ms(
    *,
    armed_deadline_ms: int,
    parent_remaining_ms: int | None,
    grace_ms: int,
) -> int:
    """Return the wall ceiling one child may run under, tighten-only.

    The candidate is the minimum of the operation's armed deadline
    (``timeout_seconds - grace``, already computed) and the parent's
    remaining wall minus the same grace. A ``None`` or non-positive
    candidate does not constrain the child; a candidate can never
    increase an existing bound. Zero means no ceiling applies — the
    caller checks the spent-parent condition (``parent_remaining_ms <=
    grace_ms``) before consulting this helper, so a spent parent never
    reaches a child launch.
    """
    candidates: list[int] = []
    if armed_deadline_ms > 0:
        candidates.append(armed_deadline_ms)
    if parent_remaining_ms is not None:
        remaining_child_ms = max(0, parent_remaining_ms - grace_ms)
        if remaining_child_ms > 0:
            candidates.append(remaining_child_ms)
    if not candidates:
        return 0
    return min(candidates)
