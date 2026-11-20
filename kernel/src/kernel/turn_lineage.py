"""One construction site for a child turn's identity (Evidence Envelope §4).

A spawned turn (fork child or delegate child) keeps its parent's ``turn_index``
— it belongs to the parent's thread turn — but mints its own ``turn_id`` and
records the two lineage edges that make the ledger a tree.

There is exactly ONE place that builds that shape. Before this module the two
spawn sites each inlined a ``dataclasses.replace`` with the same three keys, so
the invariant was reproduced by hand twice and a test that reproduced it a
third time could pass while production lost a key. Callers now share this
function and the test calls it too, so deleting a lineage edge here breaks the
build and the test together.
"""

from __future__ import annotations

from dataclasses import replace

from kernel.call_identity import mint_turn_id
from kernel.turn_hooks import TurnIdentity, TurnTaskGoal


def child_identity(
    parent: TurnIdentity,
    dispatch_id: str,
    *,
    task_goal: TurnTaskGoal | None = None,
) -> TurnIdentity:
    """Return the identity of a turn spawned by ``parent`` under ``dispatch_id``.

    Args:
        parent: The identity of the turn (or child) doing the spawning.
        dispatch_id: The fork/delegate dispatch that spawned this child, recorded
            as ``parent_dispatch_id`` so the child's rows join back to it.
        task_goal: The child's own declared goal, when it declares one (a
            delegate child does; a fork child inherits the parent's). ``None``
            inherits ``parent.task_goal``.

    Returns:
        ``parent`` with a fresh ``turn_id``, the two lineage edges set, and
        ``turn_index`` deliberately inherited.
    """
    return replace(
        parent,
        turn_id=mint_turn_id(),
        parent_turn_id=parent.turn_id,
        parent_dispatch_id=dispatch_id,
        task_goal=parent.task_goal if task_goal is None else task_goal,
    )


__all__ = ["child_identity"]
