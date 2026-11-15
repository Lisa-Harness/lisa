"""Typed identity carrier for one turn's ``TurnHooks`` (Evidence Envelope §4).

Why the type exists
-------------------
The hook builder used to read the identity-bearing fields out of a free-form
``dict[str, Any]`` with ``.get(...) or ""`` coercion. That coercion is the
defect this type removes: a caller that forgot a key got a *blank* identity
rather than an error, so ledger rows recorded ``turn_id=None`` on paths that
typechecked, linted, and passed every test. ``parent_turn_id`` and
``parent_dispatch_id`` survived a whole slice on the in-memory chain while
being structurally unable to reach a row for exactly this reason.

Every field is required, including the ones whose legitimate value is the
empty string (a top-level turn has no parent) and ``turn_index`` (``None`` is
real: work not attributable to a thread turn). Requiring them does not demand
a *populated* identity — it demands that a caller **state** the identity it is
passing, so an omission is a ``TypeError`` at the call site instead of a null
on disk.

This is an input type owned by the runtime (it carries host-side turn context
into the runtime's hook builder), so it must not live in ``kernel``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TurnContextIdentity:
    """The execution identity and lineage a turn's hooks are built from.

    Attributes:
        turn_id: The execution this turn's ledger rows and thread entry name.
        turn_index: The thread ordinal the turn ran under, or ``None`` when the
            work is not attributable to a thread turn (off-turn slice work, or
            a host with no thread).
        task_id: The task this execution belongs to.
        task_goal: The stated goal of that task (a delegate child declares its
            own; nothing else inherits it).
        parent_turn_id: The turn that spawned this one, or ``""`` at top level.
        parent_dispatch_id: The exact fork/delegate dispatch that spawned this
            one, or ``""`` at top level.
    """

    turn_id: str
    turn_index: int | None
    task_id: str
    task_goal: str
    parent_turn_id: str
    parent_dispatch_id: str


#: The identity of work that belongs to no turn: no execution id, no thread
#: position, no task, no parent. Named so a call site that genuinely has no
#: turn context says so *explicitly* instead of relying on omitted arguments.
NO_TURN_IDENTITY = TurnContextIdentity(
    turn_id="",
    turn_index=None,
    task_id="",
    task_goal="",
    parent_turn_id="",
    parent_dispatch_id="",
)

__all__ = ["NO_TURN_IDENTITY", "TurnContextIdentity"]
