"""Loop-control tagged union for the kernel turn loop.

One vocabulary for "what does the loop do next" (F2 of the 2026-10-30 turn
loop structural review). Before this module, the same decision was expressed
as five incompatible encodings — a ``Literal["continue", "break",
"dispatch"]`` on ``_PreparedToolCall``, a bare ``"break"/"continue"`` return
from the executor, ``"dispatch"/"refuse"/"stop"`` from the repeated-batch
guard, and ``"break"/"continue"/""`` from the empty-completion resolver — so
reading the loop meant holding five control conventions in working memory.

This is the application-side sibling of the lifecycle resolver's tagged
union (``_lifecycle.LifecycleAction``): a lightweight dataclass sum type,
not a Pydantic model, because these are per-iteration loop control values,
not serialized state. Consumers dispatch with ``isinstance`` so an
unhandled variant is a static (mypy) or loud (``assert_never``) failure,
not a silently-falling-through string comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinueIteration:
    """Skip to the next iteration of the enclosing loop.

    In the tool-calls dispatch loop this advances to the next tool call; in
    the turn loop it starts the next while-iteration. A guard that has
    already appended a model-visible tool-result message (a refusal, a
    per-turn cap notice) uses this so the model can observe the outcome and
    finalize instead of being mistaken for a natural empty completion.
    """


@dataclass(frozen=True, slots=True)
class BreakIteration:
    """Leave the enclosing loop now.

        In the tool-calls dispatch loop this stops dispatching the remaining
    calls in the batch; in the turn loop it proceeds to finalization with the
        current ``content``.
    """


@dataclass(frozen=True, slots=True)
class DispatchToolBatch:
    """Dispatch the prepared tool call (the normal path)."""


@dataclass(frozen=True, slots=True)
class RefuseToolBatch:
    """Refuse the batch with a model-visible message and keep the loop alive.

    ``refusal`` is the tool-result text already appended for each call in
    the batch; carried for logging and diagnostics. The model gets another
    completion to observe the refusal and finalize.
    """

    refusal: str = ""


@dataclass(frozen=True, slots=True)
class StopLoop:
    """Stop the loop and force the empty-content synthesis path.

    ``reason`` is logged. Distinct from ``BreakIteration``: a stop is a
    guard verdict that the loop cannot make progress (e.g. the repeated
    tool batch re-issued past its cap), and the caller forces a no-tools
    synthesis instead of accepting the current content as final.
    """

    reason: str = ""


LoopControl = ContinueIteration | BreakIteration | DispatchToolBatch | RefuseToolBatch | StopLoop

__all__ = [
    "BreakIteration",
    "ContinueIteration",
    "DispatchToolBatch",
    "LoopControl",
    "RefuseToolBatch",
    "StopLoop",
]
