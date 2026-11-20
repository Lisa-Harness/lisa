"""Mint and name the identifiers that join one piece of work to its cost.

A turn with several iterations makes several model calls, and some iterations
make more than one: a stream that needs the structured tool-call payload falls
back to ``complete()``, a truncated completion is retried at a doubled budget,
and a provider error is retried. Those extra calls are real spend, and before
this module the ledgers could not tell them apart \u2014 ``turn_input.calls``
recorded ``{iteration, ref}``, so two calls in one iteration were
indistinguishable and a decision could not name the call it resolved.

Identity is a tuple, not a counter
A call is identified by ``(turn_id, iteration, attempt)``. ``attempt`` names
*why* the call happened rather than which one it was in arrival order, so two
records that describe the same call agree on its identity even when they are
written at different times or by different layers. ``call_id`` is the minted
handle for that tuple, unique within its turn.

Dispatch identity is separate
The dispatch identifier (:func:`mint_dispatch_id`) belongs to the tool or
operation side, where there is no provider call to attribute: a rejected
dispatch still gets a ``dispatch_id`` so its approval row and its transcript
tool result can address each other.

Why these are not hashes of content
Identifiers are opaque and content-free by design. They are safe to put on a
durable row precisely because they carry no prompt, response, or argument text.
"""

from __future__ import annotations

import uuid
from typing import Literal

#: Why one model call happened inside its iteration. The set is closed: an
#: attempt value a reader does not know is a bug, not a new case to guess at.
CallAttempt = Literal["primary", "stream_tool_fallback", "truncation_continuation"]

#: The only attempt values a row may carry.
CALL_ATTEMPTS: frozenset[str] = frozenset(
    {"primary", "stream_tool_fallback", "truncation_continuation"}
)

#: The iteration's first call. Every other attempt is an extra call inside the
#: same iteration and must carry one of the other reasons.
PRIMARY_ATTEMPT: CallAttempt = "primary"
#: The streaming path fell back to ``complete()`` for the tool-call payload.
STREAM_TOOL_FALLBACK_ATTEMPT: CallAttempt = "stream_tool_fallback"
#: A truncation or reasoning-burnout rescue continued a cut-off completion.
TRUNCATION_CONTINUATION_ATTEMPT: CallAttempt = "truncation_continuation"


def mint_call_id() -> str:
    """Return a fresh ``call_id`` for one model call.

    Called when the request is built, so the identifier exists before the call
    is made and a failed call still has an identity to record.
    """
    return f"call-{uuid.uuid4().hex}"


def mint_dispatch_id() -> str:
    """Return a fresh ``dispatch_id`` for one tool or operation dispatch.

    Minted for rejected dispatches too: that is what lets a rejection row and
    its transcript tool result address each other.
    """
    return f"disp-{uuid.uuid4().hex}"


def mint_turn_id() -> str:
    """Return a fresh ``turn_id`` for one execution (Evidence Envelope §4).

    Minted once at turn start so every row the turn emits — ledger rows, the
    thread entry, and any child turn's ``parent_turn_id`` — can name the same
    execution. Without it a turn is only addressable by ``turn_index``, which
    is a position in one thread and cannot distinguish two overlapping turns
    or say which execution spawned a child.
    """
    return f"turn-{uuid.uuid4().hex}"


__all__ = [
    "CALL_ATTEMPTS",
    "PRIMARY_ATTEMPT",
    "STREAM_TOOL_FALLBACK_ATTEMPT",
    "TRUNCATION_CONTINUATION_ATTEMPT",
    "CallAttempt",
    "mint_call_id",
    "mint_dispatch_id",
    "mint_turn_id",
]
