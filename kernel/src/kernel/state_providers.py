"""Generic state-provider implementations for the kernel turn runner.

The :class:`~kernel.seams.protocols.StateProvider` Protocol only formally
declares ``__call__`` (returns the latest :class:`UnifiedState` or
``None``). Hosts and packs that need memory injection or opportunistic
house-keeping extend that duck-typed contract with three additional
methods:

- ``recall(ctx, query, scope, budget_tokens)`` — inject ranked memory
  context into a turn.
- ``write(ctx, query, response, scope)`` — persist a memory record.
- ``slice(ctx, policy)`` — drain opportunistic work items (reindex,
  eviction, thread compaction, self-model) inside the pressure-driven
  :mod:`kernel.slice` budget. Returns a :class:`~kernel.slice.SliceOutcome`.

This module ships a :class:`BlankStateProvider` whose four methods are
all no-ops: ``__call__`` returns ``None``, ``recall`` returns ``None``,
``write`` returns ``None``, and ``slice`` returns an empty
:class:`~kernel.slice.SliceOutcome`. The slice machinery in
:mod:`kernel.turn_runner` still fires on the pre/post/explicit triggers
and calls ``slice()`` regardless of which provider is wired — the
blank provider simply reports zero work done. This keeps the turn
loop uniform: every host has a provider, and the slot is always
available; the cost is one function call returning a frozen dataclass.

Hosts that need real work items should ship their own provider. The
reference implementations are:

- ``host/state_provider.py`` → :class:`FileBackedStateProvider`: no-op
  state-provider slot (history is never compacted; the thread is lossless).
- ``sdk/packs/memory/slice_provider.py`` → memory-consolidation slice
  provider (episodic-pressure + sealed-segment mining).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kernel.seams.protocols import StateProvider
from kernel.slice import SliceOutcome, SlicePolicy


@dataclass(slots=True)
class BlankStateProvider:
    """State provider with no state and no work items.

    All four methods are no-ops:

    - ``__call__`` → ``None``: no :class:`UnifiedState` to expose.
    - ``recall`` → ``None``: no memory to inject into the turn.
    - ``write`` → ``None``: no memory store to persist to.
    - ``slice`` → empty :class:`~kernel.slice.SliceOutcome`: no
      opportunistic work items to drain.

    The slice machinery in :mod:`kernel.turn_runner` still fires on the
    pre-turn, post-turn, and explicit triggers and calls ``slice()``
    regardless of which provider is wired. The blank provider simply
    reports zero work completed, so the slot is occupied but the
    overhead is one function call returning a frozen dataclass.

    This is the right provider for hosts that have not yet wired
    opportunistic house-keeping: they get the slot, the cost is
    effectively zero, and a future change can swap in a richer
    provider (e.g. :class:`host.state_provider.FileBackedStateProvider`)
    without touching the turn runner.
    """

    def __call__(self) -> Any:
        """Return ``None`` — no :class:`UnifiedState` to expose."""
        return None

    async def recall(  # pragma: no cover  (intentional no-op)
        self,
        ctx: Any,
        query: str,
        scope: str,
        budget_tokens: int,
    ) -> Any:
        """Return ``None`` — no memory to inject into the turn.

        A real provider ranks memory records by semantic + lexical
        overlap and returns a rendered context string bounded by
        ``budget_tokens``. See :meth:`sdk.memory.store.MemoryStore.recall`
        for the reference ranking implementation.
        """
        return None

    async def write(  # pragma: no cover  (intentional no-op)
        self,
        ctx: Any,
        query: str,
        response: str,
        scope: str,
    ) -> None:
        """Return ``None`` — no memory store to persist to.

        A real provider distills the ``query``/``response`` pair into a
        durable memory record tagged with ``scope``. See the
        ``synthesize_and_store_learning`` operation for the reference
        distillation + persistence path.
        """
        return

    async def slice(  # pragma: no cover  (intentional no-op)
        self, ctx: Any, policy: SlicePolicy
    ) -> SliceOutcome:
        """Return an empty :class:`~kernel.slice.SliceOutcome`.

        The turn runner still invokes this on every wired trigger
        (pre-turn, post-turn, explicit); the blank provider simply
        reports zero work items completed within the ``policy``
        budget. A real provider drains as many work items as it can
        before ``policy.budget_ms`` elapses and returns the count.
        """
        return SliceOutcome(
            work_items_completed=0,
            elapsed_ms=0.0,
            last_item="",
            notes=(),
        )


def get_blank_state_provider() -> StateProvider:
    """Return a fresh :class:`BlankStateProvider` instance.

    This is the default provider for hosts that have not yet wired
    opportunistic house-keeping. The instance is stateless (a slotted
    dataclass with no fields), so constructing a new one per call is
    effectively free and avoids mutable shared state across turns.
    """

    return BlankStateProvider()
