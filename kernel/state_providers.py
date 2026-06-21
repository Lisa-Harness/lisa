"""Generic state-provider implementations for the kernel turn runner.

A ``StateProvider`` exposes three methods: ``__call__`` (returns the
latest :class:`UnifiedState` or ``None``), ``recall`` (memory
injection), and ``slice`` (opportunistic house-keeping).

This module ships a :class:`BlankStateProvider` for hosts that have no
state to expose. It returns ``None`` for state, no-op for recall, and
an empty :class:`SliceOutcome` for slice. The slice machinery still
fires; the provider just has nothing to do.

Hosts that need to drain real work items (reindex, eviction, sync)
should ship their own provider; see ``app/state/sql_state_provider.py``
for the app's SQLite-backed implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kernel.io.protocols import StateProvider
from kernel.slice import SliceOutcome, SlicePolicy


@dataclass(slots=True)
class BlankStateProvider:  # noqa: D401  (no public docstring needed)
    """State provider with no state and no work items.

    The slice machinery still fires (the turn runner calls ``slice()``
    on the pre/post/explicit triggers), but the provider returns an
    empty outcome. This is the right provider for hosts that have not
    yet wired opportunistic house-keeping — they get the slot, the
    overhead is effectively zero, and a future change can swap in a
    richer provider without touching the turn runner.
    """

    def __call__(self) -> Any:  # noqa: D401
        return None

    async def recall(  # pragma: no cover  (intentional no-op)
        self,
        ctx: Any,
        query: str,
        scope: str,
        budget_tokens: int,
    ) -> Any:
        return None

    async def write(  # pragma: no cover  (intentional no-op)
        self,
        ctx: Any,
        query: str,
        response: str,
        scope: str,
    ) -> None:
        return None

    async def slice(  # pragma: no cover  (intentional no-op)
        self, ctx: Any, policy: SlicePolicy
    ) -> SliceOutcome:
        return SliceOutcome(
            work_items_completed=0,
            elapsed_ms=0.0,
            last_item="",
            notes=(),
        )


def get_blank_state_provider() -> StateProvider:
    """Return a singleton :class:`BlankStateProvider` for hosts that
    do not need opportunistic house-keeping yet."""

    return BlankStateProvider()
