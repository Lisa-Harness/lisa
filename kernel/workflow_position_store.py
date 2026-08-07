"""Kernel contract for durable workflow-position storage.

The kernel defines only the seam: ``WorkflowPositionStore`` is a Protocol
the host implements. The file-backed implementation lives in
``host/workflow_position_store.py`` — same split as
``kernel/recovery.py`` (contracts) vs ``host/recovery.py``
(``FileRecoveryService``).

The kernel never touches the store directly — the host owns the position
lifecycle (create at mode entry, advance on stage/mode moves, clear on
completion, load on boot). The Protocol mirrors ``RecoveryService`` so the
two stores compose the same way.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kernel.recovery import RecoveryIdentity
from kernel.workflow_position import WorkflowPosition

__all__ = ["WorkflowPositionStore"]


@runtime_checkable
class WorkflowPositionStore(Protocol):
    """Host-owned durable store for workflow positions.

    The kernel never touches the store directly — the host owns the
    position lifecycle (create at mode entry, advance on stage/mode moves,
    clear on completion, load on boot). The Protocol mirrors
    ``RecoveryService`` so the two stores compose the same way.
    """

    async def save(
        self, position: WorkflowPosition, identity: RecoveryIdentity | None = None
    ) -> WorkflowPosition:
        """Persist (create or update) one position.

        ``identity`` scopes the position to one user/surface so
        ``load_active`` returns the right workflow per identity. When
        ``None`` the position is stored unscoped (kernel-only tests).
        """

    async def load(self, position_id: str, identity: RecoveryIdentity) -> WorkflowPosition | None:
        """Return one position by id when its identity matches exactly."""

    async def load_active(self, identity: RecoveryIdentity) -> WorkflowPosition | None:
        """Return the single non-complete position for this identity, if any."""

    async def list_open(self, identity: RecoveryIdentity) -> list[WorkflowPosition]:
        """List non-complete positions for this identity, newest first."""

    async def clear(self, position_id: str, identity: RecoveryIdentity) -> bool:
        """Remove one position. Return True when a position was removed."""
