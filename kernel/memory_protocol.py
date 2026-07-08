"""Provider-agnostic memory seam.

The kernel talks to memory backends (lisa-context, Mem0, Letta, …) through
this Protocol, not through a concrete client class. Modes and operations
hold a reference typed as :class:`MemoryStore`; the host injects whichever
backend is wired as default (see
:class:`~kernel.memory_router.MemoryStoreRouter`).

This Protocol makes the existing implicit duck-type seam explicit:
:class:`~kernel.mcp_client.MCPMemoryClient` already satisfies it, and
:class:`~kernel.mcp_client.Mem0MemoryClient` is written to satisfy it too.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kernel.models import MemoryRecord, MemoryWriteRequest

__all__ = ["MemoryStore"]


@runtime_checkable
class MemoryStore(Protocol):
    """Provider-agnostic memory backend.

    Every memory backend implements this surface so the kernel can swap
    implementations (lisa-context SQLite, Mem0 hosted/self-hosted, Letta,
    …) without touching modes, operations, or turn hooks.

    All methods are async. Implementations are responsible for their own
    subprocess / network lifecycle.
    """

    async def record_memory(self, request: MemoryWriteRequest) -> MemoryRecord:
        """Persist one memory record. Returns the stored record (with any
        backend-assigned id / metadata)."""
        ...

    async def recall(self, query: str, **kwargs: object) -> str:
        """Free-text recall. Returns ranked memory content as a string;
        the caller is responsible for higher-level synthesis (see
        :class:`~kernel.models.MemoryRecallAnswer`)."""
        ...

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """List recent memory records for one user scope."""
        ...