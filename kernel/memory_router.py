"""MemoryStore router — generic aliases to a default backend.

The router is the piece that makes "call the generic, get the default
backend" work for both operations and the model-facing synthetic tools
(``memory.add`` / ``memory.search`` / ``memory.list``). It implements
:class:`~kernel.memory_protocol.MemoryStore` and delegates to whichever
backend the host wired as default.

Callers that want a specific backend (e.g. explicitly hitting Mem0's
semantics instead of the generic default) use :meth:`backend` to fetch it
by name, bypassing the router.
"""

from __future__ import annotations

from kernel.memory_protocol import MemoryStore
from kernel.models import MemoryRecord, MemoryWriteRequest

__all__ = ["MemoryStoreRouter"]


class MemoryStoreRouter:
    """Routes generic memory calls to whichever backend is wired as default.

    Modes and operations hold a reference to this (typed as
    :class:`MemoryStore`). The host constructs it with a ``default`` backend
    (the one backing the generic aliases) and an optional ``backends`` map
    for direct named access via :meth:`backend`.

    The router is a thin pass-through: it adds no ranking, deduplication, or
    fallback logic. Ranking lives in the turn hooks
    (``LittleLisaTurnHooks.answer_memory_recall``); fallback (when the
    default backend is unavailable) is a host concern, handled at wiring
    time by choosing which client to pass as ``default``.
    """

    def __init__(
        self,
        default: MemoryStore,
        backends: dict[str, MemoryStore] | None = None,
    ) -> None:
        self._default = default
        self._backends: dict[str, MemoryStore] = dict(backends or {})
        # The default backend is always reachable by name too.
        self._backends.setdefault("__default__", default)

    @property
    def default(self) -> MemoryStore:
        """The backend the generic aliases currently route to."""
        return self._default

    def backend(self, name: str) -> MemoryStore | None:
        """Direct access to a named backend, bypassing the generic router.

        Returns ``None`` when no backend is registered under ``name``. Use
        this when a caller explicitly wants a specific provider's semantics
        (e.g. Mem0's graph recall) rather than whatever the host wired as
        default.
        """
        return self._backends.get(name)

    def registered_backends(self) -> list[str]:
        """Names of all backends known to the router (for diagnostics)."""
        return sorted(self._backends)

    async def record_memory(self, request: MemoryWriteRequest) -> MemoryRecord:
        return await self._default.record_memory(request)

    async def recall(self, query: str, **kwargs: object) -> str:
        return await self._default.recall(query, **kwargs)

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[MemoryRecord]:
        return await self._default.list_memories(
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            limit=limit,
        )

    # Convenience aliases used only by lightweight tests/examples that hold
    # a router where they used to hold an MCPMemoryClient.
    async def store(self, query: str, response: str) -> None:
        store_fn = getattr(self._default, "store", None)
        if callable(store_fn):
            await store_fn(query, response)

    async def list_recent_memories(self, **kwargs: object) -> list[MemoryRecord]:
        list_fn = getattr(self._default, "list_recent_memories", None)
        if callable(list_fn):
            return list(list_fn(**kwargs))
        return []