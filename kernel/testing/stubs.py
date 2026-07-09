"""Host-neutral stubs for extension and data-source testing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kernel.seams.protocols import DataSourceHookSet


class StaticStateProvider:
    """Simple state-provider callable returning one static value."""

    def __init__(self, state: Any | None) -> None:
        self._state = state

    def __call__(self) -> Any | None:
        return self._state


class NullMemoryStore:
    """No-op memory store stub implementing the expected async methods."""

    async def record_memory(self, request: Any) -> Any:
        del request
        return None

    async def list_recent_memories(
        self,
        *,
        user_slug: str,
        operation_id: str = "",
        limit: int = 20,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> list[Any]:
        del user_slug, operation_id, limit, workspace_id, memory_scope, surface_id
        return []

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[Any]:
        del user_slug, workspace_id, memory_scope, surface_id, limit
        return []

    async def recall(self, query: str, **kwargs: Any) -> Any:
        del query, kwargs
        return None


class NullThreadStore:
    """No-op thread store stub for DataSourceHooks contracts."""

    def iter_cached_states(self) -> list[Any]:
        return []


class NullEventBus:
    """No-op event bus stub with async publish."""

    async def publish(self, event: Any) -> None:
        del event


def make_data_source_hooks(
    *,
    outbox_reader: Any = None,
    state: Any | None = None,
    state_provider: Callable[[], Any | None] | None = None,
    memory_store: Any = None,
    thread_store: Any = None,
    event_bus: Any = None,
) -> DataSourceHookSet:
    """Build one hook bundle compatible with kernel.seams.protocols.DataSourceHooks."""
    resolved_state_provider = state_provider or StaticStateProvider(state)
    return DataSourceHookSet(
        state_provider=resolved_state_provider,
        memory_store=memory_store or NullMemoryStore(),
        thread_store=thread_store or NullThreadStore(),
        event_bus=event_bus or NullEventBus(),
        outbox_reader=outbox_reader,
    )
