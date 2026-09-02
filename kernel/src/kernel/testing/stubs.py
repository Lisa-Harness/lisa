"""Host-neutral stubs for extension testing."""

from __future__ import annotations

from typing import Any


class StaticStateProvider:
    """Simple state-provider callable returning one static value."""

    def __init__(self, state: Any | None) -> None:
        self._state = state

    def __call__(self) -> Any | None:
        return self._state


class NullEventBus:
    """No-op event bus stub with async publish."""

    async def publish(self, event: Any) -> None:
        del event
