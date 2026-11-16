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
    """No-op event bus stub with async publish.

    Mirrors the production ``EventBus.publish(topic, payload)`` signature so
    the fire-and-forget telemetry scheduler accepts it.
    """

    async def publish(self, topic: str, payload: Any) -> None:
        del topic, payload
