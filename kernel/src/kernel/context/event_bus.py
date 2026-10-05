"""Default ``EventBus`` backed by in-process ``asyncio`` queues.

Replaces ``app.event_bus.BusBackend`` for public hosts. No external
dependencies, no persistence, no cross-process delivery. Production hosts
that need durable or cross-process events override this seam.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, object]], Awaitable[None]]


class InProcessEventBus:
    """Lightweight pub/sub keyed by topic string.

    Handlers are invoked on the running event loop. Exceptions in handlers
    are logged and swallowed so one failing handler does not block the
    publisher or sibling handlers.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register ``handler`` to receive payloads published to ``topic``."""

        self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove ``handler`` from ``topic``; unknown handlers are ignored."""

        handlers = self._handlers.get(topic)
        if handlers is None:
            return
        with contextlib.suppress(ValueError):
            handlers.remove(handler)

    async def publish(self, topic: str, payload: dict[str, object]) -> None:
        """Deliver one payload to all subscribed handlers for ``topic``."""

        handlers = list(self._handlers.get(topic, ()))
        for handler in handlers:
            try:
                await handler(payload)
            except Exception:  # pylint: disable=broad-except
                logger.exception(
                    "EventBus handler for topic %r raised; continuing.",
                    topic,
                )
