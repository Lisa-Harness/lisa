"""Transport callback protocol — the Option B seam.

``TransportCallbacks`` is the interface that ``hosts.core`` uses to deliver
turn events to a transport host (CLI console, web SSE, VS Code Chat). The
runtime calls these methods during turn execution; the transport host
implements them to render output in its native surface.

All callbacks are optional (default to no-op). A transport host wires only
the events it cares about. The ``NullTransport`` implementation is the
default — it absorbs all events silently, useful for tests and headless
turns.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TransportCallbacks(Protocol):
    """Transport-facing callbacks invoked by the host runtime during a turn.

    All methods are async and optional. The runtime never blocks on a
    transport callback beyond awaiting it — if a transport is slow, the
    turn slows down, but the runtime does not time out the transport.

    Design principles:
    - **Structured payloads, not prose**: ``on_debug_context`` and
      ``on_stage`` receive structured dicts so VS Code/web can render rich
      UI without parsing console text.
    - **No transport imports in core**: the runtime calls these methods;
      it never imports Rich, aiohttp, or VS Code APIs.
    - **Streaming is opt-in**: ``on_text_chunk`` fires only when the
      runtime has streaming enabled. Non-streaming transports ignore it.
    - **Errors are delivered, not raised**: ``on_error`` receives the
      exception; the runtime catches it and continues. The transport
      decides whether to show it to the user.
    """

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        """Notify the transport of a stage transition in the turn pipeline.

        Called at key pipeline points: ``"model_call"``,
        ``"tool_loop"``, ``"post_turn"``, ``"slice"``. The optional ``data``
        dict carries stage-specific metadata (model id, provider, iteration
        count, etc.).
        """
        ...

    async def on_text_chunk(self, text: str) -> None:
        """Deliver one streamed text chunk from the model.

        Only fires when the runtime has text streaming enabled. Transports
        that do not stream can ignore this; they will receive the full
        response via ``on_response``.
        """
        ...

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        """Deliver the final response text for this turn.

        Called exactly once per successful turn. ``data`` may carry
        token counts, mode id, model id, and other turn metadata for
        transports that want to render a status line or usage panel.
        """
        ...

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        """Deliver debug context for the current turn.

        Called when debug mode is active. The ``context`` dict mirrors the
        ``_build_turn_context`` shape (timestamp, mode_id, query, resolved
        model, handoff note, etc.). Transports render this as they see fit
        (Rich panel, web JSON view, VS Code debug panel).
        """
        ...

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        """Deliver a non-fatal error that occurred during the turn.

        The runtime catches turn-level exceptions and routes them here.
        The transport decides whether to display them inline or in a
        diagnostics panel. Fatal errors (startup failures) still raise.
        """
        ...


@dataclass(slots=True)
class NullTransport:
    """No-op transport — absorbs all events silently.

    Default transport for headless turns, tests, and the ``handle_message``
    fast path where no streaming or debug rendering is needed.
    """

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        pass

    async def on_text_chunk(self, text: str) -> None:
        pass

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        pass

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        pass

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        pass


@dataclass
class DelegatingTransport:
    """Compose multiple transport callbacks into one.

    Each event is fanned out to all registered transports in order.
    Useful for hosts that want both a console transport and a logging
    transport active simultaneously.
    """

    transports: list[TransportCallbacks] = field(default_factory=list)

    def add(self, transport: TransportCallbacks) -> None:
        self.transports.append(transport)

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        for t in self.transports:
            await t.on_stage(label, data)

    async def on_text_chunk(self, text: str) -> None:
        for t in self.transports:
            await t.on_text_chunk(text)

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        for t in self.transports:
            await t.on_response(text, data)

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        for t in self.transports:
            await t.on_debug_context(context)

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        for t in self.transports:
            await t.on_error(message, exc)