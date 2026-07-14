"""Rich-based TransportCallbacks implementation for the CLI host.

This is the terminal-transport implementation of
:class:`hosts.core.callbacks.TransportCallbacks`. The CLI host constructs a
``RichTransport`` and passes it to ``hosts.core.handle_message``; the runtime
calls these methods during turn execution to deliver stage updates, streamed
text chunks, final responses, debug context, and errors.

Rich is allowed here — this is the terminal host. The core runtime never
imports Rich; it only calls these methods on the injected transport.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

logger = logging.getLogger(__name__)

# ── Visual style constants (kept in sync with the Little Lisa palette) ──
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_CONTEXT = "gold3"
STYLE_BORDER_ERROR = "indian_red"
STYLE_STAGE = "grey58"
STYLE_TEXT_MUTED = "grey58"


class RichTransport:
    """TransportCallbacks implementation backed by a Rich Console.

    Attributes:
        console: The Rich Console to render to. Defaults to a shared
            ``Console()`` instance; callers may pass a custom one (e.g.
            for testing or file redirection).
        debug: When True, ``on_debug_context`` renders a panel. When
            False, it is a no-op (matches the ``--debug`` flag).
    """

    def __init__(self, *, console: Console | None = None, debug: bool = False) -> None:
        self.console: Console = console or Console()
        self.debug = debug
        self._status: Status | None = None
        self._thinking_stage: str = ""

    def start_thinking(self, stage: str = "routing") -> None:
        """Start a status spinner so the user sees the turn is active.
        """
        self.stop_thinking()
        self._thinking_stage = stage
        self._status = self.console.status(f"thinking… ({stage})", spinner="dots", spinner_style=STYLE_STAGE)
        self._status.start()

    def stop_thinking(self) -> None:
        """Stop the thinking spinner if it is running."""
        status = self._status
        self._status = None
        if status is not None:
            try:
                status.stop()
            except Exception:  # pylint: disable=broad-except
                pass

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        """Update the status spinner with the current stage."""
        suffix = ""
        if data:
            # Compact one-line rendering of stage metadata.
            parts: list[str] = []
            for key in ("model_id", "provider", "mode_id", "iteration", "tool", "name"):
                value = data.get(key)
                if value not in (None, "", 0):
                    parts.append(f"{key}={value}")
            if parts:
                suffix = f" ({', '.join(parts)})"
                
        text = f"stage: {label}{suffix}"
        
        if self._status is not None:
            self._status.update(text)
        else:
            self.console.print(Text(text, style=STYLE_STAGE))

    async def on_text_chunk(self, text: str) -> None:
        """Print a streamed text chunk inline (no panel, no newline)."""
        self.stop_thinking()
        self.console.file.write(text)
        self.console.file.flush()

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        """Render the final response in a Rich Panel."""
        self.stop_thinking()
        self.console.print(
            Panel(
                Markdown(text),
                title="Response",
                border_style=STYLE_BORDER_RESPONSE,
            )
        )

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        """Render turn context in a Rich Panel (only when debug is on)."""
        if not self.debug:
            return
            
        from rich.pretty import Pretty
        self.console.print(
            Panel(
                Pretty(context, expand_all=True),
                title="Turn Context",
                border_style=STYLE_BORDER_CONTEXT,
            )
        )

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        """Render an error in a red Rich Panel."""
        self.stop_thinking()
        body = message
        if exc is not None:
            body = f"{type(exc).__name__}: {exc}\n\n{message}"
        logger.error("transport error: %s", body)
        self.console.print(
            Panel(
                body,
                title="Error",
                border_style=STYLE_BORDER_ERROR,
            )
        )


__all__ = ["RichTransport"]