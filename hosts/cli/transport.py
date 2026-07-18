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
        # True once on_text_chunk fires during the current turn. Reset in
        # start_thinking so each turn starts clean. When True, on_response
        # skips the Rich Panel because the text is already on screen —
        # printing it again would double-render (raw inline + formatted panel)
        # with a misaligned header.
        self._streamed_during_turn: bool = False
        # Fork fan-out live state. Populated by ``on_stage("fork_*")`` so the
        # stage line can render one line per active child instead of a single
        # coarse "fork" label. Reset by ``fork_start`` / ``fork_complete`` /
        # ``start_thinking``.
        self._fork_children: dict[int, dict[str, Any]] = {}
        self._fork_total: int = 0

    def start_thinking(self, stage: str = "routing") -> None:
        """Start a status spinner so the user sees the turn is active."""
        self.stop_thinking()
        self._thinking_stage = stage
        self._streamed_during_turn = False
        self._fork_children = {}
        self._fork_total = 0
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
        """Update the status spinner with the current stage.

        Renders a single detailed line for normal stages. When a fork
        fan-out is active, renders one line per child so the user sees
        every parallel call progressing independently.
        """
        data = data or {}

        # ── Fork fan-out tracking ──────────────────────────────────────
        if label == "fork_start":
            self._fork_children = {}
            self._fork_total = int(data.get("children", 0)) or 0
            text = self._render_fork_stage()
            self._update_status(text)
            return
        if label == "fork_child_start":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                self._fork_children[idx] = {
                    "mode_id": data.get("mode_id", ""),
                    "stage_id": data.get("stage_id", ""),
                    "status": "running",
                    "query_preview": data.get("query_preview", ""),
                }
            text = self._render_fork_stage()
            self._update_status(text)
            return
        if label == "fork_child_complete":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                existing = self._fork_children.get(idx, {})
                existing.update({
                    "status": data.get("status", "ok"),
                    "duration_ms": data.get("duration_ms"),
                })
                self._fork_children[idx] = existing
            text = self._render_fork_stage()
            self._update_status(text)
            return
        if label == "fork_complete":
            ok = int(data.get("ok", 0))
            failed = int(data.get("failed", 0))
            self._fork_children = {}
            self._fork_total = 0
            text = f"stage: fork complete ({ok} ok, {failed} failed)"
            self._update_status(text)
            return

        # ── Normal stage line (detailed) ───────────────────────────────
        suffix = self._render_stage_suffix(data)
        text = f"stage: {label}{suffix}"
        self._update_status(text)

    def _update_status(self, text: str) -> None:
        """Push ``text`` to the spinner, or print it inline if no spinner."""
        if self._status is not None:
            self._status.update(text)
        else:
            self.console.print(Text(text, style=STYLE_STAGE))

    def _render_stage_suffix(self, data: dict[str, Any]) -> str:
        """Compact one-line rendering of stage metadata."""
        parts: list[str] = []
        for key in (
            "mode_id", "provider", "model_id", "iteration",
            "tool_calls_so_far", "tool", "tool_kind", "tool_calls",
            "duration_ms", "stage_id", "turn_count",
        ):
            value = data.get(key)
            if value in (None, "", 0):
                continue
            parts.append(f"{key}={value}")
        # Token counts are nested under model_call_complete.
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = data.get(key)
            if value:
                parts.append(f"{key}={value}")
        if parts:
            return f" ({', '.join(parts)})"
        return ""

    def _render_fork_stage(self) -> str:
        """Render the fork fan-out as one line per active child.

        Example with two children::

            stage: fork (2 children)
              #0 coding · running · query preview…
              #1 markdown · running · query preview…
        """
        if not self._fork_children:
            return f"stage: fork ({self._fork_total} children)"
        lines = [f"stage: fork ({self._fork_total} children)"]
        for idx in sorted(self._fork_children):
            child = self._fork_children[idx]
            mode_id = child.get("mode_id", "?")
            stage_id = child.get("stage_id", "")
            status = child.get("status", "running")
            stage_part = f"{stage_id} · " if stage_id else ""
            line = f"  #{idx} {mode_id} · {stage_part}{status}"
            preview = child.get("query_preview", "")
            if preview:
                line += f" · {preview}"
            duration_ms = child.get("duration_ms")
            if duration_ms is not None:
                line += f" · {duration_ms}ms"
            lines.append(line)
        return "\n".join(lines)

    async def on_text_chunk(self, text: str) -> None:
        """Print a streamed text chunk inline (no panel, no newline)."""
        self.stop_thinking()
        self._streamed_during_turn = True
        self.console.file.write(text)
        self.console.file.flush()

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        """Render the final response in a Rich Panel.

        When text was already streamed inline via ``on_text_chunk`` during
        this turn, skip the panel (the response is already visible) and just
        emit a trailing newline to separate the streamed text from the next
        prompt. The per-turn flag is reset in ``start_thinking``.
        """
        self.stop_thinking()
        if self._streamed_during_turn:
            self.console.file.write("\n")
            self.console.file.flush()
            return
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
        if self._streamed_during_turn:
            self.console.file.write("\n")
            self.console.file.flush()
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