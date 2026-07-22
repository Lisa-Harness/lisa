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
        # The mode that currently owns the turn, tracked live across stage
        # events. Seeded from ``model_call.mode_id`` and updated when a
        # mid-turn handoff lands (``tool_dispatch_complete`` with
        # ``transition="handoff"`` and ``handoff_to``). Shown as the
        # ``[mode]`` prefix on every status line so the user always knows
        # who owns their turn — including mid-turn handoffs.
        self._current_mode: str = ""

    def start_thinking(self, stage: str = "dispatch") -> None:
        """Start a status spinner so the user sees the turn is active.

        The default stage is ``dispatch`` — the pre-pipeline phase where
        ``handle_message`` parses commands, resolves the ``start_mode``
        bypass, and selects ``current_mode_id``. There is no separate
        router; mode handoff happens mid-turn via ``transition_stage``, so
        ``routing`` would be a misnomer here.

        The dispatch phase uses a distinct spinner (``dots2``) so the user
        can tell "still deciding where this goes" from "model is
        generating" once the first ``model_call`` stage event arrives and
        switches to the normal ``dots`` spinner.
        """
        self.stop_thinking()
        self._thinking_stage = stage
        self._streamed_during_turn = False
        self._fork_children = {}
        self._fork_total = 0
        self._current_mode = ""
        is_dispatch = stage == "dispatch"
        spinner = "dots2" if is_dispatch else "dots"
        label = self._render_status_line(stage)
        self._status = self.console.status(
            label, spinner=spinner, spinner_style=STYLE_STAGE,
        )
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

        # ── Live mode tracking ───────────────────────────────────────
        # model_call carries the entry mode; a mid-turn handoff surfaces
        # via tool_dispatch_complete with transition="handoff" and
        # handoff_to. Update _current_mode so the [mode] prefix on every
        # subsequent status line reflects who owns the turn now.
        if label == "model_call" and data.get("mode_id"):
            self._current_mode = str(data["mode_id"])
        elif label == "tool_dispatch_complete":
            if data.get("transition") == "handoff" and data.get("handoff_to"):
                self._current_mode = str(data["handoff_to"])

        # ── Fork fan-out tracking ──────────────────────────────────────
        if label == "fork_start":
            self._fork_children = {}
            self._fork_total = int(data.get("children", 0)) or 0
            self._update_status(self._render_fork_stage())
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
            self._update_status(self._render_fork_stage())
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
            self._update_status(self._render_fork_stage())
            return
        if label == "fork_complete":
            # Keep the per-child table visible so the user sees *which*
            # children succeeded/failed, not just counts. The table is
            # cleared on the next start_thinking.
            ok = int(data.get("ok", 0))
            failed = int(data.get("failed", 0))
            self._fork_total = 0
            text = self._render_fork_stage(summary=(ok, failed))
            self._update_status(text)
            return

        # ── Normal stage line (detailed) ───────────────────────────────
        text = self._render_status_line(label, data)
        self._update_status(text)

    def _render_status_line(self, label: str, data: dict[str, Any] | None = None) -> str:
        """Render one status line, prefixed with the active mode.

        ``[mode] stage: <label> (…meta…)``. The mode prefix is the single
        most informative fact for the user; it updates live on mid-turn
        handoff so a ``default → coding`` switch is visible on the line.
        """
        data = data or {}
        prefix = f"[{self._current_mode}] " if self._current_mode else ""
        # tool_dispatch gets a legible kind glyph instead of a flat
        # tool_kind=… suffix key, so the user can tell an in-process
        # operation from an MCP call from a mid-turn handoff at a glance.
        if label == "tool_dispatch":
            body = self._render_tool_dispatch(data)
            return f"{prefix}{body}"
        suffix = self._render_stage_suffix(data)
        return f"{prefix}stage: {label}{suffix}"

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
            "provider", "model_id", "iteration",
            "tool_calls_so_far", "tool_calls",
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

    # ── tool_dispatch legible rendering ──────────────────────────────
    # The kernel tags every dispatch with tool_kind: "operation",
    # "pack_tool", "mcp_tool", or "transition". Render a compact glyph +
    # kind word so the user can distinguish in-process work, MCP calls,
    # and mid-turn handoffs/forks without parsing a flat key=value list.
    _TOOL_KIND_GLYPH: dict[str, str] = {
        "operation": "→ op",
        "pack_tool": "→ tool",
        "mcp_tool": "→ mcp",
        "transition": "↦",
    }

    def _render_tool_dispatch(self, data: dict[str, Any]) -> str:
        tool = str(data.get("tool") or "")
        kind = str(data.get("tool_kind") or "")
        glyph = self._TOOL_KIND_GLYPH.get(kind, "→")
        # Transitions (handoff/fork/workflow) are rendered with a target
        # suffix so the direction of the mid-turn move is explicit.
        if kind == "transition":
            target = ""
            if data.get("handoff_to"):
                target = f"→ {data['handoff_to']}"
            elif data.get("transition") == "fork":
                n = data.get("fork_children") or 0
                target = f"fork ({n} children)"
            elif data.get("transitioned_to"):
                target = f"→ {data['transitioned_to']}"
            suffix_parts = [f"{glyph} {tool}"]
            if target:
                suffix_parts.append(target)
            body = " ".join(suffix_parts)
        else:
            body = f"{glyph}: {tool}"
        meta: list[str] = []
        if data.get("iteration"):
            meta.append(f"iter={data['iteration']}")
        if data.get("tool_calls_so_far"):
            meta.append(f"calls={data['tool_calls_so_far']}")
        if meta:
            body += f" ({', '.join(meta)})"
        return f"stage: tool_dispatch {body}"

    def _render_fork_stage(self, summary: tuple[int, int] | None = None) -> str:
        """Render the fork fan-out as one line per active child.

        Example with two children::

            stage: fork (2 children)
              #0 coding · running · query preview…
              #1 markdown · running · query preview…

        When ``summary`` is set (called from ``fork_complete``), the
        per-child table is kept so the user sees *which* children
        succeeded/failed — not just counts. The header line carries the
        ok/failed totals in that case.
        """
        if not self._fork_children:
            if summary is not None:
                ok, failed = summary
                return f"stage: fork complete ({ok} ok, {failed} failed)"
            return f"stage: fork ({self._fork_total} children)"
        if summary is not None:
            ok, failed = summary
            header = f"stage: fork complete ({ok} ok, {failed} failed)"
        else:
            header = f"stage: fork ({self._fork_total} children)"
        lines = [header]
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