"""Rich-based TransportCallbacks implementation for the CLI host.

This is the terminal-transport implementation of
:class:`hosts.core.callbacks.TransportCallbacks`. The CLI host constructs a
``RichTransport`` and passes it to ``hosts.core.handle_message``; the runtime
calls these methods during turn execution to deliver stage updates, streamed
text chunks, final responses, debug context, and errors.

Rich is allowed here — this is the terminal host. The core runtime never
imports Rich; it only calls these methods on the injected transport.

Layout: one flowing transcript
------------------------------
The turn renders as a single flowing column of rows — no panes, headers, or
boxes. Each stage event becomes one left-aligned line; tool calls are
one-liners that update in place as they complete; the streamed answer flows
into the transcript and *persists* (the fully formatted Markdown panel appears
only when the turn did not stream). Telemetry (pip counters, duration
sparklines, token suffixes) is opt-in via the detail tiers below.

Detail tiers
------------
* ``clean`` (default) — stage lines and tool one-liners with status glyphs.
* ``power`` (``Ctrl+O`` toggle) — the same rows plus the graphic column
  (pip counters, duration sparklines, token-flow suffixes).
* ``debug`` (``--debug``, construction-only) — the full operator rendering:
  every bookkeeping stage event gets a row with all metadata, plus the
  ``on_debug_context`` panel.

Stage-line legend
-----------------
* ``[mode]`` prefix — the mode that currently owns the turn; updates live on a
  mid-turn handoff.
* ``→ op / → tool / → mcp / ↦`` — tool-kind glyphs (in-process operation, pack
  tool, MCP call, transition).
* ``✓ / ✗ / …`` — tool-completion status glyphs.
* Power/debug tiers add: ``●``/``○`` pip counters, block-ramp duration
  sparklines (``▁▂▃▄▅▆▇█``), and token suffixes (``↑``/``↓``/``Σ``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

logger = logging.getLogger(__name__)

# ── Visual style constants (kept in sync with the Lisa palette) ──
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_CONTEXT = "gold3"
STYLE_BORDER_ERROR = "indian_red"
STYLE_STAGE = "grey58"
STYLE_TEXT_MUTED = "grey58"
STYLE_GRAPHIC = "grey58"
STYLE_ROW_TRANSITION = "gold3"
STYLE_ROW_MODEL = "dark_sea_green4"
STYLE_ROW_TOOL = "cornflower_blue"
STYLE_ROW_ERROR = "indian_red"
STYLE_STREAM_ANSWER = "bright_white"
STYLE_STREAM_REASONING = "grey42"

#: Lines reserved below the Live region for post-turn content that must stay
#: visible: the turn footer (1), the Rule separator (1), the powerline status
#: line (1), and the input prompt line (1).
_RESERVED_BELOW_LINES = 4

#: Floor for the number of transcript rows kept visible in the Live region.
#: Guarantees a usable view on very short terminals / non-tty consoles.
_MIN_VISIBLE_ROWS = 6

#: Maximum width of the pip counter. When the cap exceeds this, the pip row is
#: capped at this width and saturates with ``⁺``; when the cap is smaller, the
#: row tracks the cap exactly. When the cap is 0 (unlimited), the pip row is a
#: compact ``N`` counter (no bar), since a saturating bar against infinity is
#: meaningless.
_PIP_MAX_WIDTH = 16

#: Width of the duration sparkline shown on ``model_call_complete``.
_SPARK_WIDTH = 8

#: Block ramp used for both the duration bar and the sparkline. Index 0 is the
#: empty cell; index 7 is the full block.
_BLOCK_RAMP = " ▁▂▃▄▅▆▇█"

#: Gradient ramp for the pip counter fill: light → medium → dark → full
#: block, left to right across the filled portion. Index 0 is the empty cell
#: used for the unfilled tail; the remaining cells graduate the fill.
_PIP_GRADIENT = "░▒▓█"


def _pip_counter(count: int, cap: int) -> str:
    """Render ``count`` as a gradient block bar that tracks the cap.

    ``cap > 0`` draws a bar of ``_PIP_MAX_WIDTH`` cells whose filled fraction is
    ``count / cap`` (clamped to full); the cap is rendered faithfully no matter
    how large it is, so a 300-call budget does not saturate after 16 calls. The
    filled portion graduates left-to-right from ``░`` (light) through ``▒``/``▓``
    to ``█`` (full block) so the bar reads as a filling gradient; the unfilled
    tail is ``░``. The bar saturates with ``⁺`` only when ``count`` actually
    exceeds ``cap``. ``cap <= 0`` (unlimited) renders a compact ``N{count}``
    counter. Used only on the ``power``/``debug`` detail tiers.
    """
    if cap <= 0:
        return f"N{count}"
    width = _PIP_MAX_WIDTH
    if count >= cap:
        full = "█" * width
        return full + "⁺" if count > cap else full
    filled = round(count * width / cap)
    if filled >= width:
        filled = width - 1 if count < cap else width
    if filled <= 0:
        return "░" * width
    # Graduate the filled cells across the gradient ramp (light → full).
    # The last filled cell maps to the full block; earlier cells ramp through
    # the lighter shades so the fill reads as a gradient, not a hard edge.
    cells: list[str] = []
    grad = _PIP_GRADIENT  # ░▒▓█
    for i in range(filled):
        # Map position 0..filled-1 onto 1..len(grad)-1 (skip the empty cell at
        # index 0); the last filled cell is always the full block (█).
        idx = len(grad) - 1 if filled == 1 else round((i + 1) * (len(grad) - 1) / (filled - 1))
        cells.append(grad[max(1, min(idx, len(grad) - 1))])
    return "".join(cells) + "░" * (width - filled)


#: Glyphs for tool-dispatch completion status.
_TOOL_STATUS_GLYPH = {"ok": "✓", "error": "✗", "running": "…", "": "·"}

#: Maximum length of the primary tool argument shown inline on a tool row.
_ARG_PREVIEW_MAX = 40


def _sparkline(values: list[float], width: int = _SPARK_WIDTH) -> str:
    """Render a single-line unicode sparkline of ``values``.

    Each value maps to one block-ramp cell, normalized against the max of
    ``values`` (or 1.0 when all are zero). The most recent ``width`` values are
    kept. Returns an empty string when ``values`` is empty.
    """
    if not values:
        return ""
    recent = values[-width:]
    peak = max(recent) if recent else 0.0
    if peak <= 0:
        peak = 1.0
    cells: list[str] = []
    for v in recent:
        ratio = max(0.0, min(1.0, float(v) / peak))
        idx = round(ratio * (len(_BLOCK_RAMP) - 1))
        cells.append(_BLOCK_RAMP[idx])
    return "".join(cells)


def _block_bar(value: float, reference: float = 0.0) -> str:
    """Render a single block-ramp cell for ``value`` against ``reference``.

    When ``reference <= 0`` the value maps onto a 0–1000ms heuristic scale so
    the first iteration still shows a proportional cell.
    """
    if value <= 0:
        return _BLOCK_RAMP[0]
    scale = reference if reference > 0 else 1000.0
    ratio = max(0.0, min(1.0, value / scale))
    idx = round(ratio * (len(_BLOCK_RAMP) - 1))
    return _BLOCK_RAMP[idx]


def _truncate(value: str, limit: int = _ARG_PREVIEW_MAX) -> str:
    """Truncate ``value`` to ``limit`` chars with an ellipsis."""
    value = value.replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


class _TrailingSpinner:
    """A Live-compatible renderable that puts the spinner *after* the line.

    The fixed line content renders first (so it stays aligned with every other
    row in the transcript) and a single spinner frame glyph is appended after a
    one-space gap as a trailing activity indicator.
    """

    __slots__ = ("_line", "_spinner", "_style")

    def __init__(self, line: Text, spinner_name: str, style: str) -> None:
        self._line = line
        self._spinner = Spinner(spinner_name, style=style)
        self._style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.render(console.get_time())

    def render(self, time: float) -> RenderableType:
        frame = self._spinner.render(time)
        glyph = frame.plain if isinstance(frame, Text) else ""
        out = Text()
        out.append(self._line)
        if glyph:
            out.append(" ")
            out.append(glyph, style=self._style)
        return out


@dataclass
class _Row:
    """One row in the transcript.

    ``kind`` selects the style; ``text`` is the status words; ``graphic`` is the
    power/debug-tier leading column (pip counter / sparkline); ``active`` marks
    the spinner-bearing row; ``data`` retains the raw payload so a detail-tier
    toggle can re-render graphics on demand.

    For ``kind == "thinking"`` the row carries one reasoning segment: ``text``
    is the segment body (kept collapsed/expanded per the transport flag) and
    ``thinking_expanded`` is the per-row expand state. Multiple thinking rows
    intersperse with tool-call rows in arrival order.
    """

    kind: str
    text: str
    graphic: str = ""
    active: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    thinking_expanded: bool = False


class RichTransport:
    """TransportCallbacks implementation backed by a Rich Console.

    The Live region is a single flowing column of transcript rows — the turn
    trajectory rendered as it happens. Each stage event appends (or updates)
    one row; the streamed answer flows in and persists across ``stop_thinking``.
    The active (most recent) row carries a live spinner.

    On turn completion the Live region stops and the transcript — including the
    streamed answer — is flushed to scrollback as the permanent record. The
    fully formatted Markdown panel is printed by ``on_response`` only when the
    turn did not already stream.

    Attributes:
        console: The Rich Console to render to. Defaults to a shared
            ``Console()``; callers may pass a custom one (testing, redirection).
        debug: When True, selects the ``debug`` detail tier (every bookkeeping
            stage event renders, plus the ``on_debug_context`` panel).
    """

    def __init__(self, *, console: Console | None = None, debug: bool = False) -> None:
        self.console: Console = console or Console()
        self.debug = debug
        # Detail tier: ``clean`` (default) → ``power`` (Ctrl+O) → ``debug``
        # (--debug, construction-only). Seed from the debug flag.
        self._detail: Literal["clean", "power", "debug"] = "debug" if debug else "clean"
        # When True (default), each thinking row renders a compact
        # ``▸ Thinking… N char`` summary; when False, the full segment body is
        # shown. ``Ctrl+Y`` toggles ALL thinking rows together (they are
        # interspersed with tool calls, so a single global flag is the right
        # model, not a per-segment one). Preserved across detail tiers.
        self.thinking_collapsed: bool = True
        # ``_status`` is a non-None sentinel while the Live region is active and
        # ``None`` when stopped — the contract used by REPL cancel handling and
        # tests. The actual display is owned by ``_live``.
        self._status: Live | None = None
        self._live: Live | None = None
        self._thinking_stage: str = ""
        self._thinking_spinner: str = "dots"
        # The visible-answer deltas stream into a staging buffer that is NEVER
        # shown as the final answer — on completion the Live region tears down
        # and a clean Markdown Panel renders the full answer. The buffer exists
        # only so a mid-turn error can flush partial work to scrollback.
        self._answer_staging: Text = Text()
        # Index of the in-flight thinking row, so consecutive reasoning deltas
        # append to one segment; a tool call or model_call opens a new segment.
        self._open_thinking_row: int | None = None
        # Fork fan-out live state, populated by on_stage("fork_*") so the
        # transcript can render one row per child. Reset on start_thinking.
        self._fork_children: dict[int, dict[str, Any]] = {}
        self._fork_total: int = 0
        # The mode that currently owns the turn, tracked across stage events.
        self._current_mode: str = ""
        # ── The transcript: one ordered list of rows ────────────────────
        self._rows: list[_Row] = []
        # Per-iteration model-call durations for the current turn (sparkline).
        self._iteration_durations: list[float] = []
        # Running counters for the graphic column.
        self._handoff_count: int = 0
        self._tool_call_count: int = 0
        # ── Live status-header state (persistent, updated per stage) ─────
        # A fixed status bar rendered at the top of the Live region so
        # mode / model / token count stay visible and update instantly while
        # the turn runs — the guard against runaway sessions. The header is
        # owned by the transport, so it re-renders on every on_stage refresh
        # without waiting for the next prompt-toolkit status-line pass.
        self._status_current_model: str = ""
        self._status_model_provider: str = ""
        self._status_turn_tokens: int = 0
        self._status_session_tokens: int = 0
        # Index of the in-flight tool row, so its completion updates in place.
        self._open_tool_row: int | None = None
        # Index of the in-flight model row, so its completion updates in place.
        self._open_model_row: int | None = None
        # The streamed model output for the current turn as a single flowing
        # Text (visible answer bright, reasoning dimmed). Persists across
        # stop_thinking — the transcript flushes it to scrollback.
        self._stream_text: Text = Text()

    # ── Live region lifecycle ───────────────────────────────────────

    def start_thinking(
        self, stage: str = "dispatch", *, session_tokens: int = 0
    ) -> None:
        """Start the flowing transcript so the user sees the turn is active.

        The default stage is ``dispatch`` — the pre-pipeline phase where
        ``handle_message`` parses commands and selects ``current_mode_id``.
        It uses a distinct spinner (``dots2``) so "still deciding" reads
        differently from "model is generating" once the first ``model_call``
        arrives and switches to ``dots``.

        ``session_tokens`` seeds the live status header's cumulative session
        token count (the runtime's pre-turn total), so the header shows the
        real running total rather than starting at zero each turn.
        """
        self.stop_thinking()
        self._thinking_stage = stage
        self._streamed_during_turn = False
        self._fork_children = {}
        self._fork_total = 0
        self._current_mode = ""
        self._rows = []
        self._iteration_durations = []
        self._handoff_count = 0
        self._tool_call_count = 0
        self._status_current_model = ""
        self._status_model_provider = ""
        self._status_turn_tokens = 0
        self._status_session_tokens = session_tokens
        self._open_tool_row = None
        self._open_model_row = None
        self._open_thinking_row = None
        self._answer_staging = Text()
        is_dispatch = stage == "dispatch"
        self._thinking_spinner = "dots2" if is_dispatch else "dots"
        # Seed the transcript with the initial dispatch row so the Live region
        # shows something immediately. Mark it active so the spinner lands here.
        seed = _Row(kind="dispatch", text=self._render_status_line(stage), active=True)
        self._rows.append(seed)
        live: Live | None = Live(
            self._render_live_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        if live is not None:
            try:
                live.start()
            except Exception:  # pylint: disable=broad-except
                # Non-tty / redirected consoles can reject Live; fall back to
                # plain inline printing via _refresh_live.
                live = None
        self._live = live
        self._status = live

    def stop_thinking(self, *, flush_stream: bool = False) -> None:
        """Stop the thinking display if it is running.

        Flushes the transcript rows (tool calls + thinking segments) to
        scrollback as the permanent turn record, then tears down the Live
        region. The streamed answer is NOT flushed here — ``on_response``
        renders the clean Markdown Panel from the full response text. When
        ``flush_stream`` is True (the mid-turn error path), any partial
        answer staged so far is flushed so partial work isn't silently lost
        before the error panel shows.
        """
        live = self._live
        status = self._status
        self._live = None
        self._status = None
        if live is not None:
            for row in self._rows:
                try:
                    self.console.print(self._row_line(row))
                except Exception:  # pylint: disable=broad-except
                    pass
            if flush_stream and self._answer_staging:
                try:
                    self.console.print(self._answer_staging)
                except Exception:  # pylint: disable=broad-except
                    pass
        if live is not None:
            try:
                live.stop()
            except Exception:  # pylint: disable=broad-except
                pass
        if status is not None and status is not live:
            try:
                status.stop()
            except Exception:  # pylint: disable=broad-except
                pass

    # ── Rendering ───────────────────────────────────────────────────

    def _live_height(self) -> int:
        """Number of transcript rows visible in the Live region.

        Terminal height minus ``_RESERVED_BELOW_LINES`` so post-turn content
        doesn't push the transcript off-screen. Floored at ``_MIN_VISIBLE_ROWS``.
        """
        height = self.console.size.height or 0
        if height <= 0:
            return _MIN_VISIBLE_ROWS
        return max(_MIN_VISIBLE_ROWS, height - _RESERVED_BELOW_LINES)

    def _render_status_header(self) -> Text:
        """Render the persistent live status bar (mode · model · tokens).

        This is the always-on guard against runaway sessions: it sits at the
        top of the Live region and updates on every stage event, so the mode,
        active model, and turn/session token counts are visible and current
        for the whole turn — not just once above the input prompt.
        """
        parts: list[str] = []
        if self._current_mode:
            parts.append(f"mode {self._current_mode}")
        if self._status_current_model:
            model = self._status_current_model
            if self._status_model_provider:
                model = f"{self._status_model_provider}:{model}"
            parts.append(model)
        parts.append(
            f"tokens {self._status_turn_tokens}/{self._status_session_tokens}"
        )
        text = " │ ".join(parts)
        return Text(text, style="bold " + STYLE_STAGE)

    def _render_live_display(self) -> Group:
        """Build the Live renderable: the live status header + recent transcript rows.

        The status header is fixed at the top and updates in place on every
        stage event (mode, model, token counters). Below it the transcript
        scrolls; the most recent rows are shown and the active (last) row gets
        a trailing spinner. Thinking rows render collapsed (a compact
        ``▸ Thinking… N char`` summary) or expanded (the full dimmed segment
        body) per the global ``thinking_collapsed`` flag.
        """
        renderables: list[Any] = [self._render_status_header()]
        rows = self._rows[-self._live_height():]
        for i, row in enumerate(rows):
            is_active = row.active and self._live is not None and i == len(rows) - 1
            line = self._row_line(row)
            if is_active:
                renderables.append(_TrailingSpinner(line, self._thinking_spinner, STYLE_GRAPHIC))
            else:
                renderables.append(line)
        return Group(*renderables)

    def _row_line(self, row: _Row) -> Text:
        """Render one transcript row.

        Tool/model/transition/fork rows render as ``<graphic>  <text>`` (graphic
        on power/debug). Thinking rows render collapsed (a compact summary) or
        expanded (the full dimmed segment body) per the row's expand flag.
        """
        if row.kind == "thinking":
            return self._render_thinking_row(row)
        line = Text()
        graphic = row.graphic if self._detail in ("power", "debug") else ""
        if graphic:
            line.append(graphic, style=STYLE_GRAPHIC)
            line.append("  ")
        line.append(row.text, style=self._row_style(row.kind))
        return line

    def _render_thinking_row(self, row: _Row) -> Text:
        """Render one thinking segment: collapsed summary or expanded body."""
        if row.thinking_expanded and row.text:
            return Text(row.text, style=STYLE_STREAM_REASONING)
        char_count = len(row.text)
        label = "▸ Thinking…" if char_count else "▸ (thinking…)"
        return Text(f"{label} {char_count} char", style=STYLE_GRAPHIC)

    @staticmethod
    def _row_style(kind: str) -> str:
        return {
            "transition": STYLE_ROW_TRANSITION,
            "model": STYLE_ROW_MODEL,
            "tool": STYLE_ROW_TOOL,
            "fork": STYLE_ROW_TOOL,
            "error": STYLE_ROW_ERROR,
        }.get(kind, STYLE_STAGE)

    def _stream_line_count(self) -> int:
        """Visible thinking-char total across all thinking rows (for tests).

        Kept so existing tests that drove the old single-stream buffer can assert
        reasoning was captured. Returns the count of thinking segments whose
        bodies are non-empty.
        """
        return sum(1 for r in self._rows if r.kind == "thinking" and r.text)

    @property
    def detail(self) -> Literal["clean", "power", "debug"]:
        """Current detail tier (read-only accessor for the REPL toolbar/tests)."""
        return self._detail

    def toggle_thinking(self) -> None:
        """Collapse/expand ALL thinking rows together (Ctrl+Y).

        Thinking segments are interspersed with tool calls in the transcript;
        the toggle flips every visible thinking row's expand state at once so
        the user can inspect the model's reasoning across the whole turn
        without hunting for each segment.

        Between turns (Live region inactive) the toggle is a state-only change:
        it does NOT reprint the transcript, so no spurious row leaks into the
        input area. The bottom toolbar reflects the new state (the REPL
        invalidates it on the keystroke). During a turn (Live active) the
        in-flight thinking rows re-render expanded/collapsed immediately.
        """
        self.thinking_collapsed = not self.thinking_collapsed
        for row in self._rows:
            if row.kind == "thinking":
                row.thinking_expanded = not self.thinking_collapsed
        self._redraw_live()

    def toggle_detail(self) -> None:
        """Toggle the detail tier between ``clean`` and ``power`` (Ctrl+O).

        ``power`` re-renders rows with their graphic column (pip counters,
        sparklines, token suffixes). No-op on the construction-only ``debug``
        tier, which already shows everything.

        Between turns (Live region inactive) the toggle is state-only — no
        transcript reprint, so no spurious row leaks into the input area. The
        bottom toolbar reflects the new tier.
        """
        if self._detail == "debug":
            return
        self._detail = "power" if self._detail == "clean" else "clean"
        self._redraw_live()

    def _redraw_live(self) -> None:
        """Re-render the Live region if active; otherwise do nothing.

        Used by the detail/thinking toggles: between turns the Live region is
        down and reprinting the last transcript row would pollute the input
        area with a stale call line. The toggle's feedback surface is the
        bottom toolbar, not the transcript.
        """
        if self._live is not None:
            try:
                self._live.update(self._render_live_display())
            except Exception:  # pylint: disable=broad-except
                pass

    def _refresh_live(self) -> None:
        """Re-render the Live region if active; otherwise print the latest row inline.

        Used by row-arrival paths (stage events, streamed deltas): when the
        Live region is up, refresh it; in the headless/test path (no Live),
        print the latest row so the transcript still flows to stdout.
        """
        if self._live is not None:
            try:
                self._live.update(self._render_live_display())
            except Exception:  # pylint: disable=broad-except
                pass
        else:
            # No Live region (headless / test path): print the latest row inline.
            row = self._rows[-1] if self._rows else None
            if row is not None:
                self.console.print(self._row_line(row))

    # ── Stage events ────────────────────────────────────────────────

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        """Route one stage event into the transcript and refresh.

        * ``model_call`` opens a model row; ``model_call_complete`` /
          ``model_call_failed`` update it in place with ✓/✗ + duration
          (mirrors the tool-dispatch in-place update pattern).
        * ``tool_dispatch`` opens a tool row; ``tool_dispatch_complete`` updates
          it in place with its status glyph and duration.
        * ``fork_*`` → fork rows, one per child.
        * Mode transitions (from ``model_call`` / handoff) → transition rows.

        Bookkeeping-only events render rows only on the ``debug`` tier; on
        ``clean``/``power`` they still update internal counters.
        """
        data = data or {}

        # A tool/model/fork event closes the open thinking segment so the next
        # reasoning delta opens a fresh interspersed segment.
        if label in ("model_call", "model_call_complete", "model_call_failed", "tool_dispatch", "tool_dispatch_complete", "fork_start", "fork_child_start", "fork_child_complete", "fork_complete"):
            self._open_thinking_row = None

        # ── Live mode tracking ──────────────────────────────────────
        prev_mode = self._current_mode
        if label == "model_call" and data.get("mode_id"):
            self._current_mode = str(data["mode_id"])
        elif (
            label == "tool_dispatch_complete"
            and data.get("transition") == "handoff"
            and data.get("handoff_to")
        ):
            self._current_mode = str(data["handoff_to"])
        if self._current_mode and self._current_mode != prev_mode:
            self._handoff_count += 1
            self._append_transition(prev_mode, self._current_mode, data)

        # ── Per-iteration duration sampling ─────────────────────────
        if label == "model_call_complete":
            d_raw = data.get("duration_ms")
            if d_raw not in (None, "", 0):
                try:
                    self._iteration_durations.append(float(d_raw))
                except (TypeError, ValueError):
                    pass

        # ── Fork fan-out ────────────────────────────────────────────
        if label == "fork_start":
            self._fork_children = {}
            self._fork_total = int(data.get("children", 0)) or 0
            self._append_row(_Row(kind="fork", text=self._render_fork_stage()))
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
            self._append_row(_Row(kind="fork", text=self._render_fork_stage()))
            return
        if label == "fork_child_complete":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                existing = self._fork_children.get(idx, {})
                existing.update(
                    {"status": data.get("status", "ok"), "duration_ms": data.get("duration_ms")}
                )
                self._fork_children[idx] = existing
            self._append_row(_Row(kind="fork", text=self._render_fork_stage()))
            return
        if label == "fork_complete":
            ok = int(data.get("ok", 0))
            failed = int(data.get("failed", 0))
            self._fork_total = 0
            self._append_row(_Row(kind="fork", text=self._render_fork_stage(summary=(ok, failed))))
            return

        # ── Model calls ─────────────────────────────────────────────
        if label in ("model_call", "model_call_complete", "model_call_failed"):
            self._route_model_call(label, data)
            return

        # ── Tool dispatch ───────────────────────────────────────────
        if label == "tool_dispatch":
            self._tool_call_count += 1
            self._open_tool_row = len(self._rows)
            max_calls = int(data.get("max_tool_calls") or 0)
            graphic = _pip_counter(self._tool_call_count, max_calls)
            self._append_row(
                _Row(
                    kind="tool",
                    text=self._render_tool_dispatch(data),
                    graphic=graphic,
                    data=data,
                )
            )
            return
        if label == "tool_dispatch_complete":
            self._complete_tool_row(data)
            return

        # ── Catch-all (debug tier only) ──────────────────────────────
        if self._detail == "debug":
            self._append_row(_Row(kind="stage", text=self._render_status_line(label, data), data=data))
        else:
            # Bookkeeping events update counters but render no row on clean/power.
            self._mark_last_active()
            self._refresh_live()

    def _append_row(self, row: _Row) -> None:
        """Append ``row`` to the transcript, mark it active, and refresh."""
        for existing in self._rows:
            existing.active = False
        row.active = True
        self._rows.append(row)
        self._refresh_live()

    def _mark_last_active(self) -> None:
        for existing in self._rows:
            existing.active = False
        if self._rows:
            self._rows[-1].active = True

    def _append_transition(self, from_mode: str, to_mode: str, data: dict[str, Any]) -> None:
        """Append a mode-transition row (power/debug carry a handoff pip)."""
        arrow = "→" if from_mode else "⇥"
        text = f"[{to_mode}] handoff: {from_mode or 'start'} {arrow} {to_mode}"
        max_handoffs = int(data.get("max_handoffs") or 0)
        graphic = _pip_counter(self._handoff_count, max_handoffs)
        self._append_row(_Row(kind="transition", text=text, graphic=graphic, data=data))

    def _route_model_call(self, label: str, data: dict[str, Any]) -> None:
        """Route a model-call stage event to a single in-place-updated row.

        ``model_call`` opens a model row (storing its index in
        ``_open_model_row``); ``model_call_complete`` / ``model_call_failed``
        update that row in place with ✓/✗ + duration — the same pattern as
        ``tool_dispatch`` / ``tool_dispatch_complete``.
        """
        max_iter = int(data.get("max_iterations") or 0)
        iter_raw = data.get("iteration")
        if iter_raw in (None, "", 0):
            iter_raw = data.get("_state.iteration")
        iter_count = int(iter_raw) if iter_raw not in (None, "", 0) else 0

        # Update the persistent live status header so mode/model stay current
        # the instant the model call begins (model_call_complete below adds
        # token totals).
        if data.get("model_id") not in (None, "", 0):
            self._status_current_model = str(data["model_id"])
            provider = data.get("provider")
            self._status_model_provider = str(provider) if provider not in (None, "") else ""

        if label == "model_call":
            graphic = _pip_counter(iter_count, max_iter)
            self._open_model_row = len(self._rows)
            self._append_row(
                _Row(
                    kind="model",
                    text=self._render_model_call(data),
                    graphic=graphic,
                    data=data,
                )
            )
            return

        # model_call_complete or model_call_failed — update the open row in place.
        status_glyph = "✗" if label == "model_call_failed" else "✓"
        graphic = _pip_counter(iter_count, max_iter)
        if label == "model_call_complete":
            spark = _sparkline(self._iteration_durations)
            dur_raw = data.get("duration_ms")
            if dur_raw not in (None, "", 0):
                graphic = (
                    f"{graphic}  {spark or _block_bar(float(dur_raw))} "
                    f"{round(float(dur_raw), 3)}ms"
                )
            # Accumulate the model call's token usage into the live header so
            # the running total updates instantly and guards runaway sessions.
            total_raw = data.get("_state.total_tokens") or data.get("total_tokens")
            if total_raw not in (None, "", 0):
                self._status_turn_tokens += int(total_raw)
                self._status_session_tokens += int(total_raw)

        idx = self._open_model_row
        if idx is not None and idx < len(self._rows):
            row = self._rows[idx]
            row.text = self._render_model_call(row.data, status_glyph=status_glyph, complete=data)
            row.graphic = graphic
            row.data = {**row.data, **data}
        else:
            self._append_row(
                _Row(
                    kind="model",
                    text=self._render_model_call(data, status_glyph=status_glyph, complete=data),
                    graphic=graphic,
                    data=data,
                )
            )
        self._open_model_row = None
        self._mark_last_active()
        self._refresh_live()

    def _render_model_call(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
    ) -> str:
        """Render a model call as a one-liner: stage, model, iteration, status.

        ``[mode] stage: model_call haiku #1 ✓ 340ms``. On completion
        (``status_glyph`` set) the status and duration are appended.
        Metadata suffixes (provider, tokens) render only on the power/debug
        tiers.
        """
        prefix = f"[{self._current_mode}] " if self._current_mode else ""
        body = "stage: model_call"
        model_id = data.get("model_id")
        if model_id not in (None, "", 0):
            body += f" {model_id}"
        iter_raw = data.get("iteration") or data.get("_state.iteration")
        if iter_raw not in (None, "", 0):
            body += f" #{iter_raw}"
        if status_glyph:
            body += f" {status_glyph}"
        duration = (complete or data).get("duration_ms")
        if status_glyph and duration not in (None, "", 0):
            body += f" {round(float(duration))}ms"
        suffix = self._render_stage_suffix(data) if self._detail in ("power", "debug") else ""
        return f"{prefix}{body}{suffix}"

    def _complete_tool_row(self, data: dict[str, Any]) -> None:
        """Update the in-flight tool row in place with its status + duration."""
        ok = data.get("ok")
        status = "ok" if ok is True else ("error" if ok is False else "")
        glyph = _TOOL_STATUS_GLYPH.get(status, _TOOL_STATUS_GLYPH[""])
        if self._open_tool_row is not None and self._open_tool_row < len(self._rows):
            row = self._rows[self._open_tool_row]
            row.text = self._render_tool_dispatch(row.data, status_glyph=glyph, complete=data)
            row.data = {**row.data, **data}
        else:
            self._append_row(
                _Row(kind="tool", text=self._render_tool_dispatch(data, status_glyph=glyph, complete=data), data=data)
            )
            return
        self._open_tool_row = None
        self._mark_last_active()
        self._refresh_live()

    def _render_status_line(self, label: str, data: dict[str, Any] | None = None) -> str:
        """Render one status line, prefixed with the active mode.

        ``[mode] stage: <label> (…meta…)``. The mode prefix updates live on
        mid-turn handoff. Meta suffixes render only on the power/debug tiers.
        """
        data = data or {}
        prefix = f"[{self._current_mode}] " if self._current_mode else ""
        suffix = self._render_stage_suffix(data) if self._detail in ("power", "debug") else ""
        return f"{prefix}stage: {label}{suffix}"

    def _render_stage_suffix(self, data: dict[str, Any]) -> str:
        """Compact one-line rendering of textual stage metadata (power/debug tiers).

        Uses visual glyphs instead of ``key=value``: provider/stage_id/turn_count
        as bare values, model_id bare, ``⟁`` for reasoning models, and token
        flow arrows (``↑``/``↓``/``Σ``).
        """
        parts: list[str] = []
        for key in ("provider", "stage_id", "turn_count"):
            value = data.get(key)
            if value in (None, "", 0):
                continue
            parts.append(str(value))
        model_id = data.get("model_id")
        if model_id not in (None, "", 0):
            parts.append(str(model_id))
        if data.get("reasoning_model"):
            parts.append("⟁")
        prompt = data.get("prompt_tokens")
        if prompt:
            parts.append(f"↑{prompt}")
        completion = data.get("completion_tokens")
        if completion:
            parts.append(f"↓{completion}")
        total = data.get("total_tokens")
        if total:
            parts.append(f"Σ{total}")
        return f" ({' '.join(parts)})" if parts else ""

    # ── tool_dispatch legible rendering ──────────────────────────────
    _TOOL_KIND_GLYPH: ClassVar[dict[str, str]] = {
        "operation": "→ op",
        "pack_tool": "→ tool",
        "mcp_tool": "→ mcp",
        "transition": "↦",
    }

    def _render_tool_dispatch(
        self,
        data: dict[str, Any],
        *,
        status_glyph: str = "",
        complete: dict[str, Any] | None = None,
    ) -> str:
        """Render a tool call as a one-liner: glyph, tool, primary arg, status.

        ``→ mcp brave_search("query") ✓ 120ms``. On completion (``status_glyph``
        set) the status and duration are appended.
        """
        tool = str(data.get("tool") or "")
        kind = str(data.get("tool_kind") or "")
        glyph = self._TOOL_KIND_GLYPH.get(kind, "→")
        prefix = f"[{self._current_mode}] " if self._current_mode else ""

        if kind == "transition":
            target = ""
            if data.get("handoff_to"):
                target = f"→ {data['handoff_to']}"
            elif data.get("transition") == "fork":
                n = data.get("fork_children") or 0
                target = f"fork ({n} children)"
            elif data.get("transitioned_to"):
                target = f"→ {data['transitioned_to']}"
            body = f"stage: tool_dispatch {glyph} {tool}"
            if target:
                body += f" {target}"
        else:
            arg = self._primary_arg_preview(data)
            body = f"{glyph} {tool}"
            if arg:
                body += f" {arg}"

        if status_glyph:
            body += f" {status_glyph}"
        duration = (complete or data).get("duration_ms")
        if status_glyph and duration not in (None, "", 0):
            body += f" {round(float(duration))}ms"
        return f"{prefix}{body}"

    @staticmethod
    def _primary_arg_preview(data: dict[str, Any]) -> str:
        """Extract the primary call argument as a short inline preview."""
        for key in ("arg_preview", "query", "path", "command", "input"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return _truncate(value)
        return ""

    def _render_fork_stage(self, summary: tuple[int, int] | None = None) -> str:
        """Render the fork fan-out as one header line plus one line per child."""
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

    # ── Streaming / response / debug / error ────────────────────────

    def _render_call_summary(self) -> None:
        """Render the togglable call-stats summary after the answer.

        On the ``clean``/``power`` tier a compact one-liner (tool-call count,
        total model time, handoff note). On the ``debug`` tier a full per-row
        breakdown — one line per tool call with its status and duration, plus
        the model-call iterations with their sparkline — so the operator can
        audit the turn trajectory. No-op when there were no calls.
        """
        tools = [r for r in self._rows if r.kind == "tool"]
        models = [r for r in self._rows if r.kind == "model"]
        transitions = [r for r in self._rows if r.kind == "transition"]
        if not tools and not models and not transitions:
            return
        if self._detail == "debug":
            self._render_call_summary_debug(tools, models, transitions)
            return
        parts: list[str] = []
        if tools:
            ok = sum(1 for r in tools if "✓" in r.text)
            failed = sum(1 for r in tools if "✗" in r.text)
            parts.append(f"{len(tools)} call{'s' if len(tools) != 1 else ''} ({ok}✓/{failed}✗)")
        if models:
            total_ms = sum(self._iteration_durations)
            parts.append(f"{len(models)} model {round(total_ms)}ms")
        if transitions:
            parts.append(f"{len(transitions)} handoff")
        if parts:
            self.console.print(Text("  ·  ".join(parts), style=STYLE_TEXT_MUTED))

    def _render_call_summary_debug(
        self, tools: list[_Row], models: list[_Row], transitions: list[_Row]
    ) -> None:
        """Debug-tier call summary: a per-row breakdown in a bordered panel."""
        from rich.table import Table

        table = Table(title="Turn Trajectory", border_style=STYLE_BORDER_CONTEXT, show_lines=False)
        table.add_column("#", style=STYLE_TEXT_MUTED)
        table.add_column("Kind")
        table.add_column("Detail")
        table.add_column("Meta", style=STYLE_TEXT_MUTED)
        idx = 0
        for r in tools:
            idx += 1
            meta = r.graphic if self._detail in ("power", "debug") else ""
            table.add_row(str(idx), "tool", r.text, meta)
        for r in models:
            idx += 1
            meta = r.graphic if self._detail in ("power", "debug") else ""
            table.add_row(str(idx), "model", r.text, meta)
        for r in transitions:
            idx += 1
            table.add_row(str(idx), "handoff", r.text, "")
        self.console.print(table)

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        """Route a streamed model delta.

        ``reasoning`` (thinking) becomes an interspersed thinking row in the
        transcript — consecutive reasoning deltas append to the open segment,
        and any non-reasoning event (tool call, model_call) closes it so the
        next reasoning opens a fresh segment. Multiple segments thus interleave
        with the tool calls in arrival order.

        ``text`` (the visible answer) is buffered into ``_answer_staging`` and
        is NEVER shown as the final answer. On completion ``on_response`` tears
        down the Live region and renders a clean Markdown Panel from the full
        response text — streaming the answer inline made the final unreadable
        with interspersed calls. The staging buffer only exists so a mid-turn
        error can flush partial work to scrollback.
        """
        if reasoning:
            # Append to the open thinking segment, or open a new one. A prior
            # tool/model event closed the open segment, so a new reasoning
            # delta after a tool call starts a fresh interspersed segment.
            if self._open_thinking_row is None or self._open_thinking_row >= len(self._rows):
                row = _Row(kind="thinking", text=reasoning, thinking_expanded=not self.thinking_collapsed)
                self._append_row(row)
                self._open_thinking_row = len(self._rows) - 1
            else:
                self._rows[self._open_thinking_row].text += reasoning
            self._refresh_live()
            return
        if text:
            # Visible answer: stage only, never rendered as the final.
            self._answer_staging.append(text, style=STYLE_STREAM_ANSWER)
            # A visible-answer delta closes the open thinking segment so the
            # next reasoning opens a new one (the model emitted an answer
            # chunk, so the prior thinking block is complete).
            self._open_thinking_row = None
            self._refresh_live()

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        """Render the final response as a clean Markdown Panel.

        Tears down the Live region — the transcript (tool calls + thinking
        segments) flushes to scrollback as the turn record — then renders the
        fully formatted answer. The streamed answer is NOT reused: streaming it
        inline made the final unreadable with interspersed tool calls, so the
        clean Panel is always rendered from the complete ``text``.
        """
        self.stop_thinking()
        self._answer_staging = Text()
        if text:
            self.console.print(
                Panel(Markdown(text), title="Response", border_style=STYLE_BORDER_RESPONSE)
            )
        # Togglable call-stats summary after the answer. Clean tier shows a
        # compact one-liner; debug tier shows the full per-row breakdown.
        self._render_call_summary()

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        """Render turn context in a Rich Panel (only on the debug tier)."""
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
        """Render an error in a red Rich Panel.

        The transcript (including any partial staged answer) flushes to
        scrollback first so partial work isn't silently lost before the error
        panel shows.
        """
        self.stop_thinking(flush_stream=True)
        self._answer_staging = Text()
        body = message
        if exc is not None:
            body = f"{type(exc).__name__}: {exc}\n\n{message}"
        logger.error("transport error: %s", body)
        self.console.print(Panel(body, title="Error", border_style=STYLE_BORDER_ERROR))


__all__ = ["RichTransport"]
