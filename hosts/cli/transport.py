"""Rich-based TransportCallbacks implementation for the CLI host.

This is the terminal-transport implementation of
:class:`hosts.core.callbacks.TransportCallbacks`. The CLI host constructs a
``RichTransport`` and passes it to ``hosts.core.handle_message``; the runtime
calls these methods during turn execution to deliver stage updates, streamed
text chunks, final responses, debug context, and errors.

Rich is allowed here — this is the terminal host. The core runtime never
imports Rich; it only calls these methods on the injected transport.

Stage-line legend
-----------------
Each pane row is one left-aligned line: ``<graphic>  <words>``.

* **Graphic column** (left):
  - ``●``/``○`` pip counter — iteration / tool-call / handoff fill.
  - ``N<n>`` — unlimited-capacity compact counter.
  - block-ramp sparkline (``▁▂▃▄▅▆▇█``) — per-iteration duration trend.
* **Words column** (right of the graphic):
  - ``[mode] stage: <label> (…)`` — the status line and compact suffix.
  - Suffix glyphs (no verbose ``key=value``):
    + ``<id>``      model_id (bare; no decorative prefix).
    + ``⟁``         reasoning-model capability flag (present only when
                    ``reasoning_model`` is true — signifies the model is a
                    deep-reasoning variant).
    + ``↑<n>``      prompt_tokens
    + ``↓<n>``      completion_tokens
    + ``Σ<n>``      total_tokens
    + ``provider`` / ``stage_id`` / ``turn_count`` shown as bare values.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

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
STYLE_PANE_HEADER = "bold steel_blue3"
STYLE_PANE_TRANSITION = "gold3"
STYLE_PANE_MODEL = "dark_sea_green4"
STYLE_PANE_TOOL = "cornflower_blue"
STYLE_GRAPHIC = "grey58"

#: Lines reserved below the Live region for post-turn content that must
#: stay visible: the turn footer (1), the Rule separator (1), the
#: powerline status line (1), and the input prompt line (1). The Live
#: region's height is the terminal height minus this reserve so that
#: when the panes are flushed and the footer/status/prompt are printed
#: below, the top of the pane content is not scrolled off-screen.
_RESERVED_BELOW_LINES = 4

#: Terminal height (lines) below which the three-pane Live layout collapses
#: to a single merged status line. This is a resize fallback: the rich
#: pane view is the default on workable terminals, and only very short
#: consoles (or a live shrink past this threshold) degrade to the compact
#: one-liner so content is never crushed under three forced headers.
_COMPACT_HEIGHT_THRESHOLD = 12

#: Floor for the number of history lines kept visible in a pane. Guarantees
#: a usable pane even on very short terminals / non-tty test consoles that
#: report a height of 0 or 1.
_PANE_MIN_LINES = 2

#: Cap for how many rows a pane can claim before it starts donating its
#: unused height to the other panes. A pane that has fewer rows than its
#: proportional share gives the leftover to the panes that need it, so an
#: empty transitions pane doesn't waste 1/6 of the screen.
_PANE_MAX_LINES = 40

#: Proportional height shares for the three panes (transitions, model calls,
#: tools). Used as the *initial* allocation; the actual allocation is then
#: trimmed so no pane gets more rows than it has content (plus a small
#: overflow buffer), and the freed space goes to the panes that need it.
_PANE_WEIGHTS = (1, 3, 2)
_PANE_WEIGHT_TOTAL = sum(_PANE_WEIGHTS)

#: Maximum width of the pip counter. When the cap exceeds this, the pip
#: row is capped at this width and saturates with ``⁺``; when the cap is
#: smaller, the row tracks the cap exactly so the user sees true fill.
#: When the cap is 0 (unlimited), the pip row is replaced by a compact
#: ``N`` counter (no bar) since a saturating bar against infinity is
#: meaningless.
_PIP_MAX_WIDTH = 16

#: Width of the duration sparkline shown on ``model_call_complete``. Each
#: cell is one unicode block character, so this is also the cell count.
_SPARK_WIDTH = 8

#: Block ramp used for both the duration bar and the sparkline. Index 0
#: is the empty cell; index 7 is the full block. Unicode sub-blocks give
#: 8 perceptually-even steps in a single terminal cell.
_BLOCK_RAMP = " ▁▂▃▄▅▆▇█"

#: Glyphs for tool-dispatch completion status shown in the tools pane.
_TOOL_STATUS_GLYPH = {"ok": "✓", "error": "✗", "running": "…", "": "·"}


def _pip_counter(count: int, cap: int) -> str:
    """Render ``count`` as a pip row that tracks the cap exactly.

    * ``cap > 0``: the row is ``cap`` cells wide — ``●`` per unit, ``○``
      for the remaining capacity. When ``cap`` exceeds ``_PIP_MAX_WIDTH``,
      the row is capped at ``_PIP_MAX_WIDTH`` and saturates with ``⁺``.
      This means a 15-iteration cap draws 15 cells (not 8), so the user
      sees true fill instead of premature saturation.
    * ``cap <= 0`` (unlimited): no bar — just a compact ``N{count}``
      counter, since a saturating bar against infinity is meaningless.
    """
    if cap <= 0:
        return f"N{count}"
    width = min(cap, _PIP_MAX_WIDTH)
    if count >= width:
        if count > width:
            return "●" * width + "⁺"
        return "●" * width
    filled = "●" * count
    empty = "○" * (width - count)
    return filled + empty


def _sparkline(values: list[float], width: int = _SPARK_WIDTH) -> str:
    """Render a single-line unicode sparkline of ``values``.

    Each value maps to one block-ramp cell, normalized against the max of
    ``values`` (or 1.0 when all are zero so the first iteration still
    shows a baseline cell). The most recent ``width`` values are kept;
    older ones drop off the left. Returns an empty string when ``values``
    is empty.
    """
    if not values:
        return ""
    recent = values[-width:]
    peak = max(recent) if recent else 0.0
    if peak <= 0:
        peak = 1.0
    cells: list[str] = []
    for v in recent:
        # Clamp to [0, 1] then scale to the ramp's 8 steps.
        ratio = max(0.0, min(1.0, float(v) / peak))
        idx = int(round(ratio * (len(_BLOCK_RAMP) - 1)))
        cells.append(_BLOCK_RAMP[idx])
    return "".join(cells)


def _block_bar(value: float, reference: float = 0.0) -> str:
    """Render a single block-ramp cell for ``value`` against ``reference``.

    When ``reference <= 0`` (no known max yet) the value maps to itself
    on a 0–1000ms heuristic scale so the first iteration still shows a
    proportional cell rather than a full block. Used on stages where we
    don't yet have a series for a sparkline.
    """
    if value <= 0:
        return _BLOCK_RAMP[0]
    scale = reference if reference > 0 else 1000.0
    ratio = max(0.0, min(1.0, value / scale))
    idx = int(round(ratio * (len(_BLOCK_RAMP) - 1)))
    return _BLOCK_RAMP[idx]


class _TrailingSpinner:
    """A Live-compatible renderable that puts the spinner *after* the line.

    Rich's :class:`~rich.spinner.Spinner` renders the animation frame
    *before* its text (``frame + " " + text``), which pushes the whole
    active row right by the frame width and breaks the left-aligned
    graphic column. This renderable inverts that: it renders the fixed
    line content first (so it stays aligned with every other row in the
    pane) and appends a single spinner frame glyph as a trailing
    activity indicator after a one-space gap.

    Implements ``render(time)`` so Rich's Live refresh animates the
    trailing frame without touching the leading content.
    """

    __slots__ = ("_line", "_spinner", "_style")

    def __init__(self, line: Text, spinner_name: str, style: str) -> None:
        self._line = line
        self._spinner = Spinner(spinner_name, style=style)
        self._style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Live calls this on every refresh; delegate to ``render`` with the
        # console's current time so the trailing frame animates.
        yield self.render(console.get_time())

    def render(self, time: float) -> RenderableType:
        frame = self._spinner.render(time)
        # ``frame`` is a Text whose first appended segment is the glyph;
        # when the spinner has empty text it is just the glyph Text.
        glyph = frame.plain if isinstance(frame, Text) else ""
        out = Text()
        out.append(self._line)
        if glyph:
            out.append(" ")
            out.append(glyph, style=self._style)
        return out


@dataclass
class _PaneRow:
    """One row in a pane: graphic on the left, textual label after it.

    The pane renderer lays each row out as ``<graphic>  <text>`` on a
    single left-aligned line so the per-iteration dots form a vertically
    aligned column across every row in every pane.
    """

    text: str
    graphic: str = ""
    #: When True this row is the active (spinner-bearing) row of its pane.
    active: bool = False


def _pane_heights(total: int, counts: tuple[int, int, int]) -> tuple[int, int, int]:
    """Split ``total`` lines across the three panes, content-aware.

    The 1:3:2 weights are the *initial* allocation. Each pane is then
    trimmed to no more than ``min(_PANE_MAX_LINES, len(counts[i]) + 1)``
    (its content plus one overflow row), and the freed space is
    redistributed to the panes that have more content than their
    allocation. This means an empty transitions pane donates its space
    to the model-calls and tools panes instead of wasting 1/6 of the
    screen.
    """
    if total <= 0:
        return (_PANE_MIN_LINES, _PANE_MIN_LINES, _PANE_MIN_LINES)
    body = max(0, total - 3)
    # Initial proportional allocation.
    base = [max(_PANE_MIN_LINES, body * w // _PANE_WEIGHT_TOTAL) for w in _PANE_WEIGHTS]
    # Trim panes that have less content than their allocation, capped at
    # _PANE_MAX_LINES so a very long pane doesn't eat the whole screen.
    demand = [min(_PANE_MAX_LINES, max(_PANE_MIN_LINES, c + 1)) for c in counts]
    freed = 0
    for i in range(3):
        if base[i] > demand[i]:
            freed += base[i] - demand[i]
            base[i] = demand[i]
    # Redistribute freed space to panes that need more than they have.
    if freed > 0:
        needy = [i for i in range(3) if base[i] < demand[i]]
        if needy:
            share = freed // len(needy)
            leftover = freed - share * len(needy)
            for i in needy:
                base[i] += share
            base[needy[0]] += leftover  # model-calls pane gets the remainder
        else:
            # Nobody needs more — give it all to the model-calls pane.
            base[1] += freed
    return (base[0], base[1], base[2])


class RichTransport:
    """TransportCallbacks implementation backed by a Rich Console.

    The Live region is split into three stacked, fixed-height panes that
    scroll independently within their own window:

    * **Transitions** (1 part) — mode handoffs (``default → coding``).
    * **Model calls** (3 parts) — ``model_call`` / ``model_call_complete``
      with the iteration pip counter and duration sparkline.
    * **Tools** (2 parts) — ``tool_dispatch`` / ``tool_dispatch_complete``
      and fork fan-out rows, with the tool-call pip counter.

    Each row is a single left-aligned line with the graphical strip
    (pip-counter dots and, when present, the duration sparkline) on the
    left and the status words after it, so the per-iteration dots line
    up vertically when the pane is scrolled back. The pane that owns the
    currently-active stage shows a live spinner on its last row; the
    other two panes show only completed entries.

    Attributes:
        console: The Rich Console to render to. Defaults to a shared
            ``Console()`` instance; callers may pass a custom one (e.g.
            for testing or file redirection).
        debug: When True, ``on_debug_context`` renders a panel. When
            False, it is a no-op (matches the ``--debug`` flag).
    """

    def __init__(
        self,
        *,
        console: Console | None = None,
        debug: bool = False,
        hide_thinking: bool = False,
    ) -> None:
        self.console: Console = console or Console()
        self.debug = debug
        # When True, the streaming ``reasoning_content`` deltas are not
        # rendered. The visible answer is unaffected. Default False —
        # "show the work" (see ``--hide-thinking`` in the CLI host).
        self.hide_thinking = hide_thinking
        # True while the streamed model output is in the "thinking" phase
        # (only reasoning deltas seen so far, no visible content yet).
        # Reset in ``start_thinking`` so each turn starts clean.
        self._thinking_open: bool = False
        # ``_status`` is retained as a non-None sentinel while the Live
        # region is active and ``None`` when stopped, preserving the
        # ``transport._status is None`` contract used by REPL cancel
        # handling and tests. The actual display is owned by ``_live``.
        self._status: Live | None = None
        self._live: Live | None = None
        self._thinking_stage: str = ""
        self._thinking_spinner: str = "dots"
        # True once on_text_chunk fires during the current turn. Reset in
        # start_thinking so each turn starts clean. When True, on_response
        # skips the Rich Panel because the text is already on screen —
        # printing it again would double-render (raw inline + formatted panel)
        # with a misaligned header.
        self._streamed_during_turn: bool = False
        # Fork fan-out live state. Populated by ``on_stage("fork_*")`` so the
        # tools pane can render one row per active child instead of a single
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
        # ── Three categorized, scrolling row buffers ───────────────────
        # Each pane owns its own list of ``_PaneRow`` entries. Every
        # ``on_stage`` event is routed to exactly one pane based on its
        # label; the panes scroll independently within their proportional
        # height share of the Live region. Cleared by ``start_thinking``.
        self._transitions: list[_PaneRow] = []
        self._model_calls: list[_PaneRow] = []
        self._tools: list[_PaneRow] = []
        # Which pane owns the currently-active (spinner) row. One of
        # ``"transitions"``, ``"model_calls"``, ``"tools"``, or ``""`` when
        # no stage is active yet (the initial dispatch seed).
        self._active_pane: str = ""
        # Per-iteration durations for the current turn, used to draw the
        # duration sparkline on ``model_call_complete``. Reset in
        # ``start_thinking`` so each turn starts with a fresh series.
        self._iteration_durations: list[float] = []
        # Counters for the graphical right-edge strips. These track the
        # running totals the kernel also reports, but we keep our own copy
        # so a pane row can render its graphic even when the incoming
        # payload omits the count (e.g. a ``tool_dispatch_complete`` that
        # carries no ``tool_calls_so_far``).
        self._handoff_count: int = 0
        self._tool_call_count: int = 0
        # Last-known terminal height, used to detect a resize while the Live
        # region is active so the panes can reflow immediately instead of
        # waiting for the next stage event. Seeded on the first live start.
        self._last_live_height: int = 0
        # Previous SIGWINCH handler to restore on stop_thinking (None when
        # no handler was registered or the platform has no SIGWINCH).
        self._prev_winch: Any = None

    # ── Live region lifecycle ───────────────────────────────────────

    def _live_height(self) -> int:
        """Total lines available for the three panes + their headers.

        The terminal height minus ``_RESERVED_BELOW_LINES`` (the turn
        footer, Rule separator, status line, and input prompt) so that
        when the Live region is flushed and the post-turn content is
        printed below it, the top of the pane content is not scrolled
        off-screen. Floor at 3 panes × ``_PANE_MIN_LINES`` + 3 headers so
        non-tty / test consoles (height 0) still show a usable layout.

        Returns ``0`` on short terminals (below ``_COMPACT_HEIGHT_THRESHOLD``)
        as a signal to ``_render_live_display`` that the pane view would be
        crushed and the compact single-line layout should be used instead.
        """
        height = self.console.size.height or 0
        if height <= 0:
            return 3 * _PANE_MIN_LINES + 3
        if height < _COMPACT_HEIGHT_THRESHOLD:
            return 0
        return max(3 * _PANE_MIN_LINES + 3, height - _RESERVED_BELOW_LINES)

    def _render_pane(
        self,
        title: str,
        rows: list[_PaneRow],
        height: int,
        style: str,
    ) -> list[Any]:
        """Render one pane: a header followed by scrolling rows.

        Each row is a single left-aligned line in the order
        **graphic → text**: the pip-counter dots (and sparkline, when
        present) come first, then the status words. Putting the graphical
        strip on the left keeps the per-iteration dots in a vertically
        aligned column across every row in every pane, so the graph
        reads top-to-bottom when the user scrolls back. Rows without a
        graphic render just the text.

        The most recent ``height`` rows are kept (the pane scrolls from
        the top). The active row — if this pane owns it — gets a live
        spinner on its line.
        """
        renderables: list[Any] = []
        renderables.append(Text(title, style=STYLE_PANE_HEADER))
        visible = rows[-height:] if height > 0 else []
        for i, row in enumerate(visible):
            is_last = i == len(visible) - 1
            is_active = is_last and row.active and self._live is not None
            line = Text()
            if row.graphic:
                line.append(row.graphic, style=STYLE_GRAPHIC)
                line.append("  ")
            line.append(row.text, style=style)
            if is_active:
                # Trailing spinner: the line content (graphic + text) is
                # rendered first and stays aligned with every other row;
                # the spinner frame is appended after a one-space gap so
                # it reads as an activity indicator without offsetting the
                # row's leading column.
                renderables.append(_TrailingSpinner(line, self._thinking_spinner, STYLE_GRAPHIC))
            else:
                renderables.append(line)
        return renderables

    def _render_live_display(self) -> Group:
        """Build the Live renderable: three stacked panes (1:3:2, content-aware).

        On short terminals (``_live_height()`` returns 0) the three-pane
        view is replaced by a single merged status line — the resize
        fallback — so the Live region stays usable instead of forcing three
        headers onto a crushed screen.
        """
        height = self._live_height()
        if height <= 0:
            return Group(self._render_compact_line())
        counts = (
            len(self._transitions),
            len(self._model_calls),
            len(self._tools),
        )
        th, mh, wh = _pane_heights(height, counts)
        renderables: list[Any] = []
        renderables.extend(
            self._render_pane("Transitions", self._transitions, th, STYLE_PANE_TRANSITION)
        )
        renderables.extend(
            self._render_pane("Model calls", self._model_calls, mh, STYLE_PANE_MODEL)
        )
        renderables.extend(self._render_pane("Tools", self._tools, wh, STYLE_PANE_TOOL))
        return Group(*renderables)

    def _render_compact_line(self) -> Text:
        """Render a single merged status line for short terminals.

        Joins the most recent row of each pane into one line, prefixed with
        a compact header so the user still sees the live stage at a glance.
        """
        parts: list[str] = []
        for rows, _style in (
            (self._transitions, STYLE_PANE_TRANSITION),
            (self._model_calls, STYLE_PANE_MODEL),
            (self._tools, STYLE_PANE_TOOL),
        ):
            if rows:
                last = rows[-1]
                parts.append(f"{last.graphic + ' ' if last.graphic else ''}{last.text}")
        body = " │ ".join(parts) if parts else self._render_status_line(self._thinking_stage)
        line = Text()
        line.append("⤷ ", style=STYLE_PANE_HEADER)
        line.append(body, style=STYLE_STAGE)
        return line

    def _refresh_live(self) -> None:
        """Re-render the Live region if it is active; otherwise print inline."""
        if self._live is not None:
            try:
                # Detect a terminal resize since the last render and reflow
                # the panes immediately (they are content-aware, so a change
                # in available height reallocates the rows).
                height = self.console.size.height or 0
                if height and height != self._last_live_height:
                    self._last_live_height = height
                self._live.update(self._render_live_display())
            except Exception:  # pylint: disable=broad-except
                pass
        else:
            # No Live region: fall back to printing the latest row inline
            # (the pre-existing behavior for the no-spinner path used by
            # tests and headless callers). Print from whichever pane got
            # the most recent row.
            row = self._last_row()
            if row is not None:
                self.console.print(Text(row.text, style=STYLE_STAGE))

    def _last_row(self) -> _PaneRow | None:
        """Return the most-recently appended row across all three panes."""
        candidates: list[_PaneRow] = []
        for pane in (self._transitions, self._model_calls, self._tools):
            if pane:
                candidates.append(pane[-1])
        if not candidates:
            return None
        # The active pane's last row is the most recent by definition.
        if self._active_pane == "transitions" and self._transitions:
            return self._transitions[-1]
        if self._active_pane == "model_calls" and self._model_calls:
            return self._model_calls[-1]
        if self._active_pane == "tools" and self._tools:
            return self._tools[-1]
        return candidates[-1]

    def start_thinking(self, stage: str = "dispatch") -> None:
        """Start the three-pane Live display so the user sees the turn is active.

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
        self._thinking_open = False
        self._fork_children = {}
        self._fork_total = 0
        self._current_mode = ""
        # Clear the three panes so each turn starts fresh.
        self._transitions = []
        self._model_calls = []
        self._tools = []
        self._active_pane = ""
        self._iteration_durations = []
        self._handoff_count = 0
        self._tool_call_count = 0
        is_dispatch = stage == "dispatch"
        self._thinking_spinner = "dots2" if is_dispatch else "dots"
        # Seed the tools pane with the initial dispatch label so the Live
        # region shows something immediately (before the first on_stage
        # event arrives). Mark it active so the spinner lands here.
        seed = _PaneRow(text=self._render_status_line(stage), active=True)
        self._tools.append(seed)
        self._active_pane = "tools"
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
                # plain inline printing via _refresh_live so the turn still
                # surfaces stage events.
                live = None
        self._live = live
        # Preserve the sentinel contract: ``_status`` is non-None while the
        # display is active so REPL cancel handling and tests can assert
        # ``transport._status is None`` after stop_thinking.
        self._status = live
        # Register a SIGWINCH handler (POSIX only) so a terminal resize
        # reflows the panes immediately instead of waiting for the next
        # stage event. On Windows there is no SIGWINCH; the compact/compact
        # reflow still responds on the next _refresh_live height change.
        self._last_live_height = self.console.size.height or 0
        if live is not None and os.name != "nt":
            try:
                import signal

                # SIGWINCH is POSIX-only; guard the access so pyright/pyright
                # on Windows (where the constant is undefined) doesn't flag it.
                winch = getattr(signal, "SIGWINCH", None)
                if winch is None:
                    raise AttributeError("SIGWINCH unavailable on this platform")
                self._prev_winch = signal.getsignal(winch)

                def _on_resize(signum: int, frame: Any) -> None:
                    # Reflow synchronously; Rich's Live handles repainting
                    # the new renderable on the next refresh tick.
                    _ = (signum, frame)
                    self._refresh_live()

                signal.signal(winch, _on_resize)
            except Exception:  # pylint: disable=broad-except
                # Non-interactive / unsupported platform: ignore resize.
                self._prev_winch = None

    def stop_thinking(self) -> None:
        """Stop the thinking display if it is running.

        Before tearing down the Live region, print the accumulated pane
        rows as permanent console output so they survive in the
        scrollback. Without this, Rich's Live region reclaims its lines on
        stop and the user would lose the trajectory they just watched.
        """
        # Restore any prior SIGWINCH handler before tearing down the Live
        # region so we don't leave a dangling callback.
        prev_winch = getattr(self, "_prev_winch", None)
        if prev_winch is not None and os.name != "nt":
            try:
                import signal

                winch = getattr(signal, "SIGWINCH", None)
                if winch is not None:
                    signal.signal(winch, prev_winch)
            except Exception:  # pylint: disable=broad-except
                pass
            self._prev_winch = None
        live = self._live
        status = self._status
        self._live = None
        self._status = None
        # Flush the three panes in order (transitions, model calls, tools)
        # so the scrollback matches the on-screen layout.
        if live is not None:
            for pane, style in (
                (self._transitions, STYLE_PANE_TRANSITION),
                (self._model_calls, STYLE_PANE_MODEL),
                (self._tools, STYLE_PANE_TOOL),
            ):
                for row in pane:
                    try:
                        line = Text(row.text, style=style)
                        if row.graphic:
                            line.append("  " + row.graphic, style=STYLE_GRAPHIC)
                        self.console.print(line)
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

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        """Route one stage event to the appropriate pane and refresh.

        Each stage event is classified into one of the three panes:

        * ``model_call`` / ``model_call_complete`` → model-calls pane.
        * ``tool_dispatch`` / ``tool_dispatch_complete`` → tools pane.
        * Fork events (``fork_*``) → tools pane, one row per child.
        * Mode transitions (detected from ``model_call`` /
          ``tool_dispatch_complete`` handoffs) → transitions pane.

        The pane that received the event becomes the active pane; its
        last row carries the live spinner. All other panes show only
        completed (non-spinner) rows.
        """
        data = data or {}

        # ── Live mode tracking ───────────────────────────────────────
        # model_call carries the entry mode; a mid-turn handoff surfaces
        # via tool_dispatch_complete with transition="handoff" and
        # handoff_to. Update _current_mode so the [mode] prefix on every
        # subsequent status line reflects who owns the turn now. Each
        # change is appended to the transitions pane.
        prev_mode = self._current_mode
        if label == "model_call" and data.get("mode_id"):
            self._current_mode = str(data["mode_id"])
        elif label == "tool_dispatch_complete":
            if data.get("transition") == "handoff" and data.get("handoff_to"):
                self._current_mode = str(data["handoff_to"])
        if self._current_mode and self._current_mode != prev_mode:
            self._handoff_count += 1
            self._append_transition(prev_mode, self._current_mode, data)

        # ── Per-iteration duration sampling ────────────────────────────
        # ``model_call_complete`` carries the wall-clock duration of the
        # model call. Accumulate it so the model-calls pane can draw a
        # duration sparkline. We sample here (not in the row renderer) so
        # the series stays in arrival order.
        if label == "model_call_complete":
            d_raw = data.get("duration_ms")
            if d_raw is not None and d_raw != "" and d_raw != 0:
                try:
                    self._iteration_durations.append(float(d_raw))
                except (TypeError, ValueError):
                    pass

        # ── Fork fan-out tracking ──────────────────────────────────────
        # Forks live in the tools pane: one row per child, plus a header
        # row and a complete-summary row. The per-child table is kept
        # visible after fork_complete so the user sees which children
        # succeeded/failed.
        if label == "fork_start":
            self._fork_children = {}
            self._fork_total = int(data.get("children", 0)) or 0
            self._append_tool_row(self._render_fork_stage())
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
            self._append_tool_row(self._render_fork_stage())
            return
        if label == "fork_child_complete":
            idx = int(data.get("child_index", -1))
            if idx >= 0:
                existing = self._fork_children.get(idx, {})
                existing.update(
                    {
                        "status": data.get("status", "ok"),
                        "duration_ms": data.get("duration_ms"),
                    }
                )
                self._fork_children[idx] = existing
            self._append_tool_row(self._render_fork_stage())
            return
        if label == "fork_complete":
            ok = int(data.get("ok", 0))
            failed = int(data.get("failed", 0))
            self._fork_total = 0
            text = self._render_fork_stage(summary=(ok, failed))
            self._append_tool_row(text)
            return

        # ── Route normal stage events to their pane ────────────────────
        if label in ("model_call", "model_call_complete"):
            self._append_model_row(label, data)
        elif label in ("tool_dispatch", "tool_dispatch_complete"):
            if label == "tool_dispatch":
                self._tool_call_count += 1
            max_calls = int(data.get("max_tool_calls") or 0)
            graphic = _pip_counter(self._tool_call_count, max_calls)
            self._append_tool_row(self._render_status_line(label, data), graphic=graphic)
        else:
            # Any other stage label (e.g. the initial "dispatch" seed,
            # "workflow_stage") goes to the tools pane as a catch-all so
            # it is visible without dropping on the floor.
            self._append_tool_row(self._render_status_line(label, data))

    def _append_transition(self, from_mode: str, to_mode: str, data: dict[str, Any]) -> None:
        """Append a mode-transition row to the transitions pane.

        The graphic is a pip counter of handoffs against the
        kernel-supplied ``max_handoffs`` cap. The pane scrolls within its
        proportional height share; once the window fills, the oldest
        transition drops off the top.
        """
        arrow = "→" if from_mode else "⇥"
        text = f"{from_mode or 'start'} {arrow} {to_mode}"
        max_handoffs = int(data.get("max_handoffs") or 0)
        graphic = _pip_counter(self._handoff_count, max_handoffs)
        self._transitions.append(_PaneRow(text=text, graphic=graphic))
        self._mark_active("transitions")
        self._refresh_live()

    def _append_model_row(self, label: str, data: dict[str, Any]) -> None:
        """Append a model-call row to the model-calls pane.

        ``model_call`` carries the iteration pip counter; ``model_call_complete``
        carries the duration sparkline. Both keep the iteration pip so the
        user can see how far through the budget the turn has progressed.
        """
        text = self._render_status_line(label, data)
        max_iter = int(data.get("max_iterations") or 0)
        # The kernel emits the iteration under ``_state.iteration`` (the
        # turn-runner state field), not a bare ``iteration`` key. Read
        # both for robustness so the pip counter reflects the real
        # iteration instead of always rendering ``N0``.
        iter_raw = data.get("iteration")
        if iter_raw is None or iter_raw == "" or iter_raw == 0:
            iter_raw = data.get("_state.iteration")
        iter_count = (
            int(iter_raw) if iter_raw is not None and iter_raw != "" and iter_raw != 0 else 0
        )
        graphic = _pip_counter(iter_count, max_iter)
        if label == "model_call_complete":
            spark = _sparkline(self._iteration_durations)
            dur_raw = data.get("duration_ms")
            if dur_raw is not None and dur_raw != "" and dur_raw != 0:
                dur_str = f"{spark or _block_bar(float(dur_raw))} {round(float(dur_raw), 3)}ms"
                graphic = f"{graphic}  {dur_str}"
        self._model_calls.append(_PaneRow(text=text, graphic=graphic))
        self._mark_active("model_calls")
        self._refresh_live()

    def _append_tool_row(self, text: str, graphic: str = "") -> None:
        """Append a row to the tools pane.

        Fork rows carry their own multi-line text (header + per-child
        lines); the graphic is left empty for those. Normal tool rows get
        a tool-call pip counter graphic when the caller supplies one.
        """
        self._tools.append(_PaneRow(text=text, graphic=graphic))
        self._mark_active("tools")
        self._refresh_live()

    def _mark_active(self, pane: str) -> None:
        """Mark ``pane`` as the active pane and move the spinner to its last row.

        Clears the ``active`` flag on all rows in all panes, then sets it
        on the last row of the named pane so the Live renderer puts the
        spinner there.
        """
        self._active_pane = pane
        for p in (self._transitions, self._model_calls, self._tools):
            for row in p:
                row.active = False
        target = {
            "transitions": self._transitions,
            "model_calls": self._model_calls,
            "tools": self._tools,
        }.get(pane)
        if target:
            target[-1].active = True

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
        """Append ``text`` to the tools pane and refresh the Live display.

        Kept for backward compatibility with callers/tests that drive the
        transport directly. New code should use ``_append_tool_row``.
        """
        self._append_tool_row(text)

    def _render_stage_suffix(self, data: dict[str, Any]) -> str:
        """Compact one-line rendering of the textual stage metadata.

        Graphical quantities (iteration pip, tool-call pip, duration
        sparkline) are rendered in the pane's graphic column by
        ``_append_model_row`` / ``_append_tool_row``; this suffix carries
        only the remaining textual metadata, using compact visual glyphs
        instead of verbose ``key=value`` pairs:

        * ``provider`` / ``stage_id`` / ``turn_count`` → bare value
          (their position makes them self-evident).
        * ``model_id`` → bare id (no decorative prefix).
        * ``reasoning_model`` → ``⟁`` when true (signifies the model is a
          deep-reasoning variant); omitted when false.
        * ``prompt_tokens`` → ``↑<n>``, ``completion_tokens`` → ``↓<n>``,
          ``total_tokens`` → ``Σ<n>`` (token-flow arrows).

        A legend is documented in the module docstring rather than
        repeated on every line.
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
        # Token counts are nested under model_call_complete.
        prompt = data.get("prompt_tokens")
        if prompt:
            parts.append(f"↑{prompt}")
        completion = data.get("completion_tokens")
        if completion:
            parts.append(f"↓{completion}")
        total = data.get("total_tokens")
        if total:
            parts.append(f"Σ{total}")
        if parts:
            return f" ({' '.join(parts)})"
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

    async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
        """Print streamed model output inline (no panel, no newline).

        ``text`` is the visible-answer delta. ``reasoning``, when non-empty,
        is the model's hidden thinking delta. Reasoning is rendered as a
        dim, collapsed ``▸ Thinking…`` block that closes (``◂``) the moment
        the first visible ``text`` delta arrives, so the user sees the
        model's work without it polluting the final answer. Suppressed
        entirely when ``hide_thinking`` is set.
        """
        self.stop_thinking()
        self._streamed_during_turn = True
        # ── Thinking block ───────────────────────────────────────────
        # Reasoning-only deltas open the block; the first visible text
        # delta closes it. Both streams may arrive on separate chunks, so
        # handle the open/close boundary on each call.
        if not self.hide_thinking and reasoning:
            if not self._thinking_open:
                self.console.file.write("▸ Thinking… ")
                self._thinking_open = True
            self.console.file.write(reasoning)
            self.console.file.flush()
            return
        if self._thinking_open:
            self.console.file.write("\n")
            self._thinking_open = False
        if text:
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
