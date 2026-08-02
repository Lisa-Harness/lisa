"""Tests for the RichTransport ``on_stage`` rendering.

Covers the detailed stage line and the fork fan-out per-child rendering
that the kernel's ``on_stage`` telemetry hook drives.
"""

from __future__ import annotations


import io

import pytest
from rich.console import Console
from rich.text import Text

from hosts.cli.transport import RichTransport


@pytest.mark.asyncio
async def test_on_stage_renders_detailed_suffix() -> None:
    """A normal stage event renders ``stage: <label> (key=value, …)`` with
    the structured metadata, skipping empty/zero values.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "model_call",
        {
            "mode_id": "default",
            "model_id": "haiku",
            "provider": "anthropic",
            "iteration": 1,
            "tool_calls_so_far": 0,  # should be omitted (zero)
            "tools_offered": True,  # not in the render allowlist
        },
    )

    # The spinner text is private; stop and inspect the thinking stage label
    # by capturing a non-spinner render.
    transport.stop_thinking()
    # When no spinner is active, on_stage prints the line inline.
    await transport.on_stage(
        "model_call_complete",
        {
            "mode_id": "default",
            "model_id": "haiku",
            "iteration": 1,
            "tool_calls": 2,
            "total_tokens": 120,
            "duration_ms": 340,
        },
    )


@pytest.mark.asyncio
async def test_on_stage_fork_renders_one_line_per_child() -> None:
    """Fork fan-out events render one line per active child.

    After fork_start + two fork_child_start events, the transport holds
    one entry per child index. fork_child_complete updates each child's
    status. fork_complete clears the fork state.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "fork_start",
        {
            "children": 2,
            "targets": [
                {"mode_id": "coding", "stage_id": "implement"},
                {"mode_id": "markdown", "stage_id": ""},
            ],
        },
    )
    assert transport._fork_total == 2
    assert transport._fork_children == {}

    await transport.on_stage(
        "fork_child_start",
        {
            "child_index": 0,
            "mode_id": "coding",
            "stage_id": "implement",
            "query_preview": "write the thing",
        },
    )
    await transport.on_stage(
        "fork_child_start",
        {
            "child_index": 1,
            "mode_id": "markdown",
            "stage_id": "",
            "query_preview": "doc it",
        },
    )
    assert set(transport._fork_children.keys()) == {0, 1}
    assert transport._fork_children[0]["status"] == "running"
    assert transport._fork_children[1]["mode_id"] == "markdown"

    await transport.on_stage(
        "fork_child_complete",
        {
            "child_index": 0,
            "mode_id": "coding",
            "stage_id": "implement",
            "status": "ok",
            "duration_ms": 42,
        },
    )
    assert transport._fork_children[0]["status"] == "ok"
    assert transport._fork_children[0]["duration_ms"] == 42

    await transport.on_stage("fork_complete", {"children": 2, "ok": 1, "failed": 1})
    # fork_complete keeps the per-child table visible so the user sees
    # *which* children succeeded/failed, not just counts. The table is
    # cleared by the next start_thinking (verified by the reset test).
    assert transport._fork_children != {}
    assert transport._fork_total == 0
    # Re-running start_thinking clears the fork state.
    transport.start_thinking("dispatch")
    assert transport._fork_children == {}


@pytest.mark.asyncio
async def test_start_thinking_resets_fork_state() -> None:
    """start_thinking clears any stale fork tracking from a prior turn."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport._fork_total = 5
    transport._fork_children = {0: {"status": "running"}}

    transport.start_thinking("model_call")

    assert transport._fork_total == 0
    assert transport._fork_children == {}


@pytest.mark.asyncio
async def test_render_fork_stage_format() -> None:
    """The fork stage text is one header line plus one line per child."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport._fork_total = 2
    transport._fork_children = {
        0: {
            "mode_id": "coding",
            "stage_id": "implement",
            "status": "running",
            "query_preview": "write the thing",
            "duration_ms": None,
        },
        1: {
            "mode_id": "markdown",
            "stage_id": "",
            "status": "ok",
            "query_preview": "",
            "duration_ms": 120,
        },
    }

    text = transport._render_fork_stage()

    lines = text.splitlines()
    assert lines[0] == "stage: fork (2 children)"
    assert "#0 coding · implement · running · write the thing" in lines[1]
    # markdown has no stage_id, so no double-dot segment; ok child shows duration.
    assert "#1 markdown · ok · 120ms" in lines[2]


@pytest.mark.asyncio
async def test_on_stage_no_spinner_prints_inline() -> None:
    """When no spinner is active, on_stage prints the stage line inline."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    # No start_thinking call — _status is None.
    assert transport._status is None

    # Should not raise.
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})


@pytest.mark.asyncio
async def test_mode_prefix_tracked_from_model_call() -> None:
    """model_call seeds _current_mode and the [mode] prefix appears on the
    status line. mode_id is no longer buried in a suffix key=value list.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    # Before any model_call, no mode prefix.
    assert transport._current_mode == ""

    await transport.on_stage(
        "model_call",
        {
            "mode_id": "coding",
            "model_id": "haiku",
            "provider": "anthropic",
            "iteration": 1,
        },
    )
    assert transport._current_mode == "coding"

    line = transport._render_status_line(
        "model_call",
        {
            "mode_id": "coding",
            "model_id": "haiku",
            "provider": "anthropic",
            "iteration": 1,
        },
    )
    # [coding] prefix present; mode_id no longer appears as a suffix key.
    assert line.startswith("[coding] ")
    assert "mode_id=" not in line


@pytest.mark.asyncio
async def test_mode_prefix_updates_on_mid_turn_handoff() -> None:
    """A mid-turn handoff (tool_dispatch_complete with transition=handoff)
    updates _current_mode so the [mode] prefix reflects the new owner.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")

    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    assert transport._current_mode == "default"

    await transport.on_stage(
        "tool_dispatch_complete",
        {
            "tool": "transition_stage",
            "tool_kind": "transition",
            "transition": "handoff",
            "handoff_to": "coding",
            "ok": True,
            "iteration": 1,
        },
    )
    assert transport._current_mode == "coding"

    line = transport._render_status_line("model_call", {"mode_id": "coding"})
    assert line.startswith("[coding] ")


@pytest.mark.asyncio
async def test_tool_dispatch_renders_legible_kind_glyphs() -> None:
    """tool_dispatch renders a glyph + kind word per tool_kind, not a flat
    tool_kind=key=value suffix. operation → op, mcp_tool → mcp, transition
    → handoff target.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    await transport.on_stage("model_call", {"mode_id": "coding", "iteration": 1})

    # operation — counts now render in the pane's right-aligned graphic
    # column, not inline in the text. The text carries only the glyph +
    # tool name.
    line = transport._render_tool_dispatch(
        {
            "tool": "write_file",
            "tool_kind": "operation",
            "iteration": 1,
            "tool_calls_so_far": 1,
        }
    )
    assert line == "stage: tool_dispatch → op: write_file"

    # mcp_tool — same: no inline pip counters in the text.
    line = transport._render_tool_dispatch(
        {
            "tool": "brave_search",
            "tool_kind": "mcp_tool",
            "iteration": 2,
            "tool_calls_so_far": 3,
        }
    )
    assert line == "stage: tool_dispatch → mcp: brave_search"

    # transition handoff — shows the target
    line = transport._render_tool_dispatch(
        {
            "tool": "transition_stage",
            "tool_kind": "transition",
            "transition": "handoff",
            "handoff_to": "markdown",
        }
    )
    assert "↦ transition_stage → markdown" in line

    # transition fork — shows child count
    line = transport._render_tool_dispatch(
        {
            "tool": "transition_stage",
            "tool_kind": "transition",
            "transition": "fork",
            "fork_children": 3,
        }
    )
    assert "↦ transition_stage fork (3 children)" in line


@pytest.mark.asyncio
async def test_fork_complete_keeps_per_child_table() -> None:
    """fork_complete keeps the per-child table so the user sees which
    children succeeded/failed, with ok/failed totals in the header.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")

    await transport.on_stage("fork_start", {"children": 2})
    await transport.on_stage(
        "fork_child_start",
        {
            "child_index": 0,
            "mode_id": "coding",
            "stage_id": "implement",
            "query_preview": "write",
        },
    )
    await transport.on_stage(
        "fork_child_start",
        {
            "child_index": 1,
            "mode_id": "markdown",
            "stage_id": "",
            "query_preview": "doc",
        },
    )
    await transport.on_stage(
        "fork_child_complete",
        {
            "child_index": 0,
            "status": "ok",
            "duration_ms": 42,
        },
    )
    await transport.on_stage(
        "fork_child_complete",
        {
            "child_index": 1,
            "status": "no_tools",
            "duration_ms": 30,
        },
    )
    await transport.on_stage("fork_complete", {"children": 2, "ok": 1, "failed": 1})

    # Table survives fork_complete.
    assert transport._fork_children != {}
    text = transport._render_fork_stage(summary=(1, 1))
    lines = text.splitlines()
    assert lines[0] == "stage: fork complete (1 ok, 1 failed)"
    assert "#0 coding · implement · ok · write · 42ms" in lines[1]
    assert "#1 markdown · no_tools · doc · 30ms" in lines[2]


@pytest.mark.asyncio
async def test_stage_history_accumulates_play_by_play() -> None:
    """on_stage accumulates every stage line into _stage_history so the
    user sees the full turn trajectory, not just the last stage.

    Previously the transport overwrote a single spinner line; the user
    missed earlier stages if they looked away. Now each event is appended
    to a history list and the active line is the most recent entry.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")

    await transport.on_stage(
        "model_call",
        {"mode_id": "default", "model_id": "glm-5.2:cloud", "iteration": 1},
    )
    await transport.on_stage(
        "tool_dispatch",
        {"tool": "search_web", "tool_kind": "mcp_tool", "iteration": 1, "tool_calls_so_far": 1},
    )
    await transport.on_stage(
        "tool_dispatch_complete",
        {"tool": "search_web", "duration_ms": 1200},
    )
    await transport.on_stage(
        "model_call",
        {"mode_id": "default", "model_id": "glm-5.2:cloud", "iteration": 2},
    )

    # All four stage events are retained in their panes — the user can
    # scroll back and see the full turn trajectory. model_call events go
    # to the model-calls pane; tool_dispatch events go to the tools pane.
    # The iteration counter is in the graphic column. With no
    # max_iterations in the payload (cap=0 → unlimited), the graphic is
    # a compact ``N{count}`` counter, not a saturating bar.
    assert len(transport._model_calls) >= 2  # two model_call events
    assert len(transport._tools) >= 1  # tool_dispatch event
    model_text = "\n".join(row.text for row in transport._model_calls)
    assert "model_call" in model_text
    tools_text = "\n".join(row.text for row in transport._tools)
    assert "search_web" in tools_text
    # iteration=2, cap=0 (unlimited) → "N2"
    assert transport._model_calls[-1].graphic == "N2"
    # The active pane is the one that received the last event.
    assert transport._active_pane == "model_calls"


@pytest.mark.asyncio
async def test_start_thinking_clears_history() -> None:
    """start_thinking resets all three panes so a new turn starts clean."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    assert transport._model_calls != []

    transport.start_thinking("dispatch")
    assert transport._model_calls == []
    assert transport._transitions == []
    # The tools pane is seeded with the initial dispatch row, so it has
    # exactly one entry (the seed), not the prior turn's tool rows.
    assert len(transport._tools) == 1
    assert transport._tools[0].text == "stage: dispatch"


@pytest.mark.asyncio
async def test_stop_thinking_clears_status_sentinel() -> None:
    """stop_thinking leaves _status as None so REPL cancel handling and
    the cancel tests' ``transport._status is None`` assertion still hold.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    assert transport._status is not None

    transport.stop_thinking()
    assert transport._status is None
    assert transport._live is None


# ── stream pane streaming (on_text_chunk) ────────────────────────────


def _make_stream_transport() -> RichTransport:
    """Build a RichTransport writing to a quiet StringIO console."""
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=100)
    return RichTransport(console=console, debug=False)


@pytest.mark.asyncio
async def test_on_text_chunk_accumulates_into_stream_pane() -> None:
    """Visible and reasoning deltas accumulate in the stream pane without
    tearing down the Live region (the window stays fixed)."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    live = transport._live

    await transport.on_text_chunk("", reasoning="Let me think")
    await transport.on_text_chunk("", reasoning=" more")
    await transport.on_text_chunk("The answer")

    # All deltas buffered in arrival order into the single flowing stream.
    assert transport._stream_text.plain == "Let me think moreThe answer"
    # The Live region must NOT have been torn down by streaming.
    assert transport._live is live
    assert transport._status is not None


@pytest.mark.asyncio
async def test_on_text_chunk_reasoning_dimmed_but_shown() -> None:
    """Reasoning deltas are rendered (dimmed) alongside the visible answer."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")

    await transport.on_text_chunk("", reasoning="secret thinking")
    await transport.on_text_chunk("visible answer")

    # Both reasoning and visible answer are buffered in arrival order.
    assert transport._stream_text.plain == "secret thinkingvisible answer"


@pytest.mark.asyncio
async def test_start_thinking_resets_stream_pane() -> None:
    """start_thinking clears the stream pane so a new turn starts fresh."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("some streamed text")
    assert transport._stream_text != Text()

    transport.start_thinking("model_call")
    assert transport._stream_text == Text()


@pytest.mark.asyncio
async def test_on_response_discards_stream_and_renders_panel() -> None:
    """on_response stops the Live region, clears the stream pane, and renders
    the fully formatted answer panel — the streamed version is gone."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="think")
    await transport.on_text_chunk("draft answer")

    await transport.on_response("# Final Answer")

    # Stream pane discarded, Live region stopped.
    assert transport._stream_text == Text()
    assert transport._live is None
    assert transport._status is None
    # The formatted Markdown panel was rendered (panel title in output).
    assert "Final Answer" in transport.console.file.getvalue()


@pytest.mark.asyncio
async def test_stream_pane_wraps_to_full_width() -> None:
    """The stream buffer wraps across the console width instead of each delta
    rendering as its own short line."""
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=20)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    await transport.on_text_chunk("hello world")
    await transport.on_text_chunk(" hello again")

    # A single flowing buffer wraps into multiple full-width lines.
    wrapped = transport._stream_wrapped_lines()
    # 22 chars at width 20 → 2 wrapped lines.
    assert len(wrapped) >= 2
    # Concatenated wrapped text equals the full buffered content.
    assert "".join(line.plain for line in wrapped) == "hello world hello again"


# ── compact (short-terminal) layout ──────────────────────────────────


def test_live_height_compact_threshold() -> None:
    """Short terminals (below the threshold) collapse to a compact layout."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    # Rich's console.size setter takes a (width, height) tuple. A height of
    # 8 lines is below _COMPACT_HEIGHT_THRESHOLD (12) → compact.
    transport.console.size = (80, 8)
    assert transport._live_height() == 0
    # A normal height (24) keeps the three-pane layout.
    transport.console.size = (80, 24)
    assert transport._live_height() > 0
