"""Tests for the RichTransport ``on_stage`` rendering.

Covers the flowing-transcript stage rows and the fork fan-out per-child
rendering that the kernel's ``on_stage`` telemetry hook drives.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.text import Text

from hosts.cli.transport import RichTransport, _Row


@pytest.mark.asyncio
async def test_on_stage_renders_detailed_suffix() -> None:
    """Stage events append transcript rows; metadata suffixes render only on
    the power/debug tiers.
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
            "tool_calls_so_far": 0,
        },
    )
    transport.stop_thinking()
    await transport.on_stage(
        "model_call_complete",
        {"mode_id": "default", "model_id": "haiku", "iteration": 1, "total_tokens": 120, "duration_ms": 340},
    )
    # Rows accumulated in the transcript.
    assert any(r.kind == "model" for r in transport._rows)


@pytest.mark.asyncio
async def test_on_stage_fork_renders_one_line_per_child() -> None:
    """Fork fan-out events track one entry per child index; fork_complete
    keeps the per-child table visible.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    await transport.on_stage("fork_start", {"children": 2})
    assert transport._fork_total == 2
    assert transport._fork_children == {}

    await transport.on_stage("fork_child_start", {"child_index": 0, "mode_id": "coding", "stage_id": "implement", "query_preview": "write the thing"})
    await transport.on_stage("fork_child_start", {"child_index": 1, "mode_id": "markdown", "stage_id": "", "query_preview": "doc it"})
    assert set(transport._fork_children.keys()) == {0, 1}
    assert transport._fork_children[0]["status"] == "running"

    await transport.on_stage("fork_child_complete", {"child_index": 0, "status": "ok", "duration_ms": 42})
    assert transport._fork_children[0]["status"] == "ok"
    assert transport._fork_children[0]["duration_ms"] == 42

    await transport.on_stage("fork_complete", {"children": 2, "ok": 1, "failed": 1})
    assert transport._fork_children != {}
    assert transport._fork_total == 0
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
        0: {"mode_id": "coding", "stage_id": "implement", "status": "running", "query_preview": "write the thing", "duration_ms": None},
        1: {"mode_id": "markdown", "stage_id": "", "status": "ok", "query_preview": "", "duration_ms": 120},
    }

    text = transport._render_fork_stage()

    lines = text.splitlines()
    assert lines[0] == "stage: fork (2 children)"
    assert "#0 coding · implement · running · write the thing" in lines[1]
    assert "#1 markdown · ok · 120ms" in lines[2]


@pytest.mark.asyncio
async def test_on_stage_no_spinner_prints_inline() -> None:
    """When no Live region is active, on_stage prints the row inline without raising."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    assert transport._status is None
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})


@pytest.mark.asyncio
async def test_mode_prefix_tracked_from_model_call() -> None:
    """model_call seeds _current_mode and the [mode] prefix appears on rows."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    assert transport._current_mode == ""

    await transport.on_stage("model_call", {"mode_id": "coding", "model_id": "haiku", "provider": "anthropic", "iteration": 1})
    assert transport._current_mode == "coding"

    line = transport._render_status_line("model_call", {"mode_id": "coding"})
    assert line.startswith("[coding] ")
    assert "mode_id=" not in line


@pytest.mark.asyncio
async def test_mode_prefix_updates_on_mid_turn_handoff() -> None:
    """A mid-turn handoff updates _current_mode so the [mode] prefix reflects the new owner."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")

    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    assert transport._current_mode == "default"

    await transport.on_stage(
        "tool_dispatch_complete",
        {"tool": "transition_stage", "tool_kind": "transition", "transition": "handoff", "handoff_to": "coding", "ok": True, "iteration": 1},
    )
    assert transport._current_mode == "coding"
    line = transport._render_status_line("model_call", {"mode_id": "coding"})
    assert line.startswith("[coding] ")


@pytest.mark.asyncio
async def test_tool_dispatch_renders_legible_kind_glyphs() -> None:
    """tool_dispatch renders a glyph + tool name per tool_kind."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport._current_mode = ""

    line = transport._render_tool_dispatch({"tool": "write_file", "tool_kind": "operation"})
    assert line == "→ op write_file"

    line = transport._render_tool_dispatch({"tool": "brave_search", "tool_kind": "mcp_tool"})
    assert line == "→ mcp brave_search"

    line = transport._render_tool_dispatch({"tool": "transition_stage", "tool_kind": "transition", "transition": "handoff", "handoff_to": "markdown"})
    assert "↦ transition_stage → markdown" in line

    line = transport._render_tool_dispatch({"tool": "transition_stage", "tool_kind": "transition", "transition": "fork", "fork_children": 3})
    assert "↦ transition_stage fork (3 children)" in line


@pytest.mark.asyncio
async def test_tool_row_updates_in_place_on_completion() -> None:
    """tool_dispatch_complete updates the open tool row in place with its
    status glyph and duration instead of appending a second row.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    await transport.on_stage("tool_dispatch", {"tool": "read_file", "tool_kind": "mcp_tool", "path": "hosts/cli/transport.py"})
    tool_rows_before = [r for r in transport._rows if r.kind == "tool"]
    assert len(tool_rows_before) == 1
    assert "…" not in tool_rows_before[0].text  # no status yet

    await transport.on_stage("tool_dispatch_complete", {"tool": "read_file", "ok": True, "duration_ms": 400})
    tool_rows_after = [r for r in transport._rows if r.kind == "tool"]
    # Still exactly one tool row — updated in place, not appended.
    assert len(tool_rows_after) == 1
    assert tool_rows_after[0] is tool_rows_before[0]
    assert "✓" in tool_rows_after[0].text
    assert "400ms" in tool_rows_after[0].text


@pytest.mark.asyncio
async def test_model_call_updates_in_place_on_completion() -> None:
    """model_call_complete updates the open model row in place with its
    status glyph and duration instead of appending a second row — the
    same in-place pattern as tool_dispatch_complete.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "model_call",
        {"mode_id": "default", "model_id": "haiku", "provider": "anthropic", "iteration": 1},
    )
    model_rows_before = [r for r in transport._rows if r.kind == "model"]
    assert len(model_rows_before) == 1
    assert "✓" not in model_rows_before[0].text  # no status yet

    await transport.on_stage(
        "model_call_complete",
        {"mode_id": "default", "model_id": "haiku", "iteration": 1, "duration_ms": 340},
    )
    model_rows_after = [r for r in transport._rows if r.kind == "model"]
    # Still exactly one model row — updated in place, not appended.
    assert len(model_rows_after) == 1
    assert model_rows_after[0] is model_rows_before[0]
    assert "✓" in model_rows_after[0].text
    assert "340ms" in model_rows_after[0].text


@pytest.mark.asyncio
async def test_model_call_failed_marks_row_with_x() -> None:
    """model_call_failed updates the open model row in place with ✗."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("model_call")

    await transport.on_stage(
        "model_call",
        {"mode_id": "default", "model_id": "haiku", "iteration": 1},
    )
    await transport.on_stage(
        "model_call_failed",
        {"mode_id": "default", "model_id": "haiku", "iteration": 1},
    )
    model_rows = [r for r in transport._rows if r.kind == "model"]
    assert len(model_rows) == 1  # updated in place, not appended
    assert "✗" in model_rows[0].text


@pytest.mark.asyncio
async def test_fork_complete_keeps_per_child_table() -> None:
    """fork_complete keeps the per-child table with ok/failed totals in the header."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")

    await transport.on_stage("fork_start", {"children": 2})
    await transport.on_stage("fork_child_start", {"child_index": 0, "mode_id": "coding", "stage_id": "implement", "query_preview": "write"})
    await transport.on_stage("fork_child_start", {"child_index": 1, "mode_id": "markdown", "stage_id": "", "query_preview": "doc"})
    await transport.on_stage("fork_child_complete", {"child_index": 0, "status": "ok", "duration_ms": 42})
    await transport.on_stage("fork_child_complete", {"child_index": 1, "status": "no_tools", "duration_ms": 30})
    await transport.on_stage("fork_complete", {"children": 2, "ok": 1, "failed": 1})

    text = transport._render_fork_stage(summary=(1, 1))
    lines = text.splitlines()
    assert lines[0] == "stage: fork complete (1 ok, 1 failed)"
    assert "#0 coding · implement · ok · write · 42ms" in lines[1]
    assert "#1 markdown · no_tools · doc · 30ms" in lines[2]


@pytest.mark.asyncio
async def test_transcript_accumulates_play_by_play() -> None:
    """on_stage accumulates every event into _rows so the user sees the full
    turn trajectory, with the last row active.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")

    await transport.on_stage("model_call", {"mode_id": "default", "model_id": "glm", "iteration": 1})
    await transport.on_stage("tool_dispatch", {"tool": "search_web", "tool_kind": "mcp_tool", "iteration": 1})
    await transport.on_stage("tool_dispatch_complete", {"tool": "search_web", "ok": True, "duration_ms": 1200})
    await transport.on_stage("model_call", {"mode_id": "default", "model_id": "glm", "iteration": 2})

    model_rows = [r for r in transport._rows if r.kind == "model"]
    assert len(model_rows) >= 2
    tool_text = "\n".join(r.text for r in transport._rows if r.kind == "tool")
    assert "search_web" in tool_text
    # The last row carries the active spinner.
    assert transport._rows[-1].active is True


@pytest.mark.asyncio
async def test_start_thinking_clears_transcript() -> None:
    """start_thinking resets the transcript so a new turn starts clean, seeded
    with the initial dispatch row.
    """
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    assert len(transport._rows) > 1

    transport.start_thinking("dispatch")
    # Fresh transcript: exactly the dispatch seed row.
    assert len(transport._rows) == 1
    assert transport._rows[0].text == "stage: dispatch"


@pytest.mark.asyncio
async def test_stop_thinking_clears_status_sentinel() -> None:
    """stop_thinking leaves _status and _live as None."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    assert transport._status is not None

    transport.stop_thinking()
    assert transport._status is None
    assert transport._live is None


# ── detail-tier tests ────────────────────────────────────────────────


def test_detail_tier_defaults_clean() -> None:
    """Default detail tier is clean; --debug selects debug at construction."""
    console = Console(quiet=True, force_terminal=False)
    assert RichTransport(console=console, debug=False)._detail == "clean"
    assert RichTransport(console=console, debug=True)._detail == "debug"


def test_toggle_detail_switches_clean_and_power() -> None:
    """toggle_detail flips clean ↔ power; it is a no-op on the debug tier."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    assert transport._detail == "clean"
    transport.toggle_detail()
    assert transport._detail == "power"
    transport.toggle_detail()
    assert transport._detail == "clean"

    debug_transport = RichTransport(console=console, debug=True)
    debug_transport.toggle_detail()
    assert debug_transport._detail == "debug"


def test_row_line_hides_graphic_on_clean_shows_on_power() -> None:
    """The graphic column renders only on power/debug tiers."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    r = _Row(kind="model", text="stage: model_call", graphic="●●○")
    clean_line = transport._row_line(r)
    assert "●" not in clean_line.plain

    transport._detail = "power"
    power_line = transport._row_line(r)
    assert "●●○" in power_line.plain


# ── thinking segments + answer staging (on_text_chunk) ──────────────


def _make_stream_transport() -> RichTransport:
    """Build a RichTransport writing to a quiet StringIO console."""
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=100)
    return RichTransport(console=console, debug=False)


@pytest.mark.asyncio
async def test_reasoning_becomes_interspersed_thinking_row() -> None:
    """Reasoning deltas append to an interspersed thinking row in the
    transcript; the Live region stays up.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    live = transport._live

    await transport.on_text_chunk("", reasoning="Let me think")
    await transport.on_text_chunk("", reasoning=" more")

    thinking_rows = [r for r in transport._rows if r.kind == "thinking"]
    assert len(thinking_rows) == 1
    assert thinking_rows[0].text == "Let me think more"
    assert transport._live is live
    assert transport._status is not None


@pytest.mark.asyncio
async def test_tool_call_opens_new_thinking_segment() -> None:
    """A tool call between two reasoning deltas closes the open segment so the
    next reasoning opens a fresh interspersed segment.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="first think")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    await transport.on_stage("tool_dispatch", {"tool": "search", "tool_kind": "mcp_tool"})
    await transport.on_stage("tool_dispatch_complete", {"tool": "search", "ok": True, "duration_ms": 10})
    await transport.on_text_chunk("", reasoning="second think")

    thinking_rows = [r for r in transport._rows if r.kind == "thinking"]
    assert len(thinking_rows) == 2
    assert thinking_rows[0].text == "first think"
    assert thinking_rows[1].text == "second think"


@pytest.mark.asyncio
async def test_visible_answer_staged_not_rendered_as_final() -> None:
    """The visible-answer delta is staged (for mid-turn error flush) and is
    never rendered into the transcript — on_response renders the clean Panel.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("draft answer")

    assert transport._answer_staging.plain == "draft answer"
    # No stream/thinking row carries the visible answer.
    assert not any(r.kind == "thinking" and "draft answer" in r.text for r in transport._rows)


@pytest.mark.asyncio
async def test_start_thinking_resets_staging_and_thinking() -> None:
    """start_thinking clears staging and the transcript so a new turn is clean."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="some thinking")
    await transport.on_text_chunk("some answer")
    assert transport._answer_staging != Text()
    assert any(r.kind == "thinking" for r in transport._rows)

    transport.start_thinking("model_call")
    assert transport._answer_staging == Text()
    assert not any(r.kind == "thinking" for r in transport._rows)


@pytest.mark.asyncio
async def test_on_response_always_renders_clean_panel() -> None:
    """on_response always renders the clean Markdown Panel from the full
    response text — even when the turn streamed — so the final answer is
    readable, not interspersed with the calls.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="thinking")
    await transport.on_text_chunk("draft answer")

    await transport.on_response("# Final Answer")

    assert transport._live is None
    assert transport._status is None
    assert transport._answer_staging == Text()
    assert "Final Answer" in transport.console.file.getvalue()
    # The Response panel (border + title) is present.
    assert "Response" in transport.console.file.getvalue()


@pytest.mark.asyncio
async def test_on_response_renders_call_summary() -> None:
    """After the answer panel, a compact call-stats summary line is rendered
    when the turn had tool/model calls.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1, "duration_ms": 500})
    await transport.on_stage("model_call_complete", {"mode_id": "default", "iteration": 1, "duration_ms": 500})
    await transport.on_stage("tool_dispatch", {"tool": "search", "tool_kind": "mcp_tool"})
    await transport.on_stage("tool_dispatch_complete", {"tool": "search", "ok": True, "duration_ms": 200})

    out_before = transport.console.file.getvalue()
    await transport.on_response("answer")

    out_after = transport.console.file.getvalue()[len(out_before):]
    # The summary mentions the call count.
    assert "1 call" in out_after
    assert "✓" in out_after


@pytest.mark.asyncio
async def test_on_response_debug_tier_renders_trajectory_table() -> None:
    """On the debug tier the call summary is a per-row trajectory table."""
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=100)
    transport = RichTransport(console=console, debug=True)
    transport.start_thinking("model_call")
    await transport.on_stage("tool_dispatch", {"tool": "search", "tool_kind": "mcp_tool"})
    await transport.on_stage("tool_dispatch_complete", {"tool": "search", "ok": True, "duration_ms": 200})

    await transport.on_response("answer")

    assert "Turn Trajectory" in sink.getvalue()


@pytest.mark.asyncio
async def test_stop_thinking_does_not_flush_answer() -> None:
    """stop_thinking flushes the transcript rows but NOT the staged answer —
    the clean Panel is rendered by on_response, not the stream.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("the staged answer")

    transport.stop_thinking()
    assert "the staged answer" not in transport.console.file.getvalue()


@pytest.mark.asyncio
async def test_on_error_flushes_partial_answer() -> None:
    """On a mid-turn error, the partial staged answer flushes to scrollback so
    partial work isn't silently lost.
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("partial answer")

    await transport.on_error("boom")

    assert "partial answer" in transport.console.file.getvalue()
    assert "Error" in transport.console.file.getvalue()


# ── thinking collapsed/expanded (Ctrl+Y toggle) ──────────────────────


@pytest.mark.asyncio
async def test_thinking_rows_collapsed_by_default() -> None:
    """Thinking rows default to a compact summary (collapsed)."""
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="a long wall of reasoning text")

    assert transport.thinking_collapsed is True
    row = next(r for r in transport._rows if r.kind == "thinking")
    line = transport._row_line(row)
    assert "Thinking…" in line.plain
    assert "a long wall of reasoning text" not in line.plain


@pytest.mark.asyncio
async def test_toggle_thinking_expands_and_collapses_all_segments() -> None:
    """toggle_thinking flips every thinking row's expand state at once
    (segments are interspersed; the toggle is global).
    """
    transport = _make_stream_transport()
    transport.start_thinking("model_call")
    await transport.on_text_chunk("", reasoning="first think")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    await transport.on_text_chunk("", reasoning="second think")

    thinking_rows = [r for r in transport._rows if r.kind == "thinking"]
    assert len(thinking_rows) == 2
    assert all(not r.thinking_expanded for r in thinking_rows)

    transport.toggle_thinking()
    assert transport.thinking_collapsed is False
    assert all(r.thinking_expanded for r in thinking_rows)
    # Expanded rows show the full body.
    assert "first think" in transport._row_line(thinking_rows[0]).plain

    transport.toggle_thinking()
    assert transport.thinking_collapsed is True
    assert all(not r.thinking_expanded for r in thinking_rows)


# ── live height ───────────────────────────────────────────────────────


def test_live_height_floor_and_reserve() -> None:
    """Live height is terminal height minus the reserve, floored at the minimum."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.console.size = (80, 24)
    assert transport._live_height() == 24 - 4
    transport.console.size = (80, 5)
    assert transport._live_height() == 6
