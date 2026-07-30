"""Tests for the RichTransport ``on_stage`` rendering.

Covers the detailed stage line and the fork fan-out per-child rendering
that the kernel's ``on_stage`` telemetry hook drives.
"""

from __future__ import annotations


import pytest
from rich.console import Console

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

    # operation
    line = transport._render_tool_dispatch(
        {
            "tool": "write_file",
            "tool_kind": "operation",
            "iteration": 1,
            "tool_calls_so_far": 1,
        }
    )
    assert line == "stage: tool_dispatch → op: write_file (iter=1, calls=1)"

    # mcp_tool
    line = transport._render_tool_dispatch(
        {
            "tool": "brave_search",
            "tool_kind": "mcp_tool",
            "iteration": 2,
            "tool_calls_so_far": 3,
        }
    )
    assert "→ mcp: brave_search" in line

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

    # All four stage events are retained in order — the user can scroll
    # back and see the full play-by-play, not just the last line.
    assert len(transport._stage_history) >= 4
    history_text = "\n".join(transport._stage_history)
    assert "iteration=1" in history_text
    assert "search_web" in history_text
    assert "iteration=2" in history_text
    # The active spinner label is the most-recent stage.
    assert "iteration=2" in transport._active_label


@pytest.mark.asyncio
async def test_start_thinking_clears_history() -> None:
    """start_thinking resets _stage_history so a new turn starts clean."""
    console = Console(quiet=True, force_terminal=False)
    transport = RichTransport(console=console, debug=False)
    transport.start_thinking("dispatch")
    await transport.on_stage("model_call", {"mode_id": "default", "iteration": 1})
    assert transport._stage_history != []

    transport.start_thinking("dispatch")
    assert transport._stage_history == []


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
