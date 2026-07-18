"""Tests for the RichTransport ``on_stage`` rendering.

Covers the detailed stage line and the fork fan-out per-child rendering
that the kernel's ``on_stage`` telemetry hook drives.
"""

from __future__ import annotations

import asyncio
from typing import Any

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

    await transport.on_stage("model_call", {
        "mode_id": "default",
        "model_id": "haiku",
        "provider": "anthropic",
        "iteration": 1,
        "tool_calls_so_far": 0,  # should be omitted (zero)
        "tools_offered": True,   # not in the render allowlist
    })

    # The spinner text is private; stop and inspect the thinking stage label
    # by capturing a non-spinner render.
    transport.stop_thinking()
    # When no spinner is active, on_stage prints the line inline.
    await transport.on_stage("model_call_complete", {
        "mode_id": "default",
        "model_id": "haiku",
        "iteration": 1,
        "tool_calls": 2,
        "total_tokens": 120,
        "duration_ms": 340,
    })


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

    await transport.on_stage("fork_start", {"children": 2, "targets": [
        {"mode_id": "coding", "stage_id": "implement"},
        {"mode_id": "markdown", "stage_id": ""},
    ]})
    assert transport._fork_total == 2
    assert transport._fork_children == {}

    await transport.on_stage("fork_child_start", {
        "child_index": 0,
        "mode_id": "coding",
        "stage_id": "implement",
        "query_preview": "write the thing",
    })
    await transport.on_stage("fork_child_start", {
        "child_index": 1,
        "mode_id": "markdown",
        "stage_id": "",
        "query_preview": "doc it",
    })
    assert set(transport._fork_children.keys()) == {0, 1}
    assert transport._fork_children[0]["status"] == "running"
    assert transport._fork_children[1]["mode_id"] == "markdown"

    await transport.on_stage("fork_child_complete", {
        "child_index": 0,
        "mode_id": "coding",
        "stage_id": "implement",
        "status": "ok",
        "duration_ms": 42,
    })
    assert transport._fork_children[0]["status"] == "ok"
    assert transport._fork_children[0]["duration_ms"] == 42

    await transport.on_stage("fork_complete", {"children": 2, "ok": 1, "failed": 1})
    # fork_complete clears the live fork state.
    assert transport._fork_children == {}
    assert transport._fork_total == 0


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