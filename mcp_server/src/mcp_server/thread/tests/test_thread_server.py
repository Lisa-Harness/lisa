"""Tests for the lisa-thread MCP server tools.

Covers tool registration and the ``thread_search`` tool's behavioral
contract using MCPServer's in-process ``call_tool`` API. The read-range
and read-recent tools are thin facades over ``ThreadStore`` and are
covered by the SDK-level ``sdk/tests/test_thread_search.py`` and
``runtime/tests/test_thread.py`` suites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _tool_text(result: Any) -> str:
    """Extract text from MCPServer ``call_tool`` result.

    MCP 2.0.0 returns a ``CallToolResult`` with a ``.content`` list of
    ``TextContent`` blocks. Older versions returned a
    ``(content_blocks, result_dict)`` tuple. Both shapes are handled.
    """
    if hasattr(result, "content") and result.content:
        blocks = result.content
        if hasattr(blocks[0], "text"):
            return blocks[0].text
        return str(blocks[0])
    if isinstance(result, tuple) and len(result) >= 1:
        blocks = result[0]
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
        if blocks:
            return str(blocks[0])
    return str(result)


def _write_active_entry(
    thread_path: Path,
    *,
    query: str = "",
    response_text: str = "",
    mode_id: str = "default",
    content_source: str = "",
    reasoning_text: str = "",
) -> None:
    """Append one entry to the active segment (``.current.jsonl``)."""
    from sdk.thread._backend import _segment_paths

    paths = _segment_paths(thread_path)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, object] = {
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
    }
    if content_source:
        entry["content_source"] = content_source
    if reasoning_text:
        entry["reasoning_text"] = reasoning_text
    with paths.current.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


@pytest.fixture
def thread_deps(tmp_path: Path) -> None:
    """Initialize the lisa-thread server with a tmp thread store."""
    from mcp_server.thread.server import init_thread

    init_thread(
        data_dir=tmp_path,
        thread_dir=str(tmp_path / "threads"),
        user_slug="tester",
    )


# ── Tool registration ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_server_lists_three_tools(thread_deps: None) -> None:
    from mcp_server.thread.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"thread_read_range", "thread_read_recent", "thread_search"} == names


# ── thread_search ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_search_returns_matching_entries(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    from mcp_server.thread.server import mcp

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(
        thread_path,
        query="unrelated topic",
        response_text="something else",
    )
    _write_active_entry(
        thread_path,
        query="fix the router bug",
        response_text="patched the routing logic",
    )

    result = await mcp.call_tool(
        "thread_search",
        {"query": "router bug"},
    )
    data = json.loads(_tool_text(result))
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["query"] == "fix the router bug"
    assert data[0]["score"] > data[1]["score"]


@pytest.mark.asyncio
async def test_thread_search_empty_query_returns_recent(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    from mcp_server.thread.server import mcp

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(thread_path, query="first", response_text="a1")
    _write_active_entry(thread_path, query="second", response_text="a2")

    result = await mcp.call_tool(
        "thread_search",
        {"query": ""},
    )
    data = json.loads(_tool_text(result))
    assert len(data) == 2
    # Recency fallback: most recent first.
    assert data[0]["query"] == "second"
    assert data[1]["query"] == "first"


@pytest.mark.asyncio
async def test_thread_search_empty_store_returns_empty_array(
    thread_deps: None,
) -> None:
    from mcp_server.thread.server import mcp

    result = await mcp.call_tool(
        "thread_search",
        {"query": "anything"},
    )
    data = json.loads(_tool_text(result))
    assert data == []


@pytest.mark.asyncio
async def test_thread_search_limit_capped(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    from mcp_server.thread.server import mcp

    thread_path = tmp_path / "threads" / "tester"
    for i in range(25):
        _write_active_entry(
            thread_path,
            query=f"shared topic {i}",
            response_text=f"answer {i}",
        )

    result = await mcp.call_tool(
        "thread_search",
        {"query": "shared topic", "limit": 100},
    )
    data = json.loads(_tool_text(result))
    assert len(data) == 20


@pytest.mark.asyncio
async def test_thread_search_result_shape(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    from mcp_server.thread.server import mcp

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(
        thread_path,
        query="test query",
        response_text="test response body",
        mode_id="planner",
    )

    result = await mcp.call_tool(
        "thread_search",
        {"query": "test"},
    )
    data = json.loads(_tool_text(result))
    assert len(data) == 1
    entry = data[0]
    assert entry["query"] == "test query"
    assert entry["mode_id"] == "planner"
    assert entry["response_preview"] == "test response body"
    assert "score" in entry
    assert "turn" in entry


# ── thread_read_range + thread_read_recent (smoke) ───────────────────


@pytest.mark.asyncio
async def test_thread_read_range_returns_entries(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    from mcp_server.thread.server import mcp

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(thread_path, query="q0", response_text="a0")
    _write_active_entry(thread_path, query="q1", response_text="a1")

    result = await mcp.call_tool(
        "thread_read_range",
        {"start_turn": 0, "end_turn": 999},
    )
    data = json.loads(_tool_text(result))
    assert len(data) == 2
    assert data[0]["query"] == "q0"
    assert data[1]["query"] == "q1"


@pytest.mark.asyncio
async def test_thread_read_range_bounds_reasoning_text_playback(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    """``reasoning_tail`` entries play back a bounded, labeled reasoning
    excerpt — the FULL raw thinking stored at rest never reaches the model
    verbatim through this tool (the thread log stays exhaustive; playback
    is bounded)."""
    from mcp_server.thread.server import mcp

    long_reasoning = "".join(
        f"deliberation step {i} with substantial padding text. " for i in range(200)
    )
    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(
        thread_path,
        query="q0",
        response_text="*The model produced no visible answer — only internal "
        "reasoning. Tail of its thinking, not a response:*\n\n> …the tail",
        content_source="reasoning_tail",
        reasoning_text=long_reasoning,
    )
    _write_active_entry(thread_path, query="q1", response_text="a1")

    result = await mcp.call_tool(
        "thread_read_range",
        {"start_turn": 0, "end_turn": 999},
    )
    data = json.loads(_tool_text(result))
    assert data[0]["content_source"] == "reasoning_tail"
    excerpt = data[0]["reasoning_excerpt"]
    # Bounded: at most 800 chars (plus the ellipsis prefix).
    assert len(excerpt) <= 801
    # Tail, not head: the excerpt ends with the reasoning's own tail.
    assert excerpt.endswith("deliberation step 199 with substantial padding text. ")
    # A normal entry carries neither marker nor excerpt.
    assert "content_source" not in data[1]
    assert "reasoning_excerpt" not in data[1]

    # The raw store keeps the FULL reasoning — exhaustive at rest.
    from sdk.thread import ThreadStore

    raw = ThreadStore(thread_path).load_recent(limit=5)
    assert raw[0]["reasoning_text"] == long_reasoning


@pytest.mark.asyncio
async def test_thread_read_recent_returns_entries(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    from mcp_server.thread.server import mcp

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(thread_path, query="q0", response_text="a0")
    _write_active_entry(thread_path, query="q1", response_text="a1")

    result = await mcp.call_tool(
        "thread_read_recent",
        {"limit": 5},
    )
    data = json.loads(_tool_text(result))
    assert len(data) == 2


# ── Read-path redaction ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_read_range_redacts_default_patterns_by_default(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    """Without env overrides, served entries apply the default redaction
    policy: harness-authored cap-fire prose is struck through before it
    ever reaches the model.

    Regression for the 2026-08-30 remote-box relapse: the model called
    ``thread_read_range`` and pulled its own prior fabricated budget
    claims back into context verbatim (the host's in-process excerpt was
    redacted since 2026-08-27, but the MCP server's store was not). The
    server now builds its store through the same env-resolved policy the
    host uses.
    """
    from mcp_server.thread.server import mcp

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(
        thread_path,
        query="do the work",
        response_text=(
            "I've hit my iteration budget in default (stuck in default for 6 iterations (cap))."
        ),
    )
    _write_active_entry(thread_path, query="next", response_text="clean answer")

    result = await mcp.call_tool(
        "thread_read_range",
        {"start_turn": 0, "end_turn": 999},
    )
    data = json.loads(_tool_text(result))
    assert len(data) == 2
    assert "I've hit my iteration budget" not in data[0]["response_text"]
    assert "stuck in default for 6 iterations" not in data[0]["response_text"]
    # The clean entry and the raw store are untouched.
    assert data[1]["response_text"] == "clean answer"

    from sdk.thread import ThreadStore

    raw = ThreadStore(thread_path).load_recent(limit=5)
    assert "I've hit my iteration budget" in raw[0]["response_text"]


@pytest.mark.asyncio
async def test_thread_read_recent_env_patterns_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-empty ``LISA_THREAD_REDACTION_PATTERNS`` replaces defaults
    entirely (hard cutover, same contract as the host)."""
    from mcp_server.thread.server import mcp

    monkeypatch.setenv("LISA_THREAD_REDACTION_PATTERNS", r"hunter2")
    from mcp_server.thread.server import init_thread

    init_thread(
        data_dir=tmp_path,
        thread_dir=str(tmp_path / "threads"),
        user_slug="tester",
    )

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(
        thread_path,
        query="secret query with hunter2 inside",
        response_text="I've hit my iteration budget for this turn.",
    )

    result = await mcp.call_tool("thread_read_recent", {"limit": 5})
    data = json.loads(_tool_text(result))
    # Custom pattern fired; the default cap-fire pattern did NOT.
    assert "hunter2" not in data[0]["query"]
    assert "I've hit my iteration budget" in data[0]["response_text"]


@pytest.mark.asyncio
async def test_thread_read_range_redaction_disabled_via_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``LISA_THREAD_REDACTION_ENABLED=0`` builds an unredacted store."""
    from mcp_server.thread.server import init_thread, mcp

    monkeypatch.setenv("LISA_THREAD_REDACTION_ENABLED", "0")
    init_thread(
        data_dir=tmp_path,
        thread_dir=str(tmp_path / "threads"),
        user_slug="tester",
    )

    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(
        thread_path,
        query="q",
        response_text="stuck in default for 6 iterations (cap).",
    )

    result = await mcp.call_tool(
        "thread_read_range",
        {"start_turn": 0, "end_turn": 999},
    )
    data = json.loads(_tool_text(result))
    assert "stuck in default for 6 iterations (cap)" in data[0]["response_text"]
