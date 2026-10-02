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
from mcp_server.thread.server import mcp


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
    timestamp_utc: str | None = None,
    content_source: str = "",
    reasoning_text: str = "",
    transcript: list[dict[str, Any]] | None = None,
    task_id: str = "",
) -> None:
    """Append one entry to the active segment (``.current.jsonl``)."""
    from sdk.thread._backend import _segment_paths

    paths = _segment_paths(thread_path)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, object] = {
        "mode_id": mode_id,
        "timestamp_utc": timestamp_utc,
        "query": query,
        "response_text": response_text,
    }
    if content_source:
        entry["content_source"] = content_source
    if reasoning_text:
        entry["reasoning_text"] = reasoning_text
    if transcript is not None:
        entry["transcript"] = transcript
    if task_id:
        entry["provenance"] = {"task_id": task_id}
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


@pytest.mark.asyncio
async def test_thread_read_range_bounds_total_output(thread_deps: None, tmp_path: Path) -> None:
    thread_path = tmp_path / "threads" / "tester"
    for turn in range(376):
        _write_active_entry(thread_path, query="q" * 10000, response_text=str(turn) * 1000)

    text = _tool_text(await mcp.call_tool("thread_read_range", {"start_turn": 0, "end_turn": 375}))
    assert len(text.encode("utf-8")) <= 16384
    page = json.loads(text)
    assert page["has_more"] is True
    assert page["next_cursor"] is not None
    assert page["entries"][0]["query"]["abbreviated"] is True


# ── Tool registration ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_server_lists_five_tools(thread_deps: None) -> None:
    from mcp_server.thread.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {
        "thread_read_range",
        "thread_read_recent",
        "thread_search",
        "thread_read_content",
        "thread_find_content",
        "thread_task_view",
    } == names


@pytest.mark.asyncio
async def test_thread_read_range_include_transcript_is_restorable(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    from mcp_server.thread.server import mcp
    from sdk.thread._backend import _segment_paths

    paths = _segment_paths(tmp_path / "threads" / "tester")
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    long_output = "x" * 5000
    entry = {
        "mode_id": "worker",
        "query": "read it",
        "response_text": long_output,
        "transcript": [{"role": "tool", "tool_call_id": "c1", "content": long_output}],
    }
    paths.current.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    compact = json.loads(
        _tool_text(await mcp.call_tool("thread_read_range", {"start_turn": 0, "end_turn": 0}))
    )
    full = json.loads(
        _tool_text(
            await mcp.call_tool(
                "thread_read_range",
                {"start_turn": 0, "end_turn": 0, "include_transcript": True},
            )
        )
    )

    assert "transcript" not in compact["entries"][0]
    projected = full["entries"][0]
    assert projected["response_text"]["abbreviated"] is True
    stub = projected["transcript"][0]["content"]
    assert stub["abbreviated"] is True
    restored = json.loads(
        _tool_text(
            await mcp.call_tool(
                "thread_read_content",
                {"ref": stub["ref"], "limit_chars": 10000},
            )
        )
    )
    assert restored["content"] == long_output
    assert restored["has_more"] is False
    assert restored["source"]["scope"] == "recorded_history"
    assert restored["source"]["timestamp_utc"] is None
    assert restored["source"]["source_kind"] == "tool"


# ── thread_search ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_filtered_evidence_can_be_found_and_expanded(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    thread_path = tmp_path / "threads" / "tester"
    content = "padding " * 3000 + "inside-result evidence"
    transcript = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call",
                    "function": {"name": "files.read", "arguments": '{"path":"src/needle.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": content},
    ]
    for task_id in ("task-other", "task-1"):
        _write_active_entry(thread_path, query="read", transcript=transcript, task_id=task_id)
    search = json.loads(
        _tool_text(
            await mcp.call_tool(
                "thread_search",
                {
                    "query": "inside-result",
                    "tool_name": "read",
                    "path": "src/needle.py",
                    "task_id": "task-1",
                    "start_turn": 1,
                    "end_turn": 1,
                },
            )
        )
    )
    assert [entry["turn"] for entry in search["entries"]] == [1]
    match = search["entries"][0]["matches"][0]
    assert match["ref"] == "t1.m1.content"
    found = json.loads(
        _tool_text(
            await mcp.call_tool(
                "thread_find_content",
                {
                    "ref": match["ref"],
                    "query": "inside-result",
                },
            )
        )
    )
    location = found["matches"][0]
    assert location["match_offset_chars"] == content.index("inside-result")
    restored = json.loads(
        _tool_text(
            await mcp.call_tool(
                "thread_read_content",
                {
                    "ref": match["ref"],
                    "offset_chars": location["offset_chars"],
                    "limit_chars": 320,
                },
            )
        )
    )
    assert restored["content"] == content[location["offset_chars"] :][:320]
    assert "inside-result evidence" in restored["content"]


@pytest.mark.asyncio
async def test_thread_task_view_summarizes_latest_recorded_task(
    thread_deps: None,
    tmp_path: Path,
) -> None:
    thread_path = tmp_path / "threads" / "tester"
    _write_active_entry(
        thread_path,
        query="add the endpoint",
        response_text="done wiring",
        task_id="task-1",
        timestamp_utc="2026-10-02T16:39:06+00:00",
    )
    view = json.loads(_tool_text(await mcp.call_tool("thread_task_view", {})))
    assert view["task_id"] == "task-1"
    assert view["objective"]["text"] == "add the endpoint"
    assert view["status"] == "open"
    assert view["scope"] == "recorded_history"
    assert view["selection"] == "latest_recorded"
    assert view["status_basis"] == "recorded_observation_not_live_state"
    assert view["as_of_utc"] == "2026-10-02T16:39:06+00:00"
    restored = json.loads(
        _tool_text(await mcp.call_tool("thread_read_content", {"ref": view["objective_ref"]}))
    )
    assert restored["content"] == "add the endpoint"
    assert restored["source"]["timestamp_utc"] == view["as_of_utc"]
    assert restored["source"]["source_kind"] == "user"


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
        timestamp_utc="2026-10-02T16:39:06+00:00",
    )

    result = await mcp.call_tool(
        "thread_search",
        {"query": "router bug"},
    )
    data = json.loads(_tool_text(result))
    assert len(data["entries"]) == 2
    assert data["entries"][0]["query"] == "fix the router bug"
    assert data["entries"][0]["score"] > data["entries"][1]["score"]
    assert data["scope"] == "recorded_history"
    assert data["entries"][0]["timestamp_utc"] == "2026-10-02T16:39:06+00:00"


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
    assert len(data["entries"]) == 2
    # Recency fallback: most recent first.
    assert data["entries"][0]["query"] == "second"
    assert data["entries"][1]["query"] == "first"


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
    assert data == {
        "scope": "recorded_history",
        "interpretation": (
            "Recorded exchanges, not the active request; "
            "earlier instructions and diagnoses are historical."
        ),
        "entries": [],
        "next_cursor": None,
        "has_more": False,
    }


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
    assert len(data["entries"]) == 20


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
    assert len(data["entries"]) == 1
    entry = data["entries"][0]
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
    assert len(data["entries"]) == 2
    assert data["entries"][0]["query"] == "q0"
    assert data["entries"][1]["query"] == "q1"


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
    data = json.loads(_tool_text(result))["entries"]
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
    assert len(data["entries"]) == 2


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
    data = json.loads(_tool_text(result))["entries"]
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
    data = json.loads(_tool_text(result))["entries"]
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
    data = json.loads(_tool_text(result))["entries"]
    assert "stuck in default for 6 iterations (cap)" in data[0]["response_text"]
