"""Tests for the ``/mcp`` command render paths in ``runtime.commands._mcp``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import McpServerHealth
from kernel.tools.mcp_client import MCPToolSpec
from runtime.commands import _mcp


def _make_health(server_id: str, **overrides: Any) -> McpServerHealth:
    return McpServerHealth(server_id=server_id, **overrides)


def _fake_runtime(pool: Any) -> SimpleNamespace:
    return SimpleNamespace(
        resources=SimpleNamespace(mcp_tool_pool=pool),
        session=SimpleNamespace(
            current_mode_id="",
            current_stage_id="",
            next_model_id="",
        ),
    )


@pytest.fixture
def _stub_command_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.commands._simple._set_turn_context", lambda *a, **k: None)
    monkeypatch.setattr("runtime.commands._simple._append_turn_log", lambda *a, **k: None)
    monkeypatch.setattr("runtime.commands._simple._append_thread_entry", lambda *a, **k: None)


# ── render_mcp_summary ───────────────────────────────────────────────


def test_render_mcp_summary_empty_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {})

    assert _mcp.render_mcp_summary(_fake_runtime(object())) == "No MCP servers are configured."


def test_render_mcp_summary_renders_sorted_table(monkeypatch: pytest.MonkeyPatch) -> None:
    health = {
        "zzz": _make_health(
            "zzz",
            status="warm",
            warm=True,
            transport="http",
            url="https://example.test/mcp|pipe",
            tool_count=3,
        ),
        "aaa": _make_health(
            "aaa",
            status="cold",
            transport="stdio",
            command="/usr/bin/aaa",
            last_error="",
        ),
        "mid": _make_health(
            "mid",
            status="failed",
            transport="stdio",
            command="mid",
            last_error="boom",
        ),
    }
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: health)

    text = _mcp.render_mcp_summary(_fake_runtime(object()))

    lines = text.splitlines()
    assert lines[0] == "## MCP Servers"
    assert any(line.startswith("| Server | Status |") for line in lines)
    # Sorted by server_id: aaa, mid, zzz.
    data_rows = [
        line
        for line in lines
        if line.startswith("|") and "Server" not in line and "---" not in line
    ]
    assert [row.split("|")[1].strip() for row in data_rows] == ["aaa", "mid", "zzz"]
    # Cold server with zero tools shows an em-dash, not "0".
    assert "| aaa | cold | stdio | /usr/bin/aaa | — | — |" in lines
    # Warm HTTP server escapes the pipe in its URL cell.
    assert "| zzz | warm | http | https://example.test/mcp\\|pipe | 3 | — |" in lines
    assert "| mid | failed | stdio | mid | 0 | boom |" in lines


def test_render_mcp_summary_no_command_or_url(monkeypatch: pytest.MonkeyPatch) -> None:
    health = {
        "none": _make_health("none", status="unknown", transport="stdio", tool_count=7),
    }
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: health)

    text = _mcp.render_mcp_summary(_fake_runtime(object()))

    assert "| none | unknown | stdio | — | 7 | — |" in text.splitlines()


# ── handle_mcp_command ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_mcp_command_summary(_stub_command_log: None, monkeypatch: Any) -> None:
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {})
    runtime = _fake_runtime(object())

    result = await _mcp.handle_mcp_command(runtime, "/mcp", "/mcp")

    assert result == "No MCP servers are configured."
    assert runtime.session.current_mode_id == "mcp"
    assert runtime.session.current_stage_id == "-"
    assert runtime.session.next_model_id == "-"


@pytest.mark.asyncio
async def test_handle_mcp_command_detail(_stub_command_log: None, monkeypatch: Any) -> None:
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {})

    result = await _mcp.handle_mcp_command(_fake_runtime(object()), "/mcp  github extra", "raw")

    assert result == "No MCP server named 'github' is configured."


# ── render_mcp_server_detail ─────────────────────────────────────────


class _FakePool:
    def __init__(self, *, tools: dict[str, list[MCPToolSpec]] | None = None, boom: bool = False):
        self._tools = tools or {}
        self._boom = boom
        self.list_tools_calls: list[set[str] | None] = []

    def tools_by_server(self) -> dict[str, list[MCPToolSpec]]:
        return self._tools

    async def list_tools(self, allowed_server_ids: set[str] | None = None) -> list[MCPToolSpec]:
        self.list_tools_calls.append(allowed_server_ids)
        if self._boom:
            raise RuntimeError("connect failed")
        return []


def _spec(
    name: str, *, description: str = "desc", schema: dict[str, Any] | None = None
) -> MCPToolSpec:
    return MCPToolSpec(
        server_id="srv", name=name, description=description, input_schema=schema or {}
    )


@pytest.mark.asyncio
async def test_render_mcp_server_detail_unknown_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {})

    result = await _mcp.render_mcp_server_detail(_fake_runtime(object()), "nope")

    assert result == "No MCP server named 'nope' is configured."


@pytest.mark.asyncio
async def test_render_mcp_server_detail_full_server(monkeypatch: pytest.MonkeyPatch) -> None:
    health = _make_health(
        "srv",
        status="warm",
        warm=True,
        transport="http",
        url="https://srv.test/mcp",
        permanently_failed=True,
        restart_count=2,
        last_error="earlier failure",
    )
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {"srv": health})
    specs = [
        _spec(
            "search|things",
            schema={"required": ["query", "limit"], "properties": {"query": {}}},
        ),
        _spec("long_desc", description="x" * 200, schema={"properties": {"a": {}, "b": {}}}),
        _spec("no_schema", description=""),
    ]
    pool = _FakePool(tools={"srv": specs})

    result = await _mcp.render_mcp_server_detail(_fake_runtime(pool), "srv")

    lines = result.splitlines()
    assert lines[0] == "## MCP Server: srv"
    assert "- status: warm" in lines
    assert "- transport: http" in lines
    assert "- url: https://srv.test/mcp" in lines
    assert "- warm: True" in lines
    assert "- permanently_failed: True" in lines
    assert "- restart_count: 2" in lines
    assert "- last_error: earlier failure" in lines
    assert "- tools: 3" in lines
    # Explicit required list wins over properties; pipe in name is escaped.
    assert "| search\\|things | desc | query, limit |" in lines
    # No required list: falls back to property keys; long description truncates.
    assert any(line.startswith("| long_desc | ") and "…" in line for line in lines)
    long_row = next(line for line in lines if line.startswith("| long_desc"))
    assert long_row.endswith("| a, b |")
    # Empty schema renders "—" for required args; empty description becomes "—".
    assert "| no_schema | — | — |" in lines
    # Tools were already indexed: no on-demand warm-up happened.
    assert pool.list_tools_calls == []


@pytest.mark.asyncio
async def test_render_mcp_server_detail_stdio_and_minimal_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = _make_health(
        "srv",
        status="cold",
        warm=False,
        transport="stdio",
        command="/bin/srv",
    )
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {"srv": health})
    specs = [_spec("echo", schema={"required": []})]
    pool = _FakePool(tools={"srv": specs})

    result = await _mcp.render_mcp_server_detail(_fake_runtime(pool), "srv")

    lines = result.splitlines()
    # command branch (not url) renders; falsy flags/restart/last_error are omitted.
    assert "- command: /bin/srv" in lines
    assert not any(line.startswith("- url:") for line in lines)
    assert not any(line.startswith("- permanently_failed") for line in lines)
    assert not any(line.startswith("- restart_count") for line in lines)
    assert not any(line.startswith("- last_error") for line in lines)
    # Empty required list with empty properties dict → "—".
    assert "| echo | desc | — |" in lines


@pytest.mark.asyncio
async def test_render_mcp_server_detail_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    health = _make_health("srv", status="warm", warm=True, transport="stdio", command="srv")
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {"srv": health})
    pool = _FakePool(tools={"srv": []})

    result = await _mcp.render_mcp_server_detail(_fake_runtime(pool), "srv")

    assert "- tools: 0" in result.splitlines()
    assert "_No tools discovered (server may be cold or failed)._" in result


# ── _collect_server_tools warm-up paths ──────────────────────────────


@pytest.mark.asyncio
async def test_render_mcp_server_detail_no_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    health = _make_health("srv", status="cold")
    monkeypatch.setattr(_mcp, "gather_mcp_pool_health", lambda pool: {"srv": health})

    result = await _mcp.render_mcp_server_detail(_fake_runtime(None), "srv")

    assert "- tools: 0" in result.splitlines()
    assert "_No tools discovered" in result


@pytest.mark.asyncio
async def test_collect_server_tools_warms_cold_server(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch

    class _WarmingPool:
        def __init__(self) -> None:
            self.warmed = False

        def tools_by_server(self) -> dict[str, list[MCPToolSpec]]:
            if not self.warmed:
                return {}
            return {"srv": [_spec("late_tool")]}

        async def list_tools(self, allowed_server_ids: set[str] | None = None) -> list[Any]:
            self.warmed = True
            return []

    pool = _WarmingPool()
    runtime = _fake_runtime(pool)

    specs = await _mcp._collect_server_tools(runtime, "srv")

    assert [s.name for s in specs] == ["late_tool"]


@pytest.mark.asyncio
async def test_collect_server_tools_warm_up_failure_returns_empty() -> None:
    pool = _FakePool(boom=True)

    specs = await _mcp._collect_server_tools(_fake_runtime(pool), "srv")

    assert specs == []
    assert pool.list_tools_calls == [{"srv"}]


@pytest.mark.asyncio
async def test_collect_server_tools_warm_up_still_missing() -> None:
    # Server warms without error but never appears in the per-server index.
    pool = _FakePool(tools={})

    specs = await _mcp._collect_server_tools(_fake_runtime(pool), "srv")

    assert specs == []


# ── _required_args / _truncate helpers ───────────────────────────────


def test_required_args_explicit_required() -> None:
    spec = _spec("t", schema={"required": ["a", "b"], "properties": {"a": {}}})
    assert _mcp._required_args(spec) == "a, b"


def test_required_args_falls_back_to_properties() -> None:
    spec = _spec("t", schema={"properties": {"x": {}, "y": {}}})
    assert _mcp._required_args(spec) == "x, y"


def test_required_args_empty_schema() -> None:
    assert _mcp._required_args(_spec("t")) == "—"


def test_required_args_non_list_required() -> None:
    spec = _spec("t", schema={"required": "not-a-list", "properties": {"k": {}}})
    assert _mcp._required_args(spec) == "k"


def test_truncate_short_and_long() -> None:
    assert _mcp._truncate("hello", 10) == "hello"
    assert _mcp._truncate("x" * 100, 10) == "xxxxxxxxxx…"
    assert _mcp._truncate("", 10) == ""
    # Exactly-at-limit strings are returned unchanged.
    assert _mcp._truncate("12345", 5) == "12345"
    # Trailing whitespace before the ellipsis is stripped.
    assert _mcp._truncate("abc   ", 5) == "abc…"
    assert _mcp._truncate("abc \n" + "z" * 50, 5) == "abc…"
