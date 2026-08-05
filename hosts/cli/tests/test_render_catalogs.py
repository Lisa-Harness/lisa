"""Tests for CLI render helpers that list operations/tools for allowlist authoring.

Covers ``hosts.cli.render.render_operations`` and ``render_tools``. These
helpers read the kernel's in-process operation/tool catalogs and render
Rich tables whose first column is the ID to copy into a mode's
``allowed_operation_ids`` / ``allowed_tool_ids`` front-matter. MCP server
tool rendering (``render_mcp_servers`` / ``render_mcp_server_tools``) and
the async MCP tool collector (``hosts.cli.main._collect_mcp_tools``) are
covered here too, via stubs — we never spawn a real MCP subprocess.
"""

from __future__ import annotations

# pylint: disable=protected-access
from pathlib import Path
from unittest.mock import patch

import pytest
from host.runtime import HostRuntime
from hosts.cli import render as render_mod
from kernel.models import OperationDefinition, ToolDefinition
from kernel.operations import registry as operations_registry
from kernel.tools import registry as tools_registry
from rich.console import Console

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


# ── render_operations ─────────────────────────────────────────────────


def test_render_operations_lists_ids_and_kinds(tmp_path: Path) -> None:
    """render_operations emits one row per operation with the ID in column 1."""
    fake_catalog = {
        "synthesize_and_store_learning": OperationDefinition(
            operation_id="synthesize_and_store_learning",
            description="Persist a learning.",
            execution_kind="prompt",
            tool_ids=["memory.add"],
        ),
        "recall_memory": OperationDefinition(
            operation_id="recall_memory",
            description="Recall memories.",
            execution_kind="runtime",
        ),
    }
    console = Console(record=True, width=120)

    with patch.object(operations_registry, "get_operation_catalog", return_value=fake_catalog):
        table = render_mod.render_operations(_make_runtime(tmp_path), console=console)

    output = console.export_text()
    # Sorted by operation_id alphabetically; ID appears as the bold column.
    assert "recall_memory" in output
    assert "synthesize_and_store_learning" in output
    # recall_memory precedes synthesize_and_store_learning (alphabetical).
    assert output.index("recall_memory") < output.index("synthesize_and_store_learning")
    # Kind and description columns are rendered.
    assert "runtime" in output
    assert "prompt" in output
    # The tools column for the prompt operation lists its declared tool.
    assert "memory.add" in output
    assert len(table.rows) == 2


def test_render_operations_empty_catalog_renders_header_only(tmp_path: Path) -> None:
    """An empty operation catalog still returns a table with no data rows."""
    console = Console(record=True, width=80)
    with patch.object(operations_registry, "get_operation_catalog", return_value={}):
        table = render_mod.render_operations(_make_runtime(tmp_path), console=console)
    assert len(table.rows) == 0
    assert table.title is not None


# ── render_tools (internal tools only) ────────────────────────────────


def test_render_tools_lists_internal_tool_ids(tmp_path: Path) -> None:
    """render_tools emits one row per internal tool."""
    fake_tools = {
        "memory.add": ToolDefinition(
            tool_id="memory.add",
            description="Add a memory.",
            max_calls_per_turn=5,
            timeout_seconds=45,
        ),
    }
    console = Console(record=True, width=120)

    with patch.object(tools_registry, "get_tool_catalog", return_value=fake_tools):
        table = render_mod.render_tools(_make_runtime(tmp_path), console=console)

    output = console.export_text()
    assert "memory.add" in output
    assert "Add a memory" in output
    assert "5" in output  # max_calls_per_turn
    assert "45s" in output  # timeout_seconds
    assert "MCP Server Tools" not in output
    assert len(table.rows) == 1


# ── render_mcp_servers / render_mcp_server_tools ─────────────────────


def test_render_mcp_servers_lists_status_and_transport() -> None:
    """render_mcp_servers emits one row per server with status + transport."""
    from kernel.models import McpServerHealth

    health = {
        "lisa-catalog": McpServerHealth(
            server_id="lisa-catalog",
            status="warm",
            transport="venv_bin",
            command="lisa-catalog",
            tool_count=3,
        ),
        "mem0": McpServerHealth(
            server_id="mem0",
            status="failed",
            transport="http",
            url="https://mem0.ai",
            tool_count=0,
            last_error="401 unauthorized",
        ),
    }
    console = Console(record=True, width=160)
    table = render_mod.render_mcp_servers(health, console=console)
    output = console.export_text()
    assert "MCP Servers" in output
    assert "lisa-catalog" in output and "warm" in output and "venv_bin" in output
    assert "mem0" in output and "failed" in output and "http" in output
    assert "401 unauthorized" in output
    assert len(table.rows) == 2


def test_render_mcp_server_tools_lists_tools_with_descriptions() -> None:
    """render_mcp_server_tools emits one row per tool with description + required args."""
    from kernel.models import McpServerHealth
    from kernel.tools.mcp_client import MCPToolSpec

    health = McpServerHealth(
        server_id="lisa-catalog",
        status="warm",
        transport="venv_bin",
        tool_count=2,
    )
    specs = [
        MCPToolSpec("lisa-catalog", "list_modes", "List all modes.", {"type": "object"}),
        MCPToolSpec(
            "lisa-catalog",
            "list_tools",
            "List all tools.",
            {"type": "object", "required": ["mode_id"], "properties": {"mode_id": {}}},
        ),
    ]
    console = Console(record=True, width=120)
    table = render_mod.render_mcp_server_tools(specs, health, console=console)
    output = console.export_text()
    assert "lisa-catalog" in output
    assert "list_modes" in output and "List all modes" in output
    assert "list_tools" in output and "List all tools" in output
    assert "mode_id" in output  # required arg shown
    assert len(table.rows) == 2


# ── _collect_mcp_tools (async, in hosts.cli.main) ─────────────────────


class _FakeSpec:
    def __init__(self, server_id: str, name: str, synthetic: bool = False) -> None:
        self.server_id = server_id
        self.name = name
        self.description = ""
        self.input_schema: dict = {}
        self._synthetic = synthetic


class _FakePool:
    def __init__(
        self,
        tools: list[_FakeSpec],
        *,
        synthetic_names: set[str] | None = None,
    ) -> None:
        self._tools = tools
        self._synthetic_names = synthetic_names or set()
        self.closed = False
        self.configs: list = []

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[_FakeSpec]:
        return list(self._tools)

    def is_synthetic_tool(self, name: str) -> bool:
        # Mirrors the real MCPToolPool.aclose() which clears
        # _synthetic_handlers — once the pool is closed, no name is synthetic.
        if self.closed:
            return False
        return name in self._synthetic_names

    def warm_server_ids(self) -> list[str]:
        return []

    def warm_failed_ids(self) -> list[str]:
        return []

    def cold_failed_ids(self) -> list[str]:
        return []

    def permanently_failed_ids(self) -> list[str]:
        return []

    def restart_counts(self) -> dict:
        return {}

    def last_errors(self) -> dict:
        return {}

    def tools_by_server(self) -> dict:
        return {}

    def _aclose(self) -> None:
        self.closed = True
        self._synthetic_names = set()


class _StubRuntime:
    """Minimal runtime stand-in for _collect_mcp_tools.

    Exposes the attributes the helper touches: ``mcp_warm_configs`` /
    ``mcp_cold_configs`` (to gate the no-configs short circuit) and
    ``start`` / ``aclose`` / ``mcp_tool_pool`` (the pool lifecycle).
    ``start``/``aclose`` are async and record call order.
    """

    def __init__(self, *, warm: int, cold: int, pool: _FakePool | None) -> None:
        self.mcp_warm_configs = [object() for _ in range(warm)]
        self.mcp_cold_configs = [object() for _ in range(cold)]
        self._pool = pool
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def aclose(self) -> None:
        self.closed = True
        # Mirrors the real HostRuntime.aclose() → pool.aclose() path that
        # clears _synthetic_handlers before the caller inspects the pool.
        if self._pool is not None:
            self._pool._aclose()

    @property
    def mcp_tool_pool(self) -> _FakePool | None:
        return self._pool


@pytest.mark.asyncio
async def test_collect_mcp_tools_returns_tools_grouped_by_server() -> None:
    """_collect_mcp_tools starts the pool, reads list_tools, groups by server, closes."""
    from hosts.cli.main import _collect_mcp_tools

    pool = _FakePool(
        [
            _FakeSpec("brave", "brave_search"),
            _FakeSpec("github", "create_issue"),
            _FakeSpec("brave", "brave_search"),  # duplicate → both kept (grouped)
        ],
        synthetic_names=set(),
    )
    runtime = _StubRuntime(warm=1, cold=0, pool=pool)

    health, tools_by_server = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]

    assert set(tools_by_server.keys()) == {"brave", "github"}
    assert len(tools_by_server["brave"]) == 2  # both brave entries kept
    assert tools_by_server["github"][0].name == "create_issue"
    assert health == {}  # no configs on the fake pool → empty health
    assert runtime.started is True
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_collect_mcp_tools_no_configs_returns_empty_without_starting() -> None:
    """When there are no MCP configs, the pool is never started or closed."""
    from hosts.cli.main import _collect_mcp_tools

    runtime = _StubRuntime(warm=0, cold=0, pool=None)
    health, tools_by_server = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]
    assert health == {}
    assert tools_by_server == {}
    assert runtime.started is False
    assert runtime.closed is False  # early-returns before the try/finally


@pytest.mark.asyncio
async def test_collect_mcp_tools_pool_none_after_start_returns_empty() -> None:
    """If start() leaves mcp_tool_pool None, the helper returns empty."""
    from hosts.cli.main import _collect_mcp_tools

    runtime = _StubRuntime(warm=1, cold=0, pool=None)
    health, tools_by_server = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]
    assert health == {}
    assert tools_by_server == {}


@pytest.mark.asyncio
async def test_collect_mcp_tools_start_failure_returns_empty_and_closes() -> None:
    """A start() exception is swallowed; aclose still runs in finally."""
    from hosts.cli.main import _collect_mcp_tools

    class _BoomRuntime(_StubRuntime):
        async def start(self) -> None:
            raise RuntimeError("server down")

    runtime = _BoomRuntime(warm=1, cold=0, pool=None)
    health, tools_by_server = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]
    assert health == {}
    assert tools_by_server == {}
    assert runtime.closed is True


# ── render module exports ─────────────────────────────────────────────


def test_render_helpers_are_exported() -> None:
    """The render helpers are part of the render module's public API."""
    assert "render_operations" in render_mod.__all__
    assert "render_tools" in render_mod.__all__
    assert "render_mcp_servers" in render_mod.__all__
    assert "render_mcp_server_tools" in render_mod.__all__
    assert callable(render_mod.render_operations)
    assert callable(render_mod.render_tools)
    assert callable(render_mod.render_mcp_servers)
    assert callable(render_mod.render_mcp_server_tools)
