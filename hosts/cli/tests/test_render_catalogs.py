"""Tests for CLI render helpers that list operations/tools for allowlist authoring.

Covers ``hosts.cli.render.render_operations``, ``render_tools`` and
``render_mcp_tools``. These helpers read the kernel's in-process
operation/tool catalogs plus a pre-collected list of MCP server tool
pairs, and render Rich tables whose first column is the ID to copy into a
mode's ``allowed_operation_ids`` / ``allowed_tool_ids`` front-matter (or,
for MCP tools, the ``tool_policy`` block of ``mcp_servers.yaml``).

The kernel catalogs are populated by ``register_operation_mcp`` /
``register_tool_mcp`` at startup (which call the ``lisa-catalog`` MCP
server). To keep these tests hermetic we monkeypatch
``get_operation_catalog`` / ``get_tool_catalog`` to return fixed
definitions, and build a minimal ``HostRuntime``.

The async MCP tool collector (``hosts.cli.main._collect_mcp_tools``) is
covered here too, via a stub runtime whose ``start``/``aclose`` and pool
``list_tools`` are fakes — we never spawn a real MCP subprocess.
"""

from __future__ import annotations

# pylint: disable=protected-access

from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from hosts.cli import render as render_mod
from hosts.core.config import HostConfig
from hosts.core.runtime import HostRuntime
from kernel.models import OperationDefinition, ToolDefinition
from kernel.operations import registry as operations_registry
from kernel.tools import registry as tools_registry


# ── Helpers ───────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> HostRuntime:
    config = HostConfig.from_env(data_dir=str(tmp_path))
    return HostRuntime(config)


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
    """render_tools emits one row per internal tool; no MCP section without mcp_tools."""
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
        table = render_mod.render_tools(_make_runtime(tmp_path), mcp_tools=None, console=console)

    output = console.export_text()
    assert "memory.add" in output
    assert "Add a memory" in output
    assert "5" in output  # max_calls_per_turn
    assert "45s" in output  # timeout_seconds
    # No MCP tools provided → no MCP section.
    assert "MCP Server Tools" not in output
    assert len(table.rows) == 1


def test_render_tools_emits_mcp_section_when_pairs_provided(tmp_path: Path) -> None:
    """When mcp_tools triples are passed, a second MCP table is rendered."""
    mcp_pairs = [("brave", "brave_search", False), ("github", "create_issue", False)]
    console = Console(record=True, width=120)

    with patch.object(tools_registry, "get_tool_catalog", return_value={}):
        render_mod.render_tools(_make_runtime(tmp_path), mcp_tools=mcp_pairs, console=console)

    output = console.export_text()
    assert "MCP Server Tools" in output
    assert "brave" in output
    assert "brave_search" in output
    assert "github" in output
    assert "create_issue" in output
    # Native tools (is_alias=False) are NOT annotated with an arrow.
    assert "->" not in output


def test_render_tools_empty_mcp_pairs_omits_mcp_section(tmp_path: Path) -> None:
    """An empty mcp_tools list renders only the internal-tools table."""
    console = Console(record=True, width=120)
    with patch.object(tools_registry, "get_tool_catalog", return_value={}):
        render_mod.render_tools(_make_runtime(tmp_path), mcp_tools=[], console=console)
    assert "MCP Server Tools" not in console.export_text()


# ── render_mcp_tools ──────────────────────────────────────────────────


def test_render_mcp_tools_lists_pairs_sorted_as_given() -> None:
    """render_mcp_tools emits one row per (server_id, tool_name) triple."""
    pairs = [("zsrv", "zeta", False), ("asrv", "alpha", False)]
    console = Console(record=True, width=120)
    table = render_mod.render_mcp_tools(pairs, console=console)
    output = console.export_text()
    assert "MCP Server Tools" in output
    assert "asrv" in output and "alpha" in output
    assert "zsrv" in output and "zeta" in output
    assert len(table.rows) == 2


def test_render_mcp_tools_empty_renders_header_only() -> None:
    """An empty pair list renders the table header with no data rows."""
    console = Console(record=True, width=80)
    table = render_mod.render_mcp_tools([], console=console)
    assert len(table.rows) == 0
    assert table.title is not None


def test_render_mcp_tools_annotates_alias_with_arrow() -> None:
    """Synthetic alias entries are prefixed with → in the Tool column."""
    pairs = [
        ("lisa-memory", "memory.search", True),
        ("brave", "brave_search", False),
    ]
    console = Console(record=True, width=120)
    render_mod.render_mcp_tools(pairs, console=console)
    output = console.export_text()
    # The alias row shows the arrow annotation.
    assert "-> memory.search" in output
    # The native tool is shown without the arrow.
    assert "-> brave_search" not in output
    assert "brave_search" in output


# ── _collect_mcp_tools (async, in hosts.cli.main) ─────────────────────


class _FakeSpec:
    def __init__(self, server_id: str, name: str, synthetic: bool = False) -> None:
        self.server_id = server_id
        self.name = name
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

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[_FakeSpec]:
        return list(self._tools)

    def is_synthetic_tool(self, name: str) -> bool:
        # Mirrors the real MCPToolPool.aclose() which clears
        # _synthetic_handlers — once the pool is closed, no name is synthetic.
        if self.closed:
            return False
        return name in self._synthetic_names

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
async def test_collect_mcp_tools_returns_sorted_unique_pairs() -> None:
    """_collect_mcp_tools starts the pool, reads list_tools, dedupes, sorts, closes."""
    from hosts.cli.main import _collect_mcp_tools

    pool = _FakePool(
        [
            _FakeSpec("brave", "brave_search"),
            _FakeSpec("github", "create_issue"),
            _FakeSpec("brave", "brave_search"),  # duplicate → deduped
        ],
        synthetic_names=set(),
    )
    runtime = _StubRuntime(warm=1, cold=0, pool=pool)

    entries = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]

    assert entries == [
        ("brave", "brave_search", False),
        ("github", "create_issue", False),
    ]
    assert runtime.started is True
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_collect_mcp_tools_flags_synthetic_aliases() -> None:
    """Synthetic tools are flagged with is_alias=True; native tools with False."""
    from hosts.cli.main import _collect_mcp_tools

    pool = _FakePool(
        [
            _FakeSpec("lisa-memory", "memory.search"),
            _FakeSpec("brave", "brave_search"),
        ],
        synthetic_names={"memory.search"},
    )
    runtime = _StubRuntime(warm=1, cold=0, pool=pool)

    entries = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]

    by_name = {name: is_alias for _srv, name, is_alias in entries}
    assert by_name["memory.search"] is True
    assert by_name["brave_search"] is False


@pytest.mark.asyncio
async def test_collect_mcp_tools_no_configs_returns_empty_without_starting() -> None:
    """When there are no MCP configs, the pool is never started or closed."""
    from hosts.cli.main import _collect_mcp_tools

    runtime = _StubRuntime(warm=0, cold=0, pool=None)
    pairs = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]
    assert pairs == []
    assert runtime.started is False
    assert runtime.closed is False  # early-returns before the try/finally


@pytest.mark.asyncio
async def test_collect_mcp_tools_pool_none_after_start_returns_empty() -> None:
    """If start() leaves mcp_tool_pool None, the helper returns empty."""
    from hosts.cli.main import _collect_mcp_tools

    runtime = _StubRuntime(warm=1, cold=0, pool=None)
    pairs = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]
    assert pairs == []


@pytest.mark.asyncio
async def test_collect_mcp_tools_start_failure_returns_empty_and_closes() -> None:
    """A start() exception is swallowed; aclose still runs in finally."""
    from hosts.cli.main import _collect_mcp_tools

    class _BoomRuntime(_StubRuntime):
        async def start(self) -> None:
            raise RuntimeError("server down")

    runtime = _BoomRuntime(warm=1, cold=0, pool=None)
    pairs = await _collect_mcp_tools(runtime)  # type: ignore[arg-type]
    assert pairs == []
    assert runtime.closed is True


# ── render module exports ─────────────────────────────────────────────


def test_render_helpers_are_exported() -> None:
    """The render helpers are part of the render module's public API."""
    assert "render_operations" in render_mod.__all__
    assert "render_tools" in render_mod.__all__
    assert "render_mcp_tools" in render_mod.__all__
    assert callable(render_mod.render_operations)
    assert callable(render_mod.render_tools)
    assert callable(render_mod.render_mcp_tools)
