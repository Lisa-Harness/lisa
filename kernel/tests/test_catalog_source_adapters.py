"""Slice B: kernel adapters honor injected in-process catalog/content sources.

The kernel's sync adapters must route through the typed contracts in
``kernel.context._catalog_sources`` when the host injects a source, keep
the stdio transport otherwise, raise on a failing source instead of
falling back to a subprocess, and never import sdk/mcp_server/runtime.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from kernel.context._catalog_sources import CatalogSource, ContentSource
from kernel.context.mcp_catalog import MCPOperationCatalog, MCPToolCatalog
from kernel.context.mcp_content_registry import MCPContentRegistry
from kernel.tools.mcp_client import MCPClientConfig


class _FakeCatalogSource:
    """Minimal in-process catalog source returning a fixed catalog JSON."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.invalidated = 0

    def list_modes(self) -> str:
        return json.dumps(self._payload)

    def list_operations(self) -> str:
        return json.dumps(self._payload)

    def list_tools(self) -> str:
        return json.dumps(self._payload)

    def invalidate(self) -> None:
        self.invalidated += 1

    def list_content_roots(self) -> str:
        return json.dumps([])


class _FakeContentSource:
    """Minimal in-process content source with a hit-only read cache."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.invalidated = 0

    def read_authored_text(self, relative_path: str) -> str:
        self.reads.append(relative_path)
        return f"text:{relative_path}"

    def invalidate(self) -> None:
        self.invalidated += 1

    def roots_fingerprint(self) -> str:
        return "fake"


def _unused_config() -> MCPClientConfig:
    # Never spawned in in-process mode; present so the adapters are
    # constructible without a live server.
    return MCPClientConfig(server_id="lisa-catalog", command="python", args=["-c", "pass"])


@pytest.fixture(autouse=True)
def _no_subprocess_sessions() -> Iterator[None]:
    """Guard: these tests must never open a shared stdio session."""
    from kernel.context._shared_stdio import close_shared_sessions

    try:
        yield
    finally:
        close_shared_sessions()


def test_sources_satisfy_kernel_contracts() -> None:
    assert isinstance(_FakeCatalogSource({}), CatalogSource)
    assert isinstance(_FakeContentSource(), ContentSource)


def test_content_registry_routes_through_injected_source() -> None:
    source = _FakeContentSource()
    registry = MCPContentRegistry(_unused_config(), source=source)

    assert registry.read_authored_text("prompts/system-prompt.md") == (
        "text:prompts/system-prompt.md"
    )
    registry.invalidate()

    assert source.reads == ["prompts/system-prompt.md"]
    assert source.invalidated == 1


def test_operation_catalog_routes_through_injected_source() -> None:
    payload: dict[str, object] = {}
    source = _FakeCatalogSource(payload)
    loader = MCPOperationCatalog(_unused_config(), source=source)

    assert loader.get_operation_catalog() == {}
    loader.invalidate()

    assert source.invalidated == 1


def test_tool_catalog_routes_through_injected_source() -> None:
    source = _FakeCatalogSource({})
    loader = MCPToolCatalog(_unused_config(), source=source)

    assert loader.get_tool_catalog() == {}


def test_failing_source_raises_without_stdio_fallback() -> None:
    class _Boom(_FakeContentSource):
        def read_authored_text(self, relative_path: str) -> str:
            raise RuntimeError("in-process failure")

    registry = MCPContentRegistry(_unused_config(), source=_Boom())

    with pytest.raises(RuntimeError, match="in-process failure"):
        registry.read_authored_text("prompts/system-prompt.md")


def test_kernel_contract_module_does_not_import_sdk_or_runtime() -> None:
    source_text = Path(sys.modules["kernel.context._catalog_sources"].__file__ or "").read_text(
        encoding="utf-8"
    )
    import_lines = [
        line.strip()
        for line in source_text.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]

    for import_line in import_lines:
        assert not import_line.startswith(("import sdk", "from sdk")), import_line
        assert not import_line.startswith(("import runtime", "from runtime")), import_line
        assert "mcp_server" not in import_line, import_line
