"""Test fixtures for hosts.core tests.

Registers the lisa-catalog MCP server so ``get_workflow_catalog()`` works
in tests that exercise command dispatch (``/help``, ``/debug``, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kernel.ipsar_fsm import register_content_mcp, register_operation_callbacks
from kernel.operations.registry import (
    execute_operation,
    get_operation_catalog,
    register_operation_mcp,
)
from kernel.tools.mcp_client import MCPClientConfig

_CORE_ROOT: Path = (
    Path(__file__).resolve().parents[3] / "sdk" / "packs" / "core"
).resolve()
_LEARNING_ROOT: Path = (
    Path(__file__).resolve().parents[3] / "sdk" / "packs" / "learning"
).resolve()
_SEARCH_ROOT: Path = (
    Path(__file__).resolve().parents[3] / "sdk" / "packs" / "search"
).resolve()


def _make_catalog_mcp_config() -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-catalog with the core pack."""
    return MCPClientConfig(
        server_id="lisa-catalog-test",
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.catalog.server",
            "--data-dir",
            str(_CORE_ROOT),
        ],
        env={
            "SDK_CONTENT_ROOTS": f"{_CORE_ROOT};{_LEARNING_ROOT};{_SEARCH_ROOT}",
        },
    )


@pytest.fixture(scope="module", autouse=True)
def _mcp_catalog_setup() -> None:
    """Wire the catalog MCP and operation callbacks so the kernel can load catalogs."""
    config = _make_catalog_mcp_config()
    register_operation_mcp(config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(config)
    # Eagerly load the workflow catalog now (sync context, before any async test runs)
    from kernel.ipsar_fsm import get_workflow_catalog

    get_workflow_catalog()