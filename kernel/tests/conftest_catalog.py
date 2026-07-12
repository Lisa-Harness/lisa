"""Shared test fixture for MCP-mediated catalog loading.

Spawns the ``lisa-catalog`` MCP server with the ``sdk/packs/core`` content root
and registers it with the kernel so ``get_workflow_catalog()`` and
``get_operation_catalog()`` work in tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kernel.ipsar_fsm import register_content_mcp, register_operation_callbacks
from kernel.operations.registry import execute_operation, get_operation_catalog, register_operation_mcp
from kernel.tools.mcp_client import MCPClientConfig

_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()
_LEARNING_ROOT: Path = (
    Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "learning"
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
            "SDK_CONTENT_ROOTS": f"{_CORE_ROOT};{_LEARNING_ROOT}",
        },
    )


@pytest.fixture(scope="module", autouse=True)
def _mcp_catalog_setup() -> None:
    """Wire the catalog MCP and operation callbacks so the kernel can load catalogs.

    Order matters: operation MCP first, then operation callbacks, then content MCP.
    ``validate_workflow_catalog`` calls ``_validate_mode_operation_bindings`` which
    needs the operation callbacks registered before it can validate mode bindings.

    The catalog is eagerly loaded here during module setup (sync context) so async
    tests don't need to call ``asyncio.run()`` from inside a running event loop.
    """
    config = _make_catalog_mcp_config()
    register_operation_mcp(config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(config)
    # Eagerly load the workflow catalog now (sync context, before any async test runs)
    from kernel.ipsar_fsm import get_workflow_catalog

    get_workflow_catalog()
