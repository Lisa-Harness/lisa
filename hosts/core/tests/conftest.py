"""Test fixtures for hosts.core tests.

Registers the lisa-catalog MCP server so ``get_workflow_catalog()`` works
in tests that exercise command dispatch (``/help``, ``/debug``, etc.).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from kernel.ipsar_fsm import (
    get_workflow_catalog,
    register_content_mcp,
    register_operation_callbacks,
)
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
    Path(__file__).resolve().parents[3] / "sdk" / "packs" / "memory"
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
            "LISA_SESSION_PROVIDER_SELECTIONS": json.dumps(
                {"modes": {"default": "lisa-pack-core"}}
            ),
        },
    )


@pytest.fixture(scope="session", autouse=True)
def _mcp_catalog_setup() -> None:
    """Wire the catalog MCP, operation callbacks, and content registry.

    This mirrors what ``hosts.core.mcp_pool.setup_framework(runtime)`` does,
    but without requiring a HostRuntime: it builds the catalog and content
    MCP configs directly and registers them so the kernel can load catalogs
    and read authored text via the content registry.
    """
    config = _make_catalog_mcp_config()
    register_operation_mcp(config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(config)

    # Wire the MCPContentRegistry as the content registry for context frames.
    # This is needed by render_context_packet_markdown and turn hooks.
    from kernel.context.context_frames import set_content_registry  # noqa: PLC0415
    from kernel.context.mcp_content_registry import MCPContentRegistry  # noqa: PLC0415

    set_content_registry(MCPContentRegistry(config))

    # Eagerly load the workflow catalog now (sync context, before any async test runs).
    get_workflow_catalog()