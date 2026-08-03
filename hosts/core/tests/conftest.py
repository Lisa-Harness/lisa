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

_CORE_ROOT: Path = (Path(__file__).resolve().parents[3] / "sdk" / "packs" / "core").resolve()
_LEARNING_ROOT: Path = (Path(__file__).resolve().parents[3] / "sdk" / "packs" / "memory").resolve()
# search, markdown, and letta migrated to ../../lisa_packs/<name>
# (external submodules). Only include a content root when the submodule is
# checked out alongside the framework; tests that need a migrated mode skip
# when absent. compaction was promoted back to an in-tree workspace member
# (sdk/packs/compaction) and is discovered via entry points like checkpoint.
_PACKS_ROOT: Path = (Path(__file__).resolve().parents[3].parent / "lisa_packs").resolve()


def _external_pack_roots() -> list[Path]:
    """Return the content roots of migrated packs that are checked out."""
    roots: list[Path] = []
    for name in ("search", "markdown", "letta"):
        root = (_PACKS_ROOT / name / "lisa_packs" / name).resolve()
        if root.exists():
            roots.append(root)
    return roots


def _make_catalog_mcp_config(*, log_path: Path) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-catalog with the core pack.

    ``log_path`` routes the child's JSONL log out of the repo root (the
    ``configure_logging`` fallback writes ``<cwd>/lisa-catalog.jsonl`` when
    ``LISA_MCP_LOG_PATH`` is unset, and these configs leave cwd=None so the
    subprocess inherits the pytest cwd). Mirrors the env injection that
    ``hosts.core.mcp_pool.build_catalog_mcp_config`` does in production.
    """
    content_roots = [_CORE_ROOT, _LEARNING_ROOT, *_external_pack_roots()]
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
            "SDK_CONTENT_ROOTS": ";".join(str(r) for r in content_roots),
            "LISA_SESSION_PROVIDER_SELECTIONS": json.dumps(
                {
                    "modes": {"default": "lisa-pack-core"},
                    "tools": {"web_search": "lisa-pack-brave-search"},
                }
            ),
            "LISA_MCP_LOG_PATH": str(log_path),
        },
    )


@pytest.fixture(scope="session", autouse=True)
def _mcp_catalog_setup(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Wire the catalog MCP, operation callbacks, and content registry.

    This mirrors what ``hosts.core.mcp_pool.setup_framework(runtime)`` does,
    but without requiring a HostRuntime: it builds the catalog and content
    MCP configs directly and registers them so the kernel can load catalogs
    and read authored text via the content registry.
    """
    log_dir = tmp_path_factory.mktemp("lisa-catalog")
    config = _make_catalog_mcp_config(log_path=log_dir / "lisa-catalog.jsonl")
    register_operation_mcp(config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(config)

    # Wire the MCPContentRegistry as the content registry for context frames
    # and the sub-turn contract assets (fork/delegate briefings + midpoint
    # write-nudge). Both modules share the same registry instance.
    from kernel.context.context_frames import set_content_registry  # noqa: PLC0415
    from kernel.context.mcp_content_registry import MCPContentRegistry  # noqa: PLC0415
    from kernel.context.sub_turn_contract import (  # noqa: PLC0415
        set_content_registry as _set_sub_turn_content_registry,
    )

    _registry = MCPContentRegistry(config)
    set_content_registry(_registry)
    _set_sub_turn_content_registry(_registry)

    # Eagerly load the workflow catalog now (sync context, before any async test runs).
    get_workflow_catalog()
