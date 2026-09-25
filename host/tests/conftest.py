"""Test fixtures for host tests.

Registers the lisa-catalog MCP server so ``get_workflow_catalog()`` works
in tests that exercise command dispatch (``/help``, ``/debug``, etc.).
"""

from __future__ import annotations

import json
import os
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

_CORE_ROOT: Path = (Path(__file__).resolve().parents[2] / "sdk" / "packs" / "core").resolve()
_LEARNING_ROOT: Path = (Path(__file__).resolve().parents[2] / "sdk" / "packs" / "memory").resolve()
# search, markdown, and letta migrated to ../../lisa_packs/<name>
# (external submodules). Only include a content root when the submodule is
# checked out alongside the framework; tests that need a migrated mode skip
# when absent.
_PACKS_ROOT: Path = (Path(__file__).resolve().parents[2].parent / "lisa_packs").resolve()


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
    ``host.mcp_pool.build_catalog_mcp_config`` does in production.
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
            "SDK_CONTENT_ROOTS": os.pathsep.join(str(r) for r in content_roots),
            "LISA_SESSION_PROVIDER_SELECTIONS": json.dumps(
                {
                    "modes": {"default": "lisa-pack-core"},
                    "tools": {"web_search": "lisa-pack-brave"},
                }
            ),
            "LISA_MCP_LOG_PATH": str(log_path),
        },
    )


@pytest.fixture(scope="session", autouse=True)
def _mcp_catalog_setup(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Wire the catalog MCP, operation callbacks, and content registry.

    This mirrors what ``host.mcp_pool.setup_framework(runtime)`` does,
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
    from kernel.context.context_frames import set_content_registry
    from kernel.context.mcp_content_registry import MCPContentRegistry
    from kernel.context.sub_turn_contract import (
        set_content_registry as _set_sub_turn_content_registry,
    )

    _registry = MCPContentRegistry(config)
    set_content_registry(_registry)
    _set_sub_turn_content_registry(_registry)

    # Eagerly load the workflow catalog now (sync context, before any async test runs).
    get_workflow_catalog()


@pytest.fixture(autouse=True)
def _routing_vocabulary() -> None:
    """Register a representative routing vocabulary for each test.

    The kernel has no hardcoded fallbacks (2026-08-11 cutover): a reader
    raises ``RuntimeError`` when no session vocabulary is wired. Host tests
    that call ``handle_message`` read resume phrases via
    ``get_resume_phrases()``; register a representative set so the common
    case works without each test opting in.
    """
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )

    reset_routing_vocabulary()
    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by"],
        producer_intents=["research", "gather", "learn", "read", "analyze", "review", "architect"],
        consumer_intents=["code", "write", "implement", "review"],
        resume_phrases=["proceed", "continue", "go ahead", "go for it", "do it", "carry on", "pick up", "resume", "next step", "keep going"],
        topic_abandonment_markers=["side quest", "rewind", "never mind", "forget", "different topic", "not about"],
    )
    yield
    reset_routing_vocabulary()
