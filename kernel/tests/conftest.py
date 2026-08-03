"""Shared pytest fixtures for kernel tests.

All kernel tests that need the mode or operation catalog must use the
MCP-mediated catalog loading fixture from ``conftest_catalog.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Import the catalog setup fixture so it's auto-use across all kernel tests.
from kernel.tests.conftest_catalog import _mcp_catalog_setup  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _wire_sub_turn_content_registry() -> None:
    """Wire a directory-based content registry for the sub-turn contract assets.

    ``kernel.context.sub_turn_contract`` loads the fork/delegate contract and
    midpoint-nudge prose from authored Markdown under
    ``sdk/packs/core/content/prompts/sub_turn/``. Production wires the
    ``MCPContentRegistry`` at host startup (``hosts.core.mcp_pool``); kernel
    unit tests that call ``_build_sub_turn_contract`` directly need the same
    registry. A lightweight direct-file registry avoids spawning the catalog
    MCP subprocess for these unit tests.
    """

    class _DirContentRegistry:
        def __init__(self, root: Path) -> None:
            self._root = root

        def read_authored_text(self, relative_path: str) -> str:
            return (self._root / relative_path).read_text(encoding="utf-8")

    core_root = (
        Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core" / "content"
    ).resolve()
    from kernel.context.sub_turn_contract import set_content_registry

    set_content_registry(_DirContentRegistry(core_root))
    yield


@pytest.fixture(autouse=True)
def _reset_routing_vocabulary() -> None:
    """Reset the session routing vocabulary between tests.

    The routing-vocabulary registry is a process global injected by the
    host boot path from the session plan. Tests that construct a
    ``HostRuntime`` register their own session's connectors; tests that
    exercise the kernel directly (without a host) expect the historical
    fallback. Resetting between tests keeps the two groups from leaking
    state into each other.
    """
    from kernel.routing_vocabulary import reset_routing_vocabulary

    reset_routing_vocabulary()
    yield
    reset_routing_vocabulary()
