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
    midpoint-nudge prose from authored Markdown. Production wires the
    ``MCPContentRegistry`` at host startup (``host.mcp_pool``) against the
    ``sdk/packs/core`` content root; kernel unit tests that call
    ``_build_sub_turn_contract`` directly need the same prose but must not
    depend on the ``sdk`` package (kernel is runtime-neutral). The canonical
    prompt assets are mirrored under ``kernel/testing/fixtures/sub_turn/`` so
    the kernel test suite is self-contained. A lightweight direct-file
    registry avoids spawning the catalog MCP subprocess for these unit tests.
    """

    class _DirContentRegistry:
        def __init__(self, root: Path) -> None:
            self._root = root

        def read_authored_text(self, relative_path: str) -> str:
            return (self._root / relative_path).read_text(encoding="utf-8")

    fixtures_root = (Path(__file__).resolve().parents[1] / "testing" / "fixtures").resolve()
    from kernel.context.sub_turn_contract import set_content_registry

    set_content_registry(_DirContentRegistry(fixtures_root))
    yield


@pytest.fixture(autouse=True)
def _reset_routing_vocabulary() -> None:
    """Reset and re-register the session routing vocabulary between tests.

    The routing-vocabulary registry is a process global injected by the
    host boot path from the session plan. The kernel has no hardcoded
    fallbacks (the 2026-08-11 hard cutover): a reader raises
    ``RuntimeError`` when no session vocabulary is wired. Kernel-direct
    tests that exercise ``_compute_capability_routing_hint`` or
    ``_query_is_sequential`` need a wired registry. Reset between tests
    to clear state, then register a representative vocabulary so the
    common case works without each test opting in.
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
    )
    yield
    reset_routing_vocabulary()
    reset_routing_vocabulary()
