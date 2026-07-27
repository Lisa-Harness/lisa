"""Shared pytest fixtures for kernel tests.

All kernel tests that need the mode or operation catalog must use the
MCP-mediated catalog loading fixture from ``conftest_catalog.py``.
"""

from __future__ import annotations

import pytest

# Import the catalog setup fixture so it's auto-use across all kernel tests.
from kernel.tests.conftest_catalog import _mcp_catalog_setup  # noqa: F401


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
