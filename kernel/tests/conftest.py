"""Shared pytest fixtures for kernel tests.

All kernel tests that need the mode or operation catalog must use the
MCP-mediated catalog loading fixture from ``conftest_catalog.py``.
"""

from __future__ import annotations

# Import the catalog setup fixture so it's auto-use across all kernel tests.
from kernel.tests.conftest_catalog import _mcp_catalog_setup  # noqa: F401