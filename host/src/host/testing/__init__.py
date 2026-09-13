"""Test-only helpers for the host runtime.

This package holds host-layer test factories that were misplaced under
``kernel.testing`` (a layer violation: they import ``host.config`` and
``host.mcp_pool``). ``host.testing`` is the correct home — it can depend on
the full host stack and is consumed by host tests.
"""

from host.testing.runtime_factory import build_test_runtime

__all__ = ["build_test_runtime"]
