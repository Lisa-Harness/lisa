"""Canonical data-dir category names, re-exported from :mod:`kernel.layout`.

The kernel owns the layout contract so it never imports a higher layer;
pack authors keep importing from ``sdk.layout``.
"""

from __future__ import annotations

from kernel.layout import (
    CACHE_DIR,
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    MCP_LOGS_SUBDIR,
    RUN_DIR,
    STATE_DIR,
    data_subdir,
)

__all__ = [
    "CACHE_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "LOGS_DIR",
    "MCP_LOGS_SUBDIR",
    "RUN_DIR",
    "STATE_DIR",
    "data_subdir",
]
