"""Canonical data-dir categories shared by hosts, servers, and packs."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = "config"
STATE_DIR = "state"
DATA_DIR = "data"
CACHE_DIR = "cache"
LOGS_DIR = "logs"
RUN_DIR = "run"
MCP_LOGS_SUBDIR = "mcp"

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


def data_subdir(data_dir: Path | str, category: str, *parts: str) -> Path:
    """Return a path inside ``<data_dir>/<category>`` from relative parts."""
    base = Path(data_dir) / category
    if not parts:
        return base
    rel = Path(*parts)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"data_subdir part must be relative: {rel!s}")
    return base / rel
