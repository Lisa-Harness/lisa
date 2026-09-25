"""Canonical data-dir category names — the single source of truth.

Every file the framework (and its packs) writes inside the host data dir
belongs to one of six lifecycle categories. The names live here so all
packages (host, kernel, sdk, mcp_server, lisa_packs.*) share one spelling
and the layout can evolve in one place.

Layout (per data dir — the base or a ``profiles/<name>`` root)::

    config/    hand-editable + resolved configuration
    state/     machine-written, small, worth backing up
    data/      large durable storage
    cache/     rm-safe, never backed up
    logs/      append-only, rotated
    run/       per-session execution state, gc-able

``.env`` and the packs ``venv/`` stay at the data-dir root; they are
contracts, not data (see hosts/cli .env loading and
``host.paths.packs_venv_for``).
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = "config"
STATE_DIR = "state"
DATA_DIR = "data"
CACHE_DIR = "cache"
LOGS_DIR = "logs"
RUN_DIR = "run"

# MCP server JSONL logs live in their own subdir so the namespaced
# server ids (e.g. ``memory.lisa-memory``) stay contained.
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
    """Return ``<data_dir>/<category>[/<parts>...]`` as an absolute path.

    The first ``part`` (if any) may be a multi-segment relative name; it is
    resolved strictly under the category dir. ``os.fspath``-style inputs
    (str) are accepted for the subprocess-server call sites that receive
    the ``--data-dir`` CLI value.
    """
    root = Path(data_dir)
    base = root / category
    if not parts:
        return base
    rel = Path(*parts)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"data_subdir part must be relative: {rel!s}")
    return base / rel
