"""Lisa Storage MCP server — thin facade over :class:`sdk.storage.StorageBackend`.

This is the sole storage server for the framework. It is a transport-only
wrapper: every tool delegates to the SDK backend, which owns the sandbox
resolution, deny-glob policy, checkpoint persistence, and model-manifest
read/write. Hosts or tools that want in-process storage access can use
:class:`sdk.storage.StorageBackend` directly without spawning this server.

Tools (11 total):

- File I/O (6): ``read_file``, ``write_file``, ``list_files``, ``delete_file``,
  ``stat_file``, ``make_dir`` — scoped to ``data_dir / "files"``.
- Checkpoints (3): ``save_checkpoint``, ``restore_checkpoint``, ``list_checkpoints``
  — serialized checkpoint bundles as JSON under ``data_dir / "checkpoints"``.
- Model manifests (2): ``read_model_manifest``, ``write_model_manifest`` —
  ``models.yaml`` / ``models.proposed.yaml`` directly under ``data_dir``.

``--readonly`` disables writes, deletes, checkpoint saves, and manifest
writes. ``--deny-glob`` (repeatable) + ``LISA_STORAGE_DENY_GLOBS`` env add
glob patterns to the file-sandbox denylist.

Storage is an MCP feature, not a kernel seam — there is no ``BlobIO``
Protocol in ``kernel/seams/protocols.py``. Hosts that need full blob
semantics (leases, content-type, range reads) wire their own MCP server;
the kernel does not care.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir
from sdk.storage import StorageBackend

__all__ = ["init_storage", "main", "mcp"]

logger = logging.getLogger(__name__)

mcp = FastMCP("Lisa Storage Server")

_backend: StorageBackend | None = None


# ── Initialization ──────────────────────────────────────────────────────────


def init_storage(
    data_dir: Path,
    *,
    readonly: bool = False,
    deny_globs: tuple[str, ...] = (),
) -> None:
    """Initialize the storage server with a resolved data directory.

    Called by ``main()`` after arg parsing. Tests may call this directly.
    Creates the ``files/`` and ``checkpoints/`` subdirectories if they
    don't already exist (unless ``readonly`` is True, in which case the
    directories must pre-exist or reads will return empty/error results).

    Args:
        deny_globs: Glob patterns (forward-slash, ``fnmatch`` syntax)
            matched against the relative path within the ``files/`` sandbox.
            Any match denies read/write/list/stat/delete/make_dir on that
            path. Empty by default (no denial). Glob syntax is portable
            (``*.db``, ``threads/**``, ``repl_history.txt``).
    """
    global _backend
    _backend = StorageBackend(data_dir, readonly=readonly, deny_globs=deny_globs)


def _backend_instance() -> StorageBackend:
    """Return the active backend, raising if uninitialized."""
    if _backend is None:
        raise RuntimeError("lisa-storage is not initialized. Call init_storage(data_dir).")
    backend: StorageBackend = _backend
    return backend


# ── File I/O tools ──────────────────────────────────────────────────────────


@mcp.tool()
async def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read a text file from the files sandbox.

    Args:
        path: Relative path within the files sandbox (no ``..`` or absolute).
        encoding: Text encoding to use (default ``utf-8``).

    Returns:
        The file content as a string, or a JSON error object if the path
        is unsafe or the file doesn't exist.
    """
    return _backend_instance().read_file(path, encoding=encoding)


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """Write text content to a file in the files sandbox.

    Creates parent directories as needed. Respects ``--readonly``.

    Returns:
        JSON object with ``ok``, ``path``, and ``bytes``.
    """
    return _backend_instance().write_file(path, content)


@mcp.tool()
async def list_files(prefix: str = "") -> str:
    """List files under a prefix in the files sandbox.

    Returns a JSON array of relative paths (using forward slashes).
    """
    return _backend_instance().list_files(prefix)


@mcp.tool()
async def delete_file(path: str) -> str:
    """Delete a file from the files sandbox.

    Fails on directories (use ``make_dir`` is not reversible here —
    delete is files-only). Respects ``--readonly``.
    """
    return _backend_instance().delete_file(path)


@mcp.tool()
async def stat_file(path: str) -> str:
    """Return file metadata (size, modified timestamp, is_dir) for a path."""
    return _backend_instance().stat_file(path)


@mcp.tool()
async def make_dir(path: str) -> str:
    """Create a directory (and parents) in the files sandbox.

    Respects ``--readonly``.
    """
    return _backend_instance().make_dir(path)


# ── Checkpoint tools ────────────────────────────────────────────────────────


@mcp.tool()
async def save_checkpoint(bundle: dict[str, Any]) -> str:
    """Persist a checkpoint bundle as JSON under the checkpoints directory.

    The ``bundle`` is a serialized :class:`~kernel.models.CheckpointBundle`
    (or any JSON-serializable dict). If ``checkpoint_id`` is present in the
    bundle, it is used as the filename stem; otherwise a new ID is generated.

    Returns a JSON object with ``checkpoint_id``.
    """
    return _backend_instance().save_checkpoint(bundle)


@mcp.tool()
async def restore_checkpoint(checkpoint_id: str) -> str:
    """Load a checkpoint bundle by ID and return it as a JSON string.

    Returns the full bundle dict, or a JSON error object if not found.
    """
    return _backend_instance().restore_checkpoint(checkpoint_id)


@mcp.tool()
async def list_checkpoints(prefix: str = "") -> str:
    """List saved checkpoints as a JSON array of summary objects.

    Each entry contains ``checkpoint_id``, ``label``, ``created_at``,
    ``stage_id``, ``turn_count``, and ``mode_id`` (extracted from the
    stored JSON). Sorted by ``created_at`` descending (newest first).
    """
    return _backend_instance().list_checkpoints(prefix)


# ── Model manifest tools ────────────────────────────────────────────────────


@mcp.tool()
async def read_model_manifest(which: str = "proposal") -> str:
    """Read the active or proposed model manifest as text.

    Args:
        which: ``"active"`` for ``models.yaml`` or ``"proposal"`` for
            ``models.proposed.yaml`` (default ``"proposal"``).

    Returns:
        The manifest file content as text, or a JSON error object if the
        kind is invalid or the file does not exist.
    """
    return _backend_instance().read_model_manifest(which)


@mcp.tool()
async def write_model_manifest(content: str, which: str = "proposal") -> str:
    """Write text content to the active or proposed model manifest.

    Creates the parent directory if needed. Respects ``--readonly``.

    Args:
        content: The full manifest text (YAML) to write.
        which: ``"active"`` for ``models.yaml`` or ``"proposal"`` for
            ``models.proposed.yaml`` (default ``"proposal"``).

    Returns:
        JSON object with ``ok``, ``which``, and ``bytes``.
    """
    return _backend_instance().write_model_manifest(content, which)


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the lisa-storage MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``--data-dir`` is required; the files and checkpoints subdirectories
    are derived from it. ``--readonly`` disables all write operations.
    ``--deny-glob`` (repeatable) adds glob patterns to the file-sandbox
    denylist; ``LISA_STORAGE_DENY_GLOBS`` (comma-separated) is merged in
    too, so hosts can set policy via env without editing the launch YAML.
    """
    parser = build_parser("lisa-storage", supports_readonly=True)
    parser.add_argument(
        "--deny-glob",
        action="append",
        default=[],
        help="Glob pattern to deny in the files sandbox (repeatable). "
        "Matched against forward-slash relative paths via fnmatch.",
    )
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-storage")
    data_dir = resolve_data_dir(args.data_dir)

    env_globs = [
        g.strip() for g in os.getenv("LISA_STORAGE_DENY_GLOBS", "").split(",") if g.strip()
    ]
    deny_globs = (*args.deny_glob, *env_globs)
    log.info(
        "lisa-storage starting (data_dir=%s, readonly=%s, deny_globs=%s)",
        data_dir,
        args.readonly,
        deny_globs,
    )

    init_storage(data_dir, readonly=args.readonly, deny_globs=deny_globs)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
