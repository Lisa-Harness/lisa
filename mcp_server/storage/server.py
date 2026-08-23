"""Lisa Storage MCP server — thin facade over :class:`sdk.storage.StorageBackend`.

This is the sole storage server for the framework. It is a transport-only
wrapper: every tool delegates to the SDK backend, which owns the sandbox
resolution, deny-glob policy, and model-manifest read/write. Hosts or
tools that want in-process storage access can use
:class:`sdk.storage.StorageBackend` directly without spawning this server.

Tools (10 total):

- File I/O (8): ``read_file``, ``read_file_range``, ``write_file``,
  ``append_file``, ``list_files``, ``delete_file``, ``stat_file``,
  ``make_dir`` — scoped to ``data_dir / "files"``. ``read_file_range``
  returns a char-bounded slice (never splits multibyte codepoints);
  ``append_file`` creates parent dirs and the file if absent.
- Model manifests (2): ``read_model_manifest``, ``write_model_manifest`` —
  ``models.yaml`` / ``models.proposed.yaml`` directly under ``data_dir``.

``--readonly`` disables writes, deletes, and manifest writes.
``--deny-glob`` (repeatable) + ``LISA_STORAGE_DENY_GLOBS`` env add
glob patterns to the file-sandbox denylist.

Error contract: the SDK backend returns ``json.dumps({"error": ...})``
strings on failure (its contract is locked by ``sdk/storage/tests``). This
server seam parses those returns and raises :class:`ToolError` so the MCP
client sees a structured ``CallToolResult(is_error=True)`` instead of a
success-shaped JSON error body. Success results pass through unchanged.

Storage is an MCP feature, not a kernel seam — there is no ``BlobIO``
Protocol in ``kernel/seams/protocols.py``. Hosts that need full blob
semantics (leases, content-type, range reads) wire their own MCP server;
the kernel does not care.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from sdk.storage import StorageBackend

__all__ = ["StorageDeps", "init_storage", "main", "mcp", "set_deps_for_test"]

logger = logging.getLogger(__name__)


# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class StorageDeps:
    """Server-scoped dependencies owned by the lifespan.

    Constructed from the resolved ``--data-dir`` / ``--readonly`` /
    ``--deny-glob`` CLI args. The backend is stateless beyond paths, so
    no teardown is needed.
    """

    backend: StorageBackend


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: StorageDeps | None = None


def set_deps_for_test(deps: StorageDeps) -> None:
    """Test-only seam: inject ``StorageDeps`` without spawning stdio.

    The lifespan owns this in production; tests bypass it because the
    SDK's direct ``call_tool`` path does not run the lifespan.
    """
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> StorageDeps:
    """Return the active deps, preferring lifespan context when available.

    The lifespan context is populated during a real stdio session; the
    test seam is the fallback for direct ``call_tool`` invocations.
    """
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, StorageDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        raise ToolError("lisa-storage is not initialized. Call set_deps_for_test(deps).")
    return _deps


# ── CLI config holder (populated by _init before run()) ────────────────────


@dataclass
class _StorageConfig:
    data_dir: Path = Path()
    readonly: bool = False
    deny_globs: tuple[str, ...] = ()


_cfg = _StorageConfig()


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[StorageDeps]:
    """Construct the storage backend from resolved CLI config and yield it."""
    deps = StorageDeps(
        backend=StorageBackend(
            _cfg.data_dir,
            readonly=_cfg.readonly,
            deny_globs=_cfg.deny_globs,
        )
    )
    yield deps


mcp = MCPServer("Lisa Storage Server", lifespan=_lifespan)


def _raise_on_error(raw: str) -> str:
    """Convert a backend error-string into a ``ToolError``.

    The SDK backend returns ``json.dumps({"error": ...})`` on failure and
    a success JSON string otherwise. Parse and raise on the ``error`` key
    so the MCP client sees ``CallToolResult(is_error=True)``. Success
    results pass through unchanged.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(parsed, dict) and "error" in parsed:
        raise ToolError(str(parsed["error"]))
    return raw


def init_storage(
    data_dir: Path,
    *,
    readonly: bool = False,
    deny_globs: tuple[str, ...] = (),
) -> None:
    """Initialize the storage server with a resolved data directory.

    Kept as a thin back-compat shim for tests that call ``init_storage``
    directly. Tests should prefer :func:`set_deps_for_test` with a
    :class:`StorageDeps` instance.
    """
    set_deps_for_test(
        StorageDeps(
            backend=StorageBackend(data_dir, readonly=readonly, deny_globs=deny_globs),
        )
    )


# ── File I/O tools ──────────────────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def read_file(
    path: str,
    encoding: str = "utf-8",
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Read a text file from the files sandbox.

    Args:
        path: Relative path within the files sandbox (no ``..`` or absolute).
        encoding: Text encoding to use (default ``utf-8``).

    Returns:
        The file content as a string. Raises :class:`ToolError` if the path
        is unsafe or the file doesn't exist.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.read_file(path, encoding=encoding))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def read_file_range(
    path: str,
    start: int = 0,
    length: int = 4096,
    encoding: str = "utf-8",
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Read a char-bounded slice of a text file from the files sandbox.

    Useful for chunked ingestion of large files without loading the whole
    file into a single model context. Slicing is char-based so multibyte
    codepoints are never split.

    Args:
        path: Relative path within the files sandbox (no ``..`` or absolute).
        start: Char offset to begin reading (0-based). Negative coerced to 0.
        length: Max chars to return.
        encoding: Text encoding (default ``utf-8``).

    Returns:
        JSON object ``{"content", "start", "length", "total", "eof"}``.
        Raises :class:`ToolError` if the path is unsafe or the file is missing.
    """
    return _raise_on_error(
        _resolve_deps(ctx).backend.read_file_range(
            path, start=start, length=length, encoding=encoding
        )
    )


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def write_file(
    path: str,
    content: str,
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Write text content to a file in the files sandbox.

    Creates parent directories as needed. Respects ``--readonly``.

    Returns:
        JSON object with ``ok``, ``path``, and ``bytes``. Raises
        :class:`ToolError` on readonly or write failure.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.write_file(path, content))


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def append_file(
    path: str,
    content: str,
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Append text content to a file in the files sandbox.

    Creates parent directories and the file if absent. Respects
    ``--readonly``. Useful for incremental/segmented document persistence
    where read-modify-write of the whole file is wasteful.

    Returns:
        JSON object with ``ok``, ``path``, and ``bytes`` (the new total
        file size). Raises :class:`ToolError` on readonly or append failure.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.append_file(path, content))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_files(
    prefix: str = "",
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """List files under a prefix in the files sandbox.

    Returns a JSON array of relative paths (using forward slashes). Raises
    :class:`ToolError` if the prefix is unsafe or denied by policy.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.list_files(prefix))


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def delete_file(
    path: str,
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Delete a file from the files sandbox.

    Fails on directories (delete is files-only). Respects ``--readonly``.
    Raises :class:`ToolError` on readonly, not-found, or delete failure.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.delete_file(path))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def stat_file(
    path: str,
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Return file metadata (size, modified timestamp, is_dir) for a path.

    Raises :class:`ToolError` if the path is unsafe or not found.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.stat_file(path))


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def make_dir(
    path: str,
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Create a directory (and parents) in the files sandbox.

    Respects ``--readonly``. Raises :class:`ToolError` on readonly or
    make_dir failure.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.make_dir(path))


# ── Model manifest tools ────────────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def read_model_manifest(
    which: str = "proposal",
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Read the active or proposed model manifest as text.

    Args:
        which: ``"active"`` for ``models.yaml`` or ``"proposal"`` for
            ``models.proposed.yaml`` (default ``"proposal"``).

    Returns:
        The manifest file content as text. Raises :class:`ToolError` if the
        kind is invalid or the file does not exist.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.read_model_manifest(which))


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def write_model_manifest(
    content: str,
    which: str = "proposal",
    ctx: Context[StorageDeps, Any] | None = None,
) -> str:
    """Write text content to the active or proposed model manifest.

    Creates the parent directory if needed. Respects ``--readonly``.

    Args:
        content: The full manifest text (YAML) to write.
        which: ``"active"`` for ``models.yaml`` or ``"proposal"`` for
            ``models.proposed.yaml`` (default ``"proposal"``).

    Returns:
        JSON object with ``ok``, ``which``, and ``bytes``. Raises
        :class:`ToolError` on readonly or write failure.
    """
    return _raise_on_error(_resolve_deps(ctx).backend.write_model_manifest(content, which))


# ── Entry point ─────────────────────────────────────────────────────────────


def _add_storage_args(parser: Any) -> None:
    """Add ``--deny-glob`` (repeatable) to the standard parser."""
    parser.add_argument(
        "--deny-glob",
        action="append",
        default=[],
        help="Glob pattern to deny in the files sandbox (repeatable). "
        "Matched against forward-slash relative paths via fnmatch.",
    )


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can build the backend."""
    env_globs = [
        g.strip() for g in os.getenv("LISA_STORAGE_DENY_GLOBS", "").split(",") if g.strip()
    ]
    _cfg.data_dir = Path(args.data_dir)
    _cfg.readonly = bool(getattr(args, "readonly", False))
    _cfg.deny_globs = (*args.deny_glob, *env_globs)
    # Also populate the test seam so the server is usable immediately after
    # init (the lifespan only runs during a real stdio session).
    set_deps_for_test(
        StorageDeps(
            backend=StorageBackend(
                _cfg.data_dir,
                readonly=_cfg.readonly,
                deny_globs=_cfg.deny_globs,
            )
        )
    )


def main() -> None:
    """Entry point for the lisa-storage MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper.
    ``--data-dir`` is required; the files subdirectory is derived from it.
    ``--readonly`` disables all write operations.
    ``--deny-glob`` (repeatable) adds glob patterns to the file-sandbox
    denylist; ``LISA_STORAGE_DENY_GLOBS`` (comma-separated) is merged in
    too, so hosts can set policy via env without editing the launch YAML.
    """
    from mcp_server.config.cli import run

    run(
        "lisa-storage",
        mcp,
        _init,
        supports_readonly=True,
        add_args=_add_storage_args,
    )


if __name__ == "__main__":
    main()
