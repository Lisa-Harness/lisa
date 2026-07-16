"""Lisa Storage MCP server — file I/O and checkpoint persistence.

This is the sole storage server for the framework. It owns two concerns:

1. **File I/O** (6 tools) — a sandboxed filesystem scoped to
   ``data_dir / "files"``. The model can read, write, list, delete, stat,
   and make directories within that root. Path traversal (``..`` segments,
   absolute paths) is rejected so a crafted path cannot escape the sandbox.

2. **Checkpoints** (3 tools) — save, restore, and list serialized
   :class:`~kernel.models.CheckpointBundle` objects as JSON files under
   ``data_dir / "checkpoints"``. The kernel builds the bundle from
   ``UnifiedState`` and ``WorkflowSessionState``; the host's
   ``on_checkpoint`` hook calls ``save_checkpoint`` through the MCP tool
   pool. ``--readonly`` disables writes, deletes, and checkpoint saves.

Storage is an MCP feature, not a kernel seam — there is no ``BlobIO``
Protocol in ``kernel/seams/protocols.py``. Hosts that need full blob
semantics (leases, content-type, range reads) wire their own MCP server;
the kernel does not care.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir
from sdk.content import is_safe_relative_path

__all__ = ["init_storage", "main", "mcp"]

logger = logging.getLogger(__name__)

mcp = FastMCP("Lisa Storage Server")

_data_dir: Path | None = None
_readonly: bool = False

_FILES_SUBDIR = "files"
_CHECKPOINTS_SUBDIR = "checkpoints"


# ── Initialization ──────────────────────────────────────────────────────────


def init_storage(data_dir: Path, *, readonly: bool = False) -> None:
    """Initialize the storage server with a resolved data directory.

    Called by ``main()`` after arg parsing. Tests may call this directly.
    Creates the ``files/`` and ``checkpoints/`` subdirectories if they
    don't already exist (unless ``readonly`` is True, in which case the
    directories must pre-exist or reads will return empty/error results).
    """
    global _data_dir, _readonly
    _data_dir = data_dir
    _readonly = readonly
    if not readonly:
        (data_dir / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)
        (data_dir / _CHECKPOINTS_SUBDIR).mkdir(parents=True, exist_ok=True)


def _files_root() -> Path:
    """Return the files sandbox root, raising if uninitialized."""
    if _data_dir is None:
        raise RuntimeError("lisa-storage is not initialized. Call init_storage(data_dir).")
    return _data_dir / _FILES_SUBDIR


def _checkpoints_root() -> Path:
    """Return the checkpoints root, raising if uninitialized."""
    if _data_dir is None:
        raise RuntimeError("lisa-storage is not initialized. Call init_storage(data_dir).")
    return _data_dir / _CHECKPOINTS_SUBDIR


def _manifest_path(which: str) -> Path:
    """Return the path to the active or proposed model manifest.

    Manifests are first-class framework artifacts (like checkpoints), so
    they live directly under ``data_dir`` — not under the ``files/`` sandbox.
    The ``which`` argument selects the active manifest (``"active"`` →
    ``models.yaml``) or the proposal (``"proposal"`` →
    ``models.proposed.yaml``).
    """

    if _data_dir is None:
        raise RuntimeError("lisa-storage is not initialized. Call init_storage(data_dir).")
    normalized = which.strip().lower()
    if normalized == "active":
        return _data_dir / "models.yaml"
    if normalized == "proposal":
        return _data_dir / "models.proposed.yaml"
    raise ValueError(f"manifest kind must be 'active' or 'proposal', got {which!r}")


def _resolve_file_path(relative_path: str) -> Path | None:
    """Resolve a user-supplied relative path against the files root.

    Returns the absolute ``Path`` if safe, or ``None`` if the path is
    rejected (empty, absolute, contains ``..``, or escapes the root after
    resolution).
    """
    raw = relative_path.strip()
    if not raw or Path(raw).is_absolute():
        return None
    normalized = raw.strip("/\\")
    if not is_safe_relative_path(normalized):
        return None
    root = _files_root()
    resolved = (root / normalized).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _checkpoint_id() -> str:
    return uuid.uuid4().hex[:16]


def _checkpoint_path(checkpoint_id: str) -> Path:
    return _checkpoints_root() / f"{checkpoint_id}.json"


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
    resolved = _resolve_file_path(path)
    if resolved is None:
        return json.dumps({"error": "path rejected: unsafe or escapes sandbox"})
    if not resolved.is_file():
        return json.dumps({"error": f"file not found: {path}"})
    try:
        return resolved.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return json.dumps({"error": "file is not valid text for the requested encoding"})
    except OSError as exc:
        return json.dumps({"error": f"read failed: {exc}"})


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """Write text content to a file in the files sandbox.

    Creates parent directories as needed. Respects ``--readonly``.

    Returns:
        JSON object with ``ok``, ``path``, and ``bytes``.
    """
    if _readonly:
        return json.dumps({"error": "server is read-only"})
    resolved = _resolve_file_path(path)
    if resolved is None:
        return json.dumps({"error": "path rejected: unsafe or escapes sandbox"})
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": path, "bytes": len(content.encode("utf-8"))})
    except OSError as exc:
        return json.dumps({"error": f"write failed: {exc}"})


@mcp.tool()
async def list_files(prefix: str = "") -> str:
    """List files under a prefix in the files sandbox.

    Returns a JSON array of relative paths (using forward slashes).
    """
    root = _files_root()
    if prefix.strip():
        normalized = prefix.strip().strip("/\\")
        if not is_safe_relative_path(normalized):
            return json.dumps({"error": "prefix rejected: unsafe or escapes sandbox"})
        search_root = (root / normalized).resolve()
        try:
            search_root.relative_to(root.resolve())
        except ValueError:
            return json.dumps({"error": "prefix escapes sandbox"})
    else:
        search_root = root

    if not search_root.is_dir():
        return json.dumps([])

    files: list[str] = []
    for item in sorted(search_root.rglob("*")):
        if item.is_file():
            rel = item.relative_to(root)
            files.append(str(rel).replace("\\", "/"))
    return json.dumps(files)


@mcp.tool()
async def delete_file(path: str) -> str:
    """Delete a file from the files sandbox.

    Fails on directories (use ``make_dir`` is not reversible here —
    delete is files-only). Respects ``--readonly``.
    """
    if _readonly:
        return json.dumps({"error": "server is read-only"})
    resolved = _resolve_file_path(path)
    if resolved is None:
        return json.dumps({"error": "path rejected: unsafe or escapes sandbox"})
    if not resolved.exists():
        return json.dumps({"error": f"file not found: {path}"})
    if resolved.is_dir():
        return json.dumps({"error": "delete_file targets a directory; use a file path"})
    try:
        resolved.unlink()
        return json.dumps({"ok": True, "path": path})
    except OSError as exc:
        return json.dumps({"error": f"delete failed: {exc}"})


@mcp.tool()
async def stat_file(path: str) -> str:
    """Return file metadata (size, modified timestamp, is_dir) for a path."""
    resolved = _resolve_file_path(path)
    if resolved is None:
        return json.dumps({"error": "path rejected: unsafe or escapes sandbox"})
    if not resolved.exists():
        return json.dumps({"error": f"not found: {path}"})
    stat = resolved.stat()
    return json.dumps({
        "path": path,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "is_dir": resolved.is_dir(),
    })


@mcp.tool()
async def make_dir(path: str) -> str:
    """Create a directory (and parents) in the files sandbox.

    Respects ``--readonly``.
    """
    if _readonly:
        return json.dumps({"error": "server is read-only"})
    resolved = _resolve_file_path(path)
    if resolved is None:
        return json.dumps({"error": "path rejected: unsafe or escapes sandbox"})
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        return json.dumps({"ok": True, "path": path})
    except OSError as exc:
        return json.dumps({"error": f"make_dir failed: {exc}"})


# ── Checkpoint tools ────────────────────────────────────────────────────────


@mcp.tool()
async def save_checkpoint(bundle: dict[str, Any]) -> str:
    """Persist a checkpoint bundle as JSON under the checkpoints directory.

    The ``bundle`` is a serialized :class:`~kernel.models.CheckpointBundle`
    (or any JSON-serializable dict). If ``checkpoint_id`` is present in the
    bundle, it is used as the filename stem; otherwise a new ID is generated.

    Returns a JSON object with ``checkpoint_id``.
    """
    if _readonly:
        return json.dumps({"error": "server is read-only"})
    checkpoints_root = _checkpoints_root()
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    checkpoint_id = str(bundle.get("checkpoint_id") or "").strip() or _checkpoint_id()
    bundle["checkpoint_id"] = checkpoint_id
    if "created_at" not in bundle:
        bundle["created_at"] = _now_iso()

    path = _checkpoint_path(checkpoint_id)
    try:
        path.write_text(json.dumps(bundle, default=str, indent=2), encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"save_checkpoint failed: {exc}"})

    return json.dumps({"checkpoint_id": checkpoint_id})


@mcp.tool()
async def restore_checkpoint(checkpoint_id: str) -> str:
    """Load a checkpoint bundle by ID and return it as a JSON string.

    Returns the full bundle dict, or a JSON error object if not found.
    """
    cid = checkpoint_id.strip()
    if not cid or not is_safe_relative_path(f"{cid}.json"):
        return json.dumps({"error": "invalid checkpoint_id"})
    path = _checkpoint_path(cid)
    if not path.is_file():
        return json.dumps({"error": f"checkpoint not found: {checkpoint_id}"})
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"restore_checkpoint failed: {exc}"})


@mcp.tool()
async def list_checkpoints(prefix: str = "") -> str:
    """List saved checkpoints as a JSON array of summary objects.

    Each entry contains ``checkpoint_id``, ``label``, ``created_at``,
    ``stage_id``, ``turn_count``, and ``mode_id`` (extracted from the
    stored JSON). Sorted by ``created_at`` descending (newest first).
    """
    root = _checkpoints_root()
    if not root.is_dir():
        return json.dumps([])

    summaries: list[dict[str, Any]] = []
    for item in sorted(root.glob("*.json")):
        try:
            data = json.loads(item.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cid = str(data.get("checkpoint_id") or item.stem)
        label = str(data.get("label") or "")
        if prefix and prefix not in cid and prefix not in label:
            continue
        summaries.append({
            "checkpoint_id": cid,
            "label": label,
            "created_at": str(data.get("created_at") or ""),
            "stage_id": str(data.get("stage_id") or ""),
            "turn_count": int(data.get("turn_count") or 0),
            "mode_id": str(data.get("mode_id") or ""),
        })

    summaries.sort(key=lambda s: s["created_at"], reverse=True)
    return json.dumps(summaries, default=str)


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

    try:
        path = _manifest_path(which)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    if not path.is_file():
        return json.dumps({"error": f"manifest not found: {which} ({path.name})"})
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"read_model_manifest failed: {exc}"})


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

    if _readonly:
        return json.dumps({"error": "server is read-only"})
    try:
        path = _manifest_path(which)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "which": which, "bytes": len(content.encode("utf-8"))})
    except OSError as exc:
        return json.dumps({"error": f"write_model_manifest failed: {exc}"})


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the lisa-storage MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``--data-dir`` is required; the files and checkpoints subdirectories
    are derived from it. ``--readonly`` disables all write operations.
    """
    parser = build_parser("lisa-storage", supports_readonly=True)
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-storage")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-storage starting (data_dir=%s, readonly=%s)", data_dir, args.readonly)

    init_storage(data_dir, readonly=args.readonly)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()