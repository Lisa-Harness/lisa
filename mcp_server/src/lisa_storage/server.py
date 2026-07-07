"""Lisa Storage MCP server — filesystem-backed blob I/O.

Exposes 22 tools, one per ``BlobIO`` protocol method. The filesystem root is
``{data_dir}/storage/`` with container subdirectories. Path-safe resolution
prevents ``..`` traversal. Bytes use base64 encoding over the wire;
structured types (``UnifiedState``, ``SurfSessionState``, overlays, etc.)
serialize as JSON strings.

``--readonly`` disables all write tools (they return
``{"ok": false, "error": "readonly"}``).
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from kernel.models import SurfSessionState, UnifiedState

logger = logging.getLogger(__name__)

__all__ = ["mcp", "main", "init_storage"]

mcp = FastMCP("Lisa Storage Server")

_root: Path | None = None
_readonly: bool = False


def init_storage(data_dir: Path, *, readonly: bool = False) -> None:
    """Initialize the filesystem storage root. Called by ``main()``."""
    global _root, _readonly
    _root = data_dir / "storage"
    _root.mkdir(parents=True, exist_ok=True)
    _readonly = readonly


def _storage_root() -> Path:
    if _root is None:
        raise RuntimeError(
            "lisa_storage root is not initialized. "
            "Call init_storage(data_dir) from main() before serving tools."
        )
    return _root


def _readonly_error() -> str:
    return json.dumps({"ok": False, "error": "readonly"})


def _resolve_container_path(container: str, blob_path: str = "") -> Path:
    """Resolve a container-relative path, preventing ``..`` traversal."""
    root = _storage_root()
    safe_container = Path(container).as_posix().lstrip("/").split("/", 1)[0]
    if not safe_container or safe_container in {".", ".."}:
        raise ValueError(f"Invalid container: {container!r}")
    base = root / safe_container
    if blob_path:
        # Normalize and reject any component that escapes the container
        parts = Path(blob_path).parts
        cleaned: list[str] = []
        for part in parts:
            if part == "..":
                raise ValueError(f"Path traversal denied: {blob_path!r}")
            if part in {".", ""}:
                continue
            cleaned.append(part)
        resolved = base.joinpath(*cleaned) if cleaned else base
    else:
        resolved = base
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


# ── Unified state ────────────────────────────────────────────────────────────


@mcp.tool()
async def read_unified_state() -> str:
    """Read the unified state JSON, or ``null`` if absent."""
    target = _resolve_container_path("_state", "unified_state.json")
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_unified_state(state_json: str) -> str:
    """Write the unified state from a JSON string."""
    if _readonly:
        return _readonly_error()
    state = UnifiedState.model_validate_json(state_json)
    target = _resolve_container_path("_state", "unified_state.json")
    target.write_text(state.model_dump_json(), encoding="utf-8")
    return json.dumps({"ok": True})


# ── Blob bytes / text ────────────────────────────────────────────────────────


@mcp.tool()
async def read_blob_bytes(container: str, blob_path: str) -> str:
    """Read raw bytes as base64, or ``null`` if absent."""
    target = _resolve_container_path(container, blob_path)
    if not target.is_file():
        return "null"
    raw = target.read_bytes()
    return base64.b64encode(raw).decode("ascii")


@mcp.tool()
async def write_blob_bytes(
    container: str,
    blob_path: str,
    payload_b64: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Write raw bytes from a base64 payload."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path(container, blob_path)
    target.write_bytes(base64.b64decode(payload_b64))
    return json.dumps({"ok": True})


@mcp.tool()
async def read_blob_text(container: str, blob_path: str) -> str:
    """Read text, or ``null`` if absent."""
    target = _resolve_container_path(container, blob_path)
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_blob_text(
    container: str,
    blob_path: str,
    payload: str,
    content_type: str = "text/plain",
) -> str:
    """Write text."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path(container, blob_path)
    target.write_text(payload, encoding="utf-8")
    return json.dumps({"ok": True})


# ── Surf workflow sessions ──────────────────────────────────────────────────


@mcp.tool()
async def read_surf_workflow_session(workspace_id: str, surface_id: str) -> str:
    """Read a surf session state JSON, or ``null`` if absent."""
    target = _resolve_container_path(
        "surf_sessions", f"{workspace_id}/{surface_id}.json"
    )
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_surf_workflow_session(session_json: str) -> str:
    """Write a surf session state from a JSON string."""
    if _readonly:
        return _readonly_error()
    session = SurfSessionState.model_validate_json(session_json)
    target = _resolve_container_path(
        "surf_sessions", f"{session.workspace_id}/{session.surface_id}.json"
    )
    target.write_text(session.model_dump_json(), encoding="utf-8")
    return json.dumps({"ok": True})


@mcp.tool()
async def delete_surf_workflow_session(workspace_id: str, surface_id: str) -> str:
    """Delete a surf session. Returns ``{"ok": true}`` even if absent."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path(
        "surf_sessions", f"{workspace_id}/{surface_id}.json"
    )
    if target.is_file():
        target.unlink()
    return json.dumps({"ok": True})


# ── Surf screenshots / traces ───────────────────────────────────────────────


@mcp.tool()
async def read_surf_screenshot(blob_path: str) -> str:
    """Read a screenshot as base64, or ``null`` if absent."""
    target = _resolve_container_path("surf_screenshots", blob_path)
    if not target.is_file():
        return "null"
    return base64.b64encode(target.read_bytes()).decode("ascii")


@mcp.tool()
async def write_surf_trace(
    trace_json: str,
    screenshots_json: str,
    artifact_prefix: str,
) -> str:
    """Write a surf trace and its screenshots.

    ``screenshots_json`` is a JSON object mapping screenshot name to base64
    bytes. Screenshots are written under ``surf_screenshots/{artifact_prefix}/``.
    """
    if _readonly:
        return _readonly_error()
    trace_target = _resolve_container_path("surf_traces", f"{artifact_prefix}.json")
    trace_target.write_text(trace_json, encoding="utf-8")
    screenshots = json.loads(screenshots_json)
    for name, b64 in screenshots.items():
        shot_target = _resolve_container_path(
            "surf_screenshots", f"{artifact_prefix}/{name}"
        )
        shot_target.write_bytes(base64.b64decode(b64))
    return json.dumps({"ok": True})


# ── Overlays ────────────────────────────────────────────────────────────────


@mcp.tool()
async def read_user_context_overlays() -> str:
    """Read user context overlays JSON, or ``null`` if absent."""
    target = _resolve_container_path("_state", "user_context_overlays.json")
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_user_context_overlays(overlays_json: str) -> str:
    """Write user context overlays from a JSON string."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path("_state", "user_context_overlays.json")
    target.write_text(overlays_json, encoding="utf-8")
    return json.dumps({"ok": True})


@mcp.tool()
async def read_surface_context_overlays() -> str:
    """Read surface context overlays JSON, or ``null`` if absent."""
    target = _resolve_container_path("_state", "surface_context_overlays.json")
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_surface_context_overlays(overlays_json: str) -> str:
    """Write surface context overlays from a JSON string."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path("_state", "surface_context_overlays.json")
    target.write_text(overlays_json, encoding="utf-8")
    return json.dumps({"ok": True})


# ── Proactive recipients / reminders ─────────────────────────────────────────


@mcp.tool()
async def read_proactive_recipients() -> str:
    """Read proactive recipients registry JSON, or ``null`` if absent."""
    target = _resolve_container_path("_state", "proactive_recipients.json")
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_proactive_recipients(registry_json: str) -> str:
    """Write proactive recipients registry from a JSON string."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path("_state", "proactive_recipients.json")
    target.write_text(registry_json, encoding="utf-8")
    return json.dumps({"ok": True})


@mcp.tool()
async def read_user_reminders() -> str:
    """Read user reminders registry JSON, or ``null`` if absent."""
    target = _resolve_container_path("_state", "user_reminders.json")
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_user_reminders(registry_json: str) -> str:
    """Write user reminders registry from a JSON string."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path("_state", "user_reminders.json")
    target.write_text(registry_json, encoding="utf-8")
    return json.dumps({"ok": True})


# ── Thread state / memory snapshot ───────────────────────────────────────────


@mcp.tool()
async def read_thread_state() -> str:
    """Read thread state JSON, or ``null`` if absent."""
    target = _resolve_container_path("_state", "thread_state.json")
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_thread_state(state_json: str) -> str:
    """Write thread state from a JSON string."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path("_state", "thread_state.json")
    target.write_text(state_json, encoding="utf-8")
    return json.dumps({"ok": True})


@mcp.tool()
async def read_memory_snapshot() -> str:
    """Read memory snapshot JSON, or ``null`` if absent."""
    target = _resolve_container_path("_state", "memory_snapshot.json")
    if not target.is_file():
        return "null"
    return target.read_text(encoding="utf-8")


@mcp.tool()
async def write_memory_snapshot(snapshot_json: str) -> str:
    """Write memory snapshot from a JSON string."""
    if _readonly:
        return _readonly_error()
    target = _resolve_container_path("_state", "memory_snapshot.json")
    target.write_text(snapshot_json, encoding="utf-8")
    return json.dumps({"ok": True})


def main() -> None:
    """Entry point for the lisa-storage MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``--readonly`` disables all write tools.
    """
    from lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-storage", supports_readonly=True)
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-storage")
    data_dir = resolve_data_dir(args.data_dir)
    log.info(
        "lisa-storage starting (data_dir=%s, readonly=%s)",
        data_dir,
        args.readonly,
    )

    init_storage(data_dir, readonly=args.readonly)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()