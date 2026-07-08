"""MCP-backed ``BlobIO`` adapter.

Implements the :class:`~kernel.seams.protocols.BlobIO` protocol by calling
the 23 tools exposed by a ``lisa-storage`` MCP server. Each method opens a
per-call stdio session (open → call → close), matching the
``MCPMemoryClient`` pattern.

Bytes use base64 encoding over the wire. Structured types
(``UnifiedState``, ``SurfSessionState``) serialize as JSON strings and are
deserialized via ``model_validate`` on the adapter side.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from kernel.models import UnifiedState
from kernel.tools.mcp_client import MCPClientConfig

__all__ = ["MCPBlobIO"]


def _extract_text(result: Any) -> str:
    """Extract text from an MCP call_tool result."""
    if result.content and len(result.content) > 0:
        return result.content[0].text  # type: ignore[union-attr]
    return ""


def _parse_ok(text: str) -> bool:
    """Parse a ``{"ok": bool}`` response."""
    if not text:
        return False
    try:
        return bool(json.loads(text).get("ok", False))
    except (json.JSONDecodeError, TypeError):
        return False


def _parse_json_or_none(text: str) -> dict[str, Any] | None:
    """Parse a JSON object response, returning None for ``"null"`` or empty."""
    if not text or text == "null":
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class MCPBlobIO:
    """BlobIO backed by an MCP storage server.

    Each method calls the corresponding MCP tool on a ``lisa-storage`` server
    via a per-call stdio session.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )

    async def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        async with stdio_client(self._params) as (read, write_stream):
            async with ClientSession(read, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments=arguments)
                return _extract_text(result)

    # ── Unified state ───────────────────────────────────────────────────────

    def read_unified_state(self) -> UnifiedState | None:
        """Read the unified state. Note: MCP call is async; this sync wrapper
        runs the event loop. For async hosts, call the underlying session
        directly via ``_call``.
        """
        import asyncio

        text = asyncio.run(self._call("read_unified_state", {}))
        if not text or text == "null":
            return None
        return UnifiedState.model_validate_json(text)

    def write_unified_state(self, state: UnifiedState) -> bool:
        import asyncio

        text = asyncio.run(
            self._call("write_unified_state", {"state_json": state.model_dump_json()})
        )
        return _parse_ok(text)

    # ── Blob bytes / text ───────────────────────────────────────────────────

    def read_blob_bytes(self, container: str, blob_path: str) -> bytes | None:
        import asyncio

        text = asyncio.run(
            self._call("read_blob_bytes", {"container": container, "blob_path": blob_path})
        )
        if not text or text == "null":
            return None
        return base64.b64decode(text)

    def write_blob_bytes(
        self,
        container: str,
        blob_path: str,
        payload: bytes,
        *,
        content_type: str | None = None,
    ) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_blob_bytes",
                {
                    "container": container,
                    "blob_path": blob_path,
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "content_type": content_type or "application/octet-stream",
                },
            )
        )
        return _parse_ok(text)

    def read_blob_text(self, container: str, blob_path: str) -> str | None:
        import asyncio

        text = asyncio.run(
            self._call("read_blob_text", {"container": container, "blob_path": blob_path})
        )
        if not text or text == "null":
            return None
        return text

    def write_blob_text(
        self,
        container: str,
        blob_path: str,
        payload: str,
        *,
        content_type: str | None = None,
    ) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_blob_text",
                {
                    "container": container,
                    "blob_path": blob_path,
                    "payload": payload,
                    "content_type": content_type or "text/plain",
                },
            )
        )
        return _parse_ok(text)

    # ── Surf workflow sessions ─────────────────────────────────────────────

    def read_surf_workflow_session(self, workspace_id: str, surface_id: str) -> Any:
        import asyncio

        from lisa_sdk.models import SurfSessionState

        text = asyncio.run(
            self._call(
                "read_surf_workflow_session",
                {"workspace_id": workspace_id, "surface_id": surface_id},
            )
        )
        if not text or text == "null":
            return None
        return SurfSessionState.model_validate_json(text)

    def write_surf_workflow_session(self, session: Any) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_surf_workflow_session",
                {"session_json": session.model_dump_json()},
            )
        )
        return _parse_ok(text)

    def delete_surf_workflow_session(self, workspace_id: str, surface_id: str) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "delete_surf_workflow_session",
                {"workspace_id": workspace_id, "surface_id": surface_id},
            )
        )
        return _parse_ok(text)

    # ── Surf screenshots / traces ───────────────────────────────────────────

    def read_surf_screenshot(self, blob_path: str) -> bytes | None:
        import asyncio

        text = asyncio.run(self._call("read_surf_screenshot", {"blob_path": blob_path}))
        if not text or text == "null":
            return None
        return base64.b64decode(text)

    def write_surf_trace(
        self,
        trace_json: str,
        screenshots: dict[str, bytes],
        artifact_prefix: str,
    ) -> bool:
        import asyncio

        screenshots_json = json.dumps(
            {k: base64.b64encode(v).decode("ascii") for k, v in screenshots.items()}
        )
        text = asyncio.run(
            self._call(
                "write_surf_trace",
                {
                    "trace_json": trace_json,
                    "screenshots_json": screenshots_json,
                    "artifact_prefix": artifact_prefix,
                },
            )
        )
        return _parse_ok(text)

    # ── Overlays ────────────────────────────────────────────────────────────

    def read_user_context_overlays(self) -> dict[str, Any] | None:
        import asyncio

        return _parse_json_or_none(asyncio.run(self._call("read_user_context_overlays", {})))

    def write_user_context_overlays(self, overlays: dict[str, Any]) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_user_context_overlays",
                {"overlays_json": json.dumps(overlays)},
            )
        )
        return _parse_ok(text)

    def read_surface_context_overlays(self) -> dict[str, Any] | None:
        import asyncio

        return _parse_json_or_none(asyncio.run(self._call("read_surface_context_overlays", {})))

    def write_surface_context_overlays(self, overlays: dict[str, Any]) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_surface_context_overlays",
                {"overlays_json": json.dumps(overlays)},
            )
        )
        return _parse_ok(text)

    # ── Proactive recipients / reminders ─────────────────────────────────────

    def read_proactive_recipients(self) -> dict[str, Any] | None:
        import asyncio

        return _parse_json_or_none(asyncio.run(self._call("read_proactive_recipients", {})))

    def write_proactive_recipients(self, registry: dict[str, Any]) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_proactive_recipients",
                {"registry_json": json.dumps(registry)},
            )
        )
        return _parse_ok(text)

    def read_user_reminders(self) -> dict[str, Any] | None:
        import asyncio

        return _parse_json_or_none(asyncio.run(self._call("read_user_reminders", {})))

    def write_user_reminders(self, registry: dict[str, Any]) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_user_reminders",
                {"registry_json": json.dumps(registry)},
            )
        )
        return _parse_ok(text)

    # ── Thread state / memory snapshot ───────────────────────────────────────

    def read_thread_state(self) -> dict[str, Any] | None:
        import asyncio

        return _parse_json_or_none(asyncio.run(self._call("read_thread_state", {})))

    def write_thread_state(self, state: dict[str, Any]) -> bool:
        import asyncio

        text = asyncio.run(self._call("write_thread_state", {"state_json": json.dumps(state)}))
        return _parse_ok(text)

    def read_memory_snapshot(self) -> dict[str, Any] | None:
        import asyncio

        return _parse_json_or_none(asyncio.run(self._call("read_memory_snapshot", {})))

    def write_memory_snapshot(self, snapshot: dict[str, Any]) -> bool:
        import asyncio

        text = asyncio.run(
            self._call(
                "write_memory_snapshot",
                {"snapshot_json": json.dumps(snapshot)},
            )
        )
        return _parse_ok(text)
