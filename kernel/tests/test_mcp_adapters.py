"""Tests for the kernel MCP-client adapters.

Covers:
- Protocol compliance (MCPInputGuard, MCPOutputGuard, MCPBlobIO,
  MCPContentRegistry satisfy their respective Protocols)
- Response parsing helpers (_parse_guard_response, _parse_ok,
  _parse_json_or_none)
- Integration: MCPInputGuard.validate via a spawned lisa-guard server
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel.context import MCPBlobIO, MCPContentRegistry, MCPInputGuard, MCPOutputGuard
from kernel.models import SurfaceContext
from kernel.seams.protocols import BlobIO, ContentRegistry, GuardResult, InputGuard, OutputGuard
from kernel.tools.mcp_client import MCPClientConfig


def _make_config(
    server_id: str = "test", command: str = "python", args: list[str] | None = None
) -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command=command,
        args=args or ["-m", "lisa_guard.server"],
    )


# ── Protocol compliance ─────────────────────────────────────────────────────


def test_mcp_input_guard_satisfies_protocol() -> None:
    cfg = _make_config()
    guard = MCPInputGuard(cfg)
    assert isinstance(guard, InputGuard)


def test_mcp_output_guard_satisfies_protocol() -> None:
    cfg = _make_config()
    guard = MCPOutputGuard(cfg)
    assert isinstance(guard, OutputGuard)


def test_mcp_blob_io_satisfies_protocol() -> None:
    cfg = _make_config()
    blob = MCPBlobIO(cfg)
    assert isinstance(blob, BlobIO)


def test_mcp_content_registry_satisfies_protocol() -> None:
    cfg = _make_config()
    registry = MCPContentRegistry(cfg)
    assert isinstance(registry, ContentRegistry)


# ── Response parsing ────────────────────────────────────────────────────────


def test_parse_guard_response_allowed() -> None:
    from kernel.context.mcp_guard import _parse_guard_response

    text = json.dumps(
        {"allowed": True, "blocked_reason": None, "sanitized_text": None, "annotations": {}}
    )
    result = _parse_guard_response(text)
    assert isinstance(result, GuardResult)
    assert result.allowed is True
    assert result.blocked_reason is None


def test_parse_guard_response_blocked() -> None:
    from kernel.context.mcp_guard import _parse_guard_response

    text = json.dumps(
        {
            "allowed": False,
            "blocked_reason": "toxicity",
            "sanitized_text": "***",
            "annotations": {"score": 0.9},
        }
    )
    result = _parse_guard_response(text)
    assert result.allowed is False
    assert result.blocked_reason == "toxicity"
    assert result.sanitized_text == "***"
    assert result.annotations == {"score": 0.9}


def test_parse_guard_response_empty_defaults_to_allowed() -> None:
    from kernel.context.mcp_guard import _parse_guard_response

    result = _parse_guard_response("")
    assert result.allowed is True


def test_parse_guard_response_invalid_json_defaults_to_allowed() -> None:
    from kernel.context.mcp_guard import _parse_guard_response

    result = _parse_guard_response("not json")
    assert result.allowed is True


def test_parse_ok_true() -> None:
    from kernel.context.mcp_storage import _parse_ok

    assert _parse_ok(json.dumps({"ok": True})) is True


def test_parse_ok_false() -> None:
    from kernel.context.mcp_storage import _parse_ok

    assert _parse_ok(json.dumps({"ok": False})) is False


def test_parse_ok_empty() -> None:
    from kernel.context.mcp_storage import _parse_ok

    assert _parse_ok("") is False


def test_parse_json_or_none_valid() -> None:
    from kernel.context.mcp_storage import _parse_json_or_none

    assert _parse_json_or_none(json.dumps({"key": "value"})) == {"key": "value"}


def test_parse_json_or_none_null() -> None:
    from kernel.context.mcp_storage import _parse_json_or_none

    assert _parse_json_or_none("null") is None


def test_parse_json_or_none_empty() -> None:
    from kernel.context.mcp_storage import _parse_json_or_none

    assert _parse_json_or_none("") is None


# ── Integration: MCPInputGuard via spawned lisa-guard ───────────────────────


@pytest.mark.asyncio
async def test_mcp_input_guard_integration(tmp_path: Path) -> None:
    """Spawn lisa-guard and validate that MCPInputGuard returns a GuardResult."""
    import sys

    cfg = MCPClientConfig(
        server_id="lisa-guard-test",
        command=sys.executable,
        args=[
            "-m",
            "lisa_guard.server",
            "--data-dir",
            str(tmp_path),
        ],
    )
    guard = MCPInputGuard(cfg)
    surface = SurfaceContext(surface_id="test-surface")
    result = await guard.validate(query="hello world", surface=surface)
    assert isinstance(result, GuardResult)
    assert result.allowed is True


@pytest.mark.asyncio
async def test_mcp_output_guard_integration(tmp_path: Path) -> None:
    """Spawn lisa-guard and validate that MCPOutputGuard returns a GuardResult."""
    import sys

    cfg = MCPClientConfig(
        server_id="lisa-guard-test",
        command=sys.executable,
        args=[
            "-m",
            "lisa_guard.server",
            "--data-dir",
            str(tmp_path),
        ],
    )
    guard = MCPOutputGuard(cfg)
    surface = SurfaceContext(surface_id="test-surface")
    result = await guard.validate(text="safe output", surface=surface)
    assert isinstance(result, GuardResult)
    assert result.allowed is True


def test_mcp_content_registry_integration(tmp_path: Path) -> None:
    """Spawn lisa-content and verify read_authored_text returns empty for absent path."""
    import sys

    cfg = MCPClientConfig(
        server_id="lisa-content-test",
        command=sys.executable,
        args=[
            "-m",
            "lisa_sdk.lisa_content.server",
            "--data-dir",
            str(tmp_path),
        ],
    )
    registry = MCPContentRegistry(cfg)
    text = registry.read_authored_text("nonexistent/path.md")
    assert text == ""
