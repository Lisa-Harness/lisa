"""Tests for kernel/context/_shared_stdio.py."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from kernel.context._shared_stdio import SharedStdioSession, close_shared_sessions
from kernel.context.mcp_content_registry import MCPContentRegistry
from kernel.tools.mcp_client import MCPClientConfig
from mcp.client.stdio import StdioServerParameters

_SERVER = '''
import os

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("shared-stdio-test")


@mcp.tool()
def pid() -> str:
    """Return the server process id."""
    return str(os.getpid())


@mcp.tool()
def read_authored_text(relative_path: str) -> str:
    """Echo the requested path."""
    return "text:" + relative_path


@mcp.tool()
def die() -> str:
    """Exit the server process abruptly."""
    os._exit(0)


if __name__ == "__main__":
    mcp.run()
'''


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    script = tmp_path / "server.py"
    script.write_text(_SERVER, encoding="utf-8")
    return script


@pytest.fixture(autouse=True)
def _close_sessions() -> Iterator[None]:
    try:
        yield
    finally:
        close_shared_sessions()


def _session(script: Path) -> SharedStdioSession:
    return SharedStdioSession(StdioServerParameters(command=sys.executable, args=[str(script)]))


def _text(result: object) -> str:
    content = getattr(result, "content", [])
    return str(content[0].text)


def test_equal_configs_share_one_subprocess(server_script: Path) -> None:
    first = _session(server_script)
    second = _session(server_script)
    first.prewarm()

    pids = {_text(first.call("pid", {})), _text(second.call("pid", {}))}

    assert len(pids) == 1


def test_dead_session_is_replaced_and_call_retried(server_script: Path) -> None:
    session = _session(server_script)
    original = _text(session.call("pid", {}))

    with pytest.raises(Exception):  # noqa: B017 — transport error type varies by platform
        session.call("die", {})
    replacement = _text(session.call("pid", {}))

    assert replacement != original


def test_close_then_call_starts_new_session(server_script: Path) -> None:
    session = _session(server_script)
    original = _text(session.call("pid", {}))

    close_shared_sessions()

    assert _text(session.call("pid", {})) != original


def test_startup_failure_is_raised(tmp_path: Path) -> None:
    session = SharedStdioSession(
        StdioServerParameters(command=sys.executable, args=[str(tmp_path / "missing.py")])
    )

    with pytest.raises(Exception):  # noqa: B017 — MCP surfaces the closed stream
        session.call("pid", {})


def test_content_registry_reads_through_shared_session(server_script: Path) -> None:
    registry = MCPContentRegistry(
        MCPClientConfig(server_id="lisa-content", command=sys.executable, args=[str(server_script)])
    )

    assert registry.read_authored_text("prompts/system-prompt.md") == (
        "text:prompts/system-prompt.md"
    )
