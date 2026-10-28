"""Regression tests for the input-schema fallback warnings in MCPToolPool.

A tool whose schema is missing or empty is silently degraded to "takes no
parameters" at three sites in ``kernel.tools.mcp_client``: the synthetic-tool
rebuild, the lenient raw-POST tools/list, and the server tool-spec build. Each
site now emits a warning naming the server and tool so the degradation is
diagnosable from the log instead of presenting as a model-side failure.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import pytest
from kernel.tools.mcp_client import (
    SYNTHETIC_SERVER_ID,
    MCPClientConfig,
    MCPToolPool,
    MCPToolSpec,
)


def _config(server_id: str = "s") -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
    )


def _patch_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub _connect so no subprocess is spawned; just record a session."""

    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        self._sessions[config.server_id] = object()

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)


# ── _reindex_synthetic_tools ─────────────────────────────────────────────


def test_reindex_synthetic_warns_on_missing_prior_schema(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A synthetic tool rebuilt with no prior spec warns about the empty schema."""
    _patch_connect(monkeypatch)
    pool = MCPToolPool([_config()])
    pool._synthetic_handlers["memory.search"] = None  # type: ignore[assignment]
    # No prior spec in _tools, so the rebuild falls back to the empty schema.

    with caplog.at_level(logging.WARNING, logger="kernel.tools.mcp_client"):
        pool._reindex_synthetic_tools()

    assert any(
        "memory.search" in r.message and "empty input schema" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]
    spec = pool._tool_index["memory.search"][1]
    assert spec.input_schema == {"type": "object", "properties": {}}


def test_reindex_synthetic_no_warn_when_prior_schema_present(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A synthetic tool rebuilt from a prior spec with a real schema stays silent."""
    _patch_connect(monkeypatch)
    pool = MCPToolPool([_config()])
    pool._synthetic_handlers["memory.search"] = None  # type: ignore[assignment]
    pool._tools.append(
        MCPToolSpec(
            server_id=SYNTHETIC_SERVER_ID,
            name="memory.search",
            description="Search memory.",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
    )

    with caplog.at_level(logging.WARNING, logger="kernel.tools.mcp_client"):
        pool._reindex_synthetic_tools()

    assert not [r for r in caplog.records if "empty input schema" in r.message]
    spec = pool._tool_index["memory.search"][1]
    assert spec.input_schema == {"type": "object", "properties": {"q": {"type": "string"}}}


# ── _build_server_tool_specs ─────────────────────────────────────────────


def test_build_server_tool_specs_warns_on_empty_schema(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A discovered tool with an empty input_schema warns at spec-build time."""
    pool = MCPToolPool([_config("srv")])

    class _Tool:
        name = "fetch"
        description = "Fetch a thing."
        input_schema: ClassVar[dict[str, Any]] = {}
        annotations = None

    with caplog.at_level(logging.WARNING, logger="kernel.tools.mcp_client"):
        specs = pool._build_server_tool_specs(_config("srv"), [_Tool()])

    assert any(
        "fetch" in r.message and "empty input schema" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]
    assert specs[0].input_schema == {"type": "object", "properties": {}}


def test_build_server_tool_specs_no_warn_on_real_schema(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A discovered tool with a real schema stays silent at spec-build time."""
    pool = MCPToolPool([_config("srv")])

    class _Tool:
        name = "fetch"
        description = "Fetch a thing."
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        }
        annotations = None

    with caplog.at_level(logging.WARNING, logger="kernel.tools.mcp_client"):
        specs = pool._build_server_tool_specs(_config("srv"), [_Tool()])

    assert not [r for r in caplog.records if "empty input schema" in r.message]
    assert specs[0].input_schema == {"type": "object", "properties": {"q": {"type": "string"}}}


# ── _list_tools_lenient ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tools_lenient_warns_on_missing_inputschema(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A tools/list entry without inputSchema warns in the lenient raw-POST path."""
    _patch_connect(monkeypatch)
    config = MCPClientConfig(
        server_id="srv",
        command="python",
        args=["-m", "nonexistent.server"],
        url="http://127.0.0.1:1/",
    )
    pool = MCPToolPool([config])

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {"name": "fetch", "description": "Fetch."},
                        {
                            "name": "ok_tool",
                            "description": "OK.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        },
                    ]
                },
            }

    class _FakeClient:
        def __init__(self, timeout: float = 30.0) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def post(
            self, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> _FakeResponse:
            return _FakeResponse()

    import sys

    fake_httpx = type("fake_httpx", (), {"AsyncClient": _FakeClient})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger="kernel.tools.mcp_client"):
        result = await pool._list_tools_lenient(config, server_id="srv")

    assert any("fetch" in r.message and "no inputSchema" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]
    assert not any("ok_tool" in r.message for r in caplog.records)
    assert len(result.tools) == 2
    assert result.tools[0].input_schema == {"type": "object", "properties": {}}
    assert result.tools[1].input_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
