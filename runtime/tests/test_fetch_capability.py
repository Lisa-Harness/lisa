"""The ``fetch`` capability: built-in server, contract, and fetch_page routing.

``fetch_page`` delegates the network fetch to ``fetch.fetch_url`` so
``providers.capabilities.fetch`` selects the implementation. Before this
existed the selection was silently ignored and the tool always used an
in-process fetcher (the Go binary was never launched or downloaded).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from kernel.mcp_launch.server_manifest import load_mcp_server_manifest
from kernel.models import ToolExecutionContext, ToolResult
from kernel.tools.mcp_client import MCPToolPool
from kernel.turn_phases._turn_state import mcp_tool_caller
from mcp_server.fetch.server import mcp as fetch_mcp
from runtime.runtime import _make_in_process_resolver
from runtime.testing import build_test_runtime
from sdk.packs.core.tools.fetch_page import fetch_page

from runtime import mcp_pool

_HOST_MANIFEST = Path(mcp_pool.__file__).with_name("mcp_servers.yaml")


def _fetch_spec() -> Any:
    specs = [s for s in load_mcp_server_manifest(_HOST_MANIFEST) if s.server_id == "lisa-fetch"]
    assert len(specs) == 1
    return specs[0]


def test_host_manifest_declares_builtin_fetch_capability() -> None:
    """The built-in claims ``fetch`` with exactly the Python server's tools."""
    spec = _fetch_spec()
    served = {tool.name for tool in asyncio.run(fetch_mcp.list_tools())}

    assert spec.capability == "fetch"
    assert set(spec.capability_tools) == served == {"fetch_url", "invalidate_fetch_cache"}
    # fetch_page stays the only model-facing fetch tool.
    assert list(spec.default_available_tools) == []


def test_fetch_page_routes_through_in_process_fetch_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: compiled manifest → in-process lisa-fetch → fetch_page.

    Covers the success payload and the HTTP-refusal mapping through the real
    MCP error path (the message ``fetch_page`` parses the status from).
    """
    runtime = build_test_runtime(tmp_path, models_yaml="", user_slug="testing")
    runtime._in_process_mcp = True
    monkeypatch.setenv("FRAMEWORK_ROOT", str(tmp_path / "framework"))
    monkeypatch.setenv("LISA_TIMEZONE", "UTC")
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    async def _fake_get(self: httpx.AsyncClient, url: str, **_kw: Any) -> httpx.Response:
        if url.endswith("/blocked"):
            return httpx.Response(403, request=httpx.Request("GET", url), content=b"no")
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            content=b"<html><title>Guide</title><body><p>Use budgets.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    configs = [cfg for cfg in [*warm, *cold] if cfg.server_id == "lisa-fetch"]
    assert len(configs) == 1
    assert configs[0].runtime_kind == "in_process"
    pool = MCPToolPool(configs, in_process_resolver=_make_in_process_resolver())

    async def _drive() -> tuple[ToolResult, ToolResult]:
        await pool.start()
        try:
            call = mcp_tool_caller(pool)

            async def _run(url: str) -> ToolResult:
                ctx = ToolExecutionContext(
                    tool_id="fetch_page", parameters={"url": url}, mcp_tool_call=call
                )
                return await fetch_page.execute(ctx)

            return await _run("https://example.com/guide"), await _run(
                "https://example.com/blocked"
            )
        finally:
            await pool.aclose()

    ok, refused = asyncio.run(_drive())

    assert ok.error == ""
    assert ok.output_text.startswith("# Guide\nSource: https://example.com/guide")
    assert "Use budgets." in ok.output_text
    assert refused.error == (
        "Fetch failed: HTTP 403 for https://example.com/blocked. "
        "The site refused this request; try a different source."
    )
