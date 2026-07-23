"""Tests for the MCP proxy policy: schema projection + ambient arg injection.

The MCPToolPool is a proxy between the LLM and N external MCP servers. A
``ProxyPolicy`` on a server's config tells the proxy to enforce ambient
scope boundaries: fields the model must not see or set (user identity,
agent identity, app identity) are stripped from the tool schema before it
reaches the model and injected into the arguments before they reach the
server.

Covers:
- Schema projection: ambient fields removed from input_schema.properties/required.
- Arg injection (non-scoped path): direct native-tool calls get ambient values.
- Arg injection (scoped path): alias-routed calls get ambient values.
- Idempotency: double-application (alias → scoped) doesn't duplicate filters.
- Structured filters: ambient fields injected as filters.AND entries.
- Runtime resolvers: set_ambient_resolver + fallback defaults.
- Manifest parsing: proxy_policy block parses into ProxyPolicy.
- Compiler: proxy_policy carried onto MCPClientConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kernel.mcp_launch.compiler import compile_mcp_config
from kernel.mcp_launch.policy import (
    AmbientSource,
    MCPServerSpec,
    ProxyPolicy,
    RuntimeKind,
)
from kernel.mcp_launch.server_manifest import load_mcp_server_manifest
from kernel.tools.mcp_client import (
    MCPClientConfig,
    MCPToolPool,
    MCPToolSpec,
    namespaced_tool_name,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _config_with_proxy(
    server_id: str = "mem0",
    proxy_policy: ProxyPolicy | None = None,
) -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
        proxy_policy=proxy_policy,
    )


def _patch_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        self._sessions[config.server_id] = object()

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)


def _patch_refresh_with_proxy(
    monkeypatch: pytest.MonkeyPatch,
    server_tools: dict[str, list[tuple[str, str, dict[str, Any]]]],
) -> None:
    """Like _patch_refresh_with but applies proxy_policy schema stripping."""

    async def fake_refresh(self: MCPToolPool) -> None:
        self._tools = []
        self._tool_index = {}
        self._tool_index_by_server = {}
        self._bare_to_server_index = {}
        for config in self._configs:
            server_index: dict[str, MCPToolSpec] = {}
            for name, description, input_schema in server_tools.get(config.server_id, []):
                # Apply proxy policy schema stripping (mirrors real _refresh_tool_index)
                policy = getattr(config, "proxy_policy", None)
                if policy is not None and getattr(policy, "ambient", None):
                    input_schema = self._strip_ambient_from_schema(input_schema, policy)
                bare_spec = MCPToolSpec(
                    server_id=config.server_id,
                    name=name,
                    description=description,
                    input_schema=input_schema,
                )
                server_index[name] = bare_spec
                namespaced = namespaced_tool_name(config.server_id, name)
                namespaced_spec = MCPToolSpec(
                    server_id=config.server_id,
                    name=namespaced,
                    description=description,
                    input_schema=input_schema,
                )
                if namespaced not in self._tool_index:
                    self._tool_index[namespaced] = (config.server_id, namespaced_spec)
                    self._tools.append(namespaced_spec)
                if name not in self._bare_to_server_index:
                    self._bare_to_server_index[name] = config.server_id
            self._tool_index_by_server[config.server_id] = server_index
        self._reindex_synthetic_tools()

    monkeypatch.setattr(MCPToolPool, "_refresh_tool_index", fake_refresh)


_MEM0_PROXY_POLICY = ProxyPolicy(
    ambient={
        "user_id": AmbientSource.USER_SLUG,
        "agent_id": AmbientSource.MODE_ID,
        "app_id": AmbientSource.APP_ID,
    },
    structured_filters=True,
)

_MEM0_TOOL_SCHEMAS = {
    "mem0": [
        (
            "add_memory",
            "Add a memory",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "app_id": {"type": "string"},
                },
                "required": ["text", "user_id"],
            },
        ),
        (
            "get_memories",
            "List memories",
            {
                "type": "object",
                "properties": {
                    "filters": {"type": "object"},
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "app_id": {"type": "string"},
                    "page": {"type": "integer"},
                    "page_size": {"type": "integer"},
                },
                "required": ["user_id"],
            },
        ),
        (
            "search_memories",
            "Search memories",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {"type": "object"},
                    "user_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "app_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query", "user_id"],
            },
        ),
    ],
}


class _FakeResult:
    def __init__(self, text: str = "ok") -> None:
        self.content = [type("C", (), {"text": text})()]
        self.is_error = False


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> _FakeResult:
        self.calls.append((tool_id, dict(arguments)))
        return _FakeResult()

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            tools=[
                type(
                    "T",
                    (),
                    {
                        "name": name,
                        "description": desc,
                        "inputSchema": schema,
                    },
                )()
                for name, desc, schema in _MEM0_TOOL_SCHEMAS["mem0"]
            ]
        )


# ── Schema projection ──────────────────────────────────────────────────────


def test_schema_projection_strips_ambient_fields() -> None:
    """_strip_ambient_from_schema removes ambient fields from properties
    and required, leaving model-controlled fields intact."""
    pool = MCPToolPool([_config_with_proxy(proxy_policy=_MEM0_PROXY_POLICY)])
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "user_id": {"type": "string"},
            "agent_id": {"type": "string"},
            "app_id": {"type": "string"},
        },
        "required": ["text", "user_id"],
    }
    stripped = pool._strip_ambient_from_schema(schema, _MEM0_PROXY_POLICY)
    props = stripped["properties"]
    assert "text" in props  # model-controlled, kept
    assert "user_id" not in props
    assert "agent_id" not in props
    assert "app_id" not in props
    assert stripped["required"] == ["text"]  # user_id removed from required


def test_schema_projection_noop_without_ambient() -> None:
    """An empty ProxyPolicy doesn't strip anything."""
    pool = MCPToolPool([_config_with_proxy()])
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    result = pool._strip_ambient_from_schema(schema, ProxyPolicy())
    assert result == schema


# ── Arg injection: non-scoped (direct native call) ─────────────────────────


@pytest.mark.asyncio
async def test_direct_native_call_gets_ambient_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the model calls mem0.get_memories directly (namespaced), the
    proxy injects ambient user_id/agent_id/app_id into filters.AND."""
    _patch_connect(monkeypatch)
    _patch_refresh_with_proxy(monkeypatch, _MEM0_TOOL_SCHEMAS)

    pool = MCPToolPool([_config_with_proxy(proxy_policy=_MEM0_PROXY_POLICY)])
    await pool.start()

    session = _FakeSession()
    pool._sessions["mem0"] = session
    pool.set_ambient_resolver("user_slug", lambda: "adam")
    pool.set_ambient_resolver("mode_id", lambda: "coach")
    pool.set_ambient_resolver("app_id", lambda: "little-lisa")

    # Model calls the native tool directly with NO user_id (what caused the bug)
    result = await pool.call_tool("mem0.get_memories", {})
    assert result.is_error is False

    # The session received the bare name with ambient filters injected
    assert len(session.calls) == 1
    tool_name, args = session.calls[0]
    assert tool_name == "get_memories"
    and_list = args.get("filters", {}).get("AND", [])
    ambient_entries = {list(e.keys())[0]: list(e.values())[0] for e in and_list}
    assert ambient_entries["user_id"] == "adam"
    assert ambient_entries["agent_id"] == "coach"
    assert ambient_entries["app_id"] == "little-lisa"
    await pool.aclose()


@pytest.mark.asyncio
async def test_direct_native_call_add_memory_gets_top_level_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """add_memory has no 'filters' param, so ambient fields are top-level."""
    _patch_connect(monkeypatch)
    _patch_refresh_with_proxy(monkeypatch, _MEM0_TOOL_SCHEMAS)

    pool = MCPToolPool([_config_with_proxy(proxy_policy=_MEM0_PROXY_POLICY)])
    await pool.start()

    session = _FakeSession()
    pool._sessions["mem0"] = session
    pool.set_ambient_resolver("user_slug", lambda: "adam")
    pool.set_ambient_resolver("mode_id", lambda: "coach")
    pool.set_ambient_resolver("app_id", lambda: "little-lisa")

    result = await pool.call_tool("mem0.add_memory", {"text": "hello"})
    assert result.is_error is False
    _, args = session.calls[0]
    assert args["text"] == "hello"  # model-supplied, preserved
    assert args["user_id"] == "adam"  # ambient-injected
    assert args["agent_id"] == "coach"
    assert args["app_id"] == "little-lisa"
    await pool.aclose()


# ── Arg injection: scoped (alias-routed call) ──────────────────────────────


@pytest.mark.asyncio
async def test_scoped_call_gets_ambient_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server-scoped calls (from aliases) also get proxy ambient injection."""
    _patch_connect(monkeypatch)
    _patch_refresh_with_proxy(monkeypatch, _MEM0_TOOL_SCHEMAS)

    pool = MCPToolPool([_config_with_proxy(proxy_policy=_MEM0_PROXY_POLICY)])
    await pool.start()

    session = _FakeSession()
    pool._sessions["mem0"] = session
    pool.set_ambient_resolver("user_slug", lambda: "adam")
    pool.set_ambient_resolver("mode_id", lambda: "coach")
    pool.set_ambient_resolver("app_id", lambda: "little-lisa")

    # Scoped call (as mcp_alias_tool would do)
    result = await pool.call_tool("get_memories", {}, server_id="mem0")
    assert result.is_error is False
    _, args = session.calls[0]
    and_list = args.get("filters", {}).get("AND", [])
    ambient_entries = {list(e.keys())[0]: list(e.values())[0] for e in and_list}
    assert ambient_entries["user_id"] == "adam"
    assert ambient_entries["agent_id"] == "coach"
    assert ambient_entries["app_id"] == "little-lisa"
    await pool.aclose()


# ── Idempotency ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_idempotent_filters_no_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the alias already injected filters.AND, the proxy replaces
    ambient entries — no duplication."""
    _patch_connect(monkeypatch)
    _patch_refresh_with_proxy(monkeypatch, _MEM0_TOOL_SCHEMAS)

    pool = MCPToolPool([_config_with_proxy(proxy_policy=_MEM0_PROXY_POLICY)])
    await pool.start()

    session = _FakeSession()
    pool._sessions["mem0"] = session
    pool.set_ambient_resolver("user_slug", lambda: "adam")
    pool.set_ambient_resolver("mode_id", lambda: "coach")
    pool.set_ambient_resolver("app_id", lambda: "little-lisa")

    # Simulate what an alias would send: already has filters.AND with ambient
    pre_injected = {
        "filters": {
            "AND": [
                {"user_id": "old_user"},
                {"agent_id": "old_agent"},
                {"app_id": "old_app"},
                {"content": "model_filter"},
            ]
        },
    }
    result = await pool.call_tool("get_memories", pre_injected, server_id="mem0")
    assert result.is_error is False
    _, args = session.calls[0]
    and_list = args["filters"]["AND"]

    # Should have exactly 4 entries: 3 ambient (replaced) + 1 content (preserved)
    assert len(and_list) == 4
    ambient_entries = {
        list(e.keys())[0]: list(e.values())[0]
        for e in and_list
        if list(e.keys())[0] in ("user_id", "agent_id", "app_id")
    }
    assert ambient_entries["user_id"] == "adam"  # replaced, not "old_user"
    assert ambient_entries["agent_id"] == "coach"
    assert ambient_entries["app_id"] == "little-lisa"
    # Model-supplied content filter preserved
    content_entries = [e for e in and_list if "content" in e]
    assert len(content_entries) == 1
    assert content_entries[0]["content"] == "model_filter"
    await pool.aclose()


# ── Runtime resolvers ──────────────────────────────────────────────────────


def test_ambient_resolver_defaults() -> None:
    """Without a registered resolver, defaults are used."""
    pool = MCPToolPool([_config_with_proxy()])
    assert pool._resolve_ambient("user_slug") == "default"
    assert pool._resolve_ambient("mode_id") == "default"
    assert pool._resolve_ambient("app_id") == "lisa"


def test_ambient_resolver_registered() -> None:
    """A registered resolver overrides the default."""
    pool = MCPToolPool([_config_with_proxy()])
    pool.set_ambient_resolver("user_slug", lambda: "adam")
    assert pool._resolve_ambient("user_slug") == "adam"


def test_ambient_resolver_empty_string_falls_back() -> None:
    """An empty-string resolver falls back to the default."""
    pool = MCPToolPool([_config_with_proxy()])
    pool.set_ambient_resolver("user_slug", lambda: "")
    assert pool._resolve_ambient("user_slug") == "default"


# ── Manifest parsing ───────────────────────────────────────────────────────


def test_manifest_parses_proxy_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """proxy_policy block in mcp_servers.yaml parses into ProxyPolicy."""
    monkeypatch.setenv("MEM0_API_KEY_TEST", "secret")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: mem0
    runtime_kind: http
    url: https://mcp.mem0.ai/mcp
    headers:
      Authorization: "Bearer ${MEM0_API_KEY_TEST}"
    proxy_policy:
      ambient:
        user_id: user_slug
        agent_id: mode_id
        app_id: app_id
      structured_filters: true
""",
        encoding="utf-8",
    )
    specs, _ = load_mcp_server_manifest(manifest)
    assert len(specs) == 1
    policy = specs[0].proxy_policy
    assert policy is not None
    assert policy.ambient["user_id"] is AmbientSource.USER_SLUG
    assert policy.ambient["agent_id"] is AmbientSource.MODE_ID
    assert policy.ambient["app_id"] is AmbientSource.APP_ID
    assert policy.structured_filters is True


def test_manifest_invalid_ambient_source_raises(tmp_path: Path) -> None:
    """An unknown ambient source value raises a manifest error."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: mem0
    runtime_kind: http
    url: https://mcp.mem0.ai/mcp
    proxy_policy:
      ambient:
        user_id: bogus_source
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="must be one of"):
        load_mcp_server_manifest(manifest)


def test_manifest_no_proxy_policy_defaults_empty(tmp_path: Path) -> None:
    """A server without proxy_policy gets a default (empty) ProxyPolicy."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: lisa-memory
    runtime_kind: venv_bin
    venv_dir: "/fake/venv"
""",
        encoding="utf-8",
    )
    specs, _ = load_mcp_server_manifest(manifest)
    assert specs[0].proxy_policy.ambient == {}


# ── Compiler ───────────────────────────────────────────────────────────────


def test_compiler_carries_proxy_policy() -> None:
    """compile_mcp_config carries proxy_policy from spec to config."""
    policy = ProxyPolicy(
        ambient={"user_id": AmbientSource.USER_SLUG},
        structured_filters=True,
    )
    spec = MCPServerSpec(
        server_id="mem0",
        runtime_kind=RuntimeKind.HTTP,
        url="https://mcp.mem0.ai/mcp",
        proxy_policy=policy,
    )
    config = compile_mcp_config(spec)
    assert config.proxy_policy is policy
