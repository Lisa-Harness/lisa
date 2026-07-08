"""Tests for the MCP launch policy, runtime adapter, and tool filter.

Covers:
- MCPServerRegistry approval gate
- compile_command for each RuntimeKind
- compile_env projection (inherit, deny, allow, force)
- compile_cwd resolution (inherit, fixed, workspace)
- compile_mcp_config (data_dir injection, CLI arg construction)
- compile_registry (batch compile with host args)
- filter_tools (allow/deny lists)
"""

from __future__ import annotations

import os

import pytest

from kernel.mcp_launch.compiler import (
    compile_command,
    compile_cwd,
    compile_env,
    compile_mcp_config,
    compile_registry,
)
from kernel.mcp_launch.policy import (
    CwdPolicy,
    EnvPolicy,
    MCPServerRegistry,
    MCPServerSpec,
    RuntimeKind,
    ToolPolicy,
)
from kernel.tools.mcp_client import MCPToolSpec

# ── Registry ────────────────────────────────────────────────────────────────


def test_registry_register_and_get() -> None:
    registry = MCPServerRegistry()
    spec = MCPServerSpec(
        server_id="test",
        runtime_kind=RuntimeKind.PYTHON,
        module="test.server",
    )
    registry.register(spec)
    assert registry.is_approved("test")
    assert registry.get("test") is not None


def test_registry_rejects_duplicate() -> None:
    registry = MCPServerRegistry()
    spec = MCPServerSpec(
        server_id="test",
        runtime_kind=RuntimeKind.PYTHON,
        module="test.server",
    )
    registry.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_registry_is_approved_false_for_unknown() -> None:
    registry = MCPServerRegistry()
    assert not registry.is_approved("unknown")
    assert registry.get("unknown") is None


def test_registry_all_specs() -> None:
    registry = MCPServerRegistry()
    registry.register(MCPServerSpec(server_id="a", runtime_kind=RuntimeKind.PYTHON, module="a"))
    registry.register(MCPServerSpec(server_id="b", runtime_kind=RuntimeKind.PYTHON, module="b"))
    assert len(registry.all_specs()) == 2


# ── compile_command ─────────────────────────────────────────────────────────


def test_compile_command_python() -> None:
    spec = MCPServerSpec(server_id="t", runtime_kind=RuntimeKind.PYTHON, module="pkg.server")
    cmd, args = compile_command(spec)
    assert cmd == "python"
    assert args == ["-m", "pkg.server"]


def test_compile_command_uv() -> None:
    spec = MCPServerSpec(
        server_id="t", runtime_kind=RuntimeKind.UV, package="mcp-server", module="pkg.server"
    )
    cmd, args = compile_command(spec)
    assert cmd == "uv"
    assert "run" in args
    assert "--package" in args
    assert "mcp-server" in args
    assert "python" in args
    assert "-m" in args
    assert "pkg.server" in args


def test_compile_command_uv_with_directory() -> None:
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.UV,
        package="mcp-server",
        module="pkg.server",
        uv_directory="/path/to/framework",
    )
    cmd, args = compile_command(spec)
    assert cmd == "uv"
    assert "--directory" in args
    assert "/path/to/framework" in args


def test_compile_command_npx() -> None:
    spec = MCPServerSpec(server_id="t", runtime_kind=RuntimeKind.NPX, npx_package="@scope/server")
    cmd, args = compile_command(spec)
    assert cmd == "npx"
    assert "-y" in args
    assert "@scope/server" in args


def test_compile_command_binary() -> None:
    spec = MCPServerSpec(
        server_id="t", runtime_kind=RuntimeKind.BINARY, command_path="/usr/local/bin/server"
    )
    cmd, args = compile_command(spec)
    assert cmd == "/usr/local/bin/server"
    assert args == []


def test_compile_command_node() -> None:
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.NODE,
        command_path="node",
        script_path="server.js",
    )
    cmd, args = compile_command(spec)
    assert cmd == "node"
    assert args == ["server.js"]


def test_compile_command_python_requires_module() -> None:
    spec = MCPServerSpec(server_id="t", runtime_kind=RuntimeKind.PYTHON)
    with pytest.raises(ValueError, match="module"):
        compile_command(spec)


def test_compile_command_uv_requires_package_and_module() -> None:
    spec = MCPServerSpec(server_id="t", runtime_kind=RuntimeKind.UV, module="pkg.server")
    with pytest.raises(ValueError, match="package"):
        compile_command(spec)


def test_compile_command_venv_bin() -> None:
    spec = MCPServerSpec(
        server_id="lisa-context",
        runtime_kind=RuntimeKind.VENV_BIN,
        venv_dir="/path/to/.venv",
    )
    cmd, args = compile_command(spec)
    # Path resolves to <venv_dir>/Scripts/<entry_point>.exe on Windows
    # or <venv_dir>/bin/<entry_point> on POSIX.
    assert "lisa-context" in cmd
    assert "Scripts" in cmd or "bin" in cmd
    assert args == []


def test_compile_command_venv_bin_custom_entry_point() -> None:
    spec = MCPServerSpec(
        server_id="ctx",
        runtime_kind=RuntimeKind.VENV_BIN,
        entry_point="lisa-context",
        venv_dir="/path/to/.venv",
    )
    cmd, args = compile_command(spec)
    assert "lisa-context" in cmd
    assert args == []


def test_compile_command_venv_bin_requires_venv_dir() -> None:
    spec = MCPServerSpec(server_id="t", runtime_kind=RuntimeKind.VENV_BIN)
    with pytest.raises(ValueError, match="venv_dir"):
        compile_command(spec)


# ── compile_env ──────────────────────────────────────────────────────────────


def test_compile_env_inherit_parent() -> None:
    os.environ["TEST_MCP_KEY"] = "parent_value"
    try:
        env = compile_env(EnvPolicy(inherit_parent=True))
        assert env["TEST_MCP_KEY"] == "parent_value"
    finally:
        del os.environ["TEST_MCP_KEY"]


def test_compile_env_no_inherit() -> None:
    os.environ["TEST_MCP_KEY"] = "parent_value"
    try:
        env = compile_env(EnvPolicy(inherit_parent=False))
        assert "TEST_MCP_KEY" not in env
    finally:
        del os.environ["TEST_MCP_KEY"]


def test_compile_env_denied_keys() -> None:
    os.environ["TEST_MCP_DENY"] = "secret"
    try:
        env = compile_env(EnvPolicy(inherit_parent=True, denied_keys=["TEST_MCP_DENY"]))
        assert "TEST_MCP_DENY" not in env
    finally:
        del os.environ["TEST_MCP_DENY"]


def test_compile_env_allowed_keys() -> None:
    os.environ["TEST_MCP_KEEP"] = "keep"
    os.environ["TEST_MCP_DROP"] = "drop"
    try:
        env = compile_env(EnvPolicy(inherit_parent=True, allowed_keys=["TEST_MCP_KEEP"]))
        assert env["TEST_MCP_KEEP"] == "keep"
        assert "TEST_MCP_DROP" not in env
    finally:
        del os.environ["TEST_MCP_KEEP"]
        del os.environ["TEST_MCP_DROP"]


def test_compile_env_forced_overrides_win() -> None:
    os.environ["TEST_MCP_OVERRIDE"] = "original"
    try:
        env = compile_env(
            EnvPolicy(inherit_parent=True, forced_overrides={"TEST_MCP_OVERRIDE": "forced"})
        )
        assert env["TEST_MCP_OVERRIDE"] == "forced"
    finally:
        del os.environ["TEST_MCP_OVERRIDE"]


# ── compile_cwd ──────────────────────────────────────────────────────────────


def test_compile_cwd_inherit() -> None:
    assert compile_cwd(CwdPolicy(kind="inherit")) is None


def test_compile_cwd_fixed() -> None:
    result = compile_cwd(CwdPolicy(kind="fixed", path="/tmp"))
    assert str(result) == "/tmp" or str(result).endswith("\\tmp")


def test_compile_cwd_workspace() -> None:
    result = compile_cwd(CwdPolicy(kind="workspace", path="subdir"), workspace_root="/ws")
    assert "subdir" in str(result)
    assert "ws" in str(result)


def test_compile_cwd_workspace_without_root() -> None:
    assert compile_cwd(CwdPolicy(kind="workspace", path="subdir")) is None


# ── compile_mcp_config ──────────────────────────────────────────────────────


def test_compile_mcp_config_without_data_dir_is_optional() -> None:
    """data_dir is now optional — a spec without it compiles without injection.

    Third-party servers that manage their own storage (e.g. Mem0 via npx)
    set data_dir_arg=None and never receive a data-dir flag.
    """
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir_arg=None,
    )
    config = compile_mcp_config(spec)
    assert config.server_id == "t"
    assert "--data-dir" not in config.args
    assert config.data_dir_path is None


def test_compile_mcp_config_builds_cli_args() -> None:
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        workspace_root="/ws",
    )
    config = compile_mcp_config(spec)
    assert config.server_id == "t"
    assert config.command == "python"
    assert "--data-dir" in config.args
    assert "/data" in config.args
    assert "--workspace-root" in config.args
    assert "/ws" in config.args


def test_compile_mcp_config_readonly_flag() -> None:
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        readonly=True,
    )
    config = compile_mcp_config(spec)
    assert "--readonly" in config.args


def test_compile_mcp_config_extra_cli_args() -> None:
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        cli_args=["--backend", "azure"],
    )
    config = compile_mcp_config(spec)
    assert "--backend" in config.args
    assert "azure" in config.args


# ── compile_registry ─────────────────────────────────────────────────────────


def test_compile_registry_compiles_all() -> None:
    registry = MCPServerRegistry()
    registry.register(
        MCPServerSpec(server_id="a", runtime_kind=RuntimeKind.PYTHON, module="a.server")
    )
    registry.register(
        MCPServerSpec(server_id="b", runtime_kind=RuntimeKind.PYTHON, module="b.server")
    )
    configs = compile_registry(registry, data_dir="/data")
    assert len(configs) == 2
    assert all(c.args.count("--data-dir") == 1 for c in configs)
    assert all("/data" in c.args for c in configs)


def test_compile_registry_injects_workspace_root() -> None:
    registry = MCPServerRegistry()
    registry.register(
        MCPServerSpec(server_id="a", runtime_kind=RuntimeKind.PYTHON, module="a.server")
    )
    configs = compile_registry(registry, data_dir="/data", workspace_root="/ws")
    assert any("--workspace-root" in c.args for c in configs)


# ── data-dir adaptation ─────────────────────────────────────────────────────


def test_compile_mcp_config_data_dir_subdir_injects_namespaced_path() -> None:
    """data_dir_subdir appends a namespaced subfolder to the injected path."""
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        data_dir_subdir="mem0",
    )
    config = compile_mcp_config(spec)
    assert "--data-dir" in config.args
    idx = config.args.index("--data-dir")
    assert config.args[idx + 1].endswith("mem0")
    assert "mem0" in config.args[idx + 1]


def test_compile_mcp_config_custom_data_dir_arg() -> None:
    """data_dir_arg overrides the flag name injected for third-party servers."""
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        data_dir_arg="--store-dir",
    )
    config = compile_mcp_config(spec)
    assert "--store-dir" in config.args
    assert "--data-dir" not in config.args
    assert "/data" in config.args


def test_compile_mcp_config_null_data_dir_arg_omits_injection() -> None:
    """data_dir_arg=None means the server manages its own storage — no flag."""
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        data_dir_arg=None,
    )
    config = compile_mcp_config(spec)
    assert "--data-dir" not in config.args
    assert "/data" not in config.args


def test_compile_mcp_config_ensure_data_dir_sets_pool_fields() -> None:
    """ensure_data_dir stamps data_dir_path/ensure_data_dir for the pool to mkdir."""
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        data_dir_subdir="mem0",
        ensure_data_dir=True,
    )
    config = compile_mcp_config(spec)
    assert config.ensure_data_dir is True
    assert config.data_dir_path is not None
    assert config.data_dir_path.endswith("mem0")


def test_compile_mcp_config_ensure_data_dir_without_subdir_uses_plain_data_dir() -> None:
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
        ensure_data_dir=True,
    )
    config = compile_mcp_config(spec)
    assert config.ensure_data_dir is True
    assert config.data_dir_path == "/data"


def test_compile_mcp_config_ensure_data_dir_false_does_not_stamp_path() -> None:
    """When ensure_data_dir is False, the pool does not receive a path to mkdir."""
    spec = MCPServerSpec(
        server_id="t",
        runtime_kind=RuntimeKind.PYTHON,
        module="pkg.server",
        data_dir="/data",
    )
    config = compile_mcp_config(spec)
    assert config.ensure_data_dir is False
    assert config.data_dir_path is None


def test_compile_mcp_config_first_party_default_preserves_behavior() -> None:
    """Default data_dir_arg='--data-dir' keeps first-party behavior unchanged."""
    spec = MCPServerSpec(
        server_id="lisa-context",
        runtime_kind=RuntimeKind.PYTHON,
        module="lisa_context.server",
        data_dir="/data",
    )
    config = compile_mcp_config(spec)
    assert "--data-dir" in config.args
    idx = config.args.index("--data-dir")
    assert config.args[idx + 1] == "/data"
    assert config.data_dir_path is None
    assert config.ensure_data_dir is False


# ── filter_tools ─────────────────────────────────────────────────────────────


def test_filter_tools_allowlist() -> None:
    from kernel.mcp_launch.tool_filter import filter_tools

    tools = [
        MCPToolSpec(server_id="s", name="keep", description="", input_schema={}),
        MCPToolSpec(server_id="s", name="drop", description="", input_schema={}),
    ]
    result = filter_tools(tools, ToolPolicy(allowed_tools=["keep"]))
    assert len(result) == 1
    assert result[0].name == "keep"


def test_filter_tools_denylist() -> None:
    from kernel.mcp_launch.tool_filter import filter_tools

    tools = [
        MCPToolSpec(server_id="s", name="keep", description="", input_schema={}),
        MCPToolSpec(server_id="s", name="drop", description="", input_schema={}),
    ]
    result = filter_tools(tools, ToolPolicy(denied_tools=["drop"]))
    assert len(result) == 1
    assert result[0].name == "keep"


def test_filter_tools_no_policy_passes_all() -> None:
    from kernel.mcp_launch.tool_filter import filter_tools

    tools = [
        MCPToolSpec(server_id="s", name="a", description="", input_schema={}),
        MCPToolSpec(server_id="s", name="b", description="", input_schema={}),
    ]
    result = filter_tools(tools, ToolPolicy())
    assert len(result) == 2


def test_filter_tools_allow_and_deny_combined() -> None:
    from kernel.mcp_launch.tool_filter import filter_tools

    tools = [
        MCPToolSpec(server_id="s", name="a", description="", input_schema={}),
        MCPToolSpec(server_id="s", name="b", description="", input_schema={}),
        MCPToolSpec(server_id="s", name="c", description="", input_schema={}),
    ]
    result = filter_tools(
        tools,
        ToolPolicy(allowed_tools=["a", "b", "c"], denied_tools=["b"]),
    )
    assert len(result) == 2
    assert {t.name for t in result} == {"a", "c"}
