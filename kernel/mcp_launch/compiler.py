"""Runtime adapter — compiles approved specs into concrete MCP configs.

The compiler is the only place that produces
:class:`~kernel.tools.mcp_client.MCPClientConfig` entries for
:class:`~kernel.tools.mcp_client.MCPToolPool`. The host calls it after
registry approval.

Compilation is a pure projection: an :class:`MCPServerSpec` plus a
resolved ``data_dir`` (and optional ``workspace_root``) becomes a concrete
``MCPClientConfig`` with command, args, env, and cwd. No I/O, no spawning.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kernel.mcp_launch.policy import (
    CwdPolicy,
    EnvPolicy,
    MCPServerSpec,
    RuntimeKind,
)
from kernel.tools.mcp_client import MCPClientConfig

if TYPE_CHECKING:
    from kernel.mcp_launch.policy import MCPServerRegistry

__all__ = [
    "compile_command",
    "compile_cwd",
    "compile_env",
    "compile_mcp_config",
    "compile_registry",
]


def compile_env(policy: EnvPolicy) -> dict[str, str]:
    """Build the subprocess environment from an EnvPolicy."""
    if not policy.inherit_parent:
        env: dict[str, str] = {}
    else:
        env = dict(os.environ)
    for key in policy.denied_keys:
        env.pop(key, None)
    if policy.allowed_keys is not None:
        env = {k: v for k, v in env.items() if k in policy.allowed_keys}
    env.update(policy.forced_overrides)
    return env


def compile_cwd(policy: CwdPolicy, *, workspace_root: str | None = None) -> str | Path | None:
    """Resolve the working directory from a CwdPolicy."""
    if policy.kind == "inherit":
        return None
    if policy.kind == "fixed":
        return Path(policy.path or ".")
    if policy.kind == "workspace":
        if workspace_root is None:
            return None
        return Path(workspace_root) / (policy.path or ".")
    return None


def compile_command(spec: MCPServerSpec) -> tuple[str, list[str]]:
    """Build (command, args) for one runtime kind.

    Returns the base command and args before standard CLI contract args
    are appended.
    """
    if spec.runtime_kind == RuntimeKind.PYTHON:
        if not spec.module:
            raise ValueError(f"PYTHON runtime requires 'module' for server '{spec.server_id}'")
        return "python", ["-m", spec.module]

    if spec.runtime_kind == RuntimeKind.UV:
        if not spec.package or not spec.module:
            raise ValueError(
                f"UV runtime requires 'package' and 'module' for server '{spec.server_id}'"
            )
        args: list[str] = ["run"]
        if spec.uv_directory:
            args.extend(["--directory", spec.uv_directory])
        args.extend(["--package", spec.package, "python", "-m", spec.module])
        return "uv", args

    if spec.runtime_kind == RuntimeKind.VENV_BIN:
        if not spec.venv_dir:
            raise ValueError(f"VENV_BIN runtime requires 'venv_dir' for server '{spec.server_id}'")
        entry_point = spec.entry_point or spec.server_id
        if sys.platform == "win32":
            command_path = str(Path(spec.venv_dir) / "Scripts" / f"{entry_point}.exe")
        else:
            command_path = str(Path(spec.venv_dir) / "bin" / entry_point)
        return command_path, []

    if spec.runtime_kind == RuntimeKind.NODE:
        if not spec.command_path or not spec.script_path:
            raise ValueError(
                f"NODE runtime requires 'command_path' and 'script_path' for server '{spec.server_id}'"
            )
        return spec.command_path, [spec.script_path]

    if spec.runtime_kind == RuntimeKind.NPX:
        if not spec.npx_package:
            raise ValueError(f"NPX runtime requires 'npx_package' for server '{spec.server_id}'")
        return "npx", ["-y", spec.npx_package]

    if spec.runtime_kind == RuntimeKind.BINARY:
        if not spec.command_path:
            raise ValueError(
                f"BINARY runtime requires 'command_path' for server '{spec.server_id}'"
            )
        return spec.command_path, []

    if spec.runtime_kind == RuntimeKind.HTTP:
        if not spec.url:
            raise ValueError(f"HTTP runtime requires 'url' for server '{spec.server_id}'")
        return "", []

    raise ValueError(f"Unknown runtime kind '{spec.runtime_kind}' for server '{spec.server_id}'")


def compile_mcp_config(spec: MCPServerSpec) -> MCPClientConfig:
    """Compile one approved MCPServerSpec into a concrete MCPClientConfig.

    This is the only function that produces MCPClientConfig entries for
    MCPToolPool. The host calls this after registry approval.
    """
    # Resolve the effective data directory, applying an optional namespaced
    # subdir. A spec with data_dir=None (servers managing their own storage)
    # compiles without any data-dir injection.
    if spec.data_dir:
        resolved_data_dir = spec.data_dir
        if spec.data_dir_subdir:
            resolved_data_dir = str(Path(spec.data_dir) / spec.data_dir_subdir)
    else:
        resolved_data_dir = None

    command, base_args = compile_command(spec)

    # Build standard CLI contract args. data-dir is injected only when the
    # spec names a flag for it (data_dir_arg is not None) and a data dir was
    # resolved. First-party servers keep data_dir_arg="--data-dir" as the
    # default, preserving the original behavior. Third-party servers that
    # manage their own storage set data_dir_arg=None.
    cli_args: list[str] = []
    if spec.data_dir_arg is not None and resolved_data_dir is not None:
        cli_args.extend([spec.data_dir_arg, resolved_data_dir])
    if spec.workspace_root:
        cli_args.extend(["--workspace-root", spec.workspace_root])
    if spec.readonly:
        cli_args.append("--readonly")
    cli_args.extend(spec.cli_args)

    all_args = [*base_args, *cli_args]
    env = compile_env(spec.env_policy)
    cwd = compile_cwd(spec.cwd_policy, workspace_root=spec.workspace_root)

    # Default each first-party MCP server's logs to a per-server JSONL file
    # under its resolved data dir. The first-party servers read
    # ``LISA_MCP_LOG_PATH`` in ``mcp_server/config/cli.py:configure_logging``
    # and route through the shared ``sdk.logging_setup`` JSON formatter.
    # Without this, servers fall back to ``logging.basicConfig`` on stderr,
    # which the MCP SDK's ``StdioServerParameters`` (no ``stderr=`` field in
    # this version) inherits into the host terminal — visible on Linux as
    # stray log lines corrupting the Rich TUI. A spec may override
    # ``LISA_MCP_LOG_PATH`` via ``env_policy.forced_overrides``; respect that.
    if resolved_data_dir is not None and "LISA_MCP_LOG_PATH" not in env:
        env["LISA_MCP_LOG_PATH"] = str(Path(resolved_data_dir) / f"{spec.server_id}.jsonl")

    return MCPClientConfig(
        server_id=spec.server_id,
        command=command,
        args=all_args,
        env=env,
        cwd=cwd,
        tool_policy=spec.tool_policy,
        proxy_policy=spec.proxy_policy,
        data_dir_path=resolved_data_dir if spec.ensure_data_dir else None,
        ensure_data_dir=spec.ensure_data_dir,
        url=spec.url,
        headers=dict(spec.headers),
        warm=spec.warm,
        required=spec.required,
        aliases=spec.aliases,
    )


def compile_registry(
    registry: MCPServerRegistry,
    *,
    data_dir: str,
    workspace_root: str | None = None,
) -> list[MCPClientConfig]:
    """Compile all approved server specs from a registry into MCPClientConfig entries."""
    from dataclasses import replace

    configs: list[MCPClientConfig] = []
    for spec in registry.all_specs():
        # Inject host-provided args (slots dataclass — use replace, not __dict__)
        resolved_spec = replace(spec, data_dir=data_dir, workspace_root=workspace_root)
        configs.append(compile_mcp_config(resolved_spec))
    return configs
