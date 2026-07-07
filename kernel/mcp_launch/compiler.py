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
    "compile_env",
    "compile_cwd",
    "compile_command",
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
            raise ValueError(f"UV runtime requires 'package' and 'module' for server '{spec.server_id}'")
        return "uv", ["run", "--package", spec.package, "python", "-m", spec.module]

    if spec.runtime_kind == RuntimeKind.NODE:
        if not spec.command_path or not spec.script_path:
            raise ValueError(f"NODE runtime requires 'command_path' and 'script_path' for server '{spec.server_id}'")
        return spec.command_path, [spec.script_path]

    if spec.runtime_kind == RuntimeKind.NPX:
        if not spec.npx_package:
            raise ValueError(f"NPX runtime requires 'npx_package' for server '{spec.server_id}'")
        return "npx", ["-y", spec.npx_package]

    if spec.runtime_kind == RuntimeKind.BINARY:
        if not spec.command_path:
            raise ValueError(f"BINARY runtime requires 'command_path' for server '{spec.server_id}'")
        return spec.command_path, []

    raise ValueError(f"Unknown runtime kind '{spec.runtime_kind}' for server '{spec.server_id}'")


def compile_mcp_config(spec: MCPServerSpec) -> MCPClientConfig:
    """Compile one approved MCPServerSpec into a concrete MCPClientConfig.

    This is the only function that produces MCPClientConfig entries for
    MCPToolPool. The host calls this after registry approval.
    """
    if not spec.data_dir:
        raise ValueError(
            f"Cannot compile MCP server '{spec.server_id}': data_dir is required. "
            "The host must resolve --data-dir before compiling."
        )

    command, base_args = compile_command(spec)

    # Build standard CLI contract args
    cli_args: list[str] = ["--data-dir", spec.data_dir]
    if spec.workspace_root:
        cli_args.extend(["--workspace-root", spec.workspace_root])
    if spec.readonly:
        cli_args.append("--readonly")
    cli_args.extend(spec.cli_args)

    all_args = [*base_args, *cli_args]
    env = compile_env(spec.env_policy)
    cwd = compile_cwd(spec.cwd_policy, workspace_root=spec.workspace_root)

    return MCPClientConfig(
        server_id=spec.server_id,
        command=command,
        args=all_args,
        env=env,
        cwd=cwd,
    )


def compile_registry(
    registry: "MCPServerRegistry",
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