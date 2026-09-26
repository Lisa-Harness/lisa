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

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from kernel.mcp_launch.policy import (
    AuthKind,
    CwdPolicy,
    EnvPolicy,
    MCPServerSpec,
    RuntimeKind,
)
from kernel.tools.mcp_client import MCPClientConfig

if TYPE_CHECKING:
    from kernel.mcp_launch.policy import MCPServerRegistry

logger = logging.getLogger(__name__)

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


def _compile_python_command(spec: MCPServerSpec) -> tuple[str, list[str]]:
    """Compile a PYTHON spec: ``python -m <module>``."""
    if not spec.module:
        raise ValueError(f"PYTHON runtime requires 'module' for server '{spec.server_id}'")
    return "python", ["-m", spec.module]


def _compile_uv_command(spec: MCPServerSpec) -> tuple[str, list[str]]:
    """Compile a UV spec: ``uv run --package <pkg> python -m <module>``."""
    if not spec.package or not spec.module:
        raise ValueError(
            f"UV runtime requires 'package' and 'module' for server '{spec.server_id}'"
        )
    args: list[str] = ["run"]
    if spec.uv_directory:
        args.extend(["--directory", spec.uv_directory])
    args.extend(["--package", spec.package, "python", "-m", spec.module])
    return "uv", args


def _compile_venv_bin_command(spec: MCPServerSpec) -> tuple[str, list[str]]:
    """Compile a VENV_BIN spec: the venv's entry-point binary path."""
    if not spec.venv_dir:
        raise ValueError(f"VENV_BIN runtime requires 'venv_dir' for server '{spec.server_id}'")
    entry_point = spec.entry_point or spec.server_id
    if sys.platform == "win32":
        command_path = str(Path(spec.venv_dir) / "Scripts" / f"{entry_point}.exe")
    else:
        command_path = str(Path(spec.venv_dir) / "bin" / entry_point)
    return command_path, []


def _compile_node_command(spec: MCPServerSpec) -> tuple[str, list[str]]:
    """Compile a NODE spec: ``<command_path> <script_path>``."""
    if not spec.command_path or not spec.script_path:
        raise ValueError(
            f"NODE runtime requires 'command_path' and 'script_path' for server '{spec.server_id}'"
        )
    return spec.command_path, [spec.script_path]


def _compile_npx_command(spec: MCPServerSpec) -> tuple[str, list[str]]:
    """Compile an NPX spec: ``npx -y <package>``."""
    if not spec.npx_package:
        raise ValueError(f"NPX runtime requires 'npx_package' for server '{spec.server_id}'")
    return "npx", ["-y", spec.npx_package]


def _compile_binary_command(spec: MCPServerSpec) -> tuple[str, list[str]] | None:
    """Compile a BINARY spec, or ``None`` when the optional binary is absent.

    An empty command_path means the binary is not installed. Pack
    manifests routinely point BINARY servers/proxies at an opt-in
    third-party binary via ``command_path: ${SOME_BIN}`` and expect an
    unset var to degrade to "this server is absent" (the headroom
    pack contract). Raising here aborts the WHOLE startup for what is
    an optional capability, so signal a skip instead.
    """
    if not spec.command_path:
        return None
    return spec.command_path, []


def compile_command(spec: MCPServerSpec) -> tuple[str, list[str]] | None:
    """Build (command, args) for one runtime kind.

    Returns the base command and args before standard CLI contract args
    are appended. Returns ``None`` for a BINARY spec whose
    ``command_path`` is empty — the optional third-party binary is not
    installed (e.g. an unexpanded ``${SOME_BIN}`` env reference), so the
    caller skips the entry instead of spawning.
    """
    compiled: tuple[str, list[str]] | None
    if spec.runtime_kind == RuntimeKind.PYTHON:
        compiled = _compile_python_command(spec)
    elif spec.runtime_kind == RuntimeKind.UV:
        compiled = _compile_uv_command(spec)
    elif spec.runtime_kind == RuntimeKind.VENV_BIN:
        compiled = _compile_venv_bin_command(spec)
    elif spec.runtime_kind == RuntimeKind.NODE:
        compiled = _compile_node_command(spec)
    elif spec.runtime_kind == RuntimeKind.NPX:
        compiled = _compile_npx_command(spec)
    elif spec.runtime_kind == RuntimeKind.BINARY:
        compiled = _compile_binary_command(spec)
    elif spec.runtime_kind in (RuntimeKind.HTTP, RuntimeKind.IN_PROCESS):
        # HTTP (validated url) and IN_PROCESS (memory transport;
        # command/args unused but kept uniform) both spawn no subprocess
        # command.
        if spec.runtime_kind == RuntimeKind.HTTP and not spec.url:
            raise ValueError(f"HTTP runtime requires 'url' for server '{spec.server_id}'")
        compiled = ("", [])
    else:
        raise ValueError(
            f"Unknown runtime kind '{spec.runtime_kind}' for server '{spec.server_id}'"
        )
    return compiled


def compile_mcp_config(spec: MCPServerSpec) -> MCPClientConfig | None:
    """Compile one approved MCPServerSpec into a concrete MCPClientConfig.

    This is the only function that produces MCPClientConfig entries for
    MCPToolPool. The host calls this after registry approval.

    Returns ``None`` when the spec cannot spawn a subprocess because its
    optional binary is not installed (a BINARY spec whose ``command_path``
    is empty after ``${VAR}`` expansion). Callers must filter ``None`` —
    :func:`compile_registry` does this and logs a warning.
    """
    if spec.runtime_kind is RuntimeKind.HTTP:
        _validate_http_auth(spec)

    # Resolve the effective data directory, applying an optional namespaced
    # subdir. A spec with data_dir=None (servers managing their own storage)
    # compiles without any data-dir injection.
    if spec.data_dir:
        resolved_data_dir = spec.data_dir
        if spec.data_dir_subdir:
            resolved_data_dir = str(Path(spec.data_dir) / spec.data_dir_subdir)
    else:
        resolved_data_dir = None

    resolved_command = compile_command(spec)
    if resolved_command is None:
        logger.warning(
            "Skipping MCP server %r: runtime_kind=binary with empty command_path "
            "(optional binary not installed; check its ${VAR} env reference)",
            spec.server_id,
        )
        return None
    command, base_args = resolved_command

    # Build standard CLI contract args. data-dir is injected only when the
    # spec names a flag for it (data_dir_arg is not None) and a data dir was
    # resolved. First-party servers keep data_dir_arg="--data-dir" as the
    # default, preserving the original behavior. Third-party servers that
    # manage their own storage set data_dir_arg=None.
    cli_args: list[str] = []
    if spec.data_dir_arg is not None and resolved_data_dir is not None:
        cli_args.extend([spec.data_dir_arg, resolved_data_dir])
    if spec.workspace_root and spec.workspace_root_arg is not None:
        cli_args.extend([spec.workspace_root_arg, spec.workspace_root])
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
    # Treat a present-but-empty value as unset: an empty ``LISA_MCP_LOG_PATH``
    # would otherwise fall through to ``configure_logging``'s cwd fallback
    # and drop a ``<server_name>.jsonl`` in the subprocess's cwd (the
    # project root when the host launched from there) — the 2026-08-25
    # stray-jsonl-files-in-root symptom.
    # In-process servers skip log-path injection: there is no subprocess
    # and no stderr to leak. The host process already configures logging.
    if (
        resolved_data_dir is not None
        and not env.get("LISA_MCP_LOG_PATH")
        and spec.runtime_kind is not RuntimeKind.IN_PROCESS
    ):
        from kernel.layout import LOGS_DIR, MCP_LOGS_SUBDIR

        env["LISA_MCP_LOG_PATH"] = str(
            Path(resolved_data_dir) / LOGS_DIR / MCP_LOGS_SUBDIR / f"{spec.server_id}.jsonl"
        )

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
        auth=spec.auth,
        warm=spec.warm,
        required=spec.required,
        aliases=spec.aliases,
        runtime_kind=spec.runtime_kind.value,
    )


def _is_loopback_host(host: str) -> bool:
    """True for localhost / loopback hosts (no credential required)."""
    host = host.strip().lower()
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}:
        return True
    return host.startswith("127.")


def _validate_http_auth(spec: MCPServerSpec) -> None:
    """Enforce the network-auth gate.

    A non-loopback HTTP server **must** declare a usable ``auth:`` block, or
    an explicit ``auth: {type: none}`` whose host is allowlisted. A server
    that reaches a remote host without a credential is a config error, not a
    silent fallback to anonymous — reject it at compile time.
    """
    url = spec.url or ""
    host = url.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
    if _is_loopback_host(host):
        return

    if spec.auth.is_authenticated():
        return
    if spec.auth.kind is AuthKind.NONE:
        # Explicit opt-out is honored only for loopback hosts; a remote host
        # must carry a credential (bearer or oauth2-pkce).
        raise ValueError(
            f"HTTP MCP server '{spec.server_id}' at {url} targets a non-loopback "
            "host but declares no usable auth. Declare an 'auth:' block "
            "(type: bearer | oauth2-pkce) to proceed."
        )
    raise ValueError(
        f"HTTP MCP server '{spec.server_id}' at {url} targets a non-loopback "
        f"host with incomplete auth (kind={spec.auth.kind.value}). Provide the "
        "required credential fields."
    )


def compile_registry(
    registry: MCPServerRegistry,
    *,
    data_dir: str,
    workspace_root: str | None = None,
) -> list[MCPClientConfig]:
    """Compile all approved server specs from a registry into MCPClientConfig entries.

    A non-required server whose launch spec does not resolve (typically a
    ``${VAR}`` command_path expanding to empty because the optional tool is
    not installed) is skipped with a warning rather than aborting the compile.
    This mirrors the documented contract on optional servers ("unset -> cold
    connect fails cleanly") and the proxy pool's existing behavior for
    non-required proxies, so an opt-in pack can never brick host startup for
    an operator who has not installed its external tool.
    """
    configs: list[MCPClientConfig] = []
    for spec in registry.all_specs():
        # Inject host-provided args (slots dataclass — use replace, not __dict__)
        resolved_spec = replace(spec, data_dir=data_dir, workspace_root=workspace_root)
        config = compile_mcp_config(resolved_spec)
        if config is not None:
            configs.append(config)
    return configs
