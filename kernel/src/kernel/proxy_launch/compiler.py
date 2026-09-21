"""Compile declarative proxy specs into host-runnable process configs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from kernel.mcp_launch.compiler import compile_cwd, compile_env
from kernel.mcp_launch.policy import RuntimeKind
from kernel.proxy_launch.policy import ProxyServerRegistry, ProxyServerSpec

__all__ = ["ProxyClientConfig", "compile_proxy_config", "compile_proxy_registry"]


@dataclass(frozen=True, slots=True)
class ProxyClientConfig:
    """A compiled local proxy process plus its provider routing contract."""

    proxy_id: str
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | Path | None
    protocol: str
    providers: tuple[str, ...]
    port: int
    base_url: str
    readiness_path: str | None
    readiness_timeout_seconds: float
    adopt_existing: bool
    warm: bool
    required: bool


def _compile_command(  # noqa: C901, PLR0911, PLR0912
    spec: ProxyServerSpec,
) -> tuple[str, list[str]]:
    if spec.runtime_kind is RuntimeKind.PYTHON:
        if not spec.module:
            raise ValueError(f"PYTHON runtime requires 'module' for proxy '{spec.proxy_id}'")
        return "python", ["-m", spec.module]
    if spec.runtime_kind is RuntimeKind.UV:
        if not spec.package or not spec.module:
            raise ValueError(
                f"UV runtime requires 'package' and 'module' for proxy '{spec.proxy_id}'"
            )
        args = ["run"]
        if spec.uv_directory:
            args.extend(["--directory", spec.uv_directory])
        args.extend(["--package", spec.package, "python", "-m", spec.module])
        return "uv", args
    if spec.runtime_kind is RuntimeKind.VENV_BIN:
        if not spec.venv_dir:
            raise ValueError(f"VENV_BIN runtime requires 'venv_dir' for proxy '{spec.proxy_id}'")
        entry_point = spec.entry_point or spec.proxy_id
        if sys.platform == "win32":
            return str(Path(spec.venv_dir) / "Scripts" / f"{entry_point}.exe"), []
        return str(Path(spec.venv_dir) / "bin" / entry_point), []
    if spec.runtime_kind is RuntimeKind.NODE:
        if not spec.command_path or not spec.script_path:
            raise ValueError(
                f"NODE runtime requires command_path and script_path for proxy '{spec.proxy_id}'"
            )
        return spec.command_path, [spec.script_path]
    if spec.runtime_kind is RuntimeKind.NPX:
        if not spec.npx_package:
            raise ValueError(f"NPX runtime requires 'npx_package' for proxy '{spec.proxy_id}'")
        return "npx", ["-y", spec.npx_package]
    if spec.runtime_kind is RuntimeKind.BINARY:
        if not spec.command_path:
            raise ValueError(f"BINARY runtime requires 'command_path' for proxy '{spec.proxy_id}'")
        return spec.command_path, []
    raise ValueError(
        f"Proxy '{spec.proxy_id}' does not support runtime_kind={spec.runtime_kind.value!r}"
    )


def compile_proxy_config(spec: ProxyServerSpec) -> ProxyClientConfig:
    """Compile a resolved proxy spec into a local process config."""
    command, base_args = _compile_command(spec)
    env = compile_env(spec.env_policy)
    if spec.port_env:
        env[spec.port_env] = str(spec.port)
    base_url = f"http://127.0.0.1:{spec.port}{spec.base_path}"
    return ProxyClientConfig(
        proxy_id=spec.proxy_id,
        command=command,
        args=[*base_args, *spec.cli_args],
        env=env,
        cwd=compile_cwd(spec.cwd_policy, workspace_root=spec.workspace_root),
        protocol=spec.protocol.value,
        providers=spec.providers,
        port=spec.port,
        base_url=base_url,
        readiness_path=spec.readiness_path,
        readiness_timeout_seconds=spec.readiness_timeout_seconds,
        adopt_existing=spec.adopt_existing,
        warm=spec.warm,
        required=spec.required,
    )


def compile_proxy_registry(
    registry: ProxyServerRegistry,
    *,
    workspace_root: str | None = None,
) -> list[ProxyClientConfig]:
    """Compile every approved proxy after host runtime paths are resolved."""
    return [
        compile_proxy_config(replace(spec, workspace_root=workspace_root))
        for spec in registry.all_specs()
    ]
