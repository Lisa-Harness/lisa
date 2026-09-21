"""Compile declarative proxy specs into host-runnable process configs."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from kernel.mcp_launch.compiler import compile_cwd, compile_env
from kernel.mcp_launch.policy import RuntimeKind
from kernel.proxy_launch.policy import (
    ProviderValueSource,
    ProxyServerRegistry,
    ProxyServerSpec,
)

logger = logging.getLogger(__name__)

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
) -> tuple[str, list[str]] | None:
    """Build (command, args) for one runtime kind.

    Returns ``None`` for a BINARY spec whose ``command_path`` is empty —
    the optional third-party binary is not installed (e.g. an unexpanded
    ``${SOME_BIN}`` env reference), so the caller skips the proxy instead
    of spawning.
    """
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
        # An empty command_path means the binary is not installed. Pack
        # manifests routinely point BINARY proxies at an opt-in third-party
        # binary via ``command_path: ${SOME_BIN}`` and expect an unset var
        # to degrade to "this proxy is absent" (the headroom pack contract).
        # Raising here aborts the WHOLE startup for what is an optional
        # capability, so signal a skip instead.
        if not spec.command_path:
            return None
        return spec.command_path, []
    raise ValueError(
        f"Proxy '{spec.proxy_id}' does not support runtime_kind={spec.runtime_kind.value!r}"
    )


def compile_proxy_config(
    spec: ProxyServerSpec,
    *,
    provider_base_urls: dict[str, str] | None = None,
) -> ProxyClientConfig | None:
    """Compile a resolved proxy spec into a local process config.

    Launch options are compiled from the spec's global ``cli_args`` / ``env``
    first and then each provider overlay in ``providers`` order, so provider
    switches land after global ones. ``ProviderValueSource.DIRECT_BASE_URL``
    env references resolve through ``provider_base_urls`` — the host supplies
    the providers' *direct* endpoints, never proxy routes, so a proxy cannot
    be configured to forward to itself. Two overlays writing the same env
    key with different resolved values is a declaration conflict and raises.

    Returns ``None`` when the proxy cannot spawn because its optional binary
    is not installed (a BINARY spec whose ``command_path`` is empty after
    ``${VAR}`` expansion). Callers must filter ``None`` —
    :func:`compile_proxy_registry` does this and logs a warning.
    """
    resolved_command = _compile_command(spec)
    if resolved_command is None:
        logger.warning(
            "Skipping proxy %r: runtime_kind=binary with empty command_path "
            "(optional binary not installed; check its ${VAR} env reference)",
            spec.proxy_id,
        )
        return None
    command, base_args = resolved_command
    env = compile_env(spec.env_policy)
    if spec.port_env:
        env[spec.port_env] = str(spec.port)
    merged_args = [*base_args, *spec.cli_args]
    for provider in spec.providers:
        overlay = spec.provider_options.get(provider)
        if overlay is None:
            continue
        merged_args.extend(overlay.cli_args)
        for key, value in overlay.env.items():
            resolved = _resolve_env_value(spec, provider, key, value, provider_base_urls)
            previous = env.get(key)
            if previous is not None and previous != resolved:
                raise ValueError(
                    f"Proxy '{spec.proxy_id}': env '{key}' receives conflicting values "
                    f"across providers ({previous!r} vs {resolved!r}); split into separate proxies"
                )
            env[key] = resolved
    base_url = f"http://127.0.0.1:{spec.port}{spec.base_path}"
    return ProxyClientConfig(
        proxy_id=spec.proxy_id,
        command=command,
        args=merged_args,
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


def _resolve_env_value(
    spec: ProxyServerSpec,
    provider: str,
    key: str,
    value: str | ProviderValueSource,
    provider_base_urls: dict[str, str] | None,
) -> str:
    """Resolve one overlay env value to the literal string a child process sees."""
    if not isinstance(value, ProviderValueSource):
        return value
    if value is not ProviderValueSource.DIRECT_BASE_URL:
        raise ValueError(
            f"Proxy '{spec.proxy_id}': unsupported env source {value.value!r} for "
            f"provider_overrides.{provider}.env.{key}"
        )
    base_urls = provider_base_urls or {}
    resolved = base_urls.get(provider)
    if not resolved:
        raise ValueError(
            f"Proxy '{spec.proxy_id}': no direct base URL available for provider "
            f"'{provider}' (provider_overrides.{provider}.env.{key})"
        )
    return resolved


def compile_proxy_registry(
    registry: ProxyServerRegistry,
    *,
    workspace_root: str | None = None,
    provider_base_urls: dict[str, str] | None = None,
) -> list[ProxyClientConfig]:
    """Compile every approved proxy after host runtime paths are resolved.

    A non-required proxy whose launch spec does not resolve (typically a
    ``${VAR}`` command_path expanding to empty because the optional tool is
    not installed) is skipped rather than aborting the whole compile — the
    proxy pool already treats a non-required proxy that fails to start as a
    warning, and an opt-in pack should never brick host startup for an
    operator who has not installed its external tool.
    """
    configs: list[ProxyClientConfig] = []
    for spec in registry.all_specs():
        config = compile_proxy_config(
            replace(spec, workspace_root=workspace_root),
            provider_base_urls=provider_base_urls,
        )
        if config is not None:
            configs.append(config)
    return configs
