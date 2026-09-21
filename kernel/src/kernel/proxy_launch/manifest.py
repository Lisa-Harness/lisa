"""YAML loader for pack-contributed provider proxy declarations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kernel.mcp_launch.policy import EnvPolicy, RuntimeKind
from kernel.mcp_launch.server_manifest import _expand_env_value, _parse_cwd_policy
from kernel.proxy_launch.policy import ProxyProtocol, ProxyServerSpec

__all__ = ["ProxyServerManifestError", "load_proxy_server_manifest"]

_MAX_TCP_PORT = 65_535


class ProxyServerManifestError(ValueError):
    """Raised when a proxy declaration is invalid."""


def load_proxy_server_manifest(path: Path) -> list[ProxyServerSpec]:
    """Load ``proxies:`` from a shared pack MCP manifest.

    Missing manifests and manifests without a proxy section are intentionally
    empty so existing MCP-only packs need no migration.
    """
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProxyServerManifestError(f"Cannot load proxy manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProxyServerManifestError(f"Proxy manifest {path} must be a mapping")
    entries = raw.get("proxies") or []
    if not isinstance(entries, list):
        raise ProxyServerManifestError(f"Proxy manifest {path}: 'proxies' must be a list")

    specs: list[ProxyServerSpec] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ProxyServerManifestError(f"proxies[{index}] in {path} must be a mapping")
        if bool(entry.get("disabled", False)):
            continue
        spec = _parse_proxy_entry(entry, index, path)
        if spec.proxy_id in seen_ids:
            raise ProxyServerManifestError(
                f"proxies[{index}] in {path}: duplicate proxy_id {spec.proxy_id!r}"
            )
        seen_ids.add(spec.proxy_id)
        specs.append(spec)
    return specs


def _parse_proxy_entry(  # noqa: C901, PLR0912, PLR0915
    entry: dict[str, Any], index: int, path: Path
) -> ProxyServerSpec:
    proxy_id = str(entry.get("proxy_id") or "").strip()
    if not proxy_id:
        raise ProxyServerManifestError(f"proxies[{index}] in {path}: 'proxy_id' is required")
    try:
        runtime_kind = RuntimeKind(str(entry.get("runtime_kind") or ""))
    except ValueError as exc:
        valid = ", ".join(kind.value for kind in RuntimeKind)
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': invalid runtime_kind (valid: {valid})"
        ) from exc
    try:
        protocol = ProxyProtocol(str(entry.get("protocol") or ""))
    except ValueError as exc:
        valid = ", ".join(kind.value for kind in ProxyProtocol)
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': invalid protocol (valid: {valid})"
        ) from exc
    providers_raw = entry.get("providers") or []
    if not isinstance(providers_raw, list) or not all(
        isinstance(item, str) for item in providers_raw
    ):
        raise ProxyServerManifestError(f"proxies[{index}] '{proxy_id}': 'providers' must be a list")
    providers = tuple(item.strip().lower() for item in providers_raw if item.strip())
    if not providers:
        raise ProxyServerManifestError(f"proxies[{index}] '{proxy_id}': 'providers' is required")
    try:
        port = int(entry.get("port"))
    except (TypeError, ValueError) as exc:
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': 'port' must be an integer"
        ) from exc
    if not 1 <= port <= _MAX_TCP_PORT:
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': 'port' must be in 1..{_MAX_TCP_PORT}"
        )
    listen = entry.get("listen") or {}
    if not isinstance(listen, dict):
        raise ProxyServerManifestError(f"proxies[{index}] '{proxy_id}': 'listen' must be a mapping")
    port_env_raw = listen.get("port_env")
    port_env = str(port_env_raw).strip() if port_env_raw is not None else None
    if port_env == "":
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': listen.port_env must not be empty"
        )
    base_path = str(listen.get("base_path", "/v1" if protocol is ProxyProtocol.OPENAI else ""))
    if base_path and not base_path.startswith("/"):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': listen.base_path must start with '/'"
        )

    cli_args_raw = entry.get("cli_args") or []
    if not isinstance(cli_args_raw, list):
        raise ProxyServerManifestError(f"proxies[{index}] '{proxy_id}': 'cli_args' must be a list")
    env_raw = entry.get("env") or {}
    if not isinstance(env_raw, dict):
        raise ProxyServerManifestError(f"proxies[{index}] '{proxy_id}': 'env' must be a mapping")
    forced = {str(key): _expand_env_value(value) for key, value in env_raw.items()}
    readiness = entry.get("readiness") or {}
    if not isinstance(readiness, dict):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': 'readiness' must be a mapping"
        )
    readiness_path_raw = readiness.get("url_path", "/readyz")
    readiness_path = None if readiness_path_raw is None else str(readiness_path_raw)
    try:
        readiness_timeout_seconds = float(readiness.get("timeout_seconds", 10.0))
    except (TypeError, ValueError) as exc:
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': readiness.timeout_seconds must be numeric"
        ) from exc
    if readiness_timeout_seconds <= 0:
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': readiness.timeout_seconds must be positive"
        )

    cwd_policy = _parse_cwd_policy(entry.get("cwd"), proxy_id, index, path)
    return ProxyServerSpec(
        proxy_id=proxy_id,
        runtime_kind=runtime_kind,
        protocol=protocol,
        providers=providers,
        port=port,
        port_env=port_env,
        base_path=base_path.rstrip("/"),
        module=entry.get("module"),
        package=entry.get("package"),
        entry_point=entry.get("entry_point"),
        venv_dir=(_expand_env_value(entry.get("venv_dir")) if entry.get("venv_dir") else None),
        command_path=(
            _expand_env_value(entry.get("command_path")) if entry.get("command_path") else None
        ),
        script_path=(
            _expand_env_value(entry.get("script_path")) if entry.get("script_path") else None
        ),
        npx_package=entry.get("npx_package"),
        uv_directory=(
            _expand_env_value(entry.get("uv_directory")) if entry.get("uv_directory") else None
        ),
        cli_args=[_expand_env_value(item) for item in cli_args_raw],
        env_policy=EnvPolicy(inherit_parent=True, forced_overrides=forced),
        cwd_policy=cwd_policy,
        readiness_path=readiness_path,
        readiness_timeout_seconds=readiness_timeout_seconds,
        adopt_existing=bool(entry.get("adopt_existing", False)),
        warm=bool(entry.get("warm", True)),
        required=bool(entry.get("required", False)),
    )
