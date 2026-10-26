"""YAML loader for pack-contributed provider proxy declarations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from kernel.mcp_launch.policy import EnvPolicy, RuntimeKind
from kernel.mcp_launch.server_manifest import _expand_env_value, _parse_cwd_policy
from kernel.proxy_launch.policy import (
    ALL_PROVIDERS,
    ProviderProxyOptions,
    ProviderValueSource,
    ProxyProtocol,
    ProxyServerSpec,
    ProxyStatusSpec,
    ProxyTelemetrySpec,
)

__all__ = ["ProxyServerManifestError", "load_proxy_server_manifest"]

_MAX_TCP_PORT = 65_535


class ProxyServerManifestError(ValueError):
    """Raised when a proxy declaration is invalid."""


def load_proxy_server_manifest(
    path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> list[ProxyServerSpec]:
    """Load ``proxies:`` from a shared pack MCP manifest.

    Missing manifests and manifests without a proxy section are intentionally
    empty so existing MCP-only packs need no migration.

    ``env`` is the mapping ``${VAR}`` references expand against at parse
    time; ``None`` falls back to the ambient process environment
    (standalone callers). The host passes its resolved MCP environment so
    parsing never depends on process-global state.
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
        spec = _parse_proxy_entry(entry, index, path, env)
        if spec.proxy_id in seen_ids:
            raise ProxyServerManifestError(
                f"proxies[{index}] in {path}: duplicate proxy_id {spec.proxy_id!r}"
            )
        seen_ids.add(spec.proxy_id)
        specs.append(spec)
    return specs


def _parse_proxy_entry(  # noqa: C901, PLR0912, PLR0915
    entry: dict[str, Any], index: int, path: Path, env: Mapping[str, str] | None
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
    if ALL_PROVIDERS in providers and len(providers) > 1:
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': '{ALL_PROVIDERS}' must be the only "
            "providers entry — it already covers every manifest provider"
        )
    port_value = entry.get("port")
    if port_value is None:
        raise ProxyServerManifestError(f"proxies[{index}] '{proxy_id}': 'port' must be an integer")
    try:
        port = int(port_value)
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
    provider_options = _parse_provider_overrides(
        entry.get("provider_overrides"),
        providers,
        index=index,
        proxy_id=proxy_id,
        path=path,
        expand_env=env,
    )
    env_raw = entry.get("env") or {}
    if not isinstance(env_raw, dict):
        raise ProxyServerManifestError(f"proxies[{index}] '{proxy_id}': 'env' must be a mapping")
    forced = {str(key): _expand_env_value(value, env) for key, value in env_raw.items()}
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

    status_spec = _parse_status_block(entry.get("status"), index, proxy_id, path)

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
        venv_dir=(_expand_env_value(entry.get("venv_dir"), env) if entry.get("venv_dir") else None),
        command_path=(
            _expand_env_value(entry.get("command_path"), env) if entry.get("command_path") else None
        ),
        script_path=(
            _expand_env_value(entry.get("script_path"), env) if entry.get("script_path") else None
        ),
        npx_package=entry.get("npx_package"),
        uv_directory=(
            _expand_env_value(entry.get("uv_directory"), env) if entry.get("uv_directory") else None
        ),
        cli_args=[_expand_env_value(item, env) for item in cli_args_raw],
        provider_options=provider_options,
        env_policy=EnvPolicy(inherit_parent=True, forced_overrides=forced),
        cwd_policy=cwd_policy,
        readiness_path=readiness_path,
        readiness_timeout_seconds=readiness_timeout_seconds,
        status_spec=status_spec,
        adopt_existing=bool(entry.get("adopt_existing", False)),
        warm=bool(entry.get("warm", True)),
        required=bool(entry.get("required", False)),
    )


def _parse_provider_overrides(
    raw_overrides: Any,
    providers: tuple[str, ...],
    *,
    index: int,
    proxy_id: str,
    path: Path,
    expand_env: Mapping[str, str] | None,
) -> dict[str, ProviderProxyOptions]:
    """Parse the optional per-provider launch overlay mapping.

    Keys must be a subset of the entry's declared ``providers``. Each value
    carries optional ``cli_args`` (literal strings) and ``env`` entries whose
    values are literal strings or a single-key ``value_from: direct_base_url``
    reference the host resolves to the provider's direct endpoint.
    """
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, dict):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': 'provider_overrides' must be a mapping"
        )
    allowed = set(providers)
    options: dict[str, ProviderProxyOptions] = {}
    for raw_provider, raw_body in raw_overrides.items():
        provider = str(raw_provider or "").strip().lower()
        if not provider:
            raise ProxyServerManifestError(
                f"proxies[{index}] '{proxy_id}': provider_overrides keys must be non-empty"
            )
        if provider not in allowed:
            declared = ", ".join(providers)
            raise ProxyServerManifestError(
                f"proxies[{index}] '{proxy_id}': provider_overrides key '{provider}' "
                f"is not among declared providers: {declared}"
            )
        if not isinstance(raw_body, dict):
            raise ProxyServerManifestError(
                f"proxies[{index}] '{proxy_id}': provider_overrides.{provider} must be a mapping"
            )
        cli_args = _parse_override_cli_args(
            raw_body.get("cli_args"), index, proxy_id, provider, expand_env
        )
        env = _parse_override_env(raw_body.get("env"), index, proxy_id, provider, expand_env)
        if not cli_args and not env:
            raise ProxyServerManifestError(
                f"proxies[{index}] '{proxy_id}': provider_overrides.{provider} is empty"
            )
        options[provider] = ProviderProxyOptions(cli_args=cli_args, env=env)
    return options


def _parse_override_cli_args(
    raw_cli_args: Any,
    index: int,
    proxy_id: str,
    provider: str,
    expand_env: Mapping[str, str] | None,
) -> tuple[str, ...]:
    if raw_cli_args is None:
        return ()
    if not isinstance(raw_cli_args, list) or not all(
        isinstance(item, str) for item in raw_cli_args
    ):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': provider_overrides.{provider}.cli_args must be a list"
        )
    return tuple(_expand_env_value(item, expand_env) for item in raw_cli_args if str(item).strip())


def _parse_override_env(
    raw_env: Any,
    index: int,
    proxy_id: str,
    provider: str,
    expand_env: Mapping[str, str] | None,
) -> dict[str, str | ProviderValueSource]:
    if raw_env is None:
        return {}
    if not isinstance(raw_env, dict):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': provider_overrides.{provider}.env must be a mapping"
        )
    env: dict[str, str | ProviderValueSource] = {}
    for raw_key, raw_value in raw_env.items():
        key = str(raw_key or "").strip()
        if not key:
            raise ProxyServerManifestError(
                f"proxies[{index}] '{proxy_id}': provider_overrides.{provider}.env keys "
                "must be non-empty"
            )
        if isinstance(raw_value, dict):
            if set(raw_value) != {"value_from"}:
                raise ProxyServerManifestError(
                    f"proxies[{index}] '{proxy_id}': provider_overrides.{provider}.env.{key} "
                    "must be exactly 'value_from: <source>'"
                )
            try:
                env[key] = ProviderValueSource(str(raw_value["value_from"]))
            except ValueError as exc:
                valid = ", ".join(source.value for source in ProviderValueSource)
                raise ProxyServerManifestError(
                    f"proxies[{index}] '{proxy_id}': unknown value_from for "
                    f"provider_overrides.{provider}.env.{key} (valid: {valid})"
                ) from exc
            continue
        env[key] = _expand_env_value(raw_value, expand_env)
    return env


def _parse_status_block(
    raw_status: Any,
    index: int,
    proxy_id: str,
    path: Path,
) -> ProxyStatusSpec | None:
    """Parse the optional ``status:`` telemetry block on one proxy entry.

    ``status.url_path`` must be a non-empty path starting with ``/``;
    ``status.fields`` must be a list of non-empty strings. An absent or
    ``None`` block is a valid liveness-only proxy (no error).
    """
    if raw_status is None:
        return None
    if not isinstance(raw_status, dict):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}' in {path}: 'status' must be a mapping"
        )
    url_path = raw_status.get("url_path", "/stats")
    if not isinstance(url_path, str) or not url_path.strip():
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': status.url_path must not be empty"
        )
    url_path = url_path.strip()
    if not url_path.startswith("/") or url_path.startswith("//"):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': status.url_path must start with '/'"
        )
    fields_raw = raw_status.get("fields", [])
    if not isinstance(fields_raw, list) or not all(
        isinstance(item, str) and all(part.strip() for part in item.split("."))
        for item in fields_raw
    ):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': status.fields must be a list of strings"
        )
    fields = tuple(item.strip() for item in fields_raw)
    telemetry = _parse_telemetry_block(raw_status.get("telemetry"), index, proxy_id, path, fields)
    return ProxyStatusSpec(url_path=url_path, fields=fields, telemetry=telemetry)


def _parse_telemetry_block(
    raw_telemetry: Any,
    index: int,
    proxy_id: str,
    path: Path,
    status_fields: tuple[str, ...],
) -> ProxyTelemetrySpec | None:
    """Parse the optional ``status.telemetry`` header-counter block.

    Declared shape::

        status:
          telemetry:
            counter_unit: tokens
            savings_field: tokens_saved
            headers:
              x-acme-tokens-before: tokens_before
              x-acme-tokens-saved: tokens_saved

    ``headers`` maps a response header name to the field name it reports
    under. Values must be non-empty strings; a duplicate field name is a
    declaration error because two headers would race to set the same key.
    ``savings_field``, when set, must name a declared field — a footer that
    promises a savings figure the proxy never sends is worse than none.
    """
    if raw_telemetry is None:
        return None
    if not isinstance(raw_telemetry, dict):
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}' in {path}: status.telemetry must be a mapping"
        )
    raw_headers = raw_telemetry.get("headers")
    if raw_headers is None:
        # A telemetry block with no headers declares nothing to lift; keep the
        # savings field declaration inert rather than failing startup over it.
        return None
    if not isinstance(raw_headers, dict) or not raw_headers:
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': status.telemetry.headers must be a "
            "non-empty mapping of response-header name to field name"
        )
    headers: list[tuple[str, str]] = []
    seen_fields: set[str] = set()
    for raw_header, raw_field in raw_headers.items():
        header = str(raw_header or "").strip()
        field_name = str(raw_field or "").strip()
        if not header or not field_name:
            raise ProxyServerManifestError(
                f"proxies[{index}] '{proxy_id}': status.telemetry.headers entries "
                "must map a non-empty header name to a non-empty field name"
            )
        if field_name in seen_fields:
            raise ProxyServerManifestError(
                f"proxies[{index}] '{proxy_id}': status.telemetry.headers declares "
                f"field '{field_name}' more than once"
            )
        seen_fields.add(field_name)
        headers.append((header, field_name))

    savings_field = str(raw_telemetry.get("savings_field") or "").strip()
    if savings_field and savings_field not in seen_fields:
        raise ProxyServerManifestError(
            f"proxies[{index}] '{proxy_id}': status.telemetry.savings_field "
            f"'{savings_field}' is not one of the declared headers' field names"
        )
    counter_unit = str(raw_telemetry.get("counter_unit") or "tokens").strip() or "tokens"
    return ProxyTelemetrySpec(
        headers=tuple(headers),
        savings_field=savings_field,
        counter_unit=counter_unit,
    )
