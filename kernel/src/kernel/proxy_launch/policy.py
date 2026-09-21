"""Typed policy for pack-contributed provider proxy processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from kernel.mcp_launch.policy import CwdPolicy, EnvPolicy, RuntimeKind

__all__ = [
    "ProviderProxyOptions",
    "ProviderValueSource",
    "ProxyProtocol",
    "ProxyServerRegistry",
    "ProxyServerSpec",
]


class ProviderValueSource(StrEnum):
    """Host-resolved reference a provider overlay may inject."""

    DIRECT_BASE_URL = "direct_base_url"


class ProxyProtocol(StrEnum):
    """Provider wire protocol exposed by a proxy."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True, slots=True)
class ProviderProxyOptions:
    """Provider-specific launch options layered onto a proxy's global ones.

    The host compiles a proxy by applying the global ``cli_args`` / ``env``
    first and then each provider overlay in the spec's ``providers`` order.
    ``env`` values are either literal strings or a ``ProviderValueSource``
    the host resolves to the provider's direct endpoint (never a proxy
    route, so a proxy can never be wired to forward to itself).
    """

    cli_args: tuple[str, ...] = ()
    env: dict[str, str | ProviderValueSource] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProxyServerSpec:
    """One approved provider-facing proxy process.

    A pack declares how to launch the proxy and which provider gateway ids may
    route through its loopback endpoint. The host injects runtime paths before
    compilation, starts warm proxies before constructing provider clients, and
    exposes a route only after readiness succeeds.
    """

    proxy_id: str
    runtime_kind: RuntimeKind
    protocol: ProxyProtocol
    providers: tuple[str, ...]
    port: int
    port_env: str | None = None
    base_path: str = "/v1"

    module: str | None = None
    package: str | None = None
    entry_point: str | None = None
    venv_dir: str | None = None
    command_path: str | None = None
    script_path: str | None = None
    npx_package: str | None = None
    uv_directory: str | None = None
    cli_args: list[str] = field(default_factory=list)
    provider_options: dict[str, ProviderProxyOptions] = field(default_factory=dict)

    env_policy: EnvPolicy = field(default_factory=EnvPolicy)
    cwd_policy: CwdPolicy = field(default_factory=CwdPolicy)

    readiness_path: str | None = "/readyz"
    readiness_timeout_seconds: float = 10.0
    adopt_existing: bool = False
    warm: bool = True
    required: bool = False
    disabled: bool = False

    workspace_root: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyServerRegistry:
    """Registry of approved proxy specs a host may launch."""

    _specs: dict[str, ProxyServerSpec] = field(default_factory=dict)

    def register(self, spec: ProxyServerSpec) -> None:
        if spec.proxy_id in self._specs:
            raise ValueError(f"Proxy server '{spec.proxy_id}' is already registered")
        self._specs[spec.proxy_id] = spec

    def all_specs(self) -> list[ProxyServerSpec]:
        return list(self._specs.values())
