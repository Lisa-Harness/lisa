"""Typed policy for pack-contributed provider proxy processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from kernel.mcp_launch.policy import CwdPolicy, EnvPolicy, RuntimeKind

__all__ = [
    "ProxyProtocol",
    "ProxyServerRegistry",
    "ProxyServerSpec",
]


class ProxyProtocol(StrEnum):
    """Provider wire protocol exposed by a proxy."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


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
