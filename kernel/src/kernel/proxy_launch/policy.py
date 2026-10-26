"""Typed policy for pack-contributed provider proxy processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from kernel.mcp_launch.policy import CwdPolicy, EnvPolicy, RuntimeKind

__all__ = [
    "ALL_PROVIDERS",
    "ProviderProxyOptions",
    "ProviderValueSource",
    "ProxyProtocol",
    "ProxyServerRegistry",
    "ProxyServerSpec",
    "ProxyStatusSpec",
    "ProxyTelemetrySpec",
]


class ProviderValueSource(StrEnum):
    """Host-resolved reference a provider overlay may inject."""

    DIRECT_BASE_URL = "direct_base_url"


ALL_PROVIDERS = "all"
"""Sentinel ``providers`` value: expand to the host's manifest providers.

A pack declaring ``providers: [all]`` asks the host to route *every* provider
the session can actually call through this proxy. The runtime expands the
sentinel against the session model manifest at compile time, so the pack
never names provider ids — it stays fully decoupled from host configuration.
Expansion is host-side; the kernel never fabricates provider ids itself.
"""


class ProxyProtocol(StrEnum):
    """Provider wire protocol exposed by a proxy."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True, slots=True)
class ProxyTelemetrySpec:
    """Pack-declared per-response telemetry a proxy stamps on headers.

    A proxy that rewrites a request (compression, cache steering, routing)
    knows what it saved, and the only place it can tell the caller is the
    response. ``headers`` names the response headers carrying those counters,
    so the provider gateway can surface them per call without the harness
    knowing any particular proxy's vocabulary.

    Each entry is ``(header_name, field_name)``: ``header_name`` is matched
    case-insensitively against the response headers and the parsed integer is
    reported under ``field_name``. ``field_name`` is deliberately free-form —
    the harness records every declared counter verbatim and renders the ones
    it recognises, so a new proxy metric needs a manifest edit, not a code
    change.

    ``savings_field`` designates which of those counters is *this turn's*
    token savings, the number a footer should show. When unset, no savings
    figure is surfaced (counters are still recorded).

    ``counter_unit`` names the unit those integers are denominated in
    (``tokens`` by default) so a host never assumes.
    """

    headers: tuple[tuple[str, str], ...] = ()
    """``(response header name, field name)`` pairs to lift per call."""

    savings_field: str = ""
    """Field name of the per-turn savings counter. Empty = none surface."""

    counter_unit: str = "tokens"
    """Unit the lifted counters are denominated in."""


@dataclass(frozen=True, slots=True)
class ProxyStatusSpec:
    """Pack-declared live-stats endpoint for one proxy.

    ``url_path`` is GET-fetched from the proxy's loopback endpoint; a 2xx
    JSON body supplies the ``fields`` paths (dotted, e.g.
    ``savings.total_tokens``). Hosts render these beside liveness; an
    absent ``status`` block means liveness-only visibility.

    ``telemetry`` is the per-response counterpart: counters that only exist
    for the duration of one call and cannot be polled after the fact.
    """

    url_path: str = "/stats"
    fields: tuple[str, ...] = ()
    """Dotted JSON paths to surface. Empty = no telemetry values exposed."""

    telemetry: ProxyTelemetrySpec | None = None
    """Per-response header counters, when the proxy stamps any."""


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
    status_spec: ProxyStatusSpec | None = None
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

    def replace_spec(self, proxy_id: str, specs: list[ProxyServerSpec]) -> None:
        """Replace one registered spec with zero or more sibling specs.

        Used by the runtime's ``providers: [all]`` expansion: the wildcard
        spec is removed and per-provider siblings (``<proxy_id>.<provider>``)
        take its place. Re-registering the same ``proxy_id`` here is an
        error — the runtime must not silently duplicate a proxy process.
        """
        self._specs.pop(proxy_id, None)
        self._specs.update({spec.proxy_id: spec for spec in specs})
