"""Host-side provider-gateway provider.

The kernel's single live completion path routes through
``ContextSeams.provider_gateways`` (a :class:`ProviderGatewayProvider`). This module
turns the pack's entry-point gateway factory (which returns gateway
*classes*) into concrete, credential-bound gateway **instances** keyed by
provider name.

Environment variables are resolved here (the host owns credentials), and the
constructed gateway instances are cached per provider for the runtime's
lifetime so the underlying SDK clients (which the existing
local client factories cache on :class:`~runtime.runtime.HostRuntime`) are reused
across turns. No gateway is constructed eagerly: instances are created lazily on the first
``gateway_for(provider)`` call and cached.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI
from sdk.providers.openai_gateway import create_azure_openai_client

from runtime.provider_endpoints import direct_provider_base_url
from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)

# Optional OpenRouter attribution/telemetry headers. OpenRouter is
# OpenAI-compatible; these opt the host onto app rankings.
LISA_OPENROUTER_API_KEY_ENV = "LISA_OPENROUTER_API_KEY"
LISA_OPENROUTER_BASE_URL_ENV = "LISA_OPENROUTER_BASE_URL"
LISA_OPENROUTER_REFERER_ENV = "LISA_OPENROUTER_REFERER"
LISA_OPENROUTER_TITLE_ENV = "LISA_OPENROUTER_TITLE"


class _GatewayClassProvider(Protocol):
    """The shape of a pack's gateway-provider entry point.

    Returns the gateway *class* for one provider name, or ``None`` when the
    pack does not ship a backend for that provider. The host is responsible
    for instantiating the class with credentials/transport.
    """

    def gateway_for(self, provider: str) -> type | None: ...

    def proxy_protocol_for(self, provider: str) -> str | None:
        """Return the local-proxy wire protocol serving ``provider``, or None."""
        ...

    def direct_base_url_for(self, provider: str) -> str | None:
        """Return the provider's built-in default API endpoint, or None."""
        ...


class HostGatewayProvider:
    """Resolve a provider name to a constructed, credential-bound gateway.

    Wraps the SDK/Pack ``gateway_for(provider) -> class`` factory and is a
    :class:`ProviderGatewayProvider` itself: it returns a ready
    ``ProviderGateway`` **instance** for the kernel to call at the model-call
    site. Instances are constructed lazily and cached on this object for the
    host's lifetime, so the wrapped SDK client (cached on the runtime by the
    provider factories) is shared across turns.
    """

    def __init__(
        self,
        runtime: HostRuntime,
        discovered_provider: _GatewayClassProvider,
    ) -> None:
        self._runtime = runtime
        self._discovered = discovered_provider
        self._instances: dict[str, Any] = {}

    # ── ProviderGatewayProvider surface ─────────────────────────────
    def gateway_for(self, provider: str) -> Any | None:
        """Return the constructed gateway for ``provider``, or ``None``.

        The kernel calls this at the model-call site with
        ``model_def.provider``. ``None`` means no gateway is registered for
        the provider and the kernel raises its documented "no gateway"
        error.
        """

        normalized = (provider or "").strip().lower()
        if not normalized:
            return None
        if normalized not in self._instances:
            instance = self._build(normalized)
            if instance is None:
                return None
            self._instances[normalized] = instance
        return self._instances[normalized]

    def proxy_protocol_for(self, provider: str) -> str | None:
        """Delegate to the discovered provider: proxy routing is gateway knowledge."""

        return self._discovered.proxy_protocol_for(provider)

    def direct_base_url_for(self, provider: str) -> str | None:
        """Delegate to the discovered provider: default endpoints are gateway knowledge."""

        resolve = getattr(self._discovered, "direct_base_url_for", None)
        if not callable(resolve):
            return None
        return resolve(provider)

    # ── Construction ────────────────────────────────────────────────
    def _build(self, provider: str) -> Any | None:  # noqa: PLR0911
        """Construct the gateway instance for one provider (uncached)."""

        gateway_cls = self._discovered.gateway_for(provider)
        if gateway_cls is None:
            return None

        runtime = self._runtime
        try:
            if provider == "azure_openai":
                return gateway_cls(
                    complete_client_factory=lambda: _azure_client_instance(runtime),
                )
            if provider == "anthropic":
                # AnthropicGateway holds the raw runtime-cached httpx client;
                # auth (x-api-key) and anthropic-version are injected
                # per-request by the gateway from these values. The host's
                # prompt_cache_enabled toggle (``LISA_PROMPT_CACHE_ENABLED``)
                # controls whether typed system segments get an ephemeral
                # cache breakpoint on the stable prefix.
                return gateway_cls(
                    client_factory=lambda: _anthropic_http_client_instance(runtime),
                    api_key=_env("LISA_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
                    api_version=_env("LISA_ANTHROPIC_VERSION") or "2023-06-01",
                    prompt_cache_enabled=bool(
                        getattr(runtime.config, "prompt_cache_enabled", True)
                    ),
                )
            if provider == "google":
                # Native REST path (chosen over the legacy OpenAI-compat wire
                # shape). The gateway takes concrete credentials, not a client
                # factory.
                return gateway_cls(
                    api_key=_env("LISA_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
                    base_url=_env("LISA_GOOGLE_API_BASE_URL", "GOOGLE_API_BASE_URL")
                    or "https://generativelanguage.googleapis.com",
                )
            if provider == "aws":
                # Let boto3's default credential chain handle instances,
                # profiles, and env credentials when explicit values are absent.
                return gateway_cls(
                    region_name=_env("LISA_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
                    or "us-east-1",
                    aws_access_key_id=_env("LISA_AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID") or None,
                    aws_secret_access_key=_env(
                        "LISA_AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"
                    )
                    or None,
                    aws_session_token=_env("LISA_AWS_SESSION_TOKEN", "AWS_SESSION_TOKEN") or None,
                )
            if provider == "openrouter":
                return gateway_cls(client_factory=_openrouter_client_factory(runtime))
            if provider in {"openai", "ollama"}:
                return gateway_cls(
                    client_factory=_openai_compatible_client_factory(runtime, provider)
                )
        except Exception as exc:  # pylint: disable=broad-except
            # Construction must never break the turn loop's discovery: a
            # misconfigured provider (missing key, bad base URL) is logged once
            # and left "no gateway", matching the documented error path.
            logger.warning("provider gateway %r failed to construct: %s", provider, exc)
            return None
        return None


def _env(*names: str) -> str:
    """Return the first non-empty environment value among ``names``."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _openai_compatible_client_factory(runtime: HostRuntime, provider: str) -> Callable[[], Any]:
    """Return a client factory for the OpenAI-compatible (openai/ollama) gateways."""

    if provider == "openai":
        return lambda: _openai_client_instance(runtime)
    return lambda: _ollama_client_instance(runtime)


def _azure_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Resolve host credentials and cache the SDK-owned Azure v1 client."""

    existing_client = runtime.providers.azure_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client
    endpoint = _env("LISA_AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
    api_key = _env("LISA_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI is not configured. Set LISA_AZURE_OPENAI_ENDPOINT and "
            "LISA_AZURE_OPENAI_API_KEY (or AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)."
        )
    client = create_azure_openai_client(endpoint=endpoint, api_key=api_key)
    runtime.providers.azure_client = client
    return client


def _prune_dead_proxy_client(
    runtime: HostRuntime,
    provider: str,
    protocol: str,
) -> None:
    """Drop a cached client whose proxy route died since it was built.

    Runs *before* each factory's cache short-circuit: a cached
    ``AsyncOpenAI``/``httpx`` client is bound to the base URL it was built
    with, so returning it without re-checking the route leaves the host
    dialing a dead loopback port for the rest of the process lifetime (the
    2026-10-26 headroom death). ``proxy_route_for`` performs the liveness
    check and drops the stale route; this compares that with the view the
    client was resolved against.

    Args:
        runtime: The host runtime owning the proxy pool and client cache.
        provider: Provider gateway id.
        protocol: Wire protocol the client speaks.
    """
    previous_route = runtime.resources.proxy_routes.get(provider)
    if previous_route is None:
        return
    if runtime.resources.proxy_route_for(provider, protocol) is not None:
        return
    _invalidate_client_for_lost_route(runtime, provider, previous_route)


def _openai_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Return the cached public OpenAI client."""

    _prune_dead_proxy_client(runtime, "openai", "openai")
    existing_client = runtime.providers.openai_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client
    api_key = _env("LISA_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI is not configured. Set LISA_OPENAI_API_KEY (or OPENAI_API_KEY).")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_proxy_base_url(runtime, "openai", direct_provider_base_url(runtime, "openai")),
    )
    runtime.providers.openai_client = client
    return client


def _ollama_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Return the cached Ollama OpenAI-compatible client."""

    _prune_dead_proxy_client(runtime, "ollama", "openai")
    existing_client = runtime.providers.ollama_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client
    direct_base_url = direct_provider_base_url(runtime, "ollama")
    base_url = _proxy_base_url(runtime, "ollama", direct_base_url)
    client = AsyncOpenAI(
        api_key=_env("LISA_OLLAMA_API_KEY", "OLLAMA_API_KEY") or "ollama",
        base_url=base_url,
    )
    runtime.providers.ollama_client = client
    return client


def _anthropic_http_client_instance(runtime: HostRuntime) -> httpx.AsyncClient:
    """Return the cached raw Anthropic HTTP client."""

    _prune_dead_proxy_client(runtime, "anthropic", "anthropic")
    existing_client = runtime.providers.anthropic_http_client
    if isinstance(existing_client, httpx.AsyncClient):
        return existing_client
    direct_base_url = direct_provider_base_url(runtime, "anthropic")
    route_url = _ready_proxy_base_url(runtime, "anthropic", "anthropic")
    base_url = route_url or direct_base_url
    client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=60.0)
    runtime.providers.anthropic_http_client = client
    return client


#: Provider id → the ``ProviderRegistry`` slot holding its cached client.
#: Every factory below resolves its route through ``_ready_proxy_base_url``,
#: which invalidates the matching slot when a route disappears mid-session.
_CLIENT_SLOT_BY_PROVIDER: dict[str, str] = {
    "openai": "openai_client",
    "ollama": "ollama_client",
    "anthropic": "anthropic_http_client",
    "openrouter": "openrouter_client",
}


def _invalidate_client_for_lost_route(
    runtime: HostRuntime,
    provider: str,
    previous_route: Any,
) -> None:
    """Drop a provider's cached client when its proxy route just disappeared.

    A cached ``AsyncOpenAI``/``httpx`` client is bound to the base URL it was
    constructed with, so once the owning proxy process exits the client keeps
    dialing a dead loopback port for the rest of the process lifetime — the
    2026-10-26 headroom death, where the next model call hung until the
    stream idle watchdog fired. Clearing the slot makes the next factory call
    re-resolve against the direct endpoint.

    Only a route that was present on the previous read and is gone now
    triggers the reset: a provider that never had a proxy keeps its cache
    (there is nothing stale to invalidate), and repeating the reset on every
    call would discard a healthy direct client.

    Args:
        runtime: The host runtime holding the provider client cache.
        provider: Provider gateway id whose route was resolved.
        previous_route: The route read before resolution (``None`` when the
            provider never had one).
    """
    if previous_route is None:
        return
    slot = _CLIENT_SLOT_BY_PROVIDER.get(provider.strip().lower())
    if slot is None:
        return
    if getattr(runtime.providers, slot, None) is None:
        return
    setattr(runtime.providers, slot, None)
    logger.warning(
        "provider %r lost its proxy route mid-session; dropping the cached "
        "client so the next call re-resolves against the direct endpoint",
        provider,
    )


def _ready_proxy_base_url(
    runtime: HostRuntime,
    provider: str,
    protocol: str,
) -> str | None:
    """Return the ready proxy base URL for a provider, or ``None``.

    Single route-resolution seam for every provider client factory: it is the
    one place that compares the pre-resolution route with the resolved one, so
    a route loss invalidates the cached client for *every* proxied provider
    (OpenAI, Ollama, OpenRouter, Anthropic) instead of the two the first pass
    covered.

    Args:
        runtime: The host runtime owning the proxy pool and client cache.
        provider: Provider gateway id.
        protocol: Wire protocol the client speaks (``"openai"`` or
            ``"anthropic"``).

    Returns:
        The proxy's base URL when a ready same-protocol route exists,
        ``None`` when the provider must go direct.
    """
    previous_route = runtime.resources.proxy_routes.get(provider)
    route = runtime.resources.proxy_route_for(provider, protocol)
    if route is None:
        _invalidate_client_for_lost_route(runtime, provider, previous_route)
        return None
    return str(route.base_url)


def _proxy_base_url(runtime: HostRuntime, provider: str, direct_base_url: str) -> str:
    """Return a ready OpenAI-wire proxy route or the provider's direct URL.

    A dead proxy invalidates the provider's cached client (when the route
    just dropped): the next factory call rebuilds the client against the
    fallback base URL, so the turn proceeds direct instead of hanging on the
    dead loopback port (the 2026-10-26 headroom death).
    """
    route_url = _ready_proxy_base_url(runtime, provider, "openai")
    if route_url is None:
        return direct_base_url
    return f"{route_url.rstrip('/')}/"


def _openrouter_client_factory(runtime: HostRuntime) -> Callable[[], Any]:
    """Return a client factory that builds the OpenRouter ``AsyncOpenAI`` client.

    OpenRouter is OpenAI-compatible at ``/api/v1``. The factory honors the
    manifest-friendly ``base_url`` override that the legacy
    ``openai_client_instance`` hardwired away, and injects the optional
    attribution headers. The client is cached on the runtime's
    ``providers.openrouter_client`` slot.
    """

    def _build() -> Any:
        _prune_dead_proxy_client(runtime, "openrouter", "openai")
        existing = getattr(getattr(runtime, "providers", None), "openrouter_client", None)
        if existing is not None:
            return existing
        api_key = _env(LISA_OPENROUTER_API_KEY_ENV, "OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                f"OpenRouter is not configured. Set {LISA_OPENROUTER_API_KEY_ENV} "
                "(or OPENROUTER_API_KEY)."
            )
        direct_base_url = direct_provider_base_url(runtime, "openrouter")
        base_url = _proxy_base_url(runtime, "openrouter", direct_base_url)
        default_headers: dict[str, str] = {}
        referer = _env(LISA_OPENROUTER_REFERER_ENV, "OPENROUTER_REFERER")
        title = _env(LISA_OPENROUTER_TITLE_ENV, "OPENROUTER_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-OpenRouter-Title"] = title
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        with contextlib.suppress(AttributeError):
            runtime.providers.openrouter_client = client
        return client

    return _build


__all__ = [
    "HostGatewayProvider",
]
