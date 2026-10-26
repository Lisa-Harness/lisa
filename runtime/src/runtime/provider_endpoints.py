"""Host-owned direct provider endpoint resolution.

Resolves the *direct* (non-proxy) base URL a provider gateway uses when no
proxy route is active, and the endpoint a provider-facing proxy forwards to.
Proxy child processes are configured exclusively through this module: it never
consults ``runtime.resources.proxy_route_for`` so a proxy can never be wired to
forward to itself (or to another host-managed proxy) in a routing loop.

The host keeps no provider→endpoint table. Where a provider's API lives is
gateway knowledge — the same reason its wire protocol is (see
``ProviderGatewayProvider.proxy_protocol_for``) — so the built-in default comes
from the seam's ``direct_base_url_for``. This module owns only the *precedence*
and the loop-prevention rule.

Precedence for one provider:

1. The model manifest's per-model ``base_url``/``base_url_env`` — but only
   when every model for that provider resolves to one distinct value. Mixed
   per-model endpoints mean one process cannot serve the provider and raise
   a ``ValueError`` (declare one proxy process per endpoint instead).
2. The host's configured override (``HostConfig.provider_base_urls``), which
   covers the legacy single-provider variables (``LISA_OLLAMA_HOST`` etc.).
3. The gateway seam's built-in default for that provider.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)


def normalize_ollama_host_to_base_url(host: str) -> str:
    """Normalize an Ollama host value to an OpenAI-compatible base URL."""

    normalized = host.strip().rstrip("/")
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return f"{normalized}/" if normalized.endswith("/v1") else f"{normalized}/v1/"


def _env(*names: str) -> str:
    """Return the first non-empty environment value among ``names``."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _configured_base_url(runtime: HostRuntime, provider: str) -> str | None:
    """Return the host's configured base-URL override for ``provider``, if any."""

    config = getattr(runtime, "config", None)
    overrides = getattr(config, "provider_base_urls", None)
    if isinstance(overrides, Mapping):
        configured = str(overrides.get(provider, "") or "").strip()
        if configured:
            return configured
    return None


def _gateway_default_base_url(runtime: HostRuntime, provider: str) -> str | None:
    """Return the gateway seam's built-in endpoint for ``provider``, if any.

    Discovery is deferred to call time and failures are swallowed: an absent
    or older gateway provider yields ``None`` so the caller can raise its
    documented "no endpoint" error instead of crashing startup.
    """
    from kernel.provider_gateway_discovery import discover_provider_gateway_factories

    for entry in discover_provider_gateway_factories():
        loaded = entry.provider
        try:
            if callable(loaded) and not hasattr(loaded, "gateway_for"):
                loaded = loaded()
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "provider gateway entry %r failed to construct; skipping",
                entry.pack_name,
                exc_info=True,
            )
            continue
        resolve = getattr(loaded, "direct_base_url_for", None)
        if not callable(resolve):
            continue
        try:
            resolved = resolve(provider)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "provider gateway seam failed to report a default endpoint for %r",
                provider,
                exc_info=True,
            )
            return None
        if resolved:
            return str(resolved)
        return None
    return None


def _manifest_base_url(manifest: Any, provider: str) -> str | None:
    """Return the single distinct manifest base URL for one provider, or None.

    Considers every manifest model whose ``provider`` matches. A provider
    with no models, or whose models declare no ``base_url``/``base_url_env``,
    yields ``None`` so environment/default resolution applies. Models that
    disagree on their endpoint raise: one host-managed proxy process cannot
    forward to two different upstreams.
    """
    models = getattr(manifest, "models", ()) or ()
    values: list[str] = []
    for model in models:
        if str(getattr(model, "provider", "")).strip().lower() != provider:
            continue
        base_url = str(getattr(model, "base_url", "") or "").strip()
        if not base_url:
            base_url_env = str(getattr(model, "base_url_env", "") or "").strip()
            base_url = _env(base_url_env) if base_url_env else ""
            base_url = base_url.strip().rstrip("/")
        if not base_url:
            continue
        if base_url not in values:
            values.append(base_url)
    if not values:
        return None
    if len(values) > 1:
        joined = ", ".join(values)
        raise ValueError(
            f"Provider '{provider}' has models with mixed base URLs ({joined}); "
            "one proxy process cannot serve multiple upstreams — split the "
            "proxy declaration per endpoint"
        )
    return values[0]


def direct_provider_base_url(runtime: HostRuntime, provider: str) -> str:
    """Return the direct (non-proxy) base URL for one provider."""

    normalized = provider.strip().lower()
    manifest = getattr(getattr(runtime, "session", None), "manifest", None)
    if manifest is not None:
        from_manifest = _manifest_base_url(manifest, normalized)
        if from_manifest is not None:
            return from_manifest
    configured = _configured_base_url(runtime, normalized)
    if configured is not None:
        return configured
    default = _gateway_default_base_url(runtime, normalized)
    if default is not None:
        return default
    raise ValueError(
        f"No direct base URL is known for provider '{provider}': no manifest "
        "endpoint, no configured override, and its gateway declares no default"
    )


def direct_provider_base_urls(
    runtime: HostRuntime,
    providers: list[str] | tuple[str, ...],
) -> dict[str, str]:
    """Resolve direct base URLs for a set of providers, keyed by provider id."""

    return {
        provider.strip().lower(): direct_provider_base_url(runtime, provider)
        for provider in providers
        if provider.strip()
    }


def proxy_route_view(
    runtime: HostRuntime, provider: str, protocol: str | None = None
) -> dict[str, str]:
    """Return a transport-facing routing view for one provider's model calls.

    ``{"via": "proxy", "proxy_id": ..., "base_url": ...}`` when the provider
    routes through a ready same-protocol proxy, ``{"via": "direct"}``
    otherwise. The stage payload this feeds is informational only — the
    authoritative routing decision stays in the gateway client factories,
    which read the same pool route at client construction.
    """
    provider_id = provider.strip().lower()
    protocol = protocol or ("anthropic" if provider_id == "anthropic" else "openai")
    route = runtime.resources.proxy_route_for(provider_id, protocol)
    if route is None:
        pool = runtime.resources.proxy_pool
        if pool is not None:
            for view in pool.health():
                if provider_id in view.providers and view.protocol == protocol:
                    return {"via": "direct", "bypassed_proxy_id": view.proxy_id}
        return {"via": "direct"}
    return {"via": "proxy", "proxy_id": route.proxy_id, "base_url": route.base_url}
