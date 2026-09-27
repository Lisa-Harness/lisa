"""Host-owned direct provider endpoint resolution.

Single source of truth for each provider's *direct* base URL — the endpoint a
provider gateway uses when no proxy route is active, and the endpoint a
provider-facing proxy must forward to. Proxy child processes are configured
exclusively through this module: it never consults
``runtime.resources.proxy_route_for`` so a proxy can never be wired to forward
to itself (or to another host-managed proxy) in a routing loop.

Precedence for one provider:

1. The model manifest's per-model ``base_url``/``base_url_env`` — but only
   when every model for that provider resolves to one distinct value. Mixed
   per-model endpoints mean one process cannot serve the provider and raise
   a ``ValueError`` (declare one proxy process per endpoint instead).
2. The provider's environment configuration (``LISA_OLLAMA_HOST`` etc.).
3. The provider's built-in default.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1/"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1/"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"


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
    manifest = getattr(runtime.session, "manifest", None)
    if manifest is not None:
        from_manifest = _manifest_base_url(manifest, normalized)
        if from_manifest is not None:
            return from_manifest
    if normalized == "ollama":
        configured = _env("LISA_OLLAMA_HOST", "OLLAMA_HOST")
        return normalize_ollama_host_to_base_url(configured) or OLLAMA_DEFAULT_BASE_URL
    if normalized == "openai":
        return OPENAI_DEFAULT_BASE_URL
    if normalized == "openrouter":
        return (
            _env("LISA_OPENROUTER_BASE_URL", "OPENROUTER_BASE_URL") or OPENROUTER_DEFAULT_BASE_URL
        )
    if normalized == "anthropic":
        return _env("LISA_ANTHROPIC_BASE_URL") or ANTHROPIC_DEFAULT_BASE_URL
    raise ValueError(f"No direct base URL is known for provider '{provider}'")


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
