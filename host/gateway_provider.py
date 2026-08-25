"""Host-side provider-gateway provider.

The kernel's single live completion path now routes through
``ContextSeams.provider_gateways`` (a :class:`ProviderGatewayProvider`) —
the legacy ``host.providers.complete_live`` dispatch is removed. This module
turns the pack's entry-point gateway factory (which returns gateway
*classes*) into concrete, credential-bound gateway **instances** keyed by
provider name.

Environment variables are resolved here (the host owns credentials), and the
constructed gateway instances are cached per provider for the runtime's
lifetime so the underlying SDK clients (which the existing
``host.providers.*_client_instance`` factories cache on
:class:`~host.runtime.HostRuntime`) are reused across turns. No gateway is
constructed eagerly: instances are created lazily on the first
``gateway_for(provider)`` call and cached.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from host.providers import (
    anthropic_http_client_instance,
    azure_client_instance,
    azure_responses_client_instance,
    ollama_client_instance,
    openai_client_instance,
)
from host.runtime import HostRuntime

logger = logging.getLogger(__name__)

# Optional OpenRouter attribution/telemetry headers. OpenRouter is
# OpenAI-compatible; these opt the host onto app rankings.
LISA_OPENROUTER_API_KEY_ENV = "LISA_OPENROUTER_API_KEY"
LISA_OPENROUTER_BASE_URL_ENV = "LISA_OPENROUTER_BASE_URL"
LISA_OPENROUTER_REFERER_ENV = "LISA_OPENROUTER_REFERER"
LISA_OPENROUTER_TITLE_ENV = "LISA_OPENROUTER_TITLE"
_OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class _GatewayClassProvider(Protocol):
    """The shape of a pack's gateway-provider entry point.

    Returns the gateway *class* for one provider name, or ``None`` when the
    pack does not ship a backend for that provider. The host is responsible
    for instantiating the class with credentials/transport.
    """

    def gateway_for(self, provider: str) -> type | None: ...


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

    # ── Construction ────────────────────────────────────────────────
    def _build(self, provider: str) -> Any | None:
        """Construct the gateway instance for one provider (uncached)."""

        gateway_cls = self._discovered.gateway_for(provider)
        if gateway_cls is None:
            return None

        runtime = self._runtime
        try:
            if provider == "azure_openai":
                # Azure needs a dual-client split: chat-completions on the GA
                # api-version and the Responses API (Codex) on a preview
                # api-version. Both factories are runtime-cached.
                return gateway_cls(
                    complete_client_factory=lambda: azure_client_instance(runtime),
                    responses_client_factory=lambda: azure_responses_client_instance(runtime),
                )
            if provider == "anthropic":
                # AnthropicGateway holds raw ``httpx.AsyncClient``; the key and
                # version headers are injected by the runtime-cached client
                # factory rather than the gateway body.
                return gateway_cls(client_factory=lambda: anthropic_http_client_instance(runtime))
            if provider == "google":
                # Native REST path (chosen over the legacy OpenAI-compat wire
                # shape). The gateway takes concrete credentials, not a client
                # factory.
                return gateway_cls(
                    api_key=_env("LISA_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
                    base_url=_env("LISA_GOOGLE_API_BASE_URL", "GOOGLE_API_BASE_URL")
                    or "https://generativelanguage.googleapis.com",
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
        return lambda: openai_client_instance(runtime)
    return lambda: ollama_client_instance(runtime)


def _openrouter_client_factory(runtime: HostRuntime) -> Callable[[], Any]:
    """Return a client factory that builds the OpenRouter ``AsyncOpenAI`` client.

    OpenRouter is OpenAI-compatible at ``/api/v1``. The factory honors the
    manifest-friendly ``base_url`` override that the legacy
    ``openai_client_instance`` hardwired away, and injects the optional
    attribution headers. The client is cached on the runtime's
    ``_openrouter_client`` slot.
    """

    def _build() -> Any:
        existing = getattr(runtime, "_openrouter_client", None)
        if existing is not None:
            return existing
        from openai import AsyncOpenAI

        api_key = _env(LISA_OPENROUTER_API_KEY_ENV, "OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                f"OpenRouter is not configured. Set {LISA_OPENROUTER_API_KEY_ENV} "
                "(or OPENROUTER_API_KEY)."
            )
        base_url = (
            _env(LISA_OPENROUTER_BASE_URL_ENV, "OPENROUTER_BASE_URL")
            or _OPENROUTER_DEFAULT_BASE_URL
        )
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
        try:
            runtime._openrouter_client = client  # noqa: SLF001
        except AttributeError:
            pass
        return client

    return _build


__all__ = [
    "HostGatewayProvider",
    "httpx",
]
