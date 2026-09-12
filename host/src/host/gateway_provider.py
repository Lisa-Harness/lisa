"""Host-side provider-gateway provider.

The kernel's single live completion path routes through
``ContextSeams.provider_gateways`` (a :class:`ProviderGatewayProvider`). This module
turns the pack's entry-point gateway factory (which returns gateway
*classes*) into concrete, credential-bound gateway **instances** keyed by
provider name.

Environment variables are resolved here (the host owns credentials), and the
constructed gateway instances are cached per provider for the runtime's
lifetime so the underlying SDK clients (which the existing
local client factories cache on :class:`~host.runtime.HostRuntime`) are reused
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
from openai import AsyncAzureOpenAI, AsyncOpenAI

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
    def _build(self, provider: str) -> Any | None:  # noqa: PLR0911
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
                    complete_client_factory=lambda: _azure_client_instance(runtime),
                    responses_client_factory=lambda: _azure_responses_client_instance(runtime),
                )
            if provider == "anthropic":
                # AnthropicGateway holds the raw runtime-cached httpx client;
                # auth (x-api-key) and anthropic-version are injected
                # per-request by the gateway from these values.
                return gateway_cls(
                    client_factory=lambda: _anthropic_http_client_instance(runtime),
                    api_key=_env("LISA_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
                    api_version=_env("LISA_ANTHROPIC_VERSION") or "2023-06-01",
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
                    aws_access_key_id=_env("LISA_AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")
                    or None,
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


def _azure_client_instance(runtime: HostRuntime) -> AsyncAzureOpenAI:
    """Return the cached Azure chat-completions client."""

    existing_client = runtime.providers.azure_client
    if isinstance(existing_client, AsyncAzureOpenAI):
        return existing_client
    endpoint = _env("LISA_AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
    api_key = _env("LISA_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI is not configured. Set LISA_AZURE_OPENAI_ENDPOINT and "
            "LISA_AZURE_OPENAI_API_KEY (or AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)."
        )
    client = AsyncAzureOpenAI(
        api_version=_env("LISA_AZURE_OPENAI_API_VERSION") or "2024-10-21",
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    runtime.providers.azure_client = client
    return client


def _azure_responses_client_instance(runtime: HostRuntime) -> AsyncAzureOpenAI:
    """Return the cached Azure Responses API client with its preview version."""

    existing_client = runtime.providers.azure_responses_client
    if isinstance(existing_client, AsyncAzureOpenAI):
        return existing_client
    endpoint = _env("LISA_AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
    api_key = _env("LISA_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI is not configured. Set LISA_AZURE_OPENAI_ENDPOINT and "
            "LISA_AZURE_OPENAI_API_KEY (or AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)."
        )
    api_version = _env("LISA_AZURE_OPENAI_RESPONSES_API_VERSION")
    if not api_version:
        raise ValueError(
            "The Azure Responses API requires LISA_AZURE_OPENAI_RESPONSES_API_VERSION."
        )
    client = AsyncAzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    runtime.providers.azure_responses_client = client
    return client


def _openai_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Return the cached public OpenAI client."""

    existing_client = runtime.providers.openai_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client
    api_key = _env("LISA_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI is not configured. Set LISA_OPENAI_API_KEY (or OPENAI_API_KEY).")
    client = AsyncOpenAI(api_key=api_key)
    runtime.providers.openai_client = client
    return client


def _ollama_client_instance(runtime: HostRuntime) -> AsyncOpenAI:
    """Return the cached Ollama OpenAI-compatible client."""

    existing_client = runtime.providers.ollama_client
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client
    configured_host = _env("LISA_OLLAMA_HOST", "OLLAMA_HOST")
    base_url = _normalize_ollama_host_to_base_url(configured_host) or "http://localhost:11434/v1/"
    client = AsyncOpenAI(
        api_key=_env("LISA_OLLAMA_API_KEY", "OLLAMA_API_KEY") or "ollama",
        base_url=base_url,
    )
    runtime.providers.ollama_client = client
    return client


def _anthropic_http_client_instance(runtime: HostRuntime) -> httpx.AsyncClient:
    """Return the cached raw Anthropic HTTP client."""

    existing_client = runtime.providers.anthropic_http_client
    if isinstance(existing_client, httpx.AsyncClient):
        return existing_client
    base_url = _env("LISA_ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=60.0)
    runtime.providers.anthropic_http_client = client
    return client


def _normalize_ollama_host_to_base_url(host: str) -> str:
    """Normalize an Ollama host value to an OpenAI-compatible base URL."""

    normalized = host.strip().rstrip("/")
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return f"{normalized}/" if normalized.endswith("/v1") else f"{normalized}/v1/"


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
        with contextlib.suppress(AttributeError):
            runtime.providers.openrouter_client = client
        return client

    return _build


__all__ = [
    "HostGatewayProvider",
]
