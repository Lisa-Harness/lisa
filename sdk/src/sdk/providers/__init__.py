"""Built-in provider gateways — framework runtime backends.

Provides the reference ``ProviderGatewayProvider`` implementations for
OpenAI, Azure OpenAI, Anthropic, Google Gemini, and AWS Bedrock. Registered
via the ``lisa.extensions.provider_gateways`` entry-point group.

This is **not** a pack — it ships no modes, operations, or authored content.
It is a pluggable runtime backend for the kernel's provider-gateway seam,
kin to ``sdk/router`` (which owns the model-selection seam). It lives in
the SDK tree at ``sdk/providers`` (import path ``sdk.providers``) and is
wired into ``lisa-harness-sdk``'s entry points.
"""

from __future__ import annotations

from typing import Any


class SdkProviderGatewayProvider:
    """Return a kernel-shipped ``ProviderGateway`` for one provider name.

    The gateways themselves are constructed lazily by the caller; this
    provider only declares which provider names the SDK supports and
    hands back the gateway class or instance. Hosts wire credentials and
    transport through the gateway constructors directly.
    """

    def gateway_for(self, provider: str) -> Any | None:
        """Return the gateway class for ``provider``, or None if unsupported."""

        normalized = (provider or "").strip().lower()
        return self._gateway_class_for(normalized)

    def _gateway_class_for(self, normalized: str) -> Any | None:
        """Map one normalized provider name to its gateway class.

        Gateway modules import lazily so the SDK's import cost stays off
        the startup path; each entry point only pays for the provider it
        actually uses.
        """
        if normalized == "azure_openai":
            from sdk.providers.openai_gateway import AzureOpenAIGateway

            return AzureOpenAIGateway
        if normalized == "anthropic":
            from sdk.providers.anthropic_gateway import AnthropicGateway

            return AnthropicGateway
        if normalized == "google":
            from sdk.providers.google_gateway import GoogleGeminiGateway

            return GoogleGeminiGateway
        if normalized == "aws":
            from sdk.providers.bedrock_gateway import BedrockGateway

            return BedrockGateway
        if normalized in {"openai", "ollama", "openrouter"}:
            # OpenRouter is OpenAI-compatible; the host's client factory
            # (host/gateway_provider.py) points the shared
            # OpenAICompatibleGateway at https://openrouter.ai/api/v1 with the
            # OpenRouter attribution headers.
            from sdk.providers.openai_gateway import OpenAICompatibleGateway

            return OpenAICompatibleGateway
        return None

    def proxy_protocol_for(self, provider: str) -> str | None:
        """Return the local-proxy wire protocol serving ``provider``, or None.

        Only the OpenAI-wire and Anthropic gateways accept a loopback proxy
        base URL; they are the two this SDK's host factories resolve through
        ``runtime.resources.proxy_route_for``. Google (native REST), Bedrock
        (boto3), and Azure (endpoint-bound SDK client) build their clients
        from credentials and never consult a proxy route, so they are not
        proxy-routable.

        This is the sole authority for that split: the host asks the seam
        rather than keeping its own provider list, so a third-party pack that
        registers a new gateway (or a new wire shape) declares its own answer
        instead of editing a host table.

        Args:
            provider: Provider gateway id (any case).

        Returns:
            ``"openai"`` or ``"anthropic"`` for a proxy-routable provider,
            ``None`` otherwise (including unknown ids).
        """
        normalized = (provider or "").strip().lower()
        if normalized in {"openai", "ollama", "openrouter"}:
            return "openai"
        if normalized == "anthropic":
            return "anthropic"
        return None


def get_provider_gateway_provider_factory() -> Any:
    """Return the ``ProviderGatewayProvider`` for ``lisa.extensions.provider_gateways``.

    Unlike model routers (which receive ``(manifest, session_plan)`` and may
    return ``None`` when the manifest has no ``router:`` block), the provider
    gateway provider is always active when the SDK is installed. The factory
    returns a ``SdkProviderGatewayProvider`` instance unconditionally.
    """

    return SdkProviderGatewayProvider()
