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
        if normalized in {"openai", "ollama"}:
            from sdk.providers.openai_gateway import OpenAICompatibleGateway

            return OpenAICompatibleGateway
        if normalized == "openrouter":
            # OpenRouter is OpenAI-compatible; the host's client factory
            # (host/gateway_provider.py) points the shared
            # OpenAICompatibleGateway at https://openrouter.ai/api/v1 with the
            # OpenRouter attribution headers.
            from sdk.providers.openai_gateway import OpenAICompatibleGateway

            return OpenAICompatibleGateway
        return None


def get_provider_gateway_provider_factory() -> Any:
    """Return the ``ProviderGatewayProvider`` for ``lisa.extensions.provider_gateways``.

    Unlike model routers (which receive ``(manifest, session_plan)`` and may
    return ``None`` when the manifest has no ``router:`` block), the provider
    gateway provider is always active when the SDK is installed. The factory
    returns a ``SdkProviderGatewayProvider`` instance unconditionally.
    """

    return SdkProviderGatewayProvider()
