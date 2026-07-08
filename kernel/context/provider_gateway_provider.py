"""Default ``ProviderGatewayProvider`` returning kernel-shipped gateways.

Replaces direct ``app.providers`` imports. Returns the gateway instances
from ``kernel.provider_core`` for the supported providers. Hosts that need
custom gateway construction or provider health probing override this seam.
"""

from __future__ import annotations

from typing import Any


class KernelProviderGatewayProvider:
    """Return a kernel-shipped ``ProviderGateway`` for one provider name.

    The gateways themselves are constructed lazily by the caller; this
    provider only declares which provider names the kernel supports and
    hands back the gateway class or instance. Hosts wire credentials and
    transport through ``kernel.provider_core`` directly.
    """

    def __init__(self) -> None:
        from kernel import provider_core

        self._provider_core = provider_core

    def gateway_for(self, provider: str) -> Any | None:
        """Return the gateway class for ``provider``, or None if unsupported."""

        normalized = (provider or "").strip().lower()
        if normalized == "azure_openai":
            return self._provider_core.AzureOpenAIGateway
        if normalized == "anthropic":
            return self._provider_core.AnthropicGateway
        if normalized == "google":
            return self._provider_core.GoogleGeminiGateway
        if normalized == "aws":
            return self._provider_core.BedrockGateway
        if normalized in {"openai", "ollama"}:
            return self._provider_core.OpenAICompatibleGateway
        return None
