"""ProviderRegistry — lazily-populated provider client cache.

Holds the per-provider SDK clients the host builds on demand (Azure OpenAI,
Azure Responses, OpenAI, Anthropic HTTP, Ollama, and the OpenRouter
OpenAI-compatible client). The gateway provider (``runtime.gateway_provider``)
caches each constructed client here so repeated model calls reuse one
connection; ``runtime.mcp_pool.configure_runtime`` resets them all to ``None``
when the gateway set is recomputed after a model reload.

Extracted from ``runtime.runtime`` so the provider-client surface is a single
dataclass instead of a dozen parallel properties on the runtime god object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProviderRegistry:
    """Lazily-populated provider SDK clients keyed by provider family.

    All fields start ``None`` and are assigned by the provider client
    factory the first time that provider is exercised. They are ``Any``
    because each holds a distinct third-party SDK type; the kernel never
    sees them — gateways wrap them behind
    :class:`kernel.seams.provider_contracts` surfaces.
    """

    azure_client: Any = None
    azure_responses_client: Any = None
    openai_client: Any = None
    anthropic_http_client: Any = None
    ollama_client: Any = None
    openrouter_client: Any = None
    #: Google (Gemini) SDK client — set by the gateway provider on first use
    #: and reset by ``runtime.mcp_pool.configure_runtime``.
    google_client: Any = None


__all__ = ["ProviderRegistry"]
