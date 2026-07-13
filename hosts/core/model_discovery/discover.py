"""Discovery orchestrator.

Runs all provider probes concurrently with per-provider timeouts so one
slow/hung provider cannot stall the others. Returns a typed
:class:`DiscoveryReport`.

This module never raises. Every probe failure is captured as a typed
``ProbeResult`` with a redacted ``ProbeError``.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .probes import (
    AnthropicProbe,
    AzureOpenAIProbe,
    GoogleProbe,
    OllamaProbe,
    OpenAIProbe,
    ProviderProbe,
)
from .types import DiscoveryReport

logger = logging.getLogger(__name__)

# Per-provider probe timeout (seconds). Generous enough for a cold TLS
# handshake but bounded so a hung provider cannot block the whole
# discovery run.
_DEFAULT_PROBE_TIMEOUT = 8.0

# Default probes, in display order. Order matters only for the notes
# rendering; results are not otherwise sorted here.
_DEFAULT_PROBES: tuple[ProviderProbe, ...] = (
    OllamaProbe(),
    OpenAIProbe(),
    AzureOpenAIProbe(),
    GoogleProbe(),
)


async def discover_models(
    *,
    probes: tuple[ProviderProbe, ...] | None = None,
    timeout: float = _DEFAULT_PROBE_TIMEOUT,
) -> DiscoveryReport:
    """Run all provider probes concurrently and return a typed report.

    Failures are recorded as typed ``ProbeResult``s; this coroutine never
    raises.
    """

    probe_list = list(probes) if probes is not None else list(_DEFAULT_PROBES)
    # Anthropic uses the fallback variant so EMPTY still returns models.
    anthropic_idx: int | None = None
    for idx, probe in enumerate(probe_list):
        if isinstance(probe, AnthropicProbe):
            anthropic_idx = idx
            break
    if anthropic_idx is None:
        probe_list.append(AnthropicProbe())
        anthropic_idx = len(probe_list) - 1

    async with httpx.AsyncClient() as client:
        tasks = []
        for idx, probe in enumerate(probe_list):
            if idx == anthropic_idx and isinstance(
                probe, AnthropicProbe
            ):
                tasks.append(probe.probe_with_fallback(client, timeout))
            else:
                tasks.append(probe.probe(client, timeout))
        results = await asyncio.gather(*tasks)

    return DiscoveryReport.from_results(results)