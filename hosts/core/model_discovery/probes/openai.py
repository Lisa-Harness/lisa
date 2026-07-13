"""OpenAI probe — public ``/v1/models`` endpoint.

Filters to model ids we actually drive (``gpt*`` and the ``o``-series
reasoning models) to avoid flooding the manifest with embedding/whisper
models that the host can't invoke as chat completions.
"""

from __future__ import annotations

import httpx

from ..classify import env_any
from ..types import DiscoveredModel, ProbeResult, ProbeStatus
from .base import ProviderProbe, _extract_rows, _model

_LISA_OPENAI_API_KEY_ENV = "LISA_OPENAI_API_KEY"

# Keep this narrow: ``gpt*`` covers gpt-4o / gpt-4.1 / gpt-5 / etc. The
# ``o``-series is matched as a prefix only (``o1-``, ``o3-``, ``o4-``)
# via is_reasoning_model. Embedding, whisper, tts, dall-e are excluded.
_SUPPORTED_PREFIXES = ("gpt-",)
_SUPPORTED_EXACT_PREFIXES = ("o1-", "o3-", "o4-", "o1:", "o3:", "o4:")


def _is_supported(model_id: str) -> bool:
    lowered = model_id.lower()
    if any(lowered.startswith(prefix) for prefix in _SUPPORTED_PREFIXES):
        return True
    if any(lowered.startswith(prefix) for prefix in _SUPPORTED_EXACT_PREFIXES):
        return True
    return "reason" in lowered


class OpenAIProbe(ProviderProbe):
    provider = "openai"

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | None:
        api_key = env_any(_LISA_OPENAI_API_KEY_ENV, "OPENAI_API_KEY")
        if not api_key:
            # Signalling SKIPPED via an empty result is wrong here; the
            # base class would tag it EMPTY. Raise a sentinel instead.
            raise _Skipped()

        response = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = _extract_rows(payload, "data")
        models: list[DiscoveredModel] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "").strip()
            if not model_id or not _is_supported(model_id):
                continue
            models.append(_model(model_id, self.provider))
        return tuple(models) if models else None

    async def probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> ProbeResult:
        # Short-circuit SKIPPED before the base class's try/except turns
        # it into an UNKNOWN/EMPTY result.
        api_key = env_any(_LISA_OPENAI_API_KEY_ENV, "OPENAI_API_KEY")
        if not api_key:
            return ProbeResult(provider=self.provider, status=ProbeStatus.SKIPPED)
        return await super().probe(client, timeout)


class _Skipped(Exception):
    """Internal sentinel — never raised after the SKIPPED short-circuit."""