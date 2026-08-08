"""OpenAI probe — public ``/v1/models`` endpoint.

Filters to model ids we actually drive (``gpt*`` and the ``o``-series
reasoning models) to avoid flooding the manifest with embedding/whisper
models that the host can't invoke as chat completions.
"""

from __future__ import annotations

import httpx

from ..classify import apply_known_override, env_any
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
            models.append(_enrich_openai_model(model_id))
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


def _enrich_openai_model(model_id: str) -> DiscoveredModel:
    """Build a :class:`DiscoveredModel` with OpenAI-family capability inference.

    OpenAI's ``/v1/models`` endpoint returns only ``id``, ``owned_by`` and
    ``created`` — no capability or pricing metadata. We infer capabilities,
    cost tier, and latency tier from the model-id family:

    - ``gpt-5*`` models: tools + structured_output + vision (multimodal).
    - ``o*-`` reasoning models: reasoning + structured_output (no vision by default).
    - ``gpt-chat-*``: text + tools + structured_output (fast-chat tier).
    - ``mini`` suffix: low-latency / budget tier.
    - ``codex`` suffix: code capability.
    """

    lowered = model_id.lower()
    capabilities: list[str] = ["text", "tools", "structured_output"]
    thinking_model = False
    cost_tier = ""
    latency_tier = ""
    display_name = model_id

    if lowered.startswith("o1") or lowered.startswith("o3") or lowered.startswith("o4"):
        capabilities = ["text", "reasoning", "structured_output"]
        thinking_model = True
        cost_tier = "premium" if "pro" in lowered else "standard"
        latency_tier = "high"
    elif lowered.startswith("gpt-5"):
        capabilities = ["text", "tools", "structured_output", "code"]
        if "mini" in lowered:
            cost_tier = "standard"
            latency_tier = "medium"
            display_name = f"GPT-5 Mini ({model_id})"
        else:
            cost_tier = "frontier"
            latency_tier = "medium"
            display_name = f"GPT-5 ({model_id})"
    elif lowered.startswith("gpt-chat"):
        capabilities = ["text", "tools", "structured_output"]
        cost_tier = "standard"
        latency_tier = "low"
        display_name = f"GPT Chat ({model_id})"
    elif "codex" in lowered:
        capabilities = ["text", "tools", "structured_output", "code"]
        cost_tier = "premium"
        latency_tier = "medium"

    if "mini" in lowered and not cost_tier:
        cost_tier = "budget"
        latency_tier = "low"

    # Codex-family models on Azure only expose the Responses API; the
    # chat-completions endpoint rejects them with ``The requested
    # operation is unsupported.`` Infer ``completion_api`` so the
    # generated manifest routes them correctly without a hand edit.
    completion_api = "responses" if "codex" in lowered else "chat_completions"

    enrichment = apply_known_override(
        "openai",
        model_id,
        {
            "thinking_model": thinking_model,
            "display_name": display_name,
            "capabilities": tuple(capabilities),
            "cost_tier": cost_tier,
            "latency_tier": latency_tier,
            "completion_api": completion_api,
        },
    )
    return _model(
        model_id,
        "openai",
        thinking_model=bool(enrichment.get("thinking_model", thinking_model)),
        display_name=str(enrichment.get("display_name", display_name)),
        capabilities=tuple(enrichment.get("capabilities", tuple(capabilities))),
        cost_tier=str(enrichment.get("cost_tier", cost_tier)),
        latency_tier=str(enrichment.get("latency_tier", latency_tier)),
        structured_reliability=1.0,
        completion_api=str(enrichment.get("completion_api", completion_api)),
    )


class _Skipped(Exception):
    """Internal sentinel — never raised after the SKIPPED short-circuit."""
