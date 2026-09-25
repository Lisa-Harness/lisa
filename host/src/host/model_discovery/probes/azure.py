"""Azure OpenAI probe — data-plane deployment listing.

Root-cause of the prior "Azure doesn't work" bug: the previous probe
appended ``/openai/deployments`` to the endpoint verbatim, doubling the
``/openai`` segment for users who (reasonably) set ``AZURE_OPENAI_ENDPOINT``
to ``https://resource.openai.azure.com/openai``. This probe normalises
the endpoint to strip any trailing ``/openai`` so the constructed URL
is always ``<base>/openai/deployments`` exactly once.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..classify import apply_known_override, env_any
from ..types import DiscoveredModel, ProbeResult, ProbeStatus
from .base import ProviderProbe, _extract_rows, _model

_LISA_AZURE_ENDPOINT_ENV = "LISA_AZURE_OPENAI_ENDPOINT"
_LISA_AZURE_API_KEY_ENV = "LISA_AZURE_OPENAI_API_KEY"
_LISA_AZURE_API_VERSION_ENV = "LISA_AZURE_OPENAI_API_VERSION"

# Default API version for chat completions. The deployments list
# endpoint on AI Foundry does NOT accept this version — it needs the
# legacy ``2022-12-01`` which is the only version that returns
# ``/openai/deployments`` on ``*.services.ai.azure.com`` endpoints.
_DEFAULT_API_VERSION = "2024-10-21"
_DEPLOYMENTS_API_VERSION = "2022-12-01"


def normalize_azure_endpoint(endpoint: str) -> str:
    """Return ``endpoint`` with any trailing ``/openai`` removed.

    Ensures ``f"{base}/openai/deployments"`` never doubles the segment.
    """

    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/openai"):
        endpoint = endpoint[: -len("/openai")]
    return endpoint


class AzureOpenAIProbe(ProviderProbe):
    provider = "azure_openai"

    async def probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> ProbeResult:
        endpoint = env_any(_LISA_AZURE_ENDPOINT_ENV, "AZURE_OPENAI_ENDPOINT")
        api_key = env_any(_LISA_AZURE_API_KEY_ENV, "AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            return ProbeResult(provider=self.provider, status=ProbeStatus.SKIPPED)
        return await super().probe(client, timeout)

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | None:
        endpoint = env_any(_LISA_AZURE_ENDPOINT_ENV, "AZURE_OPENAI_ENDPOINT")
        api_key = env_any(_LISA_AZURE_API_KEY_ENV, "AZURE_OPENAI_API_KEY")
        api_version = env_any(_LISA_AZURE_API_VERSION_ENV) or _DEFAULT_API_VERSION

        base = normalize_azure_endpoint(endpoint)

        # Strategy: list *deployments* (the models actually deployed in this
        # project), not the full catalog. The ``/openai/deployments`` endpoint
        # accepts different API versions depending on the endpoint type:
        #   - Classic Azure OpenAI (``*.openai.azure.com``): accepts the
        #     configured ``api_version`` (typically ``2024-10-21``).
        #   - AI Foundry (``*.services.ai.azure.com``): rejects
        #     ``2024-10-21`` with 404, but accepts the legacy
        #     ``2022-12-01`` version which returns only deployed models.
        # We try the configured version first, then fall back to the legacy
        # version, then fall back to ``/openai/models`` (catalog) with
        # ``chat_completion`` filtering as a last resort.
        response = await client.get(
            f"{base}/openai/deployments",
            params={"api-version": api_version},
            headers={"api-key": api_key},
            timeout=timeout,
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            # Retry with the legacy deployments API version (AI Foundry).
            response = await client.get(
                f"{base}/openai/deployments",
                params={"api-version": _DEPLOYMENTS_API_VERSION},
                headers={"api-key": api_key},
                timeout=timeout,
            )
        if response.status_code == httpx.codes.NOT_FOUND:
            # Last resort: list catalog models (not deployments) and filter
            # to chat-completion-capable ones. This over-lists (returns the
            # full catalog, not just deployed models) but is better than
            # reporting nothing.
            response = await client.get(
                f"{base}/openai/models",
                params={"api-version": api_version},
                headers={"api-key": api_key},
                timeout=timeout,
            )
        response.raise_for_status()
        payload = response.json()
        rows = _extract_rows(payload, "data")
        models: list[DiscoveredModel] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Deployments have ``model`` (the underlying model) and ``id``
            # (the deployment name). Catalog models have ``id`` only.
            # Use the deployment ``id`` since that's what the OpenAI SDK
            # passes as the ``model`` parameter for chat completions.
            model_id = str(row.get("id") or row.get("model") or "").strip()
            if not model_id:
                continue
            capabilities = row.get("capabilities")
            if isinstance(capabilities, dict) and not capabilities.get("chat_completion", False):
                # On the catalog fallback, only list models that support
                # chat completions (the deployments list already filters
                # to deployed models).
                continue
            models.append(_enrich_azure_model(model_id, row))
        return tuple(models) if models else None


def _azure_catalog_capabilities(row: dict[str, Any]) -> list[str] | None:
    """Parse the Azure catalog ``capabilities`` dict into Lisa capabilities.

    Returns the sorted capability list when the catalog reported any
    capability, or ``None`` to keep the default chat surface.
    """
    raw_capabilities = row.get("capabilities")
    if not isinstance(raw_capabilities, dict):
        return None
    capability_set: set[str] = set()
    if raw_capabilities.get("chat_completion"):
        capability_set.add("text")
    if raw_capabilities.get("tools"):
        capability_set.add("tools")
    if raw_capabilities.get("structured_output") or raw_capabilities.get("json_object"):
        capability_set.add("structured_output")
    if raw_capabilities.get("vision"):
        capability_set.add("vision")
    if raw_capabilities.get("reasoning"):
        capability_set.add("reasoning")
    if capability_set:
        return sorted(capability_set)
    return None


def _azure_family_tiers(
    lowered: str,
    capabilities_list: list[str],
) -> tuple[bool, list[str], str, str]:
    """Derive ``(thinking_model, capabilities, cost_tier, latency_tier)`` from the model family.

    The family chain: o-series (thinking/reasoning, premium for pro),
    gpt-5 (code-capable; mini = standard, else frontier), gpt-chat
    (standard/low), image (image-only), embedding (budget).
    """
    thinking_model = False
    cost_tier = ""
    latency_tier = ""
    if lowered.startswith("o1") or lowered.startswith("o3") or lowered.startswith("o4"):
        thinking_model = True
        capabilities_list = sorted(set(capabilities_list) | {"reasoning"})
        cost_tier = "premium" if "pro" in lowered else "standard"
        latency_tier = "high"
    elif lowered.startswith("gpt-5"):
        capabilities_list = sorted(set(capabilities_list) | {"code"})
        if "mini" in lowered:
            cost_tier = "standard"
            latency_tier = "medium"
        else:
            cost_tier = "frontier"
            latency_tier = "medium"
    elif lowered.startswith("gpt-chat"):
        cost_tier = "standard"
        latency_tier = "low"
    elif "image" in lowered:
        capabilities_list = ["image"]
        cost_tier = "standard"
    elif "embedding" in lowered:
        capabilities_list = ["embedding"]
        cost_tier = "budget"
    return thinking_model, capabilities_list, cost_tier, latency_tier


def _enrich_azure_model(model_id: str, row: dict) -> DiscoveredModel:
    """Build a :class:`DiscoveredModel` enriched from Azure OpenAI metadata.

    Azure's deployments list carries ``model`` (underlying model family) and
    ``capabilities`` (a dict with ``chat_completion``, ``tools``, ``vision``,
    etc.) on the catalog endpoint. Deployments only carry ``id`` and ``model``,
    so we fall back to family-based inference for the enrichment fields.
    """

    lowered = model_id.lower()
    capabilities_list: list[str] = ["text", "tools", "structured_output"]
    display_name = str(row.get("model") or "").strip() or model_id
    # Codex-family models are reasoning/thinking models — they reject
    # ``temperature`` on both the Responses and chat-completions APIs.
    # Detected here (before the family chain) so it does not add a branch
    # to that chain; the o-series also sets ``thinking_model`` in its own
    # branch below.
    is_codex = "codex" in lowered
    thinking_model = is_codex

    # Parse the Azure catalog capabilities dict if present.
    catalog_caps = _azure_catalog_capabilities(row)
    if catalog_caps is not None:
        capabilities_list = catalog_caps

    family_thinking, capabilities_list, cost_tier, latency_tier = _azure_family_tiers(
        lowered, capabilities_list
    )
    if family_thinking:
        thinking_model = True

    # Codex-family models on Azure only expose the Responses API; the
    # chat-completions endpoint rejects them with ``The requested
    # operation is unsupported.`` Infer ``completion_api`` so the
    # generated manifest routes them correctly without a hand edit.
    # ``thinking_model`` for codex is set above (near the init).
    completion_api = "responses" if is_codex else "chat_completions"

    enrichment = apply_known_override(
        "azure_openai",
        model_id,
        {
            "thinking_model": thinking_model,
            "display_name": display_name,
            "capabilities": tuple(capabilities_list),
            "cost_tier": cost_tier,
            "latency_tier": latency_tier,
            "completion_api": completion_api,
        },
    )
    return _model(
        model_id,
        "azure_openai",
        thinking_model=bool(enrichment.get("thinking_model", thinking_model)),
        display_name=str(enrichment.get("display_name", display_name)),
        capabilities=tuple(enrichment.get("capabilities", tuple(capabilities_list))),
        cost_tier=str(enrichment.get("cost_tier", cost_tier)),
        latency_tier=str(enrichment.get("latency_tier", latency_tier)),
        structured_reliability=1.0,
        completion_api=str(enrichment.get("completion_api", completion_api)),
    )
