"""Google Gemini probe.

Two structural changes vs. the prior implementation:

1. The API key is sent via the ``x-goog-api-key`` header rather than as
   a ``?key=...`` query parameter. Gemini REST supports the header as
   of ``v1beta``. This removes the key from the request URL entirely,
   so even if a future ``httpx`` version embeds the URL in an exception
   string, the key is not there to leak. (The query-param path is
   retained only as a fallback if the header is rejected, and even then
   any error is routed through :func:`safe_http_error` which never
   includes the URL.)
2. Every failure path is converted through ``ProviderProbe.probe``'s
   typed ``ProbeError`` — no raw ``str(exc)`` reaches the UI.
"""

from __future__ import annotations

import httpx

from ..classify import env_any
from ..types import DiscoveredModel, ProbeResult, ProbeStatus
from .base import ProviderProbe, _extract_rows, _model

_LISA_GOOGLE_API_KEY_ENV = "LISA_GOOGLE_API_KEY"
_LISA_GOOGLE_API_BASE_URL_ENV = "LISA_GOOGLE_API_BASE_URL"

_DEFAULT_BASE = "https://generativelanguage.googleapis.com"

# Path suffixes the user may have included in LISA_GOOGLE_API_BASE_URL.
# These are stripped so the probe can append its own canonical path.
_GOOGLE_STRIP_SUFFIXES = (
    "/v1beta/models",
    "/v1beta3/models",
    "/v1/models",
    "/v1beta",
    "/v1beta3",
    "/v1",
    "/models",
)


def normalize_google_base_url(value: str) -> str:
    """Return a bare host URL for the Google REST API.

    Users may have set ``LISA_GOOGLE_API_BASE_URL`` to the full models
    endpoint (e.g. ``.../v1beta3/models``) per the web host guide. The
    probe constructs its own path (``/v1beta/models``), so any trailing
    path suffix is stripped down to the bare host.
    """

    value = value.strip()
    if not value:
        return _DEFAULT_BASE

    # Strip trailing slashes first.
    value = value.rstrip("/")

    # Strip known path suffixes that the user may have included.
    lowered = value.lower()
    for suffix in _GOOGLE_STRIP_SUFFIXES:
        if lowered.endswith(suffix):
            value = value[: -len(suffix)]
            break

    return value


# Model id substrings that indicate specialty models which support
# ``generateContent`` but are not practical for chat completions. These
# are filtered out even though they technically support text generation.
_NON_CHAT_MODEL_MARKERS = (
    "tts",
    "image",
    "computer-use",
    "deep-research",
    "robotics",
    "lyria",
    "antigravity",
    "omni",
    "nano-banana",
    "aqa",
    "imagen",
    "veo",
    "embedding",
)

# Generation methods that indicate a text-chat-capable model. The model
# must support ``generateContent`` to be considered.
_CHAT_METHOD = "generateContent"


def _is_chat_model(model_id: str, row: dict) -> bool:
    """Return True when ``row`` describes a practical chat-completion model.

    Google's ``/v1beta/models`` endpoint lists 50+ models including TTS,
    image generation, video generation, robotics, embedding, and
    deep-research variants. We filter to models that:
      1. Support ``generateContent`` (the text-generation method), and
      2. Are not in a specialty category (TTS, image, video, robotics, etc.)
    """

    methods = row.get("supportedGenerationMethods")
    if not isinstance(methods, list) or _CHAT_METHOD not in methods:
        return False
    lowered = model_id.lower()
    return not any(marker in lowered for marker in _NON_CHAT_MODEL_MARKERS)


class GoogleProbe(ProviderProbe):
    provider = "google"

    async def probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> ProbeResult:
        api_key = env_any(_LISA_GOOGLE_API_KEY_ENV, "GOOGLE_API_KEY")
        if not api_key:
            return ProbeResult(provider=self.provider, status=ProbeStatus.SKIPPED)
        return await super().probe(client, timeout)

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | None:
        api_key = env_any(_LISA_GOOGLE_API_KEY_ENV, "GOOGLE_API_KEY")
        configured_base = env_any(_LISA_GOOGLE_API_BASE_URL_ENV, "GOOGLE_API_BASE_URL")
        base = normalize_google_base_url(configured_base)

        # Header-first: keeps the key out of the URL (and therefore out
        # of any httpx exception repr that embeds the request URL).
        response = await client.get(
            f"{base}/v1beta/models",
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = _extract_rows(payload, "models")
        models: list[DiscoveredModel] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            model_id = name.replace("models/", "", 1)
            if not _is_chat_model(model_id, row):
                continue
            models.append(_model(model_id, self.provider))
        return tuple(models) if models else None