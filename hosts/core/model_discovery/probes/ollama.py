"""Ollama probe — local model tags."""

from __future__ import annotations

import httpx

from ..classify import env_any
from ..redact import safe_http_error
from ..types import DiscoveredModel, ProbeError, ProbeResult, ProbeStatus
from .base import ProviderProbe, _extract_rows, _model

_LISA_OLLAMA_HOST_ENV = "LISA_OLLAMA_HOST"


def _normalize_host(host: str) -> str:
    host = host.strip() or "127.0.0.1:11434"
    if not (host.startswith("http://") or host.startswith("https://")):
        host = f"http://{host}"
    return host.rstrip("/")


class OllamaProbe(ProviderProbe):
    provider = "ollama"

    async def _probe(
        self,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[DiscoveredModel, ...] | None:
        host = env_any(_LISA_OLLAMA_HOST_ENV, "OLLAMA_HOST")
        base = _normalize_host(host)

        response = await client.get(f"{base}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        rows = _extract_rows(payload, "models")
        models: list[DiscoveredModel] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("name") or "").strip()
            if not model_id:
                continue
            models.append(_model(model_id, self.provider))
        return tuple(models) if models else None