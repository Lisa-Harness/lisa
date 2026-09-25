"""Embedding provider for the lisa-memory MCP server.

Implements the :class:`sdk.memory.store.EmbeddingProvider` protocol backed
by an OpenAI-compatible ``/v1/embeddings`` endpoint (Ollama, vLLM, OpenAI,
Azure OpenAI, etc.). Wired into the :class:`SqliteMemoryStore` by the
server lifespan when ``--embedding-model`` is configured; when absent the
store stays lexical-only (no ``memory_vec`` rows).

The provider is best-effort by contract: the store's ``_write_embedding``
swallows failures, so a down embedding endpoint never blocks a memory
write. Embeddings are computed opportunistically on ``record_memory`` and
on ``recall``; a missing or failed embedding simply drops that record
from the semantic candidate pool and the lexical pool takes over.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

__all__ = ["HttpEmbeddingProvider", "resolve_embedding_config"]

logger = logging.getLogger(__name__)


class HttpEmbeddingProvider:
    """OpenAI-compatible ``/v1/embeddings`` embedding provider.

    Calls ``POST {endpoint}/embeddings`` with ``{"model": model_id,
    "input": [texts]}`` and returns the ``embedding`` arrays in order.
    Compatible with Ollama (``/api/embeddings`` via the OpenAI-compatible
    endpoint at ``/v1``), vLLM, and OpenAI/Azure OpenAI.

    The endpoint URL is resolved as ``{base}/embeddings`` where ``base``
    is the ``--embedding-endpoint`` value (with a trailing slash trimmed).
    When the endpoint already ends in ``/embeddings`` it is used as-is.
    """

    def __init__(
        self,
        *,
        model_id: str,
        endpoint: str,
        api_key: str | None = None,
        dimension: int = 1536,
        timeout: float = 30.0,
    ) -> None:
        self._model_id = model_id
        self._endpoint = _normalize_endpoint(endpoint)
        self._api_key = api_key
        self._dimension = int(dimension)
        self._timeout = float(timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order.

        Returns an empty list on any failure so the store's best-effort
        write path skips the embedding sidecar rather than raising.
        """
        if not texts:
            return []
        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = json.dumps({"model": self._model_id, "input": texts})
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._endpoint, content=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "embedding request failed (model=%s endpoint=%s): %s",
                self._model_id,
                self._endpoint,
                exc,
            )
            return []
        return _extract_embeddings(data)


def _normalize_endpoint(endpoint: str) -> str:
    """Resolve the embeddings URL from a base endpoint.

    Accepts either a bare base (``http://localhost:11434/v1``) or a full
    embeddings path (``http://localhost:11434/v1/embeddings``). Trims a
    trailing slash, then appends ``/embeddings`` only when the path does
    not already end with it.
    """
    base = endpoint.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


def _extract_embeddings(data: Any) -> list[list[float]]:
    """Pull the embedding vectors out of an OpenAI-shaped response.

    Handles both the OpenAI ``{"data": [{"embedding": [...]}]}`` shape
    and the Ollama native ``{"embedding": [...]}`` single-text shape.
    Returns an empty list when the response is not shaped as expected.
    """
    if not isinstance(data, dict):
        return []
    data_list = data.get("data")
    if isinstance(data_list, list):
        vectors: list[list[float]] = []
        for item in data_list:
            if isinstance(item, dict):
                vec = item.get("embedding")
                if isinstance(vec, list):
                    vectors.append([float(x) for x in vec])
        return vectors
    # Ollama native single-text shape: top-level ``embedding``.
    single = data.get("embedding")
    if isinstance(single, list):
        return [[float(x) for x in single]]
    return []


def resolve_embedding_config(args: Any) -> tuple[HttpEmbeddingProvider | None, int]:
    """Build an :class:`HttpEmbeddingProvider` from parsed CLI args + env.

    Returns ``(None, default_dim)`` when no embedding model is configured
    so the store stays lexical-only. Reads:

    - ``args.embedding_model`` / ``LISA_EMBEDDING_MODEL`` — the model id
    - ``args.embedding_endpoint`` / ``LISA_EMBEDDING_ENDPOINT`` — the
      OpenAI-compatible base URL
    - ``LISA_EMBEDDING_API_KEY`` — optional bearer token
    - ``LISA_EMBEDDING_DIMENSION`` — optional vector dimension (default 1536)
    - ``LISA_EMBEDDING_TIMEOUT`` — optional request timeout in seconds
    """
    model_id = getattr(args, "embedding_model", None) or os.getenv("LISA_EMBEDDING_MODEL", "")
    endpoint = getattr(args, "embedding_endpoint", None) or os.getenv("LISA_EMBEDDING_ENDPOINT", "")
    if not model_id or not endpoint:
        return None, 1536
    api_key = os.getenv("LISA_EMBEDDING_API_KEY", "")
    dimension = int(os.getenv("LISA_EMBEDDING_DIMENSION", "1536") or "1536")
    timeout = float(os.getenv("LISA_EMBEDDING_TIMEOUT", "30") or "30")
    provider = HttpEmbeddingProvider(
        model_id=model_id,
        endpoint=endpoint,
        api_key=api_key or None,
        dimension=dimension,
        timeout=timeout,
    )
    return provider, dimension
