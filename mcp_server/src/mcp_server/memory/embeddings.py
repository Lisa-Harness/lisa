"""Resolve the memory server's embedding adapter from CLI configuration."""

from __future__ import annotations

import os
from typing import Any

from sdk.memory.embeddings import OpenAIEmbeddingProvider

__all__ = ["resolve_embedding_config"]


def resolve_embedding_config(args: Any) -> tuple[OpenAIEmbeddingProvider | None, int]:
    """Resolve CLI values and environment; provider calls belong to the SDK."""
    model_id = getattr(args, "embedding_model", None) or os.getenv("LISA_EMBEDDING_MODEL", "")
    endpoint = getattr(args, "embedding_endpoint", None) or os.getenv("LISA_EMBEDDING_ENDPOINT", "")
    if not model_id or not endpoint:
        return None, 1536
    api_key = os.getenv("LISA_EMBEDDING_API_KEY", "")
    dimension = int(os.getenv("LISA_EMBEDDING_DIMENSION", "1536") or "1536")
    timeout = float(os.getenv("LISA_EMBEDDING_TIMEOUT", "30") or "30")
    provider = OpenAIEmbeddingProvider(
        model_id=model_id,
        endpoint=endpoint,
        api_key=api_key or None,
        dimension=dimension,
        timeout=timeout,
    )
    return provider, dimension
