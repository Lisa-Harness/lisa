"""Tests for the lisa-memory embedding provider and config resolver.

Covers endpoint normalization, response parsing (OpenAI + Ollama shapes),
and the ``resolve_embedding_config`` CLI/env resolution. The store-level
embedding behavior (write + recall) is covered by ``sdk/memory/tests/test_store.py``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server.memory.embeddings import (
    HttpEmbeddingProvider,
    _extract_embeddings,
    _normalize_endpoint,
    resolve_embedding_config,
)


def test_normalize_endpoint_appends_embeddings() -> None:
    assert _normalize_endpoint("http://localhost:11434/v1") == (
        "http://localhost:11434/v1/embeddings"
    )


def test_normalize_endpoint_preserves_full_path() -> None:
    assert _normalize_endpoint("http://localhost:11434/v1/embeddings") == (
        "http://localhost:11434/v1/embeddings"
    )


def test_normalize_endpoint_trims_trailing_slash() -> None:
    assert _normalize_endpoint("http://localhost:11434/v1/") == (
        "http://localhost:11434/v1/embeddings"
    )


def test_extract_embeddings_openai_shape() -> None:
    data = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }
    vectors = _extract_embeddings(data)
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_extract_embeddings_ollama_native_shape() -> None:
    data = {"embedding": [0.1, 0.2, 0.3]}
    vectors = _extract_embeddings(data)
    assert vectors == [[0.1, 0.2, 0.3]]


def test_extract_embeddings_malformed_returns_empty() -> None:
    assert _extract_embeddings(None) == []
    assert _extract_embeddings({}) == []
    assert _extract_embeddings({"data": "not a list"}) == []


@pytest.mark.asyncio
async def test_embed_returns_empty_on_http_failure() -> None:
    provider = HttpEmbeddingProvider(
        model_id="test-model",
        endpoint="http://localhost:1/v1",
        dimension=4,
        timeout=0.1,
    )
    # No server on port 1 — the request fails and the provider returns [].
    vectors = await provider.embed(["hello"])
    assert vectors == []


@pytest.mark.asyncio
async def test_embed_returns_vectors_on_success() -> None:
    provider = HttpEmbeddingProvider(
        model_id="test-model",
        endpoint="http://localhost:11434/v1",
        dimension=3,
    )
    fake_response = type(
        "_R",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        },
    )()
    mock_post = AsyncMock(return_value=fake_response)
    with patch("httpx.AsyncClient.post", mock_post):
        vectors = await provider.embed(["hello"])
    assert vectors == [[0.1, 0.2, 0.3]]


def test_resolve_embedding_config_returns_none_when_unset() -> None:
    args = argparse.Namespace(embedding_model=None, embedding_endpoint=None)
    os.environ.pop("LISA_EMBEDDING_MODEL", None)
    os.environ.pop("LISA_EMBEDDING_ENDPOINT", None)
    provider, dim = resolve_embedding_config(args)
    assert provider is None
    assert dim == 1536


def test_resolve_embedding_config_builds_provider_from_env() -> None:
    args = argparse.Namespace(embedding_model=None, embedding_endpoint=None)
    os.environ["LISA_EMBEDDING_MODEL"] = "nomic-embed-text"
    os.environ["LISA_EMBEDDING_ENDPOINT"] = "http://localhost:11434/v1"
    try:
        provider, dim = resolve_embedding_config(args)
    finally:
        os.environ.pop("LISA_EMBEDDING_MODEL", None)
        os.environ.pop("LISA_EMBEDDING_ENDPOINT", None)
    assert provider is not None
    assert dim == 1536


def test_resolve_embedding_config_builds_provider_from_args() -> None:
    args = argparse.Namespace(
        embedding_model="nomic-embed-text",
        embedding_endpoint="http://localhost:11434/v1",
    )
    os.environ.pop("LISA_EMBEDDING_MODEL", None)
    os.environ.pop("LISA_EMBEDDING_ENDPOINT", None)
    provider, dim = resolve_embedding_config(args)
    assert provider is not None
    assert dim == 1536