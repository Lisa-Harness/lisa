"""Tests for the lisa-memory embedding provider and config resolver.

Covers ``resolve_embedding_config`` CLI/env resolution. The store-level
embedding behavior (write + recall) is covered by ``sdk/memory/tests/test_store.py``.
"""

from __future__ import annotations

import argparse
import os

from mcp_server.memory.embeddings import resolve_embedding_config
from sdk.memory.embeddings import OpenAIEmbeddingProvider


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
    assert isinstance(provider, OpenAIEmbeddingProvider)
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
