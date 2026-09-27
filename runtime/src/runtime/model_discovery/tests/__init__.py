"""Test package for the robust model discovery subsystem."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest


def _handler_from_map(
    routes: Mapping[str, Any],
) -> httpx.MockTransport:
    """Build an ``httpx.MockTransport`` that serves canned responses.

    ``routes`` maps a URL path prefix (matched against ``request.url.path``)
    to either an ``httpx.Response`` or a callable ``(request) -> httpx.Response``.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for prefix, value in routes.items():
            if (
                path == prefix
                or path.startswith(prefix.rstrip("/") + "/")
                or path.startswith(prefix)
            ):
                if callable(value):
                    return value(request)
                return value
        return httpx.Response(404, text=f"no route for {path}")

    return httpx.MockTransport(_handler)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub all provider credential env vars so probes default to SKIPPED."""

    for var in (
        "LISA_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "LISA_AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
        "LISA_AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LISA_AZURE_OPENAI_API_VERSION",
        "LISA_AZURE_OPENAI_RESPONSES_API_VERSION",
        "LISA_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "LISA_GOOGLE_API_BASE_URL",
        "GOOGLE_API_BASE_URL",
        "LISA_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "LISA_OLLAMA_HOST",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(var, raising=False)
