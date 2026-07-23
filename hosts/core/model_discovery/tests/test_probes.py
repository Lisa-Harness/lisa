"""Tests for provider probes.

All tests use ``httpx.MockTransport`` so no real network is touched.
The primary regression guard is ``test_google_probe_redacts_on_500``
which asserts the API key never appears in any UI-facing string even
on a server error.
"""

from __future__ import annotations

import httpx
import pytest

from hosts.core.model_discovery.probes.anthropic import AnthropicProbe
from hosts.core.model_discovery.probes.azure import (
    AzureOpenAIProbe,
    normalize_azure_endpoint,
)
from hosts.core.model_discovery.probes.google import (
    GoogleProbe,
    normalize_google_base_url,
)
from hosts.core.model_discovery.probes.ollama import OllamaProbe
from hosts.core.model_discovery.probes.openai import OpenAIProbe
from hosts.core.model_discovery.types import ProbeStatus

_FAKE_KEY = "AIzaSyFAKEKEY123456789"


def _client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


# ── Azure endpoint normalization ─────────────────────────────────────


class TestNormalizeAzureEndpoint:
    def test_strips_trailing_openai(self) -> None:
        endpoint = "https://myresource.openai.azure.com/openai"
        assert normalize_azure_endpoint(endpoint) == "https://myresource.openai.azure.com"

    def test_no_trailing_openai_unchanged(self) -> None:
        endpoint = "https://myresource.openai.azure.com"
        assert normalize_azure_endpoint(endpoint) == endpoint

    def test_strips_trailing_slash_then_openai(self) -> None:
        endpoint = "https://myresource.openai.azure.com/openai/"
        assert normalize_azure_endpoint(endpoint) == "https://myresource.openai.azure.com"

    def test_idempotent(self) -> None:
        endpoint = "https://myresource.openai.azure.com/openai"
        once = normalize_azure_endpoint(endpoint)
        twice = normalize_azure_endpoint(once)
        assert once == twice

    def test_strips_trailing_slash_bare(self) -> None:
        assert (
            normalize_azure_endpoint("https://x.openai.azure.com/") == "https://x.openai.azure.com"
        )


# ── Google base URL normalization ────────────────────────────────────


class TestNormalizeGoogleBaseUrl:
    def test_empty_returns_default(self) -> None:
        assert normalize_google_base_url("") == "https://generativelanguage.googleapis.com"

    def test_strips_trailing_slash(self) -> None:
        assert (
            normalize_google_base_url("https://custom.example.com/") == "https://custom.example.com"
        )

    def test_preserves_bare_host(self) -> None:
        assert (
            normalize_google_base_url("https://custom.example.com") == "https://custom.example.com"
        )

    def test_strips_v1beta_models_suffix(self) -> None:
        assert (
            normalize_google_base_url("https://generativelanguage.googleapis.com/v1beta/models")
            == "https://generativelanguage.googleapis.com"
        )

    def test_strips_v1beta3_models_suffix(self) -> None:
        assert (
            normalize_google_base_url("https://generativelanguage.googleapis.com/v1beta3/models")
            == "https://generativelanguage.googleapis.com"
        )

    def test_strips_bare_v1beta_suffix(self) -> None:
        assert (
            normalize_google_base_url("https://generativelanguage.googleapis.com/v1beta")
            == "https://generativelanguage.googleapis.com"
        )

    def test_strips_bare_models_suffix(self) -> None:
        assert (
            normalize_google_base_url("https://generativelanguage.googleapis.com/models")
            == "https://generativelanguage.googleapis.com"
        )


# ── Ollama probe ─────────────────────────────────────────────────────


class TestOllamaProbe:
    @pytest.mark.asyncio
    async def test_ok_with_models(self) -> None:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"models": [{"name": "gemma3:latest"}, {"name": "llama3.2:3b"}]},
            )
        )
        result = await OllamaProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        assert len(result.models) == 2
        assert result.models[0].model_id == "gemma3:latest"

    @pytest.mark.asyncio
    async def test_empty_when_no_models(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"models": []}))
        result = await OllamaProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.EMPTY

    @pytest.mark.asyncio
    async def test_network_error_on_connection_failure(self) -> None:
        def _handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        transport = httpx.MockTransport(_handler)
        result = await OllamaProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.NETWORK
        assert result.error is not None
        assert "connection" in result.error.safe_message.lower()


# ── OpenAI probe ──────────────────────────────────────────────────────


class TestOpenAIProbe:
    @pytest.mark.asyncio
    async def test_skipped_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LISA_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        result = await OpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_ok_filters_to_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_OPENAI_API_KEY", "sk-test")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-4o"},
                        {"id": "gpt-4o-mini"},
                        {"id": "o3-mini"},
                        {"id": "text-embedding-3-small"},
                        {"id": "whisper-1"},
                    ]
                },
            )
        )
        result = await OpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        ids = {m.model_id for m in result.models}
        assert "gpt-4o" in ids
        assert "gpt-4o-mini" in ids
        assert "o3-mini" in ids
        assert "text-embedding-3-small" not in ids
        assert "whisper-1" not in ids

    @pytest.mark.asyncio
    async def test_auth_on_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_OPENAI_API_KEY", "sk-bad")
        transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad key"}))
        result = await OpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.AUTH
        assert "sk-bad" not in result.error.safe_message


# ── Azure probe ───────────────────────────────────────────────────────


class TestAzureProbe:
    @pytest.mark.asyncio
    async def test_skipped_without_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LISA_AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        result = await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_ok_with_deployments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "az-key")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]},
            )
        )
        result = await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        assert len(result.models) == 2

    @pytest.mark.asyncio
    async def test_normalizes_openai_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/openai")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "az-key")
        captured: dict[str, str] = {}

        def _handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})

        transport = httpx.MockTransport(_handler)
        result = await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        # URL must contain exactly one /openai/deployments
        assert captured["url"].count("/openai/deployments") == 1
        assert "/openai/openai/deployments" not in captured["url"]

    @pytest.mark.asyncio
    async def test_auth_on_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "az-bad")
        transport = httpx.MockTransport(lambda req: httpx.Response(401, json={}))
        result = await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.AUTH
        assert "az-bad" not in result.error.safe_message

    @pytest.mark.asyncio
    async def test_not_found_on_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "az-key")
        # Both /deployments and /models return 404 → NOT_FOUND
        transport = httpx.MockTransport(lambda req: httpx.Response(404, json={}))
        result = await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_api_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI Foundry: /openai/deployments 404s on 2024-10-21, but 200 on 2022-12-01."""

        monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_VERSION", "2024-10-21")
        calls: list[tuple[str, str]] = []

        def _handler(req: httpx.Request) -> httpx.Response:
            path = req.url.path
            version = req.url.params.get("api-version", "")
            calls.append((path, version))
            if path.endswith("/openai/deployments") and version == "2022-12-01":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "o4-mini", "model": "o4-mini", "object": "deployment"},
                            {"id": "gpt-4o", "model": "gpt-4o", "object": "deployment"},
                        ]
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(_handler)
        result = await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        assert len(result.models) == 2
        assert result.models[0].model_id == "o4-mini"
        # Must have tried the configured version first, then the legacy version
        assert any(p.endswith("/openai/deployments") and v == "2024-10-21" for p, v in calls)
        assert any(p.endswith("/openai/deployments") and v == "2022-12-01" for p, v in calls)

    @pytest.mark.asyncio
    async def test_falls_back_to_models_catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both deployments versions 404 → fall back to /openai/models catalog."""

        monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "az-key")
        calls: list[str] = []

        def _handler(req: httpx.Request) -> httpx.Response:
            path = req.url.path
            calls.append(path)
            if path.endswith("/openai/models"):
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "gpt-4o", "capabilities": {"chat_completion": True}},
                            {
                                "id": "text-embedding-3-small",
                                "capabilities": {"chat_completion": False},
                            },
                        ]
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(_handler)
        result = await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        assert len(result.models) == 1  # only chat_completion=True
        assert result.models[0].model_id == "gpt-4o"
        assert any(p.endswith("/openai/models") for p in calls)

    @pytest.mark.asyncio
    async def test_api_key_not_in_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Azure key must travel via the api-key header, never the URL."""

        monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
        monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "az-secret-123")
        captured: dict[str, str] = {}

        def _handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            captured["api_key_header"] = req.headers.get("api-key", "")
            return httpx.Response(200, json={"data": []})

        transport = httpx.MockTransport(_handler)
        await AzureOpenAIProbe().probe(_client(transport), timeout=5.0)
        assert "az-secret-123" not in captured["url"]
        assert captured["api_key_header"] == "az-secret-123"


# ── Google probe ──────────────────────────────────────────────────────


class TestGoogleProbe:
    @pytest.mark.asyncio
    async def test_skipped_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LISA_GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        result = await GoogleProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_ok_with_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_GOOGLE_API_KEY", _FAKE_KEY)
        monkeypatch.delenv("LISA_GOOGLE_API_BASE_URL", raising=False)
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-2.5-flash",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                        {
                            "name": "models/gemini-2.5-pro",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                    ]
                },
            )
        )
        result = await GoogleProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        assert result.models[0].model_id == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_strips_v1beta3_models_suffix_from_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LISA_GOOGLE_API_BASE_URL is the full v1beta3/models path,
        the probe must strip it and hit /v1beta/models instead."""

        monkeypatch.setenv("LISA_GOOGLE_API_KEY", _FAKE_KEY)
        monkeypatch.setenv(
            "LISA_GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta3/models"
        )
        captured: dict[str, str] = {}

        def _handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-2.5-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                },
            )

        transport = httpx.MockTransport(_handler)
        result = await GoogleProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        # Must hit /v1beta/models, NOT /v1beta3/models/v1beta/models
        assert "/v1beta/models" in captured["url"]
        assert "/v1beta3/models/v1beta/models" not in captured["url"]

    @pytest.mark.asyncio
    async def test_uses_header_not_query_for_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_GOOGLE_API_KEY", _FAKE_KEY)
        captured: dict[str, str] = {}

        def _handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            captured["x_goog_api_key"] = req.headers.get("x-goog-api-key", "")
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-2.5-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                },
            )

        transport = httpx.MockTransport(_handler)
        await GoogleProbe().probe(_client(transport), timeout=5.0)
        # Key must be in the header, NOT in the URL query string
        assert captured["x_goog_api_key"] == _FAKE_KEY
        assert _FAKE_KEY not in captured["url"]
        assert "key=" not in captured["url"]

    @pytest.mark.asyncio
    async def test_filters_to_chat_models_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-chat models (TTS, image, embedding, video, robotics) must be excluded."""

        monkeypatch.setenv("LISA_GOOGLE_API_KEY", _FAKE_KEY)
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "models": [
                        # Chat models — keep
                        {
                            "name": "models/gemini-2.5-flash",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                        {
                            "name": "models/gemini-2.5-pro",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                        # TTS — filter out (no generateContent AND name marker)
                        {
                            "name": "models/gemini-3.1-flash-tts-preview",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                        # Image — filter out (name marker)
                        {
                            "name": "models/gemini-3-pro-image",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                        # Embedding — filter out (no generateContent)
                        {
                            "name": "models/gemini-embedding-2",
                            "supportedGenerationMethods": ["embedContent", "countTokens"],
                        },
                        # Video — filter out (no generateContent)
                        {
                            "name": "models/veo-3.1-generate-preview",
                            "supportedGenerationMethods": ["predictLongRunning"],
                        },
                        # Deep research — filter out (name marker)
                        {
                            "name": "models/deep-research-preview-04-2026",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                        # Robotics — filter out (name marker)
                        {
                            "name": "models/gemini-robotics-er-1.5-preview",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        },
                    ]
                },
            )
        )
        result = await GoogleProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        ids = {m.model_id for m in result.models}
        assert "gemini-2.5-flash" in ids
        assert "gemini-2.5-pro" in ids
        assert len(result.models) == 2
        assert "gemini-3.1-flash-tts-preview" not in ids
        assert "gemini-3-pro-image" not in ids
        assert "gemini-embedding-2" not in ids
        assert "veo-3.1-generate-preview" not in ids
        assert "deep-research-preview-04-2026" not in ids
        assert "gemini-robotics-er-1.5-preview" not in ids

    @pytest.mark.asyncio
    async def test_redacts_on_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard: key must never appear in any UI-facing string on error."""

        monkeypatch.setenv("LISA_GOOGLE_API_KEY", _FAKE_KEY)
        transport = httpx.MockTransport(lambda req: httpx.Response(500, text="server error"))
        result = await GoogleProbe().probe(_client(transport), timeout=5.0)
        assert result.status is not ProbeStatus.OK
        assert result.error is not None
        safe = result.error.safe_message
        assert _FAKE_KEY not in safe
        # Also check the detail (internal) is not leaked via the public note
        assert _FAKE_KEY not in (result.error.detail or "")
        assert "generativelanguage.googleapis.com" not in safe

    @pytest.mark.asyncio
    async def test_auth_on_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_GOOGLE_API_KEY", _FAKE_KEY)
        transport = httpx.MockTransport(lambda req: httpx.Response(401, json={}))
        result = await GoogleProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.AUTH
        assert _FAKE_KEY not in result.error.safe_message


# ── Anthropic probe ───────────────────────────────────────────────────


class TestAnthropicProbe:
    @pytest.mark.asyncio
    async def test_skipped_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LISA_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        result = await AnthropicProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_ok_with_real_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_ANTHROPIC_API_KEY", "ant-key")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "data": [{"id": "claude-sonnet-4-20250514"}, {"id": "claude-opus-4-20250514"}]
                },
            )
        )
        result = await AnthropicProbe().probe(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.OK
        assert len(result.models) == 2

    @pytest.mark.asyncio
    async def test_fallback_on_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_ANTHROPIC_API_KEY", "ant-key")
        transport = httpx.MockTransport(lambda req: httpx.Response(404, json={}))
        # probe_with_fallback substitutes conservative defaults on EMPTY
        result = await AnthropicProbe().probe_with_fallback(_client(transport), timeout=5.0)
        assert result.status is ProbeStatus.EMPTY
        assert len(result.models) == 3  # fallback tuple
        assert result.models[0].provider == "anthropic"
        # Verify the fallback IDs match the current stale-free tuple
        fallback_ids = {m.model_id for m in result.models}
        assert "claude-sonnet-4-5-20250929" in fallback_ids
        assert result.error is not None
        assert "conservative" in result.error.safe_message.lower()

    @pytest.mark.asyncio
    async def test_sends_version_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_ANTHROPIC_API_KEY", "ant-key")
        captured: dict[str, str] = {}

        def _handler(req: httpx.Request) -> httpx.Response:
            captured["anthropic_version"] = req.headers.get("anthropic-version", "")
            captured["x_api_key"] = req.headers.get("x-api-key", "")
            return httpx.Response(200, json={"data": []})

        transport = httpx.MockTransport(_handler)
        await AnthropicProbe().probe_with_fallback(_client(transport), timeout=5.0)
        assert captured["anthropic_version"] == "2023-06-01"
        assert captured["x_api_key"] == "ant-key"
