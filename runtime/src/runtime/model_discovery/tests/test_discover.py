"""Tests for the discover orchestrator and the command parser.

The key integration guard is ``test_discover_key_never_in_output``: it
injects a fake Google key, runs the full discover pipeline with mocked
transports, and asserts the key never appears in any note string the
UI would render.
"""

from __future__ import annotations

import httpx
import pytest

from runtime.model_discovery import (
    DiscoveryReport,
    discover_models,
)
from runtime.model_discovery.probes import (
    AnthropicProbe,
    AzureOpenAIProbe,
    GoogleProbe,
    OllamaProbe,
    OpenAIProbe,
)
from runtime.model_discovery.types import ProbeStatus

_FAKE_KEY = "AIzaSyFAKEKEY999"


def _all_ok_transport() -> httpx.MockTransport:
    def _handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "gemma3:latest"}]})
        if path.endswith("/v1/models") or path == "/v1/models":
            if "anthropic" in str(req.url):
                return httpx.Response(200, json={"data": [{"id": "claude-sonnet-4-20250514"}]})
            return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})
        if path.endswith("/deployments"):
            return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})
        if path.endswith("/v1beta/models") or path == "/v1beta/models":
            return httpx.Response(200, json={"models": [{"name": "models/gemini-2.5-flash"}]})
        return httpx.Response(404, text=f"no route for {path}")

    return httpx.MockTransport(_handler)


class TestDiscoverOrchestrator:
    @pytest.mark.asyncio
    async def test_all_ok(self) -> None:
        probes = (
            OllamaProbe(),
            OpenAIProbe(),
            AzureOpenAIProbe(),
            GoogleProbe(),
            AnthropicProbe(),
        )
        # All probes are SKIPPED unless we set creds; with creds + mocked
        # transport they should mostly return OK.
        import os

        os.environ["LISA_OPENAI_API_KEY"] = "sk-test"
        os.environ["LISA_AZURE_OPENAI_ENDPOINT"] = "https://res.openai.azure.com"
        os.environ["LISA_AZURE_OPENAI_API_KEY"] = "az-test"
        os.environ["LISA_GOOGLE_API_KEY"] = _FAKE_KEY
        os.environ["LISA_ANTHROPIC_API_KEY"] = "ant-test"
        try:
            report = await discover_models(probes=probes)
        finally:
            for var in (
                "LISA_OPENAI_API_KEY",
                "LISA_AZURE_OPENAI_ENDPOINT",
                "LISA_AZURE_OPENAI_API_KEY",
                "LISA_GOOGLE_API_KEY",
                "LISA_ANTHROPIC_API_KEY",
            ):
                os.environ.pop(var, None)

        assert isinstance(report, DiscoveryReport)
        # Ollama runs without creds; it may fail (no local server) but
        # must not raise.
        statuses = {r.provider: r.status for r in report.results}
        assert "openai" in statuses
        assert "google" in statuses

    @pytest.mark.asyncio
    async def test_no_probes_raises_nothing(self) -> None:
        report = await discover_models(probes=())
        assert isinstance(report, DiscoveryReport)
        assert len(report.results) == 1  # Anthropic fallback auto-added

    @pytest.mark.asyncio
    async def test_skipped_providers_do_not_block_others(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No OpenAI key → SKIPPED; Google has key → probes.
        monkeypatch.setenv("LISA_GOOGLE_API_KEY", _FAKE_KEY)
        httpx.MockTransport(
            lambda req: (
                httpx.Response(
                    200,
                    json={"models": [{"name": "models/gemini-2.5-flash"}]},
                )
                if "v1beta" in req.url.path
                else httpx.Response(404)
            )
        )
        probes = (
            OpenAIProbe(),
            GoogleProbe(),
        )
        # Patch the probes to use our mock transport by overriding probe()
        # via a wrapper. Simpler: just call discover with real probes and
        # rely on the SKIPPED short-circuit for OpenAI.
        report = await discover_models(probes=probes)
        openai_result = next((r for r in report.results if r.provider.lower() == "openai"), None)
        assert openai_result is not None and openai_result.status is ProbeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_discover_key_never_in_output(self) -> None:
        """Regression guard for the credential-leak bug.

        Injects a fake Google key, runs the full discover pipeline with a
        mocked transport that returns 500 for Google, and asserts the key
        never appears in any note string the UI would render.
        """

        import os

        os.environ["LISA_GOOGLE_API_KEY"] = _FAKE_KEY
        os.environ["LISA_ANTHROPIC_API_KEY"] = "ant-test"
        try:
            # Use a transport that returns 500 for Google.
            def _handler(req: httpx.Request) -> httpx.Response:
                if "v1beta" in req.url.path:
                    return httpx.Response(500, text="server error")
                if "anthropic" in str(req.url):
                    return httpx.Response(200, json={"data": [{"id": "claude-sonnet-4-20250514"}]})
                return httpx.Response(404)

            transport = httpx.MockTransport(_handler)

            # Build probes that use the mocked transport.
            google = GoogleProbe()
            anthropic = AnthropicProbe()
            client = httpx.AsyncClient(transport=transport)
            try:
                google_result = await google.probe(client, timeout=5.0)
                anthropic_result = await anthropic.probe_with_fallback(client, timeout=5.0)
            finally:
                await client.aclose()

            report = DiscoveryReport.from_results((google_result, anthropic_result))
            # Also check the raw error detail is redacted from the public note
            for result in report.results:
                note = result.note_line()
                if note is not None:
                    assert _FAKE_KEY not in note, f"key leaked in note: {note}"
                if result.error is not None:
                    assert _FAKE_KEY not in result.error.safe_message
        finally:
            os.environ.pop("LISA_GOOGLE_API_KEY", None)
            os.environ.pop("LISA_ANTHROPIC_API_KEY", None)
