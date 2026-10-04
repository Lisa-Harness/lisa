"""Tests for the OpenRouter probe.

All tests use ``httpx.MockTransport`` — no real network. The probe consumes
OpenRouter's rich ``GET /api/v1/models`` payload (pricing, modalities,
``supported_parameters``, ``reasoning``, ``context_length``) directly into a
``DiscoveredModel``, so these tests assert field mapping from a realistic
row, the chat-capability filter (embedding-only models dropped), pagination
via ``links.next``, and the SKIPPED signal when no key is set.
"""

from __future__ import annotations

import httpx
import pytest

from runtime.model_discovery.probes.openrouter import (
    OpenRouterProbe,
    _cost_tier_from_price,
    _enrich_openrouter_model,
    _is_chat_capable,
)
from runtime.model_discovery.types import ProbeStatus

_FAKE_KEY = "sk-or-test-1234567890"


def _client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


def _row(**overrides: object) -> dict:
    """Return a realistic OpenRouter model row with sane defaults."""

    base: dict = {
        "id": "openai/gpt-5.4",
        "name": "GPT-5.4",
        "context_length": 256000,
        "pricing": {
            "prompt": "0.0000025",
            "completion": "0.000015",
            "image": "0",
            "request": "0",
        },
        "supported_parameters": [
            "temperature",
            "tools",
            "structured_outputs",
            "reasoning",
            "response_format",
        ],
        "architecture": {
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "modality": "multimodal->text",
            "tokenizer": "GPT",
        },
        "reasoning": {"mandatory": False, "default_effort": "medium"},
        "description": "OpenAI's frontier model.",
    }
    base.update(overrides)
    return base


def _single_row_payload(row: dict) -> dict:
    return {"data": [row], "links": {"next": None}, "total_count": 1}


# ── Chat-capability filter ───────────────────────────────────────────


class TestIsChatCapable:
    def test_text_to_text_is_chat(self) -> None:
        assert _is_chat_capable(_row()) is True

    def test_embedding_output_excluded(self) -> None:
        row = _row(
            architecture={
                "input_modalities": ["text"],
                "output_modalities": ["embeddings"],
                "modality": "text->embedding",
            }
        )
        assert _is_chat_capable(row) is False

    def test_no_text_input_excluded(self) -> None:
        row = _row(
            architecture={
                "input_modalities": ["image"],
                "output_modalities": ["image"],
                "modality": "image->image",
            }
        )
        assert _is_chat_capable(row) is False

    def test_modality_fallback_when_no_output_list(self) -> None:
        # modality "->text" still qualifies when output_modalities is absent.
        row = _row(
            architecture={
                "input_modalities": ["text"],
                "output_modalities": [],
                "modality": "text->text",
            }
        )
        assert _is_chat_capable(row) is True


# ── Enrichment ───────────────────────────────────────────────────────


class TestEnrichment:
    def test_maps_real_payload_fields(self) -> None:
        model = _enrich_openrouter_model("openai/gpt-5.4", _row())
        assert model.provider == "openrouter"
        assert model.model_id == "openai/gpt-5.4"
        assert model.display_name == "GPT-5.4"
        assert model.context_window == 256000
        assert model.prompt_price == pytest.approx(0.0000025)
        assert model.completion_price == pytest.approx(0.000015)
        assert model.pricing_currency == "USD"
        assert set(model.capabilities) == {
            "text",
            "tools",
            "structured_output",
            "vision",
            "reasoning",
            "code",
        }
        assert model.thinking_model is True
        assert model.cost_tier == "frontier"
        # ``known_models.yaml`` pins gpt-5.4 to medium latency (frontier cost,
        # low-latency class) — the registry overrides the probe's computed tier.
        assert model.latency_tier == "medium"
        assert model.description == "OpenAI's frontier model."

    def test_no_reasoning_block_not_thinking(self) -> None:
        model = _enrich_openrouter_model("x/m", _row(reasoning=None))
        assert model.thinking_model is False
        assert model.latency_tier == "medium"

    def test_missing_pricing_leaves_zero_and_blank_tier(self) -> None:
        model = _enrich_openrouter_model("x/m", _row(pricing={}, reasoning=None))
        assert model.prompt_price == 0.0
        assert model.completion_price == 0.0
        assert model.cost_tier == ""

    def test_falls_back_to_top_provider_context(self) -> None:
        model = _enrich_openrouter_model(
            "x/m",
            _row(context_length=0, top_provider={"context_length": 128000}),
        )
        assert model.context_window == 128000

    def test_code_capability_from_model_id(self) -> None:
        model = _enrich_openrouter_model(
            "deepseek/deepseek-coder",
            _row(id="deepseek/deepseek-coder"),
        )
        assert "code" in model.capabilities


class TestCostTierFromPrice:
    def test_blank_when_both_zero(self) -> None:
        assert _cost_tier_from_price(0.0, 0.0) == ""

    def test_budget(self) -> None:
        # e.g. a $0.1/M-token completion model.
        assert _cost_tier_from_price(0.0, 1e-7) == "budget"

    def test_frontier(self) -> None:
        assert _cost_tier_from_price(2.5e-6, 1.5e-5) == "frontier"


# ── Probe end-to-end (mocked HTTP) ──────────────────────────────────


class TestProbe:
    @pytest.mark.asyncio
    async def test_skipped_without_key(self) -> None:
        probe = OpenRouterProbe()
        result = await probe.probe(_client(httpx.MockTransport(lambda r: httpx.Response(200))), 8.0)
        assert result.status is ProbeStatus.SKIPPED
        assert result.models == ()

    @pytest.mark.asyncio
    async def test_ok_maps_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
        seen_auth: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_auth.append(str(request.headers.get("Authorization", "")))
            return httpx.Response(200, json=_single_row_payload(_row()))

        probe = OpenRouterProbe()
        result = await probe.probe(_client(httpx.MockTransport(handler)), 8.0)
        assert result.status is ProbeStatus.OK
        assert len(result.models) == 1
        model = result.models[0]
        assert model.model_id == "openai/gpt-5.4"
        assert model.prompt_price == pytest.approx(0.0000025)
        assert model.provider == "openrouter"
        assert seen_auth and seen_auth[0].startswith("Bearer ")

    @pytest.mark.asyncio
    async def test_empty_list_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [], "links": {"next": None}, "total_count": 0})

        probe = OpenRouterProbe()
        result = await probe.probe(_client(httpx.MockTransport(handler)), 8.0)
        # 200 with zero chat-capable models surfaces as EMPTY.
        assert result.status is ProbeStatus.EMPTY

    @pytest.mark.asyncio
    async def test_filters_non_chat_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
        payload = {
            "data": [
                _row(),
                _row(
                    id="openai/text-embedding-3",
                    name="text-embedding-3",
                    architecture={
                        "input_modalities": ["text"],
                        "output_modalities": ["embeddings"],
                        "modality": "text->embedding",
                    },
                ),
            ],
            "links": {"next": None},
            "total_count": 2,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        probe = OpenRouterProbe()
        result = await probe.probe(_client(httpx.MockTransport(handler)), 8.0)
        assert result.status is ProbeStatus.OK
        assert [m.model_id for m in result.models] == ["openai/gpt-5.4"]

    @pytest.mark.asyncio
    async def test_pagination_follows_links_next(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
        page_b_row = _row(id="anthropic/claude-sonnet-5", name="Claude Sonnet 5")

        def handler(request: httpx.Request) -> httpx.Response:
            if "page=2" in str(request.url.query):
                return httpx.Response(
                    200, json={"data": [page_b_row], "links": {"next": None}, "total_count": 2}
                )
            return httpx.Response(
                200,
                json={
                    "data": [_row()],
                    "links": {"next": "https://openrouter.ai/api/v1/models?page=2&limit=500"},
                    "total_count": 2,
                },
            )

        probe = OpenRouterProbe()
        result = await probe.probe(_client(httpx.MockTransport(handler)), 8.0)
        assert result.status is ProbeStatus.OK
        assert [m.model_id for m in result.models] == [
            "openai/gpt-5.4",
            "anthropic/claude-sonnet-5",
        ]
