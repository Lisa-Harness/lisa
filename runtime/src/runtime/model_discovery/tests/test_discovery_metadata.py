"""Provider discovery metadata reaches the manifest and loads in the kernel.

Each probe reads the limits and capabilities its list endpoint publishes;
the manifest emits only values the kernel loader accepts, so a discovery
proposal never fails to load because of a provider's catalog data.
"""

from __future__ import annotations

import httpx
import pytest
from kernel.model_manifest import load_model_manifest_from_text

from runtime.model_discovery.manifest import build_manifest_payload, render_manifest_yaml
from runtime.model_discovery.probes.anthropic import AnthropicProbe, _enrich_anthropic_model
from runtime.model_discovery.probes.google import _enrich_google_model
from runtime.model_discovery.probes.openrouter import _enrich_openrouter_model
from runtime.model_discovery.types import DiscoveredModel, ProbeStatus


def _anthropic_row(**overrides: object) -> dict:
    row: dict = {
        "id": "claude-sonnet-5",
        "display_name": "Claude Sonnet 5",
        "line": "sonnet",
        "lifecycle": "active",
        "max_input_tokens": 1_000_000,
        "max_tokens": 128_000,
        "capabilities": {
            "image_input": {"supported": True},
            "structured_outputs": {"supported": True},
            "thinking": {
                "supported": True,
                "types": {
                    "enabled": {"supported": True},
                    "adaptive": {"supported": True},
                },
            },
        },
    }
    row.update(overrides)
    return row


class TestAnthropicMetadata:
    def test_limits_come_from_the_row(self) -> None:
        model = _enrich_anthropic_model("claude-sonnet-5", _anthropic_row())
        assert model.context_window == 1_000_000
        assert model.max_completion_tokens == 128_000

    def test_capabilities_come_from_the_row(self) -> None:
        model = _enrich_anthropic_model("claude-sonnet-5", _anthropic_row())
        assert model.thinking_model is True
        assert {"vision", "structured_output", "reasoning"} <= set(model.capabilities)

    def test_adaptive_only_model_is_not_marked_for_manual_thinking(self) -> None:
        """The gateway sends ``type: enabled``; an adaptive-only model 400s on it."""
        caps = _anthropic_row()["capabilities"]
        caps["thinking"]["types"]["enabled"] = {"supported": False}
        model = _enrich_anthropic_model("claude-opus-9", _anthropic_row(capabilities=caps))
        assert model.thinking_model is False
        assert "reasoning" in model.capabilities

    def test_unsupported_capabilities_are_not_claimed(self) -> None:
        caps = {
            "image_input": {"supported": False},
            "structured_outputs": {"supported": False},
            "thinking": {"supported": False, "types": {"enabled": {"supported": False}}},
        }
        model = _enrich_anthropic_model("claude-haiku-9", _anthropic_row(capabilities=caps))
        assert model.thinking_model is False
        assert not {"vision", "structured_output", "reasoning"} & set(model.capabilities)

    def test_line_drives_tier_over_the_id(self) -> None:
        model = _enrich_anthropic_model("claude-x-1", _anthropic_row(line="haiku"))
        assert (model.cost_tier, model.latency_tier) == ("budget", "low")

    def test_row_without_metadata_keeps_the_heuristic(self) -> None:
        model = _enrich_anthropic_model("claude-opus-5", {})
        assert model.thinking_model is True
        assert model.cost_tier == "frontier"
        assert model.context_window == 0
        assert model.max_completion_tokens == 0

    @pytest.mark.asyncio
    async def test_retired_models_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_ANTHROPIC_API_KEY", "ant-key")
        rows = [
            _anthropic_row(),
            _anthropic_row(id="claude-old-3", lifecycle="retired"),
            _anthropic_row(id="claude-deprecated-4", lifecycle="deprecated"),
        ]
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, json={"data": rows}))
        async with httpx.AsyncClient(transport=transport) as client:
            result = await AnthropicProbe().probe(client, timeout=5.0)
        assert result.status is ProbeStatus.OK
        assert [m.model_id for m in result.models] == ["claude-sonnet-5", "claude-deprecated-4"]
        assert result.dropped_model_ids == ("claude-old-3",)


class TestGoogleMetadata:
    def test_output_limit_and_thinking_flag(self) -> None:
        model = _enrich_google_model(
            "gemini-9-pro",
            {
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 1_048_576,
                "outputTokenLimit": 65_536,
                "thinking": True,
            },
        )
        assert model.context_window == 1_048_576
        assert model.max_completion_tokens == 65_536
        assert model.thinking_model is True


class TestOpenRouterMetadata:
    def test_top_provider_completion_limit(self) -> None:
        row = {
            "id": "vendor/model",
            "name": "Model",
            "context_length": 200_000,
            "top_provider": {"context_length": 200_000, "max_completion_tokens": 32_000},
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        }
        model = _enrich_openrouter_model("vendor/model", row)
        assert model.max_completion_tokens == 32_000

    def test_null_completion_limit_is_unknown(self) -> None:
        row = {"id": "v/m", "top_provider": {"max_completion_tokens": None}}
        assert _enrich_openrouter_model("v/m", row).max_completion_tokens == 0


def _entry(model: DiscoveredModel) -> dict:
    return build_manifest_payload((model,))["models"][0]


class TestManifestCompletionCeiling:
    def test_valid_ceiling_is_emitted_and_loads(self) -> None:
        model = DiscoveredModel(
            model_id="m", provider="openrouter", context_window=200_000, max_completion_tokens=8192
        )
        payload = build_manifest_payload((model,))
        assert payload["models"][0]["max_completion_tokens"] == 8192
        manifest = load_model_manifest_from_text(render_manifest_yaml(payload))
        assert manifest.models[0].max_completion_tokens == 8192

    def test_ceiling_below_loader_floor_is_omitted(self) -> None:
        model = DiscoveredModel(model_id="m", provider="openrouter", max_completion_tokens=512)
        assert "max_completion_tokens" not in _entry(model)

    def test_ceiling_above_context_window_is_omitted(self) -> None:
        model = DiscoveredModel(
            model_id="m", provider="openrouter", context_window=8192, max_completion_tokens=16384
        )
        payload = build_manifest_payload((model,))
        assert "max_completion_tokens" not in payload["models"][0]
        load_model_manifest_from_text(render_manifest_yaml(payload))


class TestRegistryPrecedence:
    """``known_models.yaml`` fills gaps; it never replaces published fields."""

    def test_published_context_window_beats_registry(self) -> None:
        model = _enrich_anthropic_model("claude-sonnet-5", _anthropic_row())
        assert model.context_window == 1_000_000

    def test_registry_fills_unpublished_context_window(self) -> None:
        row = _anthropic_row()
        del row["max_input_tokens"]
        model = _enrich_anthropic_model("claude-sonnet-5", row)
        assert model.context_window == 200_000

    def test_published_negative_capability_beats_registry(self) -> None:
        """The registry marks claude-opus-4-8 thinking; the API says manual thinking 400s."""
        caps = _anthropic_row()["capabilities"]
        caps["thinking"]["types"]["enabled"] = {"supported": False}
        model = _enrich_anthropic_model("claude-opus-4-8", _anthropic_row(capabilities=caps))
        assert model.thinking_model is False

    def test_google_published_input_limit_beats_registry(self) -> None:
        model = _enrich_google_model(
            "gemini-2.5-pro",
            {"supportedGenerationMethods": ["generateContent"], "inputTokenLimit": 1_048_576},
        )
        assert model.context_window == 1_048_576
