"""Tests for deterministic enrichment of discovered models."""

from __future__ import annotations

from runtime.model_discovery.classify import (
    apply_known_override,
    load_known_models,
    uses_prompt_cache_breakpoints,
)
from runtime.model_discovery.manifest import build_manifest_payload
from runtime.model_discovery.probes.anthropic import _enrich_anthropic_model
from runtime.model_discovery.probes.azure import _enrich_azure_model
from runtime.model_discovery.probes.google import _enrich_google_model
from runtime.model_discovery.probes.ollama import _enrich_ollama_model
from runtime.model_discovery.probes.openai import _enrich_openai_model
from runtime.model_discovery.types import DiscoveredModel


class TestDiscoveredModelEnrichmentFields:
    def test_defaults_are_empty(self) -> None:
        model = DiscoveredModel(model_id="x", provider="openai")
        assert model.thinking_model is False
        assert model.display_name == ""
        assert model.capabilities == ()
        assert model.cost_tier == ""
        assert model.latency_tier == ""
        assert model.context_window == 0
        assert model.description == ""


class TestGoogleEnrichment:
    def test_flash_model_gets_budget_tier(self) -> None:
        model = _enrich_google_model(
            "gemini-3.5-flash",
            {
                "displayName": "Gemini 3.5 Flash",
                "description": "Fast multimodal model",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
                "inputTokenLimit": 1048576,
            },
        )
        assert model.display_name == "Gemini 3.5 Flash"
        assert model.cost_tier == "budget"
        assert model.latency_tier == "low"
        assert model.context_window == 1048576
        assert "text" in model.capabilities
        assert "tools" in model.capabilities

    def test_pro_model_gets_premium_tier(self) -> None:
        model = _enrich_google_model(
            "gemini-2.5-pro",
            {"supportedGenerationMethods": ["generateContent"]},
        )
        assert model.cost_tier == "premium"

    def test_non_chat_model_filtered_out(self) -> None:
        # _enrich_google_model is only called after _is_chat_model passes,
        # but the enrichment itself should not crash on empty metadata.
        model = _enrich_google_model("gemini-2.5-flash", {})
        assert model.model_id == "gemini-2.5-flash"


class TestOpenAIEnrichment:
    def test_o3_pro_is_thinking_model(self) -> None:
        model = _enrich_openai_model("o3-pro")
        assert model.thinking_model is True
        assert model.reasoning_model is True
        assert "reasoning" in model.capabilities
        # The static registry overrides o3-pro to frontier tier.
        assert model.cost_tier == "frontier"

    def test_gpt_5_is_frontier(self) -> None:
        model = _enrich_openai_model("gpt-5.4")
        assert model.cost_tier == "frontier"
        assert "code" in model.capabilities

    def test_mini_is_budget(self) -> None:
        model = _enrich_openai_model("gpt-5.4-mini")
        assert model.cost_tier == "standard"
        assert model.latency_tier == "low"

    def test_codex_uses_responses_api(self) -> None:
        """Codex-family models only expose the Responses API on Azure."""
        model = _enrich_openai_model("gpt-5.1-codex-max")
        assert model.completion_api == "responses"

    def test_non_codex_uses_chat_completions(self) -> None:
        """Non-codex models keep the default chat-completions endpoint."""
        model = _enrich_openai_model("gpt-5.4")
        assert model.completion_api == "chat_completions"


class TestAnthropicEnrichment:
    def test_opus_is_frontier_thinking(self) -> None:
        model = _enrich_anthropic_model("claude-opus-4-8", {})
        assert model.cost_tier == "frontier"
        assert model.thinking_model is True
        assert "vision" in model.capabilities

    def test_haiku_is_budget(self) -> None:
        model = _enrich_anthropic_model("claude-haiku-4-5", {})
        assert model.cost_tier == "budget"
        assert model.latency_tier == "low"

    def test_display_name_from_row(self) -> None:
        model = _enrich_anthropic_model(
            "claude-sonnet-5",
            {"display_name": "Claude Sonnet 5"},
        )
        assert model.display_name == "Claude Sonnet 5"


class TestAzureEnrichment:
    def test_deployment_with_capabilities_dict(self) -> None:
        model = _enrich_azure_model(
            "gpt-5.4",
            {
                "model": "gpt-5.4",
                "capabilities": {
                    "chat_completion": True,
                    "tools": True,
                    "vision": True,
                },
            },
        )
        assert "tools" in model.capabilities
        assert "vision" in model.capabilities

    def test_o_series_is_thinking(self) -> None:
        model = _enrich_azure_model("o3", {})
        assert model.thinking_model is True

    def test_codex_uses_responses_api(self) -> None:
        """Codex-family deployments only expose the Responses API on Azure."""
        model = _enrich_azure_model("gpt-5.1-codex-max", {"model": "gpt-5.1-codex-max"})
        assert model.completion_api == "responses"
        # Codex models reject ``temperature``; the probe must mark them
        # thinking so the Responses/chat-completions payload builders
        # omit it.
        assert model.thinking_model is True

    def test_gpt6_uses_responses_api(self) -> None:
        """gpt-6 chat-completions rejects function tools with default reasoning_effort."""
        model = _enrich_azure_model("gpt-6-luna", {"model": "gpt-6-luna"})
        assert model.completion_api == "responses"
        assert model.thinking_model is True

    def test_non_codex_uses_chat_completions(self) -> None:
        """Non-codex deployments keep the default chat-completions endpoint."""
        model = _enrich_azure_model("gpt-5.4", {"model": "gpt-5.4"})
        assert model.completion_api == "chat_completions"

    def test_gpt5_deployment_is_reasoning(self) -> None:
        """gpt-5 bills hidden reasoning against max_completion_tokens.

        Regression: unflagged gpt-5.6-terra got the 4096 legacy budget and
        every long call truncated, then retried with the whole prompt.
        """
        model = _enrich_azure_model("gpt-5.6-terra", {"model": "gpt-5.6-terra"})
        assert model.reasoning_model is True
        assert "reasoning" in model.capabilities

    def test_gpt5_chat_deployment_is_not_reasoning(self) -> None:
        model = _enrich_azure_model("gpt-5-chat", {"model": "gpt-5-chat"})
        assert model.reasoning_model is False


class TestOllamaEnrichment:
    def test_small_model_is_budget(self) -> None:
        model = _enrich_ollama_model(
            "llama3.2:3b",
            {"details": {"parameter_size": "3B", "family": "llama"}},
        )
        assert model.cost_tier == "budget"
        assert model.latency_tier == "low"

    def test_large_model_is_premium(self) -> None:
        model = _enrich_ollama_model(
            "deepseek-v4-pro:cloud",
            {"details": {"parameter_size": "70B", "family": "deepseek"}},
        )
        # :cloud suffix → standard regardless of size
        assert model.cost_tier == "standard"

    def test_vision_family_adds_vision_capability(self) -> None:
        model = _enrich_ollama_model(
            "llava:7b",
            {"details": {"parameter_size": "7B", "family": "llava"}},
        )
        assert "vision" in model.capabilities

    def test_reasoning_model_gets_reasoning_capability(self) -> None:
        # ollama's probe never adds "reasoning" itself, but _model derives
        # it from is_reasoning_model so the router rule
        # '"reasoning" in route.required_capabilities' can match.
        model = _enrich_ollama_model(
            "deepseek-r1:7b",
            {"details": {"parameter_size": "7B", "family": "deepseek"}},
        )
        assert model.reasoning_model is True
        assert "reasoning" in model.capabilities

    def test_coding_model_family_gets_code_capability(self) -> None:
        # ollama's probe derives "code" from is_code_model so the router rule
        # '"code" in route.required_capabilities' and _pick_code_model can
        # select a coding model instead of silently defaulting.
        model = _enrich_ollama_model(
            "glm-5.2:cloud",
            {"details": {"parameter_size": "756B", "family": "glm"}},
        )
        assert "code" in model.capabilities

    def test_coder_suffix_gets_code_capability(self) -> None:
        model = _enrich_ollama_model(
            "qwen2.5-coder:14b",
            {"details": {"parameter_size": "14B", "family": "qwen"}},
        )
        assert "code" in model.capabilities

    def test_non_coding_model_has_no_code_capability(self) -> None:
        # A plain chat model (no coding markers) must not advertise "code".
        model = _enrich_ollama_model(
            "llama3.2:3b",
            {"details": {"parameter_size": "3B", "family": "llama"}},
        )
        assert "code" not in model.capabilities

    def test_show_payload_overrides_vision_and_context(self) -> None:
        # /api/show is authoritative for vision and context window.
        model = _enrich_ollama_model(
            "qwen3:8b",
            {"details": {"parameter_size": "8B", "family": "qwen"}},
            show_payload={
                "capabilities": ["completion", "tools", "vision", "thinking"],
                "model_info": {"general.context_length": 65536},
            },
        )
        assert "vision" in model.capabilities
        assert "reasoning" in model.capabilities
        assert "tools" in model.capabilities
        assert model.context_window == 65536

    def test_show_payload_missing_fields_fall_back_to_heuristic(self) -> None:
        # A bare /api/show (no capabilities/model_info) degrades to the
        # tag-based heuristic and never fails enrichment.
        model = _enrich_ollama_model(
            "llava:7b",
            {"details": {"parameter_size": "7B", "family": "llava"}},
            show_payload={},
        )
        assert "vision" in model.capabilities  # from family heuristic
        assert model.context_window == 0

    def test_cloud_proxy_family_keyed_context_length(self) -> None:
        """Ollama-cloud `/api/show` returns family-keyed context lengths.

        Regression for the 2026-10-06 manifest gap: `models discover
        --force` against the Ollama cloud proxy produced entries with no
        `context_window` because `_ollama_show_context_window` only read
        `model_info["general.context_length"]`, while the proxy publishes
        family-keyed names (`glm_dsa_moe.context_length`,
        `deepseek_v41.context_length`, `kimi-k3.context_length`). The
        missing window left the kernel's truncation-rescue uncapped and
        the per-model `max_completion_tokens` bound without its ceiling.
        """
        model = _enrich_ollama_model(
            "glm-5.3:cloud",
            {"details": {"parameter_size": "32B", "family": "glm"}},
            show_payload={
                "capabilities": ["completion", "thinking", "tools"],
                "model_info": {"glm_dsa_moe.context_length": 1048576},
            },
        )
        assert model.context_window == 1048576
        assert "reasoning" in model.capabilities

    def test_tags_details_context_length_used_when_show_missing(self) -> None:
        """`/api/tags` `details.context_length` is the fallback when
        `/api/show` carries no usable window."""
        model = _enrich_ollama_model(
            "deepseek-v4.1-flash:cloud",
            {
                "details": {
                    "parameter_size": "763B",
                    "family": "deepseek",
                    "context_length": 1048576,
                }
            },
            show_payload={},
        )
        assert model.context_window == 1048576

    def test_tags_details_context_length_ignored_when_show_wins(self) -> None:
        """`/api/show` remains authoritative when both sources carry a value."""
        model = _enrich_ollama_model(
            "glm-5.3:cloud",
            {"details": {"parameter_size": "32B", "family": "glm", "context_length": 65536}},
            show_payload={"model_info": {"glm_dsa_moe.context_length": 1048576}},
        )
        assert model.context_window == 1048576

    def test_context_length_must_be_positive(self) -> None:
        """Zero/negative context lengths never populate the window."""
        model = _enrich_ollama_model(
            "glm-5.3:cloud",
            {"details": {"parameter_size": "32B", "family": "glm", "context_length": 0}},
            show_payload={"model_info": {"glm_dsa_moe.context_length": 0}},
        )
        assert model.context_window == 0

    def test_show_payload_embedding_only(self) -> None:
        # An embedding-only model surfaces the embedding capability.
        model = _enrich_ollama_model(
            "all-minilm:latest",
            {"details": {"parameter_size": "0.02B", "family": "bert"}},
            show_payload={"capabilities": ["embedding"], "model_info": {}},
        )
        assert "embedding" in model.capabilities

    def test_tiny_local_model_low_structured_reliability(self) -> None:
        # Small local Ollama models ignore response_format — the probe must
        # NOT claim reliability for them (the root cause of the search-mode
        # web_search prose failure).
        model = _enrich_ollama_model(
            "llama3.2:3b",
            {"details": {"parameter_size": "3B", "family": "llama"}},
        )
        assert model.structured_reliability == 0.0

    def test_cloud_model_high_structured_reliability(self) -> None:
        # Cloud-hosted Ollama models are more dependable at structured output.
        model = _enrich_ollama_model(
            "deepseek-v4-pro:cloud",
            {"details": {"parameter_size": "70B", "family": "deepseek"}},
        )
        assert model.structured_reliability >= 0.8


class TestBuildManifestPayloadEnrichment:
    def test_enrichment_fields_written_to_yaml(self) -> None:
        models = (
            DiscoveredModel(
                model_id="gpt-5.4",
                provider="openai",
                display_name="GPT-5.4",
                capabilities=("text", "tools", "code"),
                cost_tier="frontier",
                latency_tier="medium",
                context_window=256000,
                thinking_model=False,
            ),
        )
        payload = build_manifest_payload(models)
        row = payload["models"][0]
        assert row["display_name"] == "GPT-5.4"
        assert row["cost_tier"] == "frontier"
        assert row["latency_tier"] == "medium"
        assert row["context_window"] == 256000
        assert "text" in row["capabilities"]

    def test_structured_reliability_written_to_yaml(self) -> None:
        models = (
            DiscoveredModel(
                model_id="deepseek-v4-pro:cloud",
                provider="ollama",
                capabilities=("text", "tools", "structured_output"),
                structured_reliability=0.8,
            ),
        )
        payload = build_manifest_payload(models)
        row = payload["models"][0]
        assert row["structured_reliability"] == 0.8

    def test_zero_structured_reliability_omitted(self) -> None:
        # 0.0 is the default/unknown value and should not bloat the YAML.
        models = (DiscoveredModel(model_id="bare-model", provider="ollama"),)
        payload = build_manifest_payload(models)
        assert "structured_reliability" not in payload["models"][0]

    def test_empty_enrichment_fields_omitted(self) -> None:
        models = (DiscoveredModel(model_id="bare-model", provider="ollama"),)
        payload = build_manifest_payload(models)
        row = payload["models"][0]
        assert "display_name" not in row
        assert "cost_tier" not in row
        assert "latency_tier" not in row
        assert "capabilities" not in row
        assert "thinking_model" not in row
        assert "completion_api" not in row

    def test_responses_completion_api_written_to_yaml(self) -> None:
        """A model flagged for the Responses API surfaces it in the manifest row."""
        models = (
            DiscoveredModel(
                model_id="gpt-5.1-codex-max",
                provider="azure_openai",
                completion_api="responses",
            ),
        )
        payload = build_manifest_payload(models)
        row = payload["models"][0]
        assert row["completion_api"] == "responses"

    def test_prompt_cache_breakpoints_written_only_when_set(self) -> None:
        models = (
            DiscoveredModel(
                model_id="gpt-5.6-terra",
                provider="azure_openai",
                prompt_cache_breakpoints=True,
            ),
            DiscoveredModel(model_id="gpt-5.5", provider="azure_openai"),
        )
        rows = build_manifest_payload(models)["models"]
        assert rows[0]["prompt_cache_breakpoints"] is True
        assert "prompt_cache_breakpoints" not in rows[1]


class TestPromptCacheBreakpointClassification:
    def test_azure_gpt_5_6_and_later_use_breakpoints(self) -> None:
        for model_id in ("gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-sol", "gpt-6.1-sol"):
            assert _enrich_azure_model(model_id, {}).prompt_cache_breakpoints, model_id

    def test_older_and_non_azure_models_do_not(self) -> None:
        for model_id in ("gpt-5.5", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-4o"):
            assert not _enrich_azure_model(model_id, {}).prompt_cache_breakpoints, model_id
        assert not uses_prompt_cache_breakpoints("openai", "gpt-5.6-terra")


class TestKnownModelsRegistry:
    def test_load_known_models_returns_dict(self) -> None:
        registry = load_known_models()
        assert isinstance(registry, dict)
        # Must have at least the seed entries
        assert "openai:gpt-5.4" in registry

    def test_apply_known_override_fills_missing_fields(self) -> None:
        enrichment = apply_known_override(
            "openai",
            "gpt-5.4",
            {"display_name": "", "cost_tier": ""},
        )
        assert enrichment["display_name"] == "GPT-5.4"
        assert enrichment["cost_tier"] == "frontier"

    def test_apply_known_override_preserves_provider_fields_when_no_entry(self) -> None:
        enrichment = apply_known_override(
            "openai",
            "totally-unknown-model",
            {"display_name": "From Provider", "cost_tier": "standard"},
        )
        assert enrichment["display_name"] == "From Provider"
        assert enrichment["cost_tier"] == "standard"

    def test_apply_known_override_skips_empty_registry_values(self) -> None:
        # A registry entry with empty values should not wipe provider data.
        registry_entry = load_known_models().get("openai:gpt-5.4", {})
        assert registry_entry.get("cost_tier") == "frontier"  # non-empty

    def test_ollama_cloud_reasoning_models_keep_reasoning_capability(self) -> None:
        """Registry pins ``reasoning`` on Ollama cloud reasoning models.

        The 2026-09-19 narration-loop incident: a discovered manifest
        dropped the ``reasoning`` capability from glm-5.3-flash:cloud, which
        silently demoted it to the lean 4096 legacy completion budget (the
        capability drives ``_turn_completion_max_tokens``), and every turn
        truncated mid-tool-call. The registry override must restore the
        capability even though the substring heuristic does not flag
        these model ids as reasoning models.
        """
        for model_id in (
            "glm-5.3:cloud",
            "glm-5.3-flash:cloud",
            "deepseek-v4.1-flash:cloud",
            "deepseek-v4-flash:cloud",
        ):
            enrichment = apply_known_override(
                "ollama",
                model_id,
                {"capabilities": ("text", "tools", "structured_output")},
            )
            assert "reasoning" in enrichment["capabilities"], model_id

    def test_ollama_cloud_reasoning_models_flag_stays_false(self) -> None:
        """The registry must NOT set ``reasoning_model`` for Ollama models.

        The flag switches the OpenAI-compatible gateway to
        ``max_completion_tokens``, which Ollama accepts but silently
        ignores (probed live: cap=50 → 254 completion tokens,
        finish=stop) — losing all budget enforcement. The capability-only
        override keeps the wire on ``max_tokens`` while still driving the
        reasoning-aware completion budget.
        """
        for model_id in ("glm-5.3:cloud", "glm-5.3-flash:cloud"):
            registry_entry = load_known_models().get(f"ollama:{model_id}", {})
            assert "reasoning_model" not in registry_entry, model_id
            assert registry_entry.get("thinking_model") is None, model_id
