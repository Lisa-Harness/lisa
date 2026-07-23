"""Tests for manifest construction, rendering, and atomic writes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from hosts.core.model_discovery.manifest import (
    build_manifest_payload,
    render_manifest_yaml,
    write_manifest_yaml,
)
from hosts.core.model_discovery.types import DiscoveredModel


class TestBuildManifestPayload:
    def test_empty_models(self) -> None:
        payload = build_manifest_payload(())
        assert payload == {"version": 1, "models": []}

    def test_cloud_gpt5_cheapest_size_wins_default(self) -> None:
        """Cloud default = cheapest gpt-5-family model: nano < mini < full."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
            DiscoveredModel(model_id="gpt-5.4", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5-mini"}
        assert payload["roles"]["default"] == "gpt-5-mini"

    def test_cloud_gpt5_nano_beats_mini(self) -> None:
        """nano is cheaper than mini; both beat full gpt-5."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
            DiscoveredModel(model_id="gpt-5-nano", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5-nano"}

    def test_cloud_gpt5_full_when_no_mini_nano(self) -> None:
        """If only full gpt-5 variants exist, the lowest variant wins."""
        models = (
            DiscoveredModel(model_id="gpt-5.4", provider="openai"),
            DiscoveredModel(model_id="gpt-5", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5"}

    def test_cloud_falls_back_to_first_cloud_when_no_gpt5(self) -> None:
        """No gpt-5 family probed → first cloud model in discovery order."""
        models = (
            DiscoveredModel(model_id="claude-opus-4", provider="anthropic"),
            DiscoveredModel(model_id="gemini-2.5-pro", provider="google"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "claude-opus-4"}

    def test_local_mid_range_wins_when_local_present(self) -> None:
        """Local present → default is the median by context_window (mid-range)."""
        models = (
            DiscoveredModel(model_id="small:7b", provider="ollama", context_window=8192),
            DiscoveredModel(model_id="big:70b", provider="ollama", context_window=65536),
            DiscoveredModel(model_id="mid:13b", provider="ollama", context_window=16384),
        )
        payload = build_manifest_payload(models)
        # Sorted by context_window: 8192, 16384, 65536 → median = mid:13b.
        assert payload["mode_models"] == {"default": "mid:13b"}

    def test_local_even_count_picks_lower_median(self) -> None:
        """Even count: the lower of the two middle entries wins (cheaper)."""
        models = (
            DiscoveredModel(model_id="a:1b", provider="ollama", context_window=2048),
            DiscoveredModel(model_id="b:7b", provider="ollama", context_window=8192),
            DiscoveredModel(model_id="c:13b", provider="ollama", context_window=16384),
            DiscoveredModel(model_id="d:70b", provider="ollama", context_window=65536),
        )
        payload = build_manifest_payload(models)
        # Sorted: 2048, 8192, 16384, 65536 → lower median = b:7b (8192).
        assert payload["mode_models"] == {"default": "b:7b"}

    def test_local_tiebreak_prefers_reasoning(self) -> None:
        """Equal context_window → reasoning_model wins the tie."""
        models = (
            DiscoveredModel(model_id="plain:70b", provider="ollama", context_window=65536),
            DiscoveredModel(
                model_id="reason:70b",
                provider="ollama",
                context_window=65536,
                reasoning_model=True,
            ),
        )
        payload = build_manifest_payload(models)
        # Two models, even count, lower median. Same context_window, reasoning
        # sorts earlier (not reasoning=False sorts first), so reason:70b wins.
        assert payload["mode_models"] == {"default": "reason:70b"}

    def test_local_wins_default_over_cloud_when_both_present(self) -> None:
        """Local (ollama) present → local mid-range wins as default, not cloud."""
        models = (
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
            DiscoveredModel(model_id="small:7b", provider="ollama", context_window=8192),
            DiscoveredModel(model_id="big:70b", provider="ollama", context_window=65536),
        )
        payload = build_manifest_payload(models)
        # Two local models, even count, lower median = small:7b.
        assert payload["mode_models"] == {"default": "small:7b"}

    def test_deep_first_reasoning_any_provider(self) -> None:
        """deep = first reasoning model regardless of provider."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(
                model_id="deepseek-r1:70b",
                provider="ollama",
                reasoning_model=True,
                context_window=65536,
            ),
        )
        payload = build_manifest_payload(models)
        assert payload["roles"]["deep"] == "deepseek-r1:70b"

    def test_deep_falls_back_to_default_when_no_reasoning(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["roles"]["deep"] == "gpt-5"

    def test_emits_mode_models_and_roles_no_designations(self) -> None:
        """Output shape: mode_models + roles at top level; no per-model designations."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(
                model_id="o3-pro",
                provider="openai",
                reasoning_model=True,
            ),
        )
        payload = build_manifest_payload(models)
        assert "mode_models" in payload
        assert "roles" in payload
        assert payload["roles"] == {"default": "gpt-5", "deep": "o3-pro"}
        for row in payload["models"]:
            assert "designations" not in row

    def test_dedup_same_provider_same_id(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1

    def test_dedup_case_insensitive(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="GPT-5", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1

    def test_dedup_different_providers_same_id_drops_second(self) -> None:
        """The manifest loader rejects same model_id across providers,
        so _dedupe must collapse them (keeping the first-seen entry)."""

        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5", provider="azure_openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1
        assert payload["models"][0]["provider"] == "openai"


class TestRouterBlock:
    """Tests for the pack-seeded ``router:`` block in ``build_manifest_payload``.

    The router block is seeded from the first pack-declared router config
    (``lisa.extensions.model_router_configs``). Symbolic refs (``"default"``,
    ``"deep"``) are resolved to concrete model ids. Rules referencing unknown
    models are stripped. When no pack declares a config, no ``router`` key
    is emitted.
    """

    def test_no_router_block_when_no_pack_config(self) -> None:
        """When no pack declares a router config, no router key is emitted."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=None,
        ):
            payload = build_manifest_payload(models)
        assert "router" not in payload

    def test_router_block_seeded_from_pack_config(self) -> None:
        """Pack config with symbolic refs resolved to concrete model ids."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
        )
        pack_config = {
            "backend": "lisa-router",
            "rules": [
                {"when": 'required_capabilities includes "code"', "use": "deep"},
                {"when": "iteration > 1 and last_tool_result is not None", "use": "deep"},
                {"when": "tool_calls_so_far >= 5", "use": "default"},
            ],
            "fallback": "default",
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        assert "router" in payload
        router = payload["router"]
        assert router["backend"] == "lisa-router"
        # Symbolic refs resolved: "deep" → o3-pro, "default" → gpt-5.
        assert router["rules"][0]["use"] == "o3-pro"
        assert router["rules"][2]["use"] == "gpt-5"
        assert router["fallback"] == "gpt-5"

    def test_router_strips_rules_referencing_unknown_models(self) -> None:
        """Rules whose ``use`` cannot be resolved are dropped."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
        )
        pack_config = {
            "backend": "lisa-router",
            "rules": [
                {"when": "iteration > 1", "use": "deep"},  # resolvable
                {"when": "tool_calls_so_far >= 3", "use": "gpt-99-nonexistent"},  # stripped
            ],
            "fallback": "default",
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert len(router["rules"]) == 1
        assert router["rules"][0]["use"] == "o3-pro"
        assert router["fallback"] == "gpt-5"

    def test_router_strips_fallback_referencing_unknown_model(self) -> None:
        """Fallback referencing an unknown model is dropped (not the whole block)."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [{"when": "iteration > 1", "use": "default"}],
            "fallback": "gpt-99-nonexistent",
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert "fallback" not in router
        assert len(router["rules"]) == 1

    def test_router_block_dropped_when_no_rules_survive(self) -> None:
        """When all rules are stripped and no fallback resolves, no router key."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [{"when": "iteration > 1", "use": "gpt-99-nonexistent"}],
            "fallback": "also-nonexistent",
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        # No rules survive, no fallback, but backend is set. The block is
        # still emitted (backend alone is valid — just selects the router
        # backend with no rules; the router will always return None).
        assert "router" in payload
        assert payload["router"]["backend"] == "lisa-router"
        assert "rules" not in payload["router"]

    def test_router_block_resolves_literal_model_id_refs(self) -> None:
        """Literal model ids in use/fallback pass through when in the discovered set."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
        )
        pack_config = {
            "backend": "lisa-router",
            "rules": [{"when": "iteration > 1", "use": "gpt-5-mini"}],
            "fallback": "gpt-5",
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert router["rules"][0]["use"] == "gpt-5-mini"
        assert router["fallback"] == "gpt-5"


class TestRenderManifestYaml:
    def test_roundtrip(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="o3-mini", provider="openai", reasoning_model=True),
        )
        payload = build_manifest_payload(models)
        rendered = render_manifest_yaml(payload)
        assert rendered.startswith("# generated-by: lisa /models discover")
        parsed = yaml.safe_load(rendered)
        assert parsed == payload

    def test_is_valid_yaml(self) -> None:
        payload = {"version": 1, "models": []}
        rendered = render_manifest_yaml(payload)
        parsed = yaml.safe_load(rendered)
        assert parsed == payload


class TestWriteManifestYaml:
    def test_atomic_write(self, tmp_path: Path) -> None:
        path = tmp_path / "models.proposed.yaml"
        payload = build_manifest_payload(
            (DiscoveredModel(model_id="gpt-4o", provider="openai"),)
        )
        write_manifest_yaml(path, payload)
        assert path.exists()
        assert not (tmp_path / "models.proposed.yaml.tmp").exists()
        content = path.read_text(encoding="utf-8")
        assert "gpt-4o" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "models.proposed.yaml"
        write_manifest_yaml(path, {"version": 1, "models": []})
        assert path.exists()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "models.proposed.yaml"
        path.write_text("old", encoding="utf-8")
        write_manifest_yaml(
            path,
            build_manifest_payload(
                (DiscoveredModel(model_id="new-model", provider="openai"),)
            ),
        )
        assert "new-model" in path.read_text(encoding="utf-8")
        assert "old" not in path.read_text(encoding="utf-8")