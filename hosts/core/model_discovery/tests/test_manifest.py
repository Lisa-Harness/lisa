"""Tests for manifest construction, rendering, and atomic writes."""

from __future__ import annotations

from pathlib import Path

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

    def test_cloud_gpt5_lowest_tier_wins_default(self) -> None:
        """Cloud default = lowest gpt-5-family model; mini/nano allowed."""
        models = (
            DiscoveredModel(model_id="gpt-5.4", provider="openai"),
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5"}
        assert payload["roles"]["default"] == "gpt-5"

    def test_cloud_gpt5_mini_preferred_when_only_mini_available(self) -> None:
        """If only gpt-5-mini/nano are present, the mini still wins (gpt-5+ family)."""
        models = (
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
            DiscoveredModel(model_id="gpt-5-nano", provider="openai"),
        )
        payload = build_manifest_payload(models)
        # Full gpt-5 absent → first gpt-5-family model in discovery order.
        assert payload["mode_models"] == {"default": "gpt-5-mini"}

    def test_cloud_falls_back_to_first_cloud_when_no_gpt5(self) -> None:
        """No gpt-5 family probed → first cloud model in discovery order."""
        models = (
            DiscoveredModel(model_id="claude-opus-4", provider="anthropic"),
            DiscoveredModel(model_id="gemini-2.5-pro", provider="google"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "claude-opus-4"}

    def test_local_highest_context_window_wins(self) -> None:
        """Local-only: highest context_window, tiebreak reasoning, then order."""
        models = (
            DiscoveredModel(model_id="small:7b", provider="ollama", context_window=8192),
            DiscoveredModel(model_id="big:70b", provider="ollama", context_window=65536),
            DiscoveredModel(model_id="mid:13b", provider="ollama", context_window=16384),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "big:70b"}

    def test_local_tiebreak_prefers_reasoning(self) -> None:
        """Equal context_window → reasoning_model wins."""
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
        assert payload["mode_models"] == {"default": "reason:70b"}

    def test_mixed_cloud_wins_default_over_local(self) -> None:
        """When both cloud and local present, cloud gpt-5 wins as default."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="big:70b", provider="ollama", context_window=65536),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5"}
        assert payload["roles"]["default"] == "gpt-5"

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