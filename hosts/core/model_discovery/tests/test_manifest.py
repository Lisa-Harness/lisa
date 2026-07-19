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

    def test_default_low_latency_wins(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="gpt-4o-mini", provider="openai"),
        )
        payload = build_manifest_payload(models)
        # discover no longer emits mode_models; default role is via designations
        assert "mode_models" not in payload
        default_row = next(r for r in payload["models"] if r["model_id"] == "gpt-4o-mini")
        assert "default" in default_row["designations"]

    def test_deep_from_reasoning(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="o3-mini", provider="openai", reasoning_model=True),
        )
        payload = build_manifest_payload(models)
        assert "mode_models" not in payload
        deep_row = next(r for r in payload["models"] if r["model_id"] == "o3-mini")
        assert "deep" in deep_row["designations"]
        assert "reasoning" in deep_row["designations"]
        assert deep_row.get("reasoning_model") is True

    def test_dedup_same_provider_same_id(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1

    def test_dedup_case_insensitive(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="GPT-4O", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1

    def test_dedup_different_providers_same_id_drops_second(self) -> None:
        """The manifest loader rejects same model_id across providers,
        so _dedupe must collapse them (keeping the first-seen entry)."""

        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="gpt-4o", provider="azure_openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1
        assert payload["models"][0]["provider"] == "openai"

    def test_falls_back_to_first_non_reasoning(self) -> None:
        models = (
            DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
        )
        payload = build_manifest_payload(models)
        default_row = next(r for r in payload["models"] if "default" in r.get("designations", ()))
        assert default_row["model_id"] == "gpt-4o"

    def test_falls_back_to_first_model_if_all_reasoning(self) -> None:
        models = (
            DiscoveredModel(model_id="o3-mini", provider="openai", reasoning_model=True),
            DiscoveredModel(model_id="o1", provider="openai", reasoning_model=True),
        )
        payload = build_manifest_payload(models)
        # default = first model (all reasoning), deep = first reasoning
        default_row = next(r for r in payload["models"] if "default" in r.get("designations", ()))
        assert default_row["model_id"] == "o3-mini"
        deep_row = next(r for r in payload["models"] if "deep" in r.get("designations", ()))
        assert deep_row["model_id"] == "o3-mini"

    def test_mode_models_not_emitted(self) -> None:
        """discover must not emit mode_models; designations carry role info."""
        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
        )
        payload = build_manifest_payload(models)
        assert "mode_models" not in payload
        default_row = next(r for r in payload["models"] if "default" in r.get("designations", ()))
        assert default_row["model_id"] == "gpt-4o"
        deep_row = next(r for r in payload["models"] if "deep" in r.get("designations", ()))
        assert deep_row["model_id"] == "o3-pro"


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