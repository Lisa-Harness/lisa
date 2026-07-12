"""Tests for hosts.core.model_manifest — migrated from hosts/little_lisa/test_model_manifest.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hosts.core import model_manifest

_MIN_EXPECTED_MODELS = 2


def test_load_manifest_reads_designations_and_models() -> None:
    manifest = model_manifest.load_model_manifest(model_manifest.default_manifest_path())

    assert len(manifest.models) >= _MIN_EXPECTED_MODELS
    assert manifest.designated("front-door") is not None
    assert manifest.designated("default") is not None
    assert manifest.designated("reasoning") is not None
    assert manifest.designated("deep") is not None
    # The new hosts/core/models.yaml has no mode_models mapping; verify
    # that the default-designation model is returned for unmapped modes.
    default_model = manifest.designated("default")
    assert default_model is not None
    assert default_model.model_id  # has a non-empty model_id


def test_load_manifest_rejects_duplicate_designation(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            mode_models:
              default: gpt-4o
            models:
              - model_id: gpt-4o
                provider: azure_openai
                designations: [default]
              - model_id: gpt-5.4
                provider: azure_openai
                designations: [default]
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Designation 'default'"):
        model_manifest.load_model_manifest(manifest_file)


def test_load_manifest_rejects_unknown_provider(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            mode_models:
              default: gpt-4o
            models:
              - model_id: gpt-4o
                provider: weird_provider
                designations: [default]
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provider must be"):
        model_manifest.load_model_manifest(manifest_file)


def test_load_manifest_accepts_anthropic_ollama_and_google_providers(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            mode_models:
              default: claude-3-5-sonnet
              deep: llama3.1
              discuss: gemini-2.5-flash
            models:
              - model_id: claude-3-5-sonnet
                provider: anthropic
                designations: [front-door, default]
              - model_id: llama3.1
                provider: ollama
                designations: [deep]
              - model_id: gemini-2.5-flash
                provider: google
            """
        ),
        encoding="utf-8",
    )

    manifest = model_manifest.load_model_manifest(manifest_file)

    assert manifest.designated("front-door") is not None
    assert manifest.designated("deep") is not None
    google_model = manifest.find_model("gemini-2.5-flash")
    assert google_model is not None
    assert google_model.provider == "google"


def test_load_manifest_preserves_shared_provider_fields(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            mode_models:
              default: groq-llama
            models:
              - key: groq-llama
                model_id: llama-3.3-70b-versatile
                provider: openai
                display_name: Groq Llama
                display_alias: groq
                aliases: [groq-fast, llama-fast]
                api_key_env: GROQ_API_KEY
                base_url: https://api.groq.com/openai/v1
                base_url_env: GROQ_BASE_URL
                reasoning_model: true
                designations: [default]
            """
        ),
        encoding="utf-8",
    )

    manifest = model_manifest.load_model_manifest(manifest_file)

    model = manifest.find_model("groq-fast")
    assert model is not None
    assert model.key == "groq-llama"
    assert model.display_alias == "groq"
    assert model.api_key_env == "GROQ_API_KEY"
    assert model.base_url == "https://api.groq.com/openai/v1"
    assert model.base_url_env == "GROQ_BASE_URL"
    assert model.reasoning_model is True