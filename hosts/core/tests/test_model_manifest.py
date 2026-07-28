"""Tests for hosts.core.model_manifest — migrated from hosts/little_lisa/test_model_manifest.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hosts.core import model_manifest

_MIN_EXPECTED_MODELS = 2
_CONTEXT_WINDOW_TOKENS = 131072


def test_load_manifest_reads_modes_and_models(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            mode_models:
              default: test-default
            models:
              - model_id: test-default
                provider: ollama
              - model_id: test-deep
                provider: ollama
                reasoning_model: true
            """
        ),
        encoding="utf-8",
    )

    manifest = model_manifest.load_model_manifest(manifest_file)

    assert len(manifest.models) >= _MIN_EXPECTED_MODELS
    # resolve() precedence: mode_models wins for modes, then find_model.
    assert manifest.resolve("default").model_id == "test-default"
    assert manifest.resolve("test-deep").model_id == "test-deep"


def test_load_manifest_requires_mode_models(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            models:
              - model_id: m1
                provider: ollama
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must include a 'mode_models' mapping"):
        model_manifest.load_model_manifest(manifest_file)


def test_load_manifest_rejects_unknown_mode_model_ref(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            mode_models:
              default: gpt-4o
              discuss: nope
            models:
              - model_id: gpt-4o
                provider: azure_openai
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="references unknown model"):
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
            roles:
              deep: llama3.1
            models:
              - model_id: claude-3-5-sonnet
                provider: anthropic
              - model_id: llama3.1
                provider: ollama
              - model_id: gemini-2.5-flash
                provider: google
            """
        ),
        encoding="utf-8",
    )

    manifest = model_manifest.load_model_manifest(manifest_file)

    assert manifest.resolve("default").model_id == "claude-3-5-sonnet"
    assert manifest.resolve("deep").model_id == "llama3.1"
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
                context_window: 131072
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
    assert model.context_window == _CONTEXT_WINDOW_TOKENS


def test_load_manifest_rejects_negative_context_window(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        textwrap.dedent(
            """\
            version: 1
            mode_models:
              default: test-model
            models:
              - model_id: test-model
                provider: ollama
                context_window: -1
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context_window must be a non-negative integer"):
        model_manifest.load_model_manifest(manifest_file)
