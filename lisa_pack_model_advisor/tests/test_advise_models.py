"""Tests for the advise_models operation.

Tests inject a real content-registry loader (`read_authored_text_fn`) that
reads the actual authored prompt files from the pack's content root. Per the
pack-content rule, the operation has no fallbacks — tests must wire the
loader exactly as production hosts do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from kernel.models import OperationExecutionContext, OperationResult
from lisa_pack_model_advisor.operations.advise_models.advise_models import (
    _extract_yaml_block,
    _render_diff_summary,
    _validate_manifest_yaml,
    execute,
)

# Pack content root — the directory containing operations/advise_models/prompts/.
_CONTENT_ROOT = Path(__file__).resolve().parents[1]


def _make_loader() -> Any:
    """Return a ``read_authored_text`` callable that reads real prompt files."""

    def read_authored_text(relative_path: str) -> str:
        full_path = _CONTENT_ROOT / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Authored text not found: {full_path}")
        return full_path.read_text(encoding="utf-8")

    return read_authored_text


@dataclass
class _FakeModel:
    provider: str = "openai"
    model_id: str = "gpt-chat-latest"
    display_name: str = "GPT Chat"
    reasoning_model: bool = False
    thinking_model: bool = False


@dataclass
class _FakeCompletion:
    content: str
    provider: str = "openai"
    model_id: str = "gpt-chat-latest"
    display_name: str = "GPT Chat"
    request_id: str = "test"
    tool_calls: list = None
    usage: Any = None


_VALID_MANIFEST = """\
version: 1
models:
  - model_id: gpt-chat-latest
    provider: openai
  - model_id: o3
    provider: openai
    reasoning_model: true
mode_models:
  default: gpt-chat-latest
  deep: o3
"""

_EDITED_MANIFEST = """\
version: 1
models:
  - model_id: gpt-chat-latest
    provider: openai
    cost_tier: standard
  - model_id: o3
    provider: openai
    reasoning_model: true
    cost_tier: premium
mode_models:
  default: gpt-chat-latest
  deep: o3
"""


class TestExtractYamlBlock:
    def test_extracts_fenced_block(self) -> None:
        text = "Here you go:\n```yaml\nversion: 1\nmodels: []\n```\nDone."
        assert "version: 1" in _extract_yaml_block(text)

    def test_extracts_unfenced_yaml_starting_with_version(self) -> None:
        text = "version: 1\nmodels: []"
        assert _extract_yaml_block(text) == "version: 1\nmodels: []"

    def test_returns_empty_when_no_yaml(self) -> None:
        assert _extract_yaml_block("just prose, no yaml here") == ""


class TestValidateManifestYaml:
    def test_valid_manifest_passes(self) -> None:
        assert _validate_manifest_yaml(_VALID_MANIFEST) == ""

    def test_invalid_yaml_returns_error(self) -> None:
        error = _validate_manifest_yaml("not: valid: yaml: at: all")
        assert "YAML" in error or "mapping" in error.lower()

    def test_empty_models_returns_error(self) -> None:
        error = _validate_manifest_yaml("version: 1\nmodels: []")
        assert "non-empty" in error

    def test_duplicate_model_id_returns_error(self) -> None:
        dup_manifest = (
            "version: 1\n"
            "models:\n"
            "  - model_id: dup\n    provider: openai\n"
            "  - model_id: dup\n    provider: openai\n"
        )
        error = _validate_manifest_yaml(dup_manifest)
        assert "dup" in error


class TestRenderDiffSummary:
    def test_detects_cost_tier_change(self) -> None:
        summary = _render_diff_summary(_VALID_MANIFEST, _EDITED_MANIFEST, Path("models.yaml"))
        assert "Manifest updated" in summary
        assert "gpt-chat-latest" in summary
        assert "cost_tier" in summary

    def test_no_changes_detected(self) -> None:
        summary = _render_diff_summary(_VALID_MANIFEST, _VALID_MANIFEST, Path("models.yaml"))
        assert "no changes detected" in summary


class TestExecuteOperation:
    @pytest.mark.asyncio
    async def test_empty_instruction_returns_usage(self, tmp_path: Path) -> None:
        ctx = OperationExecutionContext(
            query="",
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "Usage:" in result.output_text

    @pytest.mark.asyncio
    async def test_no_manifest_returns_error(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))
        ctx = OperationExecutionContext(
            query="make gpt-5 the default",
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "No model manifest" in result.output_text

    @pytest.mark.asyncio
    async def test_writes_edited_manifest(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))
        proposal_path = tmp_path / "models.proposed.yaml"
        proposal_path.write_text(_VALID_MANIFEST, encoding="utf-8")

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content=f"```yaml\n{_EDITED_MANIFEST}```")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="mark o3 as premium tier",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "Manifest updated" in result.output_text
        written = proposal_path.read_text(encoding="utf-8")
        assert "premium" in written

    @pytest.mark.asyncio
    async def test_invalid_llm_output_returns_error(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))
        proposal_path = tmp_path / "models.proposed.yaml"
        proposal_path.write_text(_VALID_MANIFEST, encoding="utf-8")

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content="sorry, I can't help with that")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="do something",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "did not return a fenced YAML block" in result.output_text

    @pytest.mark.asyncio
    async def test_validation_failure_returns_error(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))
        proposal_path = tmp_path / "models.proposed.yaml"
        proposal_path.write_text(_VALID_MANIFEST, encoding="utf-8")

        invalid_manifest = "version: 1\nmodels: []\n"

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content=f"```yaml\n{invalid_manifest}```")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="delete everything",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "failed validation" in result.output_text


class TestAuthoredTextLoader:
    """Verify the operation loads prompts via ctx.read_authored_text_fn."""

    @pytest.mark.asyncio
    async def test_usage_loaded_from_authored_text(self) -> None:
        ctx = OperationExecutionContext(
            query="",
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "advise-models" in result.output_text

    @pytest.mark.asyncio
    async def test_no_manifest_loaded_from_authored_text(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))
        ctx = OperationExecutionContext(
            query="do something",
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "No model manifest" in result.output_text

    @pytest.mark.asyncio
    async def test_system_prompt_loaded_from_authored_text(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))
        proposal_path = tmp_path / "models.proposed.yaml"
        proposal_path.write_text(_VALID_MANIFEST, encoding="utf-8")

        captured_messages: list[dict[str, Any]] = []

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            captured_messages.extend(kwargs.get("messages", []))
            return _FakeCompletion(content=f"```yaml\n{_EDITED_MANIFEST}```")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="mark o3 as premium",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
        )
        await execute(ctx)
        system_messages = [
            msg for msg in captured_messages if msg.get("role") == "system"
        ]
        assert system_messages
        # The authored system prompt mentions "model-manifest editor".
        assert "model-manifest editor" in system_messages[0].get("content", "")

    @pytest.mark.asyncio
    async def test_validation_error_loaded_from_authored_text(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))
        proposal_path = tmp_path / "models.proposed.yaml"
        proposal_path.write_text(_VALID_MANIFEST, encoding="utf-8")

        invalid_manifest = "version: 1\nmodels: []\n"

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content=f"```yaml\n{invalid_manifest}```")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="delete everything",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "failed validation" in result.output_text


class TestNoLoaderFailure:
    """Per the pack-content rule: missing loader must fail loudly, not fall back."""

    @pytest.mark.asyncio
    async def test_missing_loader_raises_runtime_error(self) -> None:
        ctx = OperationExecutionContext(query="")
        with pytest.raises(RuntimeError, match="read_authored_text_fn is not wired"):
            await execute(ctx)

    @pytest.mark.asyncio
    async def test_empty_authored_text_raises_runtime_error(self) -> None:
        def empty_loader(path: str) -> str:
            return ""

        ctx = OperationExecutionContext(
            query="",
            read_authored_text_fn=empty_loader,
        )
        with pytest.raises(RuntimeError, match="empty or missing"):
            await execute(ctx)

    @pytest.mark.asyncio
    async def test_loader_raising_raises_runtime_error(self) -> None:
        def raising_loader(path: str) -> str:
            raise OSError("disk error")

        ctx = OperationExecutionContext(
            query="",
            read_authored_text_fn=raising_loader,
        )
        with pytest.raises(RuntimeError, match="read_authored_text_fn failed"):
            await execute(ctx)