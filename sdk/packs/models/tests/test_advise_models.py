"""Tests for the advise_models operation.

Tests inject a real content-registry loader (`read_authored_text_fn`) that
reads the actual authored prompt files from the pack's content root, and a
fake `execute_tool_fn` that simulates the lisa-storage
`read_model_manifest`/`write_model_manifest` tools. Per the pack-content rule,
the operation has no fallbacks — tests must wire both callbacks exactly as
production hosts do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kernel.models import OperationExecutionContext, ToolResult
from sdk.packs.models.operations.advise_models.advise_models import (
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
    tool_calls: list[Any] | None = None
    usage: Any = None


def _make_storage_executor(
    *,
    proposal: str | None = None,
    active: str | None = None,
) -> tuple[Any, dict[str, str]]:
    """Return a fake ``execute_tool_fn`` simulating lisa-storage manifest tools.

    Returns ``(executor, written)`` where ``written`` captures every
    ``write_model_manifest`` call keyed by ``which`` so tests can assert what
    the operation persisted.
    """

    store: dict[str, str | None] = {"proposal": proposal, "active": active}
    written: dict[str, str] = {}

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = ctx.parameters or {}
        which = str(params.get("which") or "proposal")
        if tool_id == "read_model_manifest":
            content = store.get(which)
            if content is None:
                return ToolResult(
                    tool_id=tool_id,
                    output_text=json.dumps({"error": f"manifest not found: {which}"}),
                )
            return ToolResult(tool_id=tool_id, output_text=content)
        if tool_id == "write_model_manifest":
            content = str(params.get("content") or "")
            store[which] = content
            written[which] = content
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps({"ok": True, "which": which, "bytes": len(content)}),
            )
        return ToolResult(tool_id=tool_id, error=f"unknown tool: {tool_id}")

    return execute_tool_fn, written


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
        summary = _render_diff_summary(_VALID_MANIFEST, _EDITED_MANIFEST)
        assert "Manifest updated" in summary
        assert "gpt-chat-latest" in summary
        assert "cost_tier" in summary

    def test_no_changes_detected(self) -> None:
        summary = _render_diff_summary(_VALID_MANIFEST, _VALID_MANIFEST)
        assert "no changes detected" in summary


class TestExecuteOperation:
    @pytest.mark.asyncio
    async def test_empty_instruction_returns_usage(self) -> None:
        ctx = OperationExecutionContext(
            query="",
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "Usage:" in result.output_text
        assert result.output_json.get("validated") is False

    @pytest.mark.asyncio
    async def test_no_manifest_returns_error(self) -> None:
        storage_fn, _ = _make_storage_executor()  # no proposal, no active
        ctx = OperationExecutionContext(
            query="make gpt-5 the default",
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=storage_fn,
        )
        result = await execute(ctx)
        assert "No model manifest" in result.output_text
        assert result.output_json.get("validated") is False

    @pytest.mark.asyncio
    async def test_writes_edited_manifest_via_storage_tool(self) -> None:
        storage_fn, written = _make_storage_executor(proposal=_VALID_MANIFEST)

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content=f"```yaml\n{_EDITED_MANIFEST}```")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="mark o3 as premium tier",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=storage_fn,
        )
        result = await execute(ctx)
        assert "Manifest updated" in result.output_text
        assert result.output_json.get("validated") is True
        # The operation must have written the edited manifest to the proposal.
        assert "proposal" in written
        assert "premium" in written["proposal"]

    @pytest.mark.asyncio
    async def test_reads_active_when_no_proposal(self) -> None:
        storage_fn, written = _make_storage_executor(active=_VALID_MANIFEST)

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content=f"```yaml\n{_EDITED_MANIFEST}```")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="mark o3 as premium tier",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=storage_fn,
        )
        result = await execute(ctx)
        assert "Manifest updated" in result.output_text
        # When no proposal exists, the operation still writes the edit to the proposal.
        assert "proposal" in written

    @pytest.mark.asyncio
    async def test_invalid_llm_output_returns_error(self) -> None:
        storage_fn, _ = _make_storage_executor(proposal=_VALID_MANIFEST)

        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content="sorry, I can't help with that")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        ctx = OperationExecutionContext(
            query="do something",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=storage_fn,
        )
        result = await execute(ctx)
        assert "did not return a fenced YAML block" in result.output_text
        assert result.output_json.get("validated") is False

    @pytest.mark.asyncio
    async def test_validation_failure_returns_error(self) -> None:
        storage_fn, written = _make_storage_executor(proposal=_VALID_MANIFEST)
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
            execute_tool_fn=storage_fn,
        )
        result = await execute(ctx)
        assert "failed validation" in result.output_text
        assert result.output_json.get("validated") is False
        # Validation failure must NOT write the proposal — the caller relies
        # on this to know there is nothing to apply.
        assert "proposal" not in written


class TestAuthoredTextLoader:
    """Verify the operation loads prompts via ctx.read_authored_text_fn."""

    @pytest.mark.asyncio
    async def test_usage_loaded_from_authored_text(self) -> None:
        ctx = OperationExecutionContext(
            query="",
            read_authored_text_fn=_make_loader(),
        )
        result = await execute(ctx)
        assert "/models update" in result.output_text

    @pytest.mark.asyncio
    async def test_system_prompt_loaded_from_authored_text(self) -> None:
        storage_fn, _ = _make_storage_executor(proposal=_VALID_MANIFEST)
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
            execute_tool_fn=storage_fn,
        )
        await execute(ctx)
        system_messages = [msg for msg in captured_messages if msg.get("role") == "system"]
        assert system_messages
        # The authored system prompt mentions "model-manifest editor".
        assert "model-manifest editor" in system_messages[0].get("content", "")


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


class TestNoStorageToolFailure:
    """Missing execute_tool_fn must fail loudly, not silently no-op."""

    @pytest.mark.asyncio
    async def test_missing_execute_tool_fn_raises(self) -> None:
        ctx = OperationExecutionContext(
            query="do something",
            read_authored_text_fn=_make_loader(),
        )
        with pytest.raises(RuntimeError, match="execute_tool_fn"):
            await execute(ctx)

    @pytest.mark.asyncio
    async def test_storage_read_error_raises(self) -> None:
        async def failing_storage(tool_id: str, ctx: Any) -> ToolResult:
            return ToolResult(tool_id=tool_id, error="storage offline")

        op_ctx = OperationExecutionContext(
            query="do something",
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=failing_storage,
        )
        with pytest.raises(RuntimeError, match="read_model_manifest failed"):
            await execute(op_ctx)

    @pytest.mark.asyncio
    async def test_storage_write_error_raises(self) -> None:
        async def fake_complete(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion(content=f"```yaml\n{_EDITED_MANIFEST}```")

        def fake_resolve(model_id: str) -> _FakeModel:
            return _FakeModel()

        async def failing_write_storage(tool_id: str, ctx: Any) -> ToolResult:
            if tool_id == "read_model_manifest":
                return ToolResult(tool_id=tool_id, output_text=_VALID_MANIFEST)
            if tool_id == "write_model_manifest":
                return ToolResult(tool_id=tool_id, error="disk full")
            return ToolResult(tool_id=tool_id, output_text="{}")

        op_ctx = OperationExecutionContext(
            query="mark o3 as premium",
            complete_fn=fake_complete,
            resolve_model_fn=fake_resolve,
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=failing_write_storage,
        )
        with pytest.raises(RuntimeError, match="write_model_manifest failed"):
            await execute(op_ctx)
