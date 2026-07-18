"""Tests for the markdown_forge_document operation.

Tests inject a real content-registry loader (`read_authored_text_fn`) that
reads the actual authored prompt files from the pack's content root, a fake
`complete_fn` that returns a canned markdown document, and a fake
`execute_tool_fn` that simulates the `memory.search`, `markdown_generate`,
and `write_file` tools. Per the pack-content rule, the operation has no
fallbacks — tests must wire both callbacks exactly as production hosts do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kernel.models import OperationExecutionContext, ToolResult
from sdk.packs.markdown.operations.forge_document.forge_document import execute

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
    # Default to a clean "stop" so existing tests do not trip the new
    # truncation/continuation path. Tests that exercise truncation pass
    # ``finish_reason="length"`` or ``"unknown"`` explicitly.
    finish_reason: str | None = "stop"


_VALID_DOCUMENT = (
    "# Runbook: Restart a Stuck VM\n"
    "\n"
    "## Symptoms\n"
    "- VM is unresponsive to SSH.\n"
    "- Agent reports `NotReady`.\n"
    "\n"
    "## Steps\n"
    "1. Confirm the VM state via `az vm show`.\n"
    "2. Restart with `az vm restart -g rg -n vm`.\n"
    "3. Verify health probes return 200.\n"
)


def _make_tool_executor(
    *,
    memory_records: list[str] | None = None,
    outline: dict[str, Any] | None = None,
    write_ok: bool = True,
    stored_documents: dict[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a fake ``execute_tool_fn`` and a captures dict.

    ``captures`` records the arguments of each ``write_file`` call so tests
    can assert what the operation persisted. ``stored_documents`` seeds the
    fake storage with existing document text keyed by path; ``read_file``
    reads from it and ``write_file`` updates it.
    """
    store: dict[str, str] = dict(stored_documents or {})
    captures: dict[str, Any] = {
        "writes": [],
        "memory_query": None,
        "outline_markdown": None,
        "read_paths": [],
    }

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        if tool_id == "memory.search":
            captures["memory_query"] = params.get("query")
            if memory_records is None:
                return ToolResult(tool_id=tool_id, output_text="No relevant memories found.")
            lines = [f"- {r}" for r in memory_records]
            return ToolResult(tool_id=tool_id, output_text="\n".join(lines))
        if tool_id == "markdown_generate":
            captures["outline_markdown"] = params.get("markdown")
            if outline is not None:
                return ToolResult(
                    tool_id=tool_id,
                    output_text="Outline from injected fixture.",
                    output_json=outline,
                )
            # Compute a minimal outline from the supplied markdown so the
            # operation's heading-count gate can be exercised realistically.
            md_text = str(params.get("markdown") or "")
            heading_count = sum(
                1 for line in md_text.splitlines() if line.lstrip().startswith("#")
            )
            return ToolResult(
                tool_id=tool_id,
                output_text=f"Outline: {heading_count} headings.",
                output_json={"heading_count": heading_count, "headings": []},
            )
        if tool_id == "read_file":
            path = str(params.get("path") or "")
            captures["read_paths"].append(path)
            if path in store:
                return ToolResult(tool_id=tool_id, output_text=store[path])
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps({"error": f"file not found: {path}"}),
            )
        if tool_id == "write_file":
            path = str(params.get("path") or "")
            content = str(params.get("content") or "")
            captures["writes"].append({"path": path, "content": content})
            if not write_ok:
                return ToolResult(
                    tool_id=tool_id,
                    output_text=json.dumps({"error": "server is read-only"}),
                )
            store[path] = content
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}
                ),
            )
        return ToolResult(tool_id=tool_id, error=f"unknown tool: {tool_id}")

    return execute_tool_fn, captures


def _make_ctx(
    *,
    query: str = "forge a runbook for restarting a stuck VM",
    params: dict[str, Any] | None = None,
    complete_fn: Any = None,
    resolve_model_fn: Any = None,
    execute_tool_fn: Any = None,
) -> OperationExecutionContext:
    return OperationExecutionContext(
        workspace_id="test",
        user_id="test-user",
        user_slug="test-user",
        surface_type="personal_dm",
        surface_id="test",
        memory_scope="personal",
        mode_id="markdown",
        query=query,
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=_make_loader(),
        execute_tool_fn=execute_tool_fn,
        params=params or {},
    )


def _fake_complete_returning(document: str) -> Any:
    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{document}\n```")

    return _complete


def _fake_resolve_model() -> Any:
    def _resolve(_alias: str) -> _FakeModel:
        return _FakeModel()

    return _resolve


# ── Operation tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_input_returns_no_input_message() -> None:
    ctx = _make_ctx(query="", execute_tool_fn=(lambda *_a, **_kw: None))
    result = await execute(ctx)
    assert "No instruction" in result.output_text


@pytest.mark.asyncio
async def test_forges_and_persists_document() -> None:
    tool_fn, captures = _make_tool_executor(memory_records=["Incident 42: VM stuck, restart fixed it."])
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "docs/runbook.md"},
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert "docs/runbook.md" in result.output_text
    # The operation must have recalled memory with the user query.
    assert captures["memory_query"] == "forge a runbook for restarting a stuck VM"
    # The operation must have validated the document via the outline tool.
    assert captures["outline_markdown"] is not None
    assert "Restart a Stuck VM" in str(captures["outline_markdown"])
    # The operation must have written the document to storage.
    assert len(captures["writes"]) == 1
    write = captures["writes"][0]
    assert write["path"] == "docs/runbook.md"
    assert "Restart a Stuck VM" in write["content"]


@pytest.mark.asyncio
async def test_uses_default_path_when_none_supplied() -> None:
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "docs/forge.md" in result.output_text
    assert captures["writes"][0]["path"] == "docs/forge.md"


@pytest.mark.asyncio
async def test_rejects_path_with_parent_traversal() -> None:
    tool_fn, _ = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "../escape.md"},
    )
    result = await execute(ctx)
    assert ".." in result.output_text or "document_path" in result.output_text


@pytest.mark.asyncio
async def test_empty_llm_response_returns_error() -> None:
    tool_fn, _ = _make_tool_executor()

    async def _empty_complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content="")

    ctx = _make_ctx(
        complete_fn=_empty_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "empty response" in result.output_text.lower()


@pytest.mark.asyncio
async def test_document_with_no_headings_returns_no_outline_error() -> None:
    tool_fn, _ = _make_tool_executor()
    no_headings_doc = "Just a paragraph. No headings here.\n"
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(no_headings_doc),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "no headings" in result.output_text.lower()


@pytest.mark.asyncio
async def test_storage_failure_returns_storage_failed_message() -> None:
    tool_fn, _ = _make_tool_executor(write_ok=False)
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Storage Failed" in result.output_text
    assert "read-only" in result.output_text


@pytest.mark.asyncio
async def test_outlines_injected_fixture_with_zero_headings_fails() -> None:
    tool_fn, _ = _make_tool_executor(outline={"heading_count": 0, "headings": []})
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "no headings" in result.output_text.lower()


@pytest.mark.asyncio
async def test_memory_context_passed_to_llm() -> None:
    """The recalled memory context must reach the LLM user prompt."""
    tool_fn, _ = _make_tool_executor(memory_records=["Prior incident: VM restart fixed it."])
    captured_messages: list[dict[str, Any]] = []

    async def _capturing_complete(**kwargs: Any) -> _FakeCompletion:
        captured_messages.extend(kwargs.get("messages", []))
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_capturing_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    user_prompt = next(m for m in captured_messages if m["role"] == "user")
    assert "Prior incident: VM restart fixed it." in user_prompt["content"]


@pytest.mark.asyncio
async def test_reads_existing_document_from_storage_when_omitted() -> None:
    """When existing_document is not supplied, the operation reads it from storage."""
    existing = "# Existing Doc\n\nOld content.\n"
    tool_fn, captures = _make_tool_executor(stored_documents={"docs/forge.md": existing})

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    # The operation must have read the existing document from storage.
    assert "docs/forge.md" in captures["read_paths"]


@pytest.mark.asyncio
async def test_existing_document_passed_to_llm() -> None:
    """When existing_document is supplied, it must reach the LLM user prompt."""
    existing = "# Existing Doc\n\nOld content.\n"
    tool_fn, _ = _make_tool_executor()
    captured_messages: list[dict[str, Any]] = []

    async def _capturing_complete(**kwargs: Any) -> _FakeCompletion:
        captured_messages.extend(kwargs.get("messages", []))
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_capturing_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"existing_document": existing},
    )
    await execute(ctx)
    user_prompt = next(m for m in captured_messages if m["role"] == "user")
    assert "Existing Doc" in user_prompt["content"]
    assert "Old content." in user_prompt["content"]


@pytest.mark.asyncio
async def test_iteration_revises_and_persists() -> None:
    """A second call with existing_document revises the stored document."""
    original = "# Runbook\n\n## Steps\n1. Old step.\n"
    revised = "# Runbook\n\n## Steps\n1. New step.\n2. Added step.\n"
    tool_fn, captures = _make_tool_executor(stored_documents={"docs/runbook.md": original})

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{revised}\n```")

    ctx = _make_ctx(
        query="add a step to verify the VM is healthy after restart",
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "docs/runbook.md"},
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert "docs/runbook.md" in result.output_text
    # The operation must have read the existing document before revising.
    assert "docs/runbook.md" in captures["read_paths"]
    # The persisted content must be the revised document.
    assert len(captures["writes"]) == 1
    assert "New step." in captures["writes"][0]["content"]
    assert "Added step." in captures["writes"][0]["content"]


@pytest.mark.asyncio
async def test_no_existing_document_starts_fresh() -> None:
    """When storage has no existing document, the operation starts fresh."""
    tool_fn, captures = _make_tool_executor()  # no stored_documents

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # The operation attempted a read but found nothing (not-found is normal).
    assert "docs/forge.md" in captures["read_paths"]


@pytest.mark.asyncio
async def test_document_annotated_with_model_by_default() -> None:
    """The forged document includes a non-rendering model annotation by default."""
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "<!--" in persisted
    assert "generated by lisa" in persisted
    # The fake model id and provider must appear in the annotation.
    assert "gpt-chat-latest" in persisted
    assert "openai" in persisted


@pytest.mark.asyncio
async def test_annotation_suppressed_by_param() -> None:
    """``annotate_model: False`` suppresses the model annotation."""
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"annotate_model": False},
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "generated by lisa" not in persisted


@pytest.mark.asyncio
async def test_annotation_suppressed_by_user_phrase() -> None:
    """A natural-language opt-out phrase suppresses the annotation."""
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        query="forge a runbook, but don't annotate the model",
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "generated by lisa" not in persisted


@pytest.mark.asyncio
async def test_document_formatted_by_mdformat() -> None:
    """The persisted document is normalized (single trailing newline, clean spacing)."""
    messy = (
        "# Runbook\n"
        "\n\n\n"
        "## Steps\n"
        "\n"
        "-  step one\n"
        "-   step two\n"
        "\n\n\n"
    )
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(messy),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    # mdformat collapses 3+ blank lines to one and ensures a single trailing newline.
    body = persisted.split("<!--")[0]
    assert "\n\n\n" not in body
    assert persisted.endswith("\n")


# ── Truncation / continuation tests ────────────────────────────────────
#
# These exercise the finish_reason-aware continuation contract added
# 2026-07-18 (see /memories/repo/forge-document-truncation-contract.md).
# The first pass returns a partial document with finish_reason="length"
# (or "unknown"/None), and a second fake complete_fn returns the tail.
# The operation must merge the two and persist a complete document with
# NO truncated annotation.


def _fake_complete_sequence(documents: list[str], finish_reasons: list[str | None]) -> Any:
    """Return a ``complete_fn`` that returns each document/finish_reason in order.

    The first call returns (documents[0], finish_reasons[0]); the second call
    returns (documents[1], finish_reasons[1]); and so on. Used to simulate a
    truncated first pass followed by a continuation pass.
    """
    state = {"index": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        idx = state["index"]
        state["index"] += 1
        doc = documents[idx] if idx < len(documents) else ""
        finish = finish_reasons[idx] if idx < len(finish_reasons) else "stop"
        return _FakeCompletion(
            content=f"```markdown\n{doc}\n```",
            finish_reason=finish,
        )

    return _complete


_PARTIAL_SWOT = (
    "# SWOT Analysis\n"
    "\n"
    "## Strengths\n"
    "- Deterministic state control.\n"
    "\n"
    "## Weaknesses\n"
    "- Steeper learning curve.\n"
)

_SWOT_TAIL = (
    "## Opportunities\n"
    "- Enterprise adoption.\n"
    "\n"
    "## Threats\n"
    "- Integrated IDE dominance.\n"
)

_FULL_SWOT = _PARTIAL_SWOT + "\n" + _SWOT_TAIL


@pytest.mark.asyncio
async def test_truncated_first_pass_continues_and_completes() -> None:
    """finish_reason='length' triggers a continuation pass that completes the doc."""
    tool_fn, captures = _make_tool_executor()
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _SWOT_TAIL],
        finish_reasons=["length", "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # The summary must NOT carry a truncation warning.
    assert "truncated" not in result.output_text.lower()
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    # Both the partial head and the continuation tail must be present.
    assert "Strengths" in persisted
    assert "Weaknesses" in persisted
    assert "Opportunities" in persisted
    assert "Threats" in persisted
    # The annotation must NOT mark the document as truncated.
    assert "truncated by model" not in persisted


@pytest.mark.asyncio
async def test_unknown_finish_reason_triggers_continuation() -> None:
    """finish_reason='unknown' (Ollama-style) is treated as truncation."""
    tool_fn, captures = _make_tool_executor()
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _SWOT_TAIL],
        finish_reasons=["unknown", "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    assert "Opportunities" in persisted
    assert "truncated by model" not in persisted


@pytest.mark.asyncio
async def test_none_finish_reason_triggers_continuation() -> None:
    """A missing finish_reason (None) is treated as truncation on a long doc."""
    tool_fn, captures = _make_tool_executor()
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _SWOT_TAIL],
        finish_reasons=[None, "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 1


@pytest.mark.asyncio
async def test_continuation_also_truncates_persists_with_truncated_annotation() -> None:
    """When the continuation also truncates, the artifact is marked truncated.

    Both passes return finish_reason='length'; the budget is exhausted; the
    persisted document must carry a 'truncated by model' annotation and the
    OperationResult must surface the truncation in output_json + the summary.
    """
    tool_fn, captures = _make_tool_executor()
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCompletion(
                content=f"```markdown\n{_PARTIAL_SWOT}\n```",
                finish_reason="length",
            )
        # Continuation passes: always return a tail that ends mid-sentence
        # AND reports 'length', so the merge never looks complete and the
        # continuation budget is exhausted.
        return _FakeCompletion(
            content="```markdown\n## Opportunities\n- Enterprise adoption and the\n```",
            finish_reason="length",
        )

    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is True
    assert result.output_json["continue_attempts"] >= 1
    assert "truncated" in result.output_text.lower()
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" in persisted
    assert "finish_reason: length" in persisted
    # The merged body is still persisted (head + tail), just marked.
    assert "Strengths" in persisted
    assert "Opportunities" in persisted


@pytest.mark.asyncio
async def test_clean_stop_does_not_trigger_continuation() -> None:
    """finish_reason='stop' returns the document as-is with no continuation."""
    tool_fn, captures = _make_tool_executor()
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        return _FakeCompletion(
            content=f"```markdown\n{_VALID_DOCUMENT}\n```",
            finish_reason="stop",
        )

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 0
    # Only one completion call — no continuation pass.
    assert call_count["n"] == 1
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" not in persisted


@pytest.mark.asyncio
async def test_continuation_empty_gives_up_and_marks_truncated() -> None:
    """A continuation that returns empty content stops and marks the doc truncated."""
    tool_fn, captures = _make_tool_executor()
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCompletion(
                content=f"```markdown\n{_PARTIAL_SWOT}\n```",
                finish_reason="length",
            )
        # Continuation pass: truly empty content (not fenced).
        return _FakeCompletion(content="", finish_reason="stop")

    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is True
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" in persisted
    # The partial head is still persisted.
    assert "Strengths" in persisted
    # The tail that never arrived is absent.
    assert "Opportunities" not in persisted


@pytest.mark.asyncio
async def test_continuation_duplicate_tail_gives_up_and_marks_truncated() -> None:
    """A continuation that just re-emits the existing tail is rejected as a duplicate."""
    tool_fn, captures = _make_tool_executor()
    # The continuation re-emits the partial head verbatim — a known local-model
    # failure mode where "continue" produces a loop instead of new content.
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _PARTIAL_SWOT],
        finish_reasons=["length", "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is True
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" in persisted
    # The duplicate tail must not have been merged — Opportunities still absent.
    assert "Opportunities" not in persisted