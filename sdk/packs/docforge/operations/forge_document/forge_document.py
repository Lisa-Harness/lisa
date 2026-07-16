"""Forge-document operation — memory-grounded markdown synthesis.

Pipeline:

1. Recall context from long-term memory via the ``memory.search`` MCP alias
   (routed by the host to the configured memory backend).
2. Call an LLM with an authored system prompt + the user input + recalled
   context to synthesize a markdown document.
3. Validate the synthesized document's structure by calling the
   ``docforge_outline_markdown`` tool (backed by ``markdown-it-py``). The
   outline must contain at least one heading.
4. Persist the document to the ``lisa-storage`` MCP server's ``write_file``
   tool under a caller-supplied relative path (default ``docs/``).

The operation never touches the filesystem directly — all reads (memory)
and writes (storage) go through ``ctx.execute_tool_fn``. Authored prompts
are loaded via ``ctx.read_authored_text_fn``; per the pack-content rule,
a missing/empty loader raises rather than silently falling back.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

# Tool ids — declared in the operation YAML ``tool_ids`` and routed through
# ``ctx.execute_tool_fn``. ``docforge_outline_markdown`` is a pack-contributed
# in-process tool; ``write_file`` and ``memory.search`` are MCP tools served
# by ``lisa-storage`` and the memory backend respectively.
_OUTLINE_TOOL_ID = "docforge_outline_markdown"
_WRITE_TOOL_ID = "write_file"
_READ_TOOL_ID = "read_file"
_MEMORY_SEARCH_TOOL_ID = "memory.search"

# Authored-text paths — loaded via ctx.read_authored_text_fn (SDK loader).
_PROMPT_DIR = "operations/forge_document/prompts"
_SYSTEM_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_system.md"
_USER_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_user.md"
_USAGE_PATH = f"{_PROMPT_DIR}/forge_document_usage.md"
_NO_INPUT_PATH = f"{_PROMPT_DIR}/forge_document_no_input.md"
_EMPTY_RESPONSE_PATH = f"{_PROMPT_DIR}/forge_document_empty_response.md"
_EMPTY_DOCUMENT_PATH = f"{_PROMPT_DIR}/forge_document_empty_document.md"
_NO_OUTLINE_PATH = f"{_PROMPT_DIR}/forge_document_no_outline.md"
_STORAGE_FAILED_PATH = f"{_PROMPT_DIR}/forge_document_storage_failed.md"

_MAX_COMPLETION_TOKENS = 2000
_MAX_RECALL_RECORDS = 8
_MAX_DOCUMENT_CHARS = 20_000
_MAX_QUERY_CHARS = 2_000

# Minimum structural bar for a forged document: at least one heading.
_MIN_HEADINGS = 1


def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    """Load authored markdown via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """
    if not callable(ctx.read_authored_text_fn):
        raise RuntimeError(
            f"read_authored_text_fn is not wired; cannot load authored content at {path}"
        )
    try:
        loaded = str(ctx.read_authored_text_fn(path) or "").strip()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"read_authored_text_fn failed for {path}: {exc}") from exc
    if not loaded:
        raise RuntimeError(f"Authored text at {path} is empty or missing")
    return loaded


def _format_template(template: str, **kwargs: Any) -> str:
    """Substitute ``{placeholder}`` tokens in a loaded template."""
    try:
        return template.format_map(_SafeDict(kwargs))
    except Exception:  # pylint: disable=broad-except
        return template


class _SafeDict(dict[str, str]):
    """dict subclass that returns ``{key}`` for missing keys in str.format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class Parameters(BaseModel):
    """Parameter surface for callers invoking the operation as a tool.

    The operation reads ``ctx.params`` directly (it is not registered as a
    model-facing tool by default), but exposing the shape documents the
    accepted inputs.
    """

    document_path: str = Field(
        default="docs/forge.md",
        description="Relative path within the lisa-storage files sandbox where the forged document is persisted.",
    )
    query: str = Field(
        default="",
        description="Natural-language instruction for the document. Falls back to ctx.query when empty.",
    )
    existing_document: str = Field(
        default="",
        description="Current document text to revise. When non-empty, the operation "
        "reads it from storage via read_file and passes it to the LLM as the "
        "starting point. Omit for a new document.",
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of recalled memory records to inject as context.",
    )


def _extract_document_path(params: dict[str, Any]) -> str:
    """Return the target storage path, sanitized to a safe relative form."""
    raw = str(params.get("document_path") or "").strip()
    if not raw:
        return "docs/forge.md"
    # Reject absolute paths and parent traversal before the storage server
    # sees them; the storage server re-validates, but we fail early with a
    # clearer error.
    normalized = raw.replace("\\", "/").lstrip("/")
    if ".." in normalized.split("/"):
        raise RuntimeError("document_path may not contain '..' segments")
    return normalized


def _extract_query(ctx: OperationExecutionContext, params: dict[str, Any]) -> str:
    """Return the user-supplied instruction, bounded in length."""
    raw = str(params.get("query") or ctx.query or "").strip()
    if len(raw) > _MAX_QUERY_CHARS:
        raw = raw[:_MAX_QUERY_CHARS] + "…"
    return raw


async def _read_existing_document(
    ctx: OperationExecutionContext,
    path: str,
) -> str:
    """Read the current document from lisa-storage via the ``read_file`` tool.

    Returns the document text, or an empty string when the document does not
    exist yet (a normal condition for a first forge). Raises ``RuntimeError``
    on any other tool failure.
    """
    tool_result = await ctx.execute_tool_fn(
        _READ_TOOL_ID,
        ToolExecutionContext(
            tool_id=_READ_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"path": path},
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{_READ_TOOL_ID} failed: {error_text}")
    output_text = getattr(tool_result, "output_text", "") or ""
    # The storage server returns the raw file text on success, or a JSON
    # error object on failure. A JSON error with "not found" is a normal
    # first-forge condition — return "".
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return output_text.strip()
    if isinstance(data, dict) and "error" in data:
        if "not found" in str(data["error"]):
            return ""
        raise RuntimeError(f"{_READ_TOOL_ID} failed: {data['error']}")
    return output_text.strip()


async def _recall_memory(
    ctx: OperationExecutionContext,
    query: str,
    limit: int = _MAX_RECALL_RECORDS,
) -> str:
    """Recall relevant memories via the ``memory.search`` MCP alias.

    Returns a newline-formatted context block (possibly empty). Raises
    ``RuntimeError`` only when the tool executor itself is missing.
    """
    if not callable(ctx.execute_tool_fn):
        raise RuntimeError("docforge requires execute_tool_fn to call memory/storage tools")

    recall_limit = max(1, min(int(limit), 20))
    tool_result = await ctx.execute_tool_fn(
        _MEMORY_SEARCH_TOOL_ID,
        ToolExecutionContext(
            tool_id=_MEMORY_SEARCH_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            query=query,
            parameters={
                "query": query,
                "user_slug": ctx.user_slug,
                "workspace_id": ctx.workspace_id,
                "memory_scope": ctx.memory_scope,
                "surface_id": ctx.surface_id,
                "limit": recall_limit,
            },
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        logger.warning("docforge: memory.search returned error: %s", error_text)
        return ""
    output_text = getattr(tool_result, "output_text", "") or ""
    # The memory backend returns newline-formatted text (or a JSON list).
    # If it's JSON, flatten to bullets; otherwise pass through.
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return output_text.strip()
    if isinstance(data, list):
        lines: list[str] = []
        for item in data:
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                if content:
                    lines.append(f"- {content}")
            elif isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")
        return "\n".join(lines)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        lines = []
        for item in data["results"]:
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                if content:
                    lines.append(f"- {content}")
        return "\n".join(lines)
    return output_text.strip()


async def _call_llm(
    ctx: OperationExecutionContext,
    user_input: str,
    memory_context: str,
    existing_document: str = "",
) -> str:
    """Call the host's completion function with the forging prompt."""
    if ctx.complete_fn is None or ctx.resolve_model_fn is None:
        raise RuntimeError("docforge requires complete_fn and resolve_model_fn")

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.resolve_model_fn(preferred_model_id)
    if model_def is None:
        raise RuntimeError(
            f"docforge could not resolve model '{preferred_model_id}'"
        )

    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_PATH)
    user_prompt = _format_template(
        user_template,
        user_input=user_input,
        memory_context=memory_context or "(no recalled context)",
        existing_document=existing_document or "(no existing document — start fresh)",
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = await ctx.complete_fn(
        messages=messages,
        model_def=model_def,
        max_tokens=_MAX_COMPLETION_TOKENS,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    return content.strip()


def _extract_markdown_block(text: str) -> str:
    """Extract the first fenced ```markdown ...``` block, else return text.

    Accepts fenced blocks tagged ``markdown`` or untagged. Falls back to the
    raw text when no fence is present so the outline tool can still assess it.
    """
    match = re.search(r"```(?:markdown|md)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


async def _validate_outline(ctx: OperationExecutionContext, document: str) -> dict[str, Any]:
    """Run the ``docforge_outline_markdown`` tool and return its outline dict.

    Raises ``RuntimeError`` if the tool returns an error.
    """
    tool_result = await ctx.execute_tool_fn(
        _OUTLINE_TOOL_ID,
        ToolExecutionContext(
            tool_id=_OUTLINE_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"markdown": document},
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{_OUTLINE_TOOL_ID} failed: {error_text}")
    output_json = getattr(tool_result, "output_json", {}) or {}
    if not output_json:
        # Try to parse output_text as JSON for backends that only return text.
        output_text = getattr(tool_result, "output_text", "") or ""
        try:
            output_json = json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            output_json = {}
    return output_json if isinstance(output_json, dict) else {}


async def _persist_document(
    ctx: OperationExecutionContext,
    document: str,
    path: str,
) -> dict[str, Any]:
    """Write the document to lisa-storage via the ``write_file`` tool.

    Returns the parsed JSON response (with ``ok``/``path``/``bytes``).
    Raises ``RuntimeError`` on any tool failure.
    """
    tool_result = await ctx.execute_tool_fn(
        _WRITE_TOOL_ID,
        ToolExecutionContext(
            tool_id=_WRITE_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"path": path, "content": document},
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{_WRITE_TOOL_ID} failed: {error_text}")
    output_text = getattr(tool_result, "output_text", "") or str(tool_result)
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        data = {"ok": False, "raw": output_text}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{_WRITE_TOOL_ID} failed: {data['error']}")
    return data if isinstance(data, dict) else {}


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Forge a markdown document from user input + recalled memory and persist it."""
    params = ctx.params or {}
    user_input = _extract_query(ctx, params)
    if not user_input:
        return OperationResult(output_text=_load_text(ctx, _NO_INPUT_PATH))

    try:
        document_path = _extract_document_path(params)
    except RuntimeError as exc:
        return OperationResult(output_text=str(exc))

    if not callable(ctx.execute_tool_fn):
        return OperationResult(output_text=_load_text(ctx, _USAGE_PATH))

    recall_limit = int(params.get("limit") or _MAX_RECALL_RECORDS)
    memory_context = await _recall_memory(ctx, user_input, limit=recall_limit)

    # Iteration: when the caller supplies an existing document, use it. When
    # the caller omits it but the target path already exists in storage,
    # read it so the LLM revises rather than replacing wholesale.
    existing_document = str(params.get("existing_document") or "").strip()
    if not existing_document:
        try:
            existing_document = await _read_existing_document(ctx, document_path)
        except RuntimeError as exc:
            return OperationResult(output_text=str(exc))

    completion = await _call_llm(
        ctx,
        user_input,
        memory_context,
        existing_document=existing_document,
    )
    if not completion or not completion.strip():
        return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))

    document = _extract_markdown_block(completion)
    if not document.strip():
        text = _format_template(
            _load_text(ctx, _EMPTY_DOCUMENT_PATH),
            response_preview=completion[:400],
        )
        return OperationResult(output_text=text)

    if len(document) > _MAX_DOCUMENT_CHARS:
        document = document[:_MAX_DOCUMENT_CHARS] + "\n\n<!-- truncated -->\n"

    try:
        outline = await _validate_outline(ctx, document)
    except RuntimeError as exc:
        return OperationResult(output_text=str(exc))

    heading_count = int(outline.get("heading_count", 0) or 0)
    if heading_count < _MIN_HEADINGS:
        text = _format_template(
            _load_text(ctx, _NO_OUTLINE_PATH),
            heading_count=heading_count,
        )
        return OperationResult(output_text=text)

    try:
        write_response = await _persist_document(ctx, document, document_path)
    except RuntimeError as exc:
        text = _format_template(
            _load_text(ctx, _STORAGE_FAILED_PATH),
            error=str(exc),
        )
        return OperationResult(output_text=text)

    ok = bool(write_response.get("ok", False))
    if not ok:
        text = _format_template(
            _load_text(ctx, _STORAGE_FAILED_PATH),
            error=write_response.get("raw", "unknown storage error"),
        )
        return OperationResult(output_text=text)

    stored_path = str(write_response.get("path") or document_path)
    byte_count = int(write_response.get("bytes") or len(document.encode("utf-8")))
    summary = (
        f"Document forged and stored at `{stored_path}` "
        f"({byte_count} bytes, {heading_count} headings)."
    )
    return OperationResult(
        output_text=summary,
        output_json={
            "path": stored_path,
            "bytes": byte_count,
            "heading_count": heading_count,
            "outline": outline,
        },
        extra={
            "memory_context_lines": memory_context.count("\n") + 1 if memory_context else 0,
            "document_path": stored_path,
        },
    )
