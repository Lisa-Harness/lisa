"""Forge-document operation — memory-grounded markdown synthesis.

Pipeline:

1. Recall context from long-term memory via the ``memory.search`` MCP alias
   (routed by the host to the configured memory backend).
2. Call an LLM with an authored system prompt + the user input + recalled
   context to synthesize a markdown document.
3. Validate the synthesized document's structure by calling the
   ``markdown_generate`` tool (backed by ``markdown-it-py``). The
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
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

# Tool ids — declared in the operation YAML ``tool_ids`` and routed through
# ``ctx.execute_tool_fn``. ``markdown_generate`` is a pack-contributed
# in-process tool; ``write_file`` and ``memory.search`` are MCP tools served
# by ``lisa-storage`` and the memory backend respectively.
_OUTLINE_TOOL_ID = "markdown_generate"
_WRITE_TOOL_ID = "write_file"
_READ_TOOL_ID = "read_file"
_MEMORY_SEARCH_TOOL_ID = "memory.search"

# User-instruction phrases that opt out of model attribution.
_NO_ANNOTATE_PHRASES = (
    "don't annotate the model",
    "do not annotate the model",
    "no model annotation",
    "no model attribution",
    "without model annotation",
)

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
_CONTINUE_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_continue.md"

# Token cap for the synthesizing pass. Bumped from 2000 → 7500 on
# 2026-07-18 after a local Ollama gemma4:cloud run truncated a detailed
# SWOT mid-section ("finished" at section 2.1 with finish_reason=unknown).
# 2000 tokens is tight for any document with a SWOT + architectural
# invariants section; 7500 matches the yml ``completion_max_tokens`` so
# the python-authored op and the declarative metadata agree. See
# /memories/repo/forge-document-truncation-contract.md.
_MAX_COMPLETION_TOKENS = 7500
# Continuation pass uses the same cap. The continuation appends only the
# missing tail, so the same budget is sufficient for nearly all cases.
_MAX_CONTINUE_TOKENS = _MAX_COMPLETION_TOKENS
# Hard ceiling on continuation attempts. One retry covers the
# "model ran out of tokens one section early" case; two covers the
# rare "still truncated, one more chunk needed" case. Beyond this we
# give up and annotate the document as truncated rather than looping.
_MAX_CONTINUE_ATTEMPTS = 2
_MAX_RECALL_RECORDS = 8
_MAX_DOCUMENT_CHARS = 20_000
_MAX_QUERY_CHARS = 2_000

# Minimum structural bar for a forged document: at least one heading.
_MIN_HEADINGS = 1

# Finish-reason values that signal the model stopped before completing
# the document. ``"length"`` is the OpenAI-standard "hit max_tokens"
# signal. ``"unknown"`` / ``None`` cover local backends (Ollama) that do
# not populate ``finish_reason`` — empirically these stop mid-stream on
# long documents. ``"stop"`` and ``"tool_calls"`` are NOT truncation.
_TRUNCATING_FINISH_REASONS = frozenset({"length", "unknown", None})


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
    annotate_model: bool = Field(
        default=True,
        description="When True, append a non-rendering HTML comment to the forged "
        "document noting the provider/model that wrote it. The user can also "
        "opt out by including phrases like 'don't annotate the model' in the "
        "instruction. Set False to suppress unconditionally.",
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
        raise RuntimeError("markdown requires execute_tool_fn to call memory/storage tools")

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
        logger.warning("markdown: memory.search returned error: %s", error_text)
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
) -> tuple[str, Any, str | None]:
    """Call the host's completion function with the forging prompt.

    Returns a ``(content, model_def, finish_reason)`` triple so the caller
    can annotate the document with the resolved model and detect
    mid-document truncation (``finish_reason in {"length", "unknown", None}``).
    """
    if ctx.complete_fn is None or ctx.resolve_model_fn is None:
        raise RuntimeError("markdown requires complete_fn and resolve_model_fn")

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.resolve_model_fn(preferred_model_id)
    if model_def is None:
        raise RuntimeError(
            f"markdown could not resolve model '{preferred_model_id}'"
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
    finish_reason = _coerce_finish_reason(getattr(completion, "finish_reason", None))
    return content.strip(), model_def, finish_reason


def _coerce_finish_reason(raw: Any) -> str | None:
    """Normalize a provider finish_reason into a comparable string or None.

    Providers occasionally return ``""`` or whitespace for "no signal"; coerce
    those to ``None`` so they fall into the truncation set (a missing signal
    on a long document is empirically a truncation, not a clean stop).
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    return text


def _is_truncated(finish_reason: str | None) -> bool:
    """Return True when the finish reason signals mid-stream truncation."""
    normalized = finish_reason if finish_reason is not None else None
    return normalized in _TRUNCATING_FINISH_REASONS


async def _continue_llm(
    ctx: OperationExecutionContext,
    user_input: str,
    document_so_far: str,
    model_def: Any,
) -> tuple[str, str | None]:
    """Issue one continuation pass to finish a truncated document.

    Appends a continuation instruction to the original message list (so the
    model has full context) and asks for ONLY the remaining tail. Returns
    ``(continuation_content, finish_reason)`` where ``continuation_content``
    is the raw model output (caller extracts the fenced markdown block).
    """
    if ctx.complete_fn is None:
        raise RuntimeError("markdown requires complete_fn to continue a truncated document")

    continue_template = _load_text(ctx, _CONTINUE_PROMPT_PATH)
    continue_prompt = _format_template(
        continue_template,
        user_input=user_input,
        document_so_far=document_so_far,
    )
    messages = [
        {"role": "user", "content": continue_prompt},
    ]
    completion = await ctx.complete_fn(
        messages=messages,
        model_def=model_def,
        max_tokens=_MAX_CONTINUE_TOKENS,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = _coerce_finish_reason(getattr(completion, "finish_reason", None))
    return content.strip(), finish_reason


def _looks_complete(document: str) -> bool:
    """Heuristic: does the document end on a clean paragraph/list boundary?

    Used after a continuation pass to decide whether the merged document is
    plausibly finished or whether we should try another continuation. A
    document ending mid-word, mid-sentence, or with a dangling list marker
    is treated as still-truncated.
    """
    if not document:
        return False
    stripped = document.rstrip()
    if not stripped:
        return False
    # Dangling list marker (e.g. "- " or "1. " at end of file) = truncated.
    if re.search(r"[\-\*\+] +$", stripped):
        return False
    # Dangling fenced code block (odd number of fences) = truncated.
    fence_count = len(re.findall(r"^```", stripped, flags=re.MULTILINE))
    if fence_count % 2 == 1:
        return False
    last_line = stripped.splitlines()[-1].rstrip()
    if not last_line:
        return True  # ends on a blank line — clean
    # Ends with sentence-final punctuation, a heading, or a code fence.
    if last_line.endswith(("```", ".", "!", "?", ":", ";", ">", ")")):
        return True
    # Ends with a list item that has content after the marker.
    if re.match(r"^\s*[\-\*\+] +\S", last_line) or re.match(r"^\s*\d+\. +\S", last_line):
        return True
    # Ends with a heading line.
    if last_line.lstrip().startswith("#"):
        return True
    # Otherwise: mid-sentence — treat as truncated.
    return False


def _merge_continuation(document_so_far: str, tail: str) -> str:
    """Concatenate the document-so-far with a continuation tail.

    The continuation prompt asks for ONLY the remaining body, but local
    models sometimes re-emit a leading heading or blank line. This helper
    collapses the join boundary to exactly one blank line and strips any
    leading blank lines from the tail so the merge does not produce
    doubled paragraph breaks.
    """
    head = document_so_far.rstrip()
    tail_body = tail.lstrip("\n").rstrip()
    if not tail_body:
        return document_so_far
    if not head:
        return tail_body + "\n"
    return head + "\n\n" + tail_body + "\n"


def _tail_is_duplicate(document_so_far: str, tail: str) -> bool:
    """Return True when the continuation just re-emits the existing tail.

    Some local models, when asked to "continue", repeat the last paragraph or
    the entire document-so-far instead of appending new content. Detecting
    this prevents the merge from producing a doubled or looped document. The
    check is conservative: it only flags tails whose non-empty lines are a
    contiguous suffix of the existing document.
    """
    head_lines = [ln for ln in document_so_far.splitlines() if ln.strip()]
    tail_lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not tail_lines:
        return True
    if len(tail_lines) > len(head_lines):
        # The continuation is longer than the entire document-so-far — it
        # must contain new content, so not a pure duplicate.
        return False
    # Compare the last len(tail_lines) lines of the head to the tail.
    suffix = head_lines[-len(tail_lines):]
    return suffix == tail_lines


def _extract_markdown_block(text: str) -> str:
    """Extract the first fenced ```markdown ...``` block, else return text.

    Accepts fenced blocks tagged ``markdown`` or untagged. Falls back to the
    raw text when no fence is present so the outline tool can still assess it.
    """
    match = re.search(r"```(?:markdown|md)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _format_markdown(document: str) -> str:
    """Lint/format the document via ``mdformat`` if available.

    ``mdformat`` is the canonical CommonMark formatter and is the natural
    companion to ``markdown-it-py`` (which the pack already depends on for
    outline validation). When installed it normalizes bullet markers,
    collapses redundant blank lines, fixes heading spacing, and guarantees a
    single trailing newline — producing clean, consistent output. When not
    installed the document passes through unchanged so the pack remains
    importable without the optional dependency.
    """
    try:
        import mdformat  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("mdformat not installed; skipping markdown formatting")
        return document
    try:
        return mdformat.text(document, options={"number": True}).rstrip() + "\n"
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("forge_document: mdformat failed (%s); using unformatted document", exc)
        return document


def _build_model_annotation(
    model_def: Any,
    *,
    truncated: bool = False,
    final_finish_reason: str | None = None,
    continue_attempts: int = 0,
) -> str:
    """Build a trailing HTML-comment attribution line for the resolved model.

    The comment is non-rendering in every CommonMark renderer, so it annotates
    the source without polluting the visible document.

    When ``truncated`` is True, an explicit ``truncated by model`` marker is
    appended so a reader scanning the artifact knows the document is
    incomplete even though the operation persisted it. The final
    ``finish_reason`` and the number of continuation attempts are included
    for triage.
    """
    provider = str(getattr(model_def, "provider", "") or "").strip()
    model_id = str(getattr(model_def, "model_id", "") or "").strip()
    parts: list[str] = []
    if provider:
        parts.append(f"provider: {provider}")
    if model_id:
        parts.append(f"model: {model_id}")
    if truncated:
        reason = final_finish_reason or "unknown"
        parts.append(f"truncated by model · finish_reason: {reason}")
        if continue_attempts:
            parts.append(f"continue_attempts: {continue_attempts}")
    if not parts:
        return ""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"<!-- generated by lisa · {' · '.join(parts)} · {timestamp} -->"


def _user_asked_not_to_annotate(user_input: str) -> bool:
    """Return True when the user instruction explicitly opts out of attribution."""
    lowered = user_input.lower()
    return any(phrase in lowered for phrase in _NO_ANNOTATE_PHRASES)


async def _validate_outline(ctx: OperationExecutionContext, document: str) -> dict[str, Any]:
    """Run the ``markdown_generate`` tool and return its outline dict.

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


async def _continue_until_complete(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    document: str,
    model_def: Any,
    first_finish_reason: str | None,
) -> tuple[str, bool, int, str | None]:
    """Run continuation passes until the document looks complete or budget is hit.

    Returns ``(document, was_truncated, attempts, final_finish_reason)``.
    The caller annotates and persists ``document``; the other three values
    feed the truncated-annotation marker and the triage fields in
    ``OperationResult.output_json``.

    The contract is documented in
    /memories/repo/forge-document-truncation-contract.md:
    - Only truncate-flagged first passes (``length``/``unknown``/``None``)
      trigger continuation; clean ``stop`` returns the document as-is.
    - A continuation that returns empty or duplicates the existing tail is
      treated as "model has nothing more to add" — stop and mark truncated.
    - A merged document that ends on a clean boundary (``_looks_complete``)
      AND whose continuation reported a non-truncating finish_reason is
      considered complete; otherwise we keep going up to the budget.
    """
    if not _is_truncated(first_finish_reason):
        return document, False, 0, first_finish_reason

    logger.warning(
        "forge_document: first pass finish_reason=%r; attempting continuation",
        first_finish_reason,
    )
    attempts = 0
    final_finish_reason = first_finish_reason
    current = document
    for attempt in range(1, _MAX_CONTINUE_ATTEMPTS + 1):
        attempts = attempt
        continuation_text, cont_finish_reason = await _continue_llm(
            ctx,
            user_input=user_input,
            document_so_far=current,
            model_def=model_def,
        )
        final_finish_reason = cont_finish_reason
        if not continuation_text or not continuation_text.strip():
            logger.warning(
                "forge_document: continuation pass %d returned empty; giving up",
                attempt,
            )
            break
        tail = _extract_markdown_block(continuation_text)
        # If the fenced block extracted was empty but the raw continuation
        # had content, the model emitted prose outside the fence (or an
        # empty fence). Either way, there is no usable markdown tail to
        # append — treat this as "continuation had nothing to add" and
        # stop. Persisting prose-as-tail would corrupt the document.
        if not tail.strip():
            logger.warning(
                "forge_document: continuation pass %d yielded no markdown tail; giving up",
                attempt,
            )
            break
        if _tail_is_duplicate(current, tail):
            logger.warning(
                "forge_document: continuation pass %d duplicated existing tail; giving up",
                attempt,
            )
            break
        current = _merge_continuation(current, tail)
        if _looks_complete(current) and not _is_truncated(cont_finish_reason):
            logger.info(
                "forge_document: continuation pass %d completed the document", attempt,
            )
            return current, False, attempts, cont_finish_reason
        if attempt >= _MAX_CONTINUE_ATTEMPTS:
            logger.warning(
                "forge_document: exhausted %d continuation attempts; "
                "persisting with truncated annotation (final finish_reason=%r)",
                attempt, cont_finish_reason,
            )
    return current, True, attempts, final_finish_reason


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
    # _call_llm returns (content, model_def, finish_reason).
    completion_text, model_def, first_finish_reason = completion
    if not completion_text or not completion_text.strip():
        return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))

    document = _extract_markdown_block(completion_text)
    if not document.strip():
        text = _format_template(
            _load_text(ctx, _EMPTY_DOCUMENT_PATH),
            response_preview=completion_text[:400],
        )
        return OperationResult(output_text=text)

    # Truncation handling: when the first pass stopped mid-document
    # (finish_reason in {length, unknown, None}), issue up to
    # _MAX_CONTINUE_ATTEMPTS continuation passes. Each pass appends only the
    # missing tail. If the merged document still looks truncated after the
    # budget is exhausted, mark it truncated and annotate the persisted
    # artifact so a reader knows it is incomplete. See
    # /memories/repo/forge-document-truncation-contract.md.
    document, was_truncated, continue_attempts, final_finish_reason = (
        await _continue_until_complete(
            ctx,
            user_input=user_input,
            document=document,
            model_def=model_def,
            first_finish_reason=first_finish_reason,
        )
    )

    # Lint/format the document so the persisted output is clean and consistent.
    document = _format_markdown(document)

    if len(document) > _MAX_DOCUMENT_CHARS:
        document = document[:_MAX_DOCUMENT_CHARS] + "\n\n<!-- truncated -->\n"
        # A char-budget truncation overrides the model-finish signal: the
        # artifact is definitely truncated.
        was_truncated = True

    # Annotate with the model that wrote it, unless the user opted out.
    annotate_model = bool(params.get("annotate_model", True))
    if annotate_model and not _user_asked_not_to_annotate(user_input):
        annotation = _build_model_annotation(
            model_def,
            truncated=was_truncated,
            final_finish_reason=final_finish_reason if was_truncated else None,
            continue_attempts=continue_attempts if was_truncated else 0,
        )
        if annotation:
            document = document.rstrip() + "\n\n" + annotation + "\n"

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
    if was_truncated:
        summary += (
            f" WARNING: persisted truncated after {continue_attempts} "
            f"continuation attempt(s) (final finish_reason="
            f"{final_finish_reason!r}). See annotation in document."
        )
    return OperationResult(
        output_text=summary,
        output_json={
            "path": stored_path,
            "bytes": byte_count,
            "heading_count": heading_count,
            "outline": outline,
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
            "first_finish_reason": first_finish_reason,
            "final_finish_reason": final_finish_reason,
        },
        extra={
            "memory_context_lines": memory_context.count("\n") + 1 if memory_context else 0,
            "document_path": stored_path,
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
        },
    )
