"""Advise-models operation: NL-driven model manifest editing.

Reads the current model manifest (proposal if it exists, else active), pipes
it + the user's natural-language instruction through an LLM, validates the
edited YAML against the kernel's manifest loader, and writes the result
back to the proposal file. The user still runs ``/models apply`` to promote.

The operation does NOT touch the filesystem directly. It reads and writes
the manifest through the ``lisa-storage`` MCP server's
``read_model_manifest`` and ``write_model_manifest`` tools, invoked via
``ctx.execute_tool_fn`` — exactly the pattern used by the checkpoint pack.
The host owns ``data_dir``; the pack never assumes where the manifest lives.

All system prompts, user-facing text, and error messages are externalized
as authored markdown under ``operations/advise_models/prompts/`` and loaded
through ``ctx.read_authored_text_fn`` (the SDK content-registry loader). The
operation never embeds prompt prose — only path constants and template
substitution.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext

logger = logging.getLogger(__name__)

_MAX_MANIFEST_CHARS = 24_000
_MAX_EDITED_TOKENS = 8000

# Storage tool ids — declared in the operation YAML ``tool_ids`` and routed
# through ``ctx.execute_tool_fn`` to the ``lisa-storage`` MCP server.
_READ_TOOL_ID = "read_model_manifest"
_WRITE_TOOL_ID = "write_model_manifest"

# Authored-text paths — loaded via ctx.read_authored_text_fn (SDK loader).
# Paths are relative to the pack's content root, matching the checkpoint
# pack convention (``operations/prompts/<name>.md``).
_PROMPT_DIR = "operations/advise_models/prompts"
_SYSTEM_PROMPT_PATH = f"{_PROMPT_DIR}/advise_models_system.md"
_USER_PROMPT_PATH = f"{_PROMPT_DIR}/advise_models_user.md"
_USAGE_PATH = f"{_PROMPT_DIR}/advise_models_usage.md"
_NO_MANIFEST_PATH = f"{_PROMPT_DIR}/advise_models_no_manifest.md"
_TOO_LARGE_PATH = f"{_PROMPT_DIR}/advise_models_too_large.md"
_EMPTY_RESPONSE_PATH = f"{_PROMPT_DIR}/advise_models_empty_response.md"
_NO_YAML_BLOCK_PATH = f"{_PROMPT_DIR}/advise_models_no_yaml_block.md"
_VALIDATION_FAILED_PATH = f"{_PROMPT_DIR}/advise_models_validation_failed.md"


def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    """Load authored markdown via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content. Production
    hosts wire ``read_authored_text_fn`` via the turn hooks builder; tests
    inject a fake loader that returns the authored text.
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
    """Substitute ``{placeholder}`` tokens in a loaded template.

    Uses ``str.format`` with a safe mapping so missing keys leave the
    placeholder intact rather than raising.
    """

    try:
        return template.format_map(_SafeDict(kwargs))
    except Exception:  # pylint: disable=broad-except
        return template


class _SafeDict(dict):
    """dict subclass that returns ``{key}`` for missing keys in str.format_map."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Edit the model manifest according to a natural-language instruction."""

    instruction = ctx.query.strip()
    if not instruction:
        return OperationResult(output_text=_load_text(ctx, _USAGE_PATH))

    if not callable(ctx.execute_tool_fn):
        raise RuntimeError(
            "advise_models requires execute_tool_fn to call lisa-storage tools"
        )

    current_yaml = await _read_manifest(ctx, "proposal")
    if current_yaml is None:
        current_yaml = await _read_manifest(ctx, "active")
    if current_yaml is None:
        return OperationResult(output_text=_load_text(ctx, _NO_MANIFEST_PATH))

    if len(current_yaml) > _MAX_MANIFEST_CHARS:
        text = _format_template(
            _load_text(ctx, _TOO_LARGE_PATH),
            manifest_chars=len(current_yaml),
        )
        return OperationResult(output_text=text)

    completion = await _call_llm(ctx, current_yaml, instruction)
    if not completion or not completion.strip():
        return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))

    edited_yaml = _extract_yaml_block(completion)
    if not edited_yaml:
        text = _format_template(
            _load_text(ctx, _NO_YAML_BLOCK_PATH),
            response_preview=completion[:400],
        )
        return OperationResult(output_text=text)

    validation_error = _validate_manifest_yaml(edited_yaml)
    if validation_error:
        text = _format_template(
            _load_text(ctx, _VALIDATION_FAILED_PATH),
            validation_error=validation_error,
        )
        return OperationResult(output_text=text)

    await _write_manifest(ctx, edited_yaml, "proposal")
    summary = _render_diff_summary(current_yaml, edited_yaml)
    return OperationResult(output_text=summary)


async def _read_manifest(ctx: OperationExecutionContext, which: str) -> str | None:
    """Read the active or proposed manifest via the lisa-storage tool.

    Returns the manifest text, or ``None`` if it does not exist. Raises
    ``RuntimeError`` on any other tool failure.
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
            parameters={"which": which},
        ),
    )
    return _parse_storage_text_result(tool_result, which)


async def _write_manifest(ctx: OperationExecutionContext, content: str, which: str) -> None:
    """Write the manifest text via the lisa-storage tool.

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
            parameters={"content": content, "which": which},
        ),
    )
    _raise_on_storage_error(tool_result, _WRITE_TOOL_ID)


def _parse_storage_text_result(tool_result: Any, which: str) -> str | None:
    """Extract manifest text from a read_model_manifest tool result.

    Returns ``None`` when the storage server reports the manifest is not
    found (a normal condition — the proposal may not exist yet). Any other
    error raises ``RuntimeError``.
    """

    output_text = getattr(tool_result, "output_text", "") or str(tool_result)
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{_READ_TOOL_ID} failed: {error_text}")
    if not output_text.strip():
        return None
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return output_text
    if isinstance(data, dict) and "error" in data:
        if "not found" in str(data["error"]):
            return None
        raise RuntimeError(f"{_READ_TOOL_ID} failed: {data['error']}")
    return output_text


def _raise_on_storage_error(tool_result: Any, tool_id: str) -> None:
    """Raise if a storage tool result carries an error."""

    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{tool_id} failed: {error_text}")
    output_text = getattr(tool_result, "output_text", "") or str(tool_result)
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"{tool_id} failed: {data['error']}")


async def _call_llm(
    ctx: OperationExecutionContext,
    current_yaml: str,
    instruction: str,
) -> str:
    """Call the host's completion function with the editing prompt."""

    if ctx.complete_fn is None or ctx.resolve_model_fn is None:
        raise RuntimeError("advise_models requires complete_fn and resolve_model_fn")

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.resolve_model_fn(preferred_model_id)
    if model_def is None:
        raise RuntimeError(
            f"advise_models could not resolve model '{preferred_model_id}'"
        )

    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_PATH)
    user_prompt = _format_template(
        user_template,
        instruction=instruction,
        current_manifest=current_yaml,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = await ctx.complete_fn(
        messages=messages,
        model_def=model_def,
        max_tokens=_MAX_EDITED_TOKENS,
        temperature=0.1,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    return content.strip()


def _extract_yaml_block(text: str) -> str:
    """Extract the first fenced ```yaml ...``` block from ``text``."""

    match = re.search(r"```(?:yaml|yml)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if not match:
        stripped = text.strip()
        if stripped.startswith("version:") or stripped.startswith("# "):
            return stripped
        return ""
    return match.group(1).strip()


def _validate_manifest_yaml(edited_yaml: str) -> str:
    """Validate the edited YAML through the kernel's manifest loader.

    Returns an empty string if valid, or the error message if invalid.
    """

    try:
        parsed = yaml.safe_load(edited_yaml)
    except yaml.YAMLError as exc:
        return f"YAML parse error: {exc}"

    if not isinstance(parsed, dict):
        return "Edited manifest is not a mapping."

    models = parsed.get("models")
    if not isinstance(models, list) or not models:
        return "Edited manifest must include a non-empty 'models' list."

    # Full validation through the kernel loader via a temp file.
    try:
        from kernel.model_manifest import load_model_manifest  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(edited_yaml)
            tmp_path = Path(tmp.name)
        try:
            load_model_manifest(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:  # pylint: disable=broad-except
        return f"Unexpected validation error: {exc}"
    return ""


def _render_diff_summary(
    current_yaml: str,
    edited_yaml: str,
) -> str:
    """Render a concise bullet-list summary of what changed."""

    try:
        current = yaml.safe_load(current_yaml) or {}
    except yaml.YAMLError:
        current = {}
    try:
        edited = yaml.safe_load(edited_yaml) or {}
    except yaml.YAMLError:
        edited = {}

    current_models = _index_models(current.get("models"))
    edited_models = _index_models(edited.get("models"))
    changes: list[str] = []

    for model_id, edited_row in edited_models.items():
        current_row = current_models.get(model_id, {})
        if not current_row:
            changes.append(f"+ {model_id}: added")
            continue
        for field in (
            "designations",
            "cost_tier",
            "latency_tier",
            "capabilities",
            "reasoning_model",
            "thinking_model",
            "display_name",
        ):
            old_val = current_row.get(field)
            new_val = edited_row.get(field)
            if old_val == new_val:
                continue
            old_repr = _short_repr(old_val)
            new_repr = _short_repr(new_val)
            if old_repr and not new_repr:
                changes.append(f"~ {model_id}: removed {field}")
            elif not old_repr and new_repr:
                changes.append(f"+ {model_id}: {field} = {new_repr}")
            else:
                changes.append(f"~ {model_id}: {field} {old_repr} → {new_repr}")

    for model_id in current_models:
        if model_id not in edited_models:
            changes.append(f"- {model_id}: removed")

    current_modes = current.get("mode_models") or {}
    edited_modes = edited.get("mode_models") or {}
    if current_modes != edited_modes:
        for mode_id, model_ref in edited_modes.items():
            if current_modes.get(mode_id) != model_ref:
                changes.append(f"~ mode_models: {mode_id} → {model_ref}")
        for mode_id in current_modes:
            if mode_id not in edited_modes:
                changes.append(f"- mode_models: {mode_id} removed")

    header = "Manifest updated.\n"
    if not changes:
        return header + "(no changes detected)"
    return header + "\n" + "\n".join(changes[:40])


def _index_models(rows: Any) -> dict[str, dict[str, Any]]:
    """Index a models list by model_id for diffing."""

    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return indexed
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id") or "").strip()
        if model_id:
            indexed[model_id] = row
    return indexed


def _short_repr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)