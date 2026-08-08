"""Generic runtime for prompt-authored declarative operations.

Declarative operations are authored as markdown instruction assets with YAML
front matter. They do not carry operation-specific Python execution logic. Any
imperative behavior must happen through declared tool calls issued by the model
and routed through the injected tool executor.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from kernel.approval import ApprovalRequired
from kernel.config import settings as _kernel_settings
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    ToolExecutionContext,
    ToolResult,
)
from kernel.operations.registry import OperationExecutionError
from kernel.operations.structured_output import (
    completion_content,
    is_structured_output_unsupported,
    json_only_instruction,
    json_only_retry_messages,
    parse_structured_payload,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = _kernel_settings.declarative_completion_max_tokens
_DEFAULT_TEMPERATURE = _kernel_settings.declarative_completion_temperature
_MISSING = object()


def _build_user_message(ctx: OperationExecutionContext) -> str:
    sections: list[str] = []
    if ctx.user_block.strip():
        sections.append(f"User Context:\n{ctx.user_block.strip()}")
    if ctx.conversation_excerpt_markdown.strip():
        sections.append(f"Conversation Excerpt:\n{ctx.conversation_excerpt_markdown.strip()}")
    if ctx.response.strip():
        sections.append(f"Prior Response:\n{ctx.response.strip()}")
    if ctx.params:
        sections.append(
            "Execution Parameters:\n"
            f"{json.dumps(ctx.params, indent=2, sort_keys=True, default=str)}"
        )
    if ctx.query.strip():
        sections.append(f"User Query:\n{ctx.query.strip()}")
    else:
        sections.append("User Query:\n")
    return "\n\n".join(sections)


def _extract_tool_name(tool_call: dict[str, Any]) -> str:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        return str(function_payload.get("name") or "")
    return str(tool_call.get("name") or "")


def _extract_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function")
    raw_arguments: Any
    if isinstance(function_payload, dict):
        raw_arguments = function_payload.get("arguments")
    else:
        raw_arguments = tool_call.get("arguments")

    if isinstance(raw_arguments, dict):
        return dict(raw_arguments)
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _resolve_tool_query(default_query: str, arguments: dict[str, Any]) -> str:
    """Choose the tool-call query payload when the model supplies one."""
    candidate = arguments.get("query")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return default_query


def _stringify_tool_result(result: ToolResult | str) -> str:
    if isinstance(result, str):
        return result
    if result.output_text.strip() and result.output_json:
        # Pack tools split a terse human summary (output_text) from the
        # structured payload (output_json). The model needs both: the
        # summary line plus the full data, otherwise it sees e.g.
        # "Listed 5 entries." with no entry content.
        return f"{result.output_text.strip()}\n{json.dumps(result.output_json, indent=2, sort_keys=True)}"
    if result.output_text.strip():
        return result.output_text
    if result.output_json:
        return json.dumps(result.output_json, indent=2, sort_keys=True)
    return "Done"


def _empty_content_detail(
    operation_id: str,
    finish_reason: Any,
    completion_tokens: Any,
    max_tokens: int,
) -> str:
    """Render a diagnostic suffix for the empty-content failure path.

    ``finish_reason=length`` with ``completion_tokens`` near ``max_tokens``
    means the reasoning model spent its budget on internal reasoning before
    emitting visible content — the op needs more headroom, not a retry. Any
    other finish reason with zero content is a genuine provider-empty
    response. Surfacing both fields makes the two cases distinguishable in
    logs and in the ``OperationExecutionError`` message, which is what the
    2026-07-30 ``web_search`` and ``plan_workflow`` post-mortems asked for.
    """
    parts: list[str] = []
    if finish_reason is not None:
        parts.append(f"finish_reason={finish_reason}")
    if completion_tokens is not None:
        parts.append(f"completion_tokens={completion_tokens}")
    parts.append(f"max_tokens={max_tokens}")
    if str(finish_reason) == "length" and completion_tokens is not None:
        try:
            if int(completion_tokens) >= int(max_tokens) * 0.9:
                parts.append("likely_token_budget_truncation")
        except (TypeError, ValueError):
            pass
    return f"[{', '.join(parts)}]"


# Structured-output helpers (extraction, tolerant parse, json-only retry) live in
# kernel.operations.structured_output and are re-exported here so legacy callers
# keep working. Prefer importing from structured_output directly in new code.
_is_structured_output_unsupported = is_structured_output_unsupported
_build_json_only_instruction = json_only_instruction
_parse_structured_payload = parse_structured_payload


def _build_operation_result_from_structured_payload(
    *,
    operation_id: str,
    instruction_markdown_path: str,
    used_tools: list[str],
    response_output_text_field: str,
    response_output_json_passthrough: dict[str, str],
    parsed_payload: BaseModel,
    last_tool_output_json: dict[str, Any] | None,
) -> OperationResult:
    """Build one OperationResult from a parsed structured-output payload."""
    payload = parsed_payload.model_dump(mode="json")
    if last_tool_output_json is not None:
        for destination_field, source_path in response_output_json_passthrough.items():
            resolved = _resolve_json_path(last_tool_output_json, source_path)
            if resolved is not _MISSING:
                payload[destination_field] = deepcopy(resolved)
    output_text = str(payload.get(response_output_text_field) or "").strip()
    if not output_text:
        for candidate_field in ("body_markdown", "markdown", "answer_summary", "summary"):
            candidate = str(payload.get(candidate_field) or "").strip()
            if candidate:
                output_text = candidate
                break
    if not output_text:
        output_text = json.dumps(payload, indent=2, sort_keys=True)

    return OperationResult(
        output_text=output_text,
        output_json=payload,
        extra={
            "operation_id": operation_id,
            "execution_kind": "prompt",
            "instruction_markdown_path": instruction_markdown_path,
            "used_tools": used_tools,
        },
    )


def _resolve_json_path(source: dict[str, Any], path: str) -> Any:
    """Resolve one dotted path from a JSON-like mapping, or _MISSING."""
    if not path.strip():
        return _MISSING

    current: Any = source
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


@dataclass(slots=True)
class DeclarativeOperationModule:
    """Small execute-compatible wrapper around one markdown-authored operation."""

    operation_id: str
    instruction_markdown_path: str
    instruction_body: str
    max_calls_per_turn: int = 0
    response_format_model: type[BaseModel] | None = None
    response_output_text_field: str = "output_text"
    response_output_json_passthrough: dict[str, str] | None = None
    completion_max_tokens: int = _DEFAULT_MAX_TOKENS
    completion_temperature: float = _DEFAULT_TEMPERATURE

    async def execute(self, ctx: OperationExecutionContext) -> OperationResult:
        return await execute_declarative_operation(
            ctx,
            operation_id=self.operation_id,
            instruction_markdown_path=self.instruction_markdown_path,
            instruction_body=self.instruction_body,
            max_calls_per_turn=self.max_calls_per_turn,
            response_format_model=self.response_format_model,
            response_output_text_field=self.response_output_text_field,
            response_output_json_passthrough=self.response_output_json_passthrough or {},
            completion_max_tokens=self.completion_max_tokens,
            completion_temperature=self.completion_temperature,
        )


async def execute_declarative_operation(
    ctx: OperationExecutionContext,
    *,
    operation_id: str,
    instruction_markdown_path: str,
    instruction_body: str,
    max_calls_per_turn: int,
    response_format_model: type[BaseModel] | None,
    response_output_text_field: str,
    response_output_json_passthrough: dict[str, str],
    completion_max_tokens: int,
    completion_temperature: float,
) -> OperationResult:
    """Run one markdown-authored operation through the generic inference loop."""
    if ctx.host is None:
        raise RuntimeError(
            f"Declarative operation '{operation_id}' is missing the OperationHost seam."
        )

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "gpt-chat-latest"
    model_def = ctx.host.resolve_model(preferred_model_id)
    if model_def is None:
        raise RuntimeError(
            f"Declarative operation '{operation_id}' could not resolve model '{preferred_model_id}'."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instruction_body.strip()},
        {"role": "user", "content": _build_user_message(ctx)},
    ]
    used_tools: list[str] = []
    # ``max_calls_per_turn`` follows the catalog convention where ``0`` means
    # unlimited (``OperationDefinition.max_calls_per_turn``). Treating ``0`` as
    # "zero calls allowed" makes the very first tool call return
    # "Tool call budget exhausted." — the 2026-08-07 incident: ``web_search``
    # (no authored cap, defaulting to 0) exhausted on its first invocation.
    # Only a positive authored cap bounds the loop; ``0`` disables the gate.
    remaining_tool_calls = max_calls_per_turn if max_calls_per_turn > 0 else -1
    last_tool_output_json: dict[str, Any] | None = None

    while True:
        should_retry_for_json_only = False
        try:
            completion = await ctx.host.complete(
                messages=messages,
                model_def=model_def,
                max_tokens=completion_max_tokens,
                temperature=completion_temperature,
                tools=ctx.tool_schemas or None,
                response_format=response_format_model,
            )
            # Use the canonical extractor so reasoning models (gpt-5.x,
            # glm-5.2, kimi-k2.6) that put the answer in ``reasoning_content``
            # while leaving ``content`` empty are parsed correctly. Reading
            # ``completion.content`` raw was the 2026-07-30 ``web_search``
            # root cause: the empty-content guard skipped parsing, the retry
            # hit the same condition, and the op hard-failed.
            content_text = completion_content(completion)
            parsed_payload: BaseModel | None = None
            if response_format_model is not None and content_text:
                try:
                    parsed_payload = _parse_structured_payload(
                        response_format_model,
                        content_text,
                    )
                except ValidationError:
                    should_retry_for_json_only = True
        except Exception as exc:  # pylint: disable=broad-except
            if response_format_model is None or not _is_structured_output_unsupported(exc):
                raise
            should_retry_for_json_only = True
        if should_retry_for_json_only and response_format_model is not None:
            completion = await ctx.host.complete(
                messages=json_only_retry_messages(messages, response_format_model),
                model_def=model_def,
                max_tokens=completion_max_tokens,
                temperature=completion_temperature,
                tools=ctx.tool_schemas or None,
            )
            content_text = completion_content(completion)
            # A JSON-only retry may legitimately answer with a tool-call-only
            # turn. The web_search operation must dispatch its web_search tool
            # before it can return the final JSON envelope, so a tool-call turn
            # has empty visible content by design. Firing the empty-content
            # guard for it hard-fails the operation exactly when the model
            # correctly wants to keep searching — the 2026-08-01 ``web_search``
            # incidents surfaced this as ``finish_reason=tool_calls`` followed
            # by "returned no content after JSON-only retry". Defer parsing and
            # fall through to the shared tool-execution loop below instead.
            if completion.tool_calls:
                parsed_payload = None
            elif content_text:
                try:
                    parsed_payload = _parse_structured_payload(
                        response_format_model,
                        content_text,
                    )
                except ValidationError:
                    # Retry returned non-empty prose again (the model ignored
                    # the JSON-only instruction). Degrade gracefully to the
                    # raw content rather than leaking a second ValidationError.
                    parsed_payload = None
            else:
                # No tool dispatch and no text -> genuine provider-empty
                # response or token-budget truncation. Surface a typed
                # ``OperationExecutionError`` with diagnostics so the turn
                # runner can treat an empty upstream result as a hard failure
                # and the operator can distinguish truncation from a genuine
                # provider-empty response.
                finish_reason = getattr(completion, "finish_reason", None)
                completion_tokens = getattr(completion, "completion_tokens", None)
                detail = _empty_content_detail(
                    operation_id, finish_reason, completion_tokens, completion_max_tokens
                )
                logger.warning(
                    "declarative operation '%s' empty after retry: %s", operation_id, detail
                )
                raise OperationExecutionError(
                    f"Declarative operation '{operation_id}' returned no content "
                    f"after JSON-only retry; cannot parse structured output. {detail}"
                )
        else:
            parsed_payload = parsed_payload if response_format_model is not None else None

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if content_text:
            assistant_message["content"] = content_text
        if completion.tool_calls:
            assistant_message["tool_calls"] = completion.tool_calls
        messages.append(assistant_message)

        if not completion.tool_calls:
            if parsed_payload is not None:
                return _build_operation_result_from_structured_payload(
                    operation_id=operation_id,
                    instruction_markdown_path=instruction_markdown_path,
                    used_tools=used_tools,
                    response_output_text_field=response_output_text_field,
                    response_output_json_passthrough=response_output_json_passthrough,
                    parsed_payload=parsed_payload,
                    last_tool_output_json=last_tool_output_json,
                )
            return OperationResult(
                output_text=content_text.strip() or "Done",
                output_json={
                    "operation_id": operation_id,
                    "execution_kind": "prompt",
                    "instruction_markdown_path": instruction_markdown_path,
                    "used_tools": used_tools,
                },
            )

        if ctx.host is None:
            raise RuntimeError(
                f"Declarative operation '{operation_id}' is missing the OperationHost seam."
            )

        for tool_call in completion.tool_calls:
            tool_call_id = str(tool_call.get("id") or "")
            tool_name = _extract_tool_name(tool_call)
            tool_arguments = _extract_tool_arguments(tool_call)
            tool_output = "Tool call budget exhausted."

            if remaining_tool_calls != 0:
                if remaining_tool_calls > 0:
                    remaining_tool_calls -= 1
                try:
                    tool_result = await ctx.host.execute_tool(
                        tool_name,
                        ToolExecutionContext(
                            tool_id=tool_name,
                            workspace_id=ctx.workspace_id,
                            user_id=ctx.user_id,
                            user_slug=ctx.user_slug,
                            surface_type=ctx.surface_type,
                            surface_id=ctx.surface_id,
                            memory_scope=ctx.memory_scope,
                            mode_id=ctx.mode_id,
                            query=_resolve_tool_query(ctx.query, tool_arguments),
                            parameters=tool_arguments,
                            pack_configuration_resolver=ctx.pack_configuration_resolver,
                            approval_service=ctx.approval_service,
                            approval_request_id=ctx.approval_request_id,
                            session_plan_digest=ctx.session_plan_digest,
                            code_change_artifact_store=ctx.code_change_artifact_store,
                            event_bus=ctx.event_bus,
                            state_provider=ctx.state_provider,
                            data_dir=ctx.data_dir,
                        ),
                    )
                    if tool_name and tool_name not in used_tools:
                        used_tools.append(tool_name)
                    if isinstance(tool_result, ToolResult) and tool_result.output_json:
                        last_tool_output_json = dict(tool_result.output_json)
                    tool_output = _stringify_tool_result(tool_result)
                except ApprovalRequired:
                    raise
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("declarative operation tool '%s' failed: %s", tool_name, exc)
                    tool_output = f"Tool execution failed: {type(exc).__name__}: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_output,
                }
            )
