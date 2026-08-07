"""MCP-backed catalog adapters for workflow modes and operations.

Calls ``list_modes`` and ``list_operations`` on a ``lisa-catalog`` MCP server
via per-call stdio sessions, deserializes the JSON into kernel model objects,
and for runtime operations imports the implementation module locally.

The kernel never touches the filesystem for content discovery; it delegates
all scanning and front-matter parsing to the ``lisa-catalog`` server.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel

from kernel.models import (
    DefaultIntentBridge,
    OperationDefinition,
    ToolDefinition,
    WorkflowModeConfig,
    WorkflowStageConfig,
)
from kernel.operations.declarative import (
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_TEMPERATURE,
    DeclarativeOperationModule,
)
from kernel.tools.mcp_client import MCPClientConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CatalogResponseError",
    "MCPModeCatalog",
    "MCPOperationCatalog",
    "MCPToolCatalog",
]

_MAX_CATALOG_RESPONSE_PREVIEW_CHARS = 500


class CatalogResponseError(RuntimeError):
    """Raised when a lisa-catalog tool response is not a catalog object."""


def _extract_text(result: Any) -> str:
    """Extract text from an MCP call_tool result."""
    if result.content and len(result.content) > 0:
        text = result.content[0].text
        return text if isinstance(text, str) else str(text)
    return ""


def _parse_catalog_response(raw_response: str, *, tool: str) -> dict[str, Any]:
    """Decode one catalog object or raise an actionable MCP response error."""

    normalized = raw_response.strip()
    if not normalized:
        return {}
    try:
        decoded = json.loads(normalized)
    except json.JSONDecodeError as exc:
        preview = normalized[:_MAX_CATALOG_RESPONSE_PREVIEW_CHARS]
        raise CatalogResponseError(
            f"lisa-catalog tool '{tool}' returned invalid JSON: {preview}"
        ) from exc
    if not isinstance(decoded, dict):
        raise CatalogResponseError(
            f"lisa-catalog tool '{tool}' returned a JSON {type(decoded).__name__}; "
            "expected an object."
        )
    return decoded


# ── Mode catalog ────────────────────────────────────────────────────────────


class MCPModeCatalog:
    """Workflow mode catalog backed by the ``lisa-catalog`` MCP server.

    Calls ``list_modes`` and deserializes the JSON response into
    ``WorkflowModeConfig`` objects.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )
        self._catalog: dict[str, WorkflowModeConfig] | None = None

    def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        async def _run() -> str:
            async with (
                stdio_client(self._params) as (read, write_stream),
                ClientSession(read, write_stream) as session,
            ):
                await session.initialize()
                result = await session.call_tool(tool, arguments=arguments)
                return _extract_text(result)

        # Check for a running event loop BEFORE constructing the coroutine
        # to avoid the "coroutine was never awaited" RuntimeWarning.
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if in_loop:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _run())
                return future.result()
        return asyncio.run(_run())

    def get_workflow_catalog(self) -> dict[str, WorkflowModeConfig]:
        """Return the mode catalog, loading it from the MCP server on first call."""
        if self._catalog is not None:
            return self._catalog
        raw_json = self._call("list_modes", {})
        raw_catalog = _parse_catalog_response(raw_json, tool="list_modes")
        self._catalog = {
            mode_id: _deserialize_mode(mode_dict) for mode_id, mode_dict in raw_catalog.items()
        }
        return self._catalog

    def invalidate(self) -> None:
        """Clear the cached catalog so the next call reloads from the server."""
        self._catalog = None


def _require_start_stage_id(data: dict[str, Any]) -> str:
    """Return the start_stage_id, raising for workflow modes with an empty value.

    Chat/action modes may use ``"none"`` or an empty string (no stages);
    workflow modes require a non-empty start_stage_id that names a stage
    present in the stages dict. A malformed catalog response with an empty
    start_stage_id for a workflow mode is a hard failure, not a silent
    default to ``""``.
    """
    start_stage_id = str(data.get("start_stage_id", "") or "")
    mode_kind = str(data.get("mode_kind", "workflow") or "workflow")
    if mode_kind == "workflow" and not start_stage_id:
        mode_id = data.get("mode_id", "?")
        raise CatalogResponseError(
            f"Mode '{mode_id}' from lisa-catalog is a workflow mode with no "
            "start_stage_id — workflow modes require a start stage."
        )
    return start_stage_id


def _deserialize_mode(data: dict[str, Any]) -> WorkflowModeConfig:
    """Deserialize one mode dict from JSON into a ``WorkflowModeConfig``."""

    stages = {
        stage_id: _deserialize_stage(stage_dict)
        for stage_id, stage_dict in (data.get("stages") or {}).items()
    }
    intent_bridges = [
        DefaultIntentBridge(**bridge) for bridge in (data.get("default_intent_bridges") or [])
    ]

    return WorkflowModeConfig(
        mode_id=data["mode_id"],
        start_stage_id=_require_start_stage_id(data),
        stages=stages,
        mode_kind=data.get("mode_kind", "workflow"),
        artifact_type=data.get("artifact_type", "workflow_summary"),
        description=data.get("description", ""),
        instruction_body=data.get("instruction_body", ""),
        mode_markdown_path=data.get("mode_markdown_path", ""),
        default_model=data.get("default_model"),
        routable=data.get("routable", False),
        priority=data.get("priority", 0),
        fast_path_patterns=data.get("fast_path_patterns", []),
        routing_intent_tags=data.get("routing_intent_tags", []),
        routing_intent_keywords=data.get("routing_intent_keywords", {}),
        satisfies_intents=data.get("satisfies_intents", []),
        required_capabilities=data.get("required_capabilities", []),
        command_pattern=data.get("command_pattern", ""),
        require_query=data.get("require_query", False),
        entry_operation_ids=data.get("entry_operation_ids", []),
        entry_op_kind=data.get("entry_op_kind", "work"),
        pre_turn_operation_ids=data.get("pre_turn_operation_ids", []),
        post_turn_operation_ids=data.get("post_turn_operation_ids", []),
        allowed_operation_ids=data.get("allowed_operation_ids", []),
        allowed_tool_ids=data.get("allowed_tool_ids", []),
        authored_allowed_operation_ids=data.get("authored_allowed_operation_ids", []),
        authored_allowed_tool_ids=data.get("authored_allowed_tool_ids", []),
        tools_authoritative=data.get("tools_authoritative", True),
        accept_default_capabilities=data.get("accept_default_capabilities", True),
        excluded_operation_ids=data.get("excluded_operation_ids", []),
        excluded_tool_ids=data.get("excluded_tool_ids", []),
        entry_failure_policy=data.get("entry_failure_policy", "abort"),
        default_intent_bridges=intent_bridges,
        mode_provider_id=str(data.get("mode_provider_id", "") or ""),
        context_evidence_packs=[str(p) for p in (data.get("context_evidence_packs") or [])],
    )


def _deserialize_stage(data: dict[str, Any]) -> WorkflowStageConfig:
    """Deserialize one stage dict from JSON into a ``WorkflowStageConfig``."""

    return WorkflowStageConfig(
        stage_id=data["stage_id"],
        instruction_markdown_path=data.get("instruction_markdown_path", ""),
        instruction_body=data.get("instruction_body", ""),
        description=data.get("description", ""),
        default_model=data.get("default_model"),
        required_connectors=data.get("required_connectors", []),
        allowed_next_stages=data.get("allowed_next_stages", []),
        allow_user_questions=data.get("allow_user_questions", True),
        terminal=data.get("terminal", False),
    )


# ── Operation catalog ────────────────────────────────────────────────────────


class MCPOperationCatalog:
    """Operation catalog backed by the ``lisa-catalog`` MCP server.

    Calls ``list_operations`` and deserializes the JSON response into
    ``OperationDefinition`` objects. For runtime operations, the
    implementation module is imported locally using the ``module_path``
    field returned by the server.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )
        self._catalog: dict[str, OperationDefinition] | None = None

    def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        async def _run() -> str:
            async with (
                stdio_client(self._params) as (read, write_stream),
                ClientSession(read, write_stream) as session,
            ):
                await session.initialize()
                result = await session.call_tool(tool, arguments=arguments)
                return _extract_text(result)

        # Check for a running event loop BEFORE constructing the coroutine
        # to avoid the "coroutine was never awaited" RuntimeWarning.
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if in_loop:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _run())
                return future.result()
        return asyncio.run(_run())

    def get_operation_catalog(self) -> dict[str, OperationDefinition]:
        """Return the operation catalog, loading it from the MCP server on first call."""
        if self._catalog is not None:
            return self._catalog
        raw_json = self._call("list_operations", {})
        raw_catalog = _parse_catalog_response(raw_json, tool="list_operations")
        self._catalog = {
            op_id: _deserialize_operation(op_dict) for op_id, op_dict in raw_catalog.items()
        }
        return self._catalog

    def invalidate(self) -> None:
        """Clear the cached catalog so the next call reloads from the server."""
        self._catalog = None


def _deserialize_operation(data: dict[str, Any]) -> OperationDefinition:
    """Deserialize one operation dict from JSON into an ``OperationDefinition``.

    For prompt operations, a ``DeclarativeOperationModule`` is constructed
    from the instruction body and response format fields. For runtime
    operations, the implementation module is imported locally using
    ``module_path``.
    """

    execution_kind = data.get("execution_kind", "runtime")

    response_format_model = None
    response_format_model_path = data.get("response_format_model_path", "")
    if response_format_model_path:
        response_format_model = _resolve_model_class(response_format_model_path)

    implementation_module: Any = None

    if execution_kind == "prompt":
        implementation_module = DeclarativeOperationModule(
            operation_id=data["operation_id"],
            instruction_markdown_path=data.get("instruction_markdown_path", ""),
            instruction_body=data.get("instruction_body", ""),
            max_calls_per_turn=data.get("max_calls_per_turn", 0),
            response_format_model=response_format_model,
            response_output_text_field=data.get("response_output_text_field", "output_text"),
            response_output_json_passthrough=data.get("response_output_json_passthrough", {}),
            completion_max_tokens=data.get("completion_max_tokens", _DEFAULT_MAX_TOKENS),
            completion_temperature=data.get("completion_temperature", _DEFAULT_TEMPERATURE),
        )
    else:
        module_path = data.get("module_path", "")
        if module_path:
            implementation_module = _import_operation_module(module_path)

    parameters_schema = None
    if implementation_module is not None:
        parameters_schema = getattr(implementation_module, "Parameters", None)

    return OperationDefinition(
        operation_id=data["operation_id"],
        description=data.get("description", ""),
        pack_id=str(data.get("pack_id", "") or ""),
        provider_id=str(data.get("operation_provider_id", "") or ""),
        effect=data.get("effect", "read"),
        approval_required=bool(data.get("approval_required", False)),
        approval_preview=data.get("approval_preview", ""),
        execution_kind=execution_kind,
        parameters_schema=parameters_schema,
        response_format_model=response_format_model,
        response_format_model_path=response_format_model_path,
        response_output_text_field=data.get("response_output_text_field", "output_text"),
        response_output_json_passthrough=data.get("response_output_json_passthrough", {}),
        preferred_model_id=data.get("preferred_model_id"),
        tool_ids=data.get("tool_ids", []),
        mode_allowlist=data.get("mode_allowlist", []),
        max_calls_per_turn=data.get("max_calls_per_turn", 0),
        timeout_seconds=data.get("timeout_seconds", 60),
        completion_max_tokens=data.get("completion_max_tokens", _DEFAULT_MAX_TOKENS),
        completion_temperature=data.get("completion_temperature", _DEFAULT_TEMPERATURE),
        instruction_markdown_path=data.get("instruction_markdown_path", ""),
        instruction_body=data.get("instruction_body", ""),
        implementation_module=implementation_module,
        exportable=bool(data.get("exportable", False)),
        default_available=bool(data.get("default_available", False)),
    )


def _resolve_model_class(model_path: str) -> type[BaseModel] | None:
    """Import and return one Pydantic response-format model class."""

    if not model_path.strip():
        return None
    module_path, separator, attr_name = model_path.rpartition(".")
    if not separator or not module_path or not attr_name:
        logger.warning("Invalid response_format_model_path '%s'", model_path)
        return None
    try:
        module = importlib.import_module(module_path)
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed to import response_format module '%s'", module_path, exc_info=True)
        return None
    model_class = getattr(module, attr_name, None)
    if not isinstance(model_class, type) or not issubclass(model_class, BaseModel):
        logger.warning("response_format_model_path '%s' is not a BaseModel subclass", model_path)
        return None
    return model_class


def _import_operation_module(module_path: str) -> Any:
    """Import and return one operation implementation module."""

    try:
        return importlib.import_module(module_path)
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed to import operation module '%s'", module_path, exc_info=True)
        return None


# ── Tool catalog ──────────────────────────────────────────────────────────────


class MCPToolCatalog:
    """Internal tool catalog backed by the ``lisa-catalog`` MCP server.

    Calls ``list_tools`` and deserializes the JSON response into
    ``ToolDefinition`` objects. The implementation module is imported locally
    using the ``module_path`` field returned by the server — same pattern as
    ``MCPOperationCatalog``.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )
        self._catalog: dict[str, ToolDefinition] | None = None

    def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        async def _run() -> str:
            async with (
                stdio_client(self._params) as (read, write_stream),
                ClientSession(read, write_stream) as session,
            ):
                await session.initialize()
                result = await session.call_tool(tool, arguments=arguments)
                return _extract_text(result)

        # Check for a running event loop BEFORE constructing the coroutine
        # to avoid the "coroutine was never awaited" RuntimeWarning.
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if in_loop:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _run())
                return future.result()
        return asyncio.run(_run())

    def get_tool_catalog(self) -> dict[str, ToolDefinition]:
        """Return the tool catalog, loading it from the MCP server on first call."""
        if self._catalog is not None:
            return self._catalog
        raw_json = self._call("list_tools", {})
        raw_catalog = _parse_catalog_response(raw_json, tool="list_tools")
        self._catalog = {
            tool_id: _deserialize_tool(tool_dict) for tool_id, tool_dict in raw_catalog.items()
        }
        return self._catalog

    def invalidate(self) -> None:
        """Clear the cached catalog so the next call reloads from the server."""
        self._catalog = None


def _deserialize_tool(data: dict[str, Any]) -> ToolDefinition:
    """Deserialize one tool dict from JSON into a ``ToolDefinition``.

    Tools are runtime-only: the implementation module is imported locally
    using ``module_path``, and ``Parameters`` is pulled from the module for
    the OpenAI function-tool schema.
    """

    module_path = data.get("module_path", "")
    implementation_module: Any = None
    if module_path:
        implementation_module = _import_tool_module(module_path)

    parameters_schema = None
    if implementation_module is not None:
        parameters_schema = getattr(implementation_module, "Parameters", None)

    mode_allowlist_raw = data.get("mode_allowlist")
    mode_allowlist = (
        None if mode_allowlist_raw is None else [str(mode_id) for mode_id in mode_allowlist_raw]
    )

    return ToolDefinition(
        tool_id=data["tool_id"],
        description=data.get("description", ""),
        pack_id=str(data.get("pack_id", "") or ""),
        provider_id=str(data.get("tool_provider_id", "") or ""),
        effect=data.get("effect", "read"),
        approval_required=bool(data.get("approval_required", False)),
        approval_preview=data.get("approval_preview", ""),
        parameters_schema=parameters_schema,
        mode_allowlist=mode_allowlist,
        max_calls_per_turn=data.get("max_calls_per_turn", 0),
        timeout_seconds=data.get("timeout_seconds", 30),
        module_path=module_path,
        implementation_module=implementation_module,
        exportable=bool(data.get("exportable", False)),
        default_available=bool(data.get("default_available", False)),
    )


def _import_tool_module(module_path: str) -> Any:
    """Import and return one tool implementation module."""

    try:
        return importlib.import_module(module_path)
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed to import tool module '%s'", module_path, exc_info=True)
        return None
