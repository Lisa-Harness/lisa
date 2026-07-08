"""Disk-backed registry for internal operation metadata.

Modes are the only user-facing routing surface. Operations are reusable internal
behaviors discovered from Python modules with colocated ``.yml`` metadata.

Primary discovery roots:
1. shipped baseline operations under ``sdk_packs/core/operations/``
2. additional SDK or pack operation roots such as ``lisa_sdk/operations/``

Called by: host agent loops (direct action-mode execution)
           kernel.ipsar_fsm (mode binding validation)
"""

from __future__ import annotations

# pylint: disable=global-statement
import asyncio
import importlib
import importlib.metadata
import logging
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from kernel.content_roots import (
    discover_configured_content_roots,
    discover_installed_content_roots,
    resolve_content_subdir_roots,
)
from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    OperationResult,
    ToolExecutionTelemetry,
)
from kernel.openai_function_tools import build_strict_openai_function_tool
from kernel.operations.declarative import DeclarativeOperationModule
from kernel.telemetry import get_tracer, mark_span_failed, set_span_attributes

logger = logging.getLogger(__name__)

__all__ = [
    "execute_operation",
    "get_operation_catalog",
    "is_known_tool",
    "load_operation_catalog_merged",
    "register_operation_roots",
    "register_tool_catalog_callback",
]

_OPERATION_TRACER = get_tracer("kernel.operations")

_PRIMARY_OPERATIONS_ROOT = Path(__file__).parent.parent.parent / "app" / "operations"
_OPERATIONS_ROOT = _PRIMARY_OPERATIONS_ROOT
_PRIMARY_SDK_OPERATIONS_ROOT = Path(__file__).parent.parent.parent / "lisa_sdk" / "operations"
_OPERATION_METADATA_SUFFIX = ".yml"
_DECLARATIVE_OPERATION_FILENAME = "_operation.md"
_DECLARATIVE_OPERATION_SUFFIX = ".operation.md"
_FRONT_MATTER_DELIMITER = "---"

# Host-injected discovery roots. Set by register_operation_roots() at startup so
# the kernel package does not hard-code paths relative to its install location.
_injected_primary_root: Path | None = None
_injected_sdk_roots: list[Path] = []
_get_tool_catalog_fn: Callable[[], dict[str, Any]] | None = None


class OperationValidationError(ValueError):
    """Raised when an operation folder or definition violates registry invariants."""


class OperationExecutionError(RuntimeError):
    """Raised when an operation cannot be resolved or executed."""


def is_known_tool(tool_id: str) -> bool:
    """Return True if the given ID corresponds to a registered internal tool."""
    if _get_tool_catalog_fn is None:
        return False
    return tool_id in _get_tool_catalog_fn()


def register_tool_catalog_callback(get_tool_catalog_fn: Callable[[], dict[str, Any]]) -> None:
    """Inject the tool catalog callback used for operation validation and tool schemas."""
    global _get_tool_catalog_fn  # noqa: PLW0603
    _get_tool_catalog_fn = get_tool_catalog_fn
    validate_operation_catalog(_OPERATION_CATALOG)


def register_operation_roots(primary_root: Path, sdk_roots: list[Path]) -> None:
    """Inject operation discovery roots from the host app layer and rebuild the catalog.

    The kernel package itself has no built-in operation roots; the host app
    supplies the baseline operations path plus zero or more sdk_roots so that
    the kernel can be installed and run independently of the host directory tree.

    Rebuilds and re-validates ``_OPERATION_CATALOG`` immediately so that operations
    from entry-point-discovered roots (sdk_packs, lisa_sdk, etc.) are
    available before any mode or operation execution begins.
    """
    global _injected_primary_root, _injected_sdk_roots, _OPERATION_CATALOG  # noqa: PLW0603
    _injected_primary_root = primary_root
    _injected_sdk_roots = list(sdk_roots)
    _OPERATION_CATALOG = load_operation_catalog_merged(
        core_roots=[primary_root] if primary_root else _load_core_operations_roots(),
        sdk_roots=sdk_roots,
    )
    validate_operation_catalog(_OPERATION_CATALOG)


def _parse_operation_metadata(text: str, source: Path) -> dict[str, Any]:
    """Parse one operation metadata YAML file into a mapping."""
    try:
        metadata = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise OperationValidationError(f"Invalid YAML in '{source}': {exc}") from exc

    if not isinstance(metadata, dict):
        raise OperationValidationError(
            f"Operation metadata in '{source}' must be a YAML mapping, got {type(metadata).__name__}."
        )
    return metadata


def _parse_operation_markdown(text: str, source: Path) -> tuple[dict[str, Any], str]:
    """Parse one markdown-authored operation entrypoint into front matter and body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        raise OperationValidationError(
            f"Declarative operation '{source}' must start with YAML front matter."
        )

    try:
        close_idx = next(
            idx
            for idx, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONT_MATTER_DELIMITER
        )
    except StopIteration as exc:
        raise OperationValidationError(
            f"Declarative operation '{source}' is missing a closing front-matter delimiter."
        ) from exc

    metadata = _parse_operation_metadata("\n".join(lines[1:close_idx]), source)
    body = "\n".join(lines[close_idx + 1 :]).strip()
    return metadata, body


def _require_str_field(data: dict[str, Any], field: str, source: Path) -> str:
    """Extract one required string field from an operation definition front-matter dict."""
    value = data.get(field)
    if not value:
        raise OperationValidationError(
            f"Front-matter in '{source}' is missing required field '{field}'."
        )
    return str(value)


def _require_list_field(data: dict[str, Any], field: str, source: Path) -> list[str]:
    """Extract one required list[str] field from an operation definition front-matter dict."""
    if field not in data:
        raise OperationValidationError(
            f"Front-matter in '{source}' is missing required field '{field}'."
        )
    raw = data.get(field)
    if not isinstance(raw, list):
        raise OperationValidationError(
            f"Front-matter field '{field}' in '{source}' must be a list."
        )
    return [str(item) for item in raw]


def _parse_non_negative_int_field(
    data: dict[str, Any],
    field: str,
    source: Path,
    default: int,
) -> int:
    """Parse one optional non-negative integer front-matter field."""
    raw = data.get(field, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise OperationValidationError(
            f"Front-matter field '{field}' in '{source}' must be an integer."
        ) from exc
    if value < 0:
        raise OperationValidationError(
            f"Front-matter field '{field}' in '{source}' must be non-negative."
        )
    return value


def _parse_float_field(
    data: dict[str, Any],
    field: str,
    source: Path,
    default: float,
) -> float:
    """Parse one optional floating-point front-matter field."""
    raw = data.get(field, default)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise OperationValidationError(
            f"Front-matter field '{field}' in '{source}' must be a float."
        ) from exc


def _parse_string_mapping_field(
    data: dict[str, Any],
    field: str,
    source: Path,
) -> dict[str, str]:
    """Parse one optional string-to-string mapping from front matter."""
    raw = data.get(field) or {}
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise OperationValidationError(
            f"Front-matter field '{field}' in '{source}' must be a mapping."
        )
    parsed: dict[str, str] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if not key_text or not value_text:
            raise OperationValidationError(
                f"Front-matter field '{field}' in '{source}' must map non-empty strings."
            )
        parsed[key_text] = value_text
    return parsed


def _iter_operation_module_files(operations_dir: Path) -> list[Path]:
    """Return sorted Python operation modules under one operations root.

    Skips ``.py`` files that have a sibling ``<stem>.operation.md`` declarative
    entrypoint: those modules hold the Pydantic response-format contract for a
    prompt-authored operation, not a runtime ``execute(ctx)`` entrypoint, so
    they must not be treated as runtime operations requiring a ``.yml`` file.
    """
    runtime_modules: list[Path] = []
    for module_path in operations_dir.rglob("*.py"):
        if module_path.name == "__init__.py" or "__pycache__" in module_path.parts:
            continue
        declarative_sibling = module_path.with_suffix(_DECLARATIVE_OPERATION_SUFFIX)
        if declarative_sibling.exists():
            continue
        runtime_modules.append(module_path)
    return sorted(runtime_modules)


def _iter_operation_markdown_files(operations_dir: Path) -> list[Path]:
    """Return sorted markdown-authored operation entrypoints under one operations root."""
    return sorted(
        path
        for path in operations_dir.rglob("*.md")
        if path.name == _DECLARATIVE_OPERATION_FILENAME
        or path.name.endswith(_DECLARATIVE_OPERATION_SUFFIX)
        if "__pycache__" not in path.parts
    )


def _resolve_model_class(model_path: str, source: Path) -> type[BaseModel]:
    """Import and validate one Pydantic response-format model class."""
    module_path, separator, attr_name = model_path.rpartition(".")
    if not separator or not module_path or not attr_name:
        raise OperationValidationError(
            f"response_format_model_path in '{source}' must be 'package.module.ClassName'."
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise OperationValidationError(
            f"Failed to import response_format module '{module_path}' from '{source}': {exc}"
        ) from exc
    model_class = getattr(module, attr_name, None)
    if not isinstance(model_class, type) or not issubclass(model_class, BaseModel):
        raise OperationValidationError(
            f"response_format_model_path '{model_path}' in '{source}' must resolve to a Pydantic BaseModel subclass."
        )
    return model_class


def _package_parts_for_dir(directory: Path) -> tuple[str, ...]:
    """Walk up the directory tree collecting Python package names.

    Ascends as long as ``__init__.py`` exists in each level, building the
    fully-qualified dotted name for *directory*.  For example,
    ``sdk_packs/example`` returns ``("sdk_packs", "example")``.
    """
    parts: list[str] = []
    current = directory
    while (current / "__init__.py").exists():
        parts.append(current.name)
        current = current.parent
    return tuple(reversed(parts))


def _build_module_path_from_root(
    module_file: Path,
    *,
    base_root: Path,
    base_parts: tuple[str, ...],
) -> str:
    """Build one import path from a known package root and module file."""
    relative = module_file.relative_to(base_root).with_suffix("")
    return ".".join((*base_parts, *relative.parts))


def _base_parts_for_operations_root(
    operations_root: Path,
    *,
    default_parts: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the dotted import prefix for one discovered operations root."""

    package_dir = operations_root.parent
    if (package_dir / "__init__.py").exists():
        return (*_package_parts_for_dir(package_dir), "operations")
    return default_parts


def _resolve_module_path(
    module_file: Path,
    *,
    operations_root: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> str:
    """Resolve the import path for one discovered operation module."""
    explicit_module_path = str(metadata.get("module_path") or "").strip()
    if explicit_module_path:
        return explicit_module_path

    primary_root = _injected_primary_root or _PRIMARY_OPERATIONS_ROOT
    known_roots: list[tuple[Path, tuple[str, ...]]] = [
        (
            primary_root,
            _base_parts_for_operations_root(
                primary_root,
                default_parts=("app", "operations"),
            ),
        ),
    ]
    for sdk_root in _injected_sdk_roots or [_PRIMARY_SDK_OPERATIONS_ROOT]:
        # Derive the Python package base_parts from the root path.
        # Walk up collecting all __init__.py-backed package directories so that
        # nested namespace packages (e.g. sdk_packs/example) produce the
        # full dotted prefix sdk_packs.example rather than just example.
        package_dir = sdk_root.parent
        if (package_dir / "__init__.py").exists():
            pkg_parts = _package_parts_for_dir(package_dir)
            known_roots.append((sdk_root, (*pkg_parts, "operations")))
        else:
            known_roots.append((sdk_root, (sdk_root.name,)))
    for root, base_parts in known_roots:
        try:
            return _build_module_path_from_root(module_file, base_root=root, base_parts=base_parts)
        except ValueError:
            continue

    package_root = operations_root.parent
    if (package_root / "__init__.py").exists():
        pkg_parts = _package_parts_for_dir(package_root)
        return _build_module_path_from_root(
            module_file,
            base_root=package_root,
            base_parts=pkg_parts,
        )

    raise OperationValidationError(
        f"Could not infer module path for '{module_file}' from metadata '{metadata_path}'. "
        "Add 'module_path' to the YAML or make the operations root's parent a Python package."
    )


def _load_operation_module(*, module_path: str, source: Path) -> ModuleType:
    """Import and return one operation implementation module."""
    try:
        module = importlib.import_module(module_path)
        logger.info("Loaded operation module '%s' from '%s'", module_path, source)
        return module
    except Exception as exc:  # pylint: disable=broad-except
        raise OperationValidationError(
            f"Failed to import configured module '{module_path}' from '{source}': {exc}"
        ) from exc


def _load_markdown_operation_definition(markdown_path: Path) -> OperationDefinition:
    """Load one markdown-authored declarative operation definition."""
    metadata, body = _parse_operation_markdown(
        markdown_path.read_text(encoding="utf-8"), markdown_path
    )
    operation_id = _require_str_field(metadata, "operation_id", markdown_path)
    description = _require_str_field(metadata, "description", markdown_path)
    tool_ids = [str(tool_id) for tool_id in (metadata.get("tool_ids") or [])]
    mode_allowlist = _require_list_field(metadata, "mode_allowlist", markdown_path)
    preferred_model_id = (
        str(metadata["preferred_model_id"]) if "preferred_model_id" in metadata else None
    )
    response_format_model_path = str(metadata.get("response_format_model_path") or "").strip()
    response_output_text_field = (
        str(metadata.get("response_output_text_field") or "output_text").strip() or "output_text"
    )
    response_output_json_passthrough = _parse_string_mapping_field(
        metadata,
        "response_output_json_passthrough",
        markdown_path,
    )
    timeout_seconds = _parse_non_negative_int_field(
        metadata,
        "timeout_seconds",
        markdown_path,
        default=60,
    )
    max_calls_per_turn = _parse_non_negative_int_field(
        metadata,
        "max_calls_per_turn",
        markdown_path,
        default=3,
    )
    completion_max_tokens = _parse_non_negative_int_field(
        metadata,
        "completion_max_tokens",
        markdown_path,
        default=1500,
    )
    completion_temperature = _parse_float_field(
        metadata,
        "completion_temperature",
        markdown_path,
        default=0.2,
    )
    response_format_model = (
        _resolve_model_class(response_format_model_path, markdown_path)
        if response_format_model_path
        else None
    )
    if not body:
        raise OperationValidationError(f"Declarative operation '{markdown_path}' must have a body.")

    return OperationDefinition(
        operation_id=operation_id,
        description=description,
        execution_kind="prompt",
        response_format_model=response_format_model,
        response_format_model_path=response_format_model_path,
        response_output_text_field=response_output_text_field,
        response_output_json_passthrough=response_output_json_passthrough,
        preferred_model_id=preferred_model_id,
        tool_ids=tool_ids,
        mode_allowlist=mode_allowlist,
        timeout_seconds=timeout_seconds,
        max_calls_per_turn=max_calls_per_turn,
        completion_max_tokens=completion_max_tokens,
        completion_temperature=completion_temperature,
        instruction_markdown_path=str(markdown_path),
        instruction_body=body,
        implementation_module=DeclarativeOperationModule(
            operation_id=operation_id,
            instruction_markdown_path=str(markdown_path),
            instruction_body=body,
            max_calls_per_turn=max_calls_per_turn,
            response_format_model=response_format_model,
            response_output_text_field=response_output_text_field,
            response_output_json_passthrough=response_output_json_passthrough,
            completion_max_tokens=completion_max_tokens,
            completion_temperature=completion_temperature,
        ),
    )


def _build_tool_schemas(tool_ids: list[str]) -> list[dict[str, Any]]:
    """Return filtered tool schemas for one operation from the injected tool catalog."""
    if _get_tool_catalog_fn is None or not tool_ids:
        return []

    catalog = _get_tool_catalog_fn()
    schemas: list[dict[str, Any]] = []
    for tool_id in tool_ids:
        definition = catalog.get(tool_id)
        if definition is None:
            continue
        schemas.append(
            build_strict_openai_function_tool(
                name=tool_id,
                description=definition.description,
                parameters_schema=definition.parameters_schema,
            )
        )
    return schemas


def _wrap_declarative_tool_executor(
    definition: OperationDefinition,
    execute_tool_fn: Any,
) -> Any:
    """Return a tool executor that only allows declared tools for prompt operations."""
    if definition.execution_kind != "prompt" or execute_tool_fn is None:
        return execute_tool_fn

    allowed_tool_ids = set(definition.tool_ids)

    async def _execute_allowed_tool(tool_id: str, ctx: Any) -> Any:
        if tool_id not in allowed_tool_ids:
            raise OperationExecutionError(
                f"Declarative operation '{definition.operation_id}' cannot call undeclared tool '{tool_id}'."
            )
        return await execute_tool_fn(tool_id, ctx)

    return _execute_allowed_tool


def load_operation_catalog_from_disk(
    operations_dir: Path | None = None,
) -> dict[str, OperationDefinition]:
    """Discover and load all internal operations from one operations module tree."""
    root = operations_dir or _OPERATIONS_ROOT
    if not root.exists():
        return {}

    catalog: dict[str, OperationDefinition] = {}
    for markdown_path in _iter_operation_markdown_files(root):
        definition = _load_markdown_operation_definition(markdown_path)
        if definition.operation_id in catalog:
            raise OperationValidationError(
                f"Duplicate operation_id '{definition.operation_id}' discovered in '{markdown_path}'."
            )
        catalog[definition.operation_id] = definition

    for module_file in _iter_operation_module_files(root):
        metadata_path = module_file.with_suffix(_OPERATION_METADATA_SUFFIX)
        if not metadata_path.exists():
            logger.warning(
                "Skipping operation module '%s': missing '%s'", module_file, metadata_path.name
            )
            continue

        metadata = _parse_operation_metadata(
            metadata_path.read_text(encoding="utf-8"), metadata_path
        )
        operation_id = _require_str_field(metadata, "operation_id", metadata_path)
        if operation_id in catalog:
            if catalog[operation_id].execution_kind == "prompt":
                logger.info(
                    "Skipping legacy Python operation '%s' because prompt entrypoint '%s' takes precedence.",
                    module_file,
                    catalog[operation_id].instruction_markdown_path,
                )
                continue
            raise OperationValidationError(
                f"Duplicate operation_id '{operation_id}' discovered in '{metadata_path}'."
            )

        description = _require_str_field(metadata, "description", metadata_path)
        tool_ids = [str(tool_id) for tool_id in (metadata.get("tool_ids") or [])]
        mode_allowlist = _require_list_field(metadata, "mode_allowlist", metadata_path)
        preferred_model_id = (
            str(metadata["preferred_model_id"]) if "preferred_model_id" in metadata else None
        )
        timeout_seconds = _parse_non_negative_int_field(
            metadata,
            "timeout_seconds",
            metadata_path,
            default=60,
        )
        max_calls_per_turn = _parse_non_negative_int_field(
            metadata,
            "max_calls_per_turn",
            metadata_path,
            default=3,
        )
        module_path = _resolve_module_path(
            module_file,
            operations_root=root,
            metadata=metadata,
            metadata_path=metadata_path,
        )

        impl_module = _load_operation_module(module_path=module_path, source=metadata_path)
        execute_fn = getattr(impl_module, "execute", None)
        if execute_fn is None:
            raise OperationValidationError(
                f"Operation '{operation_id}' implementation is missing execute(ctx)."
            )
        parameters_schema = getattr(impl_module, "Parameters", None) if impl_module else None

        catalog[operation_id] = OperationDefinition(
            operation_id=operation_id,
            description=description,
            execution_kind="runtime",
            preferred_model_id=preferred_model_id,
            tool_ids=tool_ids,
            mode_allowlist=mode_allowlist,
            timeout_seconds=timeout_seconds,
            max_calls_per_turn=max_calls_per_turn,
            parameters_schema=parameters_schema,
            implementation_module=impl_module,
        )
    return catalog


def validate_operation_catalog(catalog: dict[str, OperationDefinition]) -> None:
    """Validate the operation catalog for basic invariants."""
    tool_catalog = _get_tool_catalog_fn() if _get_tool_catalog_fn is not None else None
    for operation_id, definition in catalog.items():
        if operation_id != definition.operation_id:
            raise OperationValidationError(
                "Operation registry key "
                f"'{operation_id}' does not match operation_id '{definition.operation_id}'."
            )
        if not definition.description.strip():
            raise OperationValidationError(
                f"Operation '{operation_id}' must have a non-empty description."
            )
        if definition.implementation_module is None:
            raise OperationValidationError(
                f"Operation '{operation_id}' has no implementation module."
            )
        if getattr(definition.implementation_module, "execute", None) is None:
            raise OperationValidationError(
                f"Operation '{operation_id}' implementation is missing execute(ctx)."
            )
        if definition.max_calls_per_turn < 0:
            raise OperationValidationError(
                "Operation "
                f"'{operation_id}' has invalid max_calls_per_turn={definition.max_calls_per_turn}."
            )
        if definition.timeout_seconds < 0:
            raise OperationValidationError(
                f"Operation '{operation_id}' has invalid timeout_seconds={definition.timeout_seconds}."
            )
        if definition.execution_kind == "prompt":
            if not definition.instruction_markdown_path.strip():
                raise OperationValidationError(
                    f"Declarative operation '{operation_id}' is missing instruction_markdown_path."
                )
            if not definition.instruction_body.strip():
                raise OperationValidationError(
                    f"Declarative operation '{operation_id}' must have a non-empty markdown body."
                )
            if definition.completion_max_tokens < 1:
                raise OperationValidationError(
                    f"Declarative operation '{operation_id}' has invalid completion_max_tokens={definition.completion_max_tokens}."
                )
        if tool_catalog is not None:
            for tool_id in definition.tool_ids:
                if tool_id not in tool_catalog:
                    raise OperationValidationError(
                        f"Operation '{operation_id}' references unknown tool '{tool_id}'."
                    )


def load_operation_catalog_merged(
    core_root: Path | None = None,
    core_roots: list[Path] | None = None,
    sdk_roots: list[Path] | None = None,
) -> dict[str, OperationDefinition]:
    """Discover operations from core roots and zero or more SDK roots.

    Core roots are merged in order; if two roots define the same operation_id, the
    *last* one wins. SDK roots are then merged in order with the same rule.

    Args:
        core_root: One core operations directory for backward compatibility with
            callers that only pass a single path.
        core_roots: Ordered list of core operation-definition roots.
        sdk_roots: Additional operation roots contributed by lisa_sdk or
            third-party operation packages.

    Returns:
        Merged dict mapping operation_id -> OperationDefinition, ready for use.
    """
    merged: dict[str, OperationDefinition] = {}
    if core_roots is not None:
        for root in core_roots:
            merged.update(load_operation_catalog_from_disk(root))
    else:
        merged.update(load_operation_catalog_from_disk(core_root))
    for extra_root in sdk_roots or []:
        merged.update(load_operation_catalog_from_disk(extra_root))
    return merged


def _load_core_operations_roots() -> list[Path]:
    """Collect core operation module roots in ascending precedence order."""
    roots: list[Path] = []
    if _PRIMARY_OPERATIONS_ROOT.is_dir():
        roots.append(_PRIMARY_OPERATIONS_ROOT)
    return roots


def _load_sdk_operations_roots() -> list[Path]:
    """Collect SDK operation module roots via entry-points, in-repo lisa_sdk, and env config.

    Discovery order (ascending precedence — later roots override earlier ones):
    1. ``lisa.extensions.content_roots`` entry-points (installed packages).
    2. In-repo ``lisa_sdk/operations`` (convenience for local dev/test).
    3. ``SDK_CONTENT_ROOTS`` env var (explicit overrides).
    """
    roots = resolve_content_subdir_roots(discover_installed_content_roots(), "operations")
    # 2. Auto-include the in-repo lisa_sdk operations directory.
    if _PRIMARY_SDK_OPERATIONS_ROOT.is_dir() and _PRIMARY_SDK_OPERATIONS_ROOT not in roots:
        roots.append(_PRIMARY_SDK_OPERATIONS_ROOT)
    # 3. Env-configured overrides.
    for candidate in resolve_content_subdir_roots(
        discover_configured_content_roots(),
        "operations",
    ):
        if candidate not in roots:
            roots.append(candidate)
    return roots


_OPERATION_CATALOG = load_operation_catalog_merged(
    core_roots=_load_core_operations_roots(),
    sdk_roots=_load_sdk_operations_roots(),
)
validate_operation_catalog(_OPERATION_CATALOG)


def get_operation_catalog() -> dict[str, OperationDefinition]:
    """Return the validated internal operation catalog loaded at startup."""
    return _OPERATION_CATALOG


def _is_operation_allowed_for_mode(
    *,
    operation_id: str,
    definition: OperationDefinition,
    mode_id: str,
) -> bool:
    """Return whether one operation can execute for the given mode.

    Permissions are sourced from two places:
    1) Per-operation ``mode_allowlist`` in colocated ``*/operations/**/*.yml`` files.
    2) Mode-owned operation bindings declared in loaded ``*/modes/*/_mode.md`` files.

    Mode-owned bindings are treated as authoritative for mode execution so
    stale operation-level allowlists do not break chat/action/workflow routing.
    """
    if not mode_id:
        return True

    if definition.mode_allowlist and mode_id in definition.mode_allowlist:
        return True

    # pylint: disable=import-outside-toplevel
    from kernel.ipsar_fsm import get_workflow_catalog

    mode = get_workflow_catalog().get(mode_id)
    if mode:
        mode_bound_operation_ids = set(
            mode.entry_operation_ids
            + mode.pre_turn_operation_ids
            + mode.post_turn_operation_ids
            + mode.allowed_operation_ids
        )
        if operation_id in mode_bound_operation_ids:
            return True

    if definition.mode_allowlist:
        return False
    return True


async def execute_operation(
    operation_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute one internal operation by ID and return its structured result."""
    definition = _OPERATION_CATALOG.get(operation_id)
    if definition is None:
        raise OperationExecutionError(f"Unknown operation '{operation_id}'.")
    if not _is_operation_allowed_for_mode(
        operation_id=operation_id,
        definition=definition,
        mode_id=ctx.mode_id,
    ):
        raise OperationExecutionError(
            f"Operation '{operation_id}' is not allowed for mode '{ctx.mode_id}'."
        )

    # Inject preferred model from definition
    ctx.preferred_model_id = definition.preferred_model_id
    ctx.tool_schemas = _build_tool_schemas(definition.tool_ids)
    ctx.execute_tool_fn = _wrap_declarative_tool_executor(definition, ctx.execute_tool_fn)
    module = definition.implementation_module
    if module is None:
        raise OperationExecutionError(f"Operation '{operation_id}' has no implementation module.")

    execute_fn = getattr(module, "execute", None)
    if execute_fn is None:
        raise OperationExecutionError(f"Operation '{operation_id}' is missing async execute(ctx).")

    with _OPERATION_TRACER.start_as_current_span("operation.execute") as span:
        set_span_attributes(
            span,
            {
                "lisa.target_type": "operation",
                "lisa.target_id": operation_id,
                "lisa.timeout_seconds": definition.timeout_seconds,
                "lisa.mode_id": ctx.mode_id,
            },
        )
        t0 = time.perf_counter()
        status: Literal["success", "error", "timeout"] = "success"
        err_msg = None
        try:
            if definition.timeout_seconds > 0:
                result = await asyncio.wait_for(execute_fn(ctx), timeout=definition.timeout_seconds)
            else:
                result = await execute_fn(ctx)
        except TimeoutError as exc:
            logger.error(
                "operation '%s' timed out after %ss",
                operation_id,
                definition.timeout_seconds,
            )
            status = "timeout"
            err_msg = f"Timed out after {definition.timeout_seconds}s"
            mark_span_failed(span, exc)
            raise OperationExecutionError(f"Operation '{operation_id}' timed out") from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("operation '%s' failed: %s", operation_id, exc)
            status = "error"
            err_msg = f"{type(exc).__name__}: {exc}"
            mark_span_failed(span, exc)
            raise OperationExecutionError(f"Operation '{operation_id}' failed: {err_msg}") from exc
        finally:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            set_span_attributes(
                span,
                {
                    "lisa.status": status,
                    "lisa.duration_ms": duration_ms,
                    "lisa.error_message": err_msg,
                },
            )
            if ctx.event_bus:
                event = ToolExecutionTelemetry(
                    target_id=operation_id,
                    target_type="operation",
                    status=status,
                    duration_ms=duration_ms,
                    error_message=err_msg,
                )
                asyncio.create_task(ctx.event_bus.publish(event))

    if isinstance(result, OperationResult):
        return result
    if isinstance(result, str):
        return OperationResult(output_text=result)
    raise OperationExecutionError(
        f"Operation '{operation_id}' returned unsupported result type {type(result).__name__}."
    )
