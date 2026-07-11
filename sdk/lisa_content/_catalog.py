"""Mode and operation catalog loading for the ``lisa-content`` MCP server.

This module does all filesystem scanning and front-matter parsing for content
catalogs (workflow modes and internal operations) so the kernel never needs
to touch the filesystem directly. The kernel calls ``list_modes`` and
``list_operations`` MCP tools; the server runs the logic here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from sdk.lisa_content._front_matter import FrontMatterError, parse_front_matter

logger = logging.getLogger(__name__)

__all__ = [
    "build_mode_catalog_json",
    "build_operation_catalog_json",
    "build_tool_catalog_json",
    "resolve_mode_roots",
    "resolve_operation_roots",
    "resolve_tool_roots",
]

# ── Path helpers ────────────────────────────────────────────────────────────


def resolve_mode_roots(all_roots: list[Path]) -> list[Path]:
    """Return existing ``modes/`` subdirectories under all content roots."""

    return _resolve_subdir_roots(all_roots, "modes")


def resolve_tool_roots(all_roots: list[Path]) -> list[Path]:
    """Return existing ``tools/`` subdirectories under all content roots."""

    return _resolve_subdir_roots(all_roots, "tools")


def resolve_operation_roots(all_roots: list[Path]) -> list[Path]:
    """Return existing ``operations/`` subdirectories under all content roots."""

    return _resolve_subdir_roots(all_roots, "operations")


def _resolve_subdir_roots(roots: list[Path], subdir: str) -> list[Path]:
    normalized = subdir.strip("/\\")
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidate = root / normalized if normalized else root
        if not candidate.is_dir():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


# ── Mode catalog ────────────────────────────────────────────────────────────

_MODE_CONFIG_FILENAME = "_mode.md"
_MODE_PROVIDER_PREFERENCES_ENV = "LISA_MODE_PROVIDER_PREFERENCES"
_CONTENT_PROVIDER_PREFERENCES_ENV = "LISA_CONTENT_PROVIDER_PREFERENCES"


def build_mode_catalog_json(mode_roots: list[Path]) -> str:
    """Scan mode roots and return a JSON-serializable mode catalog.

    Returns a JSON string encoding ``dict[str, dict]`` where each value is a
    mode config dict with nested ``stages``. The kernel deserializes this into
    ``WorkflowModeConfig`` objects.
    """
    merged: dict[str, dict[str, Any]] = {}
    preferences = _load_content_provider_preferences()
    mode_preferences = preferences.get("modes", {})
    for root in mode_roots:
        if not root.is_dir():
            continue
        inferred_provider_id = _infer_content_provider_id(root)
        for mode_dir in sorted(root.iterdir()):
            if not mode_dir.is_dir():
                continue
            mode_dict = _load_one_mode(mode_dir, inferred_provider_id=inferred_provider_id)
            if mode_dict is not None:
                mode_id = str(mode_dict.get("mode_id") or "")
                current = merged.get(mode_id)
                if current is None:
                    merged[mode_id] = mode_dict
                else:
                    merged[mode_id] = _select_catalog_candidate(
                        current=current,
                        candidate=mode_dict,
                        provider_field="mode_provider_id",
                        preferred_providers=mode_preferences.get(mode_id, []),
                    )
    return json.dumps(merged)


def _load_one_mode(
    mode_dir: Path,
    *,
    inferred_provider_id: str,
) -> dict[str, Any] | None:
    """Load one mode directory into a JSON-serializable dict."""

    mode_config_path = mode_dir / _MODE_CONFIG_FILENAME
    if not mode_config_path.is_file():
        logger.warning("Skipping mode folder '%s': missing _mode.md", mode_dir.name)
        return None

    mode_text = mode_config_path.read_text(encoding="utf-8")
    try:
        front_matter, body = parse_front_matter(mode_text)
    except FrontMatterError as exc:
        logger.warning("Skipping mode '%s': %s", mode_dir.name, exc)
        return None

    mode_id = front_matter.get("mode_id")
    if not mode_id or not isinstance(mode_id, str):
        return None

    stages: dict[str, dict[str, Any]] = {}
    for stage_path in sorted(mode_dir.glob("*.md")):
        if stage_path.name.startswith("_"):
            continue
        stage_dict = _load_one_stage(stage_path, mode_kind=front_matter.get("mode_kind", "workflow"))
        if stage_dict is not None:
            stages[stage_dict["stage_id"]] = stage_dict

    # Resolve mode-relative file references (workshop planner/synthesis prompts).
    workshop_planner_prompt_file = _resolve_mode_relative_file(
        mode_dir,
        str(front_matter.get("workshop_planner_prompt_file", "") or ""),
    )
    workshop_synthesis_prompt_file = _resolve_mode_relative_file(
        mode_dir,
        str(front_matter.get("workshop_synthesis_prompt_file", "") or ""),
    )

    # Parse workshop panels, resolving per-panel query_prompt_file refs.
    workshop_panels: list[dict[str, Any]] = []
    for panel in front_matter.get("workshop_panels") or []:
        if not isinstance(panel, dict):
            continue
        panel_dict = {
            "panel_id": str(panel.get("panel_id", "") or "").strip(),
            "title": str(panel.get("title", "") or "").strip(),
            "mode_id": str(panel.get("mode_id", "") or ""),
            "model_alias": str(panel.get("model_alias", "") or "").strip(),
            "query_prefix": str(panel.get("query_prefix", "") or "").strip(),
            "query_prompt_file": _resolve_mode_relative_file(
                mode_dir,
                str(panel.get("query_prompt_file", "") or ""),
            ),
        }
        workshop_panels.append(panel_dict)

    mode_provider_id = str(front_matter.get("mode_provider_id", "") or "").strip()
    if not mode_provider_id:
        mode_provider_id = inferred_provider_id

    return {
        "mode_id": mode_id,
        "mode_provider_id": mode_provider_id,
        "start_stage_id": str(front_matter.get("start_stage_id", "") or ""),
        "stages": stages,
        "mode_kind": str(front_matter.get("mode_kind", "workflow") or "workflow"),
        "artifact_type": str(front_matter.get("artifact_type", "workflow_summary") or "workflow_summary"),
        "description": str(front_matter.get("description", "") or ""),
        "instruction_body": body.strip(),
        "mode_markdown_path": str(mode_config_path),
        "default_model": (str(front_matter.get("default_model", "") or "") or None),
        "frontdoor_routable": bool(front_matter.get("frontdoor_routable", False)),
        "frontdoor_priority": int(front_matter.get("frontdoor_priority", 0) or 0),
        "frontdoor_fast_path_patterns": [str(p) for p in (front_matter.get("frontdoor_fast_path_patterns") or [])],
        "frontdoor_route_hints": [str(h) for h in (front_matter.get("frontdoor_route_hints") or [])],
        "frontdoor_examples": [str(e) for e in (front_matter.get("frontdoor_examples") or [])],
        "command_pattern": str(front_matter.get("command_pattern", "") or ""),
        "require_query": bool(front_matter.get("require_query", False)),
        "entry_operation_ids": [str(o) for o in (front_matter.get("entry_operation_ids") or [])],
        "pre_turn_operation_ids": [str(o) for o in (front_matter.get("pre_turn_operation_ids") or [])],
        "post_turn_operation_ids": [str(o) for o in (front_matter.get("post_turn_operation_ids") or [])],
        "allowed_operation_ids": [str(o) for o in (front_matter.get("allowed_operation_ids") or [])],
        "allowed_tool_ids": [str(t) for t in (front_matter.get("allowed_tool_ids") or [])],
        "tools_authoritative": bool(front_matter.get("tools_authoritative", False)),
        "workshop_panels": workshop_panels,
        "workshop_budget": front_matter.get("workshop_budget") or {},
        "workshop_planner_prompt_file": workshop_planner_prompt_file,
        "workshop_synthesis_prompt_file": workshop_synthesis_prompt_file,
        "workshop_rendering": front_matter.get("workshop_rendering") or {},
        "default_intent_bridges": _parse_intent_bridges(front_matter.get("default_intent_bridges") or []),
    }


def _load_content_provider_preferences() -> dict[str, dict[str, list[str]]]:
    """Parse host-supplied provider preferences from env.

    Preferred shape:
    {
      "modes": {"default": ["lisa-pack-learning", "lisa-pack-core"]},
      "operations": {"web_search": ["lisa-pack-search"]},
      "tools": {"web_search": ["lisa-pack-brave-search", "lisa-pack-search"]}
    }

    Legacy compatibility:
    - ``LISA_MODE_PROVIDER_PREFERENCES`` may still provide mode-only mappings.
    """

    mode_preferences = _load_provider_preference_mapping(
        _MODE_PROVIDER_PREFERENCES_ENV,
        context="mode-provider preferences",
    )
    content_preferences = _load_provider_preference_mapping(
        _CONTENT_PROVIDER_PREFERENCES_ENV,
        context="content-provider preferences",
    )

    modes = _parse_preference_mapping(content_preferences.get("modes") or {})
    operations = _parse_preference_mapping(content_preferences.get("operations") or {})
    tools = _parse_preference_mapping(content_preferences.get("tools") or {})

    # Backward compatibility for hosts still setting the legacy env var.
    if not modes and mode_preferences:
        modes = mode_preferences

    return {
        "modes": modes,
        "operations": operations,
        "tools": tools,
    }


def _load_provider_preference_mapping(env_var: str, *, context: str) -> dict[str, Any]:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in %s for %s", env_var, context)
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _parse_preference_mapping(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}

    preferences: dict[str, list[str]] = {}
    for key_raw, providers_raw in raw.items():
        if not isinstance(key_raw, str):
            continue
        key = key_raw.strip().lower()
        if not key:
            continue

        providers: list[str]
        if isinstance(providers_raw, str):
            normalized = _normalize_provider_id(providers_raw)
            providers = [normalized] if normalized else []
        elif isinstance(providers_raw, list):
            providers = [
                _normalize_provider_id(item)
                for item in providers_raw
                if isinstance(item, str) and _normalize_provider_id(item)
            ]
        else:
            providers = []

        if providers:
            preferences[key] = providers
    return preferences


def _infer_content_provider_id(root: Path) -> str:
    """Infer a stable provider id from a content sub-root path."""

    resolved = root.resolve()
    raw_candidates = [
        resolved.parent.name,
        resolved.parent.parent.name,
        resolved.name,
    ]
    for raw in raw_candidates:
        normalized = _normalize_provider_id(raw)
        if normalized and normalized not in {"modes", "operations", "tools", "content"}:
            return normalized
    return ""


def _normalize_provider_id(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return value


def _provider_rank(provider_id: str, preferred_providers: list[str]) -> int:
    if not provider_id:
        return len(preferred_providers) + 1
    try:
        return preferred_providers.index(provider_id)
    except ValueError:
        return len(preferred_providers) + 1


def _select_catalog_candidate(
    *,
    current: dict[str, Any],
    candidate: dict[str, Any],
    provider_field: str,
    preferred_providers: list[str],
) -> dict[str, Any]:
    """Select the winning entry for duplicate id collisions."""

    current_provider = _normalize_provider_id(str(current.get(provider_field) or ""))
    candidate_provider = _normalize_provider_id(str(candidate.get(provider_field) or ""))
    current_rank = _provider_rank(current_provider, preferred_providers)
    candidate_rank = _provider_rank(candidate_provider, preferred_providers)

    if candidate_rank < current_rank:
        return candidate
    if current_rank < candidate_rank:
        return current
    # Tie-breaker preserves prior behavior: later-discovered mode wins.
    return candidate


def _load_one_stage(stage_path: Path, *, mode_kind: str) -> dict[str, Any] | None:
    """Load one stage markdown file into a JSON-serializable dict.

    Returns None if the file has no ``stage_id`` in its front-matter (it's an
    auxiliary asset, not a stage definition). For ``chat``/``action`` modes,
    missing ``stage_id`` is silently skipped; for workflow modes, we still skip
    but log it.
    """

    stage_text = stage_path.read_text(encoding="utf-8")
    try:
        front_matter, body = parse_front_matter(stage_text)
    except FrontMatterError as exc:
        logger.warning("Skipping stage file '%s': %s", stage_path, exc)
        return None

    if "stage_id" not in front_matter:
        if mode_kind in {"chat", "action"}:
            return None
        logger.warning("Front-matter in '%s' is missing 'stage_id'", stage_path)
        return None

    return {
        "stage_id": str(front_matter["stage_id"]),
        "instruction_markdown_path": str(stage_path),
        "instruction_body": body,
        "description": str(front_matter.get("description", "") or ""),
        "default_model": (str(front_matter.get("default_model", "") or "") or None),
        "required_connectors": [str(c) for c in (front_matter.get("required_connectors") or [])],
        "allowed_next_stages": [str(s) for s in (front_matter.get("allowed_next_stages") or [])],
        "allow_user_questions": bool(front_matter.get("allow_user_questions", True)),
        "terminal": bool(front_matter.get("terminal", False)),
    }


def _resolve_mode_relative_file(mode_dir: Path, authored_path: str) -> str:
    """Validate and normalize a mode-relative file reference.

    Returns an empty string if the path is empty. Returns the POSIX-relative
    path if the file exists and stays within the mode directory. Raises
    nothing — invalid paths are logged and returned as-is for the kernel to
    validate.
    """

    raw = authored_path.strip()
    if not raw:
        return ""

    candidate = Path(raw)
    if candidate.is_absolute():
        logger.warning("Mode-relative file path '%s' is absolute; returning as-is", raw)
        return raw

    mode_root = mode_dir.resolve()
    resolved = (mode_root / candidate).resolve()
    try:
        if not resolved.is_relative_to(mode_root):
            logger.warning("Mode-relative file '%s' escapes mode dir", raw)
            return raw
    except AttributeError:
        # Python < 3.9 compat — is_relative_to was added in 3.9
        pass

    if not resolved.is_file():
        logger.warning("Mode-relative file '%s' does not exist", raw)
        return raw

    return resolved.relative_to(mode_root).as_posix()


def _parse_intent_bridges(bridges_raw: Any) -> list[dict[str, Any]]:
    """Parse default_intent_bridges from front-matter into serializable dicts."""

    if not isinstance(bridges_raw, list):
        return []
    result: list[dict[str, Any]] = []
    for bridge in bridges_raw:
        if not isinstance(bridge, dict):
            continue
        result.append({
            "target_mode_id": str(bridge.get("target_mode_id", "") or ""),
            "matcher_patterns": [str(p) for p in (bridge.get("matcher_patterns") or [])],
            "query_extraction_strategy": str(bridge.get("query_extraction_strategy", "passthrough") or "passthrough"),
            "prerequisites": [str(p) for p in (bridge.get("prerequisites") or [])],
            "recent_handoff_source_modes": [str(m) for m in (bridge.get("recent_handoff_source_modes") or [])],
            "recent_handoff_required_artifact_keys": [str(k) for k in (bridge.get("recent_handoff_required_artifact_keys") or [])],
            "recent_handoff_context_key": str(bridge.get("recent_handoff_context_key", "") or ""),
            "response_appendix": str(bridge.get("response_appendix", "") or ""),
        })
    return result


# ── Operation catalog ────────────────────────────────────────────────────────

_OPERATION_METADATA_SUFFIX = ".yml"
_DECLARATIVE_OPERATION_FILENAME = "_operation.md"
_DECLARATIVE_OPERATION_SUFFIX = ".operation.md"


def build_operation_catalog_json(operation_roots: list[Path]) -> str:
    """Scan operation roots and return a JSON-serializable operation catalog.

    Returns a JSON string encoding ``dict[str, dict]`` where each value is an
    operation definition dict. The kernel deserializes this into
    ``OperationDefinition`` objects, importing the implementation module
    locally (using the ``module_path`` field).
    """
    merged: dict[str, dict[str, Any]] = {}
    preferences = _load_content_provider_preferences()
    operation_preferences = preferences.get("operations", {})
    for root in operation_roots:
        if not root.is_dir():
            continue
        inferred_provider_id = _infer_content_provider_id(root)
        for op_id, op_dict in _load_operations_from_root(
            root,
            inferred_provider_id=inferred_provider_id,
        ).items():
            current = merged.get(op_id)
            if current is None:
                merged[op_id] = op_dict
            else:
                merged[op_id] = _select_catalog_candidate(
                    current=current,
                    candidate=op_dict,
                    provider_field="operation_provider_id",
                    preferred_providers=operation_preferences.get(op_id, []),
                )
    return json.dumps(merged)


def _load_operations_from_root(
    root: Path,
    *,
    inferred_provider_id: str,
) -> dict[str, dict[str, Any]]:
    """Load all operations from one root directory."""

    catalog: dict[str, dict[str, Any]] = {}

    # 1. Markdown-authored (declarative/prompt) operations
    for md_path in _iter_operation_markdown_files(root):
        op_dict = _load_markdown_operation(md_path)
        if op_dict is not None:
            op_dict.setdefault("operation_provider_id", inferred_provider_id)
            catalog[op_dict["operation_id"]] = op_dict

    # 2. Python runtime operations with .yml metadata
    for py_path in _iter_operation_module_files(root):
        yml_path = py_path.with_suffix(_OPERATION_METADATA_SUFFIX)
        if not yml_path.is_file():
            logger.warning("Skipping operation module '%s': missing '%s'", py_path, yml_path.name)
            continue
        op_dict = _load_runtime_operation(py_path, yml_path, root)
        if op_dict is not None:
            op_dict.setdefault("operation_provider_id", inferred_provider_id)
            op_id = op_dict["operation_id"]
            if op_id in catalog and catalog[op_id].get("execution_kind") == "prompt":
                logger.info(
                    "Skipping legacy Python operation '%s' because prompt entrypoint takes precedence.",
                    py_path,
                )
                continue
            catalog[op_id] = op_dict

    return catalog


def _iter_operation_module_files(operations_dir: Path) -> list[Path]:
    """Return sorted Python operation modules, skipping declarative siblings."""

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
    """Return sorted markdown-authored operation entrypoints."""

    return sorted(
        path
        for path in operations_dir.rglob("*.md")
        if path.name == _DECLARATIVE_OPERATION_FILENAME
        or path.name.endswith(_DECLARATIVE_OPERATION_SUFFIX)
        if "__pycache__" not in path.parts
    )


def _load_markdown_operation(md_path: Path) -> dict[str, Any] | None:
    """Load one markdown-authored declarative operation definition."""

    text = md_path.read_text(encoding="utf-8")
    try:
        metadata, body = parse_front_matter(text)
    except FrontMatterError as exc:
        logger.warning("Skipping operation '%s': %s", md_path, exc)
        return None

    operation_id = metadata.get("operation_id")
    if not operation_id or not isinstance(operation_id, str):
        return None

    return {
        "operation_id": operation_id,
        "description": str(metadata.get("description", "") or ""),
        "execution_kind": "prompt",
        "tool_ids": [str(t) for t in (metadata.get("tool_ids") or [])],
        "mode_allowlist": [str(m) for m in (metadata.get("mode_allowlist") or [])],
        "preferred_model_id": (str(metadata["preferred_model_id"]) if "preferred_model_id" in metadata else None),
        "response_format_model_path": str(metadata.get("response_format_model_path", "") or "").strip(),
        "response_output_text_field": str(metadata.get("response_output_text_field", "output_text") or "output_text").strip() or "output_text",
        "response_output_json_passthrough": _parse_string_mapping(metadata.get("response_output_json_passthrough")),
        "timeout_seconds": int(metadata.get("timeout_seconds", 60) or 60),
        "max_calls_per_turn": int(metadata.get("max_calls_per_turn", 3) or 3),
        "completion_max_tokens": int(metadata.get("completion_max_tokens", 1500) or 1500),
        "completion_temperature": float(metadata.get("completion_temperature", 0.2) or 0.2),
        "instruction_markdown_path": str(md_path),
        "instruction_body": body,
        "module_path": "",
    }


def _load_runtime_operation(
    py_path: Path,
    yml_path: Path,
    root: Path,
) -> dict[str, Any] | None:
    """Load one Python runtime operation from its .yml metadata."""

    try:
        metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in '%s': %s", yml_path, exc)
        return None

    if not isinstance(metadata, dict):
        return None

    operation_id = metadata.get("operation_id")
    if not operation_id or not isinstance(operation_id, str):
        return None

    module_path = str(metadata.get("module_path", "") or "").strip()
    if not module_path:
        module_path = _infer_module_path(py_path, root)

    return {
        "operation_id": operation_id,
        "description": str(metadata.get("description", "") or ""),
        "execution_kind": "runtime",
        "tool_ids": [str(t) for t in (metadata.get("tool_ids") or [])],
        "mode_allowlist": [str(m) for m in (metadata.get("mode_allowlist") or [])],
        "preferred_model_id": (str(metadata["preferred_model_id"]) if "preferred_model_id" in metadata else None),
        "timeout_seconds": int(metadata.get("timeout_seconds", 60) or 60),
        "max_calls_per_turn": int(metadata.get("max_calls_per_turn", 3) or 3),
        "instruction_markdown_path": "",
        "instruction_body": "",
        "module_path": module_path,
    }


def _infer_module_path(py_path: Path, root: Path) -> str:
    """Infer a dotted Python module path from the file path relative to root.

    The ``operations`` directory name is included in the path so that
    ``sdk_packs/core/operations/research/workshop_run.py`` resolves to
    ``sdk_packs.core.operations.research.workshop_run``.
    """

    relative = py_path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    # Walk up from root's parent to find the package base
    package_dir = root.parent
    pkg_parts: list[str] = []
    while (package_dir / "__init__.py").is_file():
        pkg_parts.insert(0, package_dir.name)
        package_dir = package_dir.parent
    # Include the operations directory name itself
    if root.name:
        pkg_parts.append(root.name)
    return ".".join((*pkg_parts, *parts)) if pkg_parts else ".".join(parts)


def _parse_string_mapping(raw: Any) -> dict[str, str]:
    """Parse a string-to-string mapping from metadata."""

    if raw is None or not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        result[str(key)] = str(value)
    return result


# ── Tool catalog ─────────────────────────────────────────────────────────────

_TOOL_METADATA_SUFFIX = ".tool.yml"


def build_tool_catalog_json(tool_roots: list[Path]) -> str:
    """Scan tool roots and return a JSON-serializable internal tool catalog.

    Returns a JSON string encoding ``dict[str, dict]`` where each value is a
    tool definition dict. The kernel deserializes this into ``ToolDefinition``
    objects, importing the implementation module locally using the
    ``module_path`` field (same pattern as operations).

    Tools are runtime-only (``.py`` + ``.tool.yml`` pairs). Declarative tools
    are not supported in this iteration.
    """
    merged: dict[str, dict[str, Any]] = {}
    preferences = _load_content_provider_preferences()
    tool_preferences = preferences.get("tools", {})
    for root in tool_roots:
        if not root.is_dir():
            continue
        inferred_provider_id = _infer_content_provider_id(root)
        for tool_id, tool_dict in _load_tools_from_root(
            root,
            inferred_provider_id=inferred_provider_id,
        ).items():
            current = merged.get(tool_id)
            if current is None:
                merged[tool_id] = tool_dict
            else:
                merged[tool_id] = _select_catalog_candidate(
                    current=current,
                    candidate=tool_dict,
                    provider_field="tool_provider_id",
                    preferred_providers=tool_preferences.get(tool_id, []),
                )
    return json.dumps(merged)


def _load_tools_from_root(
    root: Path,
    *,
    inferred_provider_id: str,
) -> dict[str, dict[str, Any]]:
    """Load all internal tools from one root directory."""

    catalog: dict[str, dict[str, Any]] = {}
    for py_path in _iter_tool_module_files(root):
        yml_path = _resolve_tool_metadata_path(py_path)
        if not yml_path.is_file():
            logger.warning("Skipping tool module '%s': missing '%s'", py_path, yml_path.name)
            continue
        tool_dict = _load_runtime_tool(py_path, yml_path, root)
        if tool_dict is not None:
            tool_dict.setdefault("tool_provider_id", inferred_provider_id)
            catalog[tool_dict["tool_id"]] = tool_dict
    return catalog


def _iter_tool_module_files(tools_dir: Path) -> list[Path]:
    """Return sorted Python tool modules (``.py`` files with a ``.tool.yml`` sibling)."""

    runtime_modules: list[Path] = []
    for module_path in tools_dir.rglob("*.py"):
        if module_path.name == "__init__.py" or "__pycache__" in module_path.parts:
            continue
        runtime_modules.append(module_path)
    return sorted(runtime_modules)


def _resolve_tool_metadata_path(py_path: Path) -> Path:
    """Resolve the ``.tool.yml`` metadata path for a tool module.

    Supports two conventions:
    - ``my_tool.py`` → ``my_tool.tool.yml`` (explicit suffix, preferred)
    - ``my_tool.py`` → ``my_tool.yml`` (legacy fallback)
    """

    explicit = py_path.with_suffix(".tool.yml")
    if explicit.is_file():
        return explicit
    return py_path.with_suffix(".yml")


def _load_runtime_tool(
    py_path: Path,
    yml_path: Path,
    root: Path,
) -> dict[str, Any] | None:
    """Load one Python runtime tool from its ``.tool.yml`` metadata."""

    try:
        metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in '%s': %s", yml_path, exc)
        return None

    if not isinstance(metadata, dict):
        return None

    tool_id = metadata.get("tool_id")
    if not tool_id or not isinstance(tool_id, str):
        return None

    module_path = str(metadata.get("module_path", "") or "").strip()
    if not module_path:
        module_path = _infer_module_path(py_path, root)

    mode_allowlist_raw = metadata.get("mode_allowlist")
    mode_allowlist: list[str] | None
    if mode_allowlist_raw is None:
        mode_allowlist = None
    else:
        mode_allowlist = [str(m) for m in mode_allowlist_raw]

    return {
        "tool_id": tool_id,
        "description": str(metadata.get("description", "") or ""),
        "mode_allowlist": mode_allowlist,
        "max_calls_per_turn": int(metadata.get("max_calls_per_turn", 3) or 3),
        "timeout_seconds": int(metadata.get("timeout_seconds", 30) or 30),
        "module_path": module_path,
    }
