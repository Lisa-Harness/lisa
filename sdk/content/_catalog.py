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

from sdk.config import settings as _sdk_settings
from sdk.content._front_matter import FrontMatterError, parse_front_matter

# Config-backed completion defaults for declarative operations.
_DEFAULT_COMPLETION_MAX_TOKENS = _sdk_settings.declarative_completion_max_tokens
_DEFAULT_COMPLETION_TEMPERATURE = _sdk_settings.declarative_completion_temperature

logger = logging.getLogger(__name__)

__all__ = [
    "CatalogCollisionError",
    "build_mode_catalog_json",
    "build_operation_catalog_json",
    "build_tool_catalog_json",
    "invalidate_catalog_cache",
    "resolve_mode_roots",
    "resolve_operation_roots",
    "resolve_tool_roots",
]

# ── Catalog cache ───────────────────────────────────────────────────────────

_catalog_cache: dict[str, str] = {}


class CatalogCollisionError(ValueError):
    """Raised when duplicate catalog IDs lack one explicit session owner."""


def _roots_fingerprint(roots: list[Path]) -> str:
    """Return a stable string key for a set of catalog roots."""
    return "\n".join(str(r.resolve()) for r in roots)


def invalidate_catalog_cache() -> None:
    """Clear all cached catalog JSON (modes, operations, tools)."""
    _catalog_cache.clear()


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
_SESSION_PROVIDER_SELECTIONS_ENV = "LISA_SESSION_PROVIDER_SELECTIONS"


def build_mode_catalog_json(
    mode_roots: list[Path],
    *,
    operation_catalog_json: str | None = None,
    tool_catalog_json: str | None = None,
    default_mcp_tool_ids: set[str] | None = None,
    strict: bool = False,
) -> str:
    """Scan mode roots and return a JSON-serializable mode catalog.

    Returns a JSON string encoding ``dict[str, dict]`` where each value is a
    mode config dict with nested ``stages``. The kernel deserializes this into
    ``WorkflowModeConfig`` objects.

    When *strict* is ``True``, malformed mode/stage files raise
    :class:`CatalogCollisionError` instead of being logged and skipped. Used
    at host startup so a broken pack install fails fast.

    Three ambient-capability sources feed the merge (all optional, all in the
    SDK catalog layer so the kernel only ever sees the already-merged
    allowlists):

    * ``operation_catalog_json`` / ``tool_catalog_json``: ids flagged
      ``default_available: true`` in their YAML front matter (pack ops/tools).
    * ``default_mcp_tool_ids``: fully namespaced MCP-native tool ids
      (``server_id.tool``) the host declared ambient in ``mcp_servers.yaml``
      via ``default_available_tools``. These are not in the pack tool catalog
      (MCP-native tools are discovered at runtime from subprocesses), so the
      host passes the resolved set in directly.

    Every source is unioned into each mode's allowlists unless the mode opts
    out via ``accept_default_capabilities: false`` or lists an id in
    ``excluded_operation_ids`` / ``excluded_tool_ids``.
    """
    cache_key = (
        "modes:"
        + _roots_fingerprint(mode_roots)
        + _provider_selection_cache_key()
        + ":mcp_defaults="
        + ",".join(sorted(default_mcp_tool_ids or ()))
    )
    cached = _catalog_cache.get(cache_key)
    if cached is not None:
        return cached
    candidates_by_id: dict[str, list[dict[str, Any]]] = {}
    selections = _load_session_provider_selections()
    for root in mode_roots:
        if not root.is_dir():
            continue
        inferred_provider_id = _infer_content_provider_id(root)
        for mode_dir in sorted(root.iterdir()):
            if not mode_dir.is_dir():
                continue
            mode_dict = _load_one_mode(
                mode_dir,
                inferred_provider_id=inferred_provider_id,
                strict=strict,
            )
            if mode_dict is not None:
                mode_id = str(mode_dict.get("mode_id") or "")
                candidates_by_id.setdefault(mode_id, []).append(mode_dict)
    merged = {
        mode_id: _resolve_catalog_candidates(
            resource_kind="mode",
            resource_id=mode_id,
            candidates=candidates,
            provider_field="mode_provider_id",
            selected_provider=selections["modes"].get(mode_id, ""),
        )
        for mode_id, candidates in candidates_by_id.items()
    }
    if operation_catalog_json is not None or tool_catalog_json is not None or default_mcp_tool_ids:
        _merge_default_capabilities(
            modes=merged,
            operation_catalog=_decode_optional_catalog(operation_catalog_json),
            tool_catalog=_decode_optional_catalog(tool_catalog_json),
            default_mcp_tool_ids=set(default_mcp_tool_ids or ()),
        )
    result = json.dumps(merged)
    _catalog_cache[cache_key] = result
    return result


def _decode_optional_catalog(raw: str | None) -> dict[str, dict[str, Any]]:
    """Decode a catalog JSON string into a dict, tolerating None/empty input."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _merge_default_capabilities(
    *,
    modes: dict[str, dict[str, Any]],
    operation_catalog: dict[str, dict[str, Any]],
    tool_catalog: dict[str, dict[str, Any]],
    default_mcp_tool_ids: set[str] | None = None,
) -> None:
    """Merge ``default_available`` ops/tools into each mode's allowlists in place.

    For every mode that has ``accept_default_capabilities`` truthy (default
    True), union its ``allowed_operation_ids`` with the ids of operations
    flagged ``default_available: true`` in ``operation_catalog``, then
    subtract the mode's ``excluded_operation_ids``. Symmetric for tools.

    ``default_mcp_tool_ids`` (namespaced ``server_id.tool`` ids declared
    ambient in ``mcp_servers.yaml``) are unioned into ``allowed_tool_ids``
    alongside the pack-tool defaults, with the same opt-out and idempotent
    semantics. MCP-native tools have no pack YAML front matter, so their
    ambient set is sourced from the host manifest and passed in by the
    caller.

    The merge is idempotent (set union) so modes that already list an ambient
    id are unaffected. Modes with ``accept_default_capabilities: false`` keep
    their explicit allowlists untouched. The kernel never names ids; it only
    filters on the merged allowlists it receives.
    """
    default_op_ids = {
        op_id for op_id, op in operation_catalog.items() if bool(op.get("default_available", False))
    }
    default_tool_ids = {
        tool_id
        for tool_id, tool in tool_catalog.items()
        if bool(tool.get("default_available", False))
    }
    mcp_tool_ids = set(default_mcp_tool_ids or ())
    if not default_op_ids and not default_tool_ids and not mcp_tool_ids:
        return
    for mode in modes.values():
        if not mode.get("accept_default_capabilities", True):
            continue
        excluded_ops = set(mode.get("excluded_operation_ids") or [])
        excluded_tools = set(mode.get("excluded_tool_ids") or [])
        existing_ops = list(mode.get("allowed_operation_ids") or [])
        existing_tools = list(mode.get("allowed_tool_ids") or [])
        # Preserve the mode's authored order; append new defaults at the end.
        seen_ops = set(existing_ops)
        for op_id in sorted(default_op_ids - excluded_ops):
            if op_id not in seen_ops:
                existing_ops.append(op_id)
                seen_ops.add(op_id)
        seen_tools = set(existing_tools)
        for tool_id in sorted((default_tool_ids | mcp_tool_ids) - excluded_tools):
            if tool_id not in seen_tools:
                existing_tools.append(tool_id)
                seen_tools.add(tool_id)
        mode["allowed_operation_ids"] = existing_ops
        mode["allowed_tool_ids"] = existing_tools


def _load_one_mode(
    mode_dir: Path,
    *,
    inferred_provider_id: str,
    strict: bool = False,
) -> dict[str, Any] | None:
    """Load one mode directory into a JSON-serializable dict.

    When *strict* is ``True`` (host startup), a missing ``_mode.md`` or
    malformed front-matter raises instead of being logged and skipped.
    When *strict* is ``False`` (tests, partial installs), the historical
    warning-and-skip behavior is preserved.
    """

    mode_config_path = mode_dir / _MODE_CONFIG_FILENAME
    if not mode_config_path.is_file():
        if strict:
            raise CatalogCollisionError(
                f"Mode folder '{mode_dir.name}' is missing _mode.md — a mode "
                "without _mode.md is not a valid pack install."
            )
        logger.warning("Skipping mode folder '%s': missing _mode.md", mode_dir.name)
        return None

    mode_text = mode_config_path.read_text(encoding="utf-8")
    try:
        front_matter, body = parse_front_matter(mode_text)
    except FrontMatterError as exc:
        if strict:
            raise CatalogCollisionError(
                f"Mode '{mode_dir.name}' has malformed front-matter: {exc}"
            ) from exc
        logger.warning("Skipping mode '%s': %s", mode_dir.name, exc)
        return None

    mode_id = front_matter.get("mode_id")
    if not mode_id or not isinstance(mode_id, str):
        if strict:
            raise CatalogCollisionError(
                f"Mode '{mode_dir.name}' is missing a valid 'mode_id' in its front-matter."
            )
        return None

    stages: dict[str, dict[str, Any]] = {}
    for stage_path in sorted(mode_dir.glob("*.md")):
        if stage_path.name.startswith("_"):
            continue
        stage_dict = _load_one_stage(
            stage_path,
            mode_kind=front_matter.get("mode_kind", "workflow"),
            strict=strict,
        )
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

    # Parse routing_intent_keywords defensively. The schema requires a dict
    # mapping each intent tag to a list of substring keywords, but malformed
    # pack files have shipped a flat list instead. An unguarded ``.items()``
    # here raises ``'list' object has no attribute 'items'`` inside the
    # catalog subprocess, which surfaces as an opaque startup failure. Mirror
    # the kernel's validation (``kernel/ipsar_fsm.py``): raise in strict mode
    # (host startup fails fast on a broken pack install), skip the mode with a
    # warning otherwise.
    routing_intent_keywords_raw = front_matter.get("routing_intent_keywords") or {}
    if not isinstance(routing_intent_keywords_raw, dict):
        if strict:
            raise CatalogCollisionError(
                f"Mode '{mode_id}' has invalid routing_intent_keywords: "
                "must be a mapping of intent tag to keyword list, "
                f"got {type(routing_intent_keywords_raw).__name__} in {mode_config_path}"
            )
        logger.warning(
            "Skipping mode '%s': routing_intent_keywords must be a dict in %s",
            mode_id,
            mode_config_path,
        )
        return None
    routing_intent_keywords: dict[str, list[str]] = {}
    for tag, keywords in routing_intent_keywords_raw.items():
        routing_intent_keywords[str(tag)] = [str(k) for k in keywords]

    return {
        "mode_id": mode_id,
        "mode_provider_id": mode_provider_id,
        "start_stage_id": str(front_matter.get("start_stage_id", "") or ""),
        "stages": stages,
        "mode_kind": str(front_matter.get("mode_kind", "workflow") or "workflow"),
        "artifact_type": str(
            front_matter.get("artifact_type", "workflow_summary") or "workflow_summary"
        ),
        "description": str(front_matter.get("description", "") or ""),
        "instruction_body": body.strip(),
        "mode_markdown_path": str(mode_config_path),
        "default_model": (str(front_matter.get("default_model", "") or "") or None),
        "routable": bool(front_matter.get("routable", False)),
        "priority": int(front_matter.get("priority", 0) or 0),
        "fast_path_patterns": [str(p) for p in (front_matter.get("fast_path_patterns") or [])],
        "routing_intent_tags": [str(t) for t in (front_matter.get("routing_intent_tags") or [])],
        "routing_intent_keywords": routing_intent_keywords,
        "satisfies_intents": [str(t) for t in (front_matter.get("satisfies_intents") or [])],
        "required_capabilities": [
            str(t) for t in (front_matter.get("required_capabilities") or [])
        ],
        "command_pattern": str(front_matter.get("command_pattern", "") or ""),
        "require_query": bool(front_matter.get("require_query", False)),
        "entry_operation_ids": [str(o) for o in (front_matter.get("entry_operation_ids") or [])],
        "pre_turn_operation_ids": [
            str(o) for o in (front_matter.get("pre_turn_operation_ids") or [])
        ],
        "post_turn_operation_ids": [
            str(o) for o in (front_matter.get("post_turn_operation_ids") or [])
        ],
        "allowed_operation_ids": [
            str(o) for o in (front_matter.get("allowed_operation_ids") or [])
        ],
        "allowed_tool_ids": [str(t) for t in (front_matter.get("allowed_tool_ids") or [])],
        "authored_allowed_operation_ids": [
            str(o) for o in (front_matter.get("allowed_operation_ids") or [])
        ],
        "authored_allowed_tool_ids": [str(t) for t in (front_matter.get("allowed_tool_ids") or [])],
        "tools_authoritative": bool(front_matter.get("tools_authoritative", True)),
        "accept_default_capabilities": bool(front_matter.get("accept_default_capabilities", True)),
        "excluded_operation_ids": [
            str(o) for o in (front_matter.get("excluded_operation_ids") or [])
        ],
        "excluded_tool_ids": [str(t) for t in (front_matter.get("excluded_tool_ids") or [])],
        "entry_failure_policy": str(front_matter.get("entry_failure_policy", "abort") or "abort"),
        "workshop_panels": workshop_panels,
        "workshop_budget": front_matter.get("workshop_budget") or {},
        "workshop_planner_prompt_file": workshop_planner_prompt_file,
        "workshop_synthesis_prompt_file": workshop_synthesis_prompt_file,
        "workshop_rendering": front_matter.get("workshop_rendering") or {},
        "default_intent_bridges": _parse_intent_bridges(
            front_matter.get("default_intent_bridges") or []
        ),
        "context_evidence_packs": [
            str(p) for p in (front_matter.get("context_evidence_packs") or [])
        ],
    }


def _provider_selection_cache_key() -> str:
    """Return the raw selection payload so cached catalogs stay session-scoped."""

    return os.getenv(_SESSION_PROVIDER_SELECTIONS_ENV, "")


def _load_session_provider_selections() -> dict[str, dict[str, str]]:
    """Read the strict provider map serialized from the immutable SessionPlan."""

    raw = _provider_selection_cache_key().strip()
    selections: dict[str, dict[str, str]] = {
        "modes": {},
        "operations": {},
        "tools": {},
    }
    if not raw:
        return selections
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CatalogCollisionError(f"Invalid JSON in {_SESSION_PROVIDER_SELECTIONS_ENV}") from exc
    if not isinstance(decoded, dict):
        raise CatalogCollisionError(f"{_SESSION_PROVIDER_SELECTIONS_ENV} must contain an object")

    unexpected_kinds = set(decoded) - set(selections)
    if unexpected_kinds:
        values = ", ".join(sorted(str(kind) for kind in unexpected_kinds))
        raise CatalogCollisionError(f"Unexpected provider selection kinds: {values}")

    for kind, parsed_assignments in selections.items():
        assignments = decoded.get(kind, {})
        if not isinstance(assignments, dict):
            raise CatalogCollisionError(f"Provider selections for {kind} must be an object")
        for resource_id, provider_id in assignments.items():
            if not isinstance(resource_id, str) or not isinstance(provider_id, str):
                raise CatalogCollisionError(f"Provider selections for {kind} require string IDs")
            normalized_resource = resource_id.strip()
            normalized_provider = _normalize_provider_id(provider_id)
            if not normalized_resource or not normalized_provider:
                raise CatalogCollisionError(f"Provider selections for {kind} require non-empty IDs")
            parsed_assignments[normalized_resource] = normalized_provider
    return selections


def _infer_content_provider_id(root: Path) -> str:
    """Infer a stable provider id from a content sub-root path.

    The authoritative source is the distribution package name declared in the
    content root's ``pyproject.toml`` (``[project] name``, e.g.
    ``lisa-pack-memory``). The session manifest's
    ``resources.packs.configuration`` block is keyed by that same distribution
    name, and ``SessionPackConfigurationResolver.for_pack`` looks ops up by
    ``effective_provider_id()``. When inference returned the bare directory
    slug (``memory``) instead of the distribution name (``lisa-pack-memory``),
    the resolver silently dropped the pack's config — which left
    ``synthesize_and_store_learning`` with the 512-token module fallback
    instead of the manifest's ``memory_distill_max_tokens``, truncating the
    distillation JSON mid-fact on every learn turn (2026-08-02 incident).

    Falls back to the path-guess heuristic only when no readable
    ``pyproject.toml`` sits at the content root. This keeps synthetic test
    roots (no ``pyproject.toml``) working unchanged.
    """

    resolved = root.resolve()
    # The content root is the parent of the operations/tools/modes sub-root.
    pyproject = resolved.parent / "pyproject.toml"
    dist_name = _read_distribution_name(pyproject)
    if dist_name:
        return _normalize_provider_id(dist_name)

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


def _read_distribution_name(pyproject: Path) -> str:
    """Return the ``[project] name`` from a content-root pyproject, or ''.

    Isolated so the inference path stays testable without a real pack on disk.
    Returns the empty string for a missing/unreadable/parseable file or a
    ``pyproject.toml`` with no ``[project] name`` rather than raising — the
    caller falls back to the path-guess heuristic.
    """
    import tomllib  # noqa: PLC0415

    if not pyproject.is_file():
        return ""
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    project = payload.get("project") if isinstance(payload, dict) else None
    if not isinstance(project, dict):
        return ""
    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip()


def _normalize_provider_id(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return value


def _backfill_provider_ids(
    cap_dict: dict[str, Any],
    inferred_provider_id: str,
    *,
    provider_field: str,
) -> None:
    """Populate both provider-id fields on a capability dict in place.

    Symmetric backfill so ``pack_id`` (raw authored slug) and
    ``provider_field`` (the fully-qualified ``tool_provider_id`` /
    ``operation_provider_id``) are never empty after load. Each field is
    filled from the other when one is authored and the other is not, and
    both fall back to the content-root-inferred id. This removes the
    ``provider_id or pack_id`` fallback chains in kernel consumers: after
    load, ``ToolDefinition.effective_provider_id()`` /
    ``OperationDefinition.effective_provider_id()`` always return a
    populated value for a capability that belongs to a discoverable pack.
    """
    authored_pack_id = str(cap_dict.get("pack_id", "") or "").strip()
    if not cap_dict.get(provider_field):
        # The authored pack_id is authoritative (e.g. "lisa-pack-coding").
        # The content-root-inferred id is a fallback guess from the directory
        # structure (e.g. "lisa-packs") and may not match session plan config
        # keys. Prefer pack_id so resolver.for_pack() lookups succeed.
        cap_dict[provider_field] = authored_pack_id or inferred_provider_id
    if not authored_pack_id:
        cap_dict["pack_id"] = cap_dict[provider_field]


def _resolve_catalog_candidates(
    *,
    resource_kind: str,
    resource_id: str,
    candidates: list[dict[str, Any]],
    provider_field: str,
    selected_provider: str,
) -> dict[str, Any]:
    """Resolve duplicate IDs only through an explicit session provider choice."""

    if len(candidates) == 1:
        return candidates[0]
    if not selected_provider:
        raise CatalogCollisionError(
            f"Duplicate {resource_kind} ID '{resource_id}' requires an explicit "
            f"providers.{resource_kind}s selection."
        )

    matching = [
        candidate
        for candidate in candidates
        if _normalize_provider_id(str(candidate.get(provider_field) or "")) == selected_provider
    ]
    if len(matching) == 1:
        return matching[0]

    available = sorted(
        {
            _normalize_provider_id(str(candidate.get(provider_field) or ""))
            for candidate in candidates
        }
        - {""}
    )
    available_text = ", ".join(available) or "(none)"
    raise CatalogCollisionError(
        f"Session selected provider '{selected_provider}' for {resource_kind} "
        f"ID '{resource_id}', but matching candidates are ambiguous or absent "
        f"(available: {available_text})."
    )


def _load_one_stage(
    stage_path: Path, *, mode_kind: str, strict: bool = False
) -> dict[str, Any] | None:
    """Load one stage markdown file into a JSON-serializable dict.

    Returns None if the file has no ``stage_id`` in its front-matter (it's an
    auxiliary asset, not a stage definition). For ``chat``/``action`` modes,
    missing ``stage_id`` is silently skipped; for workflow modes, we skip
    but log it. When *strict* is ``True``, malformed front-matter raises.
    """

    stage_text = stage_path.read_text(encoding="utf-8")
    try:
        front_matter, body = parse_front_matter(stage_text)
    except FrontMatterError as exc:
        if strict:
            raise CatalogCollisionError(
                f"Stage file '{stage_path}' has malformed front-matter: {exc}"
            ) from exc
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
        result.append(
            {
                "target_mode_id": str(bridge.get("target_mode_id", "") or ""),
                "matcher_patterns": [str(p) for p in (bridge.get("matcher_patterns") or [])],
                "query_extraction_strategy": str(
                    bridge.get("query_extraction_strategy", "passthrough") or "passthrough"
                ),
                "prerequisites": [str(p) for p in (bridge.get("prerequisites") or [])],
                "recent_handoff_source_modes": [
                    str(m) for m in (bridge.get("recent_handoff_source_modes") or [])
                ],
                "recent_handoff_required_artifact_keys": [
                    str(k) for k in (bridge.get("recent_handoff_required_artifact_keys") or [])
                ],
                "recent_handoff_context_key": str(
                    bridge.get("recent_handoff_context_key", "") or ""
                ),
                "response_appendix": str(bridge.get("response_appendix", "") or ""),
            }
        )
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
    cache_key = (
        "operations:" + _roots_fingerprint(operation_roots) + _provider_selection_cache_key()
    )
    cached = _catalog_cache.get(cache_key)
    if cached is not None:
        return cached
    candidates_by_id: dict[str, list[dict[str, Any]]] = {}
    selections = _load_session_provider_selections()
    for root in operation_roots:
        if not root.is_dir():
            continue
        inferred_provider_id = _infer_content_provider_id(root)
        for op_id, op_dict in _load_operations_from_root(
            root,
            inferred_provider_id=inferred_provider_id,
        ).items():
            candidates_by_id.setdefault(op_id, []).append(op_dict)
    merged = {
        operation_id: _resolve_catalog_candidates(
            resource_kind="operation",
            resource_id=operation_id,
            candidates=candidates,
            provider_field="operation_provider_id",
            selected_provider=selections["operations"].get(operation_id, ""),
        )
        for operation_id, candidates in candidates_by_id.items()
    }
    result = json.dumps(merged)
    _catalog_cache[cache_key] = result
    return result


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
            _backfill_provider_ids(
                op_dict, inferred_provider_id, provider_field="operation_provider_id"
            )
            catalog[op_dict["operation_id"]] = op_dict

    # 2. Python runtime operations with .yml metadata
    for py_path in _iter_operation_module_files(root):
        yml_path = py_path.with_suffix(_OPERATION_METADATA_SUFFIX)
        if not yml_path.is_file():
            logger.warning("Skipping operation module '%s': missing '%s'", py_path, yml_path.name)
            continue
        op_dict = _load_runtime_operation(py_path, yml_path, root)
        if op_dict is not None:
            _backfill_provider_ids(
                op_dict, inferred_provider_id, provider_field="operation_provider_id"
            )
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
        "pack_id": str(metadata.get("pack_id", "") or "").strip(),
        "operation_provider_id": str(metadata.get("operation_provider_id", "") or "").strip(),
        "effect": str(metadata.get("effect", "read") or "read"),
        "approval_required": bool(metadata.get("approval_required", False)),
        "approval_preview": str(metadata.get("approval_preview", "") or "").strip(),
        "execution_kind": "prompt",
        "tool_ids": [str(t) for t in (metadata.get("tool_ids") or [])],
        "mode_allowlist": [str(m) for m in (metadata.get("mode_allowlist") or [])],
        "preferred_model_id": (
            str(metadata["preferred_model_id"]) if "preferred_model_id" in metadata else None
        ),
        "response_format_model_path": str(
            metadata.get("response_format_model_path", "") or ""
        ).strip(),
        "response_output_text_field": str(
            metadata.get("response_output_text_field", "output_text") or "output_text"
        ).strip()
        or "output_text",
        "response_output_json_passthrough": _parse_string_mapping(
            metadata.get("response_output_json_passthrough")
        ),
        "timeout_seconds": int(metadata.get("timeout_seconds", 60) or 60),
        "max_calls_per_turn": int(metadata.get("max_calls_per_turn", 0) or 0),
        "completion_max_tokens": int(
            metadata.get("completion_max_tokens", _DEFAULT_COMPLETION_MAX_TOKENS)
            or _DEFAULT_COMPLETION_MAX_TOKENS
        ),
        "completion_temperature": float(
            metadata.get("completion_temperature", _DEFAULT_COMPLETION_TEMPERATURE)
            or _DEFAULT_COMPLETION_TEMPERATURE
        ),
        "instruction_markdown_path": str(md_path),
        "instruction_body": body,
        "module_path": "",
        "exportable": bool(metadata.get("exportable", False)),
        "default_available": bool(metadata.get("default_available", False)),
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
        "pack_id": str(metadata.get("pack_id", "") or "").strip(),
        "operation_provider_id": str(metadata.get("operation_provider_id", "") or "").strip(),
        "effect": str(metadata.get("effect", "read") or "read"),
        "approval_required": bool(metadata.get("approval_required", False)),
        "approval_preview": str(metadata.get("approval_preview", "") or "").strip(),
        "execution_kind": "runtime",
        "tool_ids": [str(t) for t in (metadata.get("tool_ids") or [])],
        "mode_allowlist": [str(m) for m in (metadata.get("mode_allowlist") or [])],
        "preferred_model_id": (
            str(metadata["preferred_model_id"]) if "preferred_model_id" in metadata else None
        ),
        "timeout_seconds": int(metadata.get("timeout_seconds", 60) or 60),
        "max_calls_per_turn": int(metadata.get("max_calls_per_turn", 0) or 0),
        "instruction_markdown_path": "",
        "instruction_body": "",
        "module_path": module_path,
        "exportable": bool(metadata.get("exportable", False)),
        "default_available": bool(metadata.get("default_available", False)),
    }


def _is_package_dir(path: Path) -> bool:
    """Return True if ``path`` is importable as a package.

    Recognizes both regular packages (with ``__init__.py``) and PEP 420
    namespace packages, which install as a directory on ``sys.path`` with
    no ``__init__.py``. External packs ship under the ``lisa_packs`` namespace
    package; the namespace ``__init__.py`` is not always present on disk
    (it is shared across pack wheels and can be dropped during install), so
    relying solely on ``__init__.py`` would mis-infer bare module paths like
    ``coding.tools.status.status`` for namespace-pack contents.
    """

    if (path / "__init__.py").is_file():
        return True
    # PEP 420 namespace package: resolvable via the import system even
    # without an __init__.py when an ancestor is on sys.path.
    import importlib.util

    spec = importlib.util.find_spec(path.name)
    if spec is None or spec.submodule_search_locations is None:
        return False
    resolved = {Path(p).resolve() for p in spec.submodule_search_locations}
    return path.resolve() in resolved


def _infer_module_path(py_path: Path, root: Path) -> str:
    """Infer a dotted Python module path from the file path relative to root.

    The ``operations`` directory name is included in the path so that
    ``sdk/packs/introspect/operations/self_model/self_model.py`` resolves to
    ``sdk.packs.introspect.operations.self_model``.

    The package base is discovered by walking up from ``root``'s parent while
    each directory is importable as a package (regular or PEP 420 namespace),
    so namespace-pack contents under ``lisa_packs`` resolve to
    ``lisa_packs.<pack>.operations.<op>`` even when the shared namespace
    ``__init__.py`` is absent from site-packages.
    """

    relative = py_path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    # Walk up from root's parent to find the package base
    package_dir = root.parent
    pkg_parts: list[str] = []
    while _is_package_dir(package_dir):
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
    cache_key = "tools:" + _roots_fingerprint(tool_roots) + _provider_selection_cache_key()
    cached = _catalog_cache.get(cache_key)
    if cached is not None:
        return cached
    candidates_by_id: dict[str, list[dict[str, Any]]] = {}
    selections = _load_session_provider_selections()
    for root in tool_roots:
        if not root.is_dir():
            continue
        inferred_provider_id = _infer_content_provider_id(root)
        for tool_id, tool_dict in _load_tools_from_root(
            root,
            inferred_provider_id=inferred_provider_id,
        ).items():
            candidates_by_id.setdefault(tool_id, []).append(tool_dict)
    merged = {
        tool_id: _resolve_catalog_candidates(
            resource_kind="tool",
            resource_id=tool_id,
            candidates=candidates,
            provider_field="tool_provider_id",
            selected_provider=selections["tools"].get(tool_id, ""),
        )
        for tool_id, candidates in candidates_by_id.items()
    }
    result = json.dumps(merged)
    _catalog_cache[cache_key] = result
    return result


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
            _backfill_provider_ids(
                tool_dict, inferred_provider_id, provider_field="tool_provider_id"
            )
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
        "pack_id": str(metadata.get("pack_id", "") or "").strip(),
        "tool_provider_id": str(metadata.get("tool_provider_id", "") or "").strip(),
        "effect": str(metadata.get("effect", "read") or "read"),
        "approval_required": bool(metadata.get("approval_required", False)),
        "approval_preview": str(metadata.get("approval_preview", "") or "").strip(),
        "mode_allowlist": mode_allowlist,
        "max_calls_per_turn": int(metadata.get("max_calls_per_turn", 0) or 0),
        "timeout_seconds": int(metadata.get("timeout_seconds", 30) or 30),
        "module_path": module_path,
        "exportable": bool(metadata.get("exportable", False)),
        "default_available": bool(metadata.get("default_available", False)),
    }
