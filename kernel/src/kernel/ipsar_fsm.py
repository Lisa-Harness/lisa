"""Workflow finite-state-machine loader and runtime for IPSAR mode progression.

This module is a pure loader/router: it has zero hardcoded knowledge of specific
modes or stages.  The workflow catalog is loaded from the ``lisa-content`` MCP
server at startup; the kernel never touches the filesystem directly.

    {mode_name}/
      _mode.md         → YAML front-matter declares mode_id, start_stage_id,
                         mode_kind, description, command_pattern, and operation
                         bindings for the user-facing mode entry point
      {stage_id}.md    → YAML front-matter declares stage config; body text is
                         the instruction prose injected into AI prompts

Adding a new workflow mode requires only a new folder with the appropriate files;
no code changes are needed. The ``lisa-content`` server discovers and parses
mode packs from content roots.

Runtime functions (create_position, apply_stage_transition, enter_mode,
etc.) delegate to the catalog loaded by ``_ensure_catalog_loaded()`` on
first access.

Called by: host apps (position management, stage transitions, context injection)
           kernel.ipsar (parse_route for FSM mode detection)
           packaging tools (manifest summary generation)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from kernel.approval import ApprovalRequired
from kernel.execution_policy import ExecutionPolicy, resolve_capability_invocation_limit
from kernel.function_tools import build_strict_openai_function_tool
from kernel.instruction_conditionals import (
    conditional_operation_ids,
    has_conditional_markers,
    strip_conditional_operation_blocks,
)
from kernel.intake.connector_discovery import discover_installed_signal_collectors
from kernel.models import (
    CallLedger,
    DefaultIntentBridge,
    ModeKind,
    OperationExecutionContext,
    OperationResult,
    ToolExecutionContext,
    WorkflowModeConfig,
    WorkflowTransitionRequest,
    YieldPolicy,
)
from kernel.tools.mcp_client import MCPClientConfig

logger = logging.getLogger(__name__)

# The lisa-content MCP server config, injected by the host at startup.
# When None, the kernel cannot load the mode catalog.
_content_mcp_config: MCPClientConfig | None = None

_EXIT_STAGE_ID = "exit"

# Front-matter delimiter used in instruction markdown files.
_FRONT_MATTER_DELIMITER = "---"

# Baseline instructions root for ``load_catalog_from_disk()``. Injected by the
# host at startup via ``register_instructions_root()``; when ``None`` the disk
# loader requires an explicit ``instructions_dir`` argument (the MCP catalog
# path, used at runtime, is unaffected). Mirrors the ``_content_mcp_config``
# injection contract: the kernel stays decoupled from any fixed filesystem path.
_INSTRUCTIONS_ROOT: Path | None = None

# Callbacks injected by the host app layer so the kernel stays decoupled
# from the concrete operation registry module. Set at startup via
# register_operation_callbacks().
_get_operation_catalog_fn: Callable[[], dict[str, Any]] | None = None
_execute_operation_fn: Callable[[str, OperationExecutionContext], Any] | None = None


class WorkflowValidationError(ValueError):
    """Raised when a workflow mode configuration violates FSM invariants.

    Thrown at startup during catalog validation so misconfigured modes are caught
    before any user traffic arrives.
    """


class MissingModeInstructionError(WorkflowValidationError):
    """Raised when a required mode or stage instruction body is missing/empty.

    A workflow mode with no authored ``instruction_body`` in its ``_mode.md``
    (or a workflow stage with no ``instruction_body``) would silently produce
    an empty system-prompt section, causing the model to run modeless. This
    error fails the turn loudly so the missing content is surfaced instead
    of degraded into an empty prompt.
    """


@dataclass(slots=True)
class WorkflowToolCall:
    """Normalized tool call payload extracted from model completion metadata.

    Produced by the host agent loop when the model returns a 'transition_stage'
    tool call and passed to apply_transition() to advance the session.
    """

    name: str
    arguments: dict[str, object]


def register_operation_callbacks(
    get_operation_catalog_fn: Callable[[], dict[str, Any]],
    execute_operation_fn: Callable[[str, OperationExecutionContext], Any],
) -> None:
    """Inject operation-registry callbacks from the host app layer.

    The kernel cannot import the app-owned wiring layer directly (layer
    boundary), so the host calls this once at startup to wire the two halves
    together.
    """
    global _get_operation_catalog_fn, _execute_operation_fn  # noqa: PLW0603
    _get_operation_catalog_fn = get_operation_catalog_fn
    _execute_operation_fn = execute_operation_fn


def register_content_mcp(config: MCPClientConfig) -> None:
    """Inject the ``lisa-catalog`` MCP server config for catalog loading.

    Must be called before the workflow catalog is first accessed (before any
    call to ``get_workflow_catalog()``). The kernel calls ``list_modes`` on the
    server to discover and load all workflow modes. The server does all
    filesystem access; the kernel never touches the filesystem directly.
    """
    global _content_mcp_config  # noqa: PLW0603
    _content_mcp_config = config


def register_instructions_root(root: Path | None) -> None:
    """Inject the baseline instructions root for ``load_catalog_from_disk()``.

    Optional: only needed when the disk loader is invoked without an explicit
    ``instructions_dir`` (e.g. from a host that wants the kernel's own baseline
    scan root). Test code always passes ``instructions_dir`` explicitly, so it
    does not depend on this registration.
    """
    global _INSTRUCTIONS_ROOT  # noqa: PLW0603
    _INSTRUCTIONS_ROOT = root


def _known_connector_ids() -> frozenset[str]:
    """Return installed signal connector IDs discovered from extension entry points."""

    return frozenset(discover_installed_signal_collectors())


def _get_operation_catalog() -> dict[str, Any]:
    """Return the operation catalog via the injected callback."""
    if _get_operation_catalog_fn is None:
        raise RuntimeError(
            "Operation callbacks not registered. "
            "Call kernel.ipsar_fsm.register_operation_callbacks() at startup."
        )
    return _get_operation_catalog_fn()


async def _execute_operation(
    operation_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute an operation via the injected callback."""
    if _execute_operation_fn is None:
        raise RuntimeError(
            "Operation callbacks not registered. "
            "Call kernel.ipsar_fsm.register_operation_callbacks() at startup."
        )
    result = await _execute_operation_fn(operation_id, ctx)
    if isinstance(result, OperationResult):
        return result
    if isinstance(result, str):
        return OperationResult(output_text=result)
    raise WorkflowValidationError(
        f"Operation '{operation_id}' returned unsupported result type {type(result).__name__}."
    )


# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a YAML front-matter block from the top of a markdown string.

    Expects the file to start with a `---` delimiter line, followed by YAML
    content, followed by another `---` delimiter line.  The body is everything
    after the closing delimiter, stripped of leading and trailing whitespace.

    Always uses yaml.safe_load() — never yaml.load() — to avoid arbitrary
    code execution vulnerabilities in untrusted files.

    Args:
        text: Full file contents including the front-matter block.

    Returns:
        A tuple of (front_matter_dict, body_text).  If no valid front-matter
        block is found, returns ({}, text) so callers degrade gracefully.

    Raises:
        WorkflowValidationError: If the YAML in the front-matter is invalid.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        # No front-matter block: treat entire content as body.
        return {}, text

    # Find the closing --- line (must be after at least one content line).
    try:
        close_idx = next(
            i
            for i, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONT_MATTER_DELIMITER
        )
    except StopIteration:
        # Opening --- found but no closing ---: treat entire content as body.
        return {}, text

    yaml_text = "\n".join(lines[1:close_idx])
    body_text = "\n".join(lines[close_idx + 1 :]).strip()

    try:
        front_matter = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise WorkflowValidationError(f"Invalid YAML front-matter: {exc}") from exc

    if not isinstance(front_matter, dict):
        raise WorkflowValidationError(
            f"Front-matter must be a YAML mapping, got {type(front_matter).__name__}."
        )

    return front_matter, body_text


# ---------------------------------------------------------------------------
# Mode-level front-matter overrides (_overrides.yaml)
# ---------------------------------------------------------------------------


_MODE_OVERRIDES_FILENAME = "_overrides.yaml"
# Kernel-owned front-matter keys that overrides may NOT touch. This is a
# denylist, not an allowlist: the kernel names only the keys it owns and
# protects (routing, FSM, operation bindings), and passes every other key
# through opaquely as extension data. App-owned concepts like ``lenses`` are
# never named here — they flow through ``extra_front_matter`` untouched.
_MODE_OVERRIDES_PROTECTED_KEYS: frozenset[str] = frozenset(
    {
        "mode_id",
        "start_stage_id",
        "mode_kind",
        "command_pattern",
        "require_query",
        "entry_operation_ids",
        "entry_op_kind",
        "pre_turn_operation_ids",
        "post_turn_operation_ids",
        "allowed_operation_ids",
        "allowed_tool_ids",
        "tools_authoritative",
        "accept_default_capabilities",
        "excluded_operation_ids",
        "excluded_tool_ids",
        "routable",
        "priority",
        "fast_path_patterns",
        "routing_intent_tags",
        "routing_intent_keywords",
        "satisfies_intents",
        "required_capabilities",
        "completion_budget_tokens",
        "turn_completion_budget_tokens",
        "default_intent_bridges",
        "artifact_type",
        "yield_policy",
        "approval_continuation",
        "task_completion",
        "task_artifact_tool_id",
    }
)
# Every front-matter key the loader reads into a typed WorkflowModeConfig field.
# Used to compute ``extra_front_matter`` (the opaque remainder) so the kernel
# never has to name app-owned keys. Kept in sync with the field reads in
# load_catalog_from_disk(); ``description`` is included because it is read
# into ``mode_description`` even though overrides may freely replace it.
_KERNEL_OWNED_FRONT_MATTER_KEYS: frozenset[str] = _MODE_OVERRIDES_PROTECTED_KEYS | frozenset(
    {
        "description",
        "default_model",
        "mode_provider_id",
        "context_item_packs",
        "yield_policy",
        "fork_allowed",
    }
)


def _deep_merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` on top of ``base`` for list/dict values.

    - Dict values are merged recursively.
    - List values from overrides replace list values from base (no concatenation);
      this keeps extension sets deterministic and avoids duplicate-accumulation bugs.
    - Scalar values from overrides replace scalars from base.
    """
    merged = dict(base)
    for key, override_value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge_overrides(merged[key], override_value)
        else:
            merged[key] = override_value
    return merged


def _apply_mode_overrides(
    mode_dir: Path,
    base_front_matter: dict[str, Any],
) -> dict[str, Any]:
    """Apply an optional sibling ``_overrides.yaml`` to mode front matter.

    The overrides file is a YAML mapping deep-merged on top of the ``_mode.md``
    front matter. The kernel protects its own owned keys (routing, FSM,
    operation bindings) via ``_MODE_OVERRIDES_PROTECTED_KEYS`` — attempting to
    override any of them is a startup ``WorkflowValidationError``. Every other
    key is app/pack-owned extension data and passes through untouched; the
    kernel does not name, interpret, or allowlist those keys.

    Args:
        mode_dir: Directory containing ``_mode.md`` and the optional overrides file.
        base_front_matter: Parsed front matter from ``_mode.md``.

    Returns:
        A new front-matter dict with overrides deep-merged on top. If no
        overrides file exists, ``base_front_matter`` is returned unchanged.
    """
    overrides_path = mode_dir / _MODE_OVERRIDES_FILENAME
    if not overrides_path.exists():
        return base_front_matter

    try:
        raw = overrides_path.read_text(encoding="utf-8")
        overrides = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise WorkflowValidationError(
            f"Invalid YAML in overrides file '{overrides_path}': {exc}"
        ) from exc

    if not isinstance(overrides, dict):
        raise WorkflowValidationError(
            f"Overrides file '{overrides_path}' must be a YAML mapping, "
            f"got {type(overrides).__name__}."
        )

    protected = sorted(set(overrides) & _MODE_OVERRIDES_PROTECTED_KEYS)
    if protected:
        raise WorkflowValidationError(
            f"Overrides file '{overrides_path}' cannot override kernel-owned keys: {protected}."
        )

    return _deep_merge_overrides(base_front_matter, overrides)


# ---------------------------------------------------------------------------
# Disk-scanning catalog loader
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ModeOperationBindings:
    """Parsed operation/tool binding fields from one mode's front matter."""

    entry_operation_ids: list[str]
    entry_op_kind: Literal["work", "routing"]
    pre_turn_operation_ids: list[str]
    post_turn_operation_ids: list[str]
    allowed_operation_ids: list[str]
    allowed_tool_ids: list[str]
    authored_allowed_operation_ids: list[str]
    authored_allowed_tool_ids: list[str]
    tools_authoritative: bool
    accept_default_capabilities: bool
    excluded_operation_ids: list[str]
    excluded_tool_ids: list[str]


def _parse_mode_operation_bindings(
    mode_front_matter: dict[str, Any],
    mode_kind: ModeKind,
    mode_config_path: Path,
) -> _ModeOperationBindings:
    """Parse operation/tool binding fields from mode front matter.

    Args:
        mode_front_matter: Parsed (override-merged) mode front matter.
        mode_kind: The mode's kind (entry_op_kind='routing' requires 'action').
        mode_config_path: Path of ``_mode.md`` (for error messages).

    Returns:
        The parsed binding fields.

    Raises:
        WorkflowValidationError: On invalid field values.
    """
    entry_operation_ids = [
        str(operation_id) for operation_id in (mode_front_matter.get("entry_operation_ids") or [])
    ]
    entry_op_kind_raw = str(mode_front_matter.get("entry_op_kind", "work") or "work")
    if entry_op_kind_raw == "work":
        entry_op_kind: Literal["work", "routing"] = "work"
    elif entry_op_kind_raw == "routing":
        entry_op_kind = "routing"
    else:
        raise WorkflowValidationError(
            f"entry_op_kind must be 'work' or 'routing' in {mode_config_path}"
        )
    if entry_op_kind == "routing" and mode_kind != "action":
        raise WorkflowValidationError(
            f"entry_op_kind 'routing' requires mode_kind 'action' in "
            f"{mode_config_path}: only action modes run entry operations"
        )
    pre_turn_operation_ids = [
        str(operation_id)
        for operation_id in (mode_front_matter.get("pre_turn_operation_ids") or [])
    ]
    post_turn_operation_ids = [
        str(operation_id)
        for operation_id in (mode_front_matter.get("post_turn_operation_ids") or [])
    ]
    allowed_operation_ids = [
        str(operation_id) for operation_id in (mode_front_matter.get("allowed_operation_ids") or [])
    ]
    allowed_tool_ids = [
        str(tool_id) for tool_id in (mode_front_matter.get("allowed_tool_ids") or [])
    ]
    # The disk path does not run the SDK default-available merge, so the
    # authored and effective allowlists are identical. The SDK MCP path
    # sets authored_allowed_*_ids to the pre-merge values and lets the
    # merge populate allowed_*_ids with ambient capabilities.
    authored_allowed_operation_ids = list(allowed_operation_ids)
    authored_allowed_tool_ids = list(allowed_tool_ids)
    tools_authoritative = mode_front_matter.get("tools_authoritative", True)
    if not isinstance(tools_authoritative, bool):
        raise WorkflowValidationError(
            f"tools_authoritative must be a boolean in {mode_config_path}"
        )
    accept_default_capabilities = mode_front_matter.get("accept_default_capabilities", True)
    if not isinstance(accept_default_capabilities, bool):
        raise WorkflowValidationError(
            f"accept_default_capabilities must be a boolean in {mode_config_path}"
        )
    excluded_operation_ids = [
        str(op_id) for op_id in (mode_front_matter.get("excluded_operation_ids") or [])
    ]
    excluded_tool_ids = [
        str(tool_id) for tool_id in (mode_front_matter.get("excluded_tool_ids") or [])
    ]
    return _ModeOperationBindings(
        entry_operation_ids=entry_operation_ids,
        entry_op_kind=entry_op_kind,
        pre_turn_operation_ids=pre_turn_operation_ids,
        post_turn_operation_ids=post_turn_operation_ids,
        allowed_operation_ids=allowed_operation_ids,
        allowed_tool_ids=allowed_tool_ids,
        authored_allowed_operation_ids=authored_allowed_operation_ids,
        authored_allowed_tool_ids=authored_allowed_tool_ids,
        tools_authoritative=tools_authoritative,
        accept_default_capabilities=accept_default_capabilities,
        excluded_operation_ids=excluded_operation_ids,
        excluded_tool_ids=excluded_tool_ids,
    )


@dataclass(slots=True)
class _ModeRoutingFields:
    """Parsed routing fields from one mode's front matter."""

    routable: bool
    priority: int
    fast_path_patterns: list[str]
    routing_intent_tags: list[str]
    routing_intent_keywords: dict[str, list[str]]
    satisfies_intents: list[str]
    required_capabilities: list[str]


def _parse_lowercase_tag_list(
    mode_front_matter: dict[str, Any],
    field: str,
    mode_config_path: Path,
) -> list[str]:
    """Parse a list[str] field into stripped, lowercased, non-empty tags.

    Shared by ``routing_intent_tags``, ``satisfies_intents``, and
    ``required_capabilities`` parsing.

    Raises:
        WorkflowValidationError: If the field is not a list of non-empty strings.
    """
    raw_list = mode_front_matter.get(field) or []
    if not isinstance(raw_list, list):
        raise WorkflowValidationError(f"{field} must be a list in {mode_config_path}")
    parsed: list[str] = []
    for idx, raw_tag in enumerate(raw_list):
        if not isinstance(raw_tag, str):
            raise WorkflowValidationError(f"{field}[{idx}] must be a string in {mode_config_path}")
        tag = raw_tag.strip().lower()
        if not tag:
            raise WorkflowValidationError(f"{field}[{idx}] cannot be empty in {mode_config_path}")
        parsed.append(tag)
    return parsed


def _parse_routing_intent_keywords(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> dict[str, list[str]]:
    """Parse the ``routing_intent_keywords`` per-tag keyword map from mode front matter.

    Maps each intent tag to the substring keywords that signal that specific
    intent. These are case-insensitive substring matches (not anchored regexes).

    Raises:
        WorkflowValidationError: If the field is not a dict of tag -> list[str].
    """
    routing_intent_keywords_raw = mode_front_matter.get("routing_intent_keywords") or {}
    if not isinstance(routing_intent_keywords_raw, dict):
        raise WorkflowValidationError(
            f"routing_intent_keywords must be a dict in {mode_config_path}"
        )
    routing_intent_keywords: dict[str, list[str]] = {}
    for tag, keywords in routing_intent_keywords_raw.items():
        tag_str = str(tag).strip().lower()
        if not tag_str:
            raise WorkflowValidationError(
                f"routing_intent_keywords has an empty tag in {mode_config_path}"
            )
        if not isinstance(keywords, list):
            raise WorkflowValidationError(
                f"routing_intent_keywords[{tag_str}] must be a list in {mode_config_path}"
            )
        kw_list: list[str] = []
        for idx, raw_kw in enumerate(keywords):
            if not isinstance(raw_kw, str):
                raise WorkflowValidationError(
                    f"routing_intent_keywords[{tag_str}][{idx}] must be a "
                    f"string in {mode_config_path}"
                )
            kw = raw_kw.strip().lower()
            if not kw:
                raise WorkflowValidationError(
                    f"routing_intent_keywords[{tag_str}][{idx}] cannot be "
                    f"empty in {mode_config_path}"
                )
            kw_list.append(kw)
        routing_intent_keywords[tag_str] = kw_list
    return routing_intent_keywords


def _validate_routing_intent_coverage(
    routing_intent_tags: list[str],
    routing_intent_keywords: dict[str, list[str]],
    mode_config_path: Path,
) -> None:
    """Validate that every routing_intent_keywords key is declared in routing_intent_tags.

    A keyword bucket keyed by a tag the mode does not cover produces an
    uncoverable intent from any query that hits the bucket: the detector
    attributes the tag, but no routing target claims it in
    routing_intent_tags, so the gap can never be covered and the hint layer
    falls through to ambiguity (the 2026-08-11 executor incident: bucket
    keyed ``execute:`` while routing_intent_tags were ``[code, write,
    implement]``). Fail fast at catalog load so a broken pack install
    cannot reach the router.

    Raises:
        WorkflowValidationError: On keywords without declared tags.
    """
    routing_intent_tags_set = set(routing_intent_tags)
    if routing_intent_keywords and not routing_intent_tags_set:
        raise WorkflowValidationError(
            f"routing_intent_keywords cannot be set without routing_intent_tags "
            f"in {mode_config_path}"
        )
    _orphan_keyword_tags = set(routing_intent_keywords) - routing_intent_tags_set
    if _orphan_keyword_tags:
        raise WorkflowValidationError(
            f"routing_intent_keywords keys must be a subset of "
            f"routing_intent_tags in {mode_config_path}: "
            f"unknown tags {sorted(_orphan_keyword_tags)}"
        )


def _parse_mode_routing_fields(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> _ModeRoutingFields:
    """Parse routing fields (routable/priority/fast paths/intent tags) from mode front matter.

    Args:
        mode_front_matter: Parsed (override-merged) mode front matter.
        mode_config_path: Path of ``_mode.md`` (for error messages).

    Returns:
        The parsed routing fields.

    Raises:
        WorkflowValidationError: On invalid field values or orphaned
            routing_intent_keywords tags.
    """
    routable = mode_front_matter.get("routable", False)
    if not isinstance(routable, bool):
        raise WorkflowValidationError(f"routable must be a boolean in {mode_config_path}")
    priority = mode_front_matter.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise WorkflowValidationError(f"priority must be an integer in {mode_config_path}")
    fast_path_patterns_raw = mode_front_matter.get("fast_path_patterns")
    if fast_path_patterns_raw is None:
        fast_path_patterns_raw = []
    if not isinstance(fast_path_patterns_raw, list):
        raise WorkflowValidationError(f"fast_path_patterns must be a list in {mode_config_path}")
    fast_path_patterns: list[str] = []
    for idx, raw_pattern in enumerate(fast_path_patterns_raw):
        if not isinstance(raw_pattern, str):
            raise WorkflowValidationError(
                f"fast_path_patterns[{idx}] must be a string in {mode_config_path}"
            )
        pattern = raw_pattern.strip()
        if not pattern:
            raise WorkflowValidationError(
                f"fast_path_patterns[{idx}] cannot be empty in {mode_config_path}"
            )
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise WorkflowValidationError(
                f"fast_path_patterns[{idx}] is not a valid regex in {mode_config_path}: {exc}"
            ) from exc
        fast_path_patterns.append(pattern)

    # Parse routing_intent_tags: declarative intent coverage for proactive
    # routing hints. Each tag is a lowercase string (e.g. "research",
    # "reasoning", "remember", "write"). The hint generator checks whether
    # a single routing target's tags cover all detected query intents,
    # preferring it over a fork fan-out when they do.
    routing_intent_tags = _parse_lowercase_tag_list(
        mode_front_matter, "routing_intent_tags", mode_config_path
    )

    # Parse routing_intent_keywords: per-tag substring keywords for the
    # proactive routing hint generator. Maps each intent tag to the
    # substring keywords that signal that specific intent. These are
    # case-insensitive substring matches (not anchored regexes).
    routing_intent_keywords = _parse_routing_intent_keywords(mode_front_matter, mode_config_path)

    # Validate that every routing_intent_keywords key is declared in
    # routing_intent_tags. Fail fast at catalog load so a broken pack
    # install cannot reach the router (see
    # _validate_routing_intent_coverage for the incident history).
    _validate_routing_intent_coverage(
        routing_intent_tags, routing_intent_keywords, mode_config_path
    )

    # Parse satisfies_intents: intents this mode handles in-place as the
    # *current* mode, so the proactive routing hint generator knows not
    # to suggest a handoff for them. Symmetric counterpart to
    # ``routing_intent_tags`` (which declares coverage when routing *to*
    # the mode); both share the same pack-authored intent-tag vocabulary.
    # The kernel never hardcodes mode-specific intent semantics or tool
    # names — it only reads the mode's declared tags. Default empty means
    # every detected intent is a gap (safe direction for modes that have
    # not opted in, e.g. ``default``).
    satisfies_intents = _parse_lowercase_tag_list(
        mode_front_matter, "satisfies_intents", mode_config_path
    )

    # Parse required_capabilities: ``ModelCapability`` strings the resolved
    # model must declare for this mode to run safely. Mirrors
    # ``satisfies_intents`` parsing. Default empty means no capability gate
    # (backward compatible). The kernel never hardcodes specific capability
    # names — they are pack-authored vocabulary matching ``ModelCapability``
    # (text, tools, code, reasoning, ...).
    required_capabilities = _parse_lowercase_tag_list(
        mode_front_matter, "required_capabilities", mode_config_path
    )

    return _ModeRoutingFields(
        routable=routable,
        priority=priority,
        fast_path_patterns=fast_path_patterns,
        routing_intent_tags=routing_intent_tags,
        routing_intent_keywords=routing_intent_keywords,
        satisfies_intents=satisfies_intents,
        required_capabilities=required_capabilities,
    )


def _parse_completion_budget_tokens(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> tuple[int, int]:
    """Parse the mode's budget declarations (BUDGET_SCOPES_RFC scopes 2+3).

    Returns ``(completion_budget_tokens, turn_completion_budget_tokens)`` —
    the per-call work-shape budget and the whole-turn cumulative rail.
    0/absent for either = inherit (backward compatible). Declared values
    must be positive integers — a typo'd budget silently inheriting the
    settings value would defeat the declaration, so invalid values fail
    catalog load loudly.
    """
    per_call = 0
    raw = mode_front_matter.get("completion_budget_tokens")
    if raw is not None:
        per_call = _coerce_positive_budget_int(raw, "completion_budget_tokens", mode_config_path)
    per_turn = 0
    raw_turn = mode_front_matter.get("turn_completion_budget_tokens")
    if raw_turn is not None:
        per_turn = _coerce_positive_budget_int(
            raw_turn, "turn_completion_budget_tokens", mode_config_path
        )
    return per_call, per_turn


def _coerce_positive_budget_int(raw: Any, field_name: str, mode_config_path: Path) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError(
            f"{field_name} must be a positive integer in {mode_config_path}, got {raw!r}"
        ) from exc
    if value <= 0:
        raise WorkflowValidationError(
            f"{field_name} must be a positive integer in {mode_config_path}, got {value}"
        )
    return value


def _parse_default_intent_bridges(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> list[DefaultIntentBridge]:
    """Parse the ``default_intent_bridges`` list from mode front matter.

    Args:
        mode_front_matter: Parsed (override-merged) mode front matter.
        mode_config_path: Path of ``_mode.md`` (for error messages).

    Returns:
        The parsed DefaultIntentBridge list (empty when absent).

    Raises:
        WorkflowValidationError: On invalid bridge config entries.
    """
    default_intent_bridges: list[DefaultIntentBridge] = []
    bridges_raw = mode_front_matter.get("default_intent_bridges") or []
    if not isinstance(bridges_raw, list):
        raise WorkflowValidationError(
            "default_intent_bridges must be a list, got "
            f"{type(bridges_raw).__name__} in {mode_config_path}"
        )
    for idx, bridge_config in enumerate(bridges_raw):
        if not isinstance(bridge_config, dict):
            raise WorkflowValidationError(
                f"default_intent_bridges[{idx}] must be a dict, got "
                f"{type(bridge_config).__name__} in {mode_config_path}"
            )
        target_mode_id = _require_str_field(
            bridge_config,
            "target_mode_id",
            mode_config_path,
            location=f"default_intent_bridges[{idx}]",
        )
        matcher_patterns = _require_list_field(
            bridge_config,
            "matcher_patterns",
            mode_config_path,
            location=f"default_intent_bridges[{idx}]",
        )
        matcher_patterns = [str(p) for p in matcher_patterns]
        query_extraction_strategy = str(
            bridge_config.get("query_extraction_strategy", "passthrough")
        )
        prerequisites = [str(p) for p in (bridge_config.get("prerequisites") or [])]
        recent_handoff_source_modes = [
            str(source_mode)
            for source_mode in (bridge_config.get("recent_handoff_source_modes") or [])
        ]
        recent_handoff_required_artifact_keys = [
            str(artifact_key)
            for artifact_key in (bridge_config.get("recent_handoff_required_artifact_keys") or [])
        ]
        recent_handoff_context_key = str(bridge_config.get("recent_handoff_context_key", "") or "")
        response_appendix = str(bridge_config.get("response_appendix", ""))

        default_intent_bridges.append(
            DefaultIntentBridge(
                target_mode_id=target_mode_id,
                matcher_patterns=matcher_patterns,
                query_extraction_strategy=query_extraction_strategy,
                prerequisites=prerequisites,
                recent_handoff_source_modes=recent_handoff_source_modes,
                recent_handoff_required_artifact_keys=recent_handoff_required_artifact_keys,
                recent_handoff_context_key=recent_handoff_context_key,
                response_appendix=response_appendix,
            )
        )
    return default_intent_bridges


def _parse_yield_and_fork_policy(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> tuple[YieldPolicy, bool]:
    """Parse ``yield_policy`` and ``fork_allowed`` from mode front matter.

    Args:
        mode_front_matter: Parsed (override-merged) mode front matter.
        mode_config_path: Path of ``_mode.md`` (for error messages).

    Returns:
        A ``(yield_policy, fork_allowed)`` tuple.

    Raises:
        WorkflowValidationError: If either field has an invalid type.
    """
    # Parse yield_policy: the per-mode policy for the unified turn-lifecycle
    # ``yield`` action. Authored in ``_mode.md`` front matter as a dict with
    # keys ``allowed`` (bool), ``fallback_mode`` (str),
    # ``is_default_yield_target`` (bool), and ``default_question`` (str). See
    # :class:`kernel.models.YieldPolicy` and :class:`kernel.models.YieldAction`.
    # Default-empty (YieldPolicy()) means inline asks allowed, no fallback,
    # not the default target — safe for chat modes; action modes override
    # with ``allowed: false`` + a ``fallback_mode`` (typically ``discuss``).
    yield_policy_raw = mode_front_matter.get("yield_policy") or {}
    if not isinstance(yield_policy_raw, dict):
        raise WorkflowValidationError(f"yield_policy must be a mapping in {mode_config_path}")
    yield_policy = YieldPolicy(
        allowed=bool(yield_policy_raw.get("allowed", True)),
        fallback_mode=str(yield_policy_raw.get("fallback_mode", "") or ""),
        is_default_yield_target=bool(yield_policy_raw.get("is_default_yield_target", False)),
        default_question=str(yield_policy_raw.get("default_question", "") or ""),
    )
    # Parse fork_allowed: a boolean consent flag. When True, this mode
    # agrees to be a fork target — it commits to completing or failing
    # without asking the user. When absent (default False), the routing
    # fork decision and the fork enforcer refuse to fork to this mode;
    # it is reachable only via delegate or single-handoff.
    fork_allowed_raw = mode_front_matter.get("fork_allowed", False)
    if not isinstance(fork_allowed_raw, bool):
        raise WorkflowValidationError(f"fork_allowed must be a boolean in {mode_config_path}")
    fork_allowed = fork_allowed_raw
    return yield_policy, fork_allowed


def _parse_approval_continuation(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> Literal["report", "resume"]:
    """Parse ``approval_continuation`` (``report`` default, or ``resume``)."""
    raw = mode_front_matter.get("approval_continuation", "report")
    if raw == "report":
        return "report"
    if raw == "resume":
        return "resume"
    raise WorkflowValidationError(
        f"approval_continuation must be 'report' or 'resume' in {mode_config_path}"
    )


class _ModeFolderSkipped(Exception):
    """Internal control-flow sentinel: non-strict mode folder missing ``_mode.md``.

    Raised by :func:`_load_mode_from_dir` under ``strict=False`` so the scan
    loop in :func:`load_catalog_from_disk` can continue to the next directory
    without embedding the skip decision in the per-mode loader.
    """


def _load_mode_from_dir(
    mode_dir: Path,
    *,
    strict: bool,
) -> WorkflowModeConfig:
    """Load one workflow mode from its directory into a WorkflowModeConfig.

    Reads and parses ``_mode.md``, applies optional ``_overrides.yaml``,
    loads all authored stage files, and parses every front-matter section
    (operation bindings, routing fields, intent bridges, yield policy) into
    typed fields.

    Args:
        mode_dir: Directory containing ``_mode.md`` and stage files.
        strict: When True, missing ``_mode.md`` raises instead of warning
            and skipping the folder.

    Returns:
        The loaded mode config (stage graph synthesized for stageless kinds).

    Raises:
        WorkflowValidationError: On missing required front-matter fields or
            invalid values.
        _ModeFolderSkipped: Under ``strict=False`` when the folder has no
            ``_mode.md`` (the scan loop logs and continues).
    """
    mode_config_path = mode_dir / "_mode.md"
    if not mode_config_path.exists():
        if strict:
            raise WorkflowValidationError(
                f"Mode folder '{mode_dir.name}' is missing _mode.md — a mode "
                "without _mode.md is not a valid pack install."
            )
        logger.warning("Skipping instructions folder '%s': missing _mode.md", mode_dir.name)
        raise _ModeFolderSkipped(mode_dir.name)

    mode_front_matter, mode_body = parse_front_matter(mode_config_path.read_text(encoding="utf-8"))
    # Apply host/pack-authored front-matter overrides from a sibling
    # ``_overrides.yaml`` file so apps can manage extension front matter
    # (e.g. lens specs) without editing pack-owned ``_mode.md`` files.
    # Overrides are deep-merged on top of the authored front matter; the
    # override file is optional and may be absent. The kernel protects its
    # own owned keys and passes everything else through opaquely.
    mode_front_matter = _apply_mode_overrides(mode_dir, mode_front_matter)

    mode_id = _require_str_field(mode_front_matter, "mode_id", mode_config_path)
    mode_kind = cast(
        ModeKind,
        _require_str_field(mode_front_matter, "mode_kind", mode_config_path),
    )
    mode_description = _require_str_field(mode_front_matter, "description", mode_config_path)
    # command_pattern can be empty for fallback chat modes (e.g. default),
    # but the key must exist in front-matter for explicitiveness.
    raw_pattern = _require_field(mode_front_matter, "command_pattern", mode_config_path)
    command_pattern = str(raw_pattern)
    require_query = bool(mode_front_matter.get("require_query", False))
    mode_default_model = str(mode_front_matter.get("default_model", "") or "") or None
    operation_bindings = _parse_mode_operation_bindings(
        mode_front_matter, mode_kind, mode_config_path
    )
    routing = _parse_mode_routing_fields(mode_front_matter, mode_config_path)
    _mode_budget_tokens = _parse_completion_budget_tokens(mode_front_matter, mode_config_path)
    default_intent_bridges = _parse_default_intent_bridges(mode_front_matter, mode_config_path)
    yield_policy, fork_allowed = _parse_yield_and_fork_policy(mode_front_matter, mode_config_path)
    approval_continuation = _parse_approval_continuation(mode_front_matter, mode_config_path)

    # Collect app/pack-owned extension keys the kernel does not interpret.
    # Everything in the (possibly overridden) front matter that is not a
    # kernel-owned field flows through opaquely as ``extra_front_matter``.
    # This keeps the boundary honest: the kernel names only what it owns and
    # never has to allowlist or curate app-owned concepts like ``lenses``.
    extra_front_matter = {
        key: value
        for key, value in mode_front_matter.items()
        if key not in _KERNEL_OWNED_FRONT_MATTER_KEYS
    }

    # Evaluate conditional operation blocks (``<!-- if operation:<id> -->`` ...
    # ``<!-- endif -->``) against the installed operation catalog so prose
    # for uninstalled operations never reaches the model's prompt. The raw
    # body is retained on the mode for the binding-warning downgrade.
    instruction_body, instruction_body_raw = _evaluate_mode_conditionals(mode_id, mode_body.strip())

    return WorkflowModeConfig(
        mode_id=mode_id,
        mode_kind=mode_kind,
        artifact_type=str(mode_front_matter.get("artifact_type", "")),
        description=mode_description,
        instruction_body=instruction_body,
        instruction_body_raw=instruction_body_raw,
        mode_markdown_path=str(mode_config_path),
        default_model=mode_default_model,
        routable=routing.routable,
        priority=routing.priority,
        fast_path_patterns=routing.fast_path_patterns,
        routing_intent_tags=routing.routing_intent_tags,
        routing_intent_keywords=routing.routing_intent_keywords,
        satisfies_intents=routing.satisfies_intents,
        required_capabilities=routing.required_capabilities,
        completion_budget_tokens=_mode_budget_tokens[0],
        turn_completion_budget_tokens=_mode_budget_tokens[1],
        command_pattern=command_pattern,
        require_query=require_query,
        entry_operation_ids=operation_bindings.entry_operation_ids,
        entry_op_kind=operation_bindings.entry_op_kind,
        pre_turn_operation_ids=operation_bindings.pre_turn_operation_ids,
        post_turn_operation_ids=operation_bindings.post_turn_operation_ids,
        allowed_operation_ids=operation_bindings.allowed_operation_ids,
        allowed_tool_ids=operation_bindings.allowed_tool_ids,
        authored_allowed_operation_ids=operation_bindings.authored_allowed_operation_ids,
        authored_allowed_tool_ids=operation_bindings.authored_allowed_tool_ids,
        tools_authoritative=operation_bindings.tools_authoritative,
        accept_default_capabilities=operation_bindings.accept_default_capabilities,
        excluded_operation_ids=operation_bindings.excluded_operation_ids,
        excluded_tool_ids=operation_bindings.excluded_tool_ids,
        default_intent_bridges=default_intent_bridges,
        mode_provider_id=str(mode_front_matter.get("mode_provider_id", "") or ""),
        context_item_packs=[str(p) for p in (mode_front_matter.get("context_item_packs") or [])],
        yield_policy=yield_policy,
        fork_allowed=fork_allowed,
        approval_continuation=approval_continuation,
        task_completion=mode_front_matter.get("task_completion", "none"),
        task_artifact_tool_id=str(mode_front_matter.get("task_artifact_tool_id", "") or ""),
        extra_front_matter=extra_front_matter,
    )


def _evaluate_mode_conditionals(mode_id: str, raw_body: str) -> tuple[str, str]:
    """Evaluate conditional operation blocks in one mode's instruction body.

    Returns ``(instruction_body, instruction_body_raw)``. The raw body is
    non-empty only when conditional blocks were actually evaluated and the
    strip changed the body; when operation callbacks are not registered (a
    host that never dispatches operations), the body passes through
    unchanged so the disk loader keeps its legacy behavior.
    """
    if not has_conditional_markers(raw_body):
        return raw_body, ""
    try:
        available_ids = set(_get_operation_catalog())
    except RuntimeError:
        logger.debug(
            "Mode '%s': operation catalog unavailable; conditional operation "
            "blocks left unevaluated.",
            mode_id,
        )
        return raw_body, ""
    stripped = strip_conditional_operation_blocks(raw_body, available_ids)
    if stripped == raw_body:
        return raw_body, ""
    missing = sorted(conditional_operation_ids(raw_body) - available_ids)
    if missing:
        logger.debug(
            "Mode '%s': stripped conditional blocks for missing operations: %s",
            mode_id,
            ", ".join(missing),
        )
    return stripped, raw_body


def _apply_mode_conditionals(catalog: dict[str, WorkflowModeConfig]) -> None:
    """Evaluate conditional operation blocks across a loaded mode catalog.

    MCP-path counterpart of the strip in :func:`_load_mode_from_dir`: the
    catalog arrives deserialized from the ``lisa-catalog`` server, so the
    conditional blocks are evaluated here, once, before validation runs.
    """
    try:
        available_ids = set(_get_operation_catalog())
    except RuntimeError:
        logger.debug(
            "Operation catalog unavailable; conditional operation blocks left unevaluated.",
        )
        return
    for mode in catalog.values():
        raw_body = mode.instruction_body
        if not has_conditional_markers(raw_body):
            continue
        stripped = strip_conditional_operation_blocks(raw_body, available_ids)
        if stripped == raw_body:
            continue
        mode.instruction_body_raw = raw_body
        mode.instruction_body = stripped
        missing = sorted(conditional_operation_ids(raw_body) - available_ids)
        if missing:
            logger.debug(
                "Mode '%s': stripped conditional blocks for missing operations: %s",
                mode.mode_id,
                ", ".join(missing),
            )


def load_catalog_from_disk(
    instructions_dir: Path | None = None,
    *,
    strict: bool = True,
) -> dict[str, WorkflowModeConfig]:
    """Discover and load all workflow modes from the instructions directory.

    Scans `instructions_dir` (default: injected baseline mode root) for subdirectories.
    Each subdirectory becomes one workflow mode.  The discovery contract is:

    {mode_name}/_mode.md       — required; front-matter must include:
                        mode_id, start_stage_id.  It may also
                        declare mode_kind, command_pattern,
                        require_query, and operation bindings.
      {mode_name}/{stage_id}.md  — one file per stage; front-matter must include:
                                    stage_id, terminal, allowed_next_stages,
                                    allow_user_questions, required_connectors.

    When *strict* is ``True`` (the default), a mode folder missing ``_mode.md``
    raises :class:`WorkflowValidationError` instead of logging a warning and
    skipping. This catches broken pack installs at startup. Tests that
    exercise partial installs pass ``strict=False``.

    Stage files whose name starts with `_` are skipped (reserved for mode-level
    config files like `_mode.md`).

    The body text below each stage's front-matter delimiter is stored in
    WorkflowStageConfig.instruction_body so runtime prompt injection never
    needs a disk read after startup.

    Args:
        instructions_dir: Override the default scan root.  Useful in tests.

    Returns:
        Dict mapping mode_id -> WorkflowModeConfig, not yet validated.
        Call validate_workflow_catalog() on the result before use.

    Raises:
        WorkflowValidationError: If a required front-matter field is missing
            or if the YAML is malformed.
    """
    root = instructions_dir or _INSTRUCTIONS_ROOT
    if root is None:
        raise RuntimeError(
            "Instructions root not configured. Pass instructions_dir explicitly "
            "or call register_instructions_root() at startup before using "
            "load_catalog_from_disk()."
        )
    catalog: dict[str, WorkflowModeConfig] = {}

    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue

        try:
            mode_config = _load_mode_from_dir(mode_dir, strict=strict)
        except _ModeFolderSkipped:
            continue
        mode_id = mode_config.mode_id

        resolved_provider_id = mode_config.mode_provider_id
        if not resolved_provider_id and (
            mode_config.entry_operation_ids
            or mode_config.pre_turn_operation_ids
            or mode_config.post_turn_operation_ids
            or mode_config.allowed_operation_ids
            or mode_config.allowed_tool_ids
        ):
            # Empty provider id with capability references: the export-boundary
            # validator will raise a loud OperationValidationError at startup.
            # Log here so the unresolved mode_id is visible in telemetry before
            # the validator runs, and the operator can correlate the failure to
            # the missing mode_provider_id authoring.
            logger.debug(
                "Mode '%s' has an empty mode_provider_id but references "
                "capabilities (%d ops, %d tools); export-boundary validation "
                "will reject it. Set mode_provider_id in the mode front matter.",
                mode_id,
                len(mode_config.allowed_operation_ids),
                len(mode_config.allowed_tool_ids),
            )
        catalog[mode_id] = mode_config

    return catalog


def _require_str_field(data: dict[str, Any], field: str, source: Path, location: str = "") -> str:
    """Extract a required string field from a front-matter dict.

    Args:
        data: Parsed front-matter dictionary.
        field: Field name that must be present and non-empty.
        source: File path, used only in the error message for debugging.
        location: Optional nested location (e.g. "default_intent_bridges[0]") for context.

    Returns:
        The string value for the given field.

    Raises:
        WorkflowValidationError: If the field is missing or empty.
    """
    value = data.get(field)
    if not value:
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter in '{source}'{loc_str} is missing required field '{field}'."
        )
    return str(value)


def _resolve_mode_relative_file(
    *,
    mode_dir: Path,
    authored_path: str,
    mode_config_path: Path,
    location: str,
) -> str:
    """Validate a mode-authored relative file path and return normalized relative path."""

    raw = authored_path.strip()
    if not raw:
        return ""

    candidate = Path(raw)
    if candidate.is_absolute():
        raise WorkflowValidationError(
            f"Front-matter field '{location}' in '{mode_config_path}' must be a relative path."
        )

    mode_root = mode_dir.resolve()
    resolved = (mode_root / candidate).resolve()
    if not resolved.is_relative_to(mode_root):
        raise WorkflowValidationError(
            f"Front-matter field '{location}' in '{mode_config_path}' "
            "must stay within the mode folder."
        )
    if not resolved.is_file():
        raise WorkflowValidationError(
            f"Front-matter field '{location}' in '{mode_config_path}' "
            f"points to a missing file '{candidate.as_posix()}'."
        )
    return resolved.relative_to(mode_root).as_posix()


def _require_field(data: dict[str, Any], field: str, source: Path) -> Any:
    """Extract one required front-matter field regardless of value type."""
    if field not in data:
        raise WorkflowValidationError(
            f"Front-matter in '{source}' is missing required field '{field}'."
        )
    return data[field]


def _require_bool_field(data: dict[str, Any], field: str, source: Path) -> bool:
    """Extract one required boolean front-matter field."""
    value = _require_field(data, field, source)
    if not isinstance(value, bool):
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}' must be a boolean."
        )
    return value


def _require_list_field(
    data: dict[str, Any], field: str, source: Path, location: str = ""
) -> list[str]:
    """Extract one required list[str] front-matter field."""
    if field not in data:
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter in '{source}'{loc_str} is missing required field '{field}'."
        )
    raw = data[field]
    if not isinstance(raw, list):
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}'{loc_str} must be a list."
        )
    return [str(item) for item in raw]


def _optional_int_field(
    data: dict[str, Any],
    field: str,
    source: Path,
    *,
    location: str = "",
    minimum: int = 0,
) -> int | None:
    """Extract one optional integer field with a minimum bound."""

    if field not in data:
        return None
    value = data[field]
    if not isinstance(value, int):
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}'{loc_str} must be an integer."
        )
    if value < minimum:
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}'{loc_str} must be >= {minimum}."
        )
    return value


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------
def _validate_fork_allowed(mode: WorkflowModeConfig) -> None:
    """Validate ``fork_allowed`` consistency for one mode.

    A mode that declares ``fork_allowed: true`` commits to completing or
    failing without asking the user. Three self-contradictions are rejected
    at load time so the FSM never starts with an unsound fork target:

    (a) No side-effecting capability — the completion contract needs at
        least one action tool to verify. A fork-allowed mode with zero
        authored operation/tool ids has nothing to verify.
    (b) Empty ``satisfies_intents`` — a fork target that satisfies nothing
        is routing-dead; the routing hint would never select it.
    (c) ``yield_policy.allowed: true`` — the mode claims it can complete
        without asking, but its foreground policy allows asking. This is
        a direct contradiction.

    Args:
        mode: The WorkflowModeConfig to validate.

    Raises:
        WorkflowValidationError: On any ``fork_allowed`` invariant violation.
    """
    if not mode.fork_allowed:
        return

    if not mode.authored_allowed_operation_ids and not mode.authored_allowed_tool_ids:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' declares fork_allowed: true but has no "
            "authored action tools — the completion contract needs at least "
            "one side-effecting capability to verify."
        )
    if not mode.satisfies_intents:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' declares fork_allowed: true but has "
            "empty satisfies_intents — a fork target must declare what it "
            "satisfies so the routing hint can select it."
        )
    if mode.yield_policy.allowed:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' declares fork_allowed: true but its "
            "yield_policy allows inline asks — a fork target must commit to "
            "completing without asking. Set yield_policy.allowed: false or "
            "remove fork_allowed."
        )


def _validate_mode_operation_bindings(mode: WorkflowModeConfig) -> None:
    """Validate that all operation IDs referenced by one mode exist in the operation catalog.

    Operations whose owning pack is not installed are silently dropped from
    the operation catalog during validation. When a mode references such an
    operation, we warn rather than raise — the mode remains available but the
    missing operation will fail at call time if invoked. This lets the rest of
    the workflow catalog load when an external pack's content root is present
    but its Python package isn't importable.
    """
    from kernel.operations.registry import is_known_tool

    catalog = _get_operation_catalog()
    # Conditional references (``<!-- if operation:<id> -->`` blocks stripped at
    # load time) are deliberate: the author acknowledged the operation may be
    # absent. Downgrade those to debug; unconditional references keep the
    # warning because the model still sees them in the prompt.
    conditional_ids = conditional_operation_ids(mode.instruction_body_raw)
    referenced_operation_ids = (
        mode.entry_operation_ids
        + mode.pre_turn_operation_ids
        + mode.post_turn_operation_ids
        + mode.allowed_operation_ids
    )
    for operation_id in referenced_operation_ids:
        if operation_id not in catalog and not is_known_tool(operation_id):
            if operation_id in conditional_ids and operation_id not in mode.instruction_body:
                logger.debug(
                    "Mode '%s' references operation '%s' only inside a conditional "
                    "block; the block was stripped because the operation is not "
                    "installed.",
                    mode.mode_id,
                    operation_id,
                )
                continue
            logger.warning(
                "Mode '%s' references operation '%s' which is not in the catalog "
                "(the owning pack may not be installed); the operation will fail "
                "if invoked at runtime.",
                mode.mode_id,
                operation_id,
            )


def validate_workflow_mode(mode: WorkflowModeConfig) -> None:
    """Validate one mode's structural invariants.

    Only ``chat`` and ``action`` remain. The retired ``workflow`` kind is
    rejected at load so a stale pack cannot silently carry a shape the kernel
    no longer honors (see CHANGELOG — workflow stages retired).
    """
    if mode.mode_kind not in {"chat", "action"}:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' has invalid mode_kind '{mode.mode_kind}'. "
            "The 'workflow' kind (staged FSM) was removed."
        )
    _validate_fork_allowed(mode)
    _validate_mode_operation_bindings(mode)


def validate_workflow_catalog(catalog: dict[str, WorkflowModeConfig]) -> None:
    """Validate all workflow modes in the catalog.

    Called at startup after catalog loading to catch misconfigured
    modes before any user traffic arrives.

    Args:
        catalog: Mapping of mode_id -> WorkflowModeConfig to validate.

    Raises:
        WorkflowValidationError: If any mode fails validation.
    """
    for mode in catalog.values():
        validate_workflow_mode(mode)


def get_routable_mode_ids(catalog: dict[str, WorkflowModeConfig] | None = None) -> list[str]:
    """Return the ordered list of routable mode IDs from the catalog.

    Routable modes are those with ``routable=True``. Ordered by priority
    (descending) then mode_id (ascending) for deterministic iteration.
    """
    if catalog is None:
        catalog = get_workflow_catalog()
    routable = [m for m in catalog.values() if m.routable]
    routable.sort(key=lambda m: (-m.priority, m.mode_id))
    return [m.mode_id for m in routable]


def is_routable_mode(target: str, catalog: dict[str, WorkflowModeConfig] | None = None) -> bool:
    """Return True when ``target`` is a routable mode_id in the catalog.

    Used by the transition_stage interception to distinguish a mode handoff
    target from a workflow stage transition target.
    """
    if catalog is None:
        catalog = get_workflow_catalog()
    mode = catalog.get(target)
    return mode is not None and mode.routable


def synthesize_routing_transitions(
    catalog: dict[str, WorkflowModeConfig] | None = None,
) -> dict[str, list[str]]:
    """Compute the routing FSM: which modes each mode can hand off to.

    Full routable mesh: every routable mode (including ``default``) can hand
    off to every other routable mode. This unblocks the proactive fork
    routing hint (see ``kernel/turn_runner.py ::
    _compute_capability_routing_hint``) from every entry mode the user can
    reach, not just ``default`` — e.g. a remember+write request issued from
    ``coding`` can fork to ``remember`` + ``markdown`` instead of falling back
    to the current mode's approval-gated write tool.

    Per-edge restrictions (e.g. forbidding ``search`` → ``coding``) are
    deferred to a later, smaller change — the default is permissive ("all
    work items on by default, externalize later").

    Returns a dict mapping mode_id → list of allowed handoff target mode_ids.
    """
    if catalog is None:
        catalog = get_workflow_catalog()
    routable_ids = get_routable_mode_ids(catalog)
    transitions: dict[str, list[str]] = {}
    for mode_id, mode_config in catalog.items():
        if mode_config.routable:
            # Every routable mode can hand off to every other routable mode.
            transitions[mode_id] = [m for m in routable_ids if m != mode_id]
        else:
            transitions[mode_id] = []
    return transitions


def get_routing_targets(
    mode_id: str,
    catalog: dict[str, WorkflowModeConfig] | None = None,
) -> list[str]:
    """Return the list of mode_ids that ``mode_id`` can hand off to.

    Returns an empty list when the mode has no routing targets.
    """
    transitions = synthesize_routing_transitions(catalog)
    return transitions.get(mode_id, [])


def get_transition_tool_schema(
    routing_targets: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Return the structured tool schema used for stage transitions.

    ``routing_targets`` are the mode ids the current mode may hand off to;
    they are listed verbatim so the model never guesses a mode id.
    """
    if routing_targets:
        targets = ", ".join(f"`{m}`" for m in routing_targets)
        mode_form = f"(1) one of these routable mode ids: {targets} to "
    else:
        mode_form = "(1) no mode handoff is available from this mode; "
    return [
        build_strict_openai_function_tool(
            name="transition_stage",
            description=(
                "Switch modes or advance a workflow stage. Use "
                "`target_stage_id` with one of two forms: "
                f"{mode_form}"
                "hand off the rest of this turn to that mode (use this "
                "when the current mode lacks the tools the request needs); "
                "(2) a workflow stage id to advance the current mode's "
                "FSM. Pass `delegated_goal` and `context_payload` to "
                "tell the receiving mode exactly what to do and what "
                "information it needs to start. "
                "When a Routing Hint is present in the system prompt, "
                "follow it by calling this tool with the suggested target. "
                "If user input is still missing, call `yield` to ask the "
                "user instead of calling this tool."
            ),
            parameters_schema=WorkflowTransitionRequest,
        )
    ]


_YIELD_TOOL_NAME = "yield"


def get_yield_tool_schema() -> list[dict[str, object]]:
    """Return the structured tool schema for the unified-lifecycle yield action.

    ``yield`` is a first-class terminal action: it pauses the turn, checkpoints
    durable state with ``lifecycle_state="yielded"``, and surfaces a question
    to the user. The user's reply resumes the turn against the same
    ``WorkflowInstance`` step. This makes "ask" distinguishable from "done"
    (both used to be "no tool calls" — the core bug the unified lifecycle
    fixes).

    The tool is advertised to the model alongside ``transition_stage``. When
    the model calls ``yield``, the turn loop treats it as terminal (like
    ``finish``) — it breaks the loop with the question as the turn content
    and stamps the checkpoint. The harness does NOT dispatch the tool to any
    executor; it intercepts the call in the tool-dispatch loop (same pattern
    as ``transition_stage``).
    """
    from kernel.models import YieldAction

    return [
        build_strict_openai_function_tool(
            name=_YIELD_TOOL_NAME,
            description=(
                "Pause the turn and ask the user a clarifying question. Use "
                "this when you need information or a decision from the user "
                "before you can continue — e.g. when a request is ambiguous, "
                "when a step stalled and needs a retry/skip decision, or when "
                "you have hit your iteration budget. The harness checkpoints "
                "state and surfaces your question; the user's reply resumes "
                "the turn. Do NOT use this to end the turn with a final "
                "answer — just respond with text for that. Do NOT narrate "
                "asking a question in prose — call this tool."
            ),
            parameters_schema=YieldAction,
        )
    ]


_workflow_catalog: dict[str, WorkflowModeConfig] | None = None


def _ensure_catalog_loaded() -> None:
    """Lazy-load the workflow catalog on first access via the lisa-catalog MCP server.

    Deferred from module-import time so the host app has a chance to register
    operation callbacks and the content MCP config before validation runs.
    """
    global _workflow_catalog  # noqa: PLW0603
    if _workflow_catalog is not None:
        return
    if _content_mcp_config is None:
        raise RuntimeError(
            "Content MCP config not registered. "
            "Call register_content_mcp() at startup before using the workflow catalog."
        )
    from kernel.context.mcp_catalog import MCPModeCatalog

    catalog_loader = MCPModeCatalog(_content_mcp_config)
    _workflow_catalog = catalog_loader.get_workflow_catalog()
    _apply_mode_conditionals(_workflow_catalog)
    validate_workflow_catalog(_workflow_catalog)
    # Enforce the cross-pack export boundary now that both the workflow catalog
    # and the operation/tool catalogs are loaded. Generic rule over
    # ``provider_id`` + ``exportable``; the kernel never names capability ids.
    from kernel.operations.registry import validate_export_boundaries

    validate_export_boundaries()


def get_workflow_catalog() -> dict[str, WorkflowModeConfig]:
    """Return the validated workflow mode catalog.

    The catalog is populated lazily on first access via the ``lisa-catalog`` MCP
    server. The host must call ``register_content_mcp()`` at startup before any
    code accesses the catalog. The catalog is loaded fresh on each host
    launch, so new or modified instruction folders appear on the next one.
    """
    _ensure_catalog_loaded()
    return _workflow_catalog  # type: ignore[return-value]


def get_mode_system_intro(mode_id: str) -> str:
    """Return the mode-level system prompt intro string for a given mode.

    Reads the instruction_body from the WorkflowModeConfig loaded at startup
    from the mode's _mode.md body. Descriptions are catalog metadata, not
    implicit model instructions. The string is injected ahead of the
    stage-level instructions.

    Returns an empty string for unknown modes so callers can safely include it
    without a mode membership check.

    Args:
        mode_id: The route mode string (e.g. 'coach', 'chase').

    Returns:
        The authored instruction body from the mode's _mode.md, or '' if the
        mode is unknown or has no instruction body.
    """
    mode = get_workflow_catalog().get(mode_id)
    if not mode:
        return ""
    return mode.instruction_body


def get_required_mode_intro(mode_id: str) -> str:
    """Return the mode-level system prompt intro, raising if it is missing.

    Unlike :func:`get_mode_system_intro`, this fails loudly when the mode is
    unknown or has an empty ``instruction_body``. Use this in the turn runner
    and any path where the mode instruction is required — not optional.

    Args:
        mode_id: The route mode string (e.g. 'coach', 'chase').

    Returns:
        The authored instruction body from the mode's ``_mode.md``.

    Raises:
        MissingModeInstructionError: If the mode is unknown or its
            ``instruction_body`` is empty.
    """
    mode = get_workflow_catalog().get(mode_id)
    if not mode:
        raise MissingModeInstructionError(
            f"Mode '{mode_id}' is not in the workflow catalog — cannot assemble "
            "a required mode instruction. Ensure the mode pack is installed and "
            "the catalog is loaded."
        )
    instruction_body = (mode.instruction_body or "").strip()
    if not instruction_body:
        raise MissingModeInstructionError(
            f"Mode '{mode_id}' has an empty instruction_body in its _mode.md — "
            "workflow modes require authored instructions. A mode with no "
            "instruction body would run the model modeless."
        )
    return instruction_body


def _build_tool_execution_context(
    target_id: str,
    ctx: OperationExecutionContext,
) -> ToolExecutionContext:
    """Derive the tool-registry execution context for a lifecycle target.

    Args:
        target_id: The tool id being dispatched.
        ctx: Shared operation execution context for the current mode turn.

    Returns:
        The ToolExecutionContext mirroring the shared turn context fields.
    """
    return ToolExecutionContext(
        tool_id=target_id,
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        surface_type=ctx.surface_type,
        surface_id=ctx.surface_id,
        memory_scope=ctx.memory_scope,
        mode_id=ctx.mode_id,
        query=ctx.query,
        pack_config=ctx.pack_config,
        capability_binding_provider=ctx.capability_binding_provider,
        approval_service=ctx.approval_service,
        approval_request_id=ctx.approval_request_id,
        task_id=ctx.task_id,
        session_plan_digest=ctx.session_plan_digest,
        code_change_artifact_store=ctx.code_change_artifact_store,
        event_bus=ctx.event_bus,
        state_provider=ctx.state_provider,
        data_dir=ctx.data_dir,
        turn_index=ctx.turn_index,
    )


def _resolve_turn_cap(
    target_id: str,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    execution_policy: ExecutionPolicy | None,
) -> int:
    """Resolve the per-turn invocation cap for a lifecycle target.

    Args:
        target_id: Operation or tool id whose ``max_calls_per_turn`` applies.
        operation_catalog: The operation catalog (capability definitions).
        tool_catalog: The tool catalog (capability definitions).
        execution_policy: Optional execution policy overriding catalog caps.

    Returns:
        The resolved cap (0 means uncapped).
    """
    from kernel.capability import resolve_capability_for_turn_limit

    definition = resolve_capability_for_turn_limit(
        target_id,
        operation_catalog=operation_catalog,
        tool_catalog=tool_catalog,
    )
    max_calls = 0
    if definition is not None:
        max_calls = int(getattr(definition, "max_calls_per_turn", 0) or 0)
    return resolve_capability_invocation_limit(
        execution_policy,
        target_id,
        max_calls,
    )


async def _dispatch_lifecycle_target(
    target_id: str,
    target_ctx: OperationExecutionContext,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
) -> OperationResult | None:
    """Dispatch one lifecycle target polymorphically (operation or tool).

    Args:
        target_id: Operation or tool id to dispatch.
        target_ctx: Execution context for the dispatch.
        operation_catalog: The operation catalog for resolution.
        tool_catalog: The tool catalog for resolution.

    Returns:
        The result, or ``None`` when the target is neither a known
        operation nor a known tool (logged and skipped by the caller).
    """
    if target_id in operation_catalog:
        return await _execute_operation(target_id, target_ctx)
    if target_id in tool_catalog:
        from kernel.tools.registry import execute_tool

        tool_ctx = _build_tool_execution_context(target_id, target_ctx)
        tool_result = await execute_tool(target_id, tool_ctx)
        return OperationResult(
            output_text=tool_result.output_text,
            output_json=dict(tool_result.output_json) if tool_result.output_json else {},
            extra={"tool_error": tool_result.error} if tool_result.error else {},
        )
    logger.warning(
        "lifecycle target '%s' is neither a known operation nor a known tool; skipping",
        target_id,
    )
    return None


def _lifecycle_cap_reached(
    target_id: str,
    ledger: CallLedger | None,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    execution_policy: ExecutionPolicy | None,
) -> bool:
    """Check whether a lifecycle target has exhausted its per-turn cap.

    Args:
        target_id: Operation or tool id to check.
        ledger: Per-turn call ledger; ``None`` disables the check.
        operation_catalog: The operation catalog (capability definitions).
        tool_catalog: The tool catalog (capability definitions).
        execution_policy: Optional execution policy overriding catalog caps.

    Returns:
        True when the target should be skipped (cap exhausted), False
        otherwise. Always False when no ledger is provided.
    """
    if ledger is None:
        return False
    max_calls = _resolve_turn_cap(
        target_id,
        operation_catalog,
        tool_catalog,
        execution_policy,
    )
    if ledger.is_exhausted(target_id, max_calls):
        logger.debug(
            "lifecycle target '%s' skipped: per-turn cap %d reached",
            target_id,
            max_calls,
        )
        return True
    return False


def _accumulate_lifecycle_result(
    target_id: str,
    result: OperationResult,
    output_parts: list[str],
    json_results: list[tuple[str, dict[str, Any]]],
    extra: dict[str, Any],
) -> None:
    """Fold one dispatched target's result into the shared accumulation buffers.

    Args:
        target_id: The target the result came from (keyed in JSON payloads).
        result: The target's dispatch result.
        output_parts: Text output accumulator (mutated in place).
        json_results: ``(target_id, payload)`` accumulator (mutated in place).
        extra: Extra-metadata accumulator (mutated in place).
    """
    if result.output_text:
        output_parts.append(result.output_text)
    if result.output_json:
        json_results.append((target_id, result.output_json))
    if result.extra:
        extra.update(result.extra)


def _merge_lifecycle_json_results(
    json_results: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Merge per-target JSON payloads into the combined result payload.

    A single-target payload passes through directly for convenience;
    multiple payloads are keyed by their target id.
    """
    if len(json_results) == 1:
        return json_results[0][1]
    return {target_id: payload for target_id, payload in json_results}


async def _execute_mode_target_ids(
    target_ids: list[str],
    ctx: OperationExecutionContext,
    *,
    failure_policy: Literal["abort", "skip"] = "abort",
    ledger: CallLedger | None = None,
    execution_policy: ExecutionPolicy | None = None,
) -> OperationResult:
    """Execute one ordered list of lifecycle target IDs and merge results.

    Each ID is resolved polymorphically: operations dispatch through the
    injected operation callback; tools dispatch through the tool registry.
    This dissolves the historical asymmetry where only operations could run
    in lifecycle slots. A deterministic compute step (tool) no longer needs an
    operation wrapper to participate in pre/post-turn or entry slots.

    Args:
        target_ids: Ordered operation OR tool identifiers declared by one mode.
        ctx: Shared execution context for the current mode turn.
        failure_policy: ``abort`` (default) re-raises on the first failing target;
            ``skip`` logs and continues, mirroring pre/post-turn isolation.
        ledger: Optional per-turn call ledger. When provided, each target is
            checked against the catalog's ``max_calls_per_turn`` before dispatch.

    Returns:
        Combined OperationResult. Text outputs are joined with blank lines, extra
        metadata is shallow-merged, and a single-target JSON payload is passed
        through directly for convenience.
    """
    if not target_ids:
        return OperationResult()

    from kernel.operations.registry import get_operation_catalog
    from kernel.tools.registry import get_tool_catalog

    operation_catalog = get_operation_catalog()
    tool_catalog = get_tool_catalog()

    output_parts: list[str] = []
    extra: dict[str, Any] = dict(ctx.extra)
    json_results: list[tuple[str, dict[str, Any]]] = []

    for target_id in target_ids:
        # Per-turn cap enforcement (uniform across operations and tools).
        if _lifecycle_cap_reached(
            target_id, ledger, operation_catalog, tool_catalog, execution_policy
        ):
            continue

        target_ctx = ctx.model_copy(update={"operation_id": target_id, "extra": extra})
        try:
            result = await _dispatch_lifecycle_target(
                target_id, target_ctx, operation_catalog, tool_catalog
            )
            if result is None:
                continue
        except ApprovalRequired:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            if failure_policy == "abort":
                raise
            logger.warning(
                "lifecycle target '%s' failed (skip policy): %s: %s",
                target_id,
                type(exc).__name__,
                exc,
            )
            continue

        if ledger is not None:
            ledger.record(target_id)
        _accumulate_lifecycle_result(target_id, result, output_parts, json_results, extra)

    output_json = _merge_lifecycle_json_results(json_results)

    return OperationResult(
        output_text="\n\n".join(output_parts),
        output_json=output_json,
        extra=extra,
    )


async def run_mode_entry_operations(
    mode_id: str,
    ctx: OperationExecutionContext,
    *,
    ledger: CallLedger | None = None,
    execution_policy: ExecutionPolicy | None = None,
) -> OperationResult:
    """Execute the entry targets configured for one action mode.

    Resolves each entry ID polymorphically (operation or tool). The mode's
    ``entry_failure_policy`` controls whether a failing target aborts the
    mode transition (``abort``, the historical default) or is logged and
    skipped (``skip``, matching pre/post-turn isolation).
    """
    mode = get_workflow_catalog().get(mode_id)
    if mode is None:
        return OperationResult()
    return await _execute_mode_target_ids(
        mode.entry_operation_ids,
        ctx,
        failure_policy=mode.entry_failure_policy,
        ledger=ledger,
        execution_policy=execution_policy,
    )


async def run_mode_pre_turn_operations(
    mode_id: str,
    ctx: OperationExecutionContext,
    *,
    ledger: CallLedger | None = None,
    execution_policy: ExecutionPolicy | None = None,
) -> OperationResult:
    """Execute pre-turn targets for one workflow or chat mode.

    Resolves each ID polymorphically (operation or tool). Failures are always
    isolated: a flaky enrichment target logs a warning and the turn continues.
    """
    mode = get_workflow_catalog().get(mode_id)
    if mode is None:
        return OperationResult()
    if mode.pre_turn_operation_ids:
        try:
            return await _execute_mode_target_ids(
                mode.pre_turn_operation_ids,
                ctx,
                failure_policy="skip",
                ledger=ledger,
                execution_policy=execution_policy,
            )
        except ApprovalRequired:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("pre_turn targets failed for mode '%s': %s", mode_id, exc)
    return OperationResult()


async def run_mode_post_turn_operations(
    mode_id: str,
    ctx: OperationExecutionContext,
    *,
    ledger: CallLedger | None = None,
    execution_policy: ExecutionPolicy | None = None,
) -> OperationResult:
    """Execute post-turn targets for one workflow or chat mode.

    Resolves each ID polymorphically (operation or tool). Failures are always
    isolated: a flaky bookkeeping target logs a warning and the turn is
    considered successful.
    """
    mode = get_workflow_catalog().get(mode_id)
    if mode is None:
        return OperationResult()
    if mode.post_turn_operation_ids:
        try:
            return await _execute_mode_target_ids(
                mode.post_turn_operation_ids,
                ctx,
                failure_policy="skip",
                ledger=ledger,
                execution_policy=execution_policy,
            )
        except ApprovalRequired:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("post_turn targets failed for mode '%s': %s", mode_id, exc)
    return OperationResult()
