"""Test helper: fold flat ``TurnHooks`` keyword overrides into grouped forms.

Test suites historically built ``TurnHooks`` with flat keyword args
(``TurnHooks(complete=..., mcp_tool_pool=..., memory_recall_enabled=False)``).
After the hard-cutover to grouped sub-dataclasses, callers pass the *names*
as keyword overrides and this helper folds them into the right group — every
group class is applied, and unrecognized keys pass through to ``TurnHooks``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kernel.turn_hooks import (
    ContextHooks,
    HostServiceHandles,
    LifecycleHooks,
    ModelHooks,
    OperationHooks,
    RecallHooks,
    ToolHooks,
    TurnHooks,
    TurnIdentity,
    WorkflowHooks,
)


@dataclass(frozen=True)
class _FoldMap:
    field_to_group: dict[str, str]
    group_cls: dict[str, Any]
    default_kwargs: dict[str, dict[str, Any]]


_MAP = _FoldMap(
    field_to_group={
        "resolve_model": "models",
        "complete": "models",
        "complete_stream": "models",
        "assemble_system_prompt": "context",
        "assemble_user_context": "context",
        "last_context_snapshot": "context",
        "execute_operation": "operations",
        "delegate_fn": "operations",
        "fork_fn": "operations",
        "on_turn_complete": "lifecycle",
        "slice_tuning": "lifecycle",
        "pre_turn_slice": "lifecycle",
        "post_turn_slice": "lifecycle",
        "explicit_slice": "lifecycle",
        "last_slice_finished_at": "lifecycle",
        "on_stage": "lifecycle",
        "operator_signal": "lifecycle",
        "introspect": "lifecycle",
        "workflow_position": "workflow",
        "workflow_position_updater": "workflow",
        "route_coordinator": "workflow",
        "state_provider": "services",
        "pack_configuration_resolver": "services",
        "approval_service": "services",
        "recovery_service": "services",
        "code_change_artifact_store": "services",
        "data_dir": "services",
        "workspace_id": "identity",
        "user_id": "identity",
        "user_slug": "identity",
        "user_name": "identity",
        "principal_key": "identity",
        "conversation_id": "identity",
        "surface_type": "identity",
        "surface_id": "identity",
        "memory_scope": "identity",
        "session_plan_digest": "identity",
        "memory_recall_enabled": "recall",
        "memory_recall_pre_turn_limit": "recall",
        "memory_recall_min_query_chars": "recall",
        "mcp_tool_servers": "tools",
        "mcp_tool_pool": "tools",
    },
    group_cls={
        "models": ModelHooks,
        "context": ContextHooks,
        "operations": OperationHooks,
        "lifecycle": LifecycleHooks,
        "workflow": WorkflowHooks,
        "services": HostServiceHandles,
        "identity": TurnIdentity,
        "recall": RecallHooks,
        "tools": ToolHooks,
    },
    default_kwargs={  # per-group default kwarg renames to fold
        "recall": {
            "memory_recall_enabled": "enabled",
            "memory_recall_pre_turn_limit": "limit",
            "memory_recall_min_query_chars": "min_query_chars",
        },
    },
)
_RECALL_FIELD_RENAME = {
    "memory_recall_enabled": "enabled",
    "memory_recall_pre_turn_limit": "limit",
    "memory_recall_min_query_chars": "min_query_chars",
}


def build_hooks(defaults: dict[str, dict[str, Any]] | None = None, **overrides: Any) -> TurnHooks:
    """Build a grouped ``TurnHooks`` from flat keyword overrides.

    ``defaults`` optionally seeds per-group keyword dicts (e.g.
    ``{"models": {"complete": fn}, "recall": {"enabled": False}}``).
    ``overrides`` supplies flat field names; recognized flat names are
    folded into their groups, and unrecognized keys forward to
    ``TurnHooks`` (e.g. ``workflow``, ``context_seams``).
    """
    merged: dict[str, dict[str, Any]] = {
        g: dict((defaults or {}).get(g) or {}) for g in _MAP.group_cls
    }
    leftover: dict[str, Any] = {}
    for key, value in overrides.items():
        group = _MAP.field_to_group.get(key)
        if group is None:
            leftover[key] = value
            continue
        if group == "recall":
            merged["recall"][_RECALL_FIELD_RENAME[key]] = value
        else:
            merged[group][key] = value
    return TurnHooks(
        **{g: cls(**merged[g]) for g, cls in _MAP.group_cls.items() if merged[g]},
        **leftover,
    )
