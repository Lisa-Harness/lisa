"""Single resolution layer for capability ids across operation and tool catalogs.

The kernel resolves a capability id (``operation_id`` or ``tool_id``) against
two in-process catalogs: ``get_operation_catalog()`` and ``get_tool_catalog()``.
Before this module existed, the resolution pattern
``operation_catalog.get(x) or tool_catalog.get(x)`` was copy-pasted at every
call site (``ipsar_fsm``, ``turn_runner``, the operation registry's schema
builder). That made resolution-order changes fragile and left no single place
to add cross-cutting concerns (telemetry, normalized keying, audit).

This module owns the canonical resolution contract:

* ``resolve_capability(id)`` returns the first matching definition, with
  operations taking precedence over tools on id collision. This matches the
  order every existing call site used, so behavior is unchanged.
* ``resolve_operation_definition`` / ``resolve_tool_definition`` are the
  per-catalog entry points for callers that know which catalog they want.
* ``resolve_capability_for_turn_limit`` is the dedicated entry point for
  per-turn ``max_calls_per_turn`` enforcement, which previously inlined the
  dual-catalog fallback with a ``getattr(..., "max_calls_per_turn", 0)``
  tail at four sites.

Internal pack tools are keyed **bare** in the tool catalog
(e.g. ``web_search``). MCP-native tools are keyed **namespaced**
(``server_id.tool``, e.g. ``lisa-storage.read_model_manifest``); the MCP
client pool owns that namespace bridging separately (see
``kernel.tools.mcp_client``). This module resolves against the kernel
catalog only — the MCP pool's namespaced index is not consulted here, by
design, so resolution never spawns a subprocess.

Called by: kernel.ipsar_fsm (lifecycle target dispatch)
           kernel.turn_runner (tool-call dispatch, per-turn cap enforcement)
           kernel.operations.registry (schema building)
"""

from __future__ import annotations

from typing import Union

from kernel.models import OperationDefinition, ToolDefinition

__all__ = [
    "CapabilityDefinition",
    "resolve_capability",
    "resolve_capability_for_turn_limit",
    "resolve_operation_definition",
    "resolve_tool_definition",
]

CapabilityDefinition = Union[OperationDefinition, ToolDefinition]


def resolve_operation_definition(
    capability_id: str,
    *,
    operation_catalog: dict[str, OperationDefinition] | None = None,
) -> OperationDefinition | None:
    """Look up one operation id in the operation catalog.

    Args:
        capability_id: The bare ``operation_id`` to resolve.
        operation_catalog: Optional pre-resolved catalog. When omitted, the
            kernel's ``get_operation_catalog()`` is used.

    Returns:
        The matching ``OperationDefinition`` or ``None``.
    """
    catalog = (
        operation_catalog
        if operation_catalog is not None
        else _default_operation_catalog()
    )
    return catalog.get(capability_id)


def resolve_tool_definition(
    capability_id: str,
    *,
    tool_catalog: dict[str, ToolDefinition] | None = None,
) -> ToolDefinition | None:
    """Look up one tool id in the tool catalog.

    Args:
        capability_id: The tool id to resolve. Internal pack tools are keyed
            bare (e.g. ``web_search``); MCP-native tools are keyed namespaced
            (e.g. ``lisa-storage.read_model_manifest``) but are owned by the
            MCP client pool, not the kernel tool catalog.
        tool_catalog: Optional pre-resolved catalog. When omitted, the
            kernel's ``get_tool_catalog()`` is used.

    Returns:
        The matching ``ToolDefinition`` or ``None``.
    """
    catalog = tool_catalog if tool_catalog is not None else _default_tool_catalog()
    return catalog.get(capability_id)


def resolve_capability(
    capability_id: str,
    *,
    operation_catalog: dict[str, OperationDefinition] | None = None,
    tool_catalog: dict[str, ToolDefinition] | None = None,
) -> CapabilityDefinition | None:
    """Resolve a capability id against operations first, then tools.

    On id collision, operations take precedence over tools. This matches the
    order every existing inlined call site used
    (``operation_catalog.get(x) or tool_catalog.get(x)``), so behavior is
    unchanged. Callers that need a specific catalog should use
    ``resolve_operation_definition`` or ``resolve_tool_definition`` instead.

    Args:
        capability_id: The ``operation_id`` or ``tool_id`` to resolve.
        operation_catalog: Optional pre-resolved operation catalog.
        tool_catalog: Optional pre-resolved tool catalog.

    Returns:
        The matching ``OperationDefinition`` or ``ToolDefinition``, or
        ``None`` when the id is unknown to both catalogs.
    """
    op = resolve_operation_definition(
        capability_id, operation_catalog=operation_catalog
    )
    if op is not None:
        return op
    return resolve_tool_definition(capability_id, tool_catalog=tool_catalog)


def resolve_capability_for_turn_limit(
    capability_id: str,
    *,
    operation_catalog: dict[str, OperationDefinition] | None = None,
    tool_catalog: dict[str, ToolDefinition] | None = None,
) -> CapabilityDefinition | None:
    """Resolve a capability id for per-turn ``max_calls_per_turn`` enforcement.

    Same precedence as ``resolve_capability`` (operations before tools). Both
    definition types expose ``max_calls_per_turn``; callers apply
    ``getattr(definition, "max_calls_per_turn", 0)`` on the result. MCP-native
    tools have no kernel ``ToolDefinition`` and are bounded only by the global
    iteration count, so they correctly resolve to ``None`` here.
    """
    return resolve_capability(
        capability_id,
        operation_catalog=operation_catalog,
        tool_catalog=tool_catalog,
    )


def _default_operation_catalog() -> dict[str, OperationDefinition]:
    """Return the kernel operation catalog, importing lazily to avoid cycles."""
    from kernel.operations.registry import get_operation_catalog

    return get_operation_catalog()


def _default_tool_catalog() -> dict[str, ToolDefinition]:
    """Return the kernel tool catalog, importing lazily to avoid cycles."""
    from kernel.tools.registry import get_tool_catalog

    return get_tool_catalog()