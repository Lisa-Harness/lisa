"""Tests for kernel/openai_function_tools.py — strict-mode schema gating.

Locks in the behavior of :func:`build_strict_openai_function_tool`: strict
mode stays on for clean schemas and turns off for schemas OpenAI strict mode
rejects (``additionalProperties: True`` and ``anyOf``/``oneOf`` inside
``items``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from kernel.openai_function_tools import (
    _has_additional_properties_true,
    _has_union_items,
    _strict_incompatible,
    build_strict_openai_function_tool,
)


class _Empty(BaseModel):
    pass


class _OptionalField(BaseModel):
    """Optional fields emit top-level anyOf — strict mode allows this."""

    name: str | None = None


class _UnionListItems(BaseModel):
    """list[Union[...]] emits anyOf inside items — strict mode rejects this."""

    source_paths: list[str | dict[str, str]] = []


class _ArbitraryKeys(BaseModel):
    """additionalProperties: True — strict mode rejects this."""

    model_config = {"extra": "allow"}


def _is_strict(tool: dict[str, Any]) -> bool:
    return bool(tool.get("function", {}).get("strict", False))


# ── Heuristics ────────────────────────────────────────────────────────────


def test_has_union_items_flags_items_anyof() -> None:
    schema = {"items": {"anyOf": [{"type": "string"}, {"type": "object"}]}}
    assert _has_union_items(schema) is True


def test_has_union_items_ignores_top_level_optional_anyof() -> None:
    schema = {"properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
    assert _has_union_items(schema) is False


def test_has_union_items_flags_oneof() -> None:
    schema = {"items": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}
    assert _has_union_items(schema) is True


def test_has_additional_properties_true_detects() -> None:
    assert _has_additional_properties_true({"additionalProperties": True}) is True
    assert _has_additional_properties_true({"items": {"type": "string"}}) is False


def test_strict_incompatible_or_of_both() -> None:
    assert _strict_incompatible({"additionalProperties": True}) is True
    assert _strict_incompatible({"items": {"anyOf": []}}) is True
    assert _strict_incompatible({"items": {"type": "string"}}) is False


# ── build_strict_openai_function_tool ─────────────────────────────────────


def test_clean_schema_stays_strict() -> None:
    tool = build_strict_openai_function_tool(
        name="clean", description="d", parameters_schema=_Empty
    )
    assert _is_strict(tool) is True


def test_optional_field_stays_strict() -> None:
    tool = build_strict_openai_function_tool(
        name="opt", description="d", parameters_schema=_OptionalField
    )
    assert _is_strict(tool) is True


def test_union_list_items_disables_strict() -> None:
    tool = build_strict_openai_function_tool(
        name="unionlist", description="d", parameters_schema=_UnionListItems
    )
    assert _is_strict(tool) is False


def test_arbitrary_keys_disables_strict() -> None:
    tool = build_strict_openai_function_tool(
        name="arbitrary", description="d", parameters_schema=_ArbitraryKeys
    )
    assert _is_strict(tool) is False
