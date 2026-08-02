"""Helpers for OpenAI function-tool schema generation."""

from __future__ import annotations

from typing import Any, cast

from openai import pydantic_function_tool
from pydantic import BaseModel


class _EmptyToolParameters(BaseModel):
    """Placeholder schema for function tools that take no arguments."""


def _has_additional_properties_true(schema: Any) -> bool:
    if isinstance(schema, dict):
        if schema.get("additionalProperties") is True:
            return True
        return any(_has_additional_properties_true(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_additional_properties_true(item) for item in schema)
    return False


def _has_union_items(schema: Any) -> bool:
    """Detect ``anyOf``/``oneOf`` inside an ``items`` schema.

    OpenAI strict mode requires every ``items`` node to declare an explicit
    ``type``. pydantic emits ``anyOf``/``oneOf`` in ``items`` for
    ``list[Union[...]]`` fields (e.g. ``list[str | dict[str, str]]``), which
    strict mode rejects with ``schema must have a 'type' key``. Top-level
    ``anyOf`` on a property (e.g. ``Optional[str]``) is permitted by strict
    mode and is NOT flagged here, so the common optional-field case keeps
    strict mode.
    """
    if isinstance(schema, dict):
        if "items" in schema and isinstance(schema["items"], dict):
            if "anyOf" in schema["items"] or "oneOf" in schema["items"]:
                return True
        return any(_has_union_items(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_union_items(item) for item in schema)
    return False


def _strict_incompatible(schema: Any) -> bool:
    """True when the schema is incompatible with OpenAI strict mode."""
    return _has_additional_properties_true(schema) or _has_union_items(schema)


def build_strict_openai_function_tool(
    *,
    name: str,
    description: str,
    parameters_schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Return one function-tool schema compatible with SDK auto-parsing.

    Uses structured outputs (`strict=True`) unless the schema is
    incompatible with OpenAI strict mode — arbitrary object keys
    (`additionalProperties: True`) or union shapes (`anyOf`/`oneOf`),
    both of which strict mode rejects. Incompatible schemas fall back to
    JSON mode (`strict=False`) so the model can still call the tool.
    """

    tool = pydantic_function_tool(
        parameters_schema or _EmptyToolParameters,
        name=name,
        description=description,
    )

    if _strict_incompatible(tool.get("function", {}).get("parameters", {})):
        tool["function"]["strict"] = False

    return cast(dict[str, Any], tool)
