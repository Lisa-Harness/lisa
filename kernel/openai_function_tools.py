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


def build_strict_openai_function_tool(
    *,
    name: str,
    description: str,
    parameters_schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Return one function-tool schema compatible with SDK auto-parsing.

    Uses structured outputs (`strict=True`) unless the schema explicitly
    requires arbitrary object keys (`additionalProperties: True`), which
    OpenAI strict mode forbids.
    """

    tool = pydantic_function_tool(
        parameters_schema or _EmptyToolParameters,
        name=name,
        description=description,
    )

    if _has_additional_properties_true(tool.get("function", {}).get("parameters", {})):
        tool["function"]["strict"] = False

    return cast(dict[str, Any], tool)
