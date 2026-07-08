"""Translate MCP tool schemas to/from provider-specific formats.

The kernel speaks MCP tool schema internally. At the provider edge, this
module translates to whatever the provider expects:

- OpenAI function-calling: ``mcp_tool_to_openai_function`` /
  ``openai_tool_call_to_mcp``
- Anthropic tool-use: ``mcp_tool_to_anthropic_tool`` /
  ``anthropic_tool_use_to_mcp`` (stub — Phase 5)

MCP's native ``inputSchema`` is JSON-Schema, which is a superset of what
OpenAI and Anthropic accept. The translators pass it through unchanged;
providers that reject exotic keywords will need a sanitizer in a later
phase.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "anthropic_tool_use_to_mcp",
    "mcp_tool_to_anthropic_tool",
    "mcp_tool_to_openai_function",
    "openai_tool_call_to_mcp",
]


def mcp_tool_to_openai_function(
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Convert one MCP tool spec to an OpenAI function-calling tool dict.

    OpenAI's tool schema is::

        {
            "type": "function",
            "function": {
                "name": ...,
                "description": ...,
                "parameters": <JSON-Schema>,
            },
        }
    """

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema or {"type": "object", "properties": {}},
        },
    }


def openai_tool_call_to_mcp(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract (tool_name, arguments) from an OpenAI tool_call dict.

    OpenAI wraps the tool name and arguments inside a ``function`` sub-dict.
    The arguments arrive as a JSON string that must be parsed.
    """

    function = tool_call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name", ""))
        raw_args = function.get("arguments", "{}")
    else:
        name = str(tool_call.get("name", ""))
        raw_args = tool_call.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if not isinstance(parsed, dict):
                return name, {}
            return name, parsed
        except json.JSONDecodeError:
            return name, {}
    if isinstance(raw_args, dict):
        return name, raw_args
    return name, {}


def mcp_tool_to_anthropic_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Convert one MCP tool spec to an Anthropic tool-use dict.

    Anthropic's tool schema is::

        {
            "name": ...,
            "description": ...,
            "input_schema": <JSON-Schema>,
        }
    """

    return {
        "name": name,
        "description": description,
        "input_schema": input_schema or {"type": "object", "properties": {}},
    }


def anthropic_tool_use_to_mcp(tool_use: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract (tool_name, arguments) from an Anthropic tool_use block.

    Anthropic delivers tool-use as a content block with ``name`` and
    ``input`` (already a dict, not a JSON string).
    """

    name = str(tool_use.get("name", ""))
    raw_args = tool_use.get("input", {})
    if isinstance(raw_args, dict):
        return name, raw_args
    return name, {}
