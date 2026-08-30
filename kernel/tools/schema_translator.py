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
import re
from typing import Any

__all__ = [
    "anthropic_tool_use_to_mcp",
    "json_escaped_string_decode",
    "looks_like_json_escaped_text",
    "mcp_tool_to_anthropic_tool",
    "mcp_tool_to_openai_function",
    "openai_tool_call_to_mcp",
]

# A leaked escaped rendering of a document hides its line breaks either as
# ``\n`` pairs (one escaped layer) or ``\\n`` quad sequences (two layers).
# Ordinary text has at most a stray one of these; this many is the repair
# threshold. A one-off mention of ``\n`` in prose (e.g. documentation about
# escapes) stays far below it.
#
# Counting uses the two-char pair ``\\n``; a hex-escaped or quad-escaped
# layer still contains at least this many *adjacent-to-backslash* n's, and
# the decode loop re-detects each remaining layer after each pass.
_MIN_NEWLINE_ESCAPES = 3

# Cap on successive decode passes. One leaked layer is the common case;
# two occurs when the model escaped an already-escaped copy. Three is
# generous headroom; a pathological payload simply fails the repair and
# dispatches unmodified.
_MAX_DECODE_PASSES = 3


def looks_like_json_escaped_text(value: str) -> bool:
    """Return True when a whole-string arg looks like a leaked JSON rendering.

    Heuristic for the double-encoding failure mode (the 2026-08-28
    cloud-init incident: a ``write_file`` content arg arrived with zero
    real newlines and every line break baked in as a literal ``\\n``
    pair — the model echoed an escaped copy of the file from its context
    instead of the file). Four signals together:

    1. No real newline anywhere in the string;
    2. several literal ``\\n`` pairs (≥ :data:`_MIN_NEWLINE_ESCAPES`) —
       escape sequences in a document are separated by ordinary text, so
       "runs" of escapes never occur; the raw count is the signal. The
       count includes the wider ``\\\\n``/``\\u00XX`` spellings via a
       normalized view, so double-escaped layers still trip it;
    3. not a bare JSON array/object — those are legitimate other payload
       types, not the string-escaping signature;
    4. the value is decodable under the JSON string-escape grammar, so
       the repair in :func:`json_escaped_string_decode` cannot half-affect
       it (an unknown escape like ``\\$`` fails this check and stops
       further passes — real shell/nginx ``\\$`` output is preserved).
    """
    if not value or "\n" in value:
        return False
    stripped = value.strip()
    if stripped[:1] in ("[", "{"):
        return False
    # Normalize the wider escape spellings (\u006e hex-escapes,
    # \\\\n double-escapes) down to the base \\n pair so the count covers
    # every layer shape, not just the single-escaped one.
    normalized = re.sub(r"\\\\+u006[eE]", "\\\\n", value)
    normalized = re.sub(r"\\\\n", "\\\\n", normalized)
    if normalized.count("\\n") < _MIN_NEWLINE_ESCAPES:
        return False
    try:
        json.loads('"' + value + '"')
    except json.JSONDecodeError:
        return False
    return True


def json_escaped_string_decode(value: str) -> str:
    """Decode a leaked JSON-escaped rendering back to the intended text.

    The corruption round-trip: the model's intended ``content`` string was
    JSON-serialized (once or twice) *into the argument payload itself*, so
    after normal ``json.loads`` of the arguments dict the callee holds an
    escape-sequence-laden copy of the text. Decoding via ``json.loads`` on
    a synthetically quoted string recovers the real text exactly.

    Every pass requires the decode to strictly shrink the value (escape
    decoding always does: 2-char ``\\n`` → 1-char newline), so a
    misclassified or adversarial payload can never come out longer or
    altered in character count. After each pass the loop re-detects with
    a residual-quad check: the first decode of a double-escaped payload
    converts ``\\\\n`` quads into ``\\n`` pairs *while already producing*
    real newlines, so the plain full-signature check would stop early.
    Any failed check falls through to the current value: a repair attempt
    must never silently mutate a payload it cannot fully interpret.
    """

    def _residual_escape_quads(text: str) -> int:
        # ``\\\\n`` quads remaining after a decode pass (see docstring).
        return len(re.findall(r"\\\\n", text))

    decoded = value
    for _ in range(_MAX_DECODE_PASSES):
        has_real_newlines = "\n" in decoded
        if not has_real_newlines:
            if not looks_like_json_escaped_text(decoded):
                return decoded
        elif _residual_escape_quads(decoded) < _MIN_NEWLINE_ESCAPES:
            # Decoded text with real newlines and no leftover quads is a
            # healthy payload — stop.
            return decoded
        try:
            candidate = json.loads('"' + decoded + '"')
        except json.JSONDecodeError:
            return decoded
        if len(candidate) >= len(decoded):
            return decoded
        decoded = candidate
    return decoded


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

    String parameter values that carry the leaked-JSON-rendering signature
    (zero real newlines, long escape runs) are decoded once via
    :func:`json_escaped_string_decode` — the 2026-08-28 cloud-init fault
    where ``write_file`` persisted a JSON-escaped copy of the document as
    the literal file content. Repair is value-conditional, never blanket
    decoding: a string with real newlines, or without the long escape-run
    signature, passes through byte-for-byte.
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
            return name, _repair_escaped_string_args(parsed)
        except json.JSONDecodeError:
            return name, {}
    if isinstance(raw_args, dict):
        return name, _repair_escaped_string_args(raw_args)
    return name, {}


def _repair_escaped_string_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Fix leaked JSON-escaped renderings in top-level string args.

    Repairs one level (the common tool schema shape: ``path``/``content``
    string params at the top of the arguments object). Nested containers
    are left untouched — a fault deeper in the tree is a different
    signature and blanket recursion could mangle intentional escaped text
    inside user payloads. See the ``schema_translator`` module note for
    the full fault description.
    """
    repaired: dict[str, Any] = {}
    for key, val in arguments.items():
        if isinstance(val, str):
            repaired[key] = json_escaped_string_decode(val)
        else:
            repaired[key] = val
    return repaired


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
