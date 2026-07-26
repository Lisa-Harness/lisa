"""Lisa Content SDK — content catalog, authored-text reading, and diagnostics.

Public API (import from :mod:`sdk.content`, not the ``_``-prefixed submodules):

- :func:`build_mode_catalog_json` — mode catalog JSON
- :func:`build_operation_catalog_json` — operation catalog JSON
- :func:`build_tool_catalog_json` — tool catalog JSON
- :func:`resolve_mode_roots` — mode subdirectory roots
- :func:`resolve_operation_roots` — operation subdirectory roots
- :func:`resolve_tool_roots` — tool subdirectory roots
- :func:`invalidate_catalog_cache` — clear catalog caches
- :class:`ContentReader` — authored-text file reader
- :func:`is_safe_relative_path` — path-safety check
- :func:`resolve_content_roots` — merged root discovery
- :func:`roots_fingerprint` — cache key for root sets
- :func:`build_catalog_diagnostics_json` — diagnostics report

Copilot customization discovery (``.github/{instructions,agents,prompts,skills}``
and the ``.claude/`` equivalents):

- :func:`discover_copilot_customizations` — workspace walker
- :func:`matches_apply_to` / :func:`matches_any_instruction` — glob matching
- :func:`parse_instruction_file` / :func:`parse_agent_file` /
  :func:`parse_prompt_file` / :func:`parse_skill_file` /
  :func:`parse_always_on_file` — per-kind parsers
- :class:`CustomizationBundle` and the typed per-kind dataclasses

The ``_catalog`` / ``_reader`` / ``_roots`` / ``_diagnostics`` / ``_front_matter``
/ ``_copilot_customizations`` / ``_copilot_discovery`` modules are internal
layout; the names above are re-exported here as the stable surface. The MCP
servers (``lisa-content`` for file reading, ``lisa-catalog`` for catalogs and
diagnostics) are thin wrappers over these functions and classes. Hosts or
tools that want in-process content access can call them directly without
spawning an MCP server.
"""

from __future__ import annotations

from sdk.content._catalog import (
    CatalogCollisionError,
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    invalidate_catalog_cache,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)
from sdk.content._copilot_customizations import (
    AgentFile,
    CopilotInstructionsFile,
    CustomizationBundle,
    CustomizationDiagnostic,
    InstructionFile,
    PromptFile,
    SkillFile,
    matches_any_instruction,
    matches_apply_to,
    parse_agent_file,
    parse_always_on_file,
    parse_instruction_file,
    parse_prompt_file,
    parse_skill_file,
)
from sdk.content._copilot_discovery import discover_copilot_customizations
from sdk.content._diagnostics import build_catalog_diagnostics_json
from sdk.content._loader_strict import (
    MissingAuthoredContentError,
    require_authored_text,
    validate_required_content,
)
from sdk.content._reader import ContentReader, is_safe_relative_path
from sdk.content._roots import resolve_content_roots, roots_fingerprint

__all__ = [
    "AgentFile",
    "CatalogCollisionError",
    "CopilotInstructionsFile",
    "ContentReader",
    "CustomizationBundle",
    "CustomizationDiagnostic",
    "InstructionFile",
    "MissingAuthoredContentError",
    "PromptFile",
    "SkillFile",
    "build_catalog_diagnostics_json",
    "build_mode_catalog_json",
    "build_operation_catalog_json",
    "build_tool_catalog_json",
    "discover_copilot_customizations",
    "invalidate_catalog_cache",
    "is_safe_relative_path",
    "matches_any_instruction",
    "matches_apply_to",
    "parse_agent_file",
    "parse_always_on_file",
    "parse_instruction_file",
    "parse_prompt_file",
    "parse_skill_file",
    "require_authored_text",
    "resolve_content_roots",
    "resolve_mode_roots",
    "resolve_operation_roots",
    "resolve_tool_roots",
    "roots_fingerprint",
    "validate_required_content",
]
