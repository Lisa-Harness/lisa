"""Rendering and scoped matching for Copilot customization content.

This module turns a parsed :class:`~sdk.content.CustomizationBundle` into the
Markdown that belongs in a system prompt, and selects the ``applyTo``-scoped
instructions that match a path. It is the presentation half of the
customization contract: :mod:`sdk.content._copilot_customizations` owns the file
schema and :mod:`sdk.content._copilot_discovery` owns the workspace walk, while
this module owns what the model is shown and when.

Two scoping rules decide *when* a body costs context, so a turn pays only for
guidance it can use:

- **Instructions** declare ``applyTo`` globs. A scoped body is not part of the
  unconditional prompt; it is rendered only once a tool call touches a matching
  path. Instructions with no ``applyTo`` are unconditional.
- **Skills** have no ``applyTo``, so they cannot be matched to a path at all.
  They contribute an *index* — name, description, and the ``SKILL.md`` path —
  and the model reads a body when a task matches. This mirrors Copilot's
  progressive loading, where discovery is cheap and the body is on demand.

Everything here is pure: no I/O, no configuration lookup, no host state. The
caller owns discovery, caching, and per-turn bookkeeping.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sdk.content._copilot_customizations import (
    CustomizationBundle,
    InstructionFile,
    SkillFile,
    matches_apply_to,
)

__all__ = [
    "matches_any_scoped_instruction",
    "render_scoped_instruction",
    "render_skill_index",
    "render_unconditional_blocks",
    "scoped_instructions_for_paths",
    "tool_call_paths",
    "workspace_relative_posix_path",
]

#: Tool argument keys whose value names a path directly (``str``), and keys whose
#: value is a list of records each carrying a ``path`` (a batched edit tool).
#: Coding-pack file tools name their argument ``path``; ``file_path`` covers the
#: customization reader. Directory-scoped values match no ``applyTo`` glob and
#: therefore contribute nothing.
PATH_ARGUMENT_KEYS: tuple[str, ...] = ("path", "file_path")
PATH_RECORD_LIST_KEYS: tuple[str, ...] = ("edits", "files")


def tool_call_paths(arguments: Any) -> tuple[str, ...]:
    """Return the distinct paths a tool call refers to, in argument order.

    Covers the single-path tools (``path`` / ``file_path``) and the batched
    shape whose paths live inside a list of records. Non-string, empty, and
    unparseable values contribute nothing — a call that names no identifiable
    file gets no scoped guidance.
    """
    if not isinstance(arguments, dict):
        return ()
    paths: list[str] = []
    for key in PATH_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    for key in PATH_RECORD_LIST_KEYS:
        records = arguments.get(key)
        if not isinstance(records, (list, tuple)):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            value = record.get("path")
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
    # dict.fromkeys keeps first-seen order while removing repeats.
    return tuple(dict.fromkeys(paths))


def workspace_relative_posix_path(file_path: str) -> str:
    """Normalize a tool-argument path for the ``applyTo`` matcher.

    Returns a POSIX-style path with no leading ``./``, or ``""`` when the value
    names no path (empty, ``"."``) or escapes the workspace root via ``..``.
    A leading slash is stripped rather than rejected: reading an absolute path
    workspace-relative can only miss a directory-anchored glob, never attach a
    body for an unrelated file.
    """
    normalized = file_path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if not normalized:
        return ""
    parts: list[str] = []
    for part in normalized.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            # Escapes the workspace root; nothing scoped can match.
            return ""
        parts.append(part)
    return "/".join(parts)


def scoped_instructions_for_paths(
    instructions: Iterable[InstructionFile],
    file_paths: Sequence[str],
) -> tuple[InstructionFile, ...]:
    """Return the scoped instructions matching any of ``file_paths``.

    Each path is normalized first; a path that escapes the workspace
    contributes nothing. Results are deduped by instruction path and keep the
    caller's order, so a batched edit of five Python files yields one copy of
    the Python guidance.
    """
    relatives = [
        relative for relative in (workspace_relative_posix_path(p) for p in file_paths) if relative
    ]
    if not relatives:
        return ()
    matched: list[InstructionFile] = []
    seen: set[str] = set()
    for instruction in instructions:
        globs = list(instruction.apply_to)
        if not globs or instruction.path in seen:
            continue
        if any(matches_apply_to(globs, relative) for relative in relatives):
            matched.append(instruction)
            seen.add(instruction.path)
    return tuple(matched)


def matches_any_scoped_instruction(
    instructions: Iterable[InstructionFile],
    file_path: str,
) -> tuple[InstructionFile, ...]:
    """Return the scoped instructions matching one path."""
    return scoped_instructions_for_paths(instructions, (file_path,))


def render_scoped_instruction(instruction: InstructionFile) -> str:
    """Render one ``applyTo``-scoped instruction as a labelled Markdown block.

    The ``applies to`` label is kept so the model can see why the guidance
    arrived with a particular tool result.
    """
    apply_label = ", ".join(instruction.apply_to)
    return (
        f"## Instruction: {instruction.name} (applies to: {apply_label})"
        f"\n\n{instruction.body.strip()}"
    )


def render_skill_index(skills: Iterable[SkillFile]) -> str:
    """Render the skill index: name, description, and path — never the body.

    Mirrors Copilot's progressive loading. The index is the discovery surface
    (~100 tokens per skill), the body loads when a turn decides it needs the
    procedure, and a skill's own reference files load only if the body names
    them. Injecting bodies eagerly costs the full corpus on every request no
    matter what the turn does.

    Skill bodies have no ``applyTo``, so unlike instructions they cannot be
    matched to a path; the model must choose. Each entry therefore names the
    file to read, and the read goes through the ordinary file-read tool in
    source scope — no dedicated loader is required.

    ``disable_model_invocation`` is honored as an opt-out of model-driven
    loading: such a skill is neither listed nor injected. Copilot keeps it
    reachable as a slash command; a host with no skill-slash surface would
    otherwise be listing guidance the model is forbidden to load.
    """
    usable = [
        skill for skill in skills if skill.body.strip() and not skill.disable_model_invocation
    ]
    if not usable:
        return ""
    lines = [
        "## Skills",
        "",
        "Each skill below is a procedure you can load on demand. Read the named",
        "file when your task matches its description; do not guess at its",
        "contents. Skills you do not load cost nothing.",
        "",
    ]
    for skill in usable:
        description = " ".join(skill.description.split())
        lines.append(f"- **{skill.name}** — {description}")
        lines.append(f"  Load with the file-read tool: `{skill.path}`")
    return "\n".join(lines)


def render_unconditional_blocks(bundle: CustomizationBundle) -> list[str]:
    """Render the customization blocks that belong in every system prompt.

    Carries always-on files, instructions that declare no ``applyTo``, and the
    skill index. Instructions that declare one are deferred to the dispatch
    that first touches a matching path, and skill bodies are loaded on demand
    (see :func:`render_skill_index`).
    """
    parts = [
        f"## Project Instructions\n\n{file.body.strip()}"
        for file in bundle.always_on
        if file.body.strip()
    ]
    parts.extend(
        render_scoped_instruction(instruction)
        for instruction in bundle.instructions
        if instruction.body.strip() and not instruction.apply_to
    )
    index = render_skill_index(bundle.skills)
    if index:
        parts.append(index)
    return parts
