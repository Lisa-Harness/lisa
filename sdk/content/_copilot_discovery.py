"""Workspace walker that discovers VS Code Copilot customization files.

Mirrors the file-location precedence documented by VS Code:

- ``.github/copilot-instructions.md`` (always-on, workspace root only)
- ``AGENTS.md`` (always-on, workspace root + nested subfolders)
- ``CLAUDE.md`` (always-on, workspace root / ``.claude/`` / ``CLAUDE.local.md``)
- ``.github/instructions/*.instructions.md`` (VS Code format, recursive)
- ``.claude/rules/*.md`` (Claude format, recursive — uses ``paths`` not ``applyTo``)
- ``.github/prompts/*.prompt.md`` (recursive)
- ``.github/agents/*.agent.md`` (VS Code format, recursive)
- ``.claude/agents/*.md`` (Claude format, recursive — tools as comma-string)
- ``.github/skills/*/SKILL.md`` (dir-named, recursive)
- ``.claude/skills/*/SKILL.md`` (dir-named, recursive)
- ``.agents/skills/*/SKILL.md`` (dir-named, recursive)

Parent-repository discovery (the ``chat.useCustomizationsInParentRepositories``
walk-up-to-``.git`` behavior) is supported via :func:`discover_copilot_customizations`
with ``include_parents=True``.

All parsing is delegated to :mod:`sdk.content._copilot_customizations`;
malformed files produce diagnostics rather than raising.
"""

from __future__ import annotations

from pathlib import Path

from sdk.content._copilot_customizations import (
    AgentFile,
    CopilotInstructionsFile,
    CustomizationBundle,
    CustomizationDiagnostic,
    InstructionFile,
    PromptFile,
    SkillFile,
    parse_agent_file,
    parse_always_on_file,
    parse_instruction_file,
    parse_prompt_file,
    parse_skill_file,
)

__all__ = ["discover_copilot_customizations"]

# Filename patterns. Skills are discovered by ``SKILL.md`` inside a directory;
# everything else by file extension/glob.

_INSTRUCTION_SUFFIXES = (".instructions.md",)
_PROMPT_SUFFIXES = (".prompt.md",)
_AGENT_SUFFIXES = (".agent.md",)


def discover_copilot_customizations(
    workspace_root: Path,
    *,
    include_parents: bool = False,
    nested_agents_md: bool = False,
) -> CustomizationBundle:
    """Discover and parse all Copilot customization files reachable from ``workspace_root``.

    Args:
        workspace_root: The workspace folder to scan.
        include_parents: When True, walk up the folder hierarchy until a
            ``.git`` directory is found and scan all intermediate folders too
            (mirrors ``chat.useCustomizationsInParentRepositories``).
        nested_agents_md: When True, recursively search for ``AGENTS.md``
            files in subfolders of the workspace (mirrors the experimental
            ``chat.useNestedAgentsMdFiles`` setting).

    Returns:
        A :class:`CustomizationBundle` containing every successfully-parsed
        file plus structured diagnostics for malformed ones.
    """
    workspace_root = workspace_root.resolve()
    roots = _collect_scan_roots(workspace_root, include_parents=include_parents)

    always_on: list[CopilotInstructionsFile] = []
    instructions: list[InstructionFile] = []
    agents: list[AgentFile] = []
    prompts: list[PromptFile] = []
    skills: list[SkillFile] = []
    diagnostics: list[CustomizationDiagnostic] = []

    seen_paths: set[str] = set()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(workspace_root)).replace("\\", "/")
        except ValueError:
            return str(p)

    for root in roots:
        # Always-on: copilot-instructions.md (root of each scan root only).
        copilot_instr = root / ".github" / "copilot-instructions.md"
        if copilot_instr.is_file():
            _add_always_on(
                copilot_instr, _rel(copilot_instr), always_on, diagnostics, seen_paths
            )

        # Always-on: AGENTS.md at scan root.
        agents_md = root / "AGENTS.md"
        if agents_md.is_file():
            _add_always_on(
                agents_md, _rel(agents_md), always_on, diagnostics, seen_paths
            )

        # Always-on: CLAUDE.md variants.
        for claude_md in _iter_claude_md_files(root):
            if claude_md.is_file():
                _add_always_on(
                    claude_md, _rel(claude_md), always_on, diagnostics, seen_paths
                )

        # Nested AGENTS.md (experimental).
        if nested_agents_md:
            for candidate in root.rglob("AGENTS.md"):
                if candidate == agents_md:
                    continue
                _add_always_on(
                    candidate, _rel(candidate), always_on, diagnostics, seen_paths
                )

        # File-based instructions: .github/instructions/**.instructions.md
        instructions_dir = root / ".github" / "instructions"
        if instructions_dir.is_dir():
            for f in _rglob_suffix(instructions_dir, _INSTRUCTION_SUFFIXES):
                _add_instruction(
                    f,
                    _rel(f),
                    instructions,
                    diagnostics,
                    seen_paths,
                    fmt="vscode",
                )

        # File-based instructions: .claude/rules/**.md (Claude format).
        claude_rules_dir = root / ".claude" / "rules"
        if claude_rules_dir.is_dir():
            for f in claude_rules_dir.rglob("*.md"):
                _add_instruction(
                    f,
                    _rel(f),
                    instructions,
                    diagnostics,
                    seen_paths,
                    fmt="claude",
                )

        # Prompts: .github/prompts/**.prompt.md
        prompts_dir = root / ".github" / "prompts"
        if prompts_dir.is_dir():
            for f in _rglob_suffix(prompts_dir, _PROMPT_SUFFIXES):
                _add_prompt(f, _rel(f), prompts, diagnostics, seen_paths)

        # Agents (VS Code): .github/agents/**.agent.md
        vscode_agents_dir = root / ".github" / "agents"
        if vscode_agents_dir.is_dir():
            for f in _rglob_suffix(vscode_agents_dir, _AGENT_SUFFIXES):
                _add_agent(
                    f,
                    _rel(f),
                    agents,
                    diagnostics,
                    seen_paths,
                    fmt="vscode",
                )

        # Agents (Claude): .claude/agents/**.md (any .md, tools as comma-string).
        claude_agents_dir = root / ".claude" / "agents"
        if claude_agents_dir.is_dir():
            for f in claude_agents_dir.rglob("*.md"):
                _add_agent(
                    f,
                    _rel(f),
                    agents,
                    diagnostics,
                    seen_paths,
                    fmt="claude",
                )

        # Skills (dir-named SKILL.md) in all three locations.
        for skills_dir in (
            root / ".github" / "skills",
            root / ".claude" / "skills",
            root / ".agents" / "skills",
        ):
            if skills_dir.is_dir():
                for skill_md in skills_dir.rglob("SKILL.md"):
                    _add_skill(
                        skill_md, _rel(skill_md), skills, diagnostics, seen_paths
                    )

    return CustomizationBundle(
        always_on=tuple(always_on),
        instructions=tuple(instructions),
        agents=tuple(agents),
        prompts=tuple(prompts),
        skills=tuple(skills),
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------- #
# Root collection
# --------------------------------------------------------------------------- #


def _collect_scan_roots(
    workspace_root: Path, *, include_parents: bool
) -> list[Path]:
    """Return the ordered list of folders to scan.

    ``workspace_root`` is always first. When ``include_parents`` is True, we
    also walk up until we find a ``.git`` directory and include each
    intermediate folder. The repository root (the folder containing ``.git``)
    is included last so workspace-root files win on ties — but the caller
    deduplicates by resolved path, so order only affects precedence logging,
    not which files are returned.
    """
    roots: list[Path] = [workspace_root]
    if not include_parents:
        return roots

    current = workspace_root
    while current != current.parent:
        if (current / ".git").exists():
            # current is the repo root; include it and stop.
            if current not in roots:
                roots.append(current)
            break
        parent = current.parent
        if parent == current:
            break
        if parent not in roots:
            roots.append(parent)
        current = parent
    return roots


def _iter_claude_md_files(root: Path) -> list[Path]:
    """Return the candidate CLAUDE.md paths for a scan root.

    Order: root ``CLAUDE.md``, ``.claude/CLAUDE.md``, ``CLAUDE.local.md``.
    """
    return [
        root / "CLAUDE.md",
        root / ".claude" / "CLAUDE.md",
        root / "CLAUDE.local.md",
    ]


# --------------------------------------------------------------------------- #
# File-enumeration helpers
# --------------------------------------------------------------------------- #


def _rglob_suffix(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Recursively list files ending with any of ``suffixes`` under ``directory``."""
    matches: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.name.endswith(suffixes):
            matches.append(path)
    return matches


# --------------------------------------------------------------------------- #
# Add-parsed-file helpers (deduplicate and collect diagnostics)
# --------------------------------------------------------------------------- #


def _add_always_on(
    file_path: Path,
    rel: str,
    out: list[CopilotInstructionsFile],
    diagnostics: list[CustomizationDiagnostic],
    seen: set[str],
) -> None:
    if rel in seen:
        return
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(
            CustomizationDiagnostic(
                path=rel, message=f"Could not read file: {exc}", kind="always_on"
            )
        )
        return
    parsed, diag = parse_always_on_file(rel, text)
    out.append(parsed)
    seen.add(rel)
    if diag is not None:
        diagnostics.append(diag)


def _add_instruction(
    file_path: Path,
    rel: str,
    out: list[InstructionFile],
    diagnostics: list[CustomizationDiagnostic],
    seen: set[str],
    *,
    fmt: str,
) -> None:
    if rel in seen:
        return
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(
            CustomizationDiagnostic(
                path=rel, message=f"Could not read file: {exc}", kind="instructions"
            )
        )
        return
    parsed, diag = parse_instruction_file(rel, text, fmt=fmt)  # type: ignore[arg-type]
    if parsed is not None:
        out.append(parsed)
        seen.add(rel)
    if diag is not None:
        diagnostics.append(diag)


def _add_agent(
    file_path: Path,
    rel: str,
    out: list[AgentFile],
    diagnostics: list[CustomizationDiagnostic],
    seen: set[str],
    *,
    fmt: str,
) -> None:
    if rel in seen:
        return
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(
            CustomizationDiagnostic(
                path=rel, message=f"Could not read file: {exc}", kind="agents"
            )
        )
        return
    parsed, diag = parse_agent_file(rel, text, fmt=fmt)  # type: ignore[arg-type]
    if parsed is not None:
        out.append(parsed)
        seen.add(rel)
    if diag is not None:
        diagnostics.append(diag)


def _add_prompt(
    file_path: Path,
    rel: str,
    out: list[PromptFile],
    diagnostics: list[CustomizationDiagnostic],
    seen: set[str],
) -> None:
    if rel in seen:
        return
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(
            CustomizationDiagnostic(
                path=rel, message=f"Could not read file: {exc}", kind="prompts"
            )
        )
        return
    parsed, diag = parse_prompt_file(rel, text)
    if parsed is not None:
        out.append(parsed)
        seen.add(rel)
    if diag is not None:
        diagnostics.append(diag)


def _add_skill(
    file_path: Path,
    rel: str,
    out: list[SkillFile],
    diagnostics: list[CustomizationDiagnostic],
    seen: set[str],
) -> None:
    if rel in seen:
        return
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(
            CustomizationDiagnostic(
                path=rel, message=f"Could not read file: {exc}", kind="skills"
            )
        )
        return
    parsed, diag = parse_skill_file(rel, text)
    if parsed is not None:
        out.append(parsed)
        seen.add(rel)
    if diag is not None:
        diagnostics.append(diag)