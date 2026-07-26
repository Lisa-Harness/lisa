"""Schema-aware parsing for VS Code Copilot customization files.

These files live under ``.github/{instructions,agents,prompts,skills}`` and
related locations (``.claude/{rules,agents,skills}``, ``AGENTS.md``,
``CLAUDE.md``, ``copilot-instructions.md``). Each is Markdown-with-YAML
front-matter; the front-matter schema varies by file kind.

This module knows about the schema; the generic ``---``-delimited YAML
parsing is delegated to :func:`sdk.content._front_matter.parse_front_matter`.

Validation mirrors Copilot's "silent fail" behavior: malformed files
produce structured diagnostics rather than raising, so a discovery walk
over a noisy workspace returns whatever it can parse.

The Copilot specification is documented at:
- https://code.visualstudio.com/docs/agent-customization/custom-instructions
- https://code.visualstudio.com/docs/agent-customization/agent-skills
- https://code.visualstudio.com/docs/agent-customization/custom-agents
- https://code.visualstudio.com/docs/agent-customization/prompt-files
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from sdk.content._front_matter import FrontMatterError, parse_front_matter

__all__ = [
    "AgentFile",
    "CopilotInstructionsFile",
    "CustomizationBundle",
    "CustomizationDiagnostic",
    "CustomizationKind",
    "InstructionFile",
    "PromptFile",
    "SkillFile",
    "matches_apply_to",
    "matches_any_instruction",
    "parse_agent_file",
    "parse_always_on_file",
    "parse_instruction_file",
    "parse_prompt_file",
    "parse_skill_file",
]

CustomizationKind = Literal["instructions", "agents", "prompts", "skills", "always_on"]

# Skill `name` must be lowercase letters/numbers/hyphens, <=64 chars, and must
# match its parent directory name (Copilot silently fails the load otherwise).
_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_SKILL_DESCRIPTION_MAX = 1024


@dataclass(frozen=True, slots=True)
class CustomizationDiagnostic:
    """One non-fatal problem found while parsing a customization file."""

    path: str
    message: str
    kind: CustomizationKind | None = None


@dataclass(frozen=True, slots=True)
class InstructionFile:
    """A parsed ``*.instructions.md`` (or ``.claude/rules/*.md``) file."""

    path: str
    name: str
    description: str | None
    apply_to: tuple[str, ...]
    body: str
    format: Literal["vscode", "claude"]


@dataclass(frozen=True, slots=True)
class AgentFile:
    """A parsed ``*.agent.md`` (or ``.claude/agents/*.md``) file."""

    path: str
    name: str
    description: str | None
    tools: tuple[str, ...]
    agents: tuple[str, ...] | None
    model: str | None
    user_invocable: bool
    disable_model_invocation: bool
    argument_hint: str | None
    handoffs: tuple[dict[str, Any], ...]
    body: str
    format: Literal["vscode", "claude"]


@dataclass(frozen=True, slots=True)
class PromptFile:
    """A parsed ``*.prompt.md`` file."""

    path: str
    name: str
    description: str | None
    agent: str | None
    argument_hint: str | None
    tools: tuple[str, ...]
    model: str | None
    body: str


@dataclass(frozen=True, slots=True)
class SkillFile:
    """A parsed ``SKILL.md`` file."""

    path: str
    name: str
    description: str
    argument_hint: str | None
    user_invocable: bool
    disable_model_invocation: bool
    context: Literal["inline", "fork"]
    body: str
    expected_dir_name: str | None


@dataclass(frozen=True, slots=True)
class CopilotInstructionsFile:
    """An always-on instructions file (``copilot-instructions.md``, ``AGENTS.md``, ``CLAUDE.md``)."""

    path: str
    body: str


@dataclass(frozen=True, slots=True)
class CustomizationBundle:
    """All customizations discovered in a workspace, plus diagnostics."""

    always_on: tuple[CopilotInstructionsFile, ...] = ()
    instructions: tuple[InstructionFile, ...] = ()
    agents: tuple[AgentFile, ...] = ()
    prompts: tuple[PromptFile, ...] = ()
    skills: tuple[SkillFile, ...] = ()
    diagnostics: tuple[CustomizationDiagnostic, ...] = ()

    def __len__(self) -> int:
        return (
            len(self.always_on)
            + len(self.instructions)
            + len(self.agents)
            + len(self.prompts)
            + len(self.skills)
        )


# --------------------------------------------------------------------------- #
# Glob matching for applyTo / paths
# --------------------------------------------------------------------------- #


def _split_apply_to(raw: Any) -> list[str]:
    """Split a raw ``applyTo`` value into individual glob patterns.

    Copilot accepts either a single string (possibly comma-separated for
    multiple globs) or a YAML list of strings. Returns a list with empty
    entries filtered out.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [g.strip() for g in raw.split(",") if g.strip()]
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    out.append(stripped)
        return out
    return []


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _coerce_str_list(value: Any, *, comma_split: bool = False) -> tuple[str, ...]:
    """Coerce a front-matter value into a tuple of tool/agent name strings.

    VS Code format uses a YAML array; Claude format uses a comma-separated
    string. ``comma_split=True`` enables the latter.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        if comma_split:
            return tuple(p.strip() for p in value.split(",") if p.strip())
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    out.append(stripped)
        return tuple(out)
    return ()


def matches_apply_to(apply_to: list[str] | tuple[str, ...], file_path: str) -> bool:
    """Return True if ``file_path`` matches any glob in ``apply_to``.

    Globs are interpreted the way Copilot interprets ``applyTo``: relative to
    the workspace root, with ``**`` matching any number of path segments and
    brace alternation (``a.{yml,yaml}``) supported. ``file_path`` should be a
    POSIX-style relative path (forward slashes, no leading ``./``).
    """
    if not apply_to:
        return False
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    try:
        from wcmatch import glob as _wcmatch_glob
    except ImportError:  # pragma: no cover - wcmatch is a declared dep
        return _matches_fallback(apply_to, normalized)

    flags = _wcmatch_glob.GLOBSTAR | _wcmatch_glob.BRACE
    for pattern in apply_to:
        if _wcmatch_glob.globmatch(normalized, pattern, flags=flags):
            return True
    return False


def _matches_fallback(apply_to: list[str] | tuple[str, ...], file_path: str) -> bool:
    """Pure-stdlib fallback when wcmatch is unavailable.

    Does NOT support brace alternation; callers should install wcmatch for
    full Copilot parity.
    """
    import fnmatch

    for pattern in apply_to:
        # Translate ** into fnmatch-friendly form.
        fn_pattern = pattern.replace("**/", "*/").replace("/**", "/*")
        if fnmatch.fnmatch(file_path, fn_pattern):
            return True
    return False


def matches_any_instruction(
    bundle: CustomizationBundle, file_path: str
) -> tuple[InstructionFile, ...]:
    """Return all instructions in ``bundle`` whose ``apply_to`` matches ``file_path``."""
    return tuple(
        inst
        for inst in bundle.instructions
        if matches_apply_to(list(inst.apply_to), file_path)
    )


# --------------------------------------------------------------------------- #
# Per-kind parsers
# --------------------------------------------------------------------------- #


def parse_instruction_file(
    path: str, text: str, *, fmt: Literal["vscode", "claude"] = "vscode"
) -> tuple[InstructionFile | None, CustomizationDiagnostic | None]:
    """Parse a ``*.instructions.md`` (VS Code) or ``.claude/rules/*.md`` (Claude) file.

    VS Code format uses ``applyTo`` (single string, possibly comma-separated).
    Claude format uses ``paths`` (YAML list of strings, default ``**``).
    """
    try:
        fm, body = parse_front_matter(text)
    except FrontMatterError as exc:
        return None, CustomizationDiagnostic(
            path=path, message=f"Invalid front-matter: {exc}", kind="instructions"
        )

    if fmt == "claude":
        raw_apply = fm.get("paths", ["**"])
    else:
        raw_apply = fm.get("applyTo")
    apply_to = tuple(_split_apply_to(raw_apply))

    name = _coerce_str(fm.get("name")) or _default_name_from_path(path)
    description = _coerce_str(fm.get("description"))

    return (
        InstructionFile(
            path=path,
            name=name,
            description=description,
            apply_to=apply_to,
            body=body,
            format=fmt,
        ),
        None,
    )


def parse_agent_file(
    path: str, text: str, *, fmt: Literal["vscode", "claude"] = "vscode"
) -> tuple[AgentFile | None, CustomizationDiagnostic | None]:
    """Parse a ``*.agent.md`` (VS Code) or ``.claude/agents/*.md`` (Claude) file."""
    try:
        fm, body = parse_front_matter(text)
    except FrontMatterError as exc:
        return None, CustomizationDiagnostic(
            path=path, message=f"Invalid front-matter: {exc}", kind="agents"
        )

    # Claude format: tools is a comma-separated string.
    comma_split = fmt == "claude"
    tools = _coerce_str_list(fm.get("tools"), comma_split=comma_split)
    agents_raw = fm.get("agents")
    if agents_raw == "*":
        agents: tuple[str, ...] | None = ("*",)
    else:
        agents = _coerce_str_list(agents_raw) or None

    handoffs_raw = fm.get("handoffs")
    handoffs: tuple[dict[str, Any], ...] = ()
    if isinstance(handoffs_raw, list):
        handoffs = tuple(h for h in handoffs_raw if isinstance(h, dict))

    name = _coerce_str(fm.get("name")) or _default_name_from_path(path)
    description = _coerce_str(fm.get("description"))
    model = _coerce_str(fm.get("model"))
    argument_hint = _coerce_str(fm.get("argument-hint"))
    user_invocable = _coerce_bool(fm.get("user-invocable"), default=True)
    disable_model_invocation = _coerce_bool(
        fm.get("disable-model-invocation"), default=False
    )

    return (
        AgentFile(
            path=path,
            name=name,
            description=description,
            tools=tools,
            agents=agents,
            model=model,
            user_invocable=user_invocable,
            disable_model_invocation=disable_model_invocation,
            argument_hint=argument_hint,
            handoffs=handoffs,
            body=body,
            format=fmt,
        ),
        None,
    )


def parse_prompt_file(
    path: str, text: str
) -> tuple[PromptFile | None, CustomizationDiagnostic | None]:
    """Parse a ``*.prompt.md`` file."""
    try:
        fm, body = parse_front_matter(text)
    except FrontMatterError as exc:
        return None, CustomizationDiagnostic(
            path=path, message=f"Invalid front-matter: {exc}", kind="prompts"
        )

    name = _coerce_str(fm.get("name")) or _default_name_from_path(path)
    description = _coerce_str(fm.get("description"))
    agent = _coerce_str(fm.get("agent"))
    argument_hint = _coerce_str(fm.get("argument-hint"))
    tools = _coerce_str_list(fm.get("tools"))
    model = _coerce_str(fm.get("model"))

    return (
        PromptFile(
            path=path,
            name=name,
            description=description,
            agent=agent,
            argument_hint=argument_hint,
            tools=tools,
            model=model,
            body=body,
        ),
        None,
    )


def parse_skill_file(
    path: str, text: str
) -> tuple[SkillFile | None, CustomizationDiagnostic | None]:
    """Parse a ``SKILL.md`` file, validating Copilot's skill-name constraints."""
    try:
        fm, body = parse_front_matter(text)
    except FrontMatterError as exc:
        return None, CustomizationDiagnostic(
            path=path, message=f"Invalid front-matter: {exc}", kind="skills"
        )

    name = _coerce_str(fm.get("name"))
    if name is None:
        return None, CustomizationDiagnostic(
            path=path, message="Skill is missing required 'name' front-matter.", kind="skills"
        )
    if not _SKILL_NAME_RE.match(name):
        return None, CustomizationDiagnostic(
            path=path,
            message=(
                f"Skill name {name!r} must be lowercase letters, numbers, and "
                f"hyphens, <=64 chars."
            ),
            kind="skills",
        )

    description = _coerce_str(fm.get("description"))
    if description is None:
        return None, CustomizationDiagnostic(
            path=path,
            message="Skill is missing required 'description' front-matter.",
            kind="skills",
        )
    if len(description) > _SKILL_DESCRIPTION_MAX:
        return None, CustomizationDiagnostic(
            path=path,
            message=(
                f"Skill description exceeds {_SKILL_DESCRIPTION_MAX} chars "
                f"(got {len(description)})."
            ),
            kind="skills",
        )

    argument_hint = _coerce_str(fm.get("argument-hint"))
    user_invocable = _coerce_bool(fm.get("user-invocable"), default=True)
    disable_model_invocation = _coerce_bool(
        fm.get("disable-model-invocation"), default=False
    )
    context_raw = _coerce_str(fm.get("context")) or "inline"
    context: Literal["inline", "fork"] = (
        "fork" if context_raw == "fork" else "inline"
    )

    expected_dir_name = _expected_skill_dir_name(path)
    diagnostic: CustomizationDiagnostic | None = None
    if expected_dir_name is not None and expected_dir_name != name:
        diagnostic = CustomizationDiagnostic(
            path=path,
            message=(
                f"Skill name {name!r} does not match parent directory name "
                f"{expected_dir_name!r}; Copilot will silently fail to load it."
            ),
            kind="skills",
        )

    skill = SkillFile(
        path=path,
        name=name,
        description=description,
        argument_hint=argument_hint,
        user_invocable=user_invocable,
        disable_model_invocation=disable_model_invocation,
        context=context,
        body=body,
        expected_dir_name=expected_dir_name,
    )
    return skill, diagnostic


def parse_always_on_file(
    path: str, text: str
) -> tuple[CopilotInstructionsFile, CustomizationDiagnostic | None]:
    """Parse an always-on instructions file (no front-matter schema enforced)."""
    # Always-on files may carry front-matter but it is not standardized; we
    # treat the entire file as body when no front-matter is present, and
    # strip the front-matter if present (keeping the body only).
    try:
        _fm, body = parse_front_matter(text)
    except FrontMatterError as exc:
        return (
            CopilotInstructionsFile(path=path, body=text),
            CustomizationDiagnostic(
                path=path, message=f"Invalid front-matter: {exc}", kind="always_on"
            ),
        )
    return CopilotInstructionsFile(path=path, body=body), None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _default_name_from_path(path: str) -> str:
    """Derive a display name from a file path when front-matter omits one."""
    stem = PurePosixPath(path.replace("\\", "/")).stem
    # Strip common suffixes like .instructions, .agent, .prompt.
    for suffix in (".instructions", ".agent", ".prompt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or path


def _expected_skill_dir_name(path: str) -> str | None:
    """Return the parent directory name for a SKILL.md path.

    For ``.github/skills/webapp-testing/SKILL.md`` this is ``webapp-testing``.
    Returns None if the path has no parent (defensive).
    """
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if len(parts) < 2:
        return None
    return parts[-2]