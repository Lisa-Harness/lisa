"""Tests for the Copilot customization rendering and scoped-matching layer.

These functions decide what the model is shown and when. The two scoping rules
under test are that a scoped instruction body is selected by path rather than
injected unconditionally, and that a skill contributes only an index entry so
its body can be loaded on demand.
"""

from __future__ import annotations

from typing import Any

import pytest
from sdk.content import (
    CopilotInstructionsFile,
    CustomizationBundle,
    InstructionFile,
    SkillFile,
    matches_any_scoped_instruction,
    render_scoped_instruction,
    render_skill_index,
    render_unconditional_blocks,
    scoped_instructions_for_paths,
    tool_call_paths,
    workspace_relative_posix_path,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _instruction(
    name: str,
    *,
    apply_to: tuple[str, ...] = (),
    body: str | None = None,
    path: str | None = None,
) -> InstructionFile:
    return InstructionFile(
        path=path or f".github/instructions/{name}.instructions.md",
        name=name,
        description=None,
        apply_to=apply_to,
        body=body if body is not None else f"{name.upper()} BODY",
        format="vscode",
    )


def _skill(
    name: str,
    *,
    body: str | None = None,
    description: str = "A skill.",
    disable_model_invocation: bool = False,
) -> SkillFile:
    return SkillFile(
        path=f".github/skills/{name}/SKILL.md",
        name=name,
        description=description,
        argument_hint=None,
        user_invocable=True,
        disable_model_invocation=disable_model_invocation,
        context="inline",
        body=body if body is not None else f"{name.upper()} BODY",
        expected_dir_name=name,
    )


def _bundle(**overrides: Any) -> CustomizationBundle:
    return CustomizationBundle(**overrides)


# --------------------------------------------------------------------------- #
# Tool-argument path extraction
# --------------------------------------------------------------------------- #


def test_tool_call_paths_reads_a_single_path() -> None:
    assert tool_call_paths({"path": "src/app.py"}) == ("src/app.py",)


def test_tool_call_paths_accepts_file_path() -> None:
    assert tool_call_paths({"file_path": "docs/a.md"}) == ("docs/a.md",)


def test_tool_call_paths_reads_a_record_list() -> None:
    """A batched edit carries one path per record, not a top-level argument."""
    arguments = {
        "artifact_id": "a",
        "edits": [{"path": "src/a.py"}, {"path": "docs/b.md"}],
    }

    assert tool_call_paths(arguments) == ("src/a.py", "docs/b.md")


def test_tool_call_paths_dedupes_preserving_order() -> None:
    assert tool_call_paths({"edits": [{"path": "x.py"}, {"path": "x.py"}]}) == ("x.py",)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"path": ""},
        {"path": "   "},
        {"path": 3},
        {"path": ["a", "b"]},
        {"edits": "not-a-list"},
        {"edits": [{"no_path": 1}, "not-a-dict"]},
        None,
        "not-a-dict",
    ],
)
def test_tool_call_paths_ignores_unusable_shapes(arguments: Any) -> None:
    assert tool_call_paths(arguments) == ()


# --------------------------------------------------------------------------- #
# Path normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("src/app.py", "src/app.py"),
        ("./src/app.py", "src/app.py"),
        (".\\src\\app.py", "src/app.py"),
        ("src//app.py", "src/app.py"),
        ("src/./app.py", "src/app.py"),
        ("/src/app.py", "src/app.py"),
        ("", ""),
        (".", ""),
        ("../outside.py", ""),
        ("src/../../outside.py", ""),
    ],
)
def test_workspace_relative_posix_path(raw: str, expected: str) -> None:
    assert workspace_relative_posix_path(raw) == expected


# --------------------------------------------------------------------------- #
# Scoped matching
# --------------------------------------------------------------------------- #


def test_scoped_selection_returns_only_matching_instructions() -> None:
    instructions = (
        _instruction("Python", apply_to=("**/*.py",)),
        _instruction("Docs", apply_to=("docs/**",)),
    )

    matched = scoped_instructions_for_paths(instructions, ("src/app.py",))

    assert [i.name for i in matched] == ["Python"]


def test_unscoped_instructions_never_match_a_path() -> None:
    """An instruction with no ``applyTo`` is unconditional, not path-scoped."""
    instructions = (_instruction("Global"),)

    assert scoped_instructions_for_paths(instructions, ("src/app.py",)) == ()


def test_multiple_paths_dedupe_by_instruction_path() -> None:
    instructions = (_instruction("Python", apply_to=("**/*.py",)),)

    matched = scoped_instructions_for_paths(
        instructions, ("src/a.py", "src/b.py")
    )

    assert len(matched) == 1


def test_unusable_paths_match_nothing() -> None:
    instructions = (_instruction("Python", apply_to=("**/*.py",)),)

    assert scoped_instructions_for_paths(instructions, ("", "../x.py")) == ()


def test_matches_any_scoped_instruction_wraps_a_single_path() -> None:
    instructions = (_instruction("Docs", apply_to=("docs/**",)),)

    assert matches_any_scoped_instruction(instructions, "docs/a.md")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_render_scoped_instruction_labels_the_globs() -> None:
    block = render_scoped_instruction(_instruction("Python", apply_to=("**/*.py",)))

    assert "## Instruction: Python (applies to: **/*.py)" in block
    assert "PYTHON BODY" in block


def test_skill_index_renders_name_description_and_path_only() -> None:
    block = render_skill_index((_skill("golang", description="Go conventions."),))

    assert "**golang**" in block
    assert "Go conventions." in block
    assert ".github/skills/golang/SKILL.md" in block
    assert "GOLANG BODY" not in block, "the body must not be eager"


def test_skill_index_honors_disable_model_invocation() -> None:
    block = render_skill_index(
        (
            _skill("listed"),
            _skill("slash-only", disable_model_invocation=True),
        )
    )

    assert "listed" in block
    assert "slash-only" not in block


def test_skill_index_is_empty_when_every_skill_opts_out() -> None:
    assert render_skill_index((_skill("x", disable_model_invocation=True),)) == ""


def test_skill_index_skips_a_skill_with_no_body() -> None:
    assert render_skill_index((_skill("empty", body="   "),)) == ""


def test_skill_index_collapses_multi_line_descriptions() -> None:
    block = render_skill_index((_skill("multi", description="One.\n\n   Two."),))

    assert "One. Two." in block


def test_unconditional_blocks_carry_rules_and_skip_scoped_bodies() -> None:
    bundle = _bundle(
        always_on=(CopilotInstructionsFile(path="AGENTS.md", body="ALWAYS ON"),),
        instructions=(
            _instruction("Global"),
            _instruction("Python", apply_to=("**/*.py",)),
        ),
        skills=(_skill("helper"),),
    )

    blocks = "\n\n".join(render_unconditional_blocks(bundle))

    assert "ALWAYS ON" in blocks
    assert "GLOBAL BODY" in blocks
    assert "PYTHON BODY" not in blocks
    assert "**helper**" in blocks
    assert "HELPER BODY" not in blocks


def test_unconditional_blocks_are_empty_for_an_empty_bundle() -> None:
    assert render_unconditional_blocks(_bundle()) == []
