"""Tests for the Copilot customization parsing and discovery layer.

Covers per-kind front-matter parsing, glob matching, validation diagnostics,
and the workspace discovery walker. Uses ``tmp_path`` fixtures so no real
workspace state is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sdk.content import (
    CustomizationBundle,
    InstructionFile,
    discover_copilot_customizations,
    matches_any_instruction,
    matches_apply_to,
    parse_agent_file,
    parse_always_on_file,
    parse_instruction_file,
    parse_prompt_file,
    parse_skill_file,
)

# --------------------------------------------------------------------------- #
# Instruction files
# --------------------------------------------------------------------------- #


class TestParseInstructionFile:
    def test_parses_vscode_format(self) -> None:
        text = (
            "---\n"
            "name: 'Python Standards'\n"
            "description: 'Coding conventions for Python files'\n"
            "applyTo: '**/*.py'\n"
            "---\n"
            "# Python coding standards\n"
            "Follow PEP 8.\n"
        )
        parsed, diag = parse_instruction_file(".github/instructions/python.md", text)
        assert diag is None
        assert parsed is not None
        assert parsed.name == "Python Standards"
        assert parsed.description == "Coding conventions for Python files"
        assert parsed.apply_to == ("**/*.py",)
        assert parsed.format == "vscode"
        assert "PEP 8" in parsed.body

    def test_parses_comma_separated_apply_to(self) -> None:
        text = "---\napplyTo: '**/operations/**/*.py,**/modes/**/*.py'\n---\nbody\n"
        parsed, _ = parse_instruction_file(".github/instructions/pack.md", text)
        assert parsed is not None
        assert parsed.apply_to == (
            "**/operations/**/*.py",
            "**/modes/**/*.py",
        )

    def test_parses_claude_format_paths_array(self) -> None:
        text = "---\npaths: ['**/*.ts']\n---\nbody\n"
        parsed, _ = parse_instruction_file(".claude/rules/typescript.md", text, fmt="claude")
        assert parsed is not None
        assert parsed.format == "claude"
        assert parsed.apply_to == ("**/*.ts",)

    def test_claude_format_defaults_to_all_when_paths_missing(self) -> None:
        text = "---\nname: notes\n---\nbody\n"
        parsed, _ = parse_instruction_file(".claude/rules/notes.md", text, fmt="claude")
        assert parsed is not None
        assert parsed.apply_to == ("**",)

    def test_missing_apply_to_yields_empty_tuple(self) -> None:
        text = "---\nname: x\n---\nbody\n"
        parsed, _ = parse_instruction_file(".github/instructions/x.md", text)
        assert parsed is not None
        assert parsed.apply_to == ()

    def test_invalid_yaml_returns_diagnostic(self) -> None:
        text = "---\napplyTo: [unterminated\n---\nbody\n"
        parsed, diag = parse_instruction_file(".github/instructions/bad.md", text)
        assert parsed is None
        assert diag is not None
        assert diag.kind == "instructions"

    def test_no_front_matter_treats_all_as_body(self) -> None:
        text = "Just some inline instructions.\n"
        parsed, _ = parse_instruction_file(".github/instructions/plain.md", text)
        assert parsed is not None
        assert "inline instructions" in parsed.body
        assert parsed.apply_to == ()

    def test_derives_name_from_path_when_missing(self) -> None:
        text = "---\napplyTo: '**/*.py'\n---\nbody\n"
        parsed, _ = parse_instruction_file(".github/instructions/python.instructions.md", text)
        assert parsed is not None
        assert parsed.name == "python"


# --------------------------------------------------------------------------- #
# Agent files
# --------------------------------------------------------------------------- #


class TestParseAgentFile:
    def test_parses_vscode_format_with_yaml_array_tools(self) -> None:
        text = (
            "---\n"
            "name: 'Refactor Engineer'\n"
            "description: 'Refactor agent'\n"
            "tools: [vscode, edit, search]\n"
            "user-invocable: true\n"
            "argument-hint: 'file path or glob'\n"
            "---\n"
            "# Refactor\n"
            "Steps here.\n"
        )
        parsed, diag = parse_agent_file(".github/agents/refactor.agent.md", text)
        assert diag is None
        assert parsed is not None
        assert parsed.name == "Refactor Engineer"
        assert parsed.tools == ("vscode", "edit", "search")
        assert parsed.user_invocable is True
        assert parsed.argument_hint == "file path or glob"
        assert parsed.format == "vscode"

    def test_parses_claude_format_with_comma_string_tools(self) -> None:
        text = "---\ntools: 'Read, Grep, Glob, Bash'\n---\nbody\n"
        parsed, _ = parse_agent_file(".claude/agents/researcher.md", text, fmt="claude")
        assert parsed is not None
        assert parsed.format == "claude"
        assert parsed.tools == ("Read", "Grep", "Glob", "Bash")

    def test_parses_handoffs_list(self) -> None:
        text = (
            "---\n"
            "handoffs:\n"
            "  - label: 'Start Implementation'\n"
            "    agent: implementation\n"
            "    send: false\n"
            "---\n"
            "body\n"
        )
        parsed, _ = parse_agent_file(".github/agents/plan.agent.md", text)
        assert parsed is not None
        assert len(parsed.handoffs) == 1
        assert parsed.handoffs[0]["agent"] == "implementation"

    def test_agents_star_allows_all_subagents(self) -> None:
        text = "---\nagents: '*'\n---\nbody\n"
        parsed, _ = parse_agent_file(".github/agents/orchestrator.agent.md", text)
        assert parsed is not None
        assert parsed.agents == ("*",)

    def test_disable_model_invocation_default_false(self) -> None:
        text = "---\nname: x\n---\nbody\n"
        parsed, _ = parse_agent_file(".github/agents/x.agent.md", text)
        assert parsed is not None
        assert parsed.disable_model_invocation is False

    def test_invalid_yaml_returns_diagnostic(self) -> None:
        text = "---\ntools: [broken\n---\nbody\n"
        parsed, diag = parse_agent_file(".github/agents/bad.agent.md", text)
        assert parsed is None
        assert diag is not None
        assert diag.kind == "agents"


# --------------------------------------------------------------------------- #
# Prompt files
# --------------------------------------------------------------------------- #


class TestParsePromptFile:
    def test_parses_prompt_with_agent(self) -> None:
        text = (
            "---\n"
            "description: 'Review code'\n"
            "agent: 'Code Reviewer'\n"
            "---\n"
            "Resolve the changeset and hand off.\n"
        )
        parsed, diag = parse_prompt_file(".github/prompts/code-review.prompt.md", text)
        assert diag is None
        assert parsed is not None
        assert parsed.agent == "Code Reviewer"
        assert parsed.description == "Review code"

    def test_missing_agent_returns_none(self) -> None:
        text = "---\ndescription: 'x'\n---\nbody\n"
        parsed, _ = parse_prompt_file(".github/prompts/x.prompt.md", text)
        assert parsed is not None
        assert parsed.agent is None


# --------------------------------------------------------------------------- #
# Skill files
# --------------------------------------------------------------------------- #


class TestParseSkillFile:
    def test_parses_valid_skill(self) -> None:
        text = (
            "---\n"
            "name: webapp-testing\n"
            "description: 'Skill for testing webapps.'\n"
            "argument-hint: '[target page]'\n"
            "context: fork\n"
            "---\n"
            "# Webapp testing\n"
            "Steps.\n"
        )
        parsed, diag = parse_skill_file(".github/skills/webapp-testing/SKILL.md", text)
        assert parsed is not None
        assert diag is None
        assert parsed.name == "webapp-testing"
        assert parsed.context == "fork"
        assert parsed.argument_hint == "[target page]"

    def test_context_inline_when_unspecified(self) -> None:
        text = "---\nname: x\ndescription: 'd'\n---\nbody\n"
        parsed, _ = parse_skill_file(".github/skills/x/SKILL.md", text)
        assert parsed is not None
        assert parsed.context == "inline"

    def test_missing_name_returns_diagnostic(self) -> None:
        text = "---\ndescription: 'd'\n---\nbody\n"
        parsed, diag = parse_skill_file(".github/skills/x/SKILL.md", text)
        assert parsed is None
        assert diag is not None
        assert "name" in diag.message

    def test_invalid_name_returns_diagnostic(self) -> None:
        text = "---\nname: 'Bad Name!'\ndescription: 'd'\n---\nbody\n"
        parsed, diag = parse_skill_file(".github/skills/bad/SKILL.md", text)
        assert parsed is None
        assert diag is not None
        assert "lowercase" in diag.message

    def test_missing_description_returns_diagnostic(self) -> None:
        text = "---\nname: x\n---\nbody\n"
        parsed, diag = parse_skill_file(".github/skills/x/SKILL.md", text)
        assert parsed is None
        assert diag is not None
        assert "description" in diag.message

    def test_oversized_description_returns_diagnostic(self) -> None:
        text = "---\nname: x\ndescription: '" + ("x" * 2000) + "'\n---\nbody\n"
        parsed, diag = parse_skill_file(".github/skills/x/SKILL.md", text)
        assert parsed is None
        assert diag is not None
        assert "exceeds" in diag.message

    def test_name_dir_mismatch_emits_diagnostic_but_parses(self) -> None:
        text = "---\nname: right-name\ndescription: 'd'\n---\nbody\n"
        parsed, diag = parse_skill_file(".github/skills/wrong-dir/SKILL.md", text)
        # The skill still parses (we keep the data) but a diagnostic is emitted.
        assert parsed is not None
        assert diag is not None
        assert "wrong-dir" in diag.message
        assert parsed.expected_dir_name == "wrong-dir"


# --------------------------------------------------------------------------- #
# Always-on files
# --------------------------------------------------------------------------- #


class TestParseAlwaysOnFile:
    def test_strips_front_matter_if_present(self) -> None:
        text = "---\nsome: frontmatter\n---\nKeep this body.\n"
        parsed, diag = parse_always_on_file(".github/copilot-instructions.md", text)
        assert diag is None
        assert parsed.body == "Keep this body."

    def test_no_front_matter_keeps_entire_text(self) -> None:
        text = "Just instructions.\n"
        parsed, _ = parse_always_on_file("AGENTS.md", text)
        # When no front-matter is present, parse_front_matter returns the
        # raw text unchanged (including trailing newline).
        assert parsed.body == text


# --------------------------------------------------------------------------- #
# Glob matching
# --------------------------------------------------------------------------- #


class TestMatchesApplyTo:
    def test_exact_glob_match(self) -> None:
        assert matches_apply_to(["**/*.py"], "kernel/ipsar_fsm.py") is True

    def test_brace_alternation(self) -> None:
        assert matches_apply_to(["**/*.{yaml,yml}"], "artifacts/models.yaml") is True
        assert matches_apply_to(["**/*.{yaml,yml}"], "artifacts/models.yml") is True

    def test_comma_list(self) -> None:
        patterns = ["**/operations/**/*.py", "**/modes/**/*.py"]
        assert matches_apply_to(patterns, "sdk/packs/core/modes/coding/_mode.md") is False
        assert matches_apply_to(patterns, "sdk/packs/core/modes/coding/file.py") is True

    def test_no_match(self) -> None:
        assert matches_apply_to(["**/*.py"], "docs/readme.md") is False

    def test_empty_patterns(self) -> None:
        assert matches_apply_to([], "any/file.py") is False

    def test_normalizes_backslashes(self) -> None:
        assert matches_apply_to(["**/*.py"], "kernel\\ipsar_fsm.py") is True

    def test_strips_leading_dot_slash(self) -> None:
        assert matches_apply_to(["src/**"], "./src/main.py") is True


class TestMatchesAnyInstruction:
    def test_filters_bundle_by_apply_to(self) -> None:
        bundle = CustomizationBundle(
            instructions=(
                InstructionFile(
                    path=".github/instructions/python.md",
                    name="python",
                    description=None,
                    apply_to=("**/*.py",),
                    body="",
                    format="vscode",
                ),
                InstructionFile(
                    path=".github/instructions/md.md",
                    name="md",
                    description=None,
                    apply_to=("**/*.md",),
                    body="",
                    format="vscode",
                ),
            )
        )
        matched = matches_any_instruction(bundle, "kernel/ipsar_fsm.py")
        assert len(matched) == 1
        assert matched[0].name == "python"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _write(workspace: Path, rel: str, content: str = "") -> Path:
    """Write a file relative to ``workspace`` and return the full path."""
    full = workspace / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(
        content if content else f"---\nname: {full.stem}\n---\nbody\n", encoding="utf-8"
    )
    return full


class TestDiscoverCopilotCustomizations:
    def test_empty_workspace_returns_empty_bundle(self, tmp_path: Path) -> None:
        bundle = discover_copilot_customizations(tmp_path)
        assert len(bundle) == 0
        assert bundle.diagnostics == ()

    def test_discovers_all_kinds(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            ".github/copilot-instructions.md",
            "Always-on project rules.\n",
        )
        _write(
            tmp_path,
            "AGENTS.md",
            "Agent rules.\n",
        )
        _write(
            tmp_path,
            ".github/instructions/python.instructions.md",
            "---\nname: 'Python'\napplyTo: '**/*.py'\n---\nbody\n",
        )
        _write(
            tmp_path,
            ".github/agents/reviewer.agent.md",
            "---\nname: 'Reviewer'\ntools: [edit]\n---\nbody\n",
        )
        _write(
            tmp_path,
            ".github/prompts/review.prompt.md",
            "---\ndescription: 'd'\nagent: 'Reviewer'\n---\nbody\n",
        )
        _write(
            tmp_path,
            ".github/skills/webapp-testing/SKILL.md",
            "---\nname: webapp-testing\ndescription: 'd'\n---\nbody\n",
        )

        bundle = discover_copilot_customizations(tmp_path)

        assert len(bundle.always_on) == 2  # copilot-instructions.md + AGENTS.md
        assert len(bundle.instructions) == 1
        assert bundle.instructions[0].name == "Python"
        assert len(bundle.agents) == 1
        assert bundle.agents[0].name == "Reviewer"
        assert len(bundle.prompts) == 1
        assert len(bundle.skills) == 1
        assert bundle.skills[0].name == "webapp-testing"

    def test_discovers_claude_format(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            ".claude/rules/typescript.md",
            "---\npaths: ['**/*.ts']\n---\nbody\n",
        )
        _write(
            tmp_path,
            ".claude/agents/researcher.md",
            "---\ntools: 'Read, Grep'\n---\nbody\n",
        )
        _write(
            tmp_path,
            ".claude/skills/researcher/SKILL.md",
            "---\nname: researcher\ndescription: 'd'\n---\nbody\n",
        )

        bundle = discover_copilot_customizations(tmp_path)
        assert len(bundle.instructions) == 1
        assert bundle.instructions[0].format == "claude"
        assert len(bundle.agents) == 1
        assert bundle.agents[0].format == "claude"
        assert bundle.agents[0].tools == ("Read", "Grep")
        assert len(bundle.skills) == 1

    def test_discovers_claude_md_variants(self, tmp_path: Path) -> None:
        _write(tmp_path, "CLAUDE.md", "root CLAUDE\n")
        _write(tmp_path, ".claude/CLAUDE.md", "claude folder\n")
        _write(tmp_path, "CLAUDE.local.md", "local\n")

        bundle = discover_copilot_customizations(tmp_path)
        assert len(bundle.always_on) == 3

    def test_nested_agents_md_disabled_by_default(self, tmp_path: Path) -> None:
        _write(tmp_path, "AGENTS.md", "root\n")
        _write(tmp_path, "packages/frontend/AGENTS.md", "frontend\n")

        bundle = discover_copilot_customizations(tmp_path)
        # Only the root AGENTS.md is picked up by default.
        assert len(bundle.always_on) == 1

    def test_nested_agents_md_enabled(self, tmp_path: Path) -> None:
        _write(tmp_path, "AGENTS.md", "root\n")
        _write(tmp_path, "packages/frontend/AGENTS.md", "frontend\n")

        bundle = discover_copilot_customizations(tmp_path, nested_agents_md=True)
        assert len(bundle.always_on) == 2

    def test_skill_with_name_dir_mismatch_collects_diagnostic(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            ".github/skills/wrong-dir/SKILL.md",
            "---\nname: right-name\ndescription: 'd'\n---\nbody\n",
        )
        bundle = discover_copilot_customizations(tmp_path)
        assert len(bundle.skills) == 1
        assert len(bundle.diagnostics) == 1
        assert "wrong-dir" in bundle.diagnostics[0].message

    def test_recursively_discovers_instructions_in_subfolders(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            ".github/instructions/frontend/react.instructions.md",
            "---\nname: 'React'\napplyTo: '**/*.tsx'\n---\nbody\n",
        )
        _write(
            tmp_path,
            ".github/instructions/backend/api.instructions.md",
            "---\nname: 'API'\napplyTo: '**/*.py'\n---\nbody\n",
        )
        bundle = discover_copilot_customizations(tmp_path)
        names = sorted(i.name for i in bundle.instructions)
        assert names == ["API", "React"]

    def test_repository_corpus_discovers_taxonomy_files(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        bundle = discover_copilot_customizations(repo_root)

        assert len(bundle.always_on) >= 1
        expected_instruction_suffixes = (
            "engineering/debugging.instructions.md",
            "engineering/pack-content.instructions.md",
            "engineering/workflow.instructions.md",
            "architecture/runtime.instructions.md",
            "architecture/kernel.instructions.md",
            "architecture/mcp.instructions.md",
            "architecture/packs.instructions.md",
            "architecture/sdk.instructions.md",
        )
        discovered_paths = tuple(inst.path for inst in bundle.instructions)
        missing = [
            suffix
            for suffix in expected_instruction_suffixes
            if not any(path.endswith(suffix) for path in discovered_paths)
        ]
        assert not missing
        assert all(diag is None or diag.kind != "instructions" for diag in bundle.diagnostics)

    def test_deduplicates_when_seen_twice(self, tmp_path: Path) -> None:
        # include_parents walks up to repo root; if workspace_root IS the
        # repo root (has .git), it appears once. Create a .git dir so the
        # walker stops at tmp_path and doesn't add parent dirs.
        (tmp_path / ".git").mkdir(exist_ok=True)
        _write(
            tmp_path,
            ".github/instructions/x.instructions.md",
            "---\nname: 'X'\napplyTo: '**/*.py'\n---\nbody\n",
        )
        bundle = discover_copilot_customizations(tmp_path, include_parents=True)
        assert len(bundle.instructions) == 1

    def test_include_parents_walks_up_to_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build a monorepo: repo_root / packages / frontend
        repo_root = tmp_path / "repo"
        frontend = repo_root / "packages" / "frontend"
        frontend.mkdir(parents=True)
        (repo_root / ".git").mkdir(exist_ok=True)
        _write(
            repo_root,
            ".github/instructions/repo.instructions.md",
            "---\nname: 'RepoLevel'\napplyTo: '**/*.py'\n---\nbody\n",
        )
        _write(
            frontend,
            ".github/instructions/frontend.instructions.md",
            "---\nname: 'Frontend'\napplyTo: '**/*.tsx'\n---\nbody\n",
        )

        bundle = discover_copilot_customizations(frontend, include_parents=True)
        names = sorted(i.name for i in bundle.instructions)
        assert names == ["Frontend", "RepoLevel"]
