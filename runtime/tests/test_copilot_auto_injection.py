"""Tests for Copilot customization injection.

Unconditional guidance (always-on files, instructions with no ``applyTo``, and
skills) goes into the system prompt. Instructions that declare ``applyTo`` are
held back and appended to the first pack-tool result that touches a matching
path, so the guidance tracks the files the model actually works on instead of
being paid for on every turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime.context_assembly import assemble_system_prompt


def _fake_content_loader(path: str) -> str:
    if path == "prompts/system-prompt.md":
        return "BASE SYSTEM PROMPT"
    return ""


class TestAssembleSystemPromptWithCopilotCustomizations:
    def test_copilot_customizations_injected_after_session_instruction(self) -> None:
        result = assemble_system_prompt(
            mode_instruction="MODE INSTRUCTION",
            stage_instruction=None,
            content_loader=_fake_content_loader,
            session_instruction="SESSION INSTRUCTION",
            copilot_customizations=(
                "## Project Instructions\n\nAlways-on rules.\n\n"
                "## Instruction: Python (applies to: **/*.py)\n\nFollow PEP 8."
            ),
        )

        # Order: base prompt → session → copilot → mode
        assert result.index("BASE SYSTEM PROMPT") < result.index("SESSION INSTRUCTION")
        assert result.index("SESSION INSTRUCTION") < result.index("## Project Instructions")
        assert result.index("## Project Instructions") < result.index("## Instruction: Python")
        assert result.index("## Instruction: Python") < result.index("MODE INSTRUCTION")

    def test_empty_copilot_customizations_not_injected(self) -> None:
        result = assemble_system_prompt(
            mode_instruction="MODE INSTRUCTION",
            stage_instruction=None,
            content_loader=_fake_content_loader,
            session_instruction="SESSION INSTRUCTION",
            copilot_customizations="",
        )

        assert "## Project Instructions" not in result
        assert "## Instruction:" not in result
        assert "## Skills" not in result

    def test_whitespace_only_copilot_customizations_not_injected(self) -> None:
        result = assemble_system_prompt(
            mode_instruction="MODE INSTRUCTION",
            stage_instruction=None,
            content_loader=_fake_content_loader,
            session_instruction="SESSION INSTRUCTION",
            copilot_customizations="   \n  ",
        )

        assert "## Project Instructions" not in result

    def test_stage_instruction_still_appended(self) -> None:
        result = assemble_system_prompt(
            mode_instruction="MODE INSTRUCTION",
            stage_instruction="STAGE INSTRUCTION",
            content_loader=_fake_content_loader,
            session_instruction="SESSION INSTRUCTION",
            copilot_customizations="## Project Instructions\n\nRules.",
        )

        assert "STAGE INSTRUCTION" in result
        assert result.index("MODE INSTRUCTION") < result.index("STAGE INSTRUCTION")

    def test_apply_to_metadata_present_in_output(self) -> None:
        result = assemble_system_prompt(
            mode_instruction="MODE INSTRUCTION",
            stage_instruction=None,
            content_loader=_fake_content_loader,
            session_instruction="SESSION INSTRUCTION",
            copilot_customizations="## Instruction: Python (applies to: **/*.py)\n\nFollow PEP 8.",
        )

        assert "applies to: **/*.py" in result


class TestHostRuntimeCopilotCache:
    def test_copilot_customizations_text_defaults_to_none(self, tmp_path: Path) -> None:
        from runtime.config import HostConfig
        from runtime.runtime import HostRuntime

        config = HostConfig(data_dir=tmp_path)
        runtime = HostRuntime(config)
        assert runtime._copilot_customizations_text is None

    def test_copilot_customizations_text_cached_after_first_access(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from runtime.config import HostConfig
        from runtime.runtime import HostRuntime

        (tmp_path / ".git").mkdir()
        (tmp_path / ".github").mkdir()
        (tmp_path / ".github" / "copilot-instructions.md").write_text(
            "Project rules.\n", encoding="utf-8"
        )

        config = HostConfig(data_dir=tmp_path)
        runtime = HostRuntime(config)

        # Override the session plan's pack config to point at tmp_path.
        monkeypatch.setattr(
            runtime,
            "_resolve_copilot_workspace_root",
            lambda: tmp_path,
        )

        text1 = runtime.copilot_customizations_text
        text2 = runtime.copilot_customizations_text

        # Cached — same object.
        assert text1 is text2
        assert "Project rules" in text1

    def test_copilot_customizations_text_empty_when_no_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from runtime.config import HostConfig
        from runtime.runtime import HostRuntime

        (tmp_path / ".git").mkdir()

        config = HostConfig(data_dir=tmp_path)
        runtime = HostRuntime(config)

        monkeypatch.setattr(
            runtime,
            "_resolve_copilot_workspace_root",
            lambda: tmp_path,
        )

        text = runtime.copilot_customizations_text
        assert text == ""

    def test_copilot_customizations_text_graceful_on_discovery_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from runtime.config import HostConfig
        from runtime.runtime import HostRuntime

        config = HostConfig(data_dir=tmp_path)
        runtime = HostRuntime(config)

        # Make _resolve_copilot_workspace_root raise.
        monkeypatch.setattr(
            runtime,
            "_resolve_copilot_workspace_root",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        text = runtime.copilot_customizations_text
        assert text == ""


# ── Workspace-root resolution ──────────────────────────────────────────────


def _runtime_with_plan(
    tmp_path: Path,
    *,
    context_workspace_root: str | None = None,
    pack_configuration: dict[str, Any] | None = None,
) -> Any:
    """Build a real ``HostRuntime`` whose session plan carries the given config.

    Compiles an actual session manifest so the authored ``context.workspace_root``
    field is exercised through the kernel schema rather than faked.
    """
    from kernel.session_plan import compile_session_plan
    from runtime.config import HostConfig
    from runtime.runtime import HostRuntime

    context_block = ""
    if context_workspace_root is not None:
        context_block = f"context:\n  workspace_root: {context_workspace_root}\n"

    packs_block = ""
    if pack_configuration:
        packs_block = "resources:\n  packs:\n    configuration:\n"
        for pack_name, block in pack_configuration.items():
            packs_block += f"      {pack_name}:\n"
            for key, value in block.items():
                packs_block += f"        {key}: {value}\n"

    text = (
        "---\n"
        "schema_version: 2\n"
        "name: test-session\n"
        "surface:\n"
        "  type: personal_dm\n"
        "  id: host-console\n"
        "  memory_scope: testing\n"
        "routing:\n"
        "  start_mode: default\n"
        f"{context_block}"
        f"{packs_block}"
        "---\n"
        "# Primary Session\n\nAct precisely.\n"
    )
    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")
    return HostRuntime(HostConfig(data_dir=tmp_path), session_plan=plan)


# ── Scoped versus unconditional injection ──────────────────────────────────


def _workspace_with_customizations(root: Path) -> None:
    """Write one of each customization kind into a throwaway workspace."""
    (root / ".git").mkdir()
    (root / ".github").mkdir()
    (root / ".github" / "copilot-instructions.md").write_text("ALWAYS ON RULE\n", encoding="utf-8")
    instructions = root / ".github" / "instructions"
    instructions.mkdir(parents=True)
    (instructions / "python.instructions.md").write_text(
        '---\nname: Python\napplyTo: "**/*.py"\n---\nSCOPED PYTHON BODY\n',
        encoding="utf-8",
    )
    (instructions / "docs.instructions.md").write_text(
        '---\nname: Docs\napplyTo: "docs/**"\n---\nSCOPED DOCS BODY\n',
        encoding="utf-8",
    )
    (instructions / "global.instructions.md").write_text(
        "---\nname: Global\n---\nUNSCOPED BODY\n",
        encoding="utf-8",
    )
    skills = root / ".github" / "skills" / "helper"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: helper\ndescription: A helper.\n---\nSKILL BODY\n",
        encoding="utf-8",
    )


def _runtime_for_workspace(tmp_path: Path, workspace: Path) -> Any:
    from runtime.config import HostConfig
    from runtime.runtime import HostRuntime

    runtime = HostRuntime(HostConfig(data_dir=tmp_path))
    runtime._resolve_copilot_workspace_root = lambda: workspace  # type: ignore[method-assign]
    return runtime


class TestScopedInstructionDeferral:
    """``applyTo`` selects what enters context; it is not only a label."""

    def test_scoped_bodies_are_absent_from_the_system_prompt(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        runtime = _runtime_for_workspace(tmp_path, workspace)

        text = runtime.copilot_customizations_text

        assert "SCOPED PYTHON BODY" not in text
        assert "SCOPED DOCS BODY" not in text
        # Unconditional guidance stays: the always-on file and the unscoped
        # instruction.
        assert "ALWAYS ON RULE" in text
        assert "UNSCOPED BODY" in text
        # The skill is indexed, not injected.
        assert "SKILL BODY" not in text
        assert "**helper**" in text

    def test_matching_path_returns_only_the_matching_body(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        runtime = _runtime_for_workspace(tmp_path, workspace)

        python_text = runtime.scoped_instruction_text_for_path("src/app.py")

        assert "SCOPED PYTHON BODY" in python_text
        assert "SCOPED DOCS BODY" not in python_text
        assert "applies to: **/*.py" in python_text

    def test_non_matching_suffix_returns_nothing(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        runtime = _runtime_for_workspace(tmp_path, workspace)

        assert runtime.scoped_instruction_text_for_path("README.md") == ""

    def test_directory_scoped_glob_matches_a_file_under_it(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        runtime = _runtime_for_workspace(tmp_path, workspace)

        assert "SCOPED DOCS BODY" in runtime.scoped_instruction_text_for_path("docs/guide.md")

    @pytest.mark.parametrize("path", ["", "   ", ".", "../outside.py", "src/../../x.py"])
    def test_unusable_paths_match_nothing(self, tmp_path: Path, path: str) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        runtime = _runtime_for_workspace(tmp_path, workspace)

        assert runtime.scoped_instruction_text_for_path(path) == ""

    def test_backslash_and_dot_prefix_paths_normalize(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        runtime = _runtime_for_workspace(tmp_path, workspace)

        assert "SCOPED PYTHON BODY" in runtime.scoped_instruction_text_for_path(".\\src\\app.py")

    def test_absolute_path_is_read_as_workspace_relative(self, tmp_path: Path) -> None:
        """A leading slash is stripped, not treated as traversal.

        ``applyTo`` globs are workspace-relative and the coding tools document
        relative paths, so an absolute path is out of contract. Reading it
        workspace-relative can only *miss* a directory-scoped glob (a false
        negative, never a wrong body), which is the safe direction.
        """
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        runtime = _runtime_for_workspace(tmp_path, workspace)

        assert "SCOPED PYTHON BODY" in runtime.scoped_instruction_text_for_path("/src/app.py")
        # ``**/`` globs still match; a directory-anchored one does not, because
        # the stripped form starts with ``abs/`` rather than ``docs/``.
        assert runtime.scoped_instruction_text_for_path("/abs/docs/guide.md") == ""
        assert (
            runtime.scoped_instruction_text_for_path("/docs/guide.md").count("SCOPED DOCS BODY")
            == 1
        )


class TestSkillIndex:
    """Skill bodies load on demand; only name, description, and path are eager."""

    def _runtime(self, tmp_path: Path) -> Any:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        return _runtime_for_workspace(tmp_path, workspace)

    def test_body_is_absent_and_path_is_named(self, tmp_path: Path) -> None:
        text = self._runtime(tmp_path).copilot_customizations_text

        assert "SKILL BODY" not in text
        assert "A helper." in text
        assert ".github/skills/helper/SKILL.md" in text

    def test_index_instructs_reading_rather_than_guessing(self, tmp_path: Path) -> None:
        text = self._runtime(tmp_path).copilot_customizations_text

        assert "do not guess at its" in text

    def test_model_invocation_opt_out_is_honored(self, tmp_path: Path) -> None:
        """``disable-model-invocation: true`` must not be listed or injected."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        skill = workspace / ".github" / "skills" / "slash-only"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: slash-only\ndescription: Slash only.\n"
            "disable-model-invocation: true\n---\nSLASH ONLY BODY\n",
            encoding="utf-8",
        )
        runtime = _runtime_for_workspace(tmp_path, workspace)

        text = runtime.copilot_customizations_text

        assert "SLASH ONLY BODY" not in text
        assert "slash-only" not in text
        # A listed skill is still listed.
        assert "**helper**" in text

    def test_a_skill_that_is_only_an_opt_out_emits_no_section(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / ".git").mkdir()
        skill = workspace / ".github" / "skills" / "slash-only"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: slash-only\ndescription: Slash only.\n"
            "disable-model-invocation: true\n---\nBODY\n",
            encoding="utf-8",
        )
        runtime = _runtime_for_workspace(tmp_path, workspace)

        assert "## Skills" not in runtime.copilot_customizations_text

    def test_description_whitespace_is_collapsed(self, tmp_path: Path) -> None:
        """A multi-line description would otherwise break the list shape."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        skill = workspace / ".github" / "skills" / "multi"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            '---\nname: multi\ndescription: "Line one.\n\n   Line two."\n---\nBODY\n',
            encoding="utf-8",
        )
        runtime = _runtime_for_workspace(tmp_path, workspace)

        assert "Line one. Line two." in runtime.copilot_customizations_text

    def test_no_skills_emits_no_section(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / ".git").mkdir()
        (workspace / ".github").mkdir()
        (workspace / ".github" / "copilot-instructions.md").write_text(
            "ONLY RULES\n", encoding="utf-8"
        )
        runtime = _runtime_for_workspace(tmp_path, workspace)

        text = runtime.copilot_customizations_text

        assert "ONLY RULES" in text
        assert "## Skills" not in text


class TestPackToolDecoration:
    """The dispatch hook appends matching bodies, once per dispatch token."""

    def _runtime(self, tmp_path: Path) -> Any:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        _workspace_with_customizations(workspace)
        return _runtime_for_workspace(tmp_path, workspace)

    def test_matching_path_appends_guidance(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)

        out = runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "read ok", "turn-a"
        )

        assert out.startswith("read ok")
        assert "SCOPED PYTHON BODY" in out

    def test_non_matching_path_passes_through(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)

        assert (
            runtime.decorate_pack_tool_result(
                "coding_read_file", {"path": "README.md"}, "read ok", "turn-a"
            )
            == "read ok"
        )

    def test_batched_edit_paths_are_read_from_the_record_list(self, tmp_path: Path) -> None:
        """``coding_edit_files`` nests its paths inside ``edits[]``."""
        runtime = self._runtime(tmp_path)

        out = runtime.decorate_pack_tool_result(
            "coding_edit_files",
            {"artifact_id": "a", "edits": [{"kind": "create", "path": "src/new.py"}]},
            "applied",
            "turn-a",
        )

        assert out.startswith("applied")
        assert "SCOPED PYTHON BODY" in out

    def test_batched_edit_yields_each_instruction_once(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)

        out = runtime.decorate_pack_tool_result(
            "coding_edit_files",
            {
                "artifact_id": "a",
                "edits": [
                    {"kind": "create", "path": "src/a.py"},
                    {"kind": "create", "path": "src/b.py"},
                ],
            },
            "applied",
            "turn-a",
        )

        assert out.count("SCOPED PYTHON BODY") == 1

    def test_tool_without_a_path_argument_passes_through(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)

        assert (
            runtime.decorate_pack_tool_result("coding_commit", {"message": "x"}, "ok", "turn-a")
            == "ok"
        )

    def test_same_body_is_injected_once_per_token(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)

        first = runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "one", "turn-a"
        )
        second = runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/other.py"}, "two", "turn-a"
        )

        assert "SCOPED PYTHON BODY" in first
        assert second == "two", "the same body must not be appended twice in one turn"

    def test_a_new_token_gets_the_body_again(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)

        runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "one", "turn-a"
        )
        out = runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "two", "turn-b"
        )

        assert "SCOPED PYTHON BODY" in out

    def test_a_child_token_is_not_suppressed_by_the_parent(self, tmp_path: Path) -> None:
        """A fork/delegate child must receive guidance the parent already saw."""
        runtime = self._runtime(tmp_path)

        runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "parent", "parent-token"
        )
        child = runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "child", "child-token"
        )

        assert "SCOPED PYTHON BODY" in child

    def test_parent_state_survives_an_interleaved_child(self, tmp_path: Path) -> None:
        """A child token created mid-turn must not drop the parent's history."""
        runtime = self._runtime(tmp_path)

        runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "p1", "parent-token"
        )
        runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "c1", "child-token"
        )
        again = runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "p2", "parent-token"
        )

        assert again == "p2"

    def test_two_matching_bodies_are_both_appended(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)
        (tmp_path / "ws" / ".github" / "instructions" / "all-python.instructions.md").write_text(
            '---\nname: All Python\napplyTo: "**/*.py"\n---\nSECOND PYTHON BODY\n',
            encoding="utf-8",
        )
        runtime._copilot_bundle = None
        runtime._copilot_customizations_text = None

        out = runtime.decorate_pack_tool_result(
            "coding_read_file", {"path": "src/app.py"}, "read ok", "turn-a"
        )

        assert "SCOPED PYTHON BODY" in out
        assert "SECOND PYTHON BODY" in out

    def test_dedupe_sets_are_bounded(self, tmp_path: Path) -> None:
        """A long-lived host must not retain one entry per turn forever."""
        from runtime.runtime import _MAX_TRACKED_TURNS

        runtime = self._runtime(tmp_path)
        for index in range(_MAX_TRACKED_TURNS * 3):
            runtime.decorate_pack_tool_result(
                "coding_read_file", {"path": "src/app.py"}, "x", f"turn-{index}"
            )

        assert len(runtime._scoped_instructions_injected) <= _MAX_TRACKED_TURNS

    def test_discovery_failure_does_not_break_a_tool_result(self, tmp_path: Path) -> None:
        from runtime.config import HostConfig
        from runtime.runtime import HostRuntime

        runtime = HostRuntime(HostConfig(data_dir=tmp_path))

        def _boom() -> str:
            raise RuntimeError("no workspace")

        runtime._copilot_customization_bundle = _boom  # type: ignore[method-assign]

        assert (
            runtime.decorate_pack_tool_result(
                "coding_read_file", {"path": "src/app.py"}, "read ok", "turn-a"
            )
            == "read ok"
        )


class TestWorkspaceRootResolution:
    def test_runtime_source_names_no_specific_pack(self) -> None:
        """The runtime must not hardcode a pack distribution name."""
        from runtime import runtime as runtime_module

        source = Path(runtime_module.__file__).read_text(encoding="utf-8")
        assert "lisa-pack-coding" not in source

    def test_falls_back_to_cwd_when_pack_configuration_absent(self, tmp_path: Path) -> None:
        """No pack configuration at all resolves to the working directory."""
        runtime = _runtime_with_plan(tmp_path)

        assert runtime._resolve_copilot_workspace_root() == Path.cwd()

    def test_authored_workspace_root_wins_over_pack_scan(self, tmp_path: Path) -> None:
        runtime = _runtime_with_plan(
            tmp_path,
            context_workspace_root="/authored/ws",
            pack_configuration={"lisa-pack-other": {"workspace_root": "/pack/ws"}},
        )

        # Mirror the resolver: a leading "/" re-anchors onto the current drive
        # on Windows, so a bare Path("/authored/ws") is not portable.
        assert runtime._resolve_copilot_workspace_root() == Path("/authored/ws").resolve()

    def test_pack_scan_finds_workspace_root_from_any_pack(self, tmp_path: Path) -> None:
        """A pack that is not the coding pack can declare the workspace root."""
        runtime = _runtime_with_plan(
            tmp_path,
            pack_configuration={"lisa-pack-workspace": {"workspace_root": "/pack/ws"}},
        )

        assert runtime._resolve_copilot_workspace_root() == Path("/pack/ws").resolve()

    def test_pack_scan_is_deterministic_across_declarations(self, tmp_path: Path) -> None:
        """Several declaring packs resolve to the first in sorted pack-name order."""
        runtime = _runtime_with_plan(
            tmp_path,
            pack_configuration={
                "lisa-pack-zeta": {"workspace_root": "/zeta"},
                "lisa-pack-alpha": {"workspace_root": "/alpha"},
            },
        )

        assert runtime._resolve_copilot_workspace_root() == Path("/alpha").resolve()

    def test_pack_scan_accepts_mapping_blocks(self) -> None:
        """A read-only mapping block is scanned like a plain dict.

        ``test_pack_configuration`` shows pack configuration reaching the
        runtime as ``MappingProxyType``; a ``dict``-only check would silently
        skip it and fall through to ``Path.cwd()``.
        """
        from types import MappingProxyType, SimpleNamespace

        from runtime.runtime import HostRuntime

        stub = SimpleNamespace(
            session_plan=SimpleNamespace(
                manifest=SimpleNamespace(
                    resources=SimpleNamespace(
                        packs=SimpleNamespace(
                            configuration={
                                "lisa-pack-workspace": MappingProxyType(
                                    {"workspace_root": "/pack/ws"}
                                )
                            }
                        )
                    )
                )
            )
        )

        assert HostRuntime._pack_declared_workspace_root(stub) == "/pack/ws"
