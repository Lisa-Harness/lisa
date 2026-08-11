"""Tests for Copilot customization auto-injection into the system prompt.

Verifies that always-on files, applyTo-scoped instructions, and skills
are injected for CLI/web hosts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from host.context_assembly import assemble_system_prompt


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
            copilot_customizations="## Project Instructions\n\nAlways-on rules.\n\n## Instruction: Python (applies to: **/*.py)\n\nFollow PEP 8.",
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
        assert "## Skill:" not in result

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
        from host.config import HostConfig
        from host.runtime import HostRuntime

        config = HostConfig(data_dir=tmp_path)
        runtime = HostRuntime(config)
        assert runtime._copilot_customizations_text is None

    def test_copilot_customizations_text_cached_after_first_access(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from host.config import HostConfig
        from host.runtime import HostRuntime

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
        from host.config import HostConfig
        from host.runtime import HostRuntime

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
        from host.config import HostConfig
        from host.runtime import HostRuntime

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
