"""Tests for Copilot customization auto-injection into the system prompt.

Verifies that always-on files, applyTo-scoped instructions, and skills
are injected for CLI/web hosts.
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

        assert runtime._resolve_copilot_workspace_root() == Path("/authored/ws")

    def test_pack_scan_finds_workspace_root_from_any_pack(self, tmp_path: Path) -> None:
        """A pack that is not the coding pack can declare the workspace root."""
        runtime = _runtime_with_plan(
            tmp_path,
            pack_configuration={"lisa-pack-workspace": {"workspace_root": "/pack/ws"}},
        )

        assert runtime._resolve_copilot_workspace_root() == Path("/pack/ws")

    def test_pack_scan_is_deterministic_across_declarations(self, tmp_path: Path) -> None:
        """Several declaring packs resolve to the first in sorted pack-name order."""
        runtime = _runtime_with_plan(
            tmp_path,
            pack_configuration={
                "lisa-pack-zeta": {"workspace_root": "/zeta"},
                "lisa-pack-alpha": {"workspace_root": "/alpha"},
            },
        )

        assert runtime._resolve_copilot_workspace_root() == Path("/alpha")

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
