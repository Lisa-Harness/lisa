"""Boot-hook tests: every real ``HostRuntime`` boot archives its session source.

Per-message hosts (``lisa prompt``) construct a fresh ``HostRuntime`` per
turn, so the hook runs on every message; these tests prove the archive
lands, matches the booted plan, is idempotent, and never blocks startup.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from runtime.session_archive import list_archived, sessions_dir

_SESSION = """\
---
schema_version: 2
name: boothook
surface:
  type: personal_dm
  id: boothook
  memory_scope: personal
---
Boot body.
"""


def _boot(tmp_path: Path, session_text: str | None = None):
    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "models.yaml").write_text(
        "version: 1\nmode_models: {default: gemma3:latest}\n"
        "models:\n  - model_id: gemma3:latest\n    provider: ollama\n",
        encoding="utf-8",
    )
    if session_text is not None:
        (config / "session_context.md").write_text(session_text, encoding="utf-8")
    from runtime.config import HostConfig
    from runtime.runtime import HostRuntime

    return HostRuntime(HostConfig.from_env(data_dir=str(tmp_path), no_thread=True))


def test_boot_archives_the_resolved_session_source(tmp_path: Path) -> None:
    runtime = _boot(tmp_path, _SESSION)
    entries = list_archived(runtime.paths.data_dir)
    assert len(entries) == 1
    assert entries[0]["digest"] == str(runtime.session_plan.plan_digest)
    assert entries[0]["source_uri"] == str(runtime.session_plan.source_uri)
    assert entries[0]["intact"] is True


def test_boot_re_archive_is_idempotent(tmp_path: Path) -> None:
    _boot(tmp_path, _SESSION)
    _boot(tmp_path, _SESSION)
    assert len(list_archived(tmp_path)) == 1


def test_distinct_session_edits_produce_distinct_entries(tmp_path: Path) -> None:
    _boot(tmp_path, _SESSION)
    _boot(tmp_path, _SESSION.replace("Boot body.", "Edited body."))
    assert len(list_archived(tmp_path)) == 2


def test_injected_session_plan_skips_the_archive(tmp_path: Path) -> None:
    from kernel.session_plan import compile_session_plan

    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "models.yaml").write_text(
        "version: 1\nmode_models: {default: gemma3:latest}\n"
        "models:\n  - model_id: gemma3:latest\n    provider: ollama\n",
        encoding="utf-8",
    )
    plan = compile_session_plan(text=_SESSION, source_uri="injected.md")
    from runtime.config import HostConfig
    from runtime.runtime import HostRuntime

    HostRuntime(HostConfig.from_env(data_dir=str(tmp_path), no_thread=True), session_plan=plan)
    assert not sessions_dir(tmp_path).exists(), "an injected plan has no source to archive"


def test_archive_failure_never_blocks_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runtime.runtime as runtime_mod

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("archive disk full")

    monkeypatch.setattr(runtime_mod, "archive_session_source", explode)
    runtime = _boot(tmp_path, _SESSION)  # must not raise
    assert str(runtime.session_plan.plan_digest)
