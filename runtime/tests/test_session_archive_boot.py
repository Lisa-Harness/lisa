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


def test_reresolution_failure_skips_the_archive_without_failing_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.session_archive import resolve_boot_source

    def explode(**kwargs: object) -> object:
        raise RuntimeError("directive resolution failed")

    monkeypatch.setattr("runtime.session_from_resolver.resolve_session_from_directives", explode)
    out = resolve_boot_source(
        session_context=_SESSION,
        data_dir=tmp_path,  # type: ignore[arg-type]
    )
    assert out is None


def test_boot_archive_swallows_unexpected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception from re-resolution is logged; the boot hook never raises."""
    import runtime.runtime as runtime_mod

    def explode(**kwargs: object) -> object:
        raise OSError("config read failed")

    monkeypatch.setattr(runtime_mod, "resolve_boot_source", explode)
    # Must not raise:
    runtime_mod._archive_boot_session_plan(
        session_context=_SESSION,
        data_dir=tmp_path,
        plan_digest="1" * 64,
    )


def test_boot_archive_skips_when_the_digest_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-resolved source whose digest no longer matches the boot is not archived."""
    import runtime.runtime as runtime_mod

    archived: list[tuple[str, str]] = []

    def fake_archive(source_uri: str, resolved_text: str, *, data_dir: Path) -> Path:
        archived.append((source_uri, resolved_text))
        return data_dir / "ignored.md"

    monkeypatch.setattr(runtime_mod, "archive_session_source", fake_archive)
    monkeypatch.setattr(
        runtime_mod,
        "_digest_of_resolved",
        lambda source_uri, resolved_text: "0" * 64,
    )

    def failing_resolve_boot_source(*, session_context: str, data_dir: Path) -> tuple[str, str]:
        return str(data_dir / "config" / "session_context.md"), _SESSION

    monkeypatch.setattr(
        runtime_mod,
        "resolve_boot_source",
        failing_resolve_boot_source,
    )
    runtime_mod._archive_boot_session_plan(
        session_context=_SESSION,
        data_dir=tmp_path,
        plan_digest="1" * 64,
    )
    assert archived == []
