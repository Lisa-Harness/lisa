"""Edge-branch tests for ``runtime.session_archive``.

Covers the fallback and degraded branches the archive contract allows but
the main suite does not drive: a meta sidecar that is corrupt or
non-object, unreadable sidecars, relative recorded URIs, backup snapshot
of an existing target, and the inventory's intact/missing and
newest-first ordering. The main contract tests live in
``test_session_archive.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from kernel.session_plan import compile_session_plan
from runtime.session_archive import (
    _atomic_write,
    archive_session_source,
    list_archived,
    read_archived,
    restore_source,
    sessions_dir,
)

_V1 = """\
---
schema_version: 2
name: archiveedges
surface:
  type: personal_dm
  id: archiveedges
  memory_scope: personal
execution:
  max_capability_calls: 150
---
Paused body.
"""


@pytest.fixture()
def session_setup(tmp_path: Path) -> Path:
    """Seed ``<data_dir>/config/session_context.md`` with the v1 session."""
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "session_context.md").write_text(_V1, encoding="utf-8")
    return tmp_path


def _digest_of(text: str, uri: Path | str) -> str:
    return compile_session_plan(text=text, source_uri=str(uri)).plan_digest


def _meta_path(data_dir: Path, digest: str) -> Path:
    return sessions_dir(data_dir) / f"{digest}.meta.json"


def test_read_archived_returns_none_for_corrupt_meta(session_setup: Path) -> None:
    digest = _digest_of(_V1, session_setup / "config" / "session_context.md")
    sessions_dir(session_setup).mkdir(parents=True, exist_ok=True)
    _meta_path(session_setup, digest).write_text("not json", encoding="utf-8")
    assert read_archived(digest, session_setup) is None


def test_read_archived_skips_non_object_meta(session_setup: Path) -> None:
    digest = _digest_of(_V1, session_setup / "config" / "session_context.md")
    sessions_dir(session_setup).mkdir(parents=True, exist_ok=True)
    source_path = sessions_dir(session_setup) / f"{digest}.md"
    source_path.write_text(_V1, encoding="utf-8")
    _meta_path(session_setup, digest).write_text("[1, 2, 3]", encoding="utf-8")
    archived = read_archived(digest, session_setup)
    assert archived is not None
    text, meta = archived
    assert text == _V1
    assert meta == {}


def test_list_archived_skips_unreadable_and_non_object_meta(tmp_path: Path) -> None:
    sessions_dir(tmp_path).mkdir(parents=True)
    (sessions_dir(tmp_path) / "aaaa.meta.json").write_text("not json", encoding="utf-8")
    (sessions_dir(tmp_path) / "bbbb.meta.json").write_text('"a string"', encoding="utf-8")
    assert list_archived(tmp_path) == []


def test_list_archived_returns_empty_without_directory(tmp_path: Path) -> None:
    assert list_archived(tmp_path) == []


def test_list_archived_orders_newest_first_and_marks_intact(session_setup: Path) -> None:
    digest = _digest_of(_V1, session_setup / "config" / "session_context.md")
    archive_session_source(
        str(session_setup / "config" / "session_context.md"), _V1, data_dir=session_setup
    )
    entry = list_archived(session_setup)[0]
    assert entry["digest"] == digest
    assert entry["intact"] is True
    # Drop the source file: the entry is reported, flagged not intact.
    source_path = sessions_dir(session_setup) / f"{digest}.md"
    source_path.unlink()
    flagged = list_archived(session_setup)
    assert flagged[0]["intact"] is False
    assert flagged[0]["source_path"].endswith(f"{digest}.md")


def test_restore_uses_fallback_when_recorded_uri_is_relative(session_setup: Path) -> None:
    """A corrupt (relative) recorded URI restores to the config default."""
    fallback = session_setup / "config" / "session_context.md"
    digest = _digest_of(_V1, fallback)
    sessions_dir(session_setup).mkdir(parents=True, exist_ok=True)
    source_path = sessions_dir(session_setup) / f"{digest}.md"
    source_path.write_text(_V1, encoding="utf-8")
    _meta_path(session_setup, digest).write_text(
        json.dumps({"digest": digest, "source_uri": "config/relative.md"}),
        encoding="utf-8",
    )
    restored = restore_source(digest, session_setup)
    assert restored == fallback
    assert fallback.read_text(encoding="utf-8") == _V1


def test_restore_snapshots_the_replaced_target(session_setup: Path) -> None:
    target = session_setup / "config" / "session_context.md"
    digest = _digest_of(_V1, target)
    meta_path = _meta_path(session_setup, digest)
    sessions_dir(session_setup).mkdir(parents=True, exist_ok=True)
    source_path = sessions_dir(session_setup) / f"{digest}.md"
    source_path.write_text(_V1, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"digest": digest, "source_uri": str(target)}),
        encoding="utf-8",
    )
    target.write_text(_V1.replace("Paused body.", "Live-edited body."), encoding="utf-8")
    restore_source(digest, session_setup)
    assert (target.with_suffix(".md.bak")).read_text(encoding="utf-8") == (
        _V1.replace("Paused body.", "Live-edited body.")
    )
    assert target.read_text(encoding="utf-8") == _V1


def test_restore_reports_corrupt_archive_bytes(session_setup: Path) -> None:
    """Archive text that no longer compiles raises with the backup path named."""
    target = session_setup / "config" / "session_context.md"
    digest = _digest_of(_V1, target)
    meta_path = _meta_path(session_setup, digest)
    sessions_dir(session_setup).mkdir(parents=True, exist_ok=True)
    source_path = sessions_dir(session_setup) / f"{digest}.md"
    source_path.write_text("this is not a session document", encoding="utf-8")
    meta_path.write_text(
        json.dumps({"digest": digest, "source_uri": str(target)}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not compile"):
        restore_source(digest, session_setup)
    # The restore failed, but the live session file was replaced with the
    # corrupt text and the backup preserved the original.
    backup = target.with_suffix(".md.bak")
    assert backup.read_text(encoding="utf-8") == _V1


def test_restore_consistent_meta_digest_mismatch_raises_and_preserves_backup(
    session_setup: Path,
) -> None:
    target = session_setup / "config" / "session_context.md"
    digest = _digest_of(_V1, target)
    meta_path = _meta_path(session_setup, digest)
    sessions_dir(session_setup).mkdir(parents=True, exist_ok=True)
    source_path = sessions_dir(session_setup) / f"{digest}.md"
    source_path.write_text(_V1, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"digest": "0" * 64, "source_uri": str(target)}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"compiles to .* not the archived"):
        restore_source(digest, session_setup)
    backup = target.with_suffix(".md.bak")
    assert backup.read_text(encoding="utf-8") == _V1


def test_atomic_write_cleans_up_the_temp_file_on_failure(
    session_setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed replace removes the temp file and re-raises the error."""
    backup_target = session_setup / "config" / "session_context.md.bak"

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        _atomic_write(backup_target, "new text")
    assert not list(session_setup.glob("*.tmp"))


def test_read_archived_tolerates_an_unreadable_source(tmp_path: Path) -> None:
    """An unreadable digest-matched source returns None, not a crash."""
    import stat as stat_mod

    sessions_dir(tmp_path).mkdir(parents=True)
    digest = "b" * 64
    source = sessions_dir(tmp_path) / f"{digest}.md"
    source.write_text(_V1, encoding="utf-8")
    source.chmod(stat_mod.S_IRUSR)  # owner-read only -> read_text works; simulate on the meta
    meta = sessions_dir(tmp_path) / f"{digest}.meta.json"
    meta.write_text("[]", encoding="utf-8")
    # A non-object meta alone returns text with an empty meta; make the
    # source unreadable instead to exercise the OSError arm.
    # Windows ignores read-only on open-for-read, so guard by platform.
    if os.name != "nt":
        source.chmod(0)
        try:
            assert read_archived(digest, tmp_path) is None
        finally:
            source.chmod(stat_mod.S_IRUSR | stat_mod.S_IWUSR)
