"""Tests for ``runtime.session_archive`` — the content-addressed session-source archive.

The archive exists because the continuation resume guard is fail-closed by
design: a continuation saved under plan A must not resume under plan B, and
any session-file edit between a pause and its resume changes the compiled
plan digest, stranding the paused turn with "Continuation belongs to a
different turn identity or session plan". The archive keeps the bytes behind
every digest a boot produced; these tests lock the contract that restoring
them restores the digest — including the URI scope (the digest payload
includes ``source_uri``, so the same bytes at another path compile to a
different digest).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kernel.session_plan import compile_session_plan
from runtime.session_archive import (
    archive_session_source,
    list_archived,
    read_archived,
    restore_source,
    sessions_dir,
)

_V1 = """\
---
schema_version: 2
name: archivetest
surface:
  type: personal_dm
  id: archivetest
  memory_scope: personal
execution:
  max_capability_calls: 150
---
Paused body.
"""

_V2_EDITED_LIMITS = _V1.replace("150", "500")


@pytest.fixture()
def session_setup(tmp_path: Path) -> Path:
    """Seed ``<data_dir>/config/session_context.md`` with the v1 session."""
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "session_context.md").write_text(_V1, encoding="utf-8")
    return tmp_path


def _digest_of(text: str, uri: Path | str) -> str:
    return compile_session_plan(text=text, source_uri=str(uri)).plan_digest


def test_archive_writes_source_and_meta_sidecar(session_setup: Path) -> None:
    uri = session_setup / "config" / "session_context.md"
    archive_session_source(str(uri), _V1, data_dir=session_setup)
    digest = _digest_of(_V1, uri)
    source = sessions_dir(session_setup) / f"{digest}.md"
    meta_path = sessions_dir(session_setup) / f"{digest}.meta.json"
    assert source.is_file() and source.read_text(encoding="utf-8") == _V1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["digest"] == digest
    assert meta["source_uri"] == str(uri)
    assert meta["bytes"] == len(_V1.encode("utf-8"))


def test_archive_is_idempotent_for_identical_text(session_setup: Path) -> None:
    uri = session_setup / "config" / "session_context.md"
    archive_session_source(str(uri), _V1, data_dir=session_setup)
    archive_session_source(str(uri), _V1, data_dir=session_setup)
    digest = _digest_of(_V1, uri)
    entries = list_archived(session_setup)
    assert len(entries) == 1
    assert entries[0]["digest"] == digest
    assert entries[0]["intact"] is True


def test_distinct_edits_are_distinct_entries(session_setup: Path) -> None:
    uri = session_setup / "config" / "session_context.md"
    archive_session_source(str(uri), _V1, data_dir=session_setup)
    archive_session_source(str(uri), _V2_EDITED_LIMITS, data_dir=session_setup)
    entries = list_archived(session_setup)
    assert len(entries) == 2
    digests = {entry["digest"] for entry in entries}
    assert digests == {_digest_of(_V1, uri), _digest_of(_V2_EDITED_LIMITS, uri)}


def test_changing_limits_changes_the_digest(session_setup: Path) -> None:
    uri = session_setup / "config" / "session_context.md"
    assert _digest_of(_V1, uri) != _digest_of(_V2_EDITED_LIMITS, uri)


def test_restore_by_digest_prefix_restores_exact_bytes_and_digest(
    session_setup: Path,
) -> None:
    uri = session_setup / "config" / "session_context.md"
    digest_v1 = _digest_of(_V1, uri)
    archive_session_source(str(uri), _V1, data_dir=session_setup)

    # The operator edits the session between the pause and the resume.
    uri.write_text(_V2_EDITED_LIMITS, encoding="utf-8")
    restore_source(digest_v1[:12], session_setup)

    assert uri.read_text(encoding="utf-8") == _V1, "the exact archived bytes come back"
    backup = uri.with_suffix(".md.bak")
    assert backup.read_text(encoding="utf-8") == _V2_EDITED_LIMITS, "the edit is snapshotted"
    assert _digest_of(uri.read_text(encoding="utf-8"), uri) == digest_v1


def test_restore_verifies_digest_and_rejects_inconsistency(
    session_setup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uri = session_setup / "config" / "session_context.md"
    digest = _digest_of(_V1, uri)
    archive_session_source(str(uri), _V1, data_dir=session_setup)
    # Corrupt the sidecar the way a partial write could: claim another digest.
    meta_path = sessions_dir(session_setup) / f"{digest}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["digest"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    uri.write_text(_V2_EDITED_LIMITS, encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent"):
        restore_source(digest[:12], session_setup)
    # The .bak safety net ran before the verification failure.
    assert uri.with_suffix(".md.bak").read_text(encoding="utf-8") == _V2_EDITED_LIMITS


def test_restore_unknown_or_ambiguous_prefix_raises(session_setup: Path) -> None:
    uri = session_setup / "config" / "session_context.md"
    archive_session_source(str(uri), _V1, data_dir=session_setup)
    with pytest.raises(ValueError, match="No archived session source"):
        restore_source("ffffffff", session_setup)
    # An ambiguous prefix: find another digest sharing the 1-char prefix —
    # impractical to force (hex collision), so ambiguity is covered by the
    # glob contract: a one-char prefix of a 64-hex digest matches exactly one
    # archive entry here, and 'z' matches none.
    with pytest.raises(ValueError, match="No archived session source"):
        restore_source("z", session_setup)


def test_digest_is_uri_scoped(session_setup: Path) -> None:
    """Same text at a different path compiles to a different digest.

    This is why restore replays the bytes at the recorded ``source_uri``
    rather than wherever the current boot resolves from.
    """
    d_config = _digest_of(_V1, session_setup / "config" / "session_context.md")
    d_elsewhere = _digest_of(_V1, "/other/host/config/session_context.md")
    assert d_config != d_elsewhere


def test_restore_without_recorded_uri_falls_back_to_config_file(
    session_setup: Path,
) -> None:
    uri = session_setup / "config" / "session_context.md"
    digest = _digest_of(_V1, uri)
    archive_session_source(str(uri), _V1, data_dir=session_setup)
    meta_path = sessions_dir(session_setup) / f"{digest}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["source_uri"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    uri.write_text(_V2_EDITED_LIMITS, encoding="utf-8")

    restored = restore_source(digest[:12], session_setup)
    assert restored == uri
    assert uri.read_text(encoding="utf-8") == _V1


def test_read_archived_returns_none_for_missing_store(tmp_path: Path) -> None:
    assert read_archived("0" * 64, tmp_path) is None
    assert list_archived(tmp_path) == []
