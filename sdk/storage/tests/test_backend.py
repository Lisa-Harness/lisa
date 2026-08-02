"""Tests for the StorageBackend — sandbox resolution, deny-glob policy,
checkpoint persistence, and model-manifest read/write.

Covers the same behaviors the lisa-storage MCP server exposes, but
exercised directly against the SDK backend without MCPServer. These are
the in-process contract hosts and tools depend on when they use storage
without spawning the MCP server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdk.storage import PathRejected, StorageBackend

# ── Helpers ──────────────────────────────────────────────────────────────────


def _backend(
    tmp_path: Path, *, readonly: bool = False, deny_globs: tuple[str, ...] = ()
) -> StorageBackend:
    return StorageBackend(tmp_path, readonly=readonly, deny_globs=deny_globs)


def _readonly_backend(tmp_path: Path) -> StorageBackend:
    # Pre-create the files dir so reads don't error on a missing root.
    (tmp_path / "files").mkdir(parents=True, exist_ok=True)
    return StorageBackend(tmp_path, readonly=True)


# ── Construction ──────────────────────────────────────────────────────────────


def test_init_creates_subdirs(tmp_path: Path) -> None:
    StorageBackend(tmp_path)
    assert (tmp_path / "files").is_dir()
    assert (tmp_path / "checkpoints").is_dir()


def test_init_readonly_does_not_create_subdirs(tmp_path: Path) -> None:
    StorageBackend(tmp_path, readonly=True)
    assert not (tmp_path / "files").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_roots_and_manifest_path(tmp_path: Path) -> None:
    b = StorageBackend(tmp_path)
    assert b.files_root() == tmp_path / "files"
    assert b.checkpoints_root() == tmp_path / "checkpoints"
    assert b.manifest_path("active") == tmp_path / "models.yaml"
    assert b.manifest_path("proposal") == tmp_path / "models.proposed.yaml"
    assert b.manifest_path("  Proposal  ") == tmp_path / "models.proposed.yaml"


def test_manifest_path_rejects_unknown(tmp_path: Path) -> None:
    b = StorageBackend(tmp_path)
    with pytest.raises(ValueError, match="manifest kind"):
        b.manifest_path("bogus")


# ── File I/O ─────────────────────────────────────────────────────────────────


def test_write_read_roundtrip(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    w = json.loads(b.write_file("reports/daily.md", "# Hello"))
    assert w["ok"] is True
    assert w["bytes"] > 0
    assert b.read_file("reports/daily.md") == "# Hello"


def test_read_absent_returns_error(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.read_file("nonexistent.txt"))
    assert "error" in data


def test_traversal_rejected(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.read_file("../../../etc/passwd"))
    assert "error" in data


def test_absolute_path_rejected(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    abs_path = str(Path("evil.txt").resolve())
    data = json.loads(b.write_file(abs_path, "bad"))
    assert "error" in data


def test_list_files(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("a.txt", "a")
    b.write_file("sub/b.txt", "b")
    files = json.loads(b.list_files(""))
    assert "a.txt" in files
    assert "sub/b.txt" in files


def test_list_files_with_prefix(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("a.txt", "a")
    b.write_file("sub/b.txt", "b")
    files = json.loads(b.list_files("sub"))
    assert "sub/b.txt" in files
    assert "a.txt" not in files


def test_delete_file(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("to_delete.txt", "x")
    data = json.loads(b.delete_file("to_delete.txt"))
    assert data["ok"] is True
    assert json.loads(b.read_file("to_delete.txt"))["error"]


def test_delete_file_missing(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.delete_file("nope.txt"))
    assert "error" in data


def test_stat_file(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("stat_me.txt", "12345")
    data = json.loads(b.stat_file("stat_me.txt"))
    assert data["size"] == 5
    assert data["is_dir"] is False


def test_make_dir(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.make_dir("new/dir"))
    assert data["ok"] is True
    assert (b.files_root() / "new" / "dir").is_dir()


def test_read_file_range_roundtrip(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "0123456789")
    data = json.loads(b.read_file_range("range.txt", start=2, length=4))
    assert data["content"] == "2345"
    assert data["start"] == 2
    assert data["length"] == 4
    assert data["total"] == 10
    assert data["eof"] is False


def test_read_file_range_eof(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "0123456789")
    data = json.loads(b.read_file_range("range.txt", start=8, length=10))
    assert data["content"] == "89"
    assert data["length"] == 2
    assert data["eof"] is True


def test_read_file_range_start_past_end(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "abc")
    data = json.loads(b.read_file_range("range.txt", start=50, length=10))
    assert data["content"] == ""
    assert data["length"] == 0
    assert data["total"] == 3
    assert data["eof"] is True


def test_read_file_range_char_safe_multibyte(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    # "á" is one codepoint but 2 UTF-8 bytes; char slicing must not split it.
    text = "xáy"
    b.write_file("multi.txt", text)
    data = json.loads(b.read_file_range("multi.txt", start=1, length=1))
    assert data["content"] == "á"
    assert data["total"] == 3


def test_read_file_range_defaults_from_zero(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "hello")
    data = json.loads(b.read_file_range("range.txt"))
    assert data["start"] == 0
    assert data["content"] == "hello"
    assert data["eof"] is True


def test_read_file_range_negative_start_coerced(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "hello")
    data = json.loads(b.read_file_range("range.txt", start=-5, length=3))
    assert data["start"] == 0
    assert data["content"] == "hel"


def test_read_file_range_absent_returns_error(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.read_file_range("nonexistent.txt"))
    assert "error" in data


def test_read_file_range_traversal_rejected(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.read_file_range("../../../etc/passwd"))
    assert "error" in data


def test_append_file_creates_and_appends(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    first = json.loads(b.append_file("log.txt", "line1\n"))
    assert first["ok"] is True
    assert first["bytes"] == len("line1\n".encode("utf-8"))
    second = json.loads(b.append_file("log.txt", "line2\n"))
    assert second["ok"] is True
    assert second["bytes"] == len("line1\nline2\n".encode("utf-8"))
    assert b.read_file("log.txt") == "line1\nline2\n"


def test_append_file_creates_parent_dirs(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.append_file("sub/dir/log.txt", "x"))
    assert data["ok"] is True
    assert b.read_file("sub/dir/log.txt") == "x"


def test_append_file_readonly_rejected(tmp_path: Path) -> None:
    b = _readonly_backend(tmp_path)
    data = json.loads(b.append_file("log.txt", "x"))
    assert data["error"] == "server is read-only"


def test_append_file_traversal_rejected(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.append_file("../escape.txt", "x"))
    assert "error" in data


def test_readonly_rejects_writes(tmp_path: Path) -> None:
    b = _readonly_backend(tmp_path)
    assert json.loads(b.write_file("a.txt", "a"))["error"] == "server is read-only"
    assert json.loads(b.delete_file("a.txt"))["error"] == "server is read-only"
    assert json.loads(b.make_dir("d"))["error"] == "server is read-only"


# ── Deny-glob policy ────────────────────────────────────────────────────────


def _denied_backend(tmp_path: Path) -> StorageBackend:
    return StorageBackend(tmp_path, deny_globs=("*.db", "threads/**", "repl_history.txt"))


def test_resolve_raises_path_rejected(tmp_path: Path) -> None:
    b = _denied_backend(tmp_path)
    with pytest.raises(PathRejected):
        b.resolve_file_path("evil.db")


def test_resolve_or_error_denied(tmp_path: Path) -> None:
    b = _denied_backend(tmp_path)
    resolved, err = b.resolve_or_error("evil.db")
    assert resolved is None
    data = json.loads(err)  # type: ignore[arg-type]
    assert data["error"] == "path denied by policy"


def test_write_denied(tmp_path: Path) -> None:
    b = _denied_backend(tmp_path)
    data = json.loads(b.write_file("threads/x.log", "x"))
    assert data["error"] == "path denied by policy"


def test_list_files_filters_denied(tmp_path: Path) -> None:
    b = _denied_backend(tmp_path)
    b.write_file("keep.txt", "ok")
    b.write_file("secret.db", "no")
    files = json.loads(b.list_files(""))
    assert "keep.txt" in files
    assert "secret.db" not in files


def test_list_files_prefix_denied(tmp_path: Path) -> None:
    b = _denied_backend(tmp_path)
    data = json.loads(b.list_files("repl_history.txt"))
    assert data["error"] == "prefix denied by policy"


def test_list_files_ancestor_prefix_not_denied(tmp_path: Path) -> None:
    # An ancestor prefix of denied files does not match the glob, so
    # list_files returns [] (filtered), not an error.
    b = _denied_backend(tmp_path)
    files = json.loads(b.list_files("threads"))
    assert files == []


def test_denied_by_policy_case_insensitive(tmp_path: Path) -> None:
    b = StorageBackend(tmp_path, deny_globs=("*.DB",))
    assert b.denied_by_policy("evil.db") is True
    assert b.denied_by_policy("evil.DB") is True


# ── Checkpoints ─────────────────────────────────────────────────────────────


def test_save_restore_checkpoint(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    bundle = {"label": "cp1", "stage_id": "s1", "turn_count": 3, "mode_id": "default"}
    saved = json.loads(b.save_checkpoint(bundle))
    assert "checkpoint_id" in saved
    cid = saved["checkpoint_id"]

    restored = json.loads(b.restore_checkpoint(cid))
    assert restored["label"] == "cp1"
    assert restored["turn_count"] == 3


def test_restore_checkpoint_missing(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.restore_checkpoint("nope"))
    assert "error" in data


def test_list_checkpoints(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.save_checkpoint({"label": "a", "created_at": "2026-01-01T00:00:00Z"})
    b.save_checkpoint({"label": "b", "created_at": "2026-02-01T00:00:00Z"})
    summaries = json.loads(b.list_checkpoints(""))
    labels = [s["label"] for s in summaries]
    assert "a" in labels
    assert "b" in labels
    # Sorted by created_at descending
    assert summaries[0]["created_at"] == "2026-02-01T00:00:00Z"


def test_save_checkpoint_readonly(tmp_path: Path) -> None:
    b = _readonly_backend(tmp_path)
    data = json.loads(b.save_checkpoint({"label": "x"}))
    assert data["error"] == "server is read-only"


# ── Model manifests ──────────────────────────────────────────────────────────


def test_write_read_manifest(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    w = json.loads(b.write_model_manifest("version: 1\n", "proposal"))
    assert w["ok"] is True
    assert w["which"] == "proposal"
    assert b.read_model_manifest("proposal") == "version: 1\n"


def test_read_manifest_missing(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.read_model_manifest("active"))
    assert "error" in data


def test_write_manifest_readonly(tmp_path: Path) -> None:
    b = _readonly_backend(tmp_path)
    data = json.loads(b.write_model_manifest("x", "active"))
    assert data["error"] == "server is read-only"


def test_write_manifest_rejects_unknown_kind(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    data = json.loads(b.write_model_manifest("x", "bogus"))
    assert "error" in data
