"""Tests for the StorageBackend — sandbox resolution, deny-glob policy,
legacy checkpoints-dir teardown, and model-manifest read/write.

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
    (tmp_path / "data" / "files").mkdir(parents=True, exist_ok=True)
    return StorageBackend(tmp_path, readonly=True)


# ── Construction ──────────────────────────────────────────────────────────────


def test_init_creates_subdirs(tmp_path: Path) -> None:
    StorageBackend(tmp_path)
    assert (tmp_path / "data" / "files").is_dir()


def test_init_readonly_does_not_create_subdirs(tmp_path: Path) -> None:
    StorageBackend(tmp_path, readonly=True)
    assert not (tmp_path / "data" / "files").exists()


def test_init_deletes_legacy_checkpoints_dir(tmp_path: Path) -> None:
    """StorageBackend.__init__ removes a legacy ``checkpoints/`` directory.

    The checkpoint system was removed in Phase 5 of the checkpoint
    teardown; the backend best-effort ``rmtree``s any leftover
    ``checkpoints/`` dir so stale data does not accumulate. A dummy
    file inside confirms the whole tree is gone after construction.
    """
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "stale.json").write_text("{}", encoding="utf-8")
    assert checkpoints_dir.exists()

    StorageBackend(tmp_path)

    assert not checkpoints_dir.exists()


def test_roots_and_manifest_path(tmp_path: Path) -> None:
    b = StorageBackend(tmp_path)
    assert b.files_root() == tmp_path / "data" / "files"
    assert b.manifest_path("active") == tmp_path / "config" / "models.yaml"
    assert b.manifest_path("proposal") == tmp_path / "state" / "models.proposed.yaml"
    assert b.manifest_path("  Proposal  ") == tmp_path / "state" / "models.proposed.yaml"


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
    # Text content is stored newline-terminated (the POSIX text-file rule).
    assert w["trailing_newline_added"] is True
    assert b.read_file("reports/daily.md") == "# Hello\n"


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
    # The stored file is newline-terminated, so it is one byte longer than
    # the text that was written.
    assert data["size"] == 6
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
    assert data["total"] == 11
    assert data["eof"] is False


def test_read_file_range_eof(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "0123456789")
    data = json.loads(b.read_file_range("range.txt", start=8, length=10))
    assert data["content"] == "89\n"
    assert data["length"] == 3
    assert data["eof"] is True


def test_read_file_range_start_past_end(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "abc")
    data = json.loads(b.read_file_range("range.txt", start=50, length=10))
    assert data["content"] == ""
    assert data["length"] == 0
    assert data["total"] == 4
    assert data["eof"] is True


def test_read_file_range_char_safe_multibyte(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    # "á" is one codepoint but 2 UTF-8 bytes; char slicing must not split it.
    text = "xáy"
    b.write_file("multi.txt", text)
    data = json.loads(b.read_file_range("multi.txt", start=1, length=1))
    assert data["content"] == "á"
    assert data["total"] == 4


def test_read_file_range_defaults_from_zero(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("range.txt", "hello")
    data = json.loads(b.read_file_range("range.txt"))
    assert data["start"] == 0
    assert data["content"] == "hello\n"
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
    assert first["bytes"] == len(b"line1\n")
    second = json.loads(b.append_file("log.txt", "line2\n"))
    assert second["ok"] is True
    assert second["bytes"] == len(b"line1\nline2\n")
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


# ── Trailing-newline write contract ──────────────────────────────────────────
#
# The 2026-10-19 incident: a session plan was written without its final
# newline, and ``cat`` glued the shell prompt onto the last line of the
# file. The loss was NOT in storage — it was in the model's emitted tool
# arguments — but storage is the layer that owns file semantics, so it
# enforces the POSIX text-file rule and says so in the receipt.


def test_write_file_adds_missing_trailing_newline(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("plan.md", "ends the turn."))
    assert receipt["ok"] is True
    assert receipt["trailing_newline_added"] is True
    assert receipt["bytes"] == len(b"ends the turn.\n")
    assert b.read_file("plan.md") == "ends the turn.\n"


def test_write_file_keeps_existing_trailing_newline(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("plan.md", "ends the turn.\n"))
    assert receipt["trailing_newline_added"] is False
    assert receipt["bytes"] == len(b"ends the turn.\n")
    assert b.read_file("plan.md") == "ends the turn.\n"


def test_write_file_keeps_exactly_one_trailing_newline(tmp_path: Path) -> None:
    """A doubled trailing newline is content, not a missing newline."""
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("plan.md", "C1\nC2\n\n"))
    assert receipt["trailing_newline_added"] is False
    assert b.read_file("plan.md") == "C1\nC2\n\n"


def test_write_file_keeps_leading_newline(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("plan.md", "\nB1\nB2"))
    assert receipt["trailing_newline_added"] is True
    assert b.read_file("plan.md") == "\nB1\nB2\n"


def test_write_file_xy_control_keeps_its_bytes(tmp_path: Path) -> None:
    """The ``XY`` control from the 2026-10-19 probe.

    ``XY`` contains no newline, so it cannot discriminate a strip from an
    omission — the flaw that made the original diagnosis wrong. It is
    kept as the control: the payload is stored verbatim apart from the
    text-file newline rule, and the receipt states that it was added.
    """
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("probe.txt", "XY"))
    assert receipt["ok"] is True
    assert receipt["trailing_newline_added"] is True
    assert receipt["bytes"] == 3
    assert b.read_file("probe.txt") == "XY\n"


def test_write_file_empty_content_is_untouched(tmp_path: Path) -> None:
    """An empty write is empty — there is no newline to add."""
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("empty.txt", ""))
    assert receipt["ok"] is True
    assert receipt["trailing_newline_added"] is False
    assert receipt["bytes"] == 0
    assert b.read_file("empty.txt") == ""


def test_write_file_leaves_binary_suffix_content_exact(tmp_path: Path) -> None:
    """A binary format is stored byte-for-byte.

    Normalizing the trailing byte of a PNG or a database would corrupt
    it: those formats put significant data in their final bytes.
    """
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("image.png", "PNG-not-really"))
    assert receipt["trailing_newline_added"] is False
    assert receipt["bytes"] == len(b"PNG-not-really")
    assert (b.files_root() / "image.png").read_bytes() == b"PNG-not-really"


def test_write_file_rejects_non_utf8_content(tmp_path: Path) -> None:
    """Content that is not valid UTF-8 is not text and is not storable.

    A lone surrogate (which a cut multi-byte character can produce) cannot
    be encoded as UTF-8, so it can never be written as a text file. The
    write reports an error instead of raising out of the tool.
    """
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("broken.txt", "bad\ud800end"))
    assert "error" in receipt
    assert receipt["error"] == "content is not valid UTF-8 text"


def test_write_file_receipt_reports_added_newline_only_once(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    first = json.loads(b.write_file("plan.md", "one"))
    assert first["trailing_newline_added"] is True
    second = json.loads(b.write_file("plan.md", "one\n"))
    assert second["trailing_newline_added"] is False


def test_append_file_reports_appended_bytes(tmp_path: Path) -> None:
    """The receipt proves an effect landed.

    ``bytes`` is the resulting file size, so it cannot tell an append
    that moved bytes from one that moved none. ``appended_bytes`` is the
    field that can.
    """
    b = _backend(tmp_path)
    first = json.loads(b.append_file("log.txt", "line1\n"))
    assert first["ok"] is True
    assert first["appended_bytes"] == len(b"line1\n")
    assert first["bytes"] == len(b"line1\n")
    second = json.loads(b.append_file("log.txt", "line2\n"))
    assert second["appended_bytes"] == len(b"line2\n")
    assert second["bytes"] == len(b"line1\nline2\n")


def test_append_file_empty_append_reports_no_bytes_appended(tmp_path: Path) -> None:
    """An empty append is a genuine no-op, and the receipt says so."""
    b = _backend(tmp_path)
    b.append_file("log.txt", "K1\n")
    receipt = json.loads(b.append_file("log.txt", ""))
    assert receipt["ok"] is True
    assert receipt["appended_bytes"] == 0
    assert receipt["bytes"] == 3
    assert b.read_file("log.txt") == "K1\n"


def test_append_file_rejects_non_utf8_content(tmp_path: Path) -> None:
    """Content that cannot be encoded as UTF-8 is reported, not raised."""
    b = _backend(tmp_path)
    receipt = json.loads(b.append_file("broken.txt", "bad\ud800end"))
    assert receipt["error"] == "content is not valid UTF-8 text"


def test_write_file_stores_exact_bytes_on_every_platform(tmp_path: Path) -> None:
    """The stored bytes are the bytes that were sent, on every platform.

    Text mode rewrites ``\\n`` as ``\\r\\n`` on Windows, so the same
    content stored 6 bytes on POSIX and 7 on Windows — and the receipt's
    ``bytes`` reported a size the file did not have. The write is
    byte-faithful, so the receipt and the file agree everywhere.
    """
    b = _backend(tmp_path)
    receipt = json.loads(b.write_file("exact.txt", "12345\n"))
    stored = (b.files_root() / "exact.txt").read_bytes()
    assert stored == b"12345\n"
    assert receipt["bytes"] == len(stored)
    assert receipt["bytes"] == 6


def test_write_file_does_not_translate_inner_newlines(tmp_path: Path) -> None:
    """A newline inside the content stays a newline, not ``\\r\\n``."""
    b = _backend(tmp_path)
    b.write_file("inner.txt", "a\nb\nc\n")
    stored = (b.files_root() / "inner.txt").read_bytes()
    assert stored == b"a\nb\nc\n"
    assert b"\r\n" not in stored


def test_write_model_manifest_stores_exact_bytes(tmp_path: Path) -> None:
    """The manifest is stored byte-faithfully too."""
    b = _backend(tmp_path)
    receipt = json.loads(b.write_model_manifest("version: 1\n", "proposal"))
    stored = b.manifest_path("proposal").read_bytes()
    assert stored == b"version: 1\n"
    assert receipt["bytes"] == len(stored) == 11


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
