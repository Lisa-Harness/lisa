"""Tests for the StorageBackend — sandbox resolution, deny-glob policy,
legacy checkpoints-dir teardown, and model-manifest read/write.

Covers the same behaviors the lisa-storage MCP server exposes, but
exercised directly against the SDK backend without MCPServer. These are
the in-process contract hosts and tools depend on when they use storage
without spawning the MCP server.
"""

from __future__ import annotations

import hashlib
import json
import stat
import sys
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


def test_delete_file_unlinks_symlink_not_target(tmp_path: Path) -> None:
    """R7: delete_file never follows a leaf symlink.

    The pre-normalization behavior resolved the link and deleted its
    target; the nofollow normalization unlinks the link itself, so the
    target survives (POSIX ``rm`` semantics).
    """
    b = _backend(tmp_path)
    b.write_file("target.txt", "precious")
    (b.files_root() / "link.txt").symlink_to(b.files_root() / "target.txt")

    data = json.loads(b.delete_file("link.txt"))

    assert data["ok"] is True
    assert not (b.files_root() / "link.txt").is_symlink()
    assert b.read_file("target.txt") == "precious\n"


def test_delete_file_unlinks_symlink_to_directory(tmp_path: Path) -> None:
    """A link to a directory is still a leaf entry: the link goes, not the dir."""
    b = _backend(tmp_path)
    (b.files_root() / "real").mkdir()
    (b.files_root() / "real" / "keep.txt").write_text("x", encoding="utf-8")
    (b.files_root() / "dirlink").symlink_to(b.files_root() / "real", target_is_directory=True)

    data = json.loads(b.delete_file("dirlink"))

    assert data["ok"] is True
    assert not (b.files_root() / "dirlink").is_symlink()
    assert (b.files_root() / "real" / "keep.txt").is_file()


def test_delete_file_broken_symlink_is_removed(tmp_path: Path) -> None:
    """A broken link is an entry too: it is unlinked, not reported missing."""
    b = _backend(tmp_path)
    (b.files_root() / "dangling.txt").symlink_to(b.files_root() / "never-existed.txt")

    data = json.loads(b.delete_file("dangling.txt"))

    assert data["ok"] is True
    assert not (b.files_root() / "dangling.txt").is_symlink()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_renames_and_creates_parents(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("draft.md", "draft")
    before_inode = (b.files_root() / "draft.md").stat().st_ino

    receipt = json.loads(b.move_file("draft.md", "archive/final.md", create_parents=True))

    assert receipt["ok"] is True
    assert receipt["source"] == "draft.md"
    assert receipt["destination"] == "archive/final.md"
    assert receipt["bytes"] == len(b"draft\n")
    assert receipt["sha256"] == hashlib.sha256(b"draft\n").hexdigest()
    assert receipt["durable"] is True
    assert not (b.files_root() / "draft.md").exists()
    assert (b.files_root() / "archive" / "final.md").stat().st_ino == before_inode


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_refuses_existing_destination_without_overwrite(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("source.txt", "new")
    b.write_file("destination.txt", "old")

    receipt = json.loads(b.move_file("source.txt", "destination.txt"))

    assert "already exists" in receipt["error"]
    assert b.read_file("source.txt") == "new\n"
    assert b.read_file("destination.txt") == "old\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_does_not_create_parents_by_default(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("source.txt", "content")

    receipt = json.loads(b.move_file("source.txt", "missing/destination.txt"))

    assert "parent directory does not exist" in receipt["error"]
    assert not (b.files_root() / "missing").exists()
    assert b.read_file("source.txt") == "content\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_hash_guard_leaves_source_in_place(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("source.txt", "content")

    receipt = json.loads(
        b.move_file(
            "source.txt",
            "destination.txt",
            expected_sha256="0" * 64,
        )
    )

    assert "sha256 mismatch" in receipt["error"]
    assert receipt["sha256"] == hashlib.sha256(b"content\n").hexdigest()
    assert b.read_file("source.txt") == "content\n"
    assert not (b.files_root() / "destination.txt").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_rejects_symlink_source(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")
    (b.files_root() / "link.txt").symlink_to(external)

    receipt = json.loads(b.move_file("link.txt", "renamed.txt"))

    assert "not a regular file" in receipt["error"]
    assert external.read_text(encoding="utf-8") == "outside"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_rejects_symlinked_parent_and_stays_inside(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (b.files_root() / "linked").symlink_to(outside, target_is_directory=True)

    receipt = json.loads(b.move_file("linked/escaped.txt", "renamed.txt", create_parents=True))

    assert "not a real directory" in receipt["error"]
    assert not (outside / "renamed.txt").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_rolls_back_detected_source_identity_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sdk.storage._move as secure_move

    b = _backend(tmp_path)
    b.write_file("source.txt", "original")
    root = b.files_root()
    original_rename = secure_move._rename_noreplace
    raced = False

    def replace_source_before_rename(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal raced
        if not raced:
            raced = True
            (root / source_name).rename(root / "original-aside.txt")
            (root / source_name).write_bytes(b"replacement")
        original_rename(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(secure_move, "_rename_noreplace", replace_source_before_rename)

    receipt = json.loads(b.move_file("source.txt", "destination.txt"))

    assert "rolled back" in receipt["error"]
    assert (root / "source.txt").read_bytes() == b"replacement"
    assert (root / "original-aside.txt").read_bytes() == b"original\n"
    assert not (root / "destination.txt").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_rejects_directories_and_hard_links(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("source.txt", "content")
    (b.files_root() / "directory").mkdir()
    (b.files_root() / "source-link.txt").hardlink_to(b.files_root() / "source.txt")

    directory = json.loads(b.move_file("directory", "renamed-directory"))
    linked = json.loads(b.move_file("source.txt", "renamed.txt"))

    assert "not a regular file" in directory["error"]
    assert "hard links" in linked["error"]
    assert (b.files_root() / "directory").is_dir()
    assert b.read_file("source.txt") == "content\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_file_rejects_denied_source_and_destination(tmp_path: Path) -> None:
    b = _backend(tmp_path, deny_globs=("*.db", "private/**"))
    b.write_file("source.txt", "content")
    (b.files_root() / "secret.db").write_bytes(b"secret")

    denied_source = json.loads(b.move_file("secret.db", "safe.txt"))
    denied_destination = json.loads(
        b.move_file("source.txt", "private/renamed.txt", create_parents=True)
    )

    assert denied_source["error"] == "source path denied by policy"
    assert denied_destination["error"] == "destination path denied by policy"
    assert b.read_file("source.txt") == "content\n"


def test_move_file_readonly_and_unsupported_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sdk.storage._move as secure_move

    readonly = _readonly_backend(tmp_path)
    assert json.loads(readonly.move_file("a.txt", "b.txt"))["error"] == "server is read-only"

    b = _backend(tmp_path)
    b.write_file("a.txt", "content")
    monkeypatch.setattr(secure_move.sys, "platform", "win32")
    receipt = json.loads(b.move_file("a.txt", "b.txt"))
    assert "unsupported on this platform" in receipt["error"]
    assert b.read_file("a.txt") == "content\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_directory_renames_whole_subtree(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("project/src/a.py", "a")
    b.write_file("project/notes.md", "notes")
    before_inode = (b.files_root() / "project").stat().st_ino

    receipt = json.loads(b.move_directory("project", "archive/project", create_parents=True))

    assert receipt["ok"] is True
    assert receipt["source"] == "project"
    assert receipt["destination"] == "archive/project"
    assert receipt["durable"] is True
    assert "sha256" not in receipt
    assert not (b.files_root() / "project").exists()
    moved = b.files_root() / "archive" / "project"
    assert moved.stat().st_ino == before_inode
    assert b.read_file("archive/project/src/a.py") == "a\n"
    assert b.read_file("archive/project/notes.md") == "notes\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_directory_refuses_existing_destination(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("source/x.txt", "x")
    (b.files_root() / "destination").mkdir()

    receipt = json.loads(b.move_directory("source", "destination"))

    assert "already exists" in receipt["error"]
    assert (b.files_root() / "source" / "x.txt").is_file()
    assert (b.files_root() / "destination").is_dir()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_directory_refuses_destination_inside_source(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("tree/inner/leaf.txt", "leaf")

    receipt = json.loads(b.move_directory("tree", "tree/inner"))

    assert "inside the source directory" in receipt["error"]
    assert b.read_file("tree/inner/leaf.txt") == "leaf\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_directory_refuses_symlink_source(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (b.files_root() / "linked").symlink_to(outside, target_is_directory=True)

    receipt = json.loads(b.move_directory("linked", "renamed"))

    assert "not a directory" in receipt["error"]
    assert outside.is_dir()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_directory_refuses_file_source(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("plain.txt", "content")

    receipt = json.loads(b.move_directory("plain.txt", "renamed"))

    assert "not a directory" in receipt["error"]
    assert b.read_file("plain.txt") == "content\n"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_directory_creates_parents_only_when_asked(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("source/x.txt", "x")

    refused = json.loads(b.move_directory("source", "missing/destination"))
    assert "parent directory does not exist" in refused["error"]
    assert not (b.files_root() / "missing").exists()

    created = json.loads(b.move_directory("source", "missing/destination", create_parents=True))
    assert created["ok"] is True
    assert b.read_file("missing/destination/x.txt") == "x\n"


def test_move_directory_readonly_fails_closed(tmp_path: Path) -> None:
    readonly = _readonly_backend(tmp_path)
    assert json.loads(readonly.move_directory("a", "b"))["error"] == "server is read-only"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="secure moves require Linux")
def test_move_directory_denied_destination(tmp_path: Path) -> None:
    b = _backend(tmp_path, deny_globs=("private/**",))
    b.write_file("source/x.txt", "x")

    receipt = json.loads(b.move_directory("source", "private/renamed"))

    assert receipt["error"] == "destination path denied by policy"
    assert b.read_file("source/x.txt") == "x\n"


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


# ── Exact-edit contract ──────────────────────────────────────────────────────
#
# The failure ``edit_file`` exists to remove: changing one line of a
# document by rewriting the whole file means reproducing every other byte
# from memory. For a document whose value IS its evidence, that
# reconstruction is the risk. The edit is exact, ambiguous matches are
# refused, and the rest of the file is untouched.


def _seeded(tmp_path: Path) -> StorageBackend:
    """A backend holding a small tracker whose state cell must change."""
    b = _backend(tmp_path)
    b.write_file(
        "tracker.md",
        "# Tracker\n\n| Item | State |\n| --- | --- |\n| A1 | Open |\n| A2 | Closed |\n",
    )
    return b


def test_edit_file_replaces_one_region_and_preserves_the_rest(tmp_path: Path) -> None:
    b = _seeded(tmp_path)
    before = b.read_file("tracker.md")

    receipt = json.loads(b.edit_file("tracker.md", "| A1 | Open |", "| A1 | Closed |"))

    assert receipt["ok"] is True
    assert receipt["replacements"] == 1
    assert receipt["first_line"] == 5
    # The one-line change is exactly that: no other byte moved.
    after = before.replace("| A1 | Open |", "| A1 | Closed |")
    assert b.read_file("tracker.md") == after
    assert receipt["bytes_before"] == len(before.encode())
    assert receipt["bytes_after"] == len(after.encode())
    assert receipt["bytes_after"] == (b.files_root() / "tracker.md").stat().st_size


def test_edit_file_is_byte_exact_about_fragment_edges(tmp_path: Path) -> None:
    """The fragment's own newline is part of the match.

    ``edit_file`` applies no newline normalization: an empty
    ``new_string`` removes exactly the matched bytes and nothing more.
    """
    b = _backend(tmp_path)
    b.write_file("doc.md", "keep\nremove me\nkeep\n")

    receipt = json.loads(b.edit_file("doc.md", "remove me\n", ""))

    assert receipt["ok"] is True
    assert b.read_file("doc.md") == "keep\nkeep\n"


def test_edit_file_refuses_an_ambiguous_match(tmp_path: Path) -> None:
    """A fragment matching twice has no unique position, so nothing is written.

    The error names the candidate lines; the caller adds context instead
    of forcing the tool to guess which one was meant.
    """
    b = _backend(tmp_path)
    b.write_file("doc.md", "same\nmiddle\nsame\n")
    before = b.read_file("doc.md")

    receipt = json.loads(b.edit_file("doc.md", "same", "other"))

    assert "occurs 2 times" in receipt["error"]
    assert receipt["lines"] == [1, 3]
    assert b.read_file("doc.md") == before


def test_edit_file_replace_all_writes_every_occurrence(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("doc.md", "same\nmiddle\nsame\n")

    receipt = json.loads(b.edit_file("doc.md", "same", "other", replace_all=True))

    assert receipt["replacements"] == 2
    assert b.read_file("doc.md") == "other\nmiddle\nother\n"


def test_edit_file_missing_fragment_names_the_line_ending_cause(tmp_path: Path) -> None:
    """A CRLF/LF mismatch is reported as such, not as a bare "not found"."""
    b = _backend(tmp_path)
    # Written byte-faithfully, so the stored file really has CRLF.
    b.write_file("crlf.md", "one\r\ntwo\r\n")

    receipt = json.loads(b.edit_file("crlf.md", "one\ntwo\n", "changed\n"))

    assert "not found" in receipt["error"]
    assert "CRLF" in receipt["error"]


def test_edit_file_refuses_when_the_file_changed_since_read(tmp_path: Path) -> None:
    """``expected_sha256`` binds the edit to the revision the caller saw."""
    b = _seeded(tmp_path)
    stale = json.loads(b.stat_file("tracker.md"))["sha256"]
    b.edit_file("tracker.md", "| A2 | Closed |", "| A2 | Open |")

    receipt = json.loads(
        b.edit_file("tracker.md", "| A1 | Open |", "| A1 | Closed |", expected_sha256=stale)
    )

    assert "sha256 mismatch" in receipt["error"]
    assert receipt["sha256"] == json.loads(b.stat_file("tracker.md"))["sha256"]
    assert "| A2 | Open |" in b.read_file("tracker.md")


def test_edit_file_accepts_the_hash_stat_file_reported(tmp_path: Path) -> None:
    b = _seeded(tmp_path)
    digest = json.loads(b.stat_file("tracker.md"))["sha256"]

    receipt = json.loads(
        b.edit_file("tracker.md", "| A1 | Open |", "| A1 | Closed |", expected_sha256=digest)
    )

    assert receipt["ok"] is True
    assert receipt["sha256"] == json.loads(b.stat_file("tracker.md"))["sha256"]
    assert receipt["sha256"] != digest


def test_edit_file_absent_returns_error(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    receipt = json.loads(b.edit_file("missing.md", "a", "b"))
    assert "file not found" in receipt["error"]


def test_edit_file_readonly_rejected(tmp_path: Path) -> None:
    b = _readonly_backend(tmp_path)
    receipt = json.loads(b.edit_file("doc.md", "a", "b"))
    assert receipt["error"] == "server is read-only"


def test_edit_file_traversal_rejected(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    receipt = json.loads(b.edit_file("../escape.md", "a", "b"))
    assert "path rejected" in receipt["error"]


def test_edit_file_denied_glob_rejected(tmp_path: Path) -> None:
    b = _backend(tmp_path, deny_globs=("*.db",))
    (b.files_root() / "secret.db").write_text("a", encoding="utf-8")
    receipt = json.loads(b.edit_file("secret.db", "a", "b"))
    assert receipt["error"] == "path denied by policy"


def test_edit_file_rejects_empty_fragment(tmp_path: Path) -> None:
    """An empty match has a position between every pair of characters."""
    b = _seeded(tmp_path)
    receipt = json.loads(b.edit_file("tracker.md", "", "x"))
    assert "must not be empty" in receipt["error"]


def test_edit_file_rejects_a_no_op_edit(tmp_path: Path) -> None:
    b = _seeded(tmp_path)
    receipt = json.loads(b.edit_file("tracker.md", "Open", "Open"))
    assert "would change nothing" in receipt["error"]


def test_edit_file_reads_and_writes_the_requested_encoding(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("latin.md", "café\n")

    receipt = json.loads(b.edit_file("latin.md", "café", "cafe", encoding="utf-8"))

    assert receipt["ok"] is True
    assert b.read_file("latin.md") == "cafe\n"


def test_edit_file_leaves_mode_readable(tmp_path: Path) -> None:
    """The atomic swap must not narrow access to a previously readable file."""
    b = _backend(tmp_path)
    b.write_file("doc.md", "one\n")
    target = b.files_root() / "doc.md"
    target.chmod(0o644)
    # Windows reports 0o666 for any writable file, so compare to the observed mode.
    expected_mode = stat.S_IMODE(target.stat().st_mode)

    receipt = json.loads(b.edit_file("doc.md", "one", "two"))

    assert receipt["ok"] is True
    assert stat.S_IMODE(target.stat().st_mode) == expected_mode
    assert b.read_file("doc.md") == "two\n"


def test_edit_file_leaves_no_temporary_file(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("doc.md", "one\n")

    b.edit_file("doc.md", "one", "two")
    b.edit_file("doc.md", "absent", "three")

    assert sorted(p.name for p in b.files_root().iterdir()) == ["doc.md"]


def test_stat_file_reports_sha256_for_files_only(tmp_path: Path) -> None:
    b = _backend(tmp_path)
    b.write_file("hashed.txt", "abc")
    (b.files_root() / "adir").mkdir()

    file_stat = json.loads(b.stat_file("hashed.txt"))
    dir_stat = json.loads(b.stat_file("adir"))

    assert file_stat["sha256"] == hashlib.sha256(b"abc\n").hexdigest()
    assert dir_stat["sha256"] == ""


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


def test_trash_hidden_from_list_files(tmp_path: Path) -> None:
    """The default deny-glob hides the trash subtree from routine listing."""
    b = _backend(tmp_path)
    b.write_file("keep.txt", "ok")
    trash_entry = b.files_root() / ".trash" / "2026-11-18T00-00-00.000000"
    trash_entry.mkdir(parents=True)
    (trash_entry / "buried.txt").write_text("x", encoding="utf-8")

    files = json.loads(b.list_files(""))

    assert files == ["keep.txt"]


def test_trash_denied_for_routine_ops(tmp_path: Path) -> None:
    """Read/stat/write/delete refuse trash paths even with no caller deny-globs."""
    b = _backend(tmp_path)
    assert json.loads(b.read_file(".trash/buried.txt"))["error"] == "path denied by policy"
    assert json.loads(b.stat_file(".trash/buried.txt"))["error"] == "path denied by policy"
    assert json.loads(b.write_file(".trash/new.txt", "x"))["error"] == "path denied by policy"
    assert json.loads(b.delete_file(".trash/buried.txt"))["error"] == "path denied by policy"


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
