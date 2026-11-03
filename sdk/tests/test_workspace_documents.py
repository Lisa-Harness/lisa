"""Tests for bounded workspace documents and private authority transactions."""

import subprocess
from pathlib import Path

import pytest
from sdk.workspace_documents import DocumentConflict, WorkspaceDocuments


def store(tmp_path: Path) -> WorkspaceDocuments:
    """Create an existing project and separate private record root."""
    root = tmp_path / "project"
    root.mkdir()
    return WorkspaceDocuments(root, tmp_path / "authority")


def test_publication_and_stale_write(tmp_path: Path) -> None:
    """Compare-and-swap prevents lost updates and supports nested locking."""
    documents = store(tmp_path)
    with documents.transaction():
        digest = documents.write("hdae/PROJECT.md", "intent\n", expected_digest="")
    assert documents.read("hdae/PROJECT.md") == "intent\n"
    with pytest.raises(DocumentConflict):
        documents.write("hdae/PROJECT.md", "stale", expected_digest="")
    documents.write("hdae/PROJECT.md", "revised\n", expected_digest=digest)
    assert documents.list("hdae") == ("hdae/PROJECT.md",)


def test_private_records_and_move(tmp_path: Path) -> None:
    """Private state remains outside project files; moves cannot overwrite."""
    documents = store(tmp_path)
    documents.write("authorization.json", "approved", expected_digest="", private=True)
    assert not (documents.root / "authorization.json").exists()
    assert documents.read("authorization.json", private=True) == "approved"
    digest = documents.write("todo/1-gate.md", "gate", expected_digest="")
    documents.move("todo/1-gate.md", "done/1-gate.md", expected_digest=digest)
    assert not documents.exists("todo/1-gate.md")
    assert documents.exists("done/1-gate.md")
    with pytest.raises(DocumentConflict):
        documents.move("done/1-gate.md", "done/1-gate.md", expected_digest=digest)


@pytest.mark.parametrize("relative", ["../escape", "/escape", ".git/config", "a//b", "a\\b"])
def test_unsafe_paths(tmp_path: Path, relative: str) -> None:
    """Workspace methods reject traversal and Git metadata access."""
    with pytest.raises(ValueError):
        store(tmp_path).read(relative)


def test_symlink_and_bad_roots(tmp_path: Path) -> None:
    """Symlink traversal and project-contained authority are rejected."""
    documents = store(tmp_path)
    (documents.root / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        documents.resolve("linked/file")
    with pytest.raises(ValueError):
        WorkspaceDocuments(documents.root, documents.root / "private")
    with pytest.raises(FileNotFoundError):
        documents.read("missing.md")


def test_binary_fingerprints_and_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Binary build artifacts use a distinct bounded streaming fingerprint."""
    import hashlib

    from sdk import workspace_documents

    documents = store(tmp_path)
    (documents.root / "image.efi").write_bytes(b"\x00\xff\x01\x02")
    assert documents.digest("image.efi") == hashlib.sha256(b"\x00\xff\x01\x02").hexdigest()
    monkeypatch.setattr(workspace_documents, "_MAX_FINGERPRINT_BYTES", 3)
    with pytest.raises(ValueError, match="fingerprint byte limit"):
        documents.digest("image.efi")
    with pytest.raises(ValueError, match="Git repository"):
        documents.git_state()


def test_document_size_limits_and_empty_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversized text cannot enter or leave the bounded publication API."""
    from sdk import workspace_documents

    documents = store(tmp_path)
    assert documents.list("absent") == ()
    (documents.root / "large.md").write_text("oversized")
    monkeypatch.setattr(workspace_documents, "_MAX_DOCUMENT_BYTES", 2)
    with pytest.raises(ValueError, match="byte limit"):
        documents.read("large.md")
    with pytest.raises(ValueError, match="byte limit"):
        documents.write("new.md", "large", expected_digest="")


def test_git_state_observes_actual_repository(tmp_path: Path) -> None:
    """A clean commit and subsequent dirty state are observed from Git itself."""
    documents = store(tmp_path)
    commands = (
        ("init",),
        ("config", "user.name", "fixture"),
        ("config", "user.email", "fixture@example.invalid"),
        ("config", "commit.gpgsign", "false"),
        ("commit", "--allow-empty", "-m", "baseline"),
    )
    for args in commands:
        subprocess.run(
            ("git", "-C", str(documents.root), *args),
            check=True,
            capture_output=True,
            timeout=10,
        )
    state = documents.git_state()
    assert state.head and state.tree and not state.dirty
    (documents.root / "changed.txt").write_text("changed")
    assert documents.git_state().dirty


@pytest.mark.asyncio
async def test_async_transaction_reuses_existing_file_lock(tmp_path: Path) -> None:
    """Async composition safely nests compare-and-swap writes in one task."""
    documents = store(tmp_path)
    async with documents.async_transaction():
        digest = documents.write("artifact.md", "value", expected_digest="")
        assert documents.digest("artifact.md") == digest
