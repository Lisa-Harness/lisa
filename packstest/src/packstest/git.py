"""Throwaway Git repository fixtures for worktree-pack tests.

Generalizes the ``_repository()`` helper proven in the coding pack's
temporary-repository tests: a real Git repo with one commit, plus a bare
remote so push/pull paths run fully offline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

__all__ = ["GitRepoFixture", "git_repo"]


class GitRepoFixture:
    """A disposable Git work tree plus bare remote, managed from pytest."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.remote: Path | None = None

    async def _run(self, *args: str, cwd: Path | None = None) -> None:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd or self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {stderr.decode('utf-8', errors='replace')}"
            )

    async def commit_file(self, rel_path: str, content: str, message: str) -> None:
        """Write a file, commit it, and advance HEAD."""

        target = self.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        await self._run("add", rel_path)
        await self._run("commit", "-m", message)
        await self._run("config", "user.email", "test@example.invalid")

    async def head(self) -> str:
        """Return the current HEAD sha."""

        process = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "HEAD",
            cwd=str(self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        return stdout.decode().strip()

    def write(self, rel_path: str, content: str) -> Path:
        """Write a file into the work tree without committing."""

        target = self.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target


async def git_repo(tmp_path: Path, *, with_remote: bool = True) -> GitRepoFixture:
    """Create a real Git repository (one commit) plus optional bare remote."""

    root = tmp_path / "source"
    root.mkdir()
    fixture = GitRepoFixture(root)
    await fixture._run("init")
    await fixture._run("config", "user.email", "test@example.invalid")
    await fixture._run("config", "user.name", "Lisa Test")
    fixture.write("app.txt", "hello\n")
    await fixture._run("add", "app.txt")
    await fixture._run("commit", "-m", "initial")
    if with_remote:
        remote = tmp_path / "remote.git"
        await fixture._run("init", "--bare", str(remote), cwd=tmp_path)
        await fixture._run("remote", "add", "origin", str(remote))
        fixture.remote = remote
    return fixture