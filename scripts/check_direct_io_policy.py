"""Enforce runtime direct-write policy with an explicit allowlist.

Scans runtime Python code (kernel/hosts/sdk/mcp_server, excluding tests) for
filesystem write/mutation primitives that bypass higher-level storage routing.
Any detected write callsite must be explicitly allowlisted by file + snippet.

This is intentionally narrow: it checks the highest-risk direct-write APIs and
acts as a regression guard against new bypasses.
"""

from __future__ import annotations

import re
from pathlib import Path

# Direct-write primitives to guard.
_WRITE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.write_text\("),
    re.compile(r"\.write_bytes\("),
    re.compile(r"\.open\(\s*['\"](?:a|w|x|ab|wb|xb)['\"]"),
    re.compile(r"\bos\.replace\("),
    re.compile(r"\bsqlite3\.connect\("),
)

# Allowed direct-write callsites, keyed by repo-relative path.
# Each value is a tuple of line substrings that are permitted in that file.
_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "mcp_server/storage/server.py": (
        "resolved.write_text(",
        "path.write_text(",
    ),
    "hosts/core/content_roots.py": (
        "path.write_text(",
    ),
    "hosts/core/turn_hooks_builder.py": (
        "thread_path.write_text(",
    ),
    "hosts/core/code_artifacts.py": (
        "temporary_path.write_text(",
        "os.replace(temporary_path, self._path)",
    ),
    "hosts/core/approvals.py": (
        "temp_path.write_text(",
        "os.replace(temp_path, self._path)",
    ),
    "hosts/core/intake.py": (
        "state_path.write_text(",
    ),
    "hosts/core/state_provider.py": (
        "digest_path.write_text(",
        "self._thread_path.write_text(",
    ),
    "hosts/core/model_discovery/manifest.py": (
        "tmp.write_text(",
    ),
    "hosts/core/session_context_fetch.py": (
        "cache_meta.write_text(",
        "cache_md.write_bytes(",
    ),
    "hosts/core/thread.py": (
        "thread_path.open(\"a\"",
    ),
    "kernel/token_usage.py": (
        "self._path.open(\"a\"",
    ),
    "sdk/memory/store.py": (
        "sqlite3.connect(",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_runtime_python(path: Path) -> bool:
    rel = path.as_posix()
    if not rel.endswith(".py"):
        return False
    if "/tests/" in rel:
        return False
    if path.name.startswith("test_"):
        return False
    return True


def _iter_runtime_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for top in ("kernel", "hosts", "sdk", "mcp_server"):
        top_dir = root / top
        if not top_dir.is_dir():
            continue
        for path in top_dir.rglob("*.py"):
            rel = path.relative_to(root)
            if _is_runtime_python(rel):
                files.append(path)
    return sorted(files)


def _line_has_write_primitive(line: str) -> bool:
    return any(pattern.search(line) for pattern in _WRITE_PATTERNS)


def _is_allowlisted(rel_path: str, line: str) -> bool:
    allowed_snippets = _ALLOWLIST.get(rel_path)
    if not allowed_snippets:
        return False
    return any(snippet in line for snippet in allowed_snippets)


def main() -> int:
    root = _repo_root()
    violations: list[tuple[str, int, str]] = []

    for path in _iter_runtime_python_files(root):
        rel_path = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            violations.append((rel_path, 0, f"read error: {exc}"))
            continue

        for idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not _line_has_write_primitive(line):
                continue
            if _is_allowlisted(rel_path, line):
                continue
            violations.append((rel_path, idx, line))

    if violations:
        print("Direct I/O policy violations detected:")
        for rel_path, line_no, line in violations:
            print(f"- {rel_path}:{line_no}: {line}")
        print()
        print("Every runtime direct-write callsite must be explicitly allowlisted in")
        print("scripts/check_direct_io_policy.py (_ALLOWLIST) or migrated behind")
        print("lisa-storage / a documented host-internal carve-out.")
        return 1

    print("Direct I/O policy check passed (no unallowlisted runtime write callsites).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
