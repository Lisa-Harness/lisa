"""Enforce direct file I/O policy with an explicit allowlist.

Scans runtime Python code for filesystem primitives that bypass higher-level
storage routing (``lisa-storage``). Any detected callsite must be explicitly
allowlisted by file + snippet, or migrated behind the storage seam.

Scope
-----
In-tree packages (``kernel/``, ``host/``, ``sdk/``, ``mcp_server/``): scans
write/mutation primitives only. Reads are a separate, larger follow-up because
the codebase has dozens of legitimate read callsites.

External packs (``../lisa-packs/``, ``../lisa-packs-wip/``): scans **both**
reads and writes -- packs must route all file I/O through ``lisa-storage`` via
``ctx.execute_tool_fn``. No allowlist carve-outs for packs by default.

The scanner uses ``ast`` for reliable detection (catches aliased names,
multiline calls, and attribute chains that regex misses).
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Banned primitives
# ---------------------------------------------------------------------------

# Attribute-method calls on any object (e.g. ``path.write_text(...)``).
# Matched by attribute name only -- the receiver type is resolved at runtime.
_BANNED_METHODS_WRITE: frozenset[str] = frozenset(
    {
        "write_text",
        "write_bytes",
    }
)

_BANNED_METHODS_READ: frozenset[str] = frozenset(
    {
        "read_text",
        "read_bytes",
    }
)

_BANNED_METHODS_MUTATE: frozenset[str] = frozenset(
    {
        "unlink",
        "mkdir",
        "rmdir",
    }
)

# Module-level function calls (e.g. ``os.replace(...)``, ``sqlite3.connect(...)``).
# Keyed by ``(module_name, function_name)``.
_BANNED_MODULE_FUNCS_WRITE: frozenset[tuple[str, str]] = frozenset(
    {
        ("os", "replace"),
        ("os", "remove"),
        ("os", "rmdir"),
        ("shutil", "copy"),
        ("shutil", "copy2"),
        ("shutil", "copyfile"),
        ("shutil", "rmtree"),
        ("sqlite3", "connect"),
    }
)

_BANNED_MODULE_FUNCS_READ: frozenset[tuple[str, str]] = frozenset(
    {
        ("shutil", "copy"),
        ("shutil", "copy2"),
        ("shutil", "copyfile"),
    }
)

# ``open()`` mode argument prefixes that indicate write/mutation.
# ``open("r")`` is read-only (only banned for external packs).
_WRITE_OPEN_MODES: frozenset[str] = frozenset({"w", "a", "x", "wb", "ab", "xb"})

# ---------------------------------------------------------------------------
# Allowlist -- in-tree carve-outs
# ---------------------------------------------------------------------------

# Allowed direct-write callsites, keyed by repo-relative path.
# Each value is a tuple of line substrings that are permitted in that file.
_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "mcp_server/src/mcp_server/storage/server.py": (
        "resolved.write_text(",
        "path.write_text(",
    ),
    "host/src/host/content_roots.py": (
        "path.write_text(",
        "path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "host/src/host/turn_hooks_builder.py": (
        "thread_path.write_text(",
        "thread_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "host/src/host/code_artifacts.py": (
        "temporary_path.write_text(",
        "os.replace(temporary_path, self._path)",
        "self._path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "host/src/host/approvals.py": (
        "temp_path.write_text(",
        "os.replace(temp_path, self._path)",
        "self._path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "host/src/host/intake.py": (
        "state_path.write_text(",
        "data_dir.mkdir(parents=True, exist_ok=True)",
    ),
    "host/src/host/recovery.py": (
        "self._path.parent.mkdir(parents=True, exist_ok=True)",
        "temp_path.write_text(",
        "os.replace(temp_path, self._path)",
    ),
    "host/src/host/state_provider.py": (
        "digest_path.write_text(",
        "self._thread_path.write_text(",
    ),
    "host/src/host/model_discovery/manifest.py": (
        "tmp.write_text(",
        "path.parent.mkdir(parents=True, exist_ok=True)",
        "tmp.unlink(missing_ok=True)",
    ),
    "host/src/host/session_context_fetch.py": (
        "cache_meta.write_text(",
        "cache_md.write_bytes(",
        "cache_md.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "host/src/host/thread.py": (
        'thread_path.open("a"',
        "thread_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "host/src/host/workflow_instance_store.py": (
        "self._path.parent.mkdir(parents=True, exist_ok=True)",
        "temp_path.write_text(",
        "os.replace(temp_path, self._path)",
    ),
    "host/src/host/commands/_packs.py": ("packs_venv.parent.mkdir(parents=True, exist_ok=True)",),
    "kernel/src/kernel/token_usage.py": (
        'self._path.open("a"',
        "self._path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "kernel/src/kernel/tools/mcp_client.py": (
        "Path(config.data_dir_path).mkdir(parents=True, exist_ok=True)",
    ),
    "mcp_server/src/mcp_server/config/cli.py": ("path.mkdir(parents=True, exist_ok=True)",),
    "sdk/src/sdk/content/_fetcher.py": ("cache_dir.mkdir(parents=True, exist_ok=True)",),
    "sdk/src/sdk/logging_setup.py": ("log_path.parent.mkdir(parents=True, exist_ok=True)",),
    "sdk/src/sdk/memory/store.py": (
        "sqlite3.connect(",
        "path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    "sdk/src/sdk/memory/backup.py": (
        "sqlite3.connect(",
        "target.write_text(",
        "path.unlink()",
        "target_dir.mkdir(parents=True, exist_ok=True)",
    ),
    "sdk/src/sdk/storage/_backend.py": (
        "(data_dir / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)",
        "(data_dir / _CHECKPOINTS_SUBDIR).mkdir(parents=True, exist_ok=True)",
        "resolved.parent.mkdir(parents=True, exist_ok=True)",
        'resolved.write_text(content, encoding="utf-8")',
        "resolved.unlink()",
        "resolved.mkdir(parents=True, exist_ok=True)",
        "checkpoints_root.mkdir(parents=True, exist_ok=True)",
        'path.write_text(json.dumps(bundle, default=str, indent=2), encoding="utf-8")',
        "path.parent.mkdir(parents=True, exist_ok=True)",
        'path.write_text(content, encoding="utf-8")',
    ),
}


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


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
    # Exclude build artifacts in external packs (e.g. lisa-packs/<pack>/build/...).
    return "/build/" not in rel


# In-tree scan roots (relative to repo root). Packages live under <member>/src/
# but rglob from the member root finds them regardless.
_IN_TREE_ROOTS: tuple[str, ...] = ("kernel", "host", "sdk", "mcp_server")

# External pack repos (relative to repo root's parent).
_EXTERNAL_PACK_DIRS: tuple[str, ...] = ("lisa-packs", "lisa-packs-wip")


def _iter_in_tree_files(root: Path) -> list[Path]:
    """Runtime .py files in kernel/host/sdk/mcp_server (excl. tests)."""
    files: list[Path] = []
    for top in _IN_TREE_ROOTS:
        top_dir = root / top
        if not top_dir.is_dir():
            continue
        for path in top_dir.rglob("*.py"):
            if _is_runtime_python(path):
                files.append(path)
    return sorted(files)


def _iter_external_pack_files(root: Path) -> list[Path]:
    """Runtime .py files in sibling pack repos (excl. tests).

    Returns paths relative to the parent of ``root`` so violations are
    grep-able across both repos (e.g. ``lisa-packs/okf/operations/...``).
    """
    parent = root.parent
    files: list[Path] = []
    for pack_repo in _EXTERNAL_PACK_DIRS:
        repo_dir = parent / pack_repo
        if not repo_dir.is_dir():
            continue
        for path in repo_dir.rglob("*.py"):
            if _is_runtime_python(path):
                files.append(path)
    return sorted(files)


# ---------------------------------------------------------------------------
# AST-based detection
# ---------------------------------------------------------------------------


class _IOLocator(ast.NodeVisitor):
    """Walk an AST tree and record lines containing banned I/O calls."""

    def __init__(self, *, scan_reads: bool) -> None:
        self.violations: list[tuple[int, str]] = []
        self._scan_reads = scan_reads

    def _check_method(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Attribute):
            return
        attr = func.attr
        if (
            attr in _BANNED_METHODS_WRITE
            or attr in _BANNED_METHODS_MUTATE
            or (self._scan_reads and attr in _BANNED_METHODS_READ)
        ):
            self.violations.append((node.lineno, ast.unparse(node)))

    def _check_module_func(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Attribute):
            return
        # e.g. os.replace, sqlite3.connect, shutil.copy
        if not isinstance(func.value, ast.Name):
            return
        key = (func.value.id, func.attr)
        if key in _BANNED_MODULE_FUNCS_WRITE or (
            self._scan_reads and key in _BANNED_MODULE_FUNCS_READ
        ):
            self.violations.append((node.lineno, ast.unparse(node)))

    def _check_open(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "open":
            return
        if not node.args:
            return
        mode_arg = node.args[0]
        if not isinstance(mode_arg, ast.Constant) or not isinstance(mode_arg.value, str):
            return
        mode = mode_arg.value
        # Write modes are always banned; read modes only for external packs.
        if mode[0:1] in _WRITE_OPEN_MODES or (self._scan_reads and mode[0:1] == "r"):
            self.violations.append((node.lineno, ast.unparse(node)))

    def visit_Call(self, node: ast.Call) -> None:
        self._check_method(node)
        self._check_module_func(node)
        self._check_open(node)
        self.generic_visit(node)


def _find_io_violations(source: str, *, scan_reads: bool) -> list[tuple[int, str]]:
    """Parse source and return (line_number, source_snippet) for banned calls."""
    try:
        tree = ast.parse(source, type_comments=True)
    except SyntaxError:
        return []
    locator = _IOLocator(scan_reads=scan_reads)
    locator.visit(tree)
    return locator.violations


# ---------------------------------------------------------------------------
# Allowlist matching
# ---------------------------------------------------------------------------


def _is_allowlisted(rel_path: str, line_text: str) -> bool:
    allowed_snippets = _ALLOWLIST.get(rel_path)
    if not allowed_snippets:
        return False
    return any(snippet in line_text for snippet in allowed_snippets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _check_file(
    path: Path,
    rel_path: str,
    *,
    scan_reads: bool,
    allowlist_enabled: bool,
) -> list[tuple[str, int, str]]:
    """Return violations for a single file as (rel_path, line_no, snippet)."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [(rel_path, 0, f"read error: {exc}")]

    lines = source.splitlines()
    violations: list[tuple[str, int, str]] = []

    for line_no, _snippet in _find_io_violations(source, scan_reads=scan_reads):
        line_text = lines[line_no - 1].strip() if line_no <= len(lines) else ""
        if allowlist_enabled and _is_allowlisted(rel_path, line_text):
            continue
        violations.append((rel_path, line_no, line_text))

    return violations


def main() -> int:
    root = _repo_root()

    # In-tree: writes only, allowlist applies. These are enforced (exit 1).
    in_tree_violations: list[tuple[str, int, str]] = []
    for path in _iter_in_tree_files(root):
        rel_path = path.relative_to(root).as_posix()
        in_tree_violations.extend(
            _check_file(
                path,
                rel_path,
                scan_reads=False,
                allowlist_enabled=True,
            )
        )

    # External packs: reads + writes, no allowlist. Reported as warnings
    # (exit 0) until the lisa-packs agent fixes them — see
    # ../lisa-packs/docs/STORAGE_IO_POLICY_SPEC.md.
    pack_violations: list[tuple[str, int, str]] = []
    parent = root.parent
    for path in _iter_external_pack_files(root):
        rel_path = path.relative_to(parent).as_posix()
        pack_violations.extend(
            _check_file(
                path,
                rel_path,
                scan_reads=True,
                allowlist_enabled=False,
            )
        )

    if in_tree_violations:
        print("Direct I/O policy violations detected (in-tree, enforced):")
        for rel_path, line_no, line in in_tree_violations:
            print(f"- {rel_path}:{line_no}: {line}")
        print()
        print("Every in-tree direct-write callsite must be allowlisted in")
        print("scripts/check_direct_io_policy.py (_ALLOWLIST) or migrated behind")
        print("lisa-storage / a documented host-internal carve-out.")

    if pack_violations:
        print()
        print("Direct I/O policy violations detected (external packs, advisory):")
        for rel_path, line_no, line in pack_violations:
            print(f"- {rel_path}:{line_no}: {line}")
        print()
        print("External packs: ALL direct file I/O is banned -- route through")
        print("lisa-storage via ctx.execute_tool_fn. See")
        print("../lisa-packs/docs/STORAGE_IO_POLICY_SPEC.md.")

    if in_tree_violations:
        return 1

    if pack_violations:
        print()
        print("In-tree check passed. External-pack violations are advisory")
        print("(exit 0) until the lisa-packs agent fixes them.")
        return 0

    print("Direct I/O policy check passed (no unallowlisted runtime I/O callsites).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
