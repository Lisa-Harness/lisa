"""Sandboxed file I/O for the lisa-storage server.

Extracted verbatim from the former ``mcp_server/storage/server.py``: the
sandbox resolver, deny-glob matcher, ``PathRejected`` exception, and
model-manifest path logic. The MCP server is now a thin wrapper — it
wires ``--data-dir`` / ``--readonly`` / ``--deny-globs`` into a
:class:`StorageBackend` and exposes the 8 file tools plus the manifest
pair.

This gives the SDK a complete contract for storage access that can be
fulfilled without MCP — a host or tool that wants in-process file I/O
can call the backend directly instead of spawning the MCP server.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sdk.content import is_safe_relative_path
from sdk.layout import CONFIG_DIR, DATA_DIR, STATE_DIR

__all__ = ["PathRejected", "StorageBackend"]

_FILES_SUBDIR = "files"

logger = logging.getLogger(__name__)


class PathRejected(Exception):
    """Raised when a path is denied by the glob denylist policy.

    Distinct from the sandbox-escape and malformed-path rejections
    (which return ``None`` from :meth:`StorageBackend.resolve_file_path`)
    so callers can surface an explicit "denied by policy" error to the
    model/operator.
    """


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class StorageBackend:
    """Sandboxed file I/O backend.

    Owns two concerns, both derived from a single ``data_dir``:

    1. **File I/O** — a filesystem sandbox scoped to
       ``data_dir / "files"``. Path traversal (``..`` segments, absolute
       paths) is rejected so a crafted path cannot escape the sandbox.
       An optional deny-glob list (``fnmatch`` syntax, forward-slash
       paths, case-insensitive) adds defense-in-depth policy on top.
    2. **Model manifests** — ``models.yaml`` (active) and
       ``models.proposed.yaml`` (proposal) live directly under
       ``data_dir``, outside the files sandbox.

    Args:
        data_dir: The canonical data root. All storage paths derive from it.
        readonly: When ``True``, write / delete operations are rejected.
            The ``files/`` subdirectory is NOT created in readonly mode
            (it must pre-exist or reads will return empty / error results).
        deny_globs: Glob patterns (forward-slash, ``fnmatch`` syntax) matched
            against the relative path within the ``files/`` sandbox. Any
            match denies read/write/list/stat/delete/make_dir on that path.
            Empty by default (no denial). Glob syntax is portable
            (``*.db``, ``threads/**``, ``repl_history.txt``).
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        readonly: bool = False,
        deny_globs: tuple[str, ...] = (),
    ) -> None:
        self._data_dir: Path = data_dir
        self._readonly: bool = readonly
        self._deny_globs: tuple[str, ...] = tuple(deny_globs)
        if not readonly:
            (data_dir / DATA_DIR / _FILES_SUBDIR).mkdir(parents=True, exist_ok=True)
        # One-time migration: the legacy ``checkpoints/`` directory was
        # removed in Phase 5 of the checkpoint teardown. Best-effort rmtree;
        # failures (permissions, missing dir) are logged and ignored.
        legacy_checkpoints = data_dir / "checkpoints"
        if legacy_checkpoints.exists():
            try:
                shutil.rmtree(legacy_checkpoints)
            except OSError as exc:
                logger.warning(
                    "could not remove legacy checkpoints dir %s: %s",
                    legacy_checkpoints,
                    exc,
                )

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def readonly(self) -> bool:
        return self._readonly

    @property
    def deny_globs(self) -> tuple[str, ...]:
        return self._deny_globs

    # ── Roots ──────────────────────────────────────────────────────────────

    def files_root(self) -> Path:
        """Return the files sandbox root."""
        return self._data_dir / DATA_DIR / _FILES_SUBDIR

    def manifest_path(self, which: str) -> Path:
        """Return the path to the active or proposed model manifest.

        Manifests are first-class framework artifacts, so they live under
        the data dir's category dirs — not under the ``files/`` sandbox.
        The ``which`` argument selects the active manifest (``"active"``
        → ``config/models.yaml``) or the proposal (``"proposal"`` →
        ``state/models.proposed.yaml``).
        """
        normalized = which.strip().lower()
        if normalized == "active":
            return self._data_dir / CONFIG_DIR / "models.yaml"
        if normalized == "proposal":
            return self._data_dir / STATE_DIR / "models.proposed.yaml"
        raise ValueError(f"manifest kind must be 'active' or 'proposal', got {which!r}")

    # ── Policy ─────────────────────────────────────────────────────────────

    def denied_by_policy(self, normalized: str) -> bool:
        """True if the normalized relative path matches any deny glob.

        Matching is case-insensitive on the forward-slash-normalized path
        so patterns stay portable across Windows/POSIX. ``fnmatch`` is used
        (not ``pathlib.PurePath.match``) because the latter is per-segment
        and surprises with multi-segment ``**`` patterns.
        """
        if not self._deny_globs:
            return False
        posix = normalized.replace("\\", "/")
        lowered = posix.lower()
        return any(
            fnmatch.fnmatch(posix, pat) or fnmatch.fnmatch(lowered, pat.lower())
            for pat in self._deny_globs
        )

    def resolve_file_path(self, relative_path: str) -> Path | None:
        """Resolve a user-supplied relative path against the files root.

        Returns the absolute ``Path`` if safe, ``None`` if the path is
        rejected (empty, absolute, contains ``..``, or escapes the root
        after resolution), or raises :class:`PathRejected` if the path
        matches a deny glob.
        """
        raw = relative_path.strip()
        if not raw or Path(raw).is_absolute():
            return None
        normalized = raw.strip("/\\")
        if not is_safe_relative_path(normalized):
            return None
        if self.denied_by_policy(normalized):
            raise PathRejected(normalized)
        root = self.files_root()
        resolved = (root / normalized).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            return None
        return resolved

    def resolve_or_error(self, path: str) -> tuple[Path | None, str | None]:
        """Resolve a path and return ``(resolved, error_json)``.

        Consolidates the three rejection modes (unsafe/escape, policy-denied)
        into one call. On success: ``(Path, None)``. On rejection:
        ``(None, json.dumps(...))``.
        """
        try:
            resolved = self.resolve_file_path(path)
        except PathRejected:
            return None, json.dumps({"error": "path denied by policy"})
        if resolved is None:
            return None, json.dumps({"error": "path rejected: unsafe or escapes sandbox"})
        return resolved, None

    # ── File I/O ────────────────────────────────────────────────────────────

    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read a text file from the files sandbox.

        Returns the file content, or a JSON error object string if the
        path is unsafe, denied, or the file does not exist / is not valid
        text for the requested encoding.
        """
        resolved, err = self.resolve_or_error(path)
        if err is not None:
            return err
        assert resolved is not None
        if not resolved.is_file():
            return json.dumps({"error": f"file not found: {path}"})
        try:
            return resolved.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return json.dumps({"error": "file is not valid text for the requested encoding"})
        except OSError as exc:
            return json.dumps({"error": f"read failed: {exc}"})

    def write_file(self, path: str, content: str) -> str:
        """Write text content to a file in the files sandbox.

        Creates parent directories as needed. Respects ``readonly``.
        Returns a JSON object with ``ok``, ``path``, and ``bytes``.
        """
        if self._readonly:
            return json.dumps({"error": "server is read-only"})
        resolved, err = self.resolve_or_error(path)
        if err is not None:
            return err
        assert resolved is not None
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return json.dumps({"ok": True, "path": path, "bytes": len(content.encode("utf-8"))})
        except OSError as exc:
            return json.dumps({"error": f"write failed: {exc}"})

    def list_files(self, prefix: str = "") -> str:
        """List files under a prefix in the files sandbox.

        Returns a JSON array of relative paths (using forward slashes).
        Denied paths are filtered from results; a denied prefix returns
        a JSON error object.
        """
        root = self.files_root()
        if prefix.strip():
            normalized = prefix.strip().strip("/\\")
            if not is_safe_relative_path(normalized):
                return json.dumps({"error": "prefix rejected: unsafe or escapes sandbox"})
            if self.denied_by_policy(normalized):
                return json.dumps({"error": "prefix denied by policy"})
            search_root = (root / normalized).resolve()
            try:
                search_root.relative_to(root.resolve())
            except ValueError:
                return json.dumps({"error": "prefix escapes sandbox"})
        else:
            search_root = root

        if not search_root.is_dir():
            return json.dumps([])

        files: list[str] = []
        for item in sorted(search_root.rglob("*")):
            if item.is_file():
                rel = item.relative_to(root)
                rel_posix = str(rel).replace("\\", "/")
                if self._deny_globs and self.denied_by_policy(rel_posix):
                    continue
                files.append(rel_posix)
        return json.dumps(files)

    def delete_file(self, path: str) -> str:
        """Delete a file from the files sandbox.

        Fails on directories. Respects ``readonly``.
        """
        if self._readonly:
            return json.dumps({"error": "server is read-only"})
        resolved, err = self.resolve_or_error(path)
        if err is not None:
            return err
        assert resolved is not None
        if not resolved.exists():
            return json.dumps({"error": f"file not found: {path}"})
        if resolved.is_dir():
            return json.dumps({"error": "delete_file targets a directory; use a file path"})
        try:
            resolved.unlink()
        except OSError as exc:
            return json.dumps({"error": f"delete failed: {exc}"})
        return json.dumps({"ok": True, "path": path})

    def stat_file(self, path: str) -> str:
        """Return file metadata (size, modified timestamp, is_dir) for a path."""
        resolved, err = self.resolve_or_error(path)
        if err is not None:
            return err
        assert resolved is not None
        if not resolved.exists():
            return json.dumps({"error": f"not found: {path}"})
        stat = resolved.stat()
        return json.dumps(
            {
                "path": path,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                "is_dir": resolved.is_dir(),
            }
        )

    def make_dir(self, path: str) -> str:
        """Create a directory (and parents) in the files sandbox.

        Respects ``readonly``.
        """
        if self._readonly:
            return json.dumps({"error": "server is read-only"})
        resolved, err = self.resolve_or_error(path)
        if err is not None:
            return err
        assert resolved is not None
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            return json.dumps({"ok": True, "path": path})
        except OSError as exc:
            return json.dumps({"error": f"make_dir failed: {exc}"})

    def read_file_range(
        self,
        path: str,
        start: int = 0,
        length: int = 4096,
        encoding: str = "utf-8",
    ) -> str:
        """Read a char-bounded slice of a text file from the files sandbox.

        Returns a JSON object: ``{"content", "start", "length", "total",
        "eof"}``. ``total`` is the full char length of the file; ``length``
        is the chars actually returned (may be less than requested near
        EOF); ``eof`` is True when the slice reaches or passes the end.

        Slicing is char-based, not byte-based, so multibyte codepoints are
        never split. This requires decoding the whole file to slice safely;
        acceptable for text consolidation. Byte-range / lease semantics
        are intentionally out of scope (see module docstring).

        Args:
            start: Char offset to begin reading (0-based). Negative values
                and non-ints are coerced to 0.
            length: Max chars to return. Coerced to >= 0.
            encoding: Text encoding (default ``utf-8``).
        """
        resolved, err = self.resolve_or_error(path)
        if err is not None:
            return err
        assert resolved is not None
        if not resolved.is_file():
            return json.dumps({"error": f"file not found: {path}"})
        try:
            text = resolved.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return json.dumps({"error": "file is not valid text for the requested encoding"})
        except OSError as exc:
            return json.dumps({"error": f"read failed: {exc}"})

        total = len(text)
        s = max(0, int(start))
        n = max(0, int(length))
        end = min(total, s + n)
        content = text[s:end]
        return json.dumps(
            {
                "content": content,
                "start": s,
                "length": len(content),
                "total": total,
                "eof": end >= total,
            }
        )

    def append_file(self, path: str, content: str) -> str:
        """Append text content to a file in the files sandbox.

        Creates parent directories as needed. Creates the file if it does
        not exist. Respects ``readonly``. Returns a JSON object with
        ``ok``, ``path``, and ``bytes`` (the new total file size in bytes).
        """
        if self._readonly:
            return json.dumps({"error": "server is read-only"})
        resolved, err = self.resolve_or_error(path)
        if err is not None:
            return err
        assert resolved is not None
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            # newline="" disables platform newline translation so disk bytes
            # match the input exactly (a storage primitive should be
            # byte-faithful; ``write_file`` has the same property via
            # ``write_text``'s default, which also needs guarding — but
            # that's an existing concern out of scope here).
            with resolved.open("a", encoding="utf-8", newline="") as handle:
                handle.write(content)
            return json.dumps({"ok": True, "path": path, "bytes": resolved.stat().st_size})
        except OSError as exc:
            return json.dumps({"error": f"append failed: {exc}"})

    # ── Model manifests ─────────────────────────────────────────────────────

    def read_model_manifest(self, which: str = "proposal") -> str:
        """Read the active or proposed model manifest as text.

        Returns the manifest file content as text, or a JSON error object
        if the kind is invalid or the file does not exist.
        """
        try:
            path = self.manifest_path(which)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        if not path.is_file():
            return json.dumps({"error": f"manifest not found: {which} ({path.name})"})
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            return json.dumps({"error": f"read_model_manifest failed: {exc}"})

    def write_model_manifest(self, content: str, which: str = "proposal") -> str:
        """Write text content to the active or proposed model manifest.

        Creates the parent directory if needed. Respects ``readonly``.
        Returns a JSON object with ``ok``, ``which``, and ``bytes``.
        """
        if self._readonly:
            return json.dumps({"error": "server is read-only"})
        try:
            path = self.manifest_path(which)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return json.dumps({"ok": True, "which": which, "bytes": len(content.encode("utf-8"))})
        except OSError as exc:
            return json.dumps({"error": f"write_model_manifest failed: {exc}"})
