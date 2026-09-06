"""Shared workspace-root resolution for packs.

The ``workspace_root`` path is the root of the user's working project. It is
NOT carried on the execution context (which only has a ``workspace_id`` slug);
packs read it from session pack configuration. Three packs were deriving it
independently with three different strictness rules — the coding pack
hard-failing unless it is a Git worktree, the bwrap sandbox tolerantly falling
back to the process CWD, and a host command hand-rolling a third path. This
module is the single resolver they share so expansion, traversal, and the
directory check live in exactly one place, with strictness as an explicit knob
rather than an accident of which pack ran first.

Strictness contract (one policy, two caller intents):

- ``require_git=False`` (lenient) — return the first existing directory among:
  the configured ``workspace_root``, then the process CWD, then ``data_dir``,
  then the user's home. Never raises for a missing/wrong root; returns the best
  available anchor. Used by the sandbox capture path, where "no workspace" is a
  degraded-but-runnable condition, not an error.
- ``require_git=True`` (strict) — resolve only the configured
  ``workspace_root`` and raise :class:`WorkspaceRootError` unless it is a real
  Git worktree. Used by the coding pack, whose whole workflow is undefined
  outside a Git checkout.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["WorkspaceRootError", "resolve_workspace_root"]

_WORKSPACE_ROOT_KEY = "workspace_root"


class WorkspaceRootError(ValueError):
    """Raised when a strict workspace-root resolution fails."""


def _read_configured_root(pack_config: object) -> str:
    """Read the raw ``workspace_root`` string from pack configuration.

    Accepts either a mapping (the common session pack-config shape) or a typed
    config object exposing a ``workspace_root`` attribute.
    """
    if isinstance(pack_config, dict):
        return str(pack_config.get(_WORKSPACE_ROOT_KEY, "") or "")
    return str(getattr(pack_config, _WORKSPACE_ROOT_KEY, "") or "")


def _resolve_candidate(raw: str) -> Path | None:
    """Expand and resolve a configured path; None when it is not a directory."""
    if not raw:
        return None
    try:
        candidate = Path(raw).expanduser()
        root = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
    except OSError:
        return None
    return root if root.is_dir() else None


def resolve_workspace_root(
    ctx: object,
    *,
    require_git: bool = False,
) -> Path:
    """Resolve the workspace root from pack configuration on ``ctx``.

    ``ctx`` is an execution context carrying ``pack_config`` and (optionally)
    ``data_dir``; typed as ``object`` because the SDK must not import the
    kernel's higher-level models.

    Args:
        ctx: The execution context (``pack_config`` + ``data_dir`` source).
        require_git: When True, require the configured root to be a real Git
            worktree and raise :class:`WorkspaceRootError` otherwise. When
            False, fall back across CWD then ``data_dir`` then home.

    Returns:
        The resolved workspace root directory.

    Raises:
        WorkspaceRootError: In strict mode, when the configured root is absent,
            not a directory, or not a Git worktree.
    """
    pack_config = getattr(ctx, "pack_config", None)
    raw = _read_configured_root(pack_config)
    configured = _resolve_candidate(raw)

    if require_git:
        if not raw:
            raise WorkspaceRootError(
                "workspace_root is not configured for this session."
            )
        if configured is None:
            raise WorkspaceRootError(f"workspace_root is not a directory: {raw}")
        if not (configured / ".git").exists():
            raise WorkspaceRootError(
                f"workspace_root is not a Git worktree: {configured}"
            )
        return configured

    # Lenient: configured root if it is a directory, else the process CWD, then
    # the data dir, then home. "No workspace" is a degraded-but-runnable state
    # for the sandbox capture path, never an exception here.
    if configured is not None:
        return configured
    cwd = Path.cwd()
    if cwd.is_dir():
        return cwd
    data_dir = getattr(ctx, "data_dir", "")
    if data_dir and Path(str(data_dir)).is_dir():
        return Path(str(data_dir))
    return Path.home()
