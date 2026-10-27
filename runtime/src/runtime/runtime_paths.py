"""HostPaths — derived filesystem locations for a host's data layout.

A frozen, pure value object computed once from ``HostConfig``. It owns no
locks, I/O, or async context — it simply projects the resolved directories
and files a host reads or writes. ``HostRuntime.paths`` holds one instance
so callers read ``runtime.paths.thread`` instead of re-running the config's
resolution logic through a mutable object.

(Not ``runtime.paths``: that module owns repo-root/venv resolution and is
unrelated — see :mod:`runtime.paths`.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdk.layout import (
    CACHE_DIR,
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    STATE_DIR,
)

from runtime.config import HostConfig


@dataclass(frozen=True, slots=True)
class HostPaths:
    """Every filesystem path a host runtime owns.

    Attributes are plain resolved ``Path`` objects (or ``None`` when the
    feature is disabled). Consumers never recompute these and never depend
    on config-resolution order — the values are final.
    """

    data_dir: Path
    overlays_dir: Path
    thread_dir: Path
    thread_path: Path | None
    log_path: Path | None
    host_log_path: Path
    token_usage_path: Path
    decisions_path: Path
    models_path: Path
    models_proposal_path: Path

    # ── Category directories (sdk.layout naming) ────────────────────
    @property
    def config_dir(self) -> Path:
        """Hand-editable + resolved configuration (models.yaml, policy)."""
        return self.data_dir / CONFIG_DIR

    @property
    def state_dir(self) -> Path:
        """Machine-written, small, back-up-able state (approvals, lock)."""
        return self.data_dir / STATE_DIR

    @property
    def durable_dir(self) -> Path:
        """Large durable storage (memory.db, threads/, files/)."""
        return self.data_dir / DATA_DIR

    @property
    def cache_dir(self) -> Path:
        """rm-safe caches (fetch/, layers/)."""
        return self.data_dir / CACHE_DIR

    @property
    def logs_dir(self) -> Path:
        """Append-only logs (host.jsonl, mcp/, token-usage)."""
        return self.data_dir / LOGS_DIR

    def run_dir(self, session_key: str) -> Path:
        """Per-session execution state root (``run/<session_key>``).

        The bash/bwrap packs keep jobs, scratch, profile files, sandbox
        facts, and per-session logs inside this one gc-able tree.
        """
        return self.data_dir / "run" / session_key

    @classmethod
    def from_config(
        cls,
        config: HostConfig,
        *,
        models_path_override: Path | None = None,
    ) -> HostPaths:
        """Derive all filesystem paths from a resolved ``HostConfig``.

        ``models_path_override`` is the session-plan-declared model
        manifest path (``context.models.path``) when present; otherwise the
        config's ``resolved_models_path`` is used.
        """
        thread_path: Path | None = None
        if not config.no_thread and config.user_slug:
            thread_path = config.resolved_thread_dir / config.user_slug
        return cls(
            data_dir=config.data_dir,
            overlays_dir=config.resolved_overlays_dir,
            thread_dir=config.resolved_thread_dir,
            thread_path=thread_path,
            log_path=config.resolved_log_path,
            host_log_path=config.resolved_host_log_path,
            token_usage_path=config.data_dir / LOGS_DIR / "token-usage.jsonl",
            decisions_path=config.data_dir / LOGS_DIR / "decisions.jsonl",
            models_path=models_path_override or config.resolved_models_path,
            models_proposal_path=config.resolved_models_proposal_path,
        )


__all__ = ["HostPaths"]
