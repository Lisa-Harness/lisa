"""HostPaths — derived filesystem locations for a host's data layout.

A frozen, pure value object computed once from ``HostConfig``. It owns no
locks, I/O, or async context — it simply projects the resolved directories
and files a host reads or writes. ``HostRuntime.paths`` holds one instance
so callers read ``runtime.paths.thread`` instead of re-running the config's
resolution logic through a mutable object.

(Not ``host.paths``: that module owns repo-root/venv resolution and is
unrelated — see :mod:`host.paths`.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from host.config import HostConfig


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
    models_path: Path
    models_proposal_path: Path

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
            token_usage_path=config.data_dir / "token-usage.jsonl",
            models_path=models_path_override or config.resolved_models_path,
            models_proposal_path=config.resolved_models_proposal_path,
        )


__all__ = ["HostPaths"]
