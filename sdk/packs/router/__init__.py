"""Router pack — built-in rules-based model router.

Provides the ``lisa-router`` backend, a pure-Python rules engine that selects
a model per turn-loop iteration based on the manifest's ``router.rules``
block and the turn state (iteration number, tool calls so far, last tool
result, required capabilities). Zero model calls — the engine evaluates
rules locally against ``ManifestModelEntry`` metadata (``cost_tier``,
``latency_tier``, ``capabilities``) and ``RouterRequest`` fields.

Registered via the ``lisa.extensions.model_routers`` entry-point group. The
factory returns ``None`` when the manifest's ``router.backend`` is not
``"lisa-router"`` (or when no ``router`` block exists), so the host skips
wiring and falls back to static model resolution.

Lives in the framework tree at ``sdk/packs/router`` (import path
``sdk.packs.router``).
"""

from __future__ import annotations

from typing import Any

from kernel.model_manifest import ModelManifest, RouterConfig
from kernel.seams.protocols import ModelRouter


def get_router_factory():
    """Return the factory callable for ``lisa.extensions.model_routers``.

    The factory takes ``(manifest, session_plan)`` and returns a
    ``LisaRouter`` when the manifest's ``router.backend`` is ``"lisa-router"``
    (and a ``router`` block exists), or ``None`` otherwise.
    """

    return _factory


def _factory(
    manifest: ModelManifest | None,
    _session_plan: Any,
) -> ModelRouter | None:
    if manifest is None or manifest.router is None:
        return None
    if manifest.router.backend != "lisa-router":
        return None
    from sdk.packs.router.router import LisaRouter

    return LisaRouter(manifest)


__all__ = ["get_router_factory"]
