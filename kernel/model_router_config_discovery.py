"""Discovery of pack-contributed model-router configurations.

Packs declare a default ``RouterConfig`` (backend, rules, fallback) through
the ``lisa.extensions.model_router_configs`` entry-point group. Each entry
point returns a callable ``() -> dict[str, Any]`` whose output mirrors the
shape of the ``router:`` block in ``models.yaml``:

.. code-block:: python

    def get_default_router_config() -> dict[str, Any]:
        return {
            "backend": "lisa-router",
            "rules": [
                {"when": "required_capabilities includes \"code\"", "use": "gpt-5.2-codex"},
                {"when": "iteration > 1 and last_tool_result is not None", "use": "gpt-5.4"},
                {"when": "tool_calls_so_far >= 5", "use": "gpt-5.4-mini"},
            ],
            "fallback": "gemma4:cloud",
        }

The host merges pack-declared router configs with the user's manifest at
model-discovery time (:func:`hosts.core.model_discovery.manifest.build_manifest_payload`):

- A user-provided ``router:`` block in ``models.yaml`` always wins (user
  wins). The pack default is only seeded when the user has not written one.
- When multiple packs declare a router config, the first entry-point in
  discovery order wins — the same first-non-None convention used by
  ``lisa.extensions.model_routers`` and ``lisa.extensions.slice_providers``.
- Failures loading a single entry point are logged and skipped so one
  broken pack does not poison router-config discovery for the rest.

This is the config-layer parallel to :mod:`kernel.model_router_discovery`:
that module discovers *runtime factories* (``ModelRouter`` instances); this
module discovers *static config* (the ``RouterConfig`` dict that the
factories consume). Without this layer, installing a router pack alone is
not enough to activate routing — the user must also hand-write the
``router:`` block in ``models.yaml``. With this layer, the pack's default
config seeds the proposed manifest automatically.

This is also the router-side parallel of :mod:`kernel.model_roles`: that
module seeds ``roles`` from packs; this module seeds ``router:`` from packs.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ROUTER_CONFIGS_ENTRY_POINT_GROUP = "lisa.extensions.model_router_configs"


def discover_pack_router_config() -> dict[str, Any] | None:
    """Return the first pack-declared router config, or ``None``.

    Iterates entry points in the ``lisa.extensions.model_router_configs``
    group in discovery order. Each entry point is loaded and called with
    no arguments; the first one that returns a ``dict`` is used
    (first-non-None-wins). Later entry points are not consulted.

    Returns ``None`` when no pack declares a router config or all entry
    points fail to load / return a non-dict. The caller (manifest builder)
    treats ``None`` as "no pack-default router; emit no ``router:`` block".
    """

    for ep in importlib.metadata.entry_points(group=MODEL_ROUTER_CONFIGS_ENTRY_POINT_GROUP):
        try:
            factory = ep.load()
            declared = factory() if callable(factory) else factory
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load model_router_configs entry-point %r",
                ep.name,
                exc_info=True,
            )
            continue
        if isinstance(declared, dict):
            return declared
        logger.warning(
            "model_router_configs entry-point %r returned %r, expected a dict; skipping",
            ep.name,
            type(declared).__name__,
        )
    return None


__all__ = [
    "MODEL_ROUTER_CONFIGS_ENTRY_POINT_GROUP",
    "discover_pack_router_config",
]