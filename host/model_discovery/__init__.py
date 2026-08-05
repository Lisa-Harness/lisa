"""Robust model discovery for Lisa hosts.

Replaces the former flat ``host/model_discovery.py`` module with a
typed, redacted, per-provider probe package. The public surface is
preserved for callers (``commands.py``):

- :class:`DiscoveredModel`
- :class:`DiscoveryReport`
- :func:`discover_models`
- :func:`build_manifest_payload`
- :func:`render_manifest_yaml`
- :func:`write_manifest_yaml`

Internal additions (typed error surface, redaction, probe protocol) are
not part of the stable API but may be imported by tests.
"""

from __future__ import annotations

from .discover import discover_models
from .manifest import (
    build_manifest_payload,
    render_manifest_yaml,
    write_manifest_yaml,
)
from .types import (
    DiscoveredModel,
    DiscoveryReport,
    ProbeError,
    ProbeResult,
    ProbeStatus,
)

__all__ = (
    "DiscoveredModel",
    "DiscoveryReport",
    "ProbeError",
    "ProbeResult",
    "ProbeStatus",
    "build_manifest_payload",
    "discover_models",
    "render_manifest_yaml",
    "write_manifest_yaml",
)
