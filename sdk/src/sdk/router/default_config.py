"""Default router configuration for the lisa-router backend.

Declares a **minimalist** default ``router:`` block that
:func:`build_manifest_payload` seeds into ``models.proposed.yaml`` when no
user ``router:`` block is present. The goal is a small, immediately
understandable config — one capability escalation rule. When no rule
matches, the statically-resolved ``mode_models`` pin is the default (the
``fallback`` key was removed; a catch-all ``when: 'true'`` rule expresses
the old fallback explicitly). Richer examples (budget gates, phase gates,
code-model escalation, ``finish``, ``transition_to``) live in
``router/README.md`` as copy-paste recipes rather than cluttering every
generated manifest.

Model ids are *symbolic* — they reference mode names that the discovered
models may or may not contain. :func:`build_manifest_payload` removes any
rule whose ``use`` references a model that is not in the
discovered set, so the emitted YAML always loads cleanly.
"""

from __future__ import annotations

from typing import Any


def get_default_router_config() -> dict[str, Any]:
    """Return the minimal default ``router:`` block for the lisa-router backend."""

    return {
        "backend": "lisa-router",
        "rules": [
            # Escalate code-capable turns to the default model.
            {"when": '"code" in route.required_capabilities', "use": "default"},
        ],
    }
