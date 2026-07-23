"""Default router configuration for the lisa-router pack.

Declares the pack's default ``router:`` block that :func:`build_manifest_payload`
seeds into ``models.proposed.yaml`` when no user ``router:`` block is present.
This lets ``/models discover --force`` produce an immediately usable
routing config without the user hand-writing YAML.

The rules here match the documented defaults in ``docs/MODEL_ROUTER.md``:

- Route coding turns to a frontier code-capable model.
- Escalate to the deep model after the first tool returns a result.
- Downgrade to a cheaper model when the tool-call budget is running low.

Model ids are *symbolic* — they reference role names and model ids that the
discovered models may or may not contain. :func:`build_manifest_payload`
removes any rule whose ``use`` or ``fallback`` references a model that is
not in the discovered set, so the emitted YAML always loads cleanly.
"""

from __future__ import annotations

from typing import Any


def get_default_router_config() -> dict[str, Any]:
    """Return the default ``router:`` block for the lisa-router backend."""

    return {
        "backend": "lisa-router",
        "rules": [
            {
                "when": 'required_capabilities includes "code"',
                "use": "deep",
            },
            {
                "when": "iteration > 1 and last_tool_result is not None",
                "use": "deep",
            },
            {
                "when": "tool_calls_so_far >= 5",
                "use": "default",
            },
        ],
        "fallback": "default",
    }
