"""Default router configuration for the lisa-router backend.

Declares the backend's default ``router:`` block that :func:`build_manifest_payload`
seeds into ``models.proposed.yaml`` when no user ``router:`` block is present.
This lets ``/models discover --force`` produce an immediately usable
routing config without the user hand-writing YAML.

The rules are ordered so they never shadow each other:

- **Capability gates** (rules 1-3): a mode that needs code, reasoning, or
  vision gets the deep model for the whole turn. These fire first so
  capability-needing modes never hit the budget or phase gates below.
- **Budget gate** (rule 4): a long, tool-heavy turn on a plain mode
  downgrades to the default model before we pay for frontier synthesis.
- **Phase gate** (rule 5): after the first tool returns, synthesize on the
  deep model. Only reached on plain (non-capability) modes because rules
  1-3 already captured code/reasoning/vision turns.

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
                "when": 'required_capabilities includes "reasoning"',
                "use": "deep",
            },
            {
                "when": 'required_capabilities includes "vision"',
                "use": "deep",
            },
            {
                "when": "tool_calls_so_far >= 5",
                "use": "default",
            },
            {
                "when": "iteration > 1 and last_tool_result is not None",
                "use": "deep",
            },
        ],
        "fallback": "default",
    }