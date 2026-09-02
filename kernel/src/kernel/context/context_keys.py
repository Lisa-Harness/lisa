"""App-owned identifiers for dynamic prompt context sources.

These keys label assembled prompt context across runtime prompt assembly,
debug replay, and conversation-digest hygiene. They stay in the app layer
because they describe Lisa's prompt surfaces rather than kernel-neutral
contracts.
"""

from __future__ import annotations

from typing import Final

RUNTIME_SURFACE_CONTEXT_KEY: Final = "runtime-surface-context"
AUTHORED_USER_CONTEXT_KEY: Final = "authored-user-context"
SHARED_RUNTIME_SURFACE_CONTEXT_KEY: Final = "shared-runtime-surface-context"

DIGEST_HYGIENE_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        RUNTIME_SURFACE_CONTEXT_KEY,
        AUTHORED_USER_CONTEXT_KEY,
        SHARED_RUNTIME_SURFACE_CONTEXT_KEY,
    }
)

CONTEXT_SOURCE_LABELS: dict[str, str] = {
    RUNTIME_SURFACE_CONTEXT_KEY: "surface metadata",
    SHARED_RUNTIME_SURFACE_CONTEXT_KEY: "shared surface context",
    AUTHORED_USER_CONTEXT_KEY: "private user profile",
}
