"""Default ``ExtensionEvidenceBuilder`` that returns no evidence.

Production hosts override with a builder that walks installed extensions and
collects their context-evidence contributions. The pass-through default keeps
public hosts runnable without the private extension-evidence pipeline.
"""

from __future__ import annotations

from typing import Any


class NoOpExtensionEvidenceBuilder:
    """Return an empty evidence list for any state."""

    def build_extension_context_evidence(
        self,
        runtime_context: Any,  # noqa: ARG002
        state: Any,  # noqa: ARG002
    ) -> list[Any]:
        """Return an empty list. Both arguments are accepted for protocol parity."""

        return []