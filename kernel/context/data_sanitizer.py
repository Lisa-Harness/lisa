"""Default ``DataSanitizer`` that performs no redaction.

Production hosts override with a real sanitizer that applies privacy
redaction before memory writes and context assembly. The pass-through
default keeps public hosts runnable without a private redaction pipeline.
"""

from __future__ import annotations


class PassThroughSanitizer:
    """Identity sanitizer — returns input unchanged."""

    def redact(self, text: str, *, scope: str) -> str:  # noqa: ARG002
        """Return ``text`` unchanged. ``scope`` is accepted for protocol parity."""

        return text