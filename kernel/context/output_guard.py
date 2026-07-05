"""Default ``OutputGuard`` that allows everything.

Production hosts override with a real guard (Azure AI Content Safety,
Bedrock Guardrails, Presidio, etc.) that applies output validation before
the response reaches the user. The pass-through default keeps public hosts
runnable without a private safety pipeline.
"""

from __future__ import annotations

from kernel.models import SurfaceContext
from kernel.seams.protocols import GuardResult


class PassThroughOutputGuard:
    """Identity output guard — allows everything, sanitizes nothing."""

    async def validate(
        self,
        text: str,  # noqa: ARG002
        surface: SurfaceContext,  # noqa: ARG002
    ) -> GuardResult:
        """Return an allow result with no sanitized text."""

        return GuardResult(allowed=True)