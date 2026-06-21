"""Public rendering contracts shared by the host and external authors.

These types live in ``lisa_sdk`` so extension and pack authors can contribute
rich reply content and refresh-summary detail blocks.
Host layers still own wire conversion and channel-specific delivery.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import AttachmentKind, UnifiedState


class RichAttachment(BaseModel):
    """One host-neutral attachment produced by agent and extension workflows.

    Hosts convert these attachments into their own channel SDK wire objects.
    """

    attachment_kind: AttachmentKind
    content_type: str
    name: str = ""
    content_json: dict[str, Any] = Field(default_factory=dict)
    content_bytes: bytes | None = Field(default=None, exclude=True)

    @classmethod
    def adaptive_card(cls, content: dict[str, Any]) -> RichAttachment:
        """Build one Adaptive Card attachment payload."""
        return cls(
            attachment_kind="adaptive_card",
            content_type="application/vnd.microsoft.card.adaptive",
            content_json=content,
        )

    @classmethod
    def image(
        cls,
        *,
        image_bytes: bytes,
        name: str,
        content_type: str = "image/png",
    ) -> RichAttachment:
        """Build one inline image attachment payload."""
        return cls(
            attachment_kind="image",
            content_type=content_type,
            name=name,
            content_bytes=image_bytes,
        )


@dataclass(frozen=True, slots=True)
class RefreshSummarySection:
    """Extension-owned details for one connector block in the refresh card."""

    connector_id: str
    facts: tuple[tuple[str, str], ...] = ()
    lines: tuple[str, ...] = ()
    order: int = 100


RefreshSummarySectionBuilder = Callable[[UnifiedState], Iterable[RefreshSummarySection]]


__all__ = [
    "RefreshSummarySection",
    "RefreshSummarySectionBuilder",
    "RichAttachment",
]
