"""Authored context-frame helpers for prompt assembly."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from kernel.context.content_registry import DiskContentRegistry
from kernel.models import ContextEvidenceNamespace
from kernel.seams.protocols import ContentRegistry

ContextFrameId = Literal[
    "surface_context",
    "private_user_context",
    "shared_surface_context",
    "ranked_evidence",
    "live_snapshot",
    "passive_intelligence",
    "memory_recall",
    "reflection",
    "architecture_context",
    "capability_snapshot",
    "answer_planner",
]

CONTEXT_FRAME_IDS: tuple[ContextFrameId, ...] = (
    "surface_context",
    "private_user_context",
    "shared_surface_context",
    "ranked_evidence",
    "live_snapshot",
    "passive_intelligence",
    "memory_recall",
    "reflection",
    "architecture_context",
    "capability_snapshot",
    "answer_planner",
)

_FRAME_ASSET_PREFIX = "prompts/context_frames"
_RANKED_EVIDENCE_NAMESPACE_PREFIX = "prompts/context_frames/ranked_evidence"

# Module-level content registry. Hosts override via ``set_content_registry``
# when they need a custom registry; the default reads from discovered
# content roots.
_content_registry: ContentRegistry = DiskContentRegistry()


def set_content_registry(registry: ContentRegistry) -> None:
    """Override the module-level content registry and invalidate caches."""

    global _content_registry
    _content_registry = registry
    read_context_frame.cache_clear()
    read_ranked_evidence_namespace_frame.cache_clear()


@lru_cache(maxsize=len(CONTEXT_FRAME_IDS))
def read_context_frame(frame_id: ContextFrameId) -> str:
    """Load one authored context frame from the merged content registry."""

    return _content_registry.read_authored_text(f"{_FRAME_ASSET_PREFIX}/{frame_id}.md").strip()


def render_context_frame(frame_id: ContextFrameId, body: str) -> str:
    """Render one framed context body, preserving empty-body no-ops."""

    normalized_body = body.strip()
    if not normalized_body:
        return ""
    frame = read_context_frame(frame_id)
    if not frame:
        return normalized_body
    return f"{frame}\n\n{normalized_body}"


@lru_cache(maxsize=8)
def read_ranked_evidence_namespace_frame(namespace: ContextEvidenceNamespace) -> str:
    """Load the authored consumption frame for one ranked-evidence namespace."""

    return _content_registry.read_authored_text(
        f"{_RANKED_EVIDENCE_NAMESPACE_PREFIX}/{namespace}.md"
    ).strip()


def render_ranked_evidence_namespace_frame(
    namespace: ContextEvidenceNamespace,
    body: str,
) -> str:
    """Render one ranked-evidence namespace frame plus its selected evidence insert."""

    normalized_body = body.strip()
    if not normalized_body:
        return ""
    frame = read_ranked_evidence_namespace_frame(namespace)
    if not frame:
        return normalized_body
    return f"{frame}\n\n{normalized_body}"
