"""Extension-owned `/debug context` fragments for Clarity poll state."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionDebugContextFragment, ExtensionRuntimeContext
from lisa_sdk.config import settings as runtime_settings
from sdk_extensions.clarity import storage


def build_debug_context_fragments(
    runtime_context: ExtensionRuntimeContext,
) -> list[ExtensionDebugContextFragment]:
    """Expose Clarity-owned poll metadata and artifact pointers for `/debug context`."""

    blob_io = runtime_context.blob_io
    if blob_io is None:
        return []

    metadata = storage.read_poll_metadata(blob_io)
    return [
        ExtensionDebugContextFragment(
            order=10,
            poll_metadata_lines=(
                "- clarity: "
                f"date `{metadata.date_utc or 'unknown'}` | "
                f"requests `{metadata.request_count}` | "
                f"last_success `{metadata.last_success_at or 'never'}` | "
                f"last_error `{metadata.last_error or 'none'}`",
            ),
            artifact_lines=(
                "- clarity metadata blob: "
                f"`{runtime_settings.state_container}/{storage.CLARITY_POLL_METADATA_BLOB}`",
                "- clarity snapshot blob: "
                f"`{runtime_settings.intel_inbox_container}/{storage.CLARITY_INSIGHTS_BLOB}`",
            ),
        )
    ]


__all__ = ["build_debug_context_fragments"]
