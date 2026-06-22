"""Extension-owned storage helpers for Clarity poll metadata and snapshots."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol

from kernel.seams.protocols import BlobIO
from lisa_sdk.config import settings as runtime_settings
from sdk_extensions.clarity.config import settings
from sdk_extensions.clarity.connector import hydrate_live_insights_snapshot
from sdk_extensions.clarity.models import ClarityLiveInsightsResponse, ClarityPollMetadata

logger = logging.getLogger(__name__)

CLARITY_POLL_METADATA_BLOB = "clarity-poll-metadata.json"
CLARITY_INSIGHTS_BLOB = "clarity-live-insights.json"


class ClarityExportBlobIO(Protocol):
    """Narrow blob contract for raw Clarity export helpers."""

    def list_blobs(self, container: str, prefix: str = "") -> list[str]: ...
    def read_blob_text(self, container: str, blob_path: str) -> str | None: ...


def read_poll_metadata(blob_io: BlobIO) -> ClarityPollMetadata:
    """Read persisted Clarity poll metadata, or return a default payload."""

    raw = blob_io.read_blob_text(runtime_settings.state_container, CLARITY_POLL_METADATA_BLOB)
    if not raw:
        return ClarityPollMetadata()
    try:
        return ClarityPollMetadata.model_validate_json(raw)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("clarity poll metadata read failed: %s", exc)
        return ClarityPollMetadata()


def write_poll_metadata(blob_io: BlobIO, metadata: ClarityPollMetadata) -> None:
    """Persist Clarity poll metadata through the generic blob seam."""

    blob_io.write_blob_text(
        runtime_settings.state_container,
        CLARITY_POLL_METADATA_BLOB,
        metadata.model_dump_json(indent=2),
        content_type="application/json",
    )


def write_live_insights(blob_io: BlobIO, snapshot: ClarityLiveInsightsResponse) -> None:
    """Persist the latest Clarity snapshot and one daily archive copy."""

    persisted_snapshot = hydrate_live_insights_snapshot(snapshot.model_copy(deep=True))
    payload = persisted_snapshot.model_dump_json(indent=2)
    blob_io.write_blob_text(
        runtime_settings.intel_inbox_container,
        CLARITY_INSIGHTS_BLOB,
        payload,
        content_type="application/json",
    )

    date_key = datetime.now(UTC).strftime("%Y-%m-%d")
    blob_io.write_blob_text(
        runtime_settings.intel_inbox_container,
        f"clarity-history/{date_key}.json",
        payload,
        content_type="application/json",
    )


def read_live_insights(blob_io: BlobIO) -> ClarityLiveInsightsResponse | None:
    """Read the latest Clarity snapshot and hydrate legacy-compatible fields."""

    raw = blob_io.read_blob_text(runtime_settings.intel_inbox_container, CLARITY_INSIGHTS_BLOB)
    if not raw:
        return None
    try:
        snapshot = ClarityLiveInsightsResponse.model_validate_json(raw)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("clarity snapshot read failed: %s", exc)
        return None
    return hydrate_live_insights_snapshot(snapshot)


def list_export_blobs(
    blob_io: ClarityExportBlobIO,
    prefix: str | None = None,
    container: str | None = None,
) -> list[str]:
    """List raw Clarity export blobs through generic host blob helpers."""

    container_name = container or settings.clarity_blob_container
    name_prefix = prefix or settings.clarity_blob_prefix
    return blob_io.list_blobs(container_name, name_prefix)


def download_export_blob_text(
    blob_io: ClarityExportBlobIO,
    blob_name: str,
    container: str | None = None,
) -> str:
    """Read one raw Clarity export blob as UTF-8 text through generic blob helpers."""

    container_name = container or settings.clarity_blob_container
    return blob_io.read_blob_text(container_name, blob_name) or ""


__all__ = [
    "CLARITY_INSIGHTS_BLOB",
    "CLARITY_POLL_METADATA_BLOB",
    "download_export_blob_text",
    "list_export_blobs",
    "read_live_insights",
    "read_poll_metadata",
    "write_live_insights",
    "write_poll_metadata",
]
