"""Extension-owned scheduled jobs for Microsoft Clarity polling."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from kernel.extension_runtime import ExtensionRuntimeContext, ScheduledExtensionJob
from sdk_extensions.clarity.config import settings
from sdk_extensions.clarity.models import ClarityLiveInsightsResponse
from sdk_extensions.clarity.sanitizer import sanitize_clarity_record, sanitize_payload
from sdk_extensions.clarity.storage import (
    read_poll_metadata,
    write_live_insights,
    write_poll_metadata,
)

logger = logging.getLogger(__name__)


async def run_clarity_poll_job(runtime_context: ExtensionRuntimeContext) -> None:
    """Fetch, sanitize, persist, and hot-load the latest Clarity snapshot."""

    blob_io = runtime_context.blob_io
    executor = runtime_context.signal_connector_executor
    if blob_io is None or executor is None:
        logger.info("clarity poll skipped — runtime context is incomplete")
        return

    if not settings.clarity_api_token:
        logger.info("clarity poll skipped — no API token configured")
        return

    poll_started_at = datetime.now(UTC).isoformat()
    today_utc = datetime.now(UTC).strftime("%Y-%m-%d")
    poll_metadata = read_poll_metadata(blob_io)
    if poll_metadata.date_utc != today_utc:
        poll_metadata.date_utc = today_utc
        poll_metadata.request_count = 0

    if poll_metadata.request_count >= settings.clarity_daily_request_budget:
        poll_metadata.updated_at = poll_started_at
        poll_metadata.last_error = (
            f"daily request budget exhausted ({settings.clarity_daily_request_budget})"
        )
        write_poll_metadata(blob_io, poll_metadata)
        logger.warning(
            "clarity poll skipped — request budget exhausted for %s (%d/%d)",
            today_utc,
            poll_metadata.request_count,
            settings.clarity_daily_request_budget,
        )
        return

    poll_metadata.updated_at = poll_started_at
    poll_metadata.date_utc = today_utc
    poll_metadata.last_polled_at = poll_started_at
    poll_metadata.request_count += 1
    poll_metadata.last_error = ""
    write_poll_metadata(blob_io, poll_metadata)

    logger.info("clarity poll job started")
    if "clarity" not in executor.connector_ids():
        logger.info("clarity poll skipped — clarity extension not installed")
        return

    try:
        result = await executor.refresh_one("clarity")
        if result.run.status in ("unconfigured", "error"):
            poll_metadata.updated_at = datetime.now(UTC).isoformat()
            poll_metadata.last_error = (
                result.run.error.message if result.run.error else result.run.status
            )
            write_poll_metadata(blob_io, poll_metadata)
            logger.error("clarity fetch failed: status=%s", result.run.status)
            return
        snapshot = result.snapshot
        if not isinstance(snapshot, ClarityLiveInsightsResponse):
            poll_metadata.updated_at = datetime.now(UTC).isoformat()
            poll_metadata.last_error = "clarity snapshot type mismatch"
            write_poll_metadata(blob_io, poll_metadata)
            logger.error("clarity fetch failed: unexpected snapshot type")
            return
    except Exception as exc:  # pylint: disable=broad-except
        poll_metadata.updated_at = datetime.now(UTC).isoformat()
        poll_metadata.last_error = str(exc)
        write_poll_metadata(blob_io, poll_metadata)
        logger.error("clarity fetch failed: %s", exc)
        return

    typed_snapshot = snapshot.model_copy(deep=True)
    try:
        typed_snapshot.raw = sanitize_payload(typed_snapshot.raw)
        for metric in typed_snapshot.metrics:
            sanitized_info: list[dict[str, object]] = []
            for info in metric.information:
                if isinstance(info, dict):
                    sanitized_info.append(sanitize_clarity_record(info))
                else:
                    sanitized_info.append(info)
            metric.information = sanitized_info
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("clarity sanitization failed: %s", exc)

    try:
        write_live_insights(blob_io, typed_snapshot)
    except Exception as exc:  # pylint: disable=broad-except
        poll_metadata.updated_at = datetime.now(UTC).isoformat()
        poll_metadata.last_error = str(exc)
        write_poll_metadata(blob_io, poll_metadata)
        logger.error("clarity persist failed: %s", exc)
        return

    poll_metadata.updated_at = datetime.now(UTC).isoformat()
    poll_metadata.last_success_at = poll_metadata.updated_at
    poll_metadata.last_error = ""
    write_poll_metadata(blob_io, poll_metadata)

    if runtime_context.apply_state_updates is not None:
        try:
            await runtime_context.apply_state_updates(collections={"clarity": typed_snapshot})
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("clarity post-persist state update failed: %s", exc)


def build_clarity_jobs() -> list[ScheduledExtensionJob]:
    """Return the scheduled jobs owned by the Clarity extension."""

    if not settings.clarity_api_token:
        return []
    if settings.clarity_poll_schedule:
        return [
            ScheduledExtensionJob(
                job_id="clarity-poll",
                runner=run_clarity_poll_job,
                trigger="cron",
                cron_schedule=settings.clarity_poll_schedule,
                misfire_grace_time=60,
                run_immediately=True,
            )
        ]
    return [
        ScheduledExtensionJob(
            job_id="clarity-poll",
            runner=run_clarity_poll_job,
            trigger="cron",
            hour=4,
            minute=0,
            misfire_grace_time=60,
            run_immediately=True,
        )
    ]


__all__ = ["build_clarity_jobs", "run_clarity_poll_job"]
