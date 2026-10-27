"""Turn forensics — read recorded evidence about Lisa's own past turns.

Thin by design. All evidence reading lives in :mod:`sdk.forensics` (shared with
the host's introspection path) and all rendering lives in
:func:`sdk.forensics.render_bundle`. This module resolves the host's data root
and the caller's scope from ``OperationExecutionContext``, applies the
parameters, and returns the rendered bundle.

Scope rule: ``user_slug`` and the data root come from the execution context —
never from an operation parameter — so a caller cannot ask to read another
user's evidence by naming it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from kernel.models import OperationExecutionContext, OperationResult
from pydantic import BaseModel, Field
from sdk.content import require_authored_text
from sdk.forensics import ForensicReader, render_bundle
from sdk.layout import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

#: Authored content. Per the pack-content rule, user-facing wording lives in
#: versioned Markdown rather than in this module, and a missing asset fails
#: loudly instead of degrading to inline prose.
_UNAVAILABLE_PATH = "operations/forensics/prompts/unavailable.md"

#: Hard ceilings. A forensic read is a summary; a wider question is a second
#: call, not a bigger dump.
_MAX_RECORDS_LIMIT = 200
_MIN_RECORDS = 5

Focus = Literal["all", "blocked", "receipts", "cost", "latency"]


class Parameters(BaseModel):
    """Inputs for one forensic read.

    ``turns`` is intentionally a string so the model can express "the last
    turn" or "turns 3 through 7" the way it would say it. Invalid selections
    are reported in the bundle's notes rather than raising, because a
    mis-specified range still leaves useful evidence to read.
    """

    turns: str = Field(
        default="last",
        description=(
            "Which recorded turns to examine: 'last' for the most recent turn, "
            "a single index like '12', or an inclusive range like '3-7'."
        ),
    )
    focus: Focus = Field(
        default="all",
        description=(
            "Which facet to lead with: 'blocked' (contract failures and their "
            "origins), 'receipts' (what actually executed), 'cost' (tokens and "
            "spend), 'latency' (dispatch durations), or 'all'. Every focus still "
            "returns the full evidence — it changes emphasis, not filtering, so "
            "asking a narrow question cannot hide the finding."
        ),
    )
    max_records: int = Field(
        default=50,
        ge=_MIN_RECORDS,
        le=_MAX_RECORDS_LIMIT,
        description=(
            f"Cap on dispatch rows and log lines included (default 50, max "
            f"{_MAX_RECORDS_LIMIT}). A cap that fires is reported in the read notes."
        ),
    )
    include_raw: bool = Field(
        default=False,
        description=(
            "Include un-redacted raw fields where they are available. Off by "
            "default: the thread store holds raw prompts and model reasoning, "
            "and the thread readout deliberately withholds them. Only enable "
            "this when the redacted view genuinely cannot answer the question."
        ),
    )


def _resolve_data_dir(ctx: OperationExecutionContext) -> Path | None:
    """Return the host's resolved data root, or ``None`` when unwired.

    ``data_dir`` has a no-fallback contract on the context: hosts always inject
    an absolute path. ``None`` here means the host did not wire it, which the
    bundle must report as an unavailable source rather than guessing a path
    relative to the process CWD.
    """
    raw = str(getattr(ctx, "data_dir", "") or "").strip()
    return Path(raw) if raw else None


def _resolve_thread_dir(data_dir: Path, ctx: OperationExecutionContext) -> Path:
    """Return the thread directory, honoring the host's layout contract.

    ``<data-dir>/data/threads`` is the canonical location. A host that moved it
    sets ``LISA_THREAD_DIR``; the reader accepts the resolved path so this
    module does not have to duplicate the host's path resolution beyond the
    documented default.
    """
    import os

    override = os.getenv("LISA_THREAD_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    del ctx  # thread dir is derived from the data root only
    return data_dir / DATA_DIR / "threads"


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Read one forensic bundle for the caller's scope."""
    params = Parameters.model_validate(ctx.params or {})

    data_dir = _resolve_data_dir(ctx)
    if data_dir is None:
        # Report the misconfiguration as evidence about the read, not as a
        # crashed operation: the model should learn that it cannot see its own
        # logs yet, and why.
        logger.warning("turn_forensics: ctx.data_dir is empty; cannot read evidence")
        return OperationResult(
            output_text=require_authored_text(
                ctx.host.read_authored_text if ctx.host is not None else None,
                _UNAVAILABLE_PATH,
                context="turn_forensics operation",
            ),
            output_json={"available": False, "reason": "missing_data_dir"},
            status="error",
            error_detail="ctx.data_dir is empty",
        )

    if params.include_raw:
        # Deliberate, logged, per-call widening of the read surface. The raw
        # fields live in the thread store and the thread readout withholds
        # them, so this must never become the default path.
        logger.warning(
            "turn_forensics: include_raw=true; bypassing read-time redaction "
            "for turns=%r (user=%r)",
            params.turns,
            ctx.user_slug,
        )

    reader = ForensicReader(
        data_dir,
        thread_dir=_resolve_thread_dir(data_dir, ctx),
        user_slug=str(ctx.user_slug or ""),
        # Redaction is the deployment's contract, resolved by the reader from
        # the shared environment contract. ``include_raw`` is the only way to
        # opt out, and only for this call.
        resolve_redaction_from_env=not params.include_raw,
    )

    bundle = reader.bundle(
        turns=params.turns,
        max_records=params.max_records,
        focus=params.focus,
    )

    if params.include_raw:
        bundle.notes.append(
            "include_raw was requested: read-time redaction is bypassed for this "
            "bundle. Raw prompt and reasoning text may appear below, so treat it "
            "as private and do not repeat it verbatim."
        )

    rendered = render_bundle(bundle)
    return OperationResult(
        output_text=rendered,
        output_json=bundle.model_dump(mode="json"),
        extra={
            "turn_count": len(bundle.turns),
            "degraded": bundle.is_degraded,
            "missing_sources": list(bundle.sources.missing),
            "notes": list(bundle.notes),
        },
    )
