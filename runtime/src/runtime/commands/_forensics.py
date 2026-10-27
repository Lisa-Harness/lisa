"""``/forensics`` — the operator's view of Lisa's recorded turn evidence.

The command is a thin adapter. Reading lives in :mod:`sdk.forensics` and
formatting lives in :func:`sdk.forensics.render_bundle`, which is the same
formatter the ``turn_forensics`` operation returns to the model. That is the
whole point of the command: an operator asking "what did that turn actually
do?" and the model asking the same question must not be shown two different
stories about one turn.

Dependency direction: ``_forensics`` → ``_log``, ``sdk.forensics``,
``runtime.thread``. No imports from sibling ``commands._*`` modules other
than ``_log``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sdk.forensics import ForensicReader, render_bundle

from runtime.commands._log import _append_thread_entry, _append_turn_log, _set_turn_context
from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)

__all__ = ["handle_forensics_command", "render_forensics"]


def render_forensics(runtime: HostRuntime, arg: str) -> str:
    """Render the forensic bundle for ``arg``.

    ``arg`` is the stripped text after ``/forensics``: empty for the most
    recent turn, ``last`` for the same thing explicitly, ``12`` for one turn,
    or ``3-7`` for an inclusive range. An unparseable selection is reported by
    the bundle's own notes rather than rejected here: the reader already
    distinguishes "bad range" from "no evidence", and duplicating that logic
    would let the two drift.
    """
    from runtime.thread import get_thread_redaction_policy

    reader = ForensicReader(
        runtime.paths.data_dir,
        thread_dir=runtime.paths.thread_dir,
        user_slug=runtime.config.user_slug,
        # The host resolved the deployment's policy at startup; pass it
        # explicitly so the operator view is redacted exactly like every other
        # projection, regardless of ambient environment at command time.
        redaction=get_thread_redaction_policy(),
    )
    selection = arg.strip() or "last"
    return render_bundle(reader.bundle(turns=selection))


def handle_forensics_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/forensics [last | turn | start-end]``."""
    arg = stripped[len("/forensics") :].strip()
    response_text = render_forensics(runtime, arg)

    _set_turn_context(
        runtime,
        timestamp_utc=datetime.now(UTC).isoformat(),
        raw_message=message,
        mode_id="forensics",
        handoff="forensics command",
    )
    _append_turn_log(
        context={
            "mode_id": "forensics",
            "handoff": "forensics command",
            "selection": arg or "last",
        },
        response_text=response_text,
    )
    _append_thread_entry(
        runtime,
        mode_id="forensics",
        query=message,
        response_text=response_text,
    )
    runtime.session.current_mode_id = "forensics"
    runtime.session.current_stage_id = "-"
    runtime.session.next_model_id = "-"
    return response_text
