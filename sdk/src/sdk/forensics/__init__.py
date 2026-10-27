"""Forensic reading of Lisa's own turn evidence.

Answers "what actually happened, and why?" from durable records rather than
from prose: the thread store (what was said and routed), the decision ledger
(why the loop chose each action), the dispatch ledger (what ran, how long, and
what failed), and the cost log (what it spent) — joined on ``turn_index``.

Why this lives in the SDK
-------------------------
The host (``runtime.introspect``) and the introspect pack both need the same
parsing, but packs may not import ``runtime``. Keeping the reader here means
one implementation with two consumers, instead of the two independent log
parsers that previously drifted far enough apart for one to read a field the
other never wrote.

Public surface
--------------
- :class:`ForensicReader` — compose one bounded bundle
- :func:`render_bundle` — the single Markdown formatter
- :class:`ForensicBundle` and the evidence records it holds

Reading is strictly read-only and path-derived from an injected data root, so
a caller cannot address evidence outside it.
"""

from __future__ import annotations

from sdk.forensics._join import (
    build_bundle,
    cost_from_record,
    decision_from_record,
    dispatch_from_record,
    turn_from_entry,
)
from sdk.forensics._logrec import (
    DEFAULT_BACKUP_SEGMENTS,
    DEFAULT_MAX_BYTES,
    filter_window,
    in_window,
    iter_jsonl_tail,
    parse_ts,
    record_ts,
    rotated_segments,
)
from sdk.forensics._models import (
    BundleSources,
    CostRecord,
    DecisionRecord,
    DispatchRecord,
    ForensicBundle,
    TurnEvidence,
)
from sdk.forensics.reader import (
    DEFAULT_LOG_ERROR_LIMIT,
    DEFAULT_MAX_TURNS,
    DEFAULT_WINDOW_SECONDS,
    ForensicReader,
)
from sdk.forensics.render import DETAILED_TURN_LIMIT, render_bundle, render_turn_line

__all__ = [
    "DEFAULT_BACKUP_SEGMENTS",
    "DEFAULT_LOG_ERROR_LIMIT",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_WINDOW_SECONDS",
    "DETAILED_TURN_LIMIT",
    "BundleSources",
    "CostRecord",
    "DecisionRecord",
    "DispatchRecord",
    "ForensicBundle",
    "ForensicReader",
    "TurnEvidence",
    "build_bundle",
    "cost_from_record",
    "decision_from_record",
    "dispatch_from_record",
    "filter_window",
    "in_window",
    "iter_jsonl_tail",
    "parse_ts",
    "record_ts",
    "render_bundle",
    "render_turn_line",
    "rotated_segments",
    "turn_from_entry",
]
