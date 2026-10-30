# Logging and observability

Lisa uses stdlib logging. The shared
[logging setup](../sdk/src/sdk/logging_setup.py) supplies JSON formatting,
rate limiting, configuration, and structured-event emission.
Hosts own presentation; first-party MCP adapters use their shared CLI setup.

## Operator configuration

- `LISA_LOG_LEVEL` sets runtime verbosity; INFO is the effective default.
- `LISA_LOG_DEBUG` is a comma-separated list of logger names forced to DEBUG.
- Runtime events normally live in `<data-dir>/logs/runtime.jsonl`;
  token usage lives in `<data-dir>/logs/token-usage.jsonl`.
- Turn decisions and dispatches live in `<data-dir>/logs/decisions.jsonl`
  (see [decision telemetry](#decision-telemetry)).
- First-party servers accept `--log-level` and `--data-dir`; their parser
  uses `LOG_LEVEL` when no log-level argument is supplied.

See [configuration](CONFIGURATION.md) for path resolution and
[troubleshooting](TROUBLESHOOTING.md#useful-diagnostics) for investigation.
DEBUG output, threads, and `/debug` request snapshots can be sensitive.

## Contributor standards

Use a module logger and lazy formatting:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("routing: selected mode=%s reason=%s", mode_id, reason)
```

Follow the surrounding module's logger name; do not rename existing loggers
just to standardize spelling.

| Level | Use |
| --- | --- |
| INFO | Decisions, lifecycle transitions, authorization outcomes, and meaningful no-ops with reasons |
| DEBUG | Internal per-item/matching traces |
| WARNING | Degraded behavior, caught failures, unavailable optional resources |
| ERROR | Fatal failures and unrecoverable boundary errors |

When handling an exception, include the diagnostic context and traceback
with `exc_info=True` or `logger.exception`. Catch a specific exception and
either handle explicitly or re-raise. Never return a success-shaped result
after silently discarding an unexpected error.

```python
try:
    load_required_configuration()
except ValueError:
    logger.exception("configuration: invalid required input")
    raise
```

Expected control-flow exceptions are different from operational failures;
document any intentional non-reporting boundary.

## Structured events

Use namespaced events and machine-readable fields, not prose parsing:

```python
from sdk.logging_setup import emit_structured_event

emit_structured_event(
    logger,
    level=logging.INFO,
    event="routing.hint_injected",
    message="routing: hint selected",
    mode_id=mode_id,
    targets=targets,
)
```

The turn loop publishes decision telemetry through the event bus, including
mode, iteration, lifecycle action/reason, calls, and receipt counts.
Provider/usage records and exact request snapshots serve different purposes:
do not infer what the model saw solely from a routing event or final answer.
See [context lifecycle](CONTEXT_LIFECYCLE.md) and [token accounting](TOKEN_BUDGET.md).

## Decision telemetry

Two event-bus topics carry the forensic record of a turn, and both are
persisted by [`DecisionTelemetryHandler`](../kernel/src/kernel/decision_telemetry.py)
to `<data-dir>/logs/decisions.jsonl` — one JSON line per record, tagged with
its `topic`:

| Topic | Publisher | Per | Carries |
| --- | --- | --- | --- |
| `turn.decision` | [`_resolve_lifecycle`](../kernel/src/kernel/turn_runner/__init__.py) | iteration | `mode_id`, `turn_index`, `user_slug`, `iteration`, `turns_in_stage`, `action_type`, `action_reason`, `total_tool_calls`, `write_receipts`, `read_receipts` |
| `tool.telemetry` | [tools](../kernel/src/kernel/tools/registry.py) / [operations](../kernel/src/kernel/operations/registry.py) registry | dispatch | `target_id`, `target_type`, `status`, `duration_ms`, `error_message`, `turn_index`, `user_slug` |
| `model.request` | [host adapter](../kernel/src/kernel/seams/host_adapter.py) | off-turn model call | `operation_id`, `request_ref`, `turn_index`, `user_slug`, `thread_store_slug` |

Why a separate file rather than `runtime.jsonl`:

- The root file handler installs `RateLimitFilter`, which suppresses repeated
  identical records keyed on `(name, level, message)`. A mode that resolves the
  same lifecycle action on consecutive iterations would have those decisions
  dropped. The appender writes directly instead of going through stdlib
  logging.
- `decisions.jsonl` is segmented independently: the active file is sealed to
  `decisions.jsonl.<n>` once it reaches the size threshold, and sealed
  segments are never deleted, so verbose runtime logging cannot evict the
  decision ledger and the ledger cannot consume the log's rotation budget.

Operation completions also publish `model.request`: one row per model call
an operation dispatch makes outside the turn loop (memory mining,
consolidation, distills), carrying the content-addressed `request_ref` for
the archived request. The turn loop's own calls are already archived on the
thread entry's `turn_input.calls`, so this row covers precisely the calls
the thread entry never sees; its token usage rides the same
`lisa.token_usage` topic with `call_site: "operation"`. `turn_index` is
`null` when the dispatch belongs to no thread turn.

## Concurrent-ledger safety

Both telemetry ledgers are shared across host processes: overlapping Lisa
turns on one data dir append to the same `decisions.jsonl` and
`token-usage.jsonl`. The runtime serializes those appends (and the
decisions seal decision) under sidecar advisory locks
(`<file>.lock`), so every process's lines land whole and no process's
sealed segment can overwrite another's. The kernel handlers stay
runtime-neutral: they accept a `lock_factory` and the host injects the
lock ([`HostServices.build`](../runtime/src/runtime/services.py)). The
locks are best-effort with a short timeout — a stuck holder drops one
telemetry line rather than stalling a live turn.

`turn_index` is the join key. It is reserved once at turn start
(`runtime.thread.reserved_turn`), carried on `TurnIdentity` into every
operation/tool context, and matched against the thread entry, so a decision
row, a dispatch row, a token row, and a thread entry for the same turn address
each other. When overlapping turns commit out of start order the entry's `turn`
is its log position and the index the turn ran under is kept as
`provisional_turn`; the forensics reader joins on that.
Without it, per-turn cost is attributable only to a *mode*, not a turn.

`turn_index` is `null` for work that belongs to no thread turn — off-turn slice
work, or a host with no thread. That is distinct from `0`, the thread's first
real turn; a reader must not treat a missing index as turn 0 or it will
attribute background work to the first turn. Use
[`resolve_turn_index`](../kernel/src/kernel/models.py) to normalize raw values.

To read a specific turn's evidence — the thread entry, its decisions, its
dispatches, and its cost, joined — use `/forensics [last | turn | start-end]`
for the operator view, or the `turn_forensics` operation for the model view.
Both render through one formatter. See [turn forensics](FORENSICS.md).

## Decision telemetry versus spans

The event bus and OpenTelemetry are separate channels, and neither substitutes
for the other.

- **Event bus** carries decision telemetry published by the turn loop: mode,
  iteration, lifecycle action and reason, calls, and receipt counts. The
  durable subscriber persists it to `decisions.jsonl` (above). This is what
  answers "why did the turn do that?" — the logger lines are a separate,
  rate-limited diagnostic stream.
- **OpenTelemetry spans** cover dispatch timing and nesting through
  [telemetry helpers](../kernel/src/kernel/telemetry.py). Every tool dispatch
  runs under a `tool.execute` span from
  [tools/registry.py](../kernel/src/kernel/tools/registry.py) and every operation
  dispatch under an `operation.execute` span from
  [operations/registry.py](../kernel/src/kernel/operations/registry.py), both
  built with `get_tracer` from the global tracer provider.

Span emission is implemented. Span export hierarchy is not established: nothing
in this repository shows whether a host's configured exporter receives these
spans as a coherent parent/child tree. Proving it means capturing a host export
and comparing it against the spans already emitted here — it does not mean
adding instrumentation, and a second telemetry stack is not the answer.

## Safe diagnostics

Prefer identifiers, counts, hashes, status codes, and bounded reasons.
Do not log credentials, authorization headers, raw HTTP bodies, full user
prompts, embeddings, memory bodies, or serialized contexts as routine events.
Remove credentials/query secrets from URLs as well as headers.

Persisted threads and deliberate debug snapshots can contain exact content;
this logging standard does not claim those stores are sanitized. Restrict
access, define retention, and sanitize before sharing. Read-time thread
redaction is not deletion of the underlying evidence.

## Verification

Use focused tests asserting diagnostic fields and error paths, plus the
[existing quality gates](TESTING.md). This repository does **not** currently
ship the formerly advertised `scripts/audit_logging.py` or an automatic
logging fix/allowlist tool. Do not mistake contributor standards for an
enforced repository-wide logging audit.
