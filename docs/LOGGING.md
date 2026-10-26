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

## Decision telemetry versus spans

The event bus and OpenTelemetry are separate channels, and neither substitutes
for the other.

- **Event bus** carries decision telemetry published by the turn loop: mode,
  iteration, lifecycle action and reason, calls, and receipt counts, as
  structured events on the logger. This is what answers "why did the turn do
  that?"
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
