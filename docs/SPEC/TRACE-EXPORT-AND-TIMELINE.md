# Trace Export and Timeline Specification

> **Status:** Proposed.
> **Order:** 4 of 4 in the observability and replay program.
> **Depends on:** [evidence envelope](./EVIDENCE-ENVELOPE.md) (`turn_id`,
> lineage, reserved `trace_id`/`span_id`). Can start in parallel with the
> [model call archive](./MODEL-CALL-ARCHIVE.md) and
> [turn replay](./TURN-REPLAY.md). Call detail in the timeline (§6) uses the
> archive once it exists.

## Goal

Build one span tree for each turn, with two consumers:

- **External:** standard OpenTelemetry export, so Lisa traces and metrics show
  up in the observability stacks operators already run.
- **Internal:** a timeline view and a live evidence stream that work with no
  exporter configured.

Both consumers link to the durable ledgers through `trace_id`. The ledgers
remain the record; spans are a view of it.

## Current state

- Only two spans exist: `tool.execute` in [tools/registry.py](../../kernel/src/kernel/tools/registry.py)
  and `operation.execute` in [operations/registry.py](../../kernel/src/kernel/operations/registry.py),
  both from the global tracer through [telemetry.py](../../kernel/src/kernel/telemetry.py).
- No turn, iteration, model call, routing, context assembly, approval, or
  child-turn span exists, so the existing spans have no parent.
- Nothing installs a tracer provider, exporter, or meter provider.
  [LOGGING.md](../LOGGING.md#decision-telemetry-versus-spans) records that
  export hierarchy is not established.
- `kernel` declares `opentelemetry-sdk` as a dependency, but no source module
  imports it. The kernel needs only the API.
- No trace context crosses a process boundary: none goes to MCP servers, none
  is accepted from hosts over the [host protocol](./HOST_PROTOCOL.md), and
  none goes to providers.
- Forensics is pull-only text. [Behavioral signals](../../sdk/src/sdk/forensics/signals.py)
  look back only 12 turns because they run inside the post-turn slice budget.

## 1. Span tree

| Span | Parent | Key attributes |
| --- | --- | --- |
| `lisa.turn` | Host span if propagated, else root | `lisa.turn_index`, `lisa.turn_id`, `lisa.mode_id`, `lisa.run_ref`, outcome, `block_origin` |
| `lisa.routing` | `lisa.turn` | Selected mode, forced-action kind |
| `lisa.context_assembly` | `lisa.turn` | Item and byte counts per lane, budget |
| `lisa.iteration` | `lisa.turn` | `lisa.iteration`, lifecycle action and reason |
| Model call (`chat <model>`) | `lisa.iteration` | OpenTelemetry GenAI conventions: operation, provider, request and response model, token usage, finish reasons; plus `lisa.call_id`, `lisa.attempt` |
| `tool.execute`, `operation.execute` | `lisa.iteration` | Existing spans, now parented; add `lisa.dispatch_id`, `gen_ai.tool.call.id` |
| `lisa.approval.wait` | The dispatch span | `request_id`, decision, `decision_source` |
| `lisa.turn` (child) | The fork or delegate `operation.execute` span | `lisa.parent_turn_id` |

Spill, compaction, and fold events are span events on `lisa.iteration`, not
separate spans.

**Content is never placed on spans.** The GenAI conventions allow opt-in
prompt and response capture; Lisa does not use it. Spans carry
`request_ref`/`response_ref` and the archive holds the content, behind
redaction. Span attributes follow the same allowlist posture as the ledgers.

## 2. Ledger correlation

When a valid span is active, publishers fill the envelope's `trace_id` and
`span_id`. Thread entries record the `lisa.turn` span's `trace_id`. A trace in
an external backend and a forensic bundle therefore address each other in
both directions.

## 3. Export configuration

- `kernel` depends on `opentelemetry-api` only. The SDK and the OTLP exporter
  move to an optional `runtime` extra.
- The runtime installs a tracer provider and meter provider only when export
  is configured, using the standard `OTEL_*` environment variables
  (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`,
  `OTEL_TRACES_SAMPLER`, …). It adds no Lisa-specific duplicates.
- With nothing configured, the API's no-op provider stays in place and
  instrumentation cost is the no-op path.
- Resource attributes: `service.name` (default `lisa`), the service version
  from the run manifest, and the host type. No user identifiers; user-scoped
  attributes on spans use the token ledger's pseudonymous thread identity.

### Metrics

| Instrument | Kind | Dimensions |
| --- | --- | --- |
| `lisa.turns` | Counter | Mode, outcome |
| `lisa.turn.duration` | Histogram | Mode, outcome |
| `gen_ai.client.operation.duration` | Histogram | Provider, model, attempt |
| `gen_ai.client.token.usage` | Histogram | Provider, model, token type |
| `lisa.cost` | Counter | Provider, model, call site |
| `lisa.dispatches` | Counter | Target type, status |
| `lisa.telemetry.dropped` | Counter | Source |

`lisa.telemetry.dropped` increments wherever
[`schedule_telemetry_publish`](../../kernel/src/kernel/telemetry.py) logs a
lost row today, so evidence loss becomes alertable as well as countable.

## 4. Context propagation

| Boundary | Carrier | Default |
| --- | --- | --- |
| Host → Lisa | Optional W3C `traceparent` and `tracestate` on the [host protocol](./HOST_PROTOCOL.md#62-operation-envelope) `turn.start` and `approval.decide` params | On when present |
| Lisa → MCP server | `params._meta.traceparent` on tool calls, following the OpenTelemetry MCP conventions | On for first-party servers; per-server opt-in for third-party ones |
| Lisa → model provider | `traceparent` HTTP header | Off; provider-side tracing is outside Lisa's trust boundary |

First-party MCP servers (`mcp_server`, and `lisa-go` through the
[Go connector contract](../GO_CONNECTOR_SDK.md)) extract the context and parent
their own spans to it, so a dispatch's tree continues across the process
boundary. Update the host protocol spec when it adds the optional fields.

## 5. Delegation tree

Child turns' `lisa.turn` spans parent to the spawning dispatch's span. The
forensic reader renders the same tree from the envelope lineage fields
(`parent_turn_id`, `parent_dispatch_id`), so the tree is available with no
exporter. The two views must agree: a test builds both for one fork and
compares their shapes.

## 6. Timeline and live stream

### Timeline view

`/forensics <selection> --timeline` (and `turn_forensics(timeline=true)`)
renders the turn as a tree with wall-clock offsets. It is built from ledger
rows alone, so it works offline and without export:

```text
turn 912  default  0.0s → 41.2s  complete
├─ routing                   0.0s   12 ms  mode=default
├─ iteration 1               0.1s
│  ├─ chat model-a  primary  0.1s  8.4 s  tool_calls  12.1k/310 tok  $0.02
│  ├─ tool.execute bash_run  8.6s  2.1 s  ok
│  └─ approval.wait          8.6s  1.8 s  approved (operator)
└─ iteration 2               10.7s
   └─ chat model-a  primary  10.7s 30.5 s  stop        14.0k/1.2k tok $0.04
```

Per-call rows (finish reason, time to first token, outcome) come from the
[model call archive](./MODEL-CALL-ARCHIVE.md) when it is present, and are
marked unavailable otherwise.

### Live evidence stream

The host protocol gains an `evidence` event type: each ledger row, as it is
appended, projected through the same allowlist and scoped to the session's
user and thread. Hosts and UIs subscribe through the existing `turn.event`
notification and its `sequence` gap detection. No second schema exists; the
event payload is the envelope row.

### Wider signals

`/forensics --signals [--window N]` runs [`compute_behavioral_signals`](../../sdk/src/sdk/forensics/signals.py)
on demand over a caller-chosen window. The ambient post-turn scan keeps its
12-turn, budget-bound window.

## 7. Export and retention

### Evidence bundles

`lisa evidence export <selection> [--raw]` writes one archive containing:

- the thread range;
- the scoped ledger rows;
- every referenced object (requests, responses, run manifests, spilled
  results).

Redaction applies unless `--raw` is given, and a raw export is logged at
WARNING. A bundle is self-contained input to `/forensics` and to
[turn replay](./TURN-REPLAY.md), so a user can attach one to a bug report and
a maintainer can replay it.

### Retention

- Sealed ledger segments are never deleted today. Add an operator retention
  policy per data root (maximum age or size) for ledgers and archived objects.
- Retention deletes only whole sealed segments and unreferenced objects.
  Every deletion appends a `retention.applied` row naming what was removed, so
  later reads explain the gap instead of reporting silent loss.
- No retention policy is applied by default.

## Acceptance

- An in-memory exporter test runs a turn with a tool call, an approval, and a
  delegate child. It asserts one connected trace with the parent/child shape
  in §1, which closes the LOGGING.md "export hierarchy not established" gap.
- Ledger rows for that turn carry the trace's `trace_id`, and each row's
  `span_id` names an exported span.
- A first-party MCP server's span shares the dispatch's trace.
- With no `OTEL_*` configuration, no provider is installed and the
  benchmarked overhead of a no-op turn stays within noise.
- `--timeline` renders the example shape from ledger rows on a host with
  export disabled.
- `lisa evidence export` followed by `/forensics` on the bundle reproduces the
  source bundle's rendering.
- Applying retention appends `retention.applied`, and the forensic reader
  reports the retained-away range.

## Open questions

- Sampling: export decisions are head-sampled by default. Should a blocked or
  failed turn force export (tail-style) when a sampler drops it? The ledgers
  are unsampled either way.
- Does `evidence` need a server-side filter (topic allowlist) in the host
  protocol, or do hosts filter client-side?
