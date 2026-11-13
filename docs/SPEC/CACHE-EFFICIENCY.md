# Cache Efficiency and Measurement Specification

> **Status:** Proposed.
> **Scope:** Cache telemetry, history/compaction policy, mode handoffs, and
> before/after reporting.
> **Depends on:** [model data layers](./MODEL-DATA-LAYERS.md).
> **Baseline:** Local report generated on 2026-11-12. It is not committed
> because it contains user-specific usage and cost telemetry.

## Goal

Reduce uncached prompt work and total cost per successful turn without losing
tool evidence, widening tool access, weakening model quality, or treating
unknown prices/cache behavior as measured facts.

The [model data layers specification](./MODEL-DATA-LAYERS.md) is authoritative
for provider-published metadata, measured overlays, operator settings, and
field provenance. This document does not redefine those layers.

## Roadmap

The work tracks four remaining problems. They are ordered by priority, not by
a strict dependency chain: measurement ships first, the baseline replay and
prefix oracle start in parallel with it, and the mode-handoff design starts
as read-only investigation. Policy and routing changes and the final
before/after comparison wait for qualified measurements.

### Revision notes

Corrections to earlier framing, recorded so the roadmap does not propagate
them:

- The §4 telemetry substrate has landed (schema v3 durable events, the
  unknown-is-not-zero price sentinel cutover, the privacy allowlist), but §4
  is not complete: events do not yet stamp effective model-data source and
  version, the recorded upstream route is a classification rather than
  observed provider identity, and no explicit cache-break event exists.
- Cache aggregation is not absent: `_render.py` aggregates lifetime usage
  and `introspect.py` emits a trailing 24-hour summary that renders missing
  values as zero. Missing is cache-quality aggregation with honest unknown
  handling.
- Fold economics are not absent: the §5 positive-savings rule exists. What
  is missing is measured-rate input, prefix-invalidation coordination, and
  router and working-set use of qualified economics.
- Data-layer precedence composes per field (operator over measured over
  provider over host default, with field-specific authority); it is not one
  global chain.
- Section numbering below is historical and unchanged; this roadmap is
  deliberately unnumbered.

### Workstream 1 — Honest measurement

Problem: per-call telemetry is collected, but cache hit behavior is never
aggregated with honest unknown handling, so no qualified measurement exists
to drive decisions.

Remaining work:

- Aggregate the existing token-usage ledger on read paths only, off-turn or
  on report reads, never in-turn. Prefer rotation-safe incremental
  ingestion; if that cannot be implemented honestly, use a bounded full scan
  and say so.
- Compute a cache request-hit proxy and cached-token share as separate
  metrics; a missing cache counter stays unknown and is never counted as
  zero.
- Partition by provider, model, route, and cache policy where reliably
  known; label unknown dimensions and never pool them as comparable. Do not
  claim prompt-prefix attribution; no prefix identity exists yet.
- Record gates, confidence, windows, and expiry with every snapshot. The
  proposed gates (at least 30 samples, Wilson 95% half-width at most 0.15,
  14-day rolling window, 7-day snapshot expiry) are an engineering proposal,
  not normative facts; the implementation records the method it actually
  used.
- Write snapshots to the measured layer only when provider, model, route,
  policy, and prefix identity are all known; otherwise mark them
  unqualified and never let optimization read them.
- Extend the `/tokens` surface with a focused cache report: cached-token
  share, cost, unknown-cost coverage, and coverage/attribution for legacy
  rows. Fix or explicitly mark legacy aggregates that imply missing equals
  zero.
- Stamp effective model-data source and version on every emitted event (the
  remaining model-data-layers digest item).

### Workstream 2 — Economics-aware policies

Problem: folds, working-set sizing, and routing do not yet consult
qualified measured economics, so §5 decisions rest on published or
configured rates alone.

Remaining work:

- Feed quality-gated measured read/write rates into supersession fold
  economics, keeping the conservative future-request estimate and the
  iteration-budget cap; unavailable rates fall back to the configured
  conservative policy.
- Track cache-prefix state and its invalidation cause so folds happen at
  already-lost prefixes and never double-rewrite a suffix that compaction
  also rewrites.
- Tune `working_set_tokens` from measured economics within the hard floor
  and ceiling, and report the effective setting and its source.
- Route on observed end-to-end cost for successful comparable tasks with
  quality and capability constraints unchanged; unknown or sparse slices
  preserve the existing route, and every change stays explainable.
- No policy or routing change reads an unqualified snapshot.

### Workstream 3 — Safe handoffs

Problem: mode handoffs rebuild the prompt and the real cache-loss points
are assumed rather than identified; never assume every handoff loses cache.

Remaining work:

- Deliver a prompt-layout design before any code: enumerate which prompt
  components are invariant, variable, or provider-specific for each handoff
  type and state the expected cache effect of each.
- Keep invariant base instructions ahead of mode-specific content where the
  provider can cache the unchanged prefix independently.
- Never union tool schemas or widen authorization across modes to stabilize
  a prefix; any ordering change ships with an equivalence argument and
  tests.
- If an optimization cannot preserve access boundaries and correct
  instructions, accept the cache miss.

### Workstream 4 — Evidence and validation

Problem: nothing yet proves any change reduced uncached prompt work or
cost, and the 2026-11-12 baseline has not been replayed through new
readers.

Remaining work:

- Build the deterministic cache-prefix e2e oracle with per-iteration prefix
  digests and the §7 assertions, including no duplicate fold/compaction
  rewrite and expired-cache-is-not-a-changed-prefix.
- Replay the local 2026-11-12 baseline through the new aggregation readers
  and add regression tests over legacy, incomplete, and malformed rows.
- Run a minimum 14-day before/after comparison that publishes sample
  counts, coverage, and unknown share beside every result; an underpowered
  or unattributed comparison is reported as inconclusive, not as savings.

## 4. Collect and use telemetry

### Durable event contract

Every model call that contributes to a turn must be attributable by
`thread_store_id` and `turn_index`; off-turn work has an explicit null turn.
Persist the existing event fields needed for analysis:

- event time, provider, model ID, mode/stage, call site, and stable request ID;
- prompt, cached-read, cache-creation, completion, reasoning, and total tokens;
- canonical estimated cost and provider-reported cost as separate values;
- latency/duration and whether each value is measured or unavailable;
- effective model-data source/version and, when exposed, OpenRouter upstream
  provider identity.

Do not store prompts, tool arguments, or generated content in usage telemetry.
Use local, data-root-scoped pseudonymous thread identity; do not use workspace
paths or account credentials as aggregation keys.

The durable JSONL serializer, readers, `/tokens` view, and report must preserve
the fields. Missing means unknown; it must not be coerced into a measured zero.
Legacy rows remain readable but are marked unattributed or incomplete.

### Provider instrumentation

- Keep provider-reported cache counters distinct from Lisa's token estimates.
- Preserve `cache_creation_tokens` and provider-reported cost when the gateway
  supplies them; preserve canonical `cost_usd` separately.
- For providers without published prices, mark canonical cost as unavailable
  unless an explicit operator price or a confidence-qualified measured overlay
  exists. A zero value is not evidence that inference was free.
- Record cache TTL and minimum cacheable prefix only when documented or
  configured for the selected provider/model. Do not infer either from a
  single request gap.
- Keep cache behavior partitioned by provider, model, upstream route when
  available, effective prompt-prefix identity, and cache policy. Do not pool
  incompatible providers or model variants.

### Measured model overlay

Write measurements to the measured layer defined in the prerequisite
specification. Each metric records:

- model/provider/route identity and metric name;
- estimate, unit, sample count, observation start/end, and method/version;
- confidence/quality status and expiry time.

Use minimum sample requirements and confidence intervals. Small samples,
provider changes, or stale data keep a value unknown; they do not trigger
aggressive routing or history settings. Measurement writes are automatic and
never overwrite operator settings or provider-published facts.

## 5. Tune folds, compaction, working set, and routing

### Supersession folds

- Preserve the current positive-savings rule as the starting policy:
  estimate rewrite cost from the effective cache read/write rates, bytes
  removed, cached suffix rewritten, and expected future requests.
- Use published rates first, then valid measured rates, then an explicitly
  configured policy. Do not silently treat an unavailable rate as observed.
- Estimate remaining requests conservatively and cap by the turn's remaining
  iteration budget. A fold that is not expected to pay for its rewrite waits
  unless the cache has already been lost.
- Record the chosen boundary, estimated removed tokens, rewritten suffix,
  assumed price shares, expected future requests, and decision reason. Do not
  log conversation content.

### One rewrite at an already-lost prefix

- Track cache-prefix state and the reason it became invalid (provider/model
  change, prompt/tool schema change, known TTL expiry, compaction, or other
  explicit cause).
- Before soft compaction, fold pending stale results that lie wholly after a
  prefix already known to be invalid. Those edits do not create an additional
  cache miss.
- Coordinate fold and compaction so one iteration does not separately rewrite
  the same suffix. Compaction must see the post-fold prompt size.
- A time gap alone is not proof of cache expiry. Use provider-documented TTL or
  validated per-provider telemetry, accounting for an observed cache hit as a
  counterexample.
- Keep the thread store lossless; only the model-visible projection is folded.

### Working set

- Tune `working_set_tokens` from the effective context window, measured cached
  read/write economics, and the expected value of retaining recent evidence.
- Preserve a hard floor for active, unsuperseded evidence and a hard ceiling
  below the usable context budget.
- Unknown or low-confidence prices use the configured conservative policy,
  not a fabricated model-specific value.
- Report the effective setting and its source; test that increasing/decreasing
  it does not omit current-turn evidence or alter history references.

### Router

- Optimize observed end-to-end cost and latency for successful comparable
  tasks, not price tier labels alone.
- Include provider cost when available and report estimated cost separately.
- Keep quality/success and capability requirements as constraints; cost cannot
  route to a model that lacks required capabilities.
- Require minimum samples per model/task slice. Unknown or sparse measurements
  preserve the existing route rather than creating a fallback to an
  unverified model.
- Make changes explainable: show metric sample count, observation period,
  measured source, and the winning tradeoff.

## 6. Preserve cache across mode handoffs

This is a separate prompt-layout design; do not implement it as a side effect
of the model-data or telemetry work.

Requirements:

- Identify the provider-visible cacheable prefix and its invalidation rules for
  each supported gateway.
- Keep truly invariant base instructions stable across mode transitions.
- Place mode-specific instructions and state after that prefix only when the
  provider can cache the unchanged prefix independently.
- Do not union tool schemas or capabilities across modes merely to stabilize
  a prefix. A mode must retain exactly its current tool authorization and
  tool-call contract.
- Preserve mode semantics, system-priority rules, policy injection, approvals,
  and tool schema ordering. Any changed ordering must have an explicit
  equivalence argument and tests.
- Account for model/provider changes, cache breakpoints, dynamic memory,
  context compaction, and thread rehydration.
- If an optimization cannot preserve access boundaries and correct
  instructions, accept the cache miss.

Deliver a prompt-layout design and an e2e prefix trace before implementation.
The design must enumerate which prompt components are invariant, variable, or
provider-specific and state the expected cache effect for each handoff type.

## 7. Baseline, regression tests, and ongoing report

### Baseline snapshot

Use the local 2026-11-12 baseline report as the historical reference. It
contains aggregate counts only and remains outside the repository because the
usage and cost aggregates are user-specific. Do not commit raw usage logs or
account-level reports.

The baseline separates:

- per-request metrics for the whole observation window;
- per-turn metrics only where a stable turn identity is present;
- provider-reported charges, canonical estimates, and unknown/unpriced calls;
- observed request-gap cohorts from causal cache-break events.

Do not compare incomplete historical rows with complete post-change rows
without showing coverage and attribution rates.

### Deterministic cache-prefix e2e oracle

Add a scripted e2e scenario that captures the canonical provider request
shape—not prompt text in its report—and emits a digest for the cacheable prefix
on each model iteration. For every transition it records:

- prior/current prefix digest and model/provider;
- cache TTL state if the scenario configured one;
- explicit break reason;
- whether the history fold or compaction changed that prefix.

Assertions:

- Appending tool results after an unchanged prefix does not change its digest.
- A fold or compaction changes the digest only at the expected boundary.
- A fold and compaction at the same boundary cause no duplicate rewrite.
- Mode handoff does not widen tool availability or reorder required schemas.
- A known expired cache is distinguished from a changed prefix.

### Operational report

Extend the existing `/tokens` reporting surface or add a focused cache report.
It must show per model/provider and per comparable task slice:

- turns and model-call counts, plus telemetry coverage/attribution;
- prompt, cached-read, cache-creation, and non-cache-read tokens;
- cached-token share, provider cost, canonical estimated cost, and unknown
  cost coverage;
- latency percentiles only for adequately populated measured values;
- cache breaks by explicit cause, not just inferred from a long gap;
- measured overlay age, sample count, and confidence.

Never label missing price, latency, or cache data as zero.

### Success criteria

1. Replay the baseline window from raw records and reproduce the aggregates in
   the baseline report within rounding; tests exercise old/incomplete rows.
2. E2E prefix-break counts match the declared cause and prove no redundant
   fold-plus-compaction rewrite.
3. After rollout, compare a minimum 14-day stable window against a matched
   baseline by model, provider, and workload. Report both per-turn medians and
   totals with sample counts.
4. The primary outcome is lower non-cache-read prompt tokens and lower
   canonical cost per successful comparable turn. Provider-reported charges
   remain a separate validation metric.
5. Do not accept a cost improvement if success/quality evaluations regress or
   mode/tool authorization changes.
6. Publish telemetry coverage and unknown/unpriced share beside every result.
   If the comparison is underpowered or lacks cost attribution, report that
   result as inconclusive instead of claiming savings.
