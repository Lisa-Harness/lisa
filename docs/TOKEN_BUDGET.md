# Token Budget Baseline & Working Limit

Captured 2026-08-01. This document records the evidence, calculations, and
reasoning behind the recommended per-turn and daily token working limits for
this workspace. It exists so the numbers are reproducible — if the spend
profile changes materially, re-derive from `artifacts/token-usage.jsonl`
rather than guessing.

## Baseline: what this workspace actually burned (2026-08-01)

Full-session aggregate from `artifacts/token-usage.jsonl` (895 model calls):

| Model | Calls | Tokens |
| --- | --- | --- |
| deepseek-v4-flash:0731-cloud | 343 | 7,295,218 |
| glm-5.2:cloud | 264 | 7,627,042 |
| deepseek-v4-pro:cloud | 225 | 3,075,661 |
| gpt-5.4 | 59 | 1,254,539 |
| gpt-5.4-mini | 4 | 12,438 |
| **Total** | **895** | **19,264,898** |

### The split is the whole story

- **Prompt tokens: 18,718,001 (97%)**
- **Completion tokens: 546,897 (3%)**
- **Cached tokens: 35,840 (~0%)**

97% of everything spent today was the **context being re-sent**, not model
output. The largest single call ever was **107,247 total tokens** (103,151
prompt + 4,096 completion) — that is the effective per-call context ceiling.

### Hourly shape (UTC) — heavy bursts, not steady flow

| Hour | Tokens |
| --- | --- |
| 10:00 | 13,402 |
| 11:00 | 607,920 |
| 12:00 | 586,159 |
| 13:00 | 3,969,784 |
| 15:00 | 316,036 |
| 16:00 | 188,431 |
| 17:00 | 4,463,778 |
| 18:00 | 3,751,239 |
| 20:00 | 5,341,435 |
| 21:00 | 278,594 |

Peak intensity is ~4–5M in a single hour; the rest of the day is sub-1M.
The session runs in long, tool-heavy bursts separated by quiet stretches.

## Why today's 19.26M is an inflated floor, not the target

Three compounding inefficiencies inflated the raw number. A working limit
should be sized to the *natural* workload, not the waste.

1. **Four turns were truncated at 1M** (`turn budget exhausted ... forcing
   yield` at 17:00, 17:07, 22:14, 00:38). Those turns were cut off and
   re-run, so we never saw their natural peak. Re-runs add burn on top of
   the real work.

2. **97% prompt re-send.** A heavy turn runs ~30 iterations; each iteration
   re-sends a growing ~95–103K context window. The model only *output*
   ~0.55M all day. The token counter accumulates `prompt + completion`
   every iteration, so per-turn totals are dominated by context
   re-accumulation, not by model work.

3. **Provider prefix-cache reuse was inert.** Each turn restarts context from
   scratch at the *provider* layer even though the lossless thread store
   persists history externally. With no explicit cache breakpoints and a
   per-iteration system-prompt rewrite, a heavy turn re-billed the stable
   prefix on every call.

   → Now addressed: see "Prompt-cache strategy" below for the typed
   segment + breakpoint design that reuses the stable prefix across turns
   and iterations.

## Derived caps

### Working daily limit: 15M

Reasoning:

- A **max-intensity turn** ≈ 30 iterations × ~100K per-call context ≈
  **~3M cumulative**.
- Five such turns ≈ **~15M**.
- Today hit 19.26M mostly *because* truncation forced re-runs on top of a
  ~15M natural workload.
- **15M** therefore fits ~5 full-intensity turns with headroom, while
  cutting off at a point that forces compaction (or a fresh session) rather
  than blind continued burn.

A tighter **12M** is defensible (a ~35% haircut) but leaves less room for a
day with five max-intensity turns; 15M is the comfortable working target.

### Per-turn cap: ~3–4M cumulative

- One max turn ≈ 30 calls × ~100K context ≈ ~3M; 4M leaves headroom for a
  tool-heavy chain.
- This is well above the current `1M` default that fired four times today.
- Note: a max turn at ~3M cumulative means the *per-turn* counter is again
  dominated by context re-send, not work. See the accounting caveat below.

### Per-call ceiling: ~128K (physical)

- Largest real call was 107K (103K prompt + 4K completion).
- This is a **physical** bound — the model's context window. No limit
  setting changes it. It means the real lever on spend is **how many calls
  a turn makes**, which is exactly what `max_tool_iterations` and the
  `per_step_*` caps control — not the token rail.

### Per-call completion budget (updated 2026-09-19; scoped 2026-10-06)

The turn loop's per-call completion budget (`turn_completion_max_tokens`)
is a **latency and runaway-generation guard**, not a spend control —
completion tokens are ~3% of baseline spend (97% is prompt re-send, per
the table above). Two properties keep it from becoming a silent failure:

- **Reasoning models default to 16384** (was 8192; plain models keep the
  legacy 4096). The 2026-09-19 narration-loop incident: a single
  completion that must hold deliberation *plus* an artifact-sized
  tool-call payload (expanded config + multi-KB report script) truncated
  at 8192 mid-tool-call across three different models — the visible
  "response" was the truncated narration half, and no guard fired. The
  raise is cost-negligible: the 97%-prompt baseline means completion
  headroom barely moves daily spend.
- **Truncation is visible and self-heals once.** All gateways now
  populate `finish_reason`/`truncated` on `CompletionResult`, and both
  turn-loop paths (streaming and non-streaming) retry a
  `finish_reason=length` truncation once at 2× budget, capped by the
  model's context window. The pathological bound is 32768.

**Scope updates (2026-10-06, `docs/SPEC/BUDGET_SCOPES_RFC.md`):** the
budget is no longer a single process-wide knob. Manifest model entries
may declare `max_completion_tokens` (provider-behavior ceiling); modes
may declare `completion_budget_tokens` (per-call work shape) and
`turn_completion_budget_tokens` (cumulative per-turn rail). Effective
per-call budget = `min(mode, model, settings)`, context-window-capped.
The 2026-10-06 glm-5.3:cloud marathons showed why: a host override
raised for report-sized finals became every mode's ceiling, and five
reasoning-dump calls burned 128k cumulative completion in one turn
(~30% of turn spend, not 3%) — the per-turn rail now exists to bound
that pattern.

## Accounting caveat (open decision)

Two coherent ways to wire the per-turn budget; the **daily 15M is the same
either way**, but the per-turn behavior differs:

- **Cumulative (current behavior):** sum `prompt + completion` per
  iteration. Per-turn must be ~3–4M to avoid firing mid-turn, but it is
  still counting context re-send as "spend."
- **Completion-based per-turn + cumulative daily:** count *completion*
  tokens per turn (real work, ~4–50K) so the per-turn rail never fires and
  iterations govern turn length; enforce the 15M daily on a cumulative
  ledger. This matches "cost guardrail" intent and stops the rail firing on
  window size.

The recommendation is to pursue the completion-based per-turn option so
turns are governed by work + iteration count, and the daily ledger is the
true cost ceiling.

## Decision record

- **Daily working limit:** 15M tokens (cumulative across a session/day).
- **Per-turn:** ~3–4M cumulative, or completion-based (pending the
  accounting decision above).
- **Per-call:** bounded by model context (~128K); not configurable.
- **Iterations:** `max_tool_iterations = 300` remains the per-turn runaway
  governor; `per_step_max_iterations / per_step_max_tool_calls` remain the
  per-step sub-budgets.

## Prompt-cache strategy

Manus/Anthropic's prompt-caching guidance plus the KV-cache literature point
to one dominant lever for an agent like Lisa (97% prompt tokens): reuse the
*stable* prefix across turns/iterations and keep dynamic content after the
breakpoint. Lisa implements this through typed prompt segments rather than
rolling back context management.

### Segment model

- `kernel.system_segments.SystemPromptSegments` splits the system prompt
  into ordered typed segments.
- **Stable** segments hold the harness, session instructions, Copilot
  customizations, and mode/stage instructions — the cross-turn prefix.
- **Ephemeral** segments carry the routing hint, handoff context, terminal
  directives, and the per-iteration `## Turn Lifecycle` briefing. They are
  replaced in place (keyed by name), never accumulate, and always render
  after the stable prefix.
- `flatten_system_prompt_segments` renders the legacy single string with
  the historical `\n\n` join, so providers without explicit cache
  controls get byte-identical prompts.

### Wire behavior by provider

- **Anthropic:** `sdk.providers.anthropic_gateway` emits the stable prefix as
  system content blocks with an ephemeral `cache_control` breakpoint on the
  final stable block; the dynamic tail stays uncached.
- **OpenAI / Azure:** implicit prefix caching — no explicit breakpoints, but
  tool schemas are canonically sorted by function name
  (`kernel.turn_phases._tool_schemas`) so the same mode produces a
  byte-identical tool payload across turns.
- **Gemini:** implicit caching only; explicit Cached Content resources are a
  separate follow-on (distinct lifecycle/TTL semantics).

### Rollback

`LISA_PROMPT_CACHE_ENABLED=false` flattens segments to the legacy
single-string system prompt and drops the Anthropic cache breakpoint. The
default is enabled.

### Telemetry

- `cached_tokens` — provider-reported cache *reads* (Anthropic/OpenAI/Google
  already surface these).
- `cache_creation_tokens` — prompt-cache *writes* (Anthropic
  `cache_creation_input_tokens`). Tracked separately from
  `reasoning_tokens`; earlier builds conflated the two.
- `/tokens` lifetime view and the introspection 24h rollup report cache
  reads, creations, and the read share: `cached / (prompt + cached)`
  provider-reported, not a billing estimate.

### Rollout evaluation checklist

1. Baseline a workload with caching disabled (`LISA_PROMPT_CACHE_ENABLED=false`)
   and record per-provider/per-model `cached_tokens`, `cache_creation_tokens`,
   prompt tokens, and TTFT/latency where available.
2. Re-run the same workload with caching on. Expected: large gain in
   cache-read share on the stable prefix; cache-creation tokens appear once
   per session then mostly zero until the cache expires.
3. Watch quality gates, not just hit rate: iteration count, forced-final
   rate, tool-call validity, workflow completion. Roll back via the env
   toggle if any of those regress; retain the segment plumbing — it is
   behavior-neutral when caching is off.
4. Do not raise turn/iteration budgets to *spend* the cache headroom; the
   savings should land on cost/latency, not longer burn.
