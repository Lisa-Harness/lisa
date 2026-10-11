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

## Sizing the completion budget

The knobs, from widest to narrowest:

| Knob | Scope | Sets |
| --- | --- | --- |
| `TURN_COMPLETION_MAX_TOKENS` | Process | Default per-call budget. |
| Manifest `max_completion_tokens` | Model | Provider ceiling for the model. |
| Mode `completion_budget_tokens` | Mode | Per-call work shape. |
| Mode `turn_completion_budget_tokens` | Mode | Per-turn cumulative rail. |

The effective per-call budget is `min(mode, model, settings)`, further
capped by the model's declared `context_window`. Declaring a value can
only tighten; widening requires narrowing the narrower scope.

### What the settings global is for

`turn_completion_max_tokens` (16384 default) is the **default per-call
budget for reasoning-capable models** — i.e., ones whose provider bills
chain-of-thought against `max_tokens` (Ollama). It buys headroom so one
call can hold deliberation *and* an artifact-sized tool payload without
truncating mid-write; the 2026-09-19 incident had 8192 cutting tool-call
payloads in half across three models. It is deliberately generous because
per the 2026-08-01 baseline, completion tokens are ~3% of burn — a wrong
*small* value silently aborts work, a wrong *large* value costs tokens
that mostly don't matter. Plain models never receive it: unflagged models
stay on the legacy 4096, so raising the global does not loosen
non-reasoning models. OpenAI/Azure gpt-5+ models (except `*-chat`) are
reasoning models: their hidden reasoning counts against
`max_completion_tokens`. Discovery flags them `reasoning_model`; older
manifests without the flag cap them at 4096 and trigger truncation retries
that resend the whole prompt. Re-run `/models discover` or add
`reasoning_model: true` to fix this.

**If you set the global** (e.g. `TURN_COMPLETION_MAX_TOKENS=8192`), every
reasoning-mode call in every mode tightens: the truncation rescue doubles
from 8192 instead of 16384 (so pathological calls bound at 16384), and
turns at the 90th–95th percentile of completion size (per the distribution
below) start hitting the cap mid-tool-call. Raising it to 32768 doubles
the pathological ceiling to 65536 and removes the per-call brake the
2026-10-06 marathons were stopped by — the per-turn rail and iteration
cap remain, but each silent reasoning dump now burns 2× before stopping.
Treat the global as the *default shape of work*, not a cost dial. Cost
enforcement lives on the daily ledger; runaway lives on iterations and
the per-turn rail.

### What to set per model (manifest `max_completion_tokens`)

Pick the value from the model's *observed* completion distribution for the
workload, not from its context window. Current turn-loop distribution in
this workspace (`artifacts/token-usage.jsonl`, n=1243, 2026-10-06): p50
≈ 500, p75 ≈ 2500, p90 ≈ 10.5k, p95 = 16384 (the cap — a saturated
quartile, evidence the budget is binding), p99 ≈ 32k. The 16384 default
for reasoning models is therefore already the p95 and is the right
starting point for chatty reasoning classes like `glm-5.3:cloud` —
dropping it saves nothing (3% of burn) and converts p90+ calls into
truncation aborts. If a model proves tighter in practice (the
manifest-level knob's true purpose), declare the ceiling it actually
needs; e.g. DeepSeek-V4.1-Flash stayed under 17k on 10-06 even unguided,
so `max_completion_tokens: 16384` is honest headroom, not a cut. Per the
manifest loader, `0`/absent inherits the process default; the floor for
any declared value is 1024, and the value must not exceed the model's
declared `context_window` when both are set.

### `context_window` on the manifest entry

`context_window` is the physical ceiling both the budget math and the
truncation rescue respect. When discovery can't fill it in (see the
probe-shape caveat below), declaring it by hand is correct — the kernel
caps retries and budgets at `min(budget, context_window)`, and an absent
window means an uncapped 2× rescue. For the Ollama cloud family currently
in use, the provider itself reports 1M-token contexts for glm-5.3,
deepseek-v4.1-flash, and kimi-k3 (probe via `/api/tags` →
`details.context_length`, or `/api/show` → `model_info`), so a hand-set
`context_window: 1048576` matches provider truth — but it is a physical
bound, not a sizing choice; the sizing choice above is separate.

### Probe-shape caveat (2026-10-06)

`/models discover --force` is supposed to fill `context_window` from the
provider, and the Ollama probe does call `/api/show`. Two reasons it came
back zero here:

1. `/api/tags` for the cloud proxy *does* carry
   `details.context_length` (e.g. 1048576 for glm-5.3:cloud), but the
   enrichment path ignores it in favor of `/api/show`.
2. The `/api/show` reader
   (`probes/ollama.py::_ollama_show_context_window`) only looks for
   `model_info["general.context_length"]`. The cloud models return
   family-keyed names (`glm_dsa_moe.context_length`,
   `deepseek_v41.context_length`, `kimi-k3.context_length`). Real local
   Ollama GGUF builds publish `general.context_length`; the Ollama cloud
   proxy and some runtimes do not — which is why the manifest silently
   dropped windows for every cloud-pulled model on this box.

The fix is a one-bug kernel change (read `*.context_length` from
`model_info`, fall back to `/api/tags` details) — see the kernel
carve-out discussion below. Until then, hand-declaring `context_window`
in `artifacts/config/models.yaml` is the workaround.

### The per-mode rails

`completion_budget_tokens` (per call) and `turn_completion_budget_tokens`
(cumulative per turn) are the mode-local levers, and no shipped pack sets
them yet — every mode inherits the settings/model path. Reach for the
per-turn rail in coding/action modes first: it's what stops the
five-cycle, 128k reasoning-dump pattern without touching per-call
headroom the report-sized finals legitimately need (`coding_review`'s
final must fit a full review verbatim, and those finals were the largest
legitimate completions in the trace — up to 47k).

### Resuming a capped turn

A cap or operator pause saves the loop at the iteration boundary, before
the model call it would have made. The thread entry records the saved
loop as `turn_continuation_ref`. When the next message is a resume phrase
("continue", "keep going …"), the host restores that loop rather than
re-routing and replaying. The message is appended as a user note, so the
cached prefix survives. A cap resume grants the original iteration and
token allowance again, so each resume is bounded and never unlimited. Any
other message routes normally.

### Failure modes of a wrong value

- **Too small (per call):** truncation mid-tool-call → the kernel's
  finish/no-tools abort fires (`completion truncated at the token cap
  with no tool calls … ending the turn`). Turns report `blocked=True`
  with partial prose. Worse, iterations where the lifecycle resolver has
  stripped tools (budget-exhausted final completions, router terminal
  actions) cannot be rescue-retried because the rescue predicate is
  `finish_reason == "length" and effective_tools` — see the carve-out
  list below.
- **Too large:** silent multi-minute reasoning dumps (the 2026-10-06
  2–7-minute calls), rescue doubling doubles the damage (2× cap), and
  cumulative completion climbs toward the per-turn rail. The daily ledger
  and per-turn rail stay the real cost guards; a too-large per-call value
  just lets each dump burn more before the rail fires.
- **No per-turn rail on an action mode:** a turn that loops at the
  per-call cap has no token-based stop — only `max_tool_iterations`
  (default 300) bounds it, which is why 10-06 coding turns ran to 107
  tool calls before anything intervened.

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

**Resolved (2026-10-10):** `max_total_tokens` (per turn) and
`max_session_tokens` now count **processed** tokens — uncached input plus
output. Cached prefix reads are excluded. The 2026-10-10 `p920` coding
turns stopped at iterations 11–15 because ~100K full-window requests
(57–62% cached) reached the 1M raw-total rail, so work stopped
mid-implementation. With processed accounting, a cache-friendly turn can
keep its full context for many more iterations; cache misses still count
in full.

## Decision record

- **Daily working limit:** 15M tokens (cumulative across a session/day).
- **Per-turn:** ~3–4M cumulative, or completion-based (pending the
  accounting decision above).
- **Per-call:** bounded by model context (~128K); not configurable.
- **Iterations:** `max_tool_iterations = 300` remains the per-turn runaway
  governor; `per_step_max_iterations / per_step_max_tool_calls` remain the
  per-step sub-budgets.
- **Completion sizing (2026-10-06 data):** reasoning-capable per-call
  default stays **16384** — it sits at the observed p95 of the turn-loop
  completion distribution, and completion is ~3% of spend, so tightening
  buys nothing while turning the busiest tenth of calls into truncations.
  Set per-model ceilings from observed distributions, not from context
  windows; leave DeepSeek-class flash models at 16384 unless telemetry
  shows otherwise. Declare `context_window` on models the probe can't
  size (the Ollama-cloud `/api/show` shape bug) so the rescue/budget math
  has a real ceiling.

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
  final stable block. Request-level automatic `cache_control` also moves the
  conversation breakpoint forward as history grows, for both complete and
  streaming calls. The host's cache opt-out disables both. Normalized prompt
  tokens include uncached input, cache reads, and cache writes, so cached
  context still counts toward request sizing and processed-token caps.
- **OpenAI / Azure:** prefix caching. Tool schemas are canonically sorted by function name
  (`kernel.turn_phases._tool_schemas`) so the same mode produces a
  byte-identical tool payload across turns. Each request also carries an
  opaque per-conversation `prompt_cache_key` so its calls route to the same
  cache. A deployment that rejects the parameter is retried once without it
  and stops receiving it for the rest of the process. Other
  OpenAI-compatible backends never receive it.
- **Azure GPT-5.6+ (`prompt_cache_breakpoints: true`):** these models only
  reuse a prefix that ends at a cache breakpoint; the implicit one sits on
  the latest message alone. The Chat Completions path therefore marks
  explicit `prompt_cache_breakpoint` blocks at the end of the stable system
  prefix (splitting the system message into segment blocks), on every
  message that ended an earlier request (the one before each assistant
  reply), and on the final message, keeping the latest 49 boundaries.
  Positions depend only on the history, so later requests re-mark the same
  prefixes. Discovery sets the flag for Azure `gpt-5.6`+ / `gpt-6`+;
  older models reject the field, so a rejecting deployment is retried once
  without breakpoints and not sent them again. Cache writes are billed on
  these models and reported as `cache_write_tokens`, recorded as
  `cache_creation_tokens`; `scripts/cache_report.py` shows them as
  `written`.
- **Azure transport:** the SDK constructs one OpenAI v1 client at
  `{endpoint}/openai/v1/` for chat, Responses, and embeddings. The runtime
  resolves credentials and caches the client; dated chat and preview
  Responses `api-version` settings are no longer used for inference.
- **Responses continuation:** configured Azure Responses calls use
  `store=false` and request `reasoning.encrypted_content`. The SDK returns
  encrypted reasoning as opaque continuation items; the kernel stores them
  on the owning assistant message alongside, not inside, visible content.
  The SDK replays them before that message's tool calls only for the same
  provider and model. Chat and Gemini wire formats never receive those items.
  Removing an assistant message removes its continuation too.
- **Gemini:** implicit caching only; explicit Cached Content resources are a
  separate follow-on (distinct lifecycle/TTL semantics).

Memory embeddings also use the SDK's OpenAI-compatible embedding gateway.
The memory server only resolves CLI/environment configuration; it no longer
sends provider HTTP itself. Gateway failures and malformed vector counts or
dimensions propagate to the store's logged best-effort boundary, preserving
lexical memory writes and recall. The configured endpoint must implement the
OpenAI `/embeddings` shape (including Ollama's `/v1` surface).

### Reduction hysteresis

Tool results fill whatever the complete request leaves after required
content, the completion reservation, and `context_request_margin_share`.
When a call would exceed that limit, the kernel reduces resident results to
`context_reduction_target_share` (default `0.75`) of the room tool results
may occupy, not to just under the limit. Fitting just under the limit made
nearly every later call reduce again, rewriting early messages and missing
the provider prefix cache. The slack lets the following calls append to an
unchanged prefix until the next reduction.

### Mode-stable prefix

The system prompt is mode-specific, so a mode switch misses the provider
cache. A host started per message used to re-enter `start_mode` every turn,
then hand off back to the working mode: one extra routing call plus a cold
prefix per user message. The host now restores the mode the latest thread
entry left active (see `docs/ROUTER_DESIGN.md`), so follow-up turns start in
the mode whose prefix is already cached.

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
