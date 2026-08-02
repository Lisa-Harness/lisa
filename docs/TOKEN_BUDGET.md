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

3. **No cross-turn persistence.** Each turn restarts context from scratch,
   so a multi-step task re-reads the same material repeatedly across
   handoffs.

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
