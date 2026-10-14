# Token budgets and accounting

Limits have different units and lifetimes. A completion ceiling is not a
turn budget, a context window is not a spending allowance, and a session
limit is not a daily or currency cap.

## Limit scopes

| Scope | Controls | Source |
| --- | --- | --- |
| Model request | Input plus output must fit the window/margin | Model `context_window`; [allocator](../kernel/src/kernel/context/request_budget.py) |
| Completion call | Output/reasoning allowance for one call | Kernel setting, model/mode ceilings |
| Turn | Calls, processed/completion tokens, elapsed time | Host execution policy and mode rail |
| Workflow step | Local iteration/tool-call bounds | Mode/stage policy and lifecycle |
| Session | Processed tokens before admitting more turns | `execution.max_session_tokens` |
| Provider account | Currency, quotas, billing period | Provider/host infrastructure, not kernel |

## Per-call completion ceilings

The primary turn-loop path uses the narrowest applicable limit:

- `Settings.turn_completion_max_tokens` (`TURN_COMPLETION_MAX_TOKENS`),
  default **16384**.
- Model manifest `max_completion_tokens`, when declared.
- Mode `completion_budget_tokens`, when declared.
- Legacy **4096** ceiling for models not marked reasoning/thinking-capable.
- Model window and the fitted request's actual remaining output space.

Declared model ceilings must be at least 1024 and not exceed a declared
context window. Zero/absent inherits. Mode declarations must be positive
when authored; omission inherits.

```yaml
# Fields on one model entry, not a complete manifest
model_id: example-reasoning-model
provider: openai
reasoning_model: true
context_window: 131072
max_completion_tokens: 8192
```

```yaml
# Mode front matter fields
completion_budget_tokens: 8192
turn_completion_budget_tokens: 50000
```

The second mode field is cumulative over a **turn**, not one response.
It tightens an existing host completion rail or establishes one if absent.
Larger declarations cannot loosen smaller enclosing limits.

Numbers above are illustrative. Reasoning can consume output allowance
without visible text or tool dispatch. Declarative operations have separate
`completion_max_tokens` metadata; the mode knob does not size every auxiliary
call. See [operation authoring](../sdk/AUTHORING.md).

## Truncation and reasoning burnout

Provider `finish_reason`/`truncated` signals make interrupted output visible.
The loop can retry a truncated completion once with a doubled allowance,
bounded by model window and request space. The initial ceiling is therefore
not an absolute cap on a rescue call or aggregate usage.

A clean `stop` with empty content, no tool calls, substantial reasoning,
and completion usage at least **90%** of the effective allowance also
qualifies for the one-shot burnout rescue. Cheap empty completions below
the threshold do not. Turn/session limits and watchdogs remain applicable.

Inspect usage, finish reason, reasoning flags, and tool support before
raising limits; a larger ceiling can prolong runaway generation.

## Turn and session execution limits

Session `execution` requests tighten the host envelope; they do not grant
permissions. Limits cover model/capability calls, processed tokens,
completion tokens, wall time, and per-capability counts.

The [interactive runtime default](../runtime/src/runtime/execution_policy.py)
provides 500 model calls, 500 capability calls, 1,000,000 processed tokens,
and 3,600,000 ms per turn. Other kernel/step limits may stop work earlier.
Hosts can supply a different envelope.

The [core default session](../sdk/packs/core/src/sdk/packs/core/sessions/default.md)
requests 15,000,000 session tokens and 500,000 completion tokens per turn.
These are resource rails, **not** a daily reset or monetary guarantee.

```yaml
execution:
  max_model_calls: 100
  max_capability_calls: 100
  max_total_tokens: 500000
  max_completion_tokens: 50000
  max_wall_ms: 600000
  max_session_tokens: 2000000
```

The example requests tighter limits. Exhaustion can pause, yield, terminate,
or force a closing completion according to lifecycle/policy. Approval
continuation retains consumed counters; approval does not reset budgets.

## Accounting and cost

Usage distinguishes prompt, completion, reasoning, cache-read
(`cached_tokens`), cache-write (`cache_creation_tokens`), and raw totals.
Providers do not all report the same detail.

Execution rails count **processed tokens**: input plus output minus cached
prefix reads. Uncached input, output, and cache creation still count.
A cache hit does not remove input from the physical context window.

Kernel carries counts, **not currency**. Hosts/SDKs estimate cost from manifest
input/output/cache prices; compare estimates with provider billing.
Missing usage detail, hidden reasoning, and provider-specific pricing affect
accuracy.

Inspect runtime `/tokens`, `logs/token-usage.jsonl`, and the
[cache report helper](../scripts/cache_report.py). Request snapshots record
context sizing; see [context lifecycle](CONTEXT_LIFECYCLE.md).

## Calibrate from your workload

Measure distributions, saturation, dispatches, window pressure, cache
reads/writes, latency, and verified outcomes for representative tasks.
Do not use the former August/October single-workspace totals as global
recommendations. Prompt-heavy work can become output-heavy during reasoning
burnout; a large window does not imply a sensible output ceiling.
Recheck after model, provider, prompt, or workflow changes.
