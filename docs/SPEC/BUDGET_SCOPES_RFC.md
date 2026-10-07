# Per-Model and Per-Mode Completion Budgets (Design, Freeze-Gated)

Status: **implemented 2026-10-06** (commits fecc658c watchdog, 8313403d
scope 1, c4dc6fbf scope 2, 2f94f21e scope 3) after the operator lifted the
freeze gate for this work ("all freeze-gated work; no branch"). This
document is retained as the design record; the sections below document
the shipped behavior.

## Problem

One process-wide environment knob (`TURN_COMPLETION_MAX_TOKENS`, kernel
`Settings.turn_completion_max_tokens`) sizes every turn-loop model call. The
2026-10-06 marathons showed the failure mode with the override in place:

| Symptom | Mechanism |
| --- | --- |
| 5-minute silent generations, ×3 in one day | glm-5.3:cloud dumped reasoning into the completion channel until it hit the 32768 cap (`reasoning_tokens` reports 0; Ollama does not split reasoning from content) |
| 13-minute "rescue" doubling the damage | The truncation rescue (`kernel.turn_runner` § stream truncated) retried at 2× — 65536 — giving the dump an even larger budget. Mid-stream stalls ran unbounded (no idle watchdog: open freeze-compliant bug fix, separate item) |
| One turn: 128,679 completion tokens across 5 calls, ended `blocked`, no verdict | The final call filled the base cap exactly (`completion=32768`) with zero tool calls; the kernel ended the turn instead of synthesizing a dispatch |
| Completion share ~30% (128k of 430k) | The `.env` override's premise ("completion is ~3% of baseline spend", `docs/TOKEN_BUDGET.md`) holds for deliberate completions only — reasoning-dump turns invert it |

One knob cannot express two different contracts that today share it:

1. **Model capability budget** — "how much may this model bill per call"
   (provider behavior; the manifest already knows the model's context window
   and reasoning class).
2. **Mode work-shape budget** — "how big a payload does work in this mode
   produce" (a coding_execute `coding_edit_files` payload and a
   coding_report synthesis differ by an order of magnitude in practice).

## Design

Three scopes, strict precedence, each the smallest version that solves its
own problem. Where two declare a value, the narrowest wins.

### Scope 1 — per-model (`max_completion_tokens` on the manifest entry)

`ManifestModelEntry` gains:

```yaml
- model_id: glm-5.3:cloud
  provider: ollama
  max_completion_tokens: 16384   # NEW: 0/absent = host default
```

- `load_model_manifest` validates `>= 1024` and `<= context_window` (when
  both set) at load time; a violation is a load-time error, not a runtime
  clamp — the manifest is the wrong place to hide a fix-up silently.
- `StubModelDefinition` (runtime `model_runtime.py`) carries the field
  through.
- `_turn_completion_max_tokens(model_def)` becomes
  `min(host_budget, model_budget)` when a model budget is declared; the
  existing legacy/reasoning classing stays as the fallback path.

Covers: "glm-5.3-family may not bill more than 16k per call" for every mode
at once. Also the natural home for flash-model tightening — a flash model
declaring a smaller budget makes the rescue math
(`2×, context-window-capped`) proportionally smaller for free.

Ownership note: `kernel/model_manifest.py` + the kernel settings reader
change; `runtime/model_runtime.py` field pass-through; the Ollama-facing
entry in the host's `models.yaml` is data. No pack changes.

### Scope 2 — per-mode (`completion_budget_tokens` mode front matter)

`WorkflowModeConfig` gains a typed field; packs declare it in `_mode.md`
front matter (kernel-owned key, so it joins `_KERNEL_OWNED_FRONT_MATTER_KEYS`
and is parsed into the field, not left opaque in `extra_front_matter`):

```yaml
---
mode_id: coding_execute
completion_budget_tokens: 16384
---
```

- Precedence overall:

  ```
  mode_config.completion_budget_tokens      (pack-authored, work shape)
    vs model entry max_completion_tokens    (provider behavior ceiling)
    vs settings.turn_completion_max_tokens  (process default)
  → effective = min(declared values), legacy/reasoning classing only when
    nothing declared. Absent = inherited (0 = no declaration), so no pack
    update is load-bearing and there is no hard cutover of pack content.
  ```

- The turn runner already loads `mode_config` before building the completion
  request (`kernel/turn_runner` § PRE-TURN, ~line 2080); the change is
  threading it to `_turn_completion_max_tokens` and the truncation-rescue
  math (which must double from the *effective* budget, not the settings
  value).
- The per-mode field also arms the budget briefing the model already reads,
  by adding one line to the existing briefing block; models that know their
  ceiling plan their thinking instead of discovering it by truncation.

Governance note (the decision this design asks the operator to ratify): a
pack authoring a *kernel resource limit* crosses the pack/kernel ownership
boundary. Today the kernel owns turn economics (iteration caps, per-target
`max_calls_per_turn`) and packs own vocabulary; `extra_front_matter` exists
precisely so packs can carry app concepts the kernel never reads. Scope 2
makes the kernel read one more pack-authored number — deliberately, for
capability budgets that are work-shape properties. The alternative (host-only
per-mode config in `models.yaml`, pack stays vocabulary-only) does not need
the boundary change but leaves pack distributions unable to express their
own workload limits.

### Scope 3 — per-mode spend rail (mode-authored `TurnBudget`)

`budget_forces_final` already implements the completion-based turn rail
(`kernel/turn_phases/_turn_state.py` § budget_forces_final: completion
tokens, not context re-send) — the accounting direction `TOKEN_BUDGET.md`
§ "Accounting caveat" recommends. What's missing is *authority*: today the
rail exists only when a host supplies an `ExecutionPolicy`/`TurnBudget`;
the CLI's interactive default sets no completion rail.

This scope adds a mode-declared turn budget (front matter, e.g.
`turn_completion_budget_tokens: 200000`) that the turn runner folds into the
effective `TurnBudget` when the host did not supply one (host-declared wins;
fold, never tighten backwards). That converts the 2026-10-06 pattern — five
dump cycles, 128k cumulative completion, still no verdict — into a
bounded "force one final no-tools completion" stop that returns the model's
best answer instead of escalating burn per cycle.

### Out of scope, related, freeze-compliant now

- **Stream idle watchdog** in `_collect_stream` (a stalled upstream stream
  hangs the turn forever): a bug fix with a regression test, eligible under
  the freeze without this design. Listed here only as the boundary between
  "bounded" and "budgeted" behavior. **Shipped** (`fecc658c`).
- **Tools-stripped truncations skip the rescue.** The rescue predicate is
  `finish_reason == "length" and effective_tools`, but the turn loop sets
  `effective_tools = None` on pending-yield iterations and on any
  `_FinishAction` iteration (budget-exhausted finals, router terminal
  actions, approval continuations). On 2026-10-06, four of ten
  truncation aborts (including the three final turns of the session)
  landed on exactly those iterations: the model was asked for a no-tools
  closing summary, dumped reasoning into content, hit the cap, and the
  abort path surfaced the reasoning tail as the answer with
  `blocked=True`. Fix shape: run the same one-shot 2× retry (cap-bounded)
  for `finish_reason == "length"` on tools-stripped completions —
  content-free or not — instead of aborting immediately. Bug fix +
  regression test; freeze-eligible.
- **Ollama-cloud context-window probe misses.** `/api/tags` on the Ollama
  cloud proxy carries `details.context_length` (e.g. 1048576 for
  glm-5.3:cloud), and `/api/show` returns family-keyed model_info entries
  (`glm_dsa_moe.context_length`, `deepseek_v41.context_length`,
  `kimi-k3.context_length`). `_ollama_show_context_window` only reads
  `model_info["general.context_length"]`, so every Ollama-cloud model on
  this host landed in the manifest with `context_window` unset — leaving
  the rescue retry uncapped and the per-model ceiling without its bound.
  Fix shape: read `*.context_length` from `model_info`, fall back to
  `/api/tags` `details.context_length`, keep the static-registry override
  last. Bug fix + regression test; freeze-eligible.
- **Chatty-reasoning model behavior (no kernel fix).** glm-5.3:cloud and
  deepseek-v4.1-flash:cloud began emitting 20k–47k completions on
  2026-10-06 (up from ≤8k through 10-05), with per-call latencies of
  2–7 minutes. Mitigations: `context_window` declarations (above), a
  per-turn completion rail on `coding_execute` (mode front matter, not
  kernel), or switching the mode model until the behavior settles. No
  kernel change is proposed for this.
- **Router capability vocabulary**: the inert
  `"code" in route.required_capabilities` rule was worked around per-host by
  the `mode_models` pin + (optionally) declaring
  `required_capabilities: [code, tools]` in the coding modes' front matter.
  That front-matter addition is pack content, not frozen-tree change, and is
  recommended regardless of this design.

## Test plan (when implemented)

- Manifest: negative budgets, budget > context window, absent field, apply
  precedence through `_turn_completion_max_tokens`.
- Mode front matter: parse into the typed field; unknown-pack fallback to
  model/settings; budget briefing line reflects the effective value.
- Rescue math: truncation retry doubles from the effective budget and is
  capped by the smaller of model window / declared values.
- Mode-authored turn budget: host-declared wins; mode-declared rail forces
  the final completion at exhaustion (the 128k-dump scenario replays to a
  bounded end).
- Regression: the 2026-10-06 token-usage shape (5 calls, per-call 32768
  fills with dumps) replays against the new precedence and asserts the
  effective budget and cumulative rail bite.

## Deployment order after freeze lifts

1. Scope 1 (manifest field) — smallest surface, immediately tames
   reasoning-class models globally. **Shipped.**
2. Scope 2 (mode front matter + precedence) — completes work-shape
   authority; update `PACK_AUTHORING_GUIDE.md` and the SDK router README in
   the same change. **Shipped.**
3. Scope 3 (mode-authored turn rail) — closes the cumulative runaway.
   **Shipped.**
4. Host config cleanup: after 1–2, revisit whether
   `TURN_COMPLETION_MAX_TOKENS` stays an env knob at all or becomes the
   documented process default only. **Open** — the host `.env` rolled back
   to 16384 on 2026-10-06 with the runaway evidence recorded inline.

## Shipped behavior notes

- The stream idle watchdog (`Settings.stream_idle_timeout_s`, default
  300s, env `STREAM_IDLE_TIMEOUT_S`, 0 disables) shipped alongside the
  scopes as the freeze-compliant bug fix: a stalled stream now raises
  `StreamIdleTimeout` instead of hanging the turn forever.
- Scope 2's front-matter key names: `completion_budget_tokens` (per call).
- Scope 3's front-matter key name: `turn_completion_budget_tokens`
  (cumulative per turn).
- Neither key is set by any shipped pack yet; absent declarations inherit
  the pre-existing settings path unchanged. The coding pack opting in is
  pack content (unfrozen) and can land independently.
