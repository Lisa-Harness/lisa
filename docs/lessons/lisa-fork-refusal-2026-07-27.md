# Lisa — Model Refused to Fork, Kernel Forced It (2026-07-27)

> **Historical incident record (2026-07-27).** Resolved and shipped. The
> forced-fork enforcer and `_compute_fork_target_string` helper described
> here are confirmed in current code (`kernel/turn_runner.py`,
> `kernel/turn_phases/_routing_hint.py`,
> `kernel/turn_phases/_turn_state.py`). Regression tests
> (`test_forced_fork_when_model_refuses_to_fork`,
> `test_forced_fork_when_model_returns_empty_completion`) are checked in.
> Kept as a postmortem.

Follow-on to `lisa-fork-false-success-2026-07-18.md`. By 2026-07-27 the
`no_tools` guard already caught children that hallucinated success. The
remaining gap was upstream: the model refused to call `transition_stage`
with the `fork:` target at all, so the fork never ran.

---

## Prompt (user)

```plaintext
do some follow up searches on how to use fzf and zoxide together in
particular and write a guide in markdown with examples I can copy/paste
to learn
```

And later:

```plaintext
use your self awareness combined with some web searches to form an
opinion about yourself relative to other agent/harness frameworks like
langchain, smolagents, or whatever else you can find. We already
integrate with letta and mem0 but how can we better connect the
ecosystem and what else should you do? You can read your code and the
code-level documentation in /docs to form an opinion. Write it to a
markdown doc.
```

## Observed response (Lisa)

The UI sat on heartbeats for 5+ minutes. No `fork_start`, no
`fork_child_start`, no further `httpx` calls. The thread log recorded:

```plaintext
"The model did not produce a final response."
```

On the second run, the router escalated to `gpt-5.4` (not mini) at the
iteration-cap fallback. The bigger model, forced into a no-tools final
completion (`tools=None` at the iteration cap), hallucinated the tool
call as prose with injected garbage:

```plaintext
to=functions.transition_stage
北京赛车女json
{"target_stage_id":"fork:ponder,markdown,coding,search",...}
```

## Evidence (`artifacts/host.log`, `artifacts/threads/testing.jsonl`)

Two independent runs, same failure shape.

### Run A — 16:53 (fzf/zoxide guide)

```plaintext
16:53:23 routing hint injected mode=default → hand off to plan
16:53:27 post-handoff routing hint mode=plan
         hint=Hand off ... target_stage_id `fork:learn,ponder,markdown,search`
16:53:29–16:53:39  model does a flurry of fetch_page calls
                  (fzf, zoxide, bat, ripgrep repos)
16:53:39  /api/turn/stream returns 5080 bytes
          thread log: "The model did not produce a final response."
16:53:39+ only heartbeats for 5+ minutes. No fork_start, no
          fork_child_start, no further httpx calls. Turn is dead.
```

### Run B — 19:07 / 19:12 (langchain comparison)

```plaintext
19:07:42 turn loop exited with empty content mode=plan iteration=7/15
         tool_calls_total=5 handoffs=1 entry_failed=False fork_no_tools=False
         transitioned_to=None routing_hint=True
```

5 tool calls = `fetch_page` on langchain docs. **0 `transition_stage`
calls.** The model did research instead of forking.

```plaintext
19:12:38 httpx: POST .../gpt-5.4/... 200 OK   ← router escalated to gpt-5.4
19:12:42 httpx: POST .../gpt-5.4/... 200 OK   ← iteration-cap fallback
```

Thread log at `23:12:42 UTC` captured the hallucinated tool-call-as-prose
shown above.

## Root cause

The fork was **advisory**. The routing hint (`kernel/turn_runner.py`
fork branch) is prose that *asks* the model to call `transition_stage`
with `fork:...`. The kernel had guards for:

- `handoff_count > _MAX_HANDOFFS_PER_TURN` — only counts *after* a
  handoff fired.
- `fork_no_tools` — only fires *after* a fork ran.
- `entry_failed` — only on mode-entry failure.

**There was no guard for "routing hint said fork, model refused to
fork."** So `gpt-5.4-mini` in `plan` mode, faced with a strict tool
schema and a hint telling it to fan out, did the path of least
resistance: it called `fetch_page` repeatedly (those are allowed tools
in plan), burned iterations, then returned an empty completion.
`tools_dispatched` was True (fetches happened), so the loop continued
until iteration 7, where the model returned **no tools AND no content**
→ `not tools_dispatched: break` → empty-content guard → iteration-cap
fallback → junk/hallucinated tool-call-as-text.

The fork fan-out code in `kernel/turn_phases/_fork.py` was fine. It
never got called.

### Why the UI looked frozen

It was not frozen — the turn ended. The web server was alive (heartbeats
200 OK every 5s). But the turn ended with `"The model did not produce a
final response."` or with Chinese-spam-laced hallucinated prose, so the
user saw nothing useful and assumed it was still working.

## Philosophy

This incident is the same lesson as the `no_tools` guard added after the
2026-07-25 incident, applied one layer earlier:

> **Don't hand the model advisory prose and trust it.** When the kernel
> has already computed a deterministic next step, the model is the
> executor. When the executor won't execute, the kernel takes over.

The routing hint already encodes the routing decision (which modes
cover which detected intent gaps). The model's only job is to call
`transition_stage` with the named target. When it refuses — preferring
to do "useful" read-only work or returning empty — the kernel forces the
fork itself. This matches Adam's standing architectural preference for
**opportunistic, pressure-driven** kernel action over begging the model
via nudges: the hint *is* the pressure; once the model has shown it won't
act on it, the kernel acts.

The enforcer is deliberately cheap and bounded:

- **Cheap**: it re-runs the same `_compute_fork_target_string` logic
  that built the hint, so the kernel and the model share one truth
  about what the fork target is. No new routing decision is made.
- **Bounded**: it fires after `_FORCED_FORK_NON_FORK_TOOL_CAP` non-fork
  tool dispatches (default 2) or immediately on an empty completion
  with a fork pending. It fires at most once per turn
  (`forced_fork_target` is cleared after firing).
- **Tag-blind**: the kernel only inspects opaque pack-authored
  intent-tag sets and routing target ids. It names no intent
  vocabulary. The forced target is exactly what the hint named.

## Fix

Three changes, all in the turn loop:

1. **`_compute_fork_target_string`** (`kernel/turn_phases/_routing_hint.py`)
   — extracted the pure, prose-free fork-target decision out of
   `_compute_capability_routing_hint`. Returns the `fork:mode,mode,...`
   string (or `None`) for a query + routing_targets + current mode.
   The hint emitter and the enforcer share this one truth.

2. **`_TurnState.forced_fork_target` / `non_fork_tool_dispatches`**
   (`kernel/turn_phases/_turn_state.py`) — per-turn state for the
   enforcer. Populated when a fork hint is injected; refreshed after
   each handoff by the chained-hint recomputation block; cleared after
   the enforcer fires.

3. **Forced-fork enforcer** (`kernel/turn_runner.py`) — two entry
   points:
   - **Tool-dispatch path**: when `forced_fork_target` is set and the
     model calls a non-`transition_stage` tool, increment
     `non_fork_tool_dispatches`. At the cap, synthesize the
     `transition_stage(fork:...)` call by overriding the current tool
     call's name/arguments and let the existing transition interception
     dispatch the fan-out.
   - **Empty-completion path**: when `forced_fork_target` is set and the
     model returns no tool calls, synthesize the fork call and dispatch
     it on the next iteration instead of accepting the empty turn.

Both paths clear `forced_fork_target` after firing so the enforcer runs
at most once per turn and the post-fork hint recomputation can set a
fresh target if the chain continues.

### Files

- `kernel/turn_phases/_routing_hint.py` — `_compute_fork_target_string`.
- `kernel/turn_phases/_turn_state.py` — `forced_fork_target`,
  `non_fork_tool_dispatches` fields.
- `kernel/turn_runner.py` — `_FORCED_FORK_NON_FORK_TOOL_CAP` constant,
  enforcer in the tool-dispatch branch and the empty-completion branch,
  `forced_fork_target` population in the hint-injection and chained-hint
  blocks.
- `kernel/tests/test_turn_runner_transitions.py` — regression tests
  `test_forced_fork_when_model_refuses_to_fork`,
  `test_forced_fork_when_model_returns_empty_completion`, and unit
  tests for `_compute_fork_target_string`.

### Secondary (unrelated) issues in the same log

These were observed during the same session but are **not** addressed by
this fix; tracked separately.

- `lisa-pack-code` auto-install failed: `github.com/Lisa-framework/lisa-pack-code`
  returns 404. Repository does not exist or is not public.
- `session_context.md` fails `SessionManifestV2` validation (`surface`
  field required, and an `extra_forbidden` `configuration` block). The
  session manifest schema was out of date with the current
  `SessionManifestV2` definition. Fixed independently by restoring the
  `surface` block.

## Follow-on: fork children hallucinated success with read-only tools

The enforcer fixed the parent stall, but the next run exposed a child
stall one layer down. With the fork now actually running, the children
(marked `markdown`, `coding`) burned their `max_tool_iterations=5` on
read-only `fetch_page` / `web_search` calls and exited empty without
ever invoking their action tools (`markdown_forge_document`,
`coding_apply_patch`).

### Evidence

```plaintext
20:04:50 WARNING: model refused to fork (mode=plan, iteration=3,
              non_fork_tool_dispatches=2); kernel forcing fork
              target=fork:ponder,markdown,coding,search
20:05:26 WARNING: turn loop exited with empty content mode=markdown
              iteration=4/5 tool_calls_total=4 handoffs=1
20:05:29 WARNING: fork child coding completed with 0 tool calls ...
              marking no_tools
```

Final user-facing response:

```plaintext
Fork fan-out failed: the following children made zero tool calls despite
their mode declaring action tools (coding), so their reported success is
unverified — nothing was written to disk or memory. Re-issue the request.
```

### Why the existing `no_tools` guard missed it

The 2026-07-18 `no_tools` guard checked `tool_call_count == 0`. That
catches a child that produced a hallucinated "Done — wrote it to disk"
text response with **zero** tool calls. But the `markdown` child here
made **4** tool calls — they were all `fetch_page` (read-only research).
The `tool_call_count == 0` guard passed it as `ok` despite nothing
landing on disk. The guard answered the wrong question: "did the child
call *any* tool?" instead of "did the child call *its action* tool?"

### Fix — action-tool validation

`_run_fork_child_turn` (`kernel/turn_phases/_fork.py`) now also checks
that at least one of the mode's **authored action operation/tool ids**
was invoked, not just that *some* tool was called.

- **`_fork_mode_action_tool_ids`** — returns the mode's
  `authored_allowed_operation_ids ∪ authored_allowed_tool_ids`. These are
  the tools that actually *do* the mode's work (write, learn, apply).
- **`TurnResult.called_tool_names`** / **`_TurnState.called_tool_names`**
  — new set tracking every tool name dispatched this turn. Populated at
  the dispatch site; surfaced on `TurnResult`.
- The validator flags a child `status="no_tools"` when
  `called_tool_names ∩ action_tool_ids == ∅`, even if `tool_call_count > 0`.

Same philosophy as the parent enforcer, applied to the children: the
mode's authored allowlist already declares what "doing the job" means.
A child that didn't invoke any of those tools didn't do its job, and the
kernel must not report it as succeeded.

### Files (follow-on)

- `kernel/turn_phases/_fork.py` — `_fork_mode_action_tool_ids`, the new
  action-tool validation branch in `_run_fork_child_turn`.
- `kernel/turn_phases/_turn_state.py` — `called_tool_names` field.
- `kernel/turn_runner.py` — `TurnResult.called_tool_names`, populate at
  dispatch, re-export `_fork_mode_action_tool_ids`.
- `kernel/tests/test_turn_runner.py` —
  `test_run_turn_fork_marks_no_tools_child_when_action_tool_skipped`,
  `test_fork_mode_action_tool_ids_returns_authored_action_tools`.

## Root cause: the fork was the wrong routing for sequential queries

The action-tool guard and the `ponder` false-positive fix were necessary
but not sufficient. Even with gpt-5.4 on every child and the guards
correctly tuned, the workflow still failed: `markdown` did 7 read-only
`fetch_page`/`web_search` calls across all 5 iterations and never
reached `write_file`; `coding` did 0 tool calls (there was no coding
task). The guards caught the *symptom* (children not calling action
tools) but the *disease* was upstream: **the routing hint suggested a
parallel fork for a sequential query.**

"Form an opinion… write it to a markdown doc" is sequential — the
`markdown` child needs `ponder`/`search`'s output as *input*. A fork
runs all children in parallel with the whole query; none can see
another's output. The `markdown` child had no evidence to write from, so
it re-did the research the other children were doing in parallel and ran
out of iterations before reaching its action tool. `coding` was wrongly
included (no coding intent in the query — "code-level documentation"
triggered a false `coding` intent match). The sequential detector
(`_query_is_sequential`) missed it because the query used no "then"
connector.

### Fix — producer/consumer sequential guard

The kernel now refuses to suggest a fork when the gap intents mix a
**producer** intent (research/reasoning — satisfied by producing text
the next step consumes) with a **consumer** intent (write/coding —
satisfied by consuming that text to create an artifact). Instead it
routes sequential: single handoff to the first *producer* target, and
the post-handoff hint recomputation chains the consumer after the
producer produces its evidence.

- **`kernel/routing_vocabulary.py`** — new
  `DEFAULT_PRODUCER_INTENTS` / `DEFAULT_CONSUMER_INTENTS` and
  `get_producer_intents()` / `get_consumer_intents()`, following the
  existing session-injected vocabulary pattern
  (`register_routing_vocabulary` now accepts `producer_intents` /
  `consumer_intents`). Sessions override via
  `routing.producer_intents` / `routing.consumer_intents` in
  `session_context.md`. The kernel reads them as opaque tags — it never
  names the intents in code (the defaults are the historical vocabulary,
  mirroring the sequential-connector pattern).
- **`kernel/turn_phases/_routing_hint.py`** —
  `_fork_has_producer_consumer_mix` detects the mix;
  `_build_fork_target` refuses the fork when it fires;
  `_producer_first_hint` builds the sequential single-handoff hint to
  the first producer; `_compute_capability_routing_hint` calls it both
  after a refused fork (section 4) and before the single-handoff
  fallback (section 4b, so a consumer that "covers" a gap intent isn't
  picked first).

Legitimate independent forks (e.g. `learn` + `markdown`, where neither
child needs the other's output) still fire — the guard only refuses
forks that mix a producer with a consumer.

### Files (root-cause)

- `kernel/routing_vocabulary.py` — producer/consumer intent registry.
- `kernel/turn_phases/_routing_hint.py` — `_fork_has_producer_consumer_mix`,
  `_producer_first_hint`, guard in `_build_fork_target`, fallbacks in
  `_compute_capability_routing_hint`.
- `kernel/tests/test_turn_runner.py` —
  `test_routing_hint_refuses_fork_for_producer_consumer_mix`.