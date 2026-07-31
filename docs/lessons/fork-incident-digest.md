# Fork-Routing Guards — Why They Exist

> Active architectural invariants, not history. Each guard below closes a
> concrete failure mode that recurred across the 2026-07-17 → 2026-07-27
> incident chain. The incidents are resolved; the guards are load-bearing.
> Remove or "simplify" any of them and the failure returns.

## Invariant 1 — Full routable mesh in the routing FSM

**Code:** `kernel/ipsar_fsm.py :: synthesize_routing_transitions`
(`transitions[mode_id] = [m for m in routable_ids if m != mode_id]`)

**Without it:** From any non-`default` routable mode (`coding`, `search`,
`ponder`), the only FSM edge was back to `default`. The two-goal fork
detector (which checks `"learn" in routing_targets and "markdown" in
routing_targets`) silently no-oped, and the model fell back to whatever
write tool the current mode exposed — including approval-gated tools like
`coding_create_worktree`, producing a misleading approval loop instead of
the intended fork.

**What breaks if removed:** Fork fan-out (`fork:learn,markdown`) becomes
unreachable from every mode except `default`. Proactive routing hints must
be validated against every entry mode the user can reach, not just
`default`.

---

## Invariant 2 — Forced-fork enforcer (kernel takes over when the model won't)

**Code:** `kernel/turn_runner.py` (tool-dispatch branch + empty-completion
branch), `kernel/turn_phases/_turn_state.py :: forced_fork_target`,
`non_fork_tool_dispatches`; cap is the kernel setting
`forced_fork_non_fork_tool_cap`. Shared truth:
`kernel/turn_phases/_routing_hint.py :: _compute_fork_target_string`.

**Without it:** The routing hint is advisory prose that *asks* the model to
call `transition_stage(fork:...)`. The model can refuse — preferring
read-only research (`fetch_page`), burning iterations, then returning an
empty completion. The result was a 5+ minute heartbeat stall where the turn
quietly died and the UI looked frozen. There was no guard for "routing hint
said fork, model refused to fork."

**What it does:** After `forced_fork_non_fork_tool_cap` non-`transition_stage`
tool dispatches (or immediately on an empty completion with a fork pending),
the kernel synthesizes the `transition_stage(fork:...)` call itself and lets
the existing transition interception dispatch the fan-out. Fires at most
once per turn; `forced_fork_target` is cleared after firing.

**Philosophy:** Don't hand the model advisory prose and trust it. When the
kernel has already computed a deterministic next step, the model is the
executor; when the executor won't execute, the kernel takes over. This is
the opportunistic, pressure-driven pattern — the hint *is* the pressure.

---

## Invariant 3 — Action-tool validation for fork children

**Code:** `kernel/turn_phases/_fork.py :: _fork_mode_action_tool_ids` and
the validation branch in `_run_fork_child_turn`;
`kernel/turn_phases/_turn_state.py :: called_tool_names`;
`kernel/turn_runner.py :: TurnResult.called_tool_names`.

**Without it:** A fork child whose mode exists to take action (write a
file, persist a memory) could do read-only research, produce a hallucinated
"Done — wrote it to disk" text response, and be marked `status="ok"`. The
older `tool_call_count == 0` guard missed this because it asked "did the
child call *any* tool?" — a child making 4 `fetch_page` calls passed.

**What it does:** `_fork_mode_action_tool_ids` returns the mode's
`authored_allowed_operation_ids ∪ authored_allowed_tool_ids` — the tools
that actually *do* the mode's work. The validator flags
`status="no_tools"` when `called_tool_names ∩ action_tool_ids == ∅`, even
if `tool_call_count > 0`. The parent's `succeeded` count uses
`status == "ok"`, so `no_tools` children reduce the success count and the
parent surfaces them by name.

**Right question:** "did the child call *its action* tool?" — not "did it
call any tool?"

---

## Invariant 4 — Producer/consumer sequential guard

**Code:** `kernel/routing_vocabulary.py` (`DEFAULT_PRODUCER_INTENTS` =
`research`, `reasoning`; `DEFAULT_CONSUMER_INTENTS` = `write`, `coding`;
session-overridable via `routing.producer_intents` /
`routing.consumer_intents` in `session_context.md`);
`kernel/turn_phases/_routing_hint.py :: _fork_has_producer_consumer_mix`,
`_producer_first_hint`;
`kernel/workflow_instance.py` (intent-group ordering).

**Without it:** "Form an opinion… write it to a markdown doc" was routed as
a parallel fork. The `markdown` child needs `ponder`/`search`'s output as
*input*, but a fork runs all children in parallel — none can see another's
output. The `markdown` child had no evidence to write from, re-did the
research the other children were doing in parallel, and ran out of
iterations before reaching its action tool. `coding` was wrongly included
(a false intent match on "code-level documentation").

**What it does:** Refuses a fork when the gap intents mix a **producer**
(evidence-gathering: research/reasoning) with a **consumer**
(artifact-creating: write/coding). Routes sequential instead: single
handoff to the first producer, then the post-handoff hint recomputation
chains the consumer after the producer produces its evidence. Legitimate
independent forks (e.g. `learn` + `markdown`, where neither child needs the
other's output) still fire.

**Why it's subtle:** This is the most likely guard to be "cleaned up" by
someone who doesn't understand it. It encodes the distinction between
parallel and sequential multi-goal requests, which the earlier
sequential-connector detector (`_query_is_sequential`) missed because the
query used no "then" connector.

---

## Invariant 5 — `normalize_openai_messages` defaults assistant content to `""`

**Code:** `kernel/provider_core.py :: normalize_openai_messages`
(`normalized_message["content"] = ""` for assistant messages with no
content key). Single chokepoint for both `build_openai_completion_kwargs`
and `build_openai_responses_input`.

**Without it:** When a model emits tool calls without accompanying text,
the assistant message has no `content` key. The official OpenAI API accepts
`content: <nil>`; **Ollama's OpenAI-compatible endpoint rejects it** with
`invalid message content type: <nil>` (HTTP 400). The bug was silent
against OpenAI and only surfaced on strict local backends.

**Principle:** OpenAI-compatible ≠ OpenAI-identical. The single chokepoint
must enforce a shape acceptable to the strictest backend in the fleet.
Defaulting to `""` is safe everywhere.

---

## Invariant 6 — Distillation retry changes inputs, catches all failures

**Code:** `sdk/packs/memory/operations/memory/synthesize_learning.py`
(retry loop over two attempt payloads; `distill_max_tokens = 512`;
`response_format=DistilledLearning if attempt_idx == 0 else None`);
`kernel/operations/structured_output.py :: json_only_retry_messages`.

**Without it:** The local model ignored `response_format` and returned
empty content, or emitted JSON that truncated mid-array because
`max_tokens=300` was too small for the envelope + 3 facts. A first retry
attempt re-issued the *identical* request (same `response_format`, same
`max_tokens`) and `except RuntimeError: raise` bypassed the retry for the
JSON-truncation case entirely.

**What it does:** First attempt uses `response_format=DistilledLearning`;
the retry clears it (`response_format=None`) and appends a JSON-only
instruction. `max_tokens` raised 300 → 512. One `except Exception` catches
ALL first-attempt failures (empty output, JSON-truncation
`RuntimeError`, `ValidationError`) and continues to the retry.

**Principle:** A retry that re-issues the identical request is not a retry.
A retry must change at least one input (clear `response_format`, append a
JSON-only instruction, raise the token cap) and must catch all first-attempt
failure modes, not just the one that motivated the fix. The retry *is* the
structured-output contract — a "follows the contract" call site that parses
tolerantly but never retries is not actually following the contract.

---

## Cross-cutting principles

1. **A successful fork is not a successful turn is not a successful write.**
   Each layer can fail independently and mask the next.
2. **Don't trust advisory prose.** When the kernel has computed a
   deterministic next step, the model is the executor; when the executor
   won't execute, the kernel takes over.
3. **Verify success against the declared action surface**, not just "did it
   return" or "did it call any tool."
4. **A retry must change an input** and catch all first-attempt failure
   modes.
5. **OpenAI-compatible ≠ OpenAI-identical.** Normalize to the strictest
   backend in the fleet.
6. **Parallel forks are wrong for sequential queries.** Producer/consumer
   intent mixes route sequentially.
7. **All on by default, externalize later.** The mesh FSM and the
   forced-fork enforcer both follow the house pattern: default to the
   permissive behavior, defer per-edge restrictions to a later, smaller
   change.
