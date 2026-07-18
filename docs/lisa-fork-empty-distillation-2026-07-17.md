# Lisa — Fork Reached, Both Halves Silently Failed (Empty Distillation + Nil Content)

> Captured 2026-07-17. Follow-on to `lisa-coding-mode-fork-failure-2026-07-17.md`.
> The earlier FSM fix worked: Lisa now forks `learn` + `markdown` correctly.
> But both fork halves silently produced nothing — "The model did not produce
> a final response." and no memory records in `memory.db`. This file documents
> the two independent downstream failures and the fixes.

---

## Prompt (user)

```text
write a markdown document of your entire response and use it to remember
the most important parts.
```

## Observed response (Lisa)

```text
The model did not produce a final response.
```

No markdown document. No memories in `memory.db`.

## Evidence (`artifacts/host.log`)

Two distinct failures, both on the local Ollama backend:

### Failure 1 — Distillation empty output (learn half)

```plaintext
23:41:23 kernel.operations.registry ERROR: operation 'synthesize_and_store_learning'
  failed: synthesize_and_store_learning distillation failed: empty distillation output
23:41:23 kernel.turn_runner WARNING: Tool 'synthesize_and_store_learning' failed
  in turn runner: ... RuntimeError: ... empty distillation output
```

Repeated at 23:41:29, 23:48:09, 23:48:24. The distillation completion
returned **HTTP 200 OK** but the content was empty. `synthesize_learning.py`
raised immediately on empty output — no retry.

### Failure 2 — Nil content type 400 (markdown/final half)

```plaintext
23:42:04 httpx INFO: POST .../v1/chat/completions "HTTP/1.1 400 Bad Request"
23:42:04 kernel.turn_runner WARNING: iteration-cap final completion failed:
  Error code: 400 - {'error': {'message': 'invalid message content type: <nil>',
  'type': 'invalid_request_error', ...}}
```

The iteration-cap final no-tools completion reuses the accumulated
`messages` list. During the tool loop, the turn runner appended assistant
messages with `tool_calls` and **no `content` key** (the model emitted tool
calls without accompanying text). Ollama's OpenAI-compatible endpoint
rejects `content: <nil>` on those messages with `invalid message content type: <nil>`.

## Root causes

### Failure 1 — No structured-output retry on empty distillation

`sdk/packs/learning/operations/memory/synthesize_learning.py` called
`ctx.complete_fn(..., response_format=DistilledLearning)` once and parsed
the result. When the local model ignored `response_format` and returned
empty content, `_parse_distilled_learning` raised `ValueError("empty
distillation output")` and the operation failed with no retry.

The structured-output contract (`/memories/repo/structured-output-contract.md`)
requires a retry with an explicit JSON-only instruction when the first
attempt fails to produce valid output. The distillation operation was
listed as "follows this contract" via `parse_structured_payload` but did
NOT use `run_structured_completion_with_retry` and had no inline retry —
so the contract was honored in parsing only, not in the completion loop.

### Failure 2 — Assistant messages with tool_calls but nil content

`kernel/turn_runner.py` line ~1551:

```python
assistant_msg: dict[str, Any] = {"role": "assistant"}
if getattr(completion, "content", None):
    assistant_msg["content"] = completion.content
tool_calls = getattr(completion, "tool_calls", None) or []
if tool_calls:
    assistant_msg["tool_calls"] = tool_calls
```

When `completion.content` is falsy (model emits tool calls without text),
`assistant_msg` has no `content` key. The OpenAI spec treats `content` as
optional for assistant tool-call messages, and the official OpenAI API
accepts it. But **Ollama's OpenAI-compatible endpoint rejects `content: <nil>`**
with a 400.

`normalize_openai_messages` (`kernel/provider_core.py`) — the single
chokepoint that builds every OpenAI-compatible request — did not repair
this. The bug is silent against the real OpenAI API and only surfaces on
strict local backends.

## Fixes (APPLIED 2026-07-17)

### Fix 1 — Distillation retry on empty/failed output

`sdk/packs/learning/operations/memory/synthesize_learning.py`:

When the first distillation completion fails (empty output, parse failure,
or any exception), re-issue the call with `json_only_retry_messages(...)`
and `response_format=None` before surfacing the failure. This adopts the
structured-output contract's retry pattern inline (the operation keeps its
custom bullet/prose fallbacks via `_parse_distilled_learning`, which
`run_structured_completion_with_retry` does not provide).

### Fix 2 — Default assistant content to empty string

`kernel/provider_core.py :: normalize_openai_messages`:

```python
if str(normalized_message.get("role", "") or "") == "assistant":
    if normalized_message.get("content") is None:
        normalized_message["content"] = ""
```

Every assistant message now carries a string `content` (empty when absent).
This is the single chokepoint — `build_openai_completion_kwargs` and
`build_openai_responses_input` both flow through it. Strict and lenient
backends both accept `""`.

## Key files

- `kernel/provider_core.py` — `normalize_openai_messages` (assistant content fix)
- `kernel/tests/test_provider_core.py` — 4 new `test_normalize_*` regression tests
- `sdk/packs/learning/operations/memory/synthesize_learning.py` — distillation retry
- `kernel/operations/structured_output.py` — `json_only_retry_messages` (used by fix)
- `artifacts/host.log` — incident evidence
- `docs/lisa-coding-mode-fork-failure-2026-07-17.md` — prior FSM fix (this incident's prerequisite)

## Lesson

A successful fork is not a successful turn. The FSM fix unblocked routing,
but both downstream mode operations still had to complete their work. Two
independent silent failures (empty distillation, nil content 400) combined
to produce a total no-op: no markdown, no memory. Each failure mode was
masked by the other — fixing only one would still lose half the turn.

**Structured-output contracts must cover the completion loop, not just the
parser.** A "follows the contract" call site that parses tolerantly but never
retries is not actually following the contract. The retry is the contract.

**OpenAI-compatible does not mean OpenAI-identical.** Lenient providers
accept `content: <nil>` on assistant tool-call messages; strict providers
(Ollama) reject it. `normalize_openai_messages` is the single place that
must enforce a shape acceptable to the strictest backend in the fleet.
Defaulting to `""` is safe everywhere.
