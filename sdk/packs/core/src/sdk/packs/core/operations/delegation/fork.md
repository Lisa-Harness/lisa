# Fork operation instruction

## Purpose

Fan out a parallel sub-turn to multiple modes. The parent turn stays
alive, waits for all children to finish, and synthesizes across their
structured results. This is the parallel counterpart to `delegate`
(which runs a single sequential child and keeps the parent alive to use
its output).

## When to call

Call `fork` when:

- The request has two or more independent concurrent goals that no
  single mode covers (e.g. `fork` to `remember` + `markdown` so each
  child handles its own intent in parallel).
- The goals are independent — no child needs another child's output as
  input. If the goals are sequential (one feeds the next), use `delegate`
  or `transition_stage` chaining instead.

Do NOT call `fork` when:

- You need a single sequential sub-turn whose result you use in the
  same turn. Use `delegate` for that.
- You just need to hand off to another mode and end your turn. Use
  `transition_stage` for that.
- The goals are sequential ("search ... then code ... then write"). A
  fork runs children in parallel with the whole query, so none can use
  another's output. Use `transition_stage` to the first target and let
  post-handoff hints chain the rest.

## Available Modes

Each `targets[].mode_id` must be a routable mode from the installed
catalog. Non-routable modes cannot be forked to.

## Parameters

- `targets` (required): A list of fork targets. At least one is
  required. Each entry has:
  - `mode_id` (required): The routable mode to fan out to.
  - `query` (optional): The scoped input text for this child. This is
    the child's user query — not your full query. Scope it tightly per
    child. When empty, the parent turn's query is used.
  - `stage_id` (optional): A stage within the target mode. Defaults to
    the mode's start stage.
- `max_tool_iterations` (optional): Per-child iteration cap applied to
  every child. 0 means use the kernel default.
- `model_alias` (optional): Model override applied to every child turn.
  Empty means use each target mode's resolved model.

## Result

The operation returns a `ForkResult` with:

- `succeeded`: the count of children whose `status` is `ok`.
- `children`: a list of `ForkChildResult`, one per target, in order. Each
  has:
  - `status`: one of `ok`, `no_tools`, `timeout`, `error`, `cancelled`,
    `interrupted`, `model_yield`, `blocked`.
  - `output_text`: the child's response text.
  - `called_tool_names`: the tools the child actually invoked.
  - `error_message`: present when `status` is not `ok`.
  - `yield_action`: present when the child yielded (asked a question).

Inspect each child's `status` before synthesizing. `no_tools` means the
child completed without invoking its action tools — its reported success
is unverified. `interrupted` means the child hit its iteration cap before
finishing. `blocked` means the capability gate declined to escalate. A
failed child's work is lost (the framework is turn-by-turn with no
background scheduler); surface the failure honestly and suggest
re-issuing the request rather than claiming deferred completion.