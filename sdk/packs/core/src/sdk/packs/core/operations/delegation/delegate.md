# Delegate operation instruction

## Purpose

Delegate a sequential sub-turn to another mode. The parent turn stays
alive, waits for the child's structured result, and can use the child's
output to make further decisions. This is the sequential counterpart to
the `fork` operation (which fans out in parallel and returns all
children's results to the parent for synthesis).

## When to call

Call `delegate` when:

- You need another mode's capability and want to use its result in the
  same turn (e.g. delegate to `search` for evidence, then synthesize).
- You want to build a multi-step agent workflow where each step's output
  feeds the next.
- You need recursive delegation (a delegate child can itself delegate,
  bounded by `DELEGATE_MAX_DEPTH`).

Do NOT call `delegate` when:

- You just need to hand off to another mode and end your turn. Use
  `transition_stage` for that.
- You need parallel fan-out to multiple modes. Call the `fork` operation
  for that.

## Available Modes

The target `target_mode_id` must be a routable mode from the installed
catalog. Non-routable modes cannot be delegated to.

## Parameters

- `target_mode_id` (required): The routable mode to delegate to.
- `query` (required): The scoped input text for the child. This is the
  child's user query — not your full query. Scope it tightly.
- `stage_id` (optional): A stage within the target mode. Defaults to the
  mode's start stage.
- `max_tool_iterations` (optional): Per-child iteration cap. 0 means use
  the kernel default.
- `model_alias` (optional): Model override for the child. Empty means
  use the target mode's resolved model.

## Result

The operation returns a `DelegationResult` with:

- `status`: one of `ok`, `no_tools`, `timeout`, `error`, `cancelled`,
  `interrupted`, `model_yield`, `max_depth`.
- `output_text`: the child's response text.
- `tool_call_count` and `called_tool_names`: what the child actually did.
- `error_message`: present when `status` is not `ok`.
- `yield_action`: present when the child yielded (asked a question).

Inspect `status` before acting on the result. `no_tools` means the
child completed without invoking its action tools — its reported success
is unverified. `interrupted` means the child hit its iteration cap before
finishing. `max_depth` means the delegation chain exceeded the recursion
limit.