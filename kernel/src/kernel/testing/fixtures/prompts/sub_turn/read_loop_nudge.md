# Read-loop budget warning

READ-LOOP BUDGET WARNING. You have spent many iterations on read-only
calls without landing your findings. So far this turn: {retrieval_calls}
thread-archive reads, {retrieval_repeats} of them repeating calls you
already made. Each read re-bills the context it adds on every later
iteration. Synthesize what you have now instead of gathering more:

- Call your mode's write/document tool with the evidence collected so
  far, or
- Delegate the remaining gathering to a child — scope the query to the
  exact fact you still need and pass `max_tool_iterations` (a small
  cap, e.g. 8) so the child is bounded; prefer a read-only diagnostic
  mode when one is available over a mode that can mutate the
  environment, or
- Hand off to the mode that can land the work.

Do not issue more of the same read calls without a new, specific reason.