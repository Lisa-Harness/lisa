---
name: Debugging Workflow
applyTo: "**/*.py"
description: "Use when investigating Python bugs, stalls, or test failures; preserves evidence-first diagnosis and safe worktree handling."
---

# Debugging workflow

- After a large automated edit to dense or control-flow-critical code, inspect the targeted diff before continuing. Prefer small edits around handoffs, approvals, and lifecycle boundaries.
- Do not attribute a failure to the current change until its relationship to the touched files is established. Preserve unrelated worktree changes; use an isolated worktree or an explicitly named stash when comparison with `HEAD` is required.
- Diagnose opaque provider failures from runtime evidence before changing behavior. Capture finish reason, completion tokens, configured limits, and mode or stage context.
- A newly reachable path can reveal a latent failure in its next stage. Test and diagnose that downstream boundary independently rather than reverting the earlier fix by default.
- For context or resume symptoms, compare persisted turns and allocator decisions with the observed output before diagnosing lifecycle state.