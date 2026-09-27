---
name: Pack Testing
applyTo: "packstest/**,sdk/packs/**,../lisa-packs*/**"
description: "Use when testing or debugging content packs; mandates the packstest in-process harness as the first verification loop."
---

# Pack testing

- Verify pack changes with the `packstest` harness **before** any installed-package route: `validate_pack(<pack-path>)` for structure, `run_tool`/`run_scenario` for behavior, `git_repo` for worktree flows. See `packstest/README.md` for the API.
- Eval files live in the pack under `tests/evals/` and carry the `eval` marker (register it in the pack's `pyproject.toml`). The marker contract is deterministic: no model calls, no I/O beyond throwaway tmp dirs. Scripted tool-call plans stand in for the model.
- Pack tests must not import `runtime` or any private runtime module — use `packstest.store.MemoryWorkProductStore` where pack code expects a `WorkProductStore`. Packs import `kernel` and `sdk` only.
- The harness calls the real SDK catalog scanners; do not add parallel path-inference or provider-id logic to it. Validate tool `effect` values via `kernel.approval.EffectKind`, not hardcoded sets.
- When an eval fails, fix the pack (or, if the harness itself mis-modeled a production contract, fix the harness and add a self-test) — never loosen the assertion to force green.