# Mode taxonomy

Target-state design for Lisa's mode set: how the modes spread across
packs (core, personas, bash, coding) become a complementary surface
instead of overlapping escape hatches. Evidence base: the 307-turn
corpus from the October 2026 host logs
(`~/.lisa/logs/runtime.jsonl*`, `token-usage.jsonl`).

## Design principle

**Personas own ways of thinking; packs own surfaces of action; a mode is
one persona-shaped contract bound to one tool surface.** Stages of work
(design → execute → review) belong to workflow position or mode-internal
stages, not to whole lookalike modes.

The corpus showed the modes themselves behave once correctly entered;
the failures clustered at the seams — mixed-intent routing bouncing
between lookalike chat modes, and stuck turns teleporting to the one
permissive action mode and running away there.

## Observed failure fingerprints

| Failure | Turn evidence | Root cause |
| --- | --- | --- |
| Chain bounce | 7-intent query routed default → planner → researcher → coding_review (2026-10-25 18:45) | review/research/analyze intents split across four near-identical chat modes |
| Runaway delegate | coding_execute read-looped 40 iterations, delegated to a shell child, child burned 19 bash calls over 14 min to the wall-budget kill (2026-10-25 20:38–21:00) | shell mode is the universal stuck-hatch with effectively unbounded runway |
| Lost turn | child force-yielded → parent silent 300 s → StreamIdleTimeout | interrupted child does not inject a terminal result into the parent stream |
| Zero-write endings | 52/307 turns (17%) ended with no write tool call | three modes reference `okf_emit_concept`, whose owning pack was not installed — no sanctioned "emit my output" verb |

## Target mode set

| Mode | Pack | Kind | Contract |
| --- | --- | --- | --- |
| `chat` (was `default`) | core | chat | The only entry-point mode: answer directly, clarify, or hand off decisively. Absorbs `thinker` (deliberation is a prompt contract, not a tool surface) and `planner` (`plan_workflow` becomes a callable operation). |
| `notebook` (was `explainer`) | core | chat | Terminal artifact sink: every mode can land here with a synthesis document, OKF concept, or memory capture. Owns the emit-your-output verbs (`markdown_forge_document`, `okf_emit_concept`). |
| `coding` | coding | chat | One mode, three stages: `design → execute → review`. Stages keep their current tool allowlists; stage transitions are artifact-gated (brief → worktree → review findings). `/design`, `/exec`, `/review` remain entry aliases that pin the starting stage. |
| `console` (was `shell`) | bash | action | User-facing interactive terminal: long-running, posture-stamped, bare-host-capable. Router targets it only on explicit shell/terminal intents. A future powershell pack is the same mode shape with a different executor backend. |
| `probe` | bash | chat | Bounded read-only diagnostician, delegate-only (`routable: false`): a handful of sandboxed inspection commands, then findings back to the parent. Never mutates; never the escape hatch for stuck turns. |
| `researcher` | personas | chat | Web-gathering. Unchanged. |
| `headroom` | headroom | chat | Token-budget specialist. Unchanged — the exemplar of a narrow capability mode. |

### What dies

- `planner` as a mode — `plan_workflow` survives as an operation
  callable from `chat`; the kernel already drives handoffs from plan
  results.
- `thinker` as a mode — `/think` survives as an entry alias that primes
  a deliberation contract in `chat`.
- `coding_design` / `coding_execute` / `coding_review` as separate
  modes — folded into `coding`'s stage machine. Their current
  `satisfies_intents` lists collapse into one `code` home; the stage
  comes from workflow position instead of keyword overlap.

## Migration

### Phase 1 — content-only (no taxonomy change)

Landed:

- `probe` mode in the bash pack, delegate-only, with its catalog
  regression test. The read-loop nudge is now authored Markdown
  (`prompts/sub_turn/read_loop_nudge.md`) and describes the bounded
  delegate *shape* — tightly scoped query, small `max_tool_iterations`,
  prefer read-only targets — without naming any pack's mode: the
  framework recommends contracts, not installed-content vocabulary.
  Likewise the delegate instruction (`delegate.md`) describes when to
  pass a small iteration cap and prefer read-only targets. The
  recommendation of `probe` as the shell-workload complement belongs to
  the bash pack's own content (its mode descriptions), not to core.
- Host `.env` retuned from the corpus: nudge threshold 9, stage cap 30,
  child caps 40 — see the inline evidence comments there.

Not landed (previously listed here as landed):

- `lisa-pack-okf` is **not** installed in the runtime environment. The
  pack exists (`Lisa-Harness/lisa-pack-okf`, provides `okf_emit_concept`
  via the `lisa.extensions.content_roots` entry point), but the live
  runtime log still warns on every mode load that `coding_design`,
  `coding_review`, and `researcher` reference `okf_emit_concept`, which
  is not in the catalog (observed 2026-10-28T01:13:26Z, same continuous
  session). Install it (`/pack install lisa-pack-okf`) or remove the
  reference from those modes. Until then the zero-write regression gate
  below cannot pass in this environment.

Layering rule this phase established: **framework-owned prose (kernel
fixtures, core pack prompts, operation instructions) never names a
mode owned by another pack.** Modes advertise themselves; the framework
describes shapes and lets the installed catalog speak.

### Phase 2 — mode surgery

- Fold `thinker` into `chat` and `planner` into the `plan_workflow`
  operation. Keep `/think` and `/plan` as aliases.
- Merge the three coding modes into one `coding` mode with a stage
  machine; keep `/design`, `/exec`, `/review` aliases. Requires the
  kernel to accept stage-pinned entry aliases for a single mode.
- Rename `shell` → `console`; keep `/shell` as an alias.

### Phase 3 — harness hardening

- Interrupted delegate/fork children must inject a terminal chunk into
  the parent's stream instead of going silent (the StreamIdleTimeout
  turn-loss path).
- Per-mode child budget defaults: the delegate operation's
  `timeout_seconds` is the child wall budget; diagnostic-class children
  want far smaller budgets than coding children. Mode frontmatter gains a
  declarable child-budget shape.

## Regression gates

- **Runaway delegate**: a packstest scenario that drives a stuck coding
  turn to the read-loop nudge and asserts the recommended delegate
  shape is bounded (small `max_tool_iterations`) and read-only-first —
  and that framework-owned prose names no other pack's mode.
- **Probe contract**: the bash pack catalog test pins `routable: false`,
  `bash_run`-only tool surface, posture pre-warm, and the read-only
  contract prose.
- **Zero-write endings**: an eval that runs a design-stage turn to
  completion and asserts an `okf_emit_concept` (or notebook landing)
  receipt exists.
- **Turn survival**: an e2e scenario where a child is force-yielded at
  its wall budget and the parent still completes with the partial
  child result.
