# Safe-handoff prompt layout (design)

Cache-efficiency Workstream 3 — the design deliverable SPEC §6 required
before implementation: which prompt components are invariant, variable,
or provider-specific, and the expected cache effect for each handoff
type.

> **Status: implemented.** The implementation sketch at the end of this
> document shipped in ``9d76130d997d7e7bbaed7d49cdb4de87c35b1dbd``:
> ``ContextHooks.assemble_system_prompt_parts``, the factored
> ``SystemPromptSegments`` base (``from_parts`` / ``replace_mode_tail``),
> the S1-only handoff swap, and the runtime parts wiring. The
> ``Problem`` section below is historical — it describes the
> pre-implementation merged-base layout — and the ``Implementation
> sketch`` section is the plan that landed. Everything this document
> says about cache *effects* is an oracle-declared request shape plus a
> provider-behavior hypothesis: the ``CacheOracle`` trace proves the
> shape of the requests the kernel emits, and no assertion here
> measures what a provider actually served from cache. The honest
> before/after evidence is the SPEC's Success-criteria measurement
> window, which has not run; see ``Measurement protocol`` at the end.

## Problem

Every mid-turn mode handoff rebuilds the system prompt through
``assemble_system_prompt`` and swaps the **single** stable segment
wholesale (``SystemPromptSegments.replace_base`` plus a
``messages[0]`` rewrite, ``kernel/turn_runner/__init__.py`` in the
handoff transition). The stable base is the concatenation of:

1. the harness system prompt (content root, byte-stable across turns),
2. the session instruction (per conversation),
3. copilot customizations (per workspace/session),
4. the mode instruction, and
5. the stage instruction.

Only (4) and (5) actually change at a handoff, but the wire-visible
result is one system blob whose bytes change from position 0. On
breakpoint-based providers that re-declares the whole cached prefix;
on every provider it moves the first divergent byte of the canonical
shape to entry 0.

## Component inventory

| Component | Cache class | Changes when |
| --- | --- | --- |
| Harness system prompt | invariant (S0) | host upgrade/content-root edit |
| Session instruction | invariant per conversation (S0) | new conversation |
| Copilot customizations | invariant per session (S0) | workspace edits |
| Mode + stage instruction | mode-variable (S1) | mode handoff, stage advance |
| Routing hint | ephemeral (E) | per iteration |
| Handoff context | ephemeral (E) | per handoff |
| Stage instruction (dynamic copy) | ephemeral (E) | mid-turn stage change |
| Terminal directive / lifecycle briefing | ephemeral (E) | per resolution / iteration |
| Tool schemas | mode-authorized, never unioned | mode handoff |
| Pack-state message | transcript tail (post-fold placement) | owner snapshot changes |
| `prompt_cache_key` / `cache_control` | provider-specific emission | never (see below) |

Ordering rule already enforced by ``SystemPromptSegments.segments()``:
all stable segments precede all ephemeral segments in canonical name
order, regardless of mutation history.

## Provider-visible cacheable prefix and invalidation rules

- **OpenAI / Azure (OpenAI-shaped).** Implicit prefix caching over the
  serialized request prefix (tools included). ``prompt_cache_key`` is
  sent only to providers whose endpoints accept it
  (``sdk/providers/openai_gateway.py`` allowlist) and is stripped on an
  unknown-argument retry. The key is
  ``sha256(workspace:principal:conversation)[:32]`` and deliberately
  omits the mode, so affinity survives handoffs.
- **Anthropic.** Explicit ephemeral ``cache_control`` breakpoints. The
  gateway marks the **last stable segment block**
  (``_build_anthropic_system_blocks``), so the cacheable region is
  tools → system blocks → transcript. Cache-prefix order puts tools
  first: a tool-schema change at a handoff invalidates from the tools
  region onward, ahead of any system content. Breakpoints carry a
  short provider-documented TTL; an expired breakpoint is a
  ``cache_expired`` event, not a changed prefix. There is exactly
  **one** explicit breakpoint, on the final stable block — after
  factoring, that block is S1, not S0. Whether the provider reuses the
  S0 prefix behind that single breakpoint is provider behavior
  (incremental prefix reads behind the breakpoint); this design does
  not measure it, and the S0/S1 oracle trace alone does not establish
  it.
- **Ollama / local gateways.** No provider prompt cache of this kind;
  the oracle vocabulary still records shape transitions for parity.

## Target layout: factored stable base

Split the stable region into two stable segments:

- **S0 (invariant):** harness + session + copilot customizations.
  Written once per conversation at turn start; no mid-turn path rewrites it.
- **S1 (mode-variable):** mode + stage instruction. The handoff swaps
  **only this segment**; S0 bytes and the S0/S1 boundary are untouched.
- **E (ephemeral):** unchanged — dynamics already sit after the stable
  tail and after the Anthropic breakpoint.

``SystemPromptSegments`` already stores a *list* of stable segments, so
the kernel-side change is confined to the handoff path (replace the
last stable segment instead of the whole base) and to a two-part host
assembly hook; ``flatten()`` keeps byte-identical output. The runtime's
``assemble_system_prompt`` factors into ``invariant_part()`` +
``mode_part()``; a host that does not opt in keeps today's single
segment (S0+S1 merged) and today's behavior.

## Expected cache effect per handoff type

Declared in ``CacheOracle`` break-reason vocabulary. The trace tests
pin both columns' *oracle shapes* — the requests the kernel emits per
layout. Whether a provider rewards either shape with cache reads is
not established by these tests; that is the measured window's job (see
``Measurement protocol``).

| Transition | Today (merged base) | Target (S0/S1) |
| --- | --- | --- |
| Mid-turn handoff, same tools | `prefix_changed` @ 0 | `prefix_changed` @ first S1 entry; carried digest = S0 prefix |
| Mid-turn handoff, tools differ | `prefix_changed` @ 0 + tool digest change | `prefix_changed` @ first S1 entry + tool digest change; provider prefix ahead of S1 still breaks (accepted miss) |
| Mode re-entry, identical base | `identical_request` | `identical_request` |
| Routing-hint / briefing update | tail append / tail-only change | unchanged (already ephemeral) |
| History fold / compaction | digest moves at the rewritten boundary only; fold+compaction at one boundary never double-rewrites | unchanged |
| Provider or model change | `provider_or_model_changed` | unchanged |
| Gap beyond cache TTL | `cache_expired`, distinct from `prefix_changed` | unchanged |

**The honest structural miss.** Providers hash the tools region into
the cache prefix *before* the system region (Anthropic explicitly;
OpenAI includes tools in the prefix hash). A handoff that changes tool
authorization therefore invalidates the prefix ahead of S1 on current
providers no matter how the system prompt is factored. Factoring pays
where tool authorization is unchanged — same-authorization handoffs and
every same-mode turn boundary — and nowhere else. Per §6, we do not
union or reorder tools to widen the win.

## Equivalence argument

1. **Segment-blind gateways (OpenAI wire, Gemini).** ``flatten()`` joins
   stable segments S0, S1 then dynamics with ``\n\n``. Splitting one
   stable blob into S0 + S1 preserves the joined bytes exactly, so the
   prompt text and token count are identical for providers that never
   read segment metadata.
2. **Anthropic.** Block count grows by one (S0 and S1 become separate
   text blocks); each block's text is byte-identical to its slice of
   today's single block, and the breakpoint still sits at the end of the
   stable region. The change is additive wire metadata; instruction
   content is unchanged.
3. **Tool authorization.** No union, no widening, no reordering: the
   handoff rebuilds schemas through the same assembler the initial
   phase used. Any authorization change remains a declared, observable
   break (tool-schema digest), never masked by layout work.
4. **System-priority rules, policy injection, approvals.** S0 carries
   the harness rules verbatim; the handoff swaps S1 text only. Approval
   resumption re-enters the owning mode directly and does not traverse
   the base-swap path. Mode semantics and tool-call contracts are
   preserved by construction; the synthetic trace pins tool digests,
   and the real-path test
   (``test_real_handoff_tool_surface_matches_target_mode_phase1_surface``)
   asserts the post-handoff surface equals the target mode's Phase-1
   ``build_tool_surface`` output exactly — same schemas, same order,
   nothing from the entry mode's surface carried across.
5. **Empty-base edge.** Today a bare target mode may clear the whole
   base. In the target, only S1 may be empty; S0 is never empty after
   turn start (the ``from_base`` invariant), so an empty S1 renders as
   the S0 bytes plus the ephemeral tail.

## Instrumentation and trace

- ``kernel/cache_oracle.py`` (Workstream 4) is the evidence tool: the
  trace test scripts the merged-base layout and the factored S0/S1
  layout through the same ``CacheOracle`` and asserts the declared
  cause, first divergent entry, carried digest, and tool-schema digest
  for each transition above.
- Baseline (historical): a merged-base mid-turn handoff shows
  ``prefix_changed`` with first divergence **0** — the invariant head
  is lost with the mode tail. Still asserted against hand-built
  segment tuples so the regression stays visible.
- Target (implemented, ``9d76130``): the real-path test drives the
  production handoff through ``run_turn`` with
  ``ContextHooks.assemble_system_prompt_parts`` wired and a real
  ``transition_stage`` call, and asserts ``prefix_changed`` at the
  first S1 entry with carried digest equal to the S0-only shape
  digest. The stable entries are scripted into the oracle with fixed
  synthetic tools and an empty transcript, so this pins kernel request
  *shape* — it is not a provider cache measurement. Tool authorization
  is covered on the real path separately: the post-handoff surface
  must equal the target mode's Phase-1 surface exactly (see the
  equivalence argument), and the synthetic-oracle test keeps
  tool-digest changes observable.

## Implementation sketch (implemented)

Shipped in ``9d76130d997d7e7bbaed7d49cdb4de87c35b1dbd``; kept for the
design record. The hook landed on ``ContextHooks`` (the context
hooks family), not this sketch's earlier ``TurnHooks`` shorthand.

1. ``ContextHooks`` gains an optional ``assemble_system_prompt_parts``
   returning ``(invariant, mode)``; the legacy hook keeps working as
   the merged-base path.
2. ``SystemPromptSegments``: ``from_parts(invariant, mode)`` and
   ``replace_mode_tail(mode)``; ``replace_base`` stays for the legacy
   path.
3. Turn-runner handoff: when parts are available, swap the mode tail
   and rebuild ``messages[0]`` from ``flatten()`` (byte-identical).
4. Runtime: split ``assemble_system_prompt`` into the invariant and
   mode parts; wire ``assemble_system_prompt_parts`` in the hooks
   builder.
5. Flip the trace test's target-layout assertions from "simulated via
   segments" to "driven through the real handoff path".

## Measurement protocol (post-deployment; not yet run)

No measurement in this repository claims a provider cache result. The
honest sequence, per the SPEC's Success criteria:

1. Record the UTC deployment boundary — the moment a runtime carrying
   ``9d76130`` (or later) starts serving real turns. No deployment has
   been performed or verified as part of this change; until one is
   recorded, every before/after figure is structurally inconclusive.
2. Baseline window: replay the pre-deployment ledger through the
   aggregation readers —
   ``python -m runtime.cache_replay ~/.lisa/logs/token-usage.jsonl
   --as-of <deployment-UTC> --window-days 14 --json``. The replay CLI
   is ``runtime.cache_replay`` (stateless, read-only, checkpoint-free);
   ``runtime.cache_report`` is the ``/tokens`` rendering library, not a
   CLI. Verify the ledger exists and telemetry is enabled before
   trusting any window. Replay reproduces aggregates; it is not by
   itself a matched before/after analysis.
3. After the post window accumulates the SPEC-minimum 14 stable days,
   compare matched slices by provider, model, route, and workload:
   non-cache-read prompt tokens and canonical cost per successful
   comparable turn, with sample counts, coverage, and unknown share
   beside every figure, per-turn medians and totals. Regressions in
   success/quality or any authorization change veto the comparison.
4. Report underpowered or unattributed comparisons as **inconclusive**
   — never as savings. Missing or malformed telemetry is reported
   unavailable, never zero.
5. ``scripts/cache_report.py --since <UTC>`` labels (``first`` /
   ``mode`` / ``trim`` / ``node`` / ``unexplained``) are heuristics for
   triage; they are not causal evidence and do not replace the matched
   comparison.
6. Evaluating a second, dedicated S0 breakpoint on Anthropic is
   deliberately out of scope: it changes billing and wire shape,
   depends on provider-documented breakpoint limits and TTLs, and
   requires controlled provider measurements before any design change.
   Investigate separately; do not assume an S0 breakpoint pays.
