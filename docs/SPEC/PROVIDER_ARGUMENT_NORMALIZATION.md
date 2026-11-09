# SPEC: Provider argument normalization at the gateway boundary

> **Status:** SPEC ONLY — implementation and design decisions deferred;
> this document fixes the evidence, the seam, and the open questions for
> the design discussion. Nothing here is implemented.
> Updated 2026-11-09.

Mitigations shipped in the meantime (context, not the plan):

- kernel: [structured-output ladder](../../kernel/src/kernel/operations/structured_output.py)
  (`run_distillation_ladder` — one shared escalation for offline model
  calls);
- lisa-pack-coding: sentinel-coercing optional field types
  (`OptionalIdString` / `OptionalRevString`) and `coerce_update_edit`.

Those contain one provider's failure modes at two chosen seams. This spec
is about the layer below both of them.

## 1. The defect: model output is trusted at three unguarded boundaries

The Lisa agent's tool calls and structured completions arrive as
model-written JSON. Three boundaries accept that JSON, and each one has
absorbed a stream of per-shape patches instead of owning a normalization
contract:

| Boundary | Reads model JSON at | Patch history |
| --- | --- | --- |
| In-turn tool dispatch | `Parameters.model_validate(ctx.parameters)` in `kernel.tools.registry._validate_parameters` / `kernel.operations.registry._validate_parameters` | rejections → actionable error → coercion validators (`validate_artifact_id`, `validate_revision`, `coerce_update_edit`), string-encoded `edits` decoder, flat-record lift, ripgrep exit-code handling |
| Offline model calls | `parse_structured_payload` + the distillation ladder (`kernel.operations.structured_output`) | empty-content guard, fenced-JSON extraction, JSON-only re-ask, validation-feedback re-ask, bootstrap prefill, budget escalation |
| Gateway transport | `sdk.providers.*` argument sanitizers (`_parse_tool_arguments_object`, `_repair_openai_tool_arguments`, strict-schema `additionalProperties` enforcement) | 400-on-replay repair, non-finite constant rejection, recursion-depth catch |

Every patch is local and correct, and the incident count says the class is
not shrinking. The two distiller files alone absorbed 9 fix commits between
2026-10-01 and 2026-11-09; the coding pack's per-field coercion commit
chain (2026-10-30, 2026-11-09 twice) covered one sentinel at a time across
40 candidate fields.

### Evidence — recorded incident shapes, one log, nine days

`~/.lisa/logs/runtime.jsonl(+.1)`, level=warning, "invalid parameters",
2026-10-25 → 2026-11-09 (~252 events across 9 days, accelerating: 5/day in
late Oct, 39 on Nov 8, 85 on Nov 9 — heavier use, same per-call rate):

| Count | Tool | Field | Shape the model sent |
| --- | --- | --- | --- |
| 48 | coding_search_code | artifact_id | `'null'`/`''` string instead of JSON null |
| 44 | coding_record_fact | claim | over the 200-char cap |
| 35 | coding_read_file | artifact_id | `''` string |
| ~44 | coding_edit_files | edits.* | string-encoded array, flat shape, `content: ""`, null digest |
| 14 | coding_git_log | artifact_id | `''` string |
| 12 | coding_read_file | revision | `''` string |
| 9 | fetch_page | max_chars | out-of-range / wrong-typed |
| 7 | bash_run | commands | shape drift |
| 7 | coding_git_diff | base | `''` string |

Not one of the top shapes is an unknown-tool or permission error. They are
*encodings of the right intent*: null-as-string, cap-overflow, structurally
correct payloads that miss a schema bound. The kernel refused them; the
model often retried identically (the 2026-11-09 sentinel fix commit's
19-repeats finding), because a JSON Schema error message does not state the
*intent* contract, only the shape violation.

### Why the boundaries produce whack-a-mole

1. **No shared vocabulary for "optional sentinel".** Ten field types
   (`artifact_id`, `revision`, `base`, `head`, `content`, `claim`, …)
   mean "absent" in ten places. The tool type fix (OptionalIdString)
   works per-field per-pack; nothing teaches the *provider* that
   `""` ⇄ null.
2. **Schema-to-model gap.** `min_length`/`max_length` bounds are enforced
   but the failure message is positional (`string_too_long`), and most
   schemas do not state the cap in the description (fixed per-field in the
   pack, still per-field).
3. **Response-format is not universal.** Offline calls to
   non-reasoning-first hosts ignore `response_format`; the ladder (re-ask,
   re-aim, prefill) exists because the *provider layer* does not guarantee
   JSON.
4. **Tool-argument repair is lossy by design.** The gateway's honest repair
   (empty object) keeps history well-formed but discards the intent — the
   model then re-issues the same malformed call.

## 2. Goal and non-goals

**Goal — one normalization contract at the gateway boundary**, so that a
model-shaped JSON payload is corrected-or-rejected *above* every pack, and
pack schemas declare bounds without repeating coercion per field.

**Non-goals**

- No change to what the model *is allowed to call* (policy, mode allowlists).
- No silent "repair" of semantically ambiguous payloads (a wrong artifact id
  stays an error; only *encodings* of absent/null/caps are normalized).
- No per-provider fork of the contract: one spec, provider adapters
  implement it.
- No relaxation of security boundaries (path traversal, SSRF, sandbox
  floors) — normalization runs before those checks and can only *tighten*
  what reaches them.

## 3. Proposed design space (to finalize in review)

Three candidate layers; the discussion should pick one (or a hybrid).

### Option A — normalize at the gateway (`sdk.providers.*`)

The gateway owns `arguments` for tool calls and `content` for structured
completions. Add a `PayloadNormalizer` protocol:

- `normalize_tool_arguments(raw: Any, schema: ToolSchema | None) -> dict`
  — canonicalizes encodings: `"null"`/`""` → null for optional fields
  *when the schema marks them optional*, string-encoded arrays → list,
  flat-record shapes → current record contract (the pack's
  `_lift_flat_string_edits` logic moves here and generalizes).
- `normalize_structured(raw: str, schema: type[BaseModel]) -> str` —
  fence stripping, prose extraction (already `extract_json_object`),
  + the ladder's *first* rung equivalents.

Pros: one implementation covers every pack and both boundaries; schema-aware
(the gateway already receives tool defs; structured calls carry the pydantic
model). Cons: gateways are transport adapters by contract
([BOUNDARIES.md](../BOUNDARIES.md)) — putting semantic normalization there
blurs the transport/domain line; the kernel registry already validates.

### Option B — normalize in the kernel dispatch (`kernel.tools.registry`)

Extend `_validate_parameters` with a schema-driven pre-normalization pass:
for each `Parameters` model, an automatically-derived normalizer built from
the schema itself (optional fields coerce null-sentinels; bounded strings
truncate-or-reject by declared policy; nested lists decode string-encoded
arrays). Pack schemas declare a marker mixin (`SentinelSafeModel`) to opt in
per tool.

Pros: stays inside the kernel boundary; schema-driven — zero per-field
authoring, applies to every pack at once; the rejections that remain are
genuinely semantic. Cons: still after-the-fact (the model already spent the
call); doesn't fix the *replayed-history 400* (that one does need gateway
work); needs careful design so a normalizing pass never masks a real error.

### Option C — schema-first authoring (prevent, not normalize)

Make bounds and null-contracts visible in the *wire schema* so the model
rarely errs: every tool schema emission includes per-field `description`
constraints (the pack fix), plus JSON Schema `default: null`, and an
`x-absent-encoding: null` convention the dispatch layer reads.

Pros: attacks the cause (the model never learns `""` means null); no new
runtime layer. Cons: depends on provider adherence — glm-5.3 ignored field
descriptions repeatedly; description-following is exactly the capability
that was absent in every incident. Cheap, but insufficient alone.

### Hybrid recommendation (draft for discussion)

C is table stakes and already partially landed. A is the *complete* fix for
the transport-side shape (replay-400, fence stripping) where the gateway is
already doing normalization work. B is the broad wins for parameter
encodings. Concretely:

1. **B now** — `SentinelSafeModel` + schema-derived normalizer in
   `kernel.tools.registry._validate_parameters`; migrate the coding pack's
   `OptionalIdString`/`OptionalRevString`/`coerce_update_edit` onto it as
   the first consumer (deletes the pack's coercion helpers again, third
   time these moved — they are seed corn, not product).
2. **A next** — move/extend `_parse_tool_arguments_object` /
   `_repair_openai_tool_arguments` into a shared `PayloadNormalizer`
   protocol consumed by every gateway; add the missing
   string-encoded-array decode there so replay-400 repairs stop losing
   intent.
3. **Offline structured calls** — keep the ladder (it models a *conversation*,
   which normalization cannot replace), but make rung 1 the normalizer's
   job too.

## 4. Open questions for the design review

1. **Is the kernel the right home for B** (`kernel.tools.registry`), or does
   a pre-dispatch `PayloadNormalizer` seam belong in `kernel.seams` so hosts
   can extend it without kernel changes?
2. **Opt-in vs opt-in-by-default.** `SentinelSafeModel` opt-in keeps
   blast radius zero but re-creates the whack-a-mole loop for packs that
   forget. Default-on with a strict-escape hatch (`StrictModel`) inverts
   it — which failure mode do we prefer?
3. **Truncation policy for bounded strings.** When a value exceeds
   `max_length`: reject (current), truncate (silent data loss in memory
   records?), or re-ask offline? The 44 `coding_record_fact` claim
   overflows suggest reject-with-feedback (the ladder's rung 3) is right
   for offline calls; for in-turn calls, reject with an intent-naming
   message (the existing pattern that *did* eventually recover the model
   once messages named the cap).
4. **Does normalization log?** Every coercion today lands a warning line so
   mistypes stay visible. Normalization at scale needs the same contract,
   or forensics loses the provider-signal (the warnings are part of how
   this entire incident family was diagnosed).
5. **Telemetry gap.** `tool.telemetry` records rejections (landed
   2026-11-09) but does not record *normalizations*. Should a normalized
   dispatch stamp the row (`normalized_fields: [...]`) so the provider's
   error rate stays measurable per shape?
6. **Relationship to the Go broker** (`lisa-go`): the broker proxies
   provider traffic — is normalization in scope there, or does it stay
   Python-side?
7. **Test strategy**: the pack's field-type tests are the model; the
   normalizer needs per-shape fixtures drawn from the incident log (the
   9 shapes above are the fixture list) — conformance tests per gateway?
8. **Sequencing**: which lands first — B, or the coding pack adoption of
   it? (Commit `5f87fe3` moved coercions into field types; option B
   supersedes that placement.)

## 5. Success criteria

- The sentinel/encoding incident class (the 9 shapes) produces zero
  rejections on a replayed incident log: every recorded malformed payload,
  replayed through the normalizer, either dispatches identically to the
  intent or is rejected with an intent-naming message on a *semantic* (not
  encoding) failure.
- No pack contains per-field coercion validators; packs declare bounds in
  schemas, the normalizer owns encodings.
- `tool.telemetry` carries normalization provenance such that a
  provider-quality regression is visible in per-shape normalization counts
  without reading logs.
- The distillation ladder shrinks to conversation-level corrections only
  (no argument-encoding rungs); gateway replay-400 repairs stop
  discarding intent.
