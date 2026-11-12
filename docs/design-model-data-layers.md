# Implementation Design: Model-Data-Layers

## Context and Goals

The `MODEL-DATA-LAYERS` spec is Proposed and replaces the current hardcoded model registry with a typed, reviewable provider catalog and a separate operator settings file. The implementation must remove `runtime/src/runtime/model_discovery/classify.py` registry logic (lines 199–260 for `load_known_models` and `apply_known_override`; line 60 for `_KNOWN_MODELS_PATH`), update all probe call sites, and introduce a catalog/settings split with strict validation.

The design is a hard cutover: no compatibility reader for the old registry format. This avoids silently supporting two ownership models and forces migration decisions to be explicit. The new catalog is `models.proposed.yaml`, maintained as a candidate provider catalog with field-level source attribution and observed timestamps. Operator configuration remains in `models.yaml` and is never mutated by `/models apply`.

The resulting effective model manifest is composed at reload time from four layers, with explicit precedence and a report for unknown provenance in existing files.

## Target Data Model and Catalog/Settings Split

The catalog file `models.proposed.yaml` becomes the reviewable candidate provider catalog. It uses a typed, versioned schema. Every field that may originate from measurement, provider defaults, host defaults, or operator overrides carries:

- `source`: one of `operator`, `measured`, `provider`, or `host_default`
- `observed_at`: timestamp for measured/provider fields
- `value`: the typed field value
- for capabilities, a three-valued state: `supported`, `unsupported`, or `unknown`

The key semantic rule is that `unknown` is distinct from `unsupported`. An omitted capability or a provider that does not publish a field remains `unknown`; only explicit provider documentation may mark a capability `unsupported`. The code must not infer IDs, aliases, or capabilities from identifier shape.

At runtime, the effective manifest is composed field-by-field at reload with precedence:

1. `operator`
2. `measured`
3. `provider`
4. `host_default`

Unknown values are explicit fields and never fall through silently to a lower layer. IO ceilings are tighten-only: a stricter measured ceiling may lower an effective ceiling, but no layer may raise an operator-set ceiling.

`/models apply` uses the existing import helpers in `runtime/src/runtime/commands/__init__.py`:

- `_import_write_manifest_yaml` at line 299
- `_import_proposal_manifest_path` at line 325
- `_import_load_model_manifest` at line 313

Apply promotes the proposed catalog only after schema validation and provider-reference validation. It does not touch the operator `models.yaml`. The kernel-side validator in `kernel/src/kernel/model_manifest.py` `load_model_manifest` (lines 187 and 206) continues to enforce hard validation: provider limits, provider references, `mode_models` references, and router CEL expressions.

## Registry Removal and Cutover

Slice 1 removes the registry entirely. Delete the load/override functions and path constant from `runtime/src/runtime/model_discovery/classify.py`:

- `_KNOWN_MODELS_PATH` around line 60
- `load_known_models` and `apply_known_override` from lines 199–260

Remove all import and call sites from probes:

- `probes/anthropic.py` lines 16 and 195
- `probes/azure.py` lines 17 and 234
- `probes/google.py` lines 23 and 264
- `probes/ollama.py` lines 17 and 288
- `probes/openai.py` lines 12 and 135
- `probes/openrouter.py` lines 32 and 242

Update stale docstrings that mention the registry or old override behavior:

- `types.py` line 54
- `probes/base.py` line 131
- `openrouter.py` line 9
- `ollama.py` lines 154, 186, 257, and 281

No compatibility reader is introduced. Any old registry file path is ignored or reported as legacy, but it is not parsed.

## Provider Capture Matrix and Probe Behavior

The catalog schema is populated only by the provider-specific fields named in the spec table.

Anthropic:

- `max_input_tokens`
- `max_tokens`
- `lifecycle`
- `deprecated_at`
- effort levels

Google:

- `inputTokenLimit`
- `outputTokenLimit`
- generation methods
- thinking configuration

OpenRouter:

- context limits
- `top_provider.max_completion_tokens`
- prices
- supported parameters
- modalities
- expiration
- upstream identity

Ollama:

- `/api/show` capabilities
- model-info context length

OpenAI:

- identifiers
- actually-returned metadata from the API response

Azure:

- deployment identity
- capability fields as returned for the deployment

For every measured field, the capture layer records the source provider and the observation time. Raw provider JSON is never retained; it is parsed into typed fields and discarded. Credentials are handled only by the probe transport and are never copied into the catalog or manifest.

Probes must fail cleanly. A failed or partial probe cannot replace the last good catalog. The probe must complete extraction and validate the candidate catalog as a whole before an atomic swap. If any required field is inconsistent or missing in a way that would invalidate the target model, the candidate is rejected and the previous catalog remains active.

## Limit Validation and Failure Semantics

Output limits are treated as per-call ceilings and checked against the effective context window. A limit greater than the effective context window is inconsistent. The implementation must reject or omit inconsistent data with a visible diagnostic. It must never silently clamp a value.

For example, if a provider reports `max_tokens` larger than the composed context window, the catalog entry is rejected for that field with a diagnostic naming the provider, model, raw value, and effective context window. If the probe cannot determine whether the field is inconsistent, the field remains `unknown` and the model is allowed with the limit omitted rather than guessed.

Silent clamping is explicitly forbidden. Validation occurs before apply and at load time. The kernel hard validator remains the final authoritative check because it already enforces model limits and provider references.

## Migration Report and Operator Pin Handling

Existing model files predate the ownership metadata. During migration, the system surfaces fields whose provenance cannot be established. It does not guess provenance. Fields found only in old files are reported as `unestablishable ownership`; they are not automatically assigned to a source layer.

Retired models remain visible in lifecycle comparisons. The effective manifest shows removals and lifecycle changes rather than hiding them. If an operator settings file pins a model that is no longer available, the reload emits a report entry naming the operator pin and the unavailable model. The pin is not erased: operator configuration is preserved, and the manifest reports the mismatch. This prevents surprise deletion and keeps rollback possible.

The migration report is generated on demand and at apply/reload when legacy fields are encountered. It is written as a readable report, not as mutation of the operator file.

## Landable Implementation Slices

The work is split into three landable slices.

**Slice 1 — Registry removal and probe/test cleanup.** Delete registry code and call sites; remove stale docstrings; delete registry-focused tests in `model_discovery/tests/test_enrichment.py` lines 468–531, `test_discovery_metadata.py` line 162, and `test_openrouter_probe.py` line 132. Update imports so probes no longer depend on `classify.py`. All existing network-free tests must pass.

**Slice 2 — Per-provider field capture into the typed catalog schema plus limit validation.** Add the catalog schema, source/observed metadata, and three-valued capability representation. Implement the capture matrix for Anthropic, Google, OpenRouter, Ollama, OpenAI, and Azure. Add limit validation against the effective context window with reject/omit diagnostics. Add atomic catalog swap on successful probe. No raw provider JSON retention.

**Slice 3 — Apply/reload composition and migration report.** Wire `/models apply` to promote the proposed catalog after schema and provider-reference validation. Implement field-by-field composition with precedence `operator > measured > provider > host_default`; unknown remains explicit; IO ceilings tighten only. Add migration report for unestablishable ownership and operator pins to unavailable models.

## Test Plan and Files to Extend or Create

Existing tests to extend:

- `runtime/src/runtime/model_discovery/tests/test_enrichment.py`
- `runtime/src/runtime/model_discovery/tests/test_discovery_metadata.py`
- `runtime/src/runtime/model_discovery/tests/test_openrouter_probe.py`

These currently assert registry behavior. Replace registry assertions with catalog/schema assertions. Remove regression blocks tied to `load_known_models` and `_KNOWN_MODELS_PATH`.

New network-free unit test files:

- `runtime/src/runtime/model_discovery/tests/test_catalog_schema.py` — schema typing, field sources, observed timestamps, three-valued capabilities, versioning
- `runtime/src/runtime/model_discovery/tests/test_probe_limit_validation.py` — per-call ceiling vs effective context window, reject/omit behavior, no silent clamp, failed/partial probe atomicity
- `runtime/src/runtime/model_discovery/tests/test_apply_promotion.py` — `/models apply` validates schema and provider references, promotes candidate, and does not mutate operator `models.yaml`
- `runtime/src/runtime/model_discovery/tests/test_reload_composition.py` — field-by-field precedence, explicit unknown, tighten-only IO ceilings, retired lifecycle display
- `runtime/src/runtime/model_discovery/tests/test_migration_report.py` — unestablishable ownership surface, operator pin reporting without erasure

All new tests are network-free and use small typed fakes. Test files must not import each other; duplicate small fakes where necessary.

## Alternatives Considered

- **Compatibility reader for the old registry format.** Rejected. A dual reader doubles ownership paths and allows silent fallback during migration. The hard cutover forces explicit migration and keeps the runtime simpler.
- **Merged YAML registry with runtime defaults.** Rejected. Mixing operator settings with measured/provider data makes review and promotion ambiguous. The catalog/settings split separates reviewable provider data from local operator decisions.
- **Permissive null fallback for unknown capabilities.** Rejected. Treating `null` as `unsupported` hides provider gaps and causes false negatives. Three-valued state makes unknown explicit.
- **Silent clamp for oversized limits.** Rejected. Silent clamping removes evidence needed to correct provider data or operator pins. Visible rejection or omission is required.
- **Identifier heuristics for unpublished fields.** Rejected. Guessing IDs or aliases from model names is unreliable and violates provenance rules. Unpublished fields remain `unknown`.
- **Retain raw provider JSON for later re-extraction.** Rejected. Storing raw payloads risks credential leakage and duplicates unstructured data. Only typed fields are retained.

## Acceptance Checks and Repo Validation

Acceptance checks are mapped to spec criteria:

- Registry removal: no references to `load_known_models`, `apply_known_override`, or `_KNOWN_MODELS_PATH` remain in probes or runtime model discovery.
- Catalog/settings split: `/models apply` promotes only `models.proposed.yaml`; operator `models.yaml` is unchanged.
- Typed schema: catalog has version, field sources, observed timestamps, and three-valued capabilities.
- Composition precedence: effective manifest follows `operator > measured > provider > host_default`; unknown is explicit.
- IO ceilings: measured/provider ceilings can tighten but never loosen an operator ceiling.
- Limit validation: oversized output limits are rejected or omitted with a diagnostic; no silent clamp.
- Probe safety: failed or partial probes never replace the last good catalog.
- Privacy: no raw provider JSON or credentials in catalog or manifest.
- Migration: unestablishable ownership is reported, not guessed; pinned unavailable models are reported, not erased.

Repository validation is the standard seven-check set:

1. `autofix`
2. `format`
3. `lint`
4. `policy`
5. `types`
6. `dead`
7. `test`

The three slices must each pass all seven checks independently before landing.

<!-- generated by lisa · provider: openrouter · model: ~deepseek/deepseek-pro-latest · 2026-11-12 -->
