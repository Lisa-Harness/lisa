# Model Data Layers Specification

> **Status:** Proposed.
> **Scope:** Model discovery, provider catalog, and the active model settings.
> **Related plan:** [cache efficiency and measurement](./CACHE-EFFICIENCY.md).

## Goal

Separate provider-published facts from Lisa's measured facts and operator
choices. Remove the hand-maintained model registry: it duplicates provider
catalogs, can silently become stale, and has no provenance for its values.

## Invariants

1. A provider catalog row contains only fields the provider actually returned.
   Missing means unknown, not zero, false, or a guessed model-family value.
2. `known_models.yaml` is not a runtime fallback, override layer, or source of
   model-specific facts. Remove it and the `load_known_models` /
   `apply_known_override` path.
3. Operator choices survive discovery. Discovery never rewrites mode pins,
   router rules, explicit overrides, or other settings.
4. Every effective field has a source: `operator`, `measured`, `provider`, or
   `host_default`. Unknown is represented explicitly and is not reported as a
   provider claim.
5. Do not retain raw provider responses, credentials, or request content in
   the catalog.

## Data ownership and precedence

| Layer | Writer | Contents |
| --- | --- | --- |
| Operator settings | Operator via config/editor | `mode_models`, router rules, endpoint overrides, deliberate per-model overrides |
| Measured model data | Lisa's telemetry aggregation | Confidence-qualified observations with sample count and observation window |
| Provider catalog | `/models discover` | Normalized facts returned by provider discovery, with field-level provenance and observation time |
| Host defaults | Lisa configuration | General limits/policies used when no model-specific value is known |

Composition is field-by-field, not whole-row replacement. Authority is
field-specific:

- Operator choices control selection, routing, endpoints, and deliberate
  overrides. Input/output safety ceilings are tighten-only: an operator or
  mode may lower a provider ceiling, never raise it.
- A provider's published fact is authoritative for that fact. A conflicting
  measurement is recorded as a discrepancy for review; it does not silently
  replace the catalog value.
- Measured data fills fields providers do not publish, such as observed
  latency or cache behavior, and may supply a qualified estimate when a
  provider fact is absent.
- Host defaults apply only after the model-specific provider and measured
  sources are unknown.

For example, an operator may pin a model while the catalog supplies its
published context window and the measured layer supplies cache lifetime. The
effective model view reports the source of each value.

The active `models.yaml` becomes the operator-settings layer. The provider
catalog is a separate generated artifact using the data-root storage layout;
its exact path must be resolved through the existing storage/layout APIs, not
hard-coded into a host. A discovery run writes a proposal for the catalog.
Applying it validates and promotes the catalog without modifying operator
settings. Reloading models composes the layers and validates the effective
manifest.

## Work items

### 1. Remove the hand-maintained registry

- Delete `runtime/src/runtime/model_discovery/known_models.yaml` and the
  registry-loading/override helpers.
- Remove all comments and tests that describe the registry as a fallback or
  override. Search the model-discovery package and docs for stale references.
- Remove unsupported entries such as the current AWS rows from the discovery
  registry; do not make them appear discovered. Do not delete operator-
  configured models for a provider merely because Lisa has no discovery probe
  for it.
- Keep shared host defaults, such as a configured default context window,
  separate from model-specific facts. Do not copy such defaults into catalog
  rows.
- Derive price-based routing metadata only where a provider publishes prices.
  Leave price and latency unknown otherwise; do not infer them from model IDs.

### 2. Split discovery data from operator settings

- Define typed, versioned schemas for the provider catalog and operator
  settings. The existing active model manifest is the starting point for the
  settings schema; do not introduce a compatibility reader for the removed
  registry.
- Keep discovery's candidate catalog reviewable. `/models apply` promotes the
  candidate catalog only after schema and provider-reference validation.
- Compose an effective manifest at load time, using the field precedence above.
  Preserve the existing loader's hard validation for limits, providers,
  mode-model references, and router CEL.
- Keep catalog facts out of the operator-owned settings file. Do not rewrite
  settings merely to materialize the merged view.
- Migrate existing model files without guessing provenance. Preserve
  `mode_models`, router rules, endpoints, and clearly intentional overrides.
  Fields whose ownership cannot be established are surfaced in a migration
  report for operator review; they are not silently promoted to either a
  provider fact or permanent override.
- When a discovered model disappears or becomes retired, update the catalog
  and show the change. Do not silently erase operator pins; report settings
  that now reference unavailable models and require an explicit resolution.

### 3. Capture useful provider-published metadata

Normalize documented fields into the catalog. Store field-level source and
observation time; do not preserve the provider's raw JSON.

| Probe | Fields to retain when present | Unknowns that must remain unknown |
| --- | --- | --- |
| Anthropic | `max_input_tokens`, `max_tokens`, `line`, `lifecycle`, `deprecated_at`, model capabilities including manual/adaptive thinking and effort levels | Prices and cache lifetime when not supplied |
| Google | `inputTokenLimit`, `outputTokenLimit`, supported generation methods and documented thinking support | Prices and cache lifetime when not supplied |
| OpenRouter | Context limits, `top_provider.max_completion_tokens`, prompt/cache prices, supported parameters, modalities, expiration date, and any reported upstream/provider identity | Fields absent from its catalog response |
| Ollama | `/api/show` capabilities and authoritative model-info context length | Prices and output ceilings not exposed by its API |
| OpenAI | Model identifiers and other useful catalog metadata actually returned | Context/output limits, prices, and capabilities not returned by the models endpoint |
| Azure OpenAI | Deployment/model identity and capability fields actually returned by the configured endpoint | Limits and prices not returned by that endpoint |

Probe behavior:

- Preserve three-valued capability semantics: supported, unsupported, unknown.
  An absent field or a null capability object must not be interpreted as
  unsupported.
- Treat provider output limits as per-call ceilings, validate them against the
  effective context window, and reject or omit inconsistent provider data with
  a visible diagnostic. Never silently clamp a published value.
- Retain Anthropic's supported effort levels as capabilities; do not use an
  effort level unless the selected gateway can express it.
- Retain lifecycle/deprecation metadata. Do not offer retired models as
  available; surface deprecated models with their status.
- Pagination and provider errors remain explicit. A failed or partial probe
  must not replace the last good catalog with an empty one.
- A provider that publishes no limit does not gain a model-specific limit from
  an ID heuristic. The effective runtime may still use the host-level default.

## Acceptance criteria

- There is no runtime import or test dependency on `known_models.yaml`.
- A provider-published field wins over no hidden static override because no
  such override exists.
- Missing, null, and explicitly unsupported capabilities are distinguishable.
- Discover/apply/reload leaves operator settings byte-stable unless the
  operator explicitly edits them.
- Catalog proposals show additions, removals, lifecycle changes, and changed
  values with their provider and observation time.
- Tests cover each provider's positive, missing, null, malformed, retired, and
  limit-validation cases, plus field-by-field composition and migration.
- Runtime discovery and model-manifest tests pass; no provider credentials or
  live network access are required by unit tests.

## Implementation boundaries

- Runtime owns provider discovery and persistence integration.
- Kernel owns the effective manifest schema and validation.
- SDK gateways consume the effective model capabilities/limits; they do not
  read the discovery catalog directly.
- The operator-visible `/models` commands explain which layer supplied a
  value and distinguish “unknown” from “provider says no.”
