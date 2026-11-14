# Model Data Layers — Completion Digest

> **Status:** Implemented (2026-11-12).
> **Scope:** Model discovery, provider catalog, and the active model settings.
> **Related plan:** [cache efficiency and measurement](./CACHE-EFFICIENCY.md).

## What shipped

The hand-maintained model registry (`known_models.yaml` and its
`load_known_models` / `apply_known_override` path) is removed. Discovery now
captures provider-published fields into a typed, versioned candidate catalog
(`runtime/src/runtime/model_discovery/catalog.py`) and validates per-call
ceilings against the context window — oversized values are omitted with a
diagnostic, never silently clamped. `/models discover` writes
`models.catalog.yaml` as a proposal only when every probe succeeds, so a
failed run never replaces the last good catalog. `/models apply` promotes the
candidate after schema and provider-reference validation and reports a
migration report without touching operator settings. `/models reload`
composes the layers in memory
(`runtime/src/runtime/model_discovery/compose.py`): operator > measured >
provider > host_default, field by field, with byte-stable operator rows,
tighten-only input/output ceilings, and explicit unknowns.

The measured model-data layer is **future work**: telemetry collection for
observed cache behavior, latency, and confidence-qualified estimates lands
with the cache-efficiency work and is not part of this digest.

## Invariants (still normative)

1. A provider catalog row contains only fields the provider actually returned.
   Missing means unknown, not zero, false, or a guessed model-family value.
2. There is no registry fallback: `known_models.yaml` and the
   `load_known_models` / `apply_known_override` path no longer exist.
3. Operator choices survive discovery. Discovery never rewrites mode pins,
   router rules, explicit overrides, or other settings.
4. Every effective field has a source: `operator`, `measured`, `provider`, or
   `host_default`. Unknown is represented explicitly and is not reported as a
   provider claim.
5. Do not retain raw provider responses, credentials, or request content in
   the catalog.

## Data ownership and precedence (still normative)

| Layer | Writer | Contents |
| --- | --- | --- |
| Operator settings | Operator via config/editor | `mode_models`, router rules, endpoint overrides, deliberate per-model overrides |
| Measured model data | Lisa's telemetry aggregation (future) | Confidence-qualified observations with sample count and observation window |
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
  provider fact is absent. (Collection lands with the cache-efficiency work.)
- Host defaults apply only after the model-specific provider and measured
  sources are unknown.

The active `models.yaml` is the operator-settings layer. The provider catalog
is a separate generated artifact using the data-root storage layout; its path
resolves through the existing storage/layout APIs, not hard-coded into a host.
A discovery run writes a proposal for the catalog. Applying it validates and
promotes the catalog without modifying operator settings. Reloading models
composes the layers and validates the effective manifest.

## Implementation boundaries (still normative)

- Runtime owns provider discovery and persistence integration.
- Kernel owns the effective manifest schema and validation.
- SDK gateways consume the effective model capabilities/limits; they do not
  read the discovery catalog directly.
- The operator-visible `/models` commands explain which layer supplied a
  value and distinguish "unknown" from "provider says no."
