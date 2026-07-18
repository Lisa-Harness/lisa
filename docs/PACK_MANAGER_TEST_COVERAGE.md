# Pack Manager Test Coverage

This document summarizes test coverage for runtime pack management, pack-contributed MCP server behavior, and the installed-packs lock lifecycle. It is kept in sync with the actual test file.

## Test File

- [hosts/core/tests/test_pack_manager.py](../hosts/core/tests/test_pack_manager.py) — 21 tests

## Source Modules Under Test

| Module | Role |
| --- | --- |
| `hosts/core/commands.py` | `/pack install`, `/pack uninstall`, `/pack enable`, `/pack disable`, `/pack list`, trust policy validation |
| `hosts/core/content_roots.py` | Installed-packs lock load/write, bundled-pack seeding, SDK content root filtering, always-on resolution, pack name inference |
| `hosts/core/mcp_pool.py` | MCP config compilation, pack server exclusion, memory provider selection |

## Covered Areas

### 1. Runtime Install / Uninstall

- Install rejected by trust policy does not mutate restart state.
- Successful package install records provenance in lock file (source_kind=package) and marks restart required.
- Successful local-folder install records `source_kind=local` with the folder path as `source_ref`.
- Successful uninstall removes the lock-file entry, clears any provider selections pointing to the uninstalled pack, and marks restart required.

### 2. Trust Policy Enforcement

- Package refs are constrained by `LISA_PACK_ALLOWED_PREFIXES` (default: `lisa-pack-`).
- Git refs are constrained by `LISA_PACK_GIT_ALLOW_HOSTS` (default: `github.com`).
- Git refs must pin `@tag` / `@branch` / `@sha` when `LISA_PACK_REQUIRE_GIT_REF` is enabled (default: true).
- Local folder refs are validated by `pyproject.toml` `project.name` and constrained by allowed prefixes.
- Local folder refs with a missing `project.name` are rejected.

### 3. Restart + Resume Signaling

- `runtime.request_restart(reason)` sets the restart flag and reason.
- The resume message is stored in the `LISA_RESTART_RESUME_MESSAGE` env var and consumed on next launch.

### 4. Pack Enable / Disable

- Disabling an always-on pack (e.g. `lisa-pack-core`) is rejected.
- Enabling a bundled pack records `enabled=true` in the lock and marks restart required.
- Disabling a pack clears any `selected_capability_providers` entries pointing to it.

### 5. Pack Server Exclusion Policy

- Host exclusion policy supports multiple formats: `<pack_slug>.<server_id>`, `<pack_slug>:<server_id>`, `<pack_slug>/<server_id>`, `<pack_slug>.*`.
- Exclusions are applied before MCP config registration.

### 6. MCP Config Compilation — Namespacing

- Two packs with the same declared `server_id` compile to distinct namespaced runtime ids: `<pack_slug>.<server_id>`.
- Collision is avoided without one-wins behavior.
- Memory servers from packs are namespaced the same way (e.g. `memory.lisa-memory`, `memory.mem0`).

### 7. MCP Config Compilation — Memory Provider Selection

- **Session plan provider**: When the session plan declares a memory provider (e.g. `providers.capabilities.memory: lisa-pack-mem0`), the pack matching that provider is selected regardless of lock state.
- **Explicit lock selection**: When `selected_capability_providers.memory` is set in the lock, the pack matching that selection is used.
- **Auto-selection on install**: When a newly-installed pack contributes a memory capability and no provider is currently selected, it is auto-selected as the memory provider.

### 8. SDK Content Roots Filtering

- Disabled bundled packs are excluded from `sdk_content_roots()`.
- Enabled bundled packs are included.
- Always-on packs (e.g. `lisa-pack-core`) are always included regardless of lock state.
- Configured content roots (from `SDK_CONTENT_ROOTS` env var) are always included.

### 9. Bundled Pack Lock Seeding

- **First boot**: All discovered bundled packs are seeded into the lock with `enabled=true`, `source_kind=bundled`.
- **Idempotent**: Second seed call does not duplicate or mutate existing entries.
- **Preserves explicit disabled state**: An explicitly disabled bundled pack stays disabled across seeding.
- **Preserves runtime-installed entries**: Runtime-installed packs and `selected_capability_providers` survive seeding.
- **Empty discovery**: No discovered packs → no write, no lock file created.
- **Excludes `lisa-sdk`**: The `lisa-sdk` umbrella package is never seeded into the lock.

### 10. Pack List Rendering

- `lisa-sdk` is excluded from `/pack list` output even if a stale lock entry references it.
- `render_modes_list`, `render_operations_list`, and `render_tools_list` exist in `commands.py` but are not yet covered by dedicated tests.

## Known Issues

- `test_compile_mcp_configs_namespaces_pack_memory_servers` is recorded as a last-failed test in the pytest cache. This test validates that memory servers from pack manifests get namespaced correctly and that `runtime.memory_config_pack_slug` is set.

## How To Run

Targeted run:

```powershell
uv run pytest hosts/core/tests/test_pack_manager.py -q
```

Focused run by behavior:

```powershell
uv run pytest hosts/core/tests/test_pack_manager.py -k "namespace or exclusion or validate_pack_spec" -q
uv run pytest hosts/core/tests/test_pack_manager.py -k "seed or memory_provider or enable" -q
```

Broader core suite:

```powershell
uv run pytest hosts/core/tests/ -q
```

## Pre-Release TODO

- [ ] **Fix `test_compile_mcp_configs_namespaces_pack_memory_servers`** — investigate and resolve the recorded failure.
- [ ] **Add tests for `render_modes_list`, `render_operations_list`, `render_tools_list`** — these functions exist in `commands.py` but have no dedicated test coverage.
- [ ] **Add test for `/pack disable` success path** — `set_pack_enabled(enabled=False)` for a non-always-on pack is not directly tested (only the always-on rejection path is).
- [ ] **Add test for `run_pack_uninstall` clearing provider selections** — the uninstall test verifies lock entry removal but not the `_clear_selected_provider_for_package` side effect.
- [ ] **Add test for `_auto_select_capability_provider_on_install` when provider already set** — the auto-select test only covers the "unset" path.
- [ ] **Add test for `resolve_pack_name` with alias resolution** — pack alias resolution through `pack_alias_map` is not directly tested.
- [ ] **Add test for `pack_aliases` with `_overrides.yaml`** — pack aliases from override files are not tested.
- [ ] **Add test for `infer_pack_package_name` edge cases** — sdk/packs nesting, lisa_packs nesting, and fallback paths.
- [ ] **Add test for `load_pack_policy_dict`** — host manifest pack policy loading is not tested.
- [ ] **Add test for `pack_is_active`** — the combined enabled+always-on+installed check is not directly tested.
- [ ] **Add test for `runtime_installed_pack_names` and `runtime_enabled_pack_map`** — these lock-derived helpers are not directly tested.
- [ ] **Add test for `save_installed_packs_lock`** — the write path is exercised indirectly through install/uninstall/enable but has no dedicated round-trip test.
- [ ] **Add test for `discover_framework` / `setup_framework`** — the shared discovery cache and framework setup are not tested.
- [ ] **Add test for `compile_mcp_configs` with warm/cold split** — only the cold path is tested; warm-server compilation is not.
- [ ] **Add test for `handle_pack_command` dispatch** — the `/pack list|enable|disable|install|uninstall` command router in `commands.py` is not tested end-to-end.
