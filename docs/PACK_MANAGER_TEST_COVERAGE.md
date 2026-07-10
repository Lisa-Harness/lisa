# Pack Manager Test Coverage

This document summarizes the high-depth test coverage for runtime pack management and pack-contributed MCP server behavior.

## Test File

- [hosts/little_lisa/test_pack_manager.py](hosts/little_lisa/test_pack_manager.py)

## Covered Areas

## 1. Runtime install/uninstall behavior

- Install path rejected by trust policy does not mutate restart state.
- Successful install:
  - records pack provenance in lock file
  - marks runtime restart required
- Successful uninstall:
  - removes lock-file entry
  - marks runtime restart required

## 2. Trust policy enforcement

- Package refs are constrained by allowed prefixes.
- Git refs are constrained by host allowlist.
- Git refs must pin `@tag` / `@branch` / `@sha` when pin policy is enabled.
- Local folder refs are validated by `pyproject.toml` `project.name` and
  constrained by allowed prefixes.

## 3. Restart + resume signaling

- Restart request toggles runtime restart flags.
- Resume message is persisted and consumed exactly once.

## 4. Pack-contributed MCP server coexistence

- Two packs with the same declared `server_id` compile to distinct namespaced runtime ids:
  - `<pack_slug>.<server_id>`
- Collision is avoided without one-wins behavior.

## 5. Pack server exclusion policy

- Host exclusion policy supports multiple formats:
  - `<pack_slug>.<server_id>`
  - `<pack_slug>:<server_id>`
  - `<pack_slug>/<server_id>`
  - `<pack_slug>.*`
- Exclusions are applied before registration.

## How To Run

Targeted run:

```powershell
uv run pytest hosts/little_lisa/test_pack_manager.py -q
```

Focused run by behavior:

```powershell
uv run pytest hosts/little_lisa/test_pack_manager.py -k "namespace or exclusion or validate_pack_spec" -q
```

Broader little_lisa suite:

```powershell
uv run pytest hosts/little_lisa/test_little_lisa.py hosts/little_lisa/test_pack_manager.py -q
```

