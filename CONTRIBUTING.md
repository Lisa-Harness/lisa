# Contributing to Lisa Framework

Thank you for your interest in contributing.

## Getting Started

1. Fork the repo and clone it locally.
2. Install dependencies: `uv sync --extra dev`
3. Run the verification: `uv run little-lisa verify`
4. Make your changes on a feature branch.

## Development Loop

```powershell
# Lint
uv run ruff check kernel lisa_sdk sdk_extensions sdk_packs examples

# Type check
uv run mypy kernel lisa_sdk

# Run tests (no live API calls)
uv run pytest -m "not live" --tb=short -v

# Run verification smoke test
uv run little-lisa verify
```

## Pull Request Checklist

- [ ] Code passes `ruff check` and `mypy` with no new errors.
- [ ] All non-live tests pass (`pytest -m "not live"`).
- [ ] `little-lisa verify` exits 0.
- [ ] New public APIs have docstrings.
- [ ] Logging best practices followed (module-level logger and appropriate debug/info/warning/error statements).
- [ ] Breaking changes are noted in the PR description.

## Package Layout

| Package | Path | Description |
| --- | --- | --- |
| `lisa-kernel` | `kernel/` | Runtime-neutral contracts, FSM engine, operation registry |
| `lisa-sdk` | `lisa_sdk/` | Author-facing helpers for packs and extensions |
| `lisa-ext-*` | `sdk_extensions/` | Public extensions (Clarity, Cloudflare, GitHub, Stripe) |
| `lisa-pack-*` | `sdk_packs/` | Public mode/operation packs |
| `little-lisa` | `examples/little_lisa/` | Local terminal host CLI |

## Commit Conventions

- `feat:` — new behavior
- `fix:` — bug fix
- `imp:` — non-breaking improvement
- `docs:` — documentation only
- `infra:` — CI/CD, build, or tooling

## Questions?

Open a discussion or issue on GitHub.
