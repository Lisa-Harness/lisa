# Changelog

All notable changes to the Lisa Framework will be documented in this file.

## [0.1.0] — Unreleased

### Added

- `lisa-kernel`: runtime-neutral FSM engine, contracts, operation registry, and telemetry primitives.
- `lisa-sdk`: author-facing helper contracts for packs, extensions, workshop execution, and rendering.
- `lisa-ext-clarity`: public Clarity extension.
- `lisa-ext-cloudflare`: public Cloudflare extension.
- `lisa-ext-github`: public GitHub extension.
- `lisa-ext-stripe`: public Stripe extension.
- `lisa-pack-core`: baseline public modes, operations, and prompt assets.
- `lisa-pack-extras`: optional general-purpose modes and authoring operations.
- `little-lisa`: local terminal host CLI with `--doctor`, `--setup`, `--version`, and interactive REPL.
- Hermes installer scripts (`install.sh`, `install.ps1`) for one-command setup.
- CI matrix (Ubuntu, macOS, Windows × Python 3.11, 3.12).
- PyPI publish workflow on `v*` tags.

### Changed

- `little-lisa` turn logs now emit top-level resolved model fields and per-invocation token totals for both front-door planning and mode execution, making `turn.completed` JSONL entries readable without deep payload expansion.
