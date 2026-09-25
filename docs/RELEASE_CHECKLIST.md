# Lisa Harness Release Readiness Checklist

This checklist governs the release readiness of two repositories:

- **lisa** — `github.com/Lisa-Harness/lisa` (private core framework)
- **lisa-packs** — `github.com/Lisa-Harness/lisa-packs` (public pack registry)

Host distribution release (lisa-harness-cli, lisa-host meta-package) is governed
by `../lisa-hosts/docs/RELEASE_CHECKLIST.md`.

Each section applies to one or both repositories as indicated. Items are marked with checkboxes for tracking. Complete all items before tagging a release.

______________________________________________________________________

## 1. Repository Hygiene

### 1.1 lisa (core framework)

- [ ] Repository visibility confirmed (currently private; decide whether release requires public visibility or a public mirror)
- [ ] `README.md` exists and accurately describes the framework, installation, and usage
- [ ] `LICENSE` file present and matches intended license
- [ ] `CONTRIBUTING.md` present with contribution guidelines
- [ ] `CODE_OF_CONDUCT.md` present
- [ ] `.gitignore` reviewed for stale or overly broad entries
- [ ] No secrets, API keys, or `.env` files committed (scan history if necessary)
- [ ] Docs reference the `lisa` CLI (shipped by `lisa-harness-cli`) — no `little-lisa` references remain
- [ ] Branch protection enabled on `main`
- [ ] Default branch is `main`
- [ ] Stale branches pruned or archived
- [ ] Tagging convention agreed (e.g., `vMAJOR.MINOR.PATCH`)
- [ ] GitHub Actions / CI workflows present and passing on `main`
- [ ] Issue and PR templates configured
- [ ] Repository description and topics set in GitHub settings

### 1.2 lisa-packs

- [ ] `README.md` accurately describes the pack registry, pack lifecycle, and how to add packs
- [ ] `CONTRIBUTING.md` present and aligned with the promotion lifecycle (private standalone → packs-wip → lisa-packs)
- [ ] `docs/PROMOTION_GUIDE.md` reviewed for accuracy and completeness
- [ ] `.gitmodules` reviewed — all 10 submodule URLs resolve and point to `github.com/Lisa-Harness/lisa-pack-<name>` (brave, cloudservers, letta, markdown, mem0, okf, personas, search, serper, tavily)
- [ ] All submodule SHAs pinned to intended commits (not floating `HEAD`)
- [ ] `.cspell` dictionary reviewed; no false positives blocking CI
- [ ] `.github` workflows reviewed and passing
- [ ] `.vscode` settings reviewed for committed user-specific paths
- [ ] `.gitignore` reviewed
- [ ] No secrets or sensitive content committed
- [ ] Branch protection enabled on `main`
- [ ] Only 3 commits on `main` — confirm this is intentional and not a sign of lost history
- [ ] 0 issues / 0 PRs / 0 stars / 0 forks — confirm this is expected for a pre-release state
- [ ] Issue and PR templates configured
- [ ] Repository description and topics set

______________________________________________________________________

## 2. Documentation Completeness

### 2.1 lisa (core framework)

- [ ] `/docs` directory reviewed
- [ ] GAP/SWOT analysis documents reviewed and marked as current or superseded:
  - [ ] `docs/research/architecture_gap_swot.md` — **superseded point-in-time snapshot** (2026-07-25); lists shipped capabilities as missing. Marked superseded with a banner. Either rewrite against the current architecture or keep as history and point the checklist at a replacement.
  - [ ] `docs/research/agent_framework_swot.md`, `docs/research/agent_framework_analysis.md` — market-level snapshots, partly speculative; marked as point-in-time.
  - [ ] `docs/research/release-readiness-assessment.md` — external snapshot whose fork-unfixed claim is contradicted by `docs/lessons/`; marked superseded.
- [ ] Architecture overview document exists and covers:
  - [ ] Mode-based system (markdown, coding, search, and other modes)
  - [ ] Tool inventory and descriptions (`memory.search`, `recall_memory`, `dispute_memory`, `synthesize_and_store_learning`, `fetch_page`, `markdown_forge_document`, `transition_stage`). Verify each against the live catalog with `lisa tools` / `lisa operations` before release.
  - [ ] Memory tiers (`daily`, `weekly`, `long_term`, `permanent` — see `kernel/models.py:MemoryTier`) and salience scoring (0.0–1.0)
  - [ ] Pack system architecture and loading mechanism
  - [ ] Storage layer (`lisa-storage`) and document persistence
  - [ ] Workflow stages and FSM transitions
  - [ ] Fork capability for parallel sub-workflows (`ForkTarget` / `ForkChildResult` with `no_tools` / `blocked` status, `EffectReceipt` completion validation via `classify_child_turn`)
- [ ] Installation / setup guide exists
- [ ] Configuration reference exists (environment variables, config files)
- [ ] Tool reference documentation exists for each tool
- [ ] Known issues document exists and includes:
  - [ ] `transition_stage` handoff behavior (note: the 2026-07-17/18 fork fan-out reliability issues are **fixed** — see `docs/lessons/` and `docs/ROADMAP.md` §0.2; verify no regression before release)
  - [ ] Multiple approval cycles needed for workflow handoffs
  - [ ] Coding mode handoff attempted multiple times
- [ ] Changelog or release notes draft exists

### 2.2 lisa-packs

- [ ] `README.md` documents the pack lifecycle (private standalone → packs-wip → lisa-packs)
- [ ] `docs/PROMOTION_GUIDE.md` is complete and actionable
- [ ] Each pack has or links to its own README
- [ ] Pack compatibility matrix exists (which packs work with which lisa core versions)
- [ ] Changelog or release notes draft exists

______________________________________________________________________

## 3. Code Quality and Testing

### 3.1 lisa (core framework)

- [ ] Linting configured and passing (e.g., ruff, eslint, or equivalent)
- [ ] Type checking configured and passing (e.g., mypy, pyright, or equivalent)
- [ ] Unit tests exist for core modules
- [ ] Unit tests for each tool:
  - [ ] `memory.search`
  - [ ] `recall_memory`
  - [ ] `dispute_memory`
  - [ ] `synthesize_and_store_learning`
  - [ ] `fetch_page`
  - [ ] `markdown_generate`
  - [ ] `markdown_forge_document`
  - [ ] `transition_stage` (include regression tests for known reliability issues)
- [ ] Integration tests for workflow stage transitions (FSM)
- [ ] Integration tests for fork / parallel sub-workflow capability
- [ ] Integration tests for memory tier interactions (long_term, daily, weekly)
- [ ] Integration tests for salience scoring
- [ ] Integration tests for pack loading
- [ ] Integration tests for `lisa-storage` persistence
- [ ] Test coverage threshold defined and met
- [ ] CI pipeline runs all tests on PR and on `main`
- [ ] No `TODO`, `FIXME`, or `HACK` comments blocking release (review and triage)
- [ ] Error handling reviewed for all tool invocations
- [ ] Logging configured and reviewed for sensitive data leakage

### 3.2 lisa-packs

- [ ] Linting configured and passing
- [ ] Spell check (`.cspell`) passing
- [ ] Submodule integrity test — all 6 packs clone and load successfully
- [ ] Smoke test for each pack:
  - [ ] `brave_search`
  - [ ] `letta`
  - [ ] `markdown`
  - [ ] `mem0`
  - [ ] `okf`
  - [ ] `search`
- [ ] Pack manifest / metadata schema validated for each pack
- [ ] CI pipeline runs on PR and on `main`

______________________________________________________________________

## 4. Pack Readiness

For each of the 6 packs, verify the following in the individual `lisa-pack-<name>` repository:

### 4.1 brave_search

- [ ] README present and accurate
- [ ] License file present
- [ ] API key / credential handling documented and secure
- [ ] Tests present and passing
- [ ] Version tagged
- [ ] Submodule SHA in lisa-packs matches latest stable tag

### 4.2 letta

- [ ] README present and accurate
- [ ] License file present
- [ ] Tests present and passing
- [ ] Version tagged
- [ ] Submodule SHA in lisa-packs matches latest stable tag

### 4.3 markdown

- [ ] README present and accurate
- [ ] License file present
- [ ] Tests present and passing
- [ ] Version tagged
- [ ] Submodule SHA in lisa-packs matches latest stable tag

### 4.4 mem0

- [ ] README present and accurate
- [ ] License file present
- [ ] Tests present and passing
- [ ] Version tagged
- [ ] Submodule SHA in lisa-packs matches latest stable tag

### 4.5 okf

- [ ] README present and accurate
- [ ] License file present
- [ ] Tests present and passing
- [ ] Version tagged
- [ ] Submodule SHA in lisa-packs matches latest stable tag

### 4.6 search

- [ ] README present and accurate
- [ ] License file present
- [ ] Tests present and passing
- [ ] Version tagged
- [ ] Submodule SHA in lisa-packs matches latest stable tag

### 4.7 Cross-cutting pack checks

- [ ] Pack naming convention consistent (`lisa-pack-<name>`)
- [ ] All packs follow the promotion lifecycle (none promoted from `packs-wip` prematurely)
- [ ] Pack dependency graph documented (if packs depend on each other)
- [ ] Pack version compatibility with lisa core documented

______________________________________________________________________

## 5. Security Review

### 5.1 lisa (core framework)

- [ ] Dependency audit completed (e.g., `pip-audit`, `npm audit`, or equivalent)
- [ ] No known vulnerable dependencies
- [ ] Secret scanning run (GitHub secret scanning or equivalent)
- [ ] No hardcoded credentials in code or config
- [ ] Memory storage layer (`lisa-storage`) reviewed for unauthorized access paths
- [ ] `fetch_page` tool reviewed for SSRF and URL validation
- [ ] Tool permission model reviewed — can tools access resources beyond their scope?
- [ ] Workflow FSM reviewed for state injection or unauthorized transitions
- [ ] Fork / sub-workflow capability reviewed for privilege escalation
- [ ] Salience scoring reviewed for manipulation vectors
- [ ] `dispute_memory` tool reviewed for abuse potential (memory poisoning)
- [ ] Rate limiting or guardrails considered for tool invocations
- [ ] If repository goes public, full history scanned for leaked secrets

### 5.2 lisa-packs

- [ ] Dependency audit for each pack
- [ ] Secret scanning run
- [ ] Each pack reviewed for excessive permissions or network access
- [ ] `brave_search` API key handling reviewed
- [ ] `search` pack reviewed for SSRF if it fetches URLs
- [ ] `letta` and `mem0` packs reviewed for data exfiltration risks
- [ ] Submodule URLs verified to point only to `Lisa-Harness` org
- [ ] No pack executes arbitrary code on load without documented consent

______________________________________________________________________

## 6. Release Artifacts

### 6.1 lisa (core framework)

- [ ] `CHANGELOG.md` created or updated with all changes since last release
- [ ] Release notes drafted (summary, new features, fixes, known issues)
- [ ] Known issues section includes:
  - [ ] `transition_stage` reliability issues
  - [ ] Workflow handoff approval cycle behavior
  - [ ] Coding mode handoff behavior
- [ ] Version number decided and consistent across all files
- [ ] Git tag created (`vX.Y.Z`)
- [ ] GitHub Release created with notes
- [ ] Release artifacts built and attached (if applicable: wheel, tarball, Docker image)
- [ ] Installation instructions verified end-to-end on a clean environment
- [ ] Upgrade / migration guide written (if breaking changes exist)

### 6.2 lisa-packs

- [ ] `CHANGELOG.md` created or updated
- [ ] Release notes drafted
- [ ] Version number decided
- [ ] Git tag created (`vX.Y.Z`)
- [ ] GitHub Release created with notes
- [ ] All 6 submodule SHAs pinned to tagged releases (not commit SHAs from `main`)
- [ ] Clone-and-verify test: fresh clone of lisa-packs with `--recursive` succeeds and all packs load

______________________________________________________________________

## 7. Post-Release Verification

- [ ] Fresh install of lisa core from release artifact succeeds
- [ ] Fresh clone of lisa-packs with submodules succeeds
- [ ] All 6 packs load into a running lisa instance
- [ ] End-to-end smoke test: markdown mode, coding mode, search mode each function
- [ ] Memory tools (search, recall, dispute, synthesize) function end-to-end
- [ ] `transition_stage` tested in a real workflow — confirm fix or document limitation
- [ ] `lisa-storage` persistence verified across session restart
- [ ] Fork / parallel sub-workflow verified
- [ ] Release announcement prepared (if applicable)
- [ ] Issue tracker updated — known issues filed and linked from release notes

______________________________________________________________________

## Sign-off

| Role | Name | Date | Approved |
| ------ | ------ | ------ | ---------- |
| Core framework maintainer | | | |
| Pack registry maintainer | | | |
| Security reviewer | | | |
| QA / testing | | | |
| Release manager | | | |

<!-- generated by lisa · provider: ollama · model: glm-5.2:cloud · 2026-07-26 -->
