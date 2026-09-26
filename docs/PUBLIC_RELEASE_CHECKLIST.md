# Lisa Harness — public-release checklist

> Generated 2026-09-22 from a live readiness audit of the working tree.
> Governs flipping `github.com/Lisa-Harness/lisa` to **public** and cutting
> the first tagged release (`v0.1.0`). Long-form detail stays in
> [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md); this file is the follow-by-hand
> tracker. Check items off as they land.

Legend: ✅ verified done · ⚠️ needs attention · ❌ missing/broken

---

## 0. Snapshot — audit verdict

| Area | State | Evidence |
| --- | --- | --- |
| Runtime/personal data in git | ✅ clean | `logs/`, `data/`, `config/…` untracked via ignore chain; `git log` on those paths = 0 commits; no `sk-`/`ghp_`/PEM in last 200 commits |
| LICENSE | ✅ MIT, "The Migus Group" | [LICENSE](../LICENSE) |
| CI | ✅ strong matrix (3 OS × Py 3.11/3.12) + format/lint/mypy/vulture/pytest | [ci.yml](../.github/workflows/ci.yml) · but **red**, see §1 |
| Publish | ✅ tag-triggered PyPI trusted publishing after CI and package-version checks (`v*`) | [publish.yml](../.github/workflows/publish.yml) |
| Working tree | ⚠️ | 5 modified files, 2 unpushed commits, 2 untracked paths |
| Community files | ❌ | `SECURITY.md` and `CODE_OF_CONDUCT.md` missing |
| Hardcoded paths | ⚠️ | `/home/adam/…` in `packstest/run_all_validations.py`, "p920" anecdotes in CHANGELOG |
| Registry drift | ⚠️ | [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) says "6 packs"; org actually ships 10 |

---

## 1. Blockers — fix before anything else

- [ ] **CI red — direct-I/O security gate.** `uv run python scripts/check_direct_io_policy.py` exits **1** (verified locally today). Apply the reviewed patch in
  [STORAGE_IO_POLICY_FIX_REPORT.md](../STORAGE_IO_POLICY_FIX_REPORT.md) (25+/28−,
  allowlist 21→20 keys, all live) as its own commit, then run the gate to green.
  The report's "pre-existing 4 modified kernel files" note is **stale** — those
  were the `ea0d315`/`4beda22` unpushed commits, not WIP.
- [ ] **Push the 2 unpushed commits.** `origin/main` sits at `318aafb`; local has
  `ea0d315` + `4beda22` (kernel strict-mode fix + guard-layering docs). Public
  flip must carry them.
- [ ] **Review & commit the pending release-prep diff** — README.md (+163/−),
  CONTRIBUTING.md, kernel/README.md, sdk/README.md, uv.lock. The README rewrite
  (packstest blurb, CLI-host link fix) is release-critical; land or discard
  deliberately.
- [ ] **Decide `docs/blog/`** — `introducing-lisa.md` + `why-lisa-is-built-the-way-it-is.md`
  are untracked launch content. Commit if intended public.

---

## 2. Repository hygiene

- [ ] Delete `/home/adam/…` hardcodes in [packstest/run_all_validations.py](../packstest/run_all_validations.py#L12) (`REPOS`, `sys.path`, `PY`) — parameterize or fail loudly.
- [ ] Scrub personal anecdotes from [CHANGELOG.md](../CHANGELOG.md#L480) ("p920 stuck-loop") → neutral wording ("a stuck-loop where Lisa ended eight …").
- [ ] `little-lisa` references — verify current scope: [CHANGELOG.md](../CHANGELOG.md#L1249) entries are history (OK), but [host/memory_aliases.py](../host/src/host/memory_aliases.py#L60) docstring + [test_proxy_policy.py](../kernel/tests/test_proxy_policy.py#L255) fixtures still use the retired name — confirm they're examples-only, not live behavior.
- [ ] Add **SECURITY.md** (vuln-report policy) and **CODE_OF_CONDUCT.md**. RELEASE_CHECKLIST requires CoC; neither exists.
- [ ] GitHub settings: branch protection on `main`, repo description/topics, issue & PR templates.
- [ ] Confirm `.coverage` (tracked, 1 file) should stay in git — it's a local artifact; likely `git rm` + ignore.
- [ ] Stash: 1 entry — pop/drop deliberately (see repo memory: pop carries real blast radius here).

---

## 3. Docs / metadata sync

- [ ] Update [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md#L57) — "6 packs" → **10**
  (brave, cloudservers, letta, markdown, mem0, okf, personas, search, serper,
  tavily) and the §3.2 smoke-test list (it currently omits cloudservers,
  personas, serper, tavily).
- [ ] Confirm each pack repo + submodule scope (lisa-packs registry is out of
  tree — spot-check submodule SHAs pinned to tags, not `main`, per §6.2 there).
- [ ] Verify README's sibling-repo links resolve publicly (`lisa-hosts/cli`,
  `lisa-packs`) — they must also be public for links to work.
- [ ] Docs sweep: ensure [`/docs`](./) references `lisa` CLI (not `little-lisa`) in
  user-facing guides.
- [ ] Draft `v0.1.0` release notes from CHANGELOG's `[Unreleased]` section.

---

## 4. Verification gates (run, confirm green)

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run python scripts/check_direct_io_policy.py` ← the red one
- [ ] Production mypy gate passes (see the type-check step in [ci.yml](../.github/workflows/ci.yml)).
- [ ] `uv run vulture … vulture_allowlist.py`
- [ ] `uv run pytest -m "not live" --tb=short`
- [ ] Secret scan (gitleaks / GitHub secret scanning) over **full history**, not last-N.

---

## 5. Release mechanics

- [ ] Flip repo visibility → **public** (only after §1 + §2 merge to `origin/main`).
- [ ] Bump versions: all packages are `0.1.0` — confirm intended first-tag.
- [ ] `git tag v0.1.0 && git push --tags` → triggers trusted-publisher PyPI publish
  of the 8 build targets in [publish.yml](../.github/workflows/publish.yml).
- [ ] Verify the 8 wheels on PyPI (kernel, sdk, mcp-servers, pack-core,
  pack-introspect, pack-memory, pack-recovery, host-runtime) — non-zero entry
  count each (see `uv-packaging-gotchas` memory: stale-wheel trap).
- [ ] Create GitHub Release from the tag with the drafted notes.
- [ ] Post: fresh `pipx install` / `uv tool install` smoke-test on a clean machine.

---

## 6. Deferred / out-of-scope (noted, not blocking)

- `lisa-harness-cli` distribution checklist — lives at `../lisa-hosts/docs/RELEASE_CHECKLIST.md`.
- lisa-packs registry §4 per-pack audits (READMEs, licenses, version tags).
- `docs/research/` GAP/SWOT superseded banners (already flagged in RELEASE_CHECKLIST).

---

*Owner: @adam · Generated by release audit · See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) §Sign-off for approver table.*
