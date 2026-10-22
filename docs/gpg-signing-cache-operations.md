# GPG signing cache — operational runbook

Signed commits fail in non-interactive contexts because gpg-agent launches
curses pinentry and there is no TTY. This runbook describes how to make
signed commits work on demand **without ever handing a passphrase to a
program**.

All environment facts were verified on the developer's machine on
2026-10-22. See `docs/trailing-newline-root-cause-and-gpg-signing-2026-10-22.md`
in the host workspace for the full analysis and rationale.

## Current state: signing already works

Verified 2026-10-22 on this host: the agent's own commits **are already
signed, and the signature is good**.

```sh
$ git log --format='%G? %GS %GK' -2
G|Adam C. Migus <adam@migusgroup.com>|86DEDFC7398CEC63
G|Adam C. Migus <adam@migusgroup.com>|86DEDFC7398CEC63

$ git cat-file -p HEAD | head -3
tree 9d84c8c77bb4908d0c262d6ab5688a92f64f1bbf
parent 89e22ddab590b98d612f37d9c582be9d10b48426
author Adam C. Migus <adam@migus.org> 1792688866 -0400
```

`%G?` = `G` means a **good** signature, verified against the signer's key.
The commit object carries a `gpgsig` header. The worktree's *local* config
has no `commit.gpgsign` key — the value `true` comes only from global
`~/.gitconfig`.

Two consequences:

1. The agent's commit path reaches the operator's real gpg-agent and signs
   correctly. **The passphrase cache is already primed on this host.**
2. "Commits do not happen" is **not** a signing problem. Look elsewhere
   first — a blocked approval, a failed check, a missing worktree, or a
   dirty tree all block a commit just as effectively as a signing failure,
   and only the last of those is documented here.

Author identity and signer identity differ (`adam@migus.org` vs
`adam@migusgroup.com`). Git does not require them to match, but a policy
that verifies author identity against signer identity will not be
satisfied by this configuration.

## Environment facts

| Fact | Value |
|---|---|
| GnuPG | 2.5.24 |
| Default TTLs | `default-cache-ttl` 600 s, `max-cache-ttl` 7200 s |
| TTL options | Work — writing them to `gpg-agent.conf` and reloading shows the new current values in `gpgconf --list-options gpg-agent` |
| `gpg-preset-passphrase` | Not installed (ships in `gnupg2-utils`) |
| `PRESET_PASSPHRASE` Assuan command | Refused unless the agent is started with `allow-preset-passphrase` |
| Global git config | `commit.gpgsign=true`, `user.signingkey=<fingerprint>` |
| Cold non-interactive commit | Fails: `gpg: signing failed: Operation cancelled` → `fatal: failed to write commit object` |
| Keygrip | `gpg --list-secret-keys --with-colons \| awk -F: '/^grp:/{print $10; exit}'` |
| Cache indicator | `gpg-connect-agent 'keyinfo --list' /bye` — the 4th field after the keygrip |

## The mechanism

gpg-agent caches a passphrase **only** after a successful **pinentry** (the
interactive path). Once cached, later signing operations reuse it —
**including operations from processes with no TTY**.

You type the passphrase once, in your own terminal; the agent signs on your
behalf for a bounded window. The agent can *use* the key but cannot
*extract* the secret. That distinction is the entire point.

Two conditions must hold for an agent-made commit to be signed:

1. The committing process reaches the **same gpg-agent instance** as the
   interactive session.
2. That agent holds the passphrase in cache.

Condition 1 can silently fail: a sandboxed process with a private
`GNUPGHOME` or a private agent socket starts its own agent with an empty
cache. Verify it before assuming the rest works.

## Procedure

### Step 0 — (optional) extend the window

Add to `~/.gnupg/gpg-agent.conf`:

```plaintext
default-cache-ttl 28800
max-cache-ttl 28800
```

`gpg-agent.conf` is read at agent startup, so restart it:

```sh
gpgconf --kill gpg-agent
```

The next `gpg` invocation starts a fresh agent that reads the new file.
Killing the agent also clears the cache, so prime *after* this step.

Without it, the stock window is 600 s.

### Step 1 — prime the cache

```sh
gpg --clearsign --local-user <fingerprint> -o /dev/null /dev/null
```

`pinentry-curses` appears; type the passphrase. It succeeds.

### Step 2 — confirm the cache is warm

```sh
gpg-connect-agent 'keyinfo --list' /bye
```

The KEYINFO reply has this field order:

```plaintext
S KEYINFO <keygrip> <type> <serial> <idstr> <cached> <protection> <fpr> <ttl> <flags>
```

Counting tokens of the whole line, the **cached** indicator is token 7 —
equivalently the 4th field *after* the keygrip (token 3). An uncached
secret key reads `S KEYINFO <keygrip> D - - - P - - -`, where the cached
field is `-`. Once the passphrase is cached that field holds the remaining
TTL in seconds.

Do not rely on counting positions: take a **before/after diff** of the
KEYINFO line instead, and treat "does a commit actually sign?" as the
authoritative test.

### Step 3 — confirm the agent process shares the agent

From the same environment commits are made in:

```sh
gpg-connect-agent 'GETINFO version' /bye
gpg-connect-agent 'keyinfo --list' /bye
```

A cached entry for your key means the agent is shared and the procedure
works. No cached entry means a private agent — see below.

Compare the agent socket path with the one the operator's terminal uses:

```sh
gpgconf --list-dirs agent
```

If the paths differ, the two are talking to different agents and the
cache will never be shared. Point the process at the operator's agent
with `GNUPGHOME=$HOME/.gnupg`.

### Step 4 — work

Commits sign while the cache is warm.

### Step 5 — revoke

```sh
gpgconf --kill gpg-agent
```

The cache dies with the agent. The next signing operation needs a human.
No passphrase required to revoke.

## Safety properties

- The passphrase never leaves the terminal: it enters agent memory, not a
  file, not an env var, not a readable config.
- The agent can use the key but cannot extract the secret.
- The window is bounded by the TTL and closes on its own.
- Revocation is instant and needs no passphrase.

A passphrase in a file or env var is readable by every process running as
the user, permanently. This design is the only one where the secret is used
but never disclosed.

## Do not enable `allow-preset-passphrase`

It would allow a program to hand the passphrase to the agent directly. That
converts "an agent may use your key" into "an agent may read your
passphrase" and removes the boundary this design exists to keep.

## If the agent process has a private agent

Symptom: step 3 shows no cached entry and commits still fail while
interactive signing works.

Preferred fix: point the process at the real agent socket
(`GNUPGHOME=$HOME/.gnupg`) so one agent per user is the topology. The
alternative is a dedicated no-passphrase key for the agent, which removes
the TTY problem and makes agent commits honestly attributable — a
provenance decision, made deliberately.

## Diagnostic signatures

| Signal | Meaning |
|---|---|
| exit 128 + `fatal: failed to write commit object` | commit object could not be written |
| `gpg: signing failed: Operation cancelled` | pinentry cancelled; no TTY or no cached passphrase |
| `[GNUPG:] FAILURE sign 83886179` | the agent refused to sign |
| `ERR ... no --allow-preset-passphrase` | `PRESET_PASSPHRASE` attempted without the agent flag |
| `PINENTRY_LAUNCHED` | pinentry started and died for lack of a TTY |

## Expected agent behavior

A signing failure is a *specific, classified* condition, not a generic
commit error. When a commit fails this way, report the signing key
fingerprint, the fact that no passphrase is cached, the exact command to
prime it, the current `default-cache-ttl`, and the revocation command. The
operator should never have to interpret gpg's stderr to know what to do.

Never embed, persist, or request the passphrase: no file, no env var, no
agent-driven prompt, and never `allow-preset-passphrase`. If a passphrase
is needed, a human types it into a terminal the agent does not control.

Keep signing enabled for real commits. Test fixtures that need
`commit.gpgsign false` should set it locally in the repo config they
create, and say why — disabling it globally would silently stop signing
the operator's own work.
