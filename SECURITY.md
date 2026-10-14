# Security policy

## Report a vulnerability privately

Use GitHub's **Security → Report a vulnerability** for
[this repository](https://github.com/Lisa-Harness/lisa/security/advisories/new).
Include the affected version/commit, configuration, prerequisites, impact,
and a minimal reproduction with synthetic data.

Do not post exploit details, credentials, or private conversation data in
public issues or pull requests. Do not test against systems you do not own
or have permission to test.

**Maintainer setup requirement:** enable GitHub private vulnerability reporting
before public release and verify the reporting link works for a non-maintainer.
If the reporting form is unavailable, open an issue requesting a private contact
channel **without vulnerability details**. No response-time or bounty commitment
is currently made.

## Supported versions

The project is pre-1.0. Security fixes target the current development line and
the latest release when available; there is no published LTS/backport matrix.
Maintainers will identify affected/fixed versions in an advisory. Hosts and
external packs have their own release and reporting policies.

## Trust and deployment boundaries

- Installed packs and in-process tools execute Python. Treat them as trusted
  code; catalog restrictions are not a process sandbox.
- Model/provider requests can contain session instructions, conversation,
  evidence, and tool results. Apply your provider's data-processing policy.
- Default input/output guard implementations pass through. Applications that
  require content filtering must configure an appropriate implementation.
- Approval policy and containment are distinct. A policy decision does not
  create an OS sandbox, and a missing/degraded sandbox must not be treated
  as contained.
- Surface and memory-scope contracts need authenticated host identities and
  enforcement in storage, retrieval, and delivery. Single-user defaults are
  not a multi-tenant isolation guarantee.
- Threads, request/result objects, logs, memory, and approval state are
  sensitive local data. Read-time redaction is not at-rest encryption or
  secure deletion.

See [architecture boundaries](docs/BOUNDARIES.md),
[host integration](docs/HOST_INTEGRATION.md), [approval policy](docs/POLICY_GUIDE.md),
and [storage](docs/STORAGE.md) before exposing a deployment to untrusted users.
