"""Single source of truth for the reader-understood ledger schema versions.

The forensic reader consumes rows written by the kernel's durable writers.
Every writer stamps ``schema_version`` on its rows (or wrote rows before the
stamp existed); the reader must understand every version up to the highest
one any current writer emits, read rows without the stamp as legacy, and
treat anything newer as unreadable-but-reported.

Verified against the current writers:

- ``TokenUsageEvent`` (kernel.models) -- ``schema_version: int = 3``
- evidence topics (kernel.evidence TOPIC_SPECS) -- ``schema_version: int = 2``
- approvals / errors / runtime log rows -- no ``schema_version`` today, so
  they read as version 0 (legacy) and stay legal to parse

Bump :data:`SCHEMA_VERSION_CURRENT` only when a kernel writer starts
emitting a higher version, and update this docstring with the evidence.
"""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION_CURRENT"]

#: Highest ``schema_version`` any current kernel writer emits. Token rows
#: carry 3; evidence-topic rows carry 2; the other ledgers are unstamped and
#: therefore version 0 on the read path.
SCHEMA_VERSION_CURRENT = 3

# The read-side contract this constant enforces is specified in
# docs/design/spec1-s5-forensic-reader.md.
