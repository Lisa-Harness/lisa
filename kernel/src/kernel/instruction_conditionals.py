"""Conditional operation blocks in mode instruction markup.

Mode authors can wrap procedural instruction text in a conditional block that
is evaluated once at mode-load time against the installed operation catalog:

    <!-- if operation:okf_emit_concept -->
    When you have a structured result, call `okf_emit_concept` to emit it.
    <!-- endif -->

When the operation is installed, both marker lines are removed and the inner
content is kept byte-identical. When it is not installed, the whole block —
markers and content — is removed and the surrounding whitespace is collapsed,
so no prose naming a missing operation reaches the model's prompt.

Rules:

- **Operations only.** The condition tests presence in the operation catalog
  (``get_operation_catalog()`` keys) — installed, independent of any mode's
  allowlist. Tool IDs are out of scope: live tool schemas are a per-turn
  artifact and do not exist at mode-load time.
- **Fences are literal.** Markers inside fenced code blocks are literal text,
  never evaluated. Fence state tracks the opening fence's character and run
  length (CommonMark: the closing fence must use the same character, be at
  least as long, and be followed only by whitespace), so bodies mixing ```
  and ~~~ fences cannot desync the scanner.
- **Idempotent.** After one pass no conditional marker remains, so applying
  the transform twice is a no-op: ``f(f(x)) == f(x)``.
- **Malformed markup fails visible, not silent.** An unterminated opener or a
  stray ``<!-- endif -->`` drops only the marker line; the authored content
  around it is preserved (and the missing-operation warning stays, because
  the content still reaches the prompt). A nested opener makes the outer
  block malformed: the outer marker line is dropped and the inner block is
  evaluated on its own.
- **Whitespace.** Removing a block (or a malformed marker line) collapses
  runs of three or more newlines in the result down to two. Kept blocks only
  lose their two marker lines; inner content is byte-identical and no
  collapse is performed when nothing was removed.

This module is pure: no I/O, no logging, no catalog access. Callers pass
``available_ids`` explicitly.
"""

from __future__ import annotations

import re

__all__ = [
    "conditional_operation_ids",
    "has_conditional_markers",
    "strip_conditional_operation_blocks",
]

# Marker syntax: <!-- if operation:<id> --> ... <!-- endif -->
# Each marker must sit on its own line (leading/trailing whitespace is
# ignored). Operation IDs are identifier-shaped; anything else is not a
# directive and stays literal so an authoring typo remains visible in the
# rendered prompt instead of silently vanishing.
_OPEN_MARKER_RE = re.compile(r"^<!--\s*if\s+operation:(?P<op_id>[A-Za-z0-9_]+)\s*-->$")
_CLOSE_MARKER_RE = re.compile(r"^<!--\s*endif\s*-->$")

# Conservative fast-path substring: markers are HTML comments, so a body
# containing no "<!--" can never change. No longer hint is safe — the marker
# regexes tolerate arbitrary whitespace inside the comment, so e.g.
# "<!--   if   operation:x -->" shares only "<!--" with the canonical form.
_MARKER_HINT = "<!--"

# Fenced code blocks (CommonMark): an opening fence is 3+ backticks or
# tildes at the start of the (optionally indented) line; the closing fence
# must use the same character, be at least as long, and carry nothing else.
_FENCE_OPEN_RE = re.compile(r"(?P<fence>```+|~~~+)")


def _opening_fence(line: str) -> tuple[str, int] | None:
    """Return ``(fence_char, run_length)`` when ``line`` opens a fenced block.

    CommonMark: the info string following a backtick fence may not contain
    backticks (and symmetrically for tildes), so a prose line like
    ````` ```foo``` ````` is not a fence opener.
    """
    lstripped = line.lstrip()
    match = _FENCE_OPEN_RE.match(lstripped)
    if match is None:
        return None
    fence = match.group("fence")
    if fence[0] in lstripped[len(fence) :]:
        return None
    return fence[0], len(fence)


def _is_closing_fence(line: str, fence_char: str, run_length: int) -> bool:
    """Return True when ``line`` closes the fence opened with ``fence_char``."""
    stripped = line.strip()
    return len(stripped) >= run_length and all(ch == fence_char for ch in stripped)


def _marker_op_id(line: str) -> str | None:
    """Return the operation ID when ``line`` is a conditional opener marker."""
    match = _OPEN_MARKER_RE.match(line.strip())
    if match is None:
        return None
    return match.group("op_id")


def _is_close_marker(line: str) -> bool:
    """Return True when ``line`` is a conditional ``<!-- endif -->`` marker."""
    return _CLOSE_MARKER_RE.match(line.strip()) is not None


def has_conditional_markers(instruction_body: str) -> bool:
    """Return True when the body might contain a conditional marker.

    Conservative substring pre-check (no fence awareness): any HTML comment
    matches, so marker-free comment bodies fall through to the full scan and
    are returned unchanged.
    """
    return _MARKER_HINT in instruction_body


def conditional_operation_ids(instruction_body: str) -> set[str]:
    """Return every operation ID referenced by a conditional opener marker.

    Fence-aware: markers inside fenced code blocks are literal text and do
    not count. Includes IDs from malformed (unterminated or nested) openers —
    callers gate the actual warning downgrade on the effective body, so an
    over-broad set here is harmless while an under-broad one would miss
    deliberate conditional references.
    """
    if _MARKER_HINT not in instruction_body:
        return set()
    ids: set[str] = set()
    fence_char: str | None = None
    fence_length = 0
    for line in instruction_body.split("\n"):
        if fence_char is not None:
            if _is_closing_fence(line, fence_char, fence_length):
                fence_char = None
                fence_length = 0
            continue
        opened = _opening_fence(line)
        if opened is not None:
            fence_char, fence_length = opened
            continue
        op_id = _marker_op_id(line)
        if op_id is not None:
            ids.add(op_id)
    return ids


def _find_block_end(lines: list[str], start: int) -> int | None:
    """Return the index of the ``<!-- endif -->`` closing an opened block.

    Scans fence-aware from ``start``: a close marker inside a fenced code
    block is literal text. Returns ``None`` when the block is unterminated or
    when a nested opener makes it malformed — the caller then drops only the
    opener line so the inner block is evaluated on its own.
    """
    fence_char: str | None = None
    fence_length = 0
    for index in range(start, len(lines)):
        line = lines[index]
        if fence_char is not None:
            if _is_closing_fence(line, fence_char, fence_length):
                fence_char = None
                fence_length = 0
            continue
        opened = _opening_fence(line)
        if opened is not None:
            fence_char, fence_length = opened
            continue
        if _marker_op_id(line) is not None:
            return None  # nested opener: the outer block is malformed
        if _is_close_marker(line):
            return index
    return None


def _handle_conditional_block(
    lines: list[str],
    opener_index: int,
    op_id: str,
    available_ids: set[str],
    out: list[str],
) -> int:
    """Process one conditional block starting at ``lines[opener_index]``.

    Returns the index of the first line after the processed block, and
    appends the kept content (if any) to ``out``. Malformed blocks
    (unterminated or nested-opener) drop only the opener marker line so the
    authored content stays and the authoring error stays visible in the
    prompt.
    """
    end = _find_block_end(lines, opener_index + 1)
    if end is None:
        return opener_index + 1
    if op_id in available_ids:
        out.extend(lines[opener_index + 1 : end])
    return end + 1


def strip_conditional_operation_blocks(
    instruction_body: str,
    available_ids: set[str],
) -> str:
    """Evaluate conditional operation blocks in one instruction body.

    Blocks whose operation is in ``available_ids`` keep their inner content
    byte-identical and lose only the two marker lines. Blocks whose operation
    is absent are removed entirely. Markers inside fenced code blocks are
    literal text. Malformed markers (unterminated opener, stray close, nested
    opener) drop only the marker line. When anything was removed, runs of
    three or more newlines in the result collapse to two; otherwise the body
    is returned with only marker lines removed. A body with no marker
    substring is returned byte-identical.

    Args:
        instruction_body: The authored mode instruction body.
        available_ids: Operation IDs present in the installed operation
            catalog. The caller owns resolution; this function never reads
            the catalog itself.

    Returns:
        The instruction body with conditional blocks evaluated. Applying the
        transform to its own output is a no-op.
    """
    if not has_conditional_markers(instruction_body):
        return instruction_body

    # split("\n") (not splitlines) is lossless: the trailing-newline element
    # and any \r from \r\n line endings survive the round trip.
    lines = instruction_body.split("\n")
    out: list[str] = []
    removed_any = False
    index = 0
    fence_char: str | None = None
    fence_length = 0

    while index < len(lines):
        line = lines[index]

        if fence_char is not None:
            out.append(line)
            if _is_closing_fence(line, fence_char, fence_length):
                fence_char = None
                fence_length = 0
            index += 1
            continue

        opened = _opening_fence(line)
        if opened is not None:
            fence_char, fence_length = opened
            out.append(line)
            index += 1
            continue

        op_id = _marker_op_id(line)
        if op_id is not None:
            next_index = _handle_conditional_block(lines, index, op_id, available_ids, out)
            if next_index == index + 1 or op_id not in available_ids:
                removed_any = True
            index = next_index
            continue

        if _is_close_marker(line):
            # Stray close with no open block: drop the marker line.
            removed_any = True
            index += 1
            continue

        out.append(line)
        index += 1

    result = "\n".join(out)
    if removed_any:
        result = re.sub(r"\n{3,}", "\n\n", result)
    return result
