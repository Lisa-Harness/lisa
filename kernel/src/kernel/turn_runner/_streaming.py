"""StreamedCompletionAdapter — makes a StreamedCompletion look like a CompletionResult."""

from __future__ import annotations

import re
from collections import Counter, deque

from kernel.provider_contracts import StreamedCompletion

_INLINE_LITERAL = re.compile(r'`[^`\n]*`|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_NUMBER = re.compile(r"\b\d+\b")
_NON_SIGNATURE = re.compile(r"[^a-z0-9#<> ]+")
_WHITESPACE = re.compile(r"\s+")
_MIN_SIGNATURE_CHARS = 8


class StreamRepetitionGuard:
    """Detect sustained structurally repeated lines while a stream is live."""

    __slots__ = (
        "_content_pending",
        "_counts",
        "_line_limit",
        "_reasoning_pending",
        "_window",
        "_window_chars",
        "_window_size",
    )

    def __init__(self, *, line_limit: int, window_chars: int) -> None:
        self._content_pending = ""
        self._reasoning_pending = ""
        self._line_limit = max(0, int(line_limit))
        self._window_chars = max(0, int(window_chars))
        self._window: deque[tuple[str, int]] = deque()
        self._window_size = 0
        self._counts: Counter[str] = Counter()

    @staticmethod
    def _signature(line: str) -> str:
        normalized = _INLINE_LITERAL.sub("<literal>", line.lower())
        normalized = _NUMBER.sub("#", normalized)
        normalized = _NON_SIGNATURE.sub(" ", normalized)
        normalized = _WHITESPACE.sub(" ", normalized).strip()
        return normalized if len(normalized) >= _MIN_SIGNATURE_CHARS else ""

    def _observe_lines(self, text: str, pending: str) -> tuple[str, bool]:
        pending += text
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            signature = self._signature(line)
            line_chars = len(line) + 1
            if self._observe_signature(signature, line_chars):
                return pending, True
        return pending, False

    def _observe_signature(self, signature: str, line_chars: int) -> bool:
        self._window.append((signature, line_chars))
        self._window_size += line_chars
        if signature:
            self._counts[signature] += 1
        while self._window and self._window_size > self._window_chars:
            old_signature, old_chars = self._window.popleft()
            self._window_size -= old_chars
            if old_signature:
                self._counts[old_signature] -= 1
                if self._counts[old_signature] <= 0:
                    del self._counts[old_signature]
        return bool(signature and self._counts[signature] >= self._line_limit)

    def observe(self, *, content: str = "", reasoning: str = "") -> bool:
        """Return ``True`` once one repeated shape reaches the configured cap."""
        if self._line_limit <= 0 or self._window_chars <= 0:
            return False
        if content:
            self._content_pending, repeated = self._observe_lines(content, self._content_pending)
            if repeated:
                return True
        if reasoning:
            self._reasoning_pending, repeated = self._observe_lines(
                reasoning, self._reasoning_pending
            )
            if repeated:
                return True
        return False


# ── Streaming adapter ──────────────────────────────────────────────────────


class StreamedCompletionAdapter:
    """Adapter that makes a ``StreamedCompletion`` look like a ``CompletionResult``.

    The turn runner's tool loop expects ``completion.content``,
    ``completion.tool_calls``, and ``completion.usage``. When the streaming
    path produces a ``StreamedCompletion`` (no tool calls in v1), this adapter
    bridges the two shapes without forcing the streaming code to construct a
    full ``CompletionResult``.

    For gpt-5.x reasoning models the visible ``content`` can be empty while
    the model's substantive answer lives in ``reasoning_content``. This
    adapter keeps the two SEPARATE: ``content`` stays the visible answer and
    ``reasoning_content`` carries the thinking. Folding reasoning into
    ``content`` here would defeat the kernel's user-facing guards, which rely
    on distinguishing a real answer from leaked chain-of-thought so they can
    bound it (the 2026-10-04 non-streaming 119KB monologue incident). Callers
    that need the folded view (structured-output parsers) use
    ``completion_content``, which folds ``reasoning_content`` when ``content``
    is empty.
    """

    __slots__ = (
        "_aggregated",
        "content",
        "duration_ms",
        "finish_reason",
        "reasoning_content",
        "tool_calls",
        "usage",
    )

    def __init__(self, aggregated: StreamedCompletion) -> None:
        self._aggregated = aggregated
        self.reasoning_content = aggregated.reasoning_content
        # Keep the visible answer and the reasoning SEPARATE — the kernel's
        # display sites bound leaked reasoning via ``reasoning_tail_excerpt``,
        # which only fires when ``content`` is empty for a reasoning-only turn.
        self.content = aggregated.content
        self.tool_calls = aggregated.tool_calls
        self.usage = aggregated.usage
        # Surfaced flat for the same diagnostics the non-streaming
        # ``CompletionResult`` carries: the reasoning-tail caveat's
        # truncation clause and the turn-loop empty-completion warnings read
        # ``finish_reason`` via ``getattr``. The 2026-10-13 turn 789 streamed
        # burnout caveat reported ``finish_reason=unknown`` because the
        # adapter hid the aggregate's termination signal.
        self.finish_reason = aggregated.finish_reason
        # Surfaced flat for the turn-runner's duration extraction; the stream
        # carries no per-chunk duration, so the caller measures and stamps it.
        self.duration_ms = 0.0

    def __repr__(self) -> str:
        return (
            f"StreamedCompletionAdapter(content={self.content!r}, "
            f"reasoning_content={self.reasoning_content!r}, "
            f"finish_reason={self._aggregated.finish_reason!r})"
        )
