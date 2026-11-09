"""Shared lexical ranking for memory recall.

One tokenizer, one overlap scorer, used by both the SQLite store's recall
pools and the memory pack's evidence re-ranking. Keeping the two layers on
the same token definition prevents a record the backend surfaces from
looking disjoint to the pack ranker (or the reverse): before this module
the store split on whitespace while the pack matched ``[a-z0-9]{3,}``
against content *and* the JSON-serialized ``content_json``, so identical
records scored differently at each layer.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Minimum token length — 1-2 character fragments (x, to, of) are noise.
_MIN_TOKEN_LEN = 3


def tokenize(text: str) -> set[str]:
    """Lowercase ``text`` into alphanumeric token sets, dropping 1-2 char tokens."""
    return {
        match.group(0)
        for match in _TOKEN_RE.finditer(text.lower())
        if len(match.group(0)) >= _MIN_TOKEN_LEN
    }


def lexical_score(query_tokens: set[str], text: str) -> float:
    """Overlap fraction of ``query_tokens`` found in ``text`` (0.0-1.0).

    Normalized by query size: a one-word hit on a five-word query scores
    0.2. An empty query (or all-short-token query) scores 0.0 — callers
    treat that as "no lexical signal" and fall back to their other signals.
    """
    if not query_tokens:
        return 0.0
    return len(query_tokens.intersection(tokenize(text))) / len(query_tokens)
