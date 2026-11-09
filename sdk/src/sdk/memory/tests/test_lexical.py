"""Tests for the shared memory lexical scoring contract."""

import pytest
from sdk.memory.lexical import lexical_score, tokenize


def test_tokenize_normalizes_case_and_drops_short_fragments() -> None:
    assert tokenize("AI and GPT-4: GPT, Model!") == {"and", "gpt", "model"}


@pytest.mark.parametrize(
    ("tokens", "text", "expected"),
    [
        (set(), "anything", 0.0),
        ({"alpha", "beta", "gamma"}, "ALPHA and beta", 2 / 3),
        ({"alpha"}, "unrelated", 0.0),
    ],
)
def test_lexical_score_is_query_normalized(tokens: set[str], text: str, expected: float) -> None:
    assert lexical_score(tokens, text) == pytest.approx(expected)
