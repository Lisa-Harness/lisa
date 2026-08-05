"""Tests for the CLI REPL prompt history wrapper.

Regression coverage for the bug where ``_SurrogateSafeFileHistory`` delegated
``append_string`` to the inner ``FileHistory``, leaving the wrapper's in-memory
``_loaded_strings`` stale. The symptom was that up-arrow history within a
session omitted the commands you just typed.
"""

from __future__ import annotations

import pytest

prompt_toolkit = pytest.importorskip("prompt_toolkit")

from hosts.cli.repl import _SurrogateSafeFileHistory


def test_append_string_updates_wrapper_in_memory(tmp_path) -> None:
    """``append_string`` must update the wrapper's ``_loaded_strings``.

    prompt_toolkit's ``Buffer`` reads history via ``History.load()``, which
    yields from ``self._loaded_strings``. If the wrapper delegates
    ``append_string`` to the inner ``FileHistory``, the wrapper's list stays
    empty and up-arrow history is stale within the session.
    """
    history_file = tmp_path / "repl_history.txt"
    history = _SurrogateSafeFileHistory(str(history_file))

    history.append_string("first command")
    history.append_string("second command")

    # get_strings returns oldest-first; the just-typed entries must appear.
    strings = history.get_strings()
    assert "first command" in strings
    assert "second command" in strings


def test_store_string_still_sanitises_unpaired_surrogates(tmp_path) -> None:
    """The store_string override must strip unpaired surrogates before writing."""
    history_file = tmp_path / "repl_history.txt"
    history = _SurrogateSafeFileHistory(str(history_file))

    # A lone high surrogate (U+D83D) is invalid UTF-8 on its own.
    history.store_string("\ud83d broken emoji")

    # File must be writable and decodable — the surrogate is replaced, not
    # written as a raw lone surrogate byte. The replacement renders as ``?``
    # under the ``replace`` encode handler.
    raw = history_file.read_bytes()
    decoded = raw.decode("utf-8", errors="replace")
    assert "broken emoji" in decoded


def test_load_history_strings_round_trips_through_file(tmp_path) -> None:
    """Entries stored in one session are visible to a fresh wrapper instance."""
    history_file = tmp_path / "repl_history.txt"

    first = _SurrogateSafeFileHistory(str(history_file))
    first.append_string("persisted command")

    second = _SurrogateSafeFileHistory(str(history_file))
    loaded = second.load_history_strings()
    assert "persisted command" in list(loaded)
