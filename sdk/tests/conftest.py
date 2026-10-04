"""Shared pytest fixtures for sdk tests.

Only hermeticity guards live here; fixtures for specific subsystems belong in
``sdk/tests/<subsystem>/conftest.py``.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _settings_hermeticity_guard() -> Any:
    """Snapshot/restore ``sdk.config.settings`` and fail on leaks.

    Mirrors ``kernel/tests/conftest.py`` — see that module for the rationale.
    """

    from sdk.config import settings

    snapshot = settings.model_copy(deep=True)
    yield

    leaked: dict[str, tuple[Any, Any]] = {}
    for field, value in settings.__dict__.items():
        if field not in snapshot.__dict__:
            leaked[field] = (None, value)
        elif snapshot.__dict__[field] != value:
            leaked[field] = (snapshot.__dict__[field], value)

    for field, value in snapshot.__dict__.items():
        object.__setattr__(settings, field, value)

    if leaked:
        diff = "\n".join(
            f"  {field}: snapshot={before!r} leaked={after!r}"
            for field, (before, after) in sorted(leaked.items())
        )
        raise AssertionError(
            "Test mutated sdk.config.settings without restoring it.\n"
            "The freeze requires a hermetic suite; wrap mutations in\n"
            "try/finally (or use monkeypatch) so later tests are isolated.\n"
            f"Leaked fields:\n{diff}"
        )
