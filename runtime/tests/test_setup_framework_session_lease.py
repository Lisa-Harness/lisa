"""Slice A integration tests: shared-session lease ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from runtime.testing.runtime_factory import build_test_runtime


def test_setup_framework_releases_lease_when_build_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing build releases its lease and clears the runtime field."""

    from runtime import mcp_pool

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("build failed")

    monkeypatch.setattr(mcp_pool, "_setup_framework_impl", _boom)
    runtime = build_test_runtime(tmp_path)

    with pytest.raises(RuntimeError, match="build failed"):
        mcp_pool.setup_framework(runtime)

    assert runtime.resources.shared_session_lease is None

    from kernel.context._shared_stdio import close_shared_sessions

    close_shared_sessions()


@pytest.mark.asyncio
async def test_runtime_aclose_releases_shared_session_lease(tmp_path: Path) -> None:
    """Runtime teardown releases the owning build's sessions."""

    from kernel.context._shared_stdio import acquire_lease, close_shared_sessions

    runtime = build_test_runtime(tmp_path)
    lease = acquire_lease()
    runtime.resources.shared_session_lease = lease

    await runtime.resources.aclose()

    assert lease.released
    assert runtime.resources.shared_session_lease is None

    close_shared_sessions()
