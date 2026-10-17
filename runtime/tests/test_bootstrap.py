"""Tests for the ``runtime.bootstrap`` composition root."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from runtime.bootstrap import _configure_logging, build_runtime
from runtime.config import HostConfig
from sdk.thread import DEFAULT_REDACTION_PATTERNS, ThreadRedactionPolicy


class _RecordingConfigurator:
    def __init__(self) -> None:
        self.calls: list[HostConfig] = []

    def __call__(self, config: HostConfig) -> None:
        self.calls.append(config)


def _config(**overrides: Any) -> HostConfig:
    """Build a real ``HostConfig`` via ``from_env`` with the given field overrides."""
    import dataclasses

    config = HostConfig.from_env()
    return dataclasses.replace(config, **overrides) if overrides else config


def _install_bootstrap_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: HostConfig,
    pack_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Install every eager-wiring fake ``build_runtime`` needs; return probes."""
    probes: dict[str, Any] = {"redact": [], "registers": 0}

    fake_runtime_cls = lambda cfg, in_process_mcp=False: SimpleNamespace(  # noqa: E731
        config=cfg,
        in_process_mcp=in_process_mcp,
        session=SimpleNamespace(state_snapshot=None),
        resources=SimpleNamespace(
            set_mcp_configs=lambda w, c: None,
            set_proxy_configs=lambda p: None,
        ),
    )

    monkeypatch.setattr("runtime.mcp_pool.compile_mcp_configs", lambda r, discovery=None: ([], []))
    monkeypatch.setattr("runtime.mcp_pool.compile_proxy_configs", lambda r, discovery=None: [])
    monkeypatch.setattr("runtime.mcp_pool.configure_runtime", lambda r: None)
    monkeypatch.setattr("runtime.mcp_pool.discover_framework", lambda r: object())
    monkeypatch.setattr("runtime.mcp_pool.setup_framework", lambda r, discovery=None: None)
    monkeypatch.setattr("runtime.runtime.HostRuntime", fake_runtime_cls)
    monkeypatch.setattr(
        "runtime.state_provider.load_state_snapshot",
        lambda *, explicit_path, data_dir: {"source": explicit_path},
    )
    monkeypatch.setattr(
        "runtime.content_roots.discover_pack_redaction_patterns",
        lambda data_dir, always_on_packs: list(pack_patterns or []),
    )
    redact_sink = probes["redact"]
    monkeypatch.setattr("runtime.thread.set_thread_redaction_policy", redact_sink.append)
    monkeypatch.setattr(
        "runtime.host_commands.register_builtin_host_commands",
        lambda: probes.__setitem__("registers", probes["registers"] + 1),
    )
    monkeypatch.setattr(HostConfig, "from_env", classmethod(lambda cls, **kw: config))
    return probes


# ── _configure_logging ───────────────────────────────────────────────


def test_configure_logging_uses_injected_configurator() -> None:
    recorder = _RecordingConfigurator()
    config = _config(host_kind="cli")

    _configure_logging(config, recorder)

    assert recorder.calls == [config]


def test_configure_logging_defaults_to_sdk_configurator(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "sdk.logging_setup.configure_logging",
        lambda **kwargs: called.append(kwargs),
    )
    config = _config(host_kind="cli")

    _configure_logging(config, None)

    assert called == [
        {
            "log_path": config.resolved_host_log_path,
            "level": config.effective_log_level,
            "debug_modules": list(config.debug_log_modules),
            "host_kind": config.host_kind,
        }
    ]


# ── build_runtime ────────────────────────────────────────────────────


def test_build_runtime_full_eager_path(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(thread_redaction_enabled=True)
    probes = _install_bootstrap_fakes(
        monkeypatch,
        config=config,
        pack_patterns=["pack-extra", "pack-extra"],  # duplicates are deduped
    )

    runtime = build_runtime(host_kind="cli", state_snapshot_path="/x/state.json")

    assert runtime.config.host_kind == "cli"
    assert runtime.in_process_mcp is False
    assert runtime.session.state_snapshot == {"source": "/x/state.json"}
    assert probes["registers"] == 1
    policy: ThreadRedactionPolicy = probes["redact"][0]
    assert isinstance(policy, ThreadRedactionPolicy)
    # Framework defaults merged with the (deduped) pack pattern.
    assert pattern_count(policy) == len(DEFAULT_REDACTION_PATTERNS) + 1
    assert policy.redact("pack-extra text") != "pack-extra text"


def test_build_runtime_env_override_replaces_pack_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        thread_redaction_enabled=True,
        thread_redaction_patterns=("custom-only",),
    )
    probes = _install_bootstrap_fakes(
        monkeypatch,
        config=config,
        pack_patterns=["pack-pat"],
    )

    build_runtime(host_kind="web")

    policy: ThreadRedactionPolicy = probes["redact"][0]
    # Env override is a hard cutover: only the configured pattern is compiled —
    # no framework defaults, no pack-contributed patterns.
    assert pattern_count(policy) == 1
    assert policy.redact("custom-only") != "custom-only"
    assert policy.redact("pack-pat") == "pack-pat"


def test_build_runtime_redaction_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(thread_redaction_enabled=False)
    probes = _install_bootstrap_fakes(monkeypatch, config=config)

    build_runtime(host_kind="cli")

    assert probes["redact"] == [None]


def test_build_runtime_in_process_mcp_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(thread_redaction_enabled=False)
    _install_bootstrap_fakes(monkeypatch, config=config)

    runtime = build_runtime(host_kind="cli", in_process_mcp=True)

    assert runtime.in_process_mcp is True


def pattern_count(policy: ThreadRedactionPolicy) -> int:
    return len(policy._compiled)
