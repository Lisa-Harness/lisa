"""Tests for the ``/policy`` command — render, mode dial, bare toggle.

``/policy`` is the host-side surface for the approval policy engine:

- bare ``/policy`` renders the active document (mode, guards, rule lists).
- ``/policy mode <m>`` persists a new fallback dial to ``policy.yaml``
  (preserving any rule lists) and the reply confirms it.
- ``/policy bare on|off`` toggles ``bare_requires_approval``; bad input
  echoes the current state and does not write.
- Unknown subcommands return usage, never a crash.

Writes go through ``host.policy.write_policy`` so the document stays
hand-editable; the engine re-reads the file on every approval request, so an
edit must be observable by the very next ``load_policy``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from host.commands import handle_policy_command, render_policy_document
from host.policy import PolicyDocument, load_policy


@pytest.fixture(autouse=True)
def _clean_timezone_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop LISA_TIMEZONE/TZ before each test.

    The full suite's env-propagation tests export ``LISA_TIMEZONE`` into
    ``os.environ`` (compile_mcp_configs writes the host's resolved zone for
    subprocesses). On hosts whose system-local tz is a POSIX alias
    (``EDT`` → ``EST5EDT``) that exported value is not re-resolvable by
    ``HostConfig.from_env``, which poisons every later ``build_test_runtime``
    call. This file builds runtimes directly, so scrub the var first.
    """
    for env_name in ("LISA_TIMEZONE", "TZ", "TIMEZONE", "LOCAL_TIMEZONE"):
        monkeypatch.delenv(env_name, raising=False)


def _make_runtime(tmp_path: Path) -> object:
    from host.testing import build_test_runtime

    models_yaml = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
        ]
    )
    return build_test_runtime(tmp_path, models_yaml=models_yaml)


# ── Render ────────────────────────────────────────────────────────────


def test_render_policy_document_defaults(tmp_path: Path) -> None:
    """A missing policy.yaml renders the fail-closed defaults."""
    runtime = _make_runtime(tmp_path)
    text = render_policy_document(runtime)  # type: ignore[arg-type]
    assert "## Approval Policy" in text
    assert "mode: **ask**" in text
    assert "sandbox_downgrade: true" in text
    assert "bare_requires_approval: true" in text
    assert "deny rules (0)" in text
    assert "(none)" in text


def _write_doc(tmp_path: Path, doc: PolicyDocument) -> None:
    """Serialize a PolicyDocument to policy.yaml (mode as its string value)."""
    payload = doc.model_dump()
    payload["mode"] = doc.mode.value
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_render_policy_document_reflects_file(tmp_path: Path) -> None:
    """An existing policy.yaml's mode and rules appear in the render."""
    runtime = _make_runtime(tmp_path)
    doc = PolicyDocument(
        mode="trust",
        allow=[{"tool": "bash_run", "command_prefix": "git status"}],
    )
    _write_doc(tmp_path, doc)
    text = render_policy_document(runtime)  # type: ignore[arg-type]
    assert "mode: **trust**" in text
    assert "allow rules (1)" in text
    assert "git status" in text


# ── Mode dial ─────────────────────────────────────────────────────────


def test_policy_mode_set_persists_and_preserves_rules(tmp_path: Path) -> None:
    """mode trust writes the dial and keeps existing rule lists."""
    runtime = _make_runtime(tmp_path)
    doc = PolicyDocument(deny=[{"tool": "bash_run"}])
    _write_doc(tmp_path, doc)
    reply = handle_policy_command(runtime, "/policy mode trust", "/policy mode trust")  # type: ignore[arg-type]
    assert "**trust**" in reply
    reloaded = load_policy(tmp_path)
    assert reloaded.mode.value == "trust"
    # The pre-existing deny rule survived the write.
    assert any(r.tool == "bash_run" for r in reloaded.deny)


def test_policy_mode_unknown_is_rejected_without_write(tmp_path: Path) -> None:
    """An invalid mode returns usage and leaves the document untouched."""
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy mode banana", "/policy mode banana")  # type: ignore[arg-type]
    assert "Unknown mode" in reply
    assert not (tmp_path / "policy.yaml").exists()


# ── Bare toggle ───────────────────────────────────────────────────────


def test_policy_bare_off_toggles_guard(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy bare off", "/policy bare off")  # type: ignore[arg-type]
    assert "**false**" in reply
    assert load_policy(tmp_path).bare_requires_approval is False


def test_policy_bare_bad_input_reports_current(tmp_path: Path) -> None:
    """Bad bare input echoes the current state and writes nothing."""
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy bare", "/policy bare")  # type: ignore[arg-type]
    assert "currently true" in reply
    assert not (tmp_path / "policy.yaml").exists()


# ── Unknown / usage ───────────────────────────────────────────────────


def test_policy_unknown_subcommand_returns_usage(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy frobnicate", "/policy frobnicate")  # type: ignore[arg-type]
    assert "Usage: /policy" in reply


def test_policy_mode_roundtrip_via_engine(tmp_path: Path) -> None:
    """End-to-end: set trust, then the engine's decide() honors it.

    A write-mode 'write' action would prompt under ask but is allowed under
    trust (no rules, no containment consulted for non-execute effects).
    """
    from host.policy import decide

    runtime = _make_runtime(tmp_path)
    from kernel.approval import ApprovalIdentity, build_approval_invocation

    identity = ApprovalIdentity(workspace_id="w", user_slug="u")
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="write_file",
        pack_id="lisa-pack-core",
        effect="write",
        parameters={"path": "x.txt"},
        identity=identity,
        pack_config={},
        safe_preview="write x.txt",
    )
    # Default ask mode -> prompt.
    policy = load_policy(tmp_path)
    assert decide(policy=policy, invocation=invocation, data_dir=tmp_path).verdict == "ask"
    handle_policy_command(runtime, "/policy mode trust", "/policy mode trust")  # type: ignore[arg-type]
    policy = load_policy(tmp_path)
    assert decide(policy=policy, invocation=invocation, data_dir=tmp_path).verdict == "allow"
