"""quality_monitor_cli.main() branch dispatch + exit-code contract.

Argo CronWorkflows key run success/failure off these process exit codes, and
only --annotation-cycle was exercised before. These pin the --annotation-feedback
--argo-url guard, the result-driven codes (errored agents → 1), the --once
force-cycle codes (status ok → 0, else 1), and that the --once path awaits
monitor.close() in the same loop (the "Event loop is closed" guard against a
second asyncio.run).
"""

from __future__ import annotations

import sys

import pytest

from cogniverse_runtime import quality_monitor_cli as qm

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubMonitor:
    force_result = {"status": "ok"}
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        type(self).instances.append(self)

    async def force_optimization_cycle(self):
        return type(self).force_result

    async def run(self):
        return None

    async def close(self):
        self.closed = True


class _StubTelemetry:
    def __init__(self):
        self.config = type("_Cfg", (), {"provider_config": {}})()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(qm, "_build_phoenix_provider", lambda **k: None)
    monkeypatch.setattr(qm, "_workflow_pod_spec_from_env", lambda: None)
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda *a, **k: _StubTelemetry(),
    )
    monkeypatch.setattr(
        "cogniverse_evaluation.quality_monitor.QualityMonitor", _StubMonitor
    )
    _StubMonitor.force_result = {"status": "ok"}
    _StubMonitor.instances = []
    return monkeypatch


_BASE = ["--tenant-id", "acme", "--llm-model", "gemma"]


def _main_exit(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["quality_monitor_cli.py", *argv])
    with pytest.raises(SystemExit) as exc:
        qm.main()
    return exc.value.code


def test_annotation_feedback_without_argo_url_exits_2(patched):
    assert _main_exit(patched, [*_BASE, "--annotation-feedback"]) == 2


def test_annotation_feedback_success_exits_0(patched):
    async def _ok(**kwargs):
        return {"submitted": ["routing"], "errored_agents": []}

    patched.setattr(qm, "run_annotation_feedback_cycle", _ok)
    code = _main_exit(
        patched, [*_BASE, "--annotation-feedback", "--argo-url", "http://argo"]
    )
    assert code == 0


def test_annotation_feedback_errored_agents_exits_1(patched):
    async def _err(**kwargs):
        return {"submitted": [], "errored_agents": ["routing"]}

    patched.setattr(qm, "run_annotation_feedback_cycle", _err)
    code = _main_exit(
        patched, [*_BASE, "--annotation-feedback", "--argo-url", "http://argo"]
    )
    assert code == 1


def test_once_status_ok_exits_0_and_closes(patched):
    _StubMonitor.force_result = {"status": "ok"}
    code = _main_exit(patched, [*_BASE, "--once"])
    assert code == 0
    assert _StubMonitor.instances[-1].closed is True


def test_once_status_error_exits_1_and_closes(patched):
    _StubMonitor.force_result = {"status": "error", "reason": "eval failed"}
    code = _main_exit(patched, [*_BASE, "--once"])
    assert code == 1
    # close() ran in the same loop as the cycle even on the failure exit path.
    assert _StubMonitor.instances[-1].closed is True
