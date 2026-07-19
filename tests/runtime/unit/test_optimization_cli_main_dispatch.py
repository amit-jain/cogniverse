"""optimization_cli.main() branch dispatch + exit-code contract.

Argo CronWorkflow steps key success/failure off main()'s process exit code, so
each --mode must (1) canonicalize the tenant id once before dispatch, (2) call
the one worker its branch selects with the parsed args, and (3) exit 1 when the
worker result reports failure else 0. Only rollback / ab-compare / egress-netpol
were driven through the real main() (via subprocess); these pin the other modes'
dispatch wiring with the workers stubbed so no Argo/Phoenix/Vespa/LM is touched.
"""

from __future__ import annotations

import sys

import pytest

from cogniverse_runtime import optimization_cli as oc

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Recorder:
    """Stub worker that records its call args and returns a controlled result.

    main() invokes each worker as ``asyncio.run(worker(...))``, so ``__call__``
    records synchronously and hands back a coroutine for asyncio.run to await.
    """

    def __init__(self, result):
        self._result = result
        self.calls = 0
        self.args: tuple = ()
        self.kwargs: dict = {}

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.args = args
        self.kwargs = kwargs
        return self._coro()

    async def _coro(self):
        return self._result


_OK = {"status": "success"}
_FAIL = {"status": "error", "reason": "worker failed"}


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["optimization_cli.py", *argv])
    with pytest.raises(SystemExit) as exc:
        oc.main()
    return exc.value.code


# Modes whose branch calls run_*(tenant_id=..., lookback_hours=...) identically.
_LOOKBACK_MODES = [
    ("simba", "run_simba_optimization"),
    ("workflow", "run_workflow_optimization"),
    ("gateway-thresholds", "run_gateway_thresholds_optimization"),
    ("online-routing-eval", "run_online_routing_evaluation"),
    ("online-eval", "run_online_evaluation"),
    ("profile", "run_profile_optimization"),
    ("entity-extraction", "run_entity_extraction_optimization"),
]


@pytest.mark.parametrize("mode,worker_attr", _LOOKBACK_MODES)
def test_lookback_mode_success_dispatch(monkeypatch, mode, worker_attr):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, worker_attr, rec)
    code = _run_main(
        monkeypatch, ["--mode", mode, "--tenant-id", "acme", "--lookback-hours", "2.5"]
    )
    assert code == 0
    assert rec.calls == 1
    assert rec.args == ()
    assert rec.kwargs == {"tenant_id": "acme:acme", "lookback_hours": 2.5}


@pytest.mark.parametrize("mode,worker_attr", _LOOKBACK_MODES)
def test_lookback_mode_failure_result_exits_1(monkeypatch, mode, worker_attr):
    rec = _Recorder(_FAIL)
    monkeypatch.setattr(oc, worker_attr, rec)
    code = _run_main(
        monkeypatch, ["--mode", mode, "--tenant-id", "acme", "--lookback-hours", "1.0"]
    )
    assert code == 1
    assert rec.calls == 1
    assert rec.kwargs["tenant_id"] == "acme:acme"


def test_cleanup_success_dispatch(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_cleanup", rec)
    code = _run_main(
        monkeypatch,
        [
            "--mode",
            "cleanup",
            "--tenant-id",
            "acme",
            "--log-retention-days",
            "5",
            "--memory-retention-days",
            "15",
        ],
    )
    assert code == 0
    assert rec.calls == 1
    # run_cleanup(tenant_id, log_retention_days, memory_retention_days) positional
    assert rec.args == ("acme:acme", 5, 15)
    assert rec.kwargs == {}


def test_cleanup_failure_result_exits_1(monkeypatch):
    rec = _Recorder(_FAIL)
    monkeypatch.setattr(oc, "run_cleanup", rec)
    code = _run_main(monkeypatch, ["--mode", "cleanup", "--tenant-id", "acme"])
    assert code == 1
    assert rec.args[0] == "acme:acme"


def test_cleanup_without_tenant_runs_global(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_cleanup", rec)
    code = _run_main(monkeypatch, ["--mode", "cleanup"])
    assert code == 0
    assert rec.calls == 1
    # cleanup is tenant-optional: omitted --tenant-id runs globally with None.
    assert rec.args == (None, 7, 30)


def test_monthly_reports_success_dispatch(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_monthly_reports", rec)
    code = _run_main(
        monkeypatch,
        [
            "--mode",
            "monthly-reports",
            "--reports-output-dir",
            "/tmp/reports_x",
            "--lookback-hours",
            "3.0",
        ],
    )
    assert code == 0
    assert rec.calls == 1
    # monthly-reports is global: no tenant_id forwarded.
    assert rec.args == ()
    assert rec.kwargs == {"output_dir": "/tmp/reports_x", "lookback_hours": 3.0}


def test_monthly_reports_failure_result_exits_1(monkeypatch):
    rec = _Recorder(_FAIL)
    monkeypatch.setattr(oc, "run_monthly_reports", rec)
    code = _run_main(monkeypatch, ["--mode", "monthly-reports"])
    assert code == 1
    assert rec.calls == 1


def test_triggered_success_dispatch(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_triggered_optimization", rec)
    code = _run_main(
        monkeypatch,
        [
            "--mode",
            "triggered",
            "--tenant-id",
            "acme",
            "--agents",
            "search_agent, routing_agent",
            "--trigger-dataset",
            "trig_ds",
        ],
    )
    assert code == 0
    assert rec.calls == 1
    assert rec.kwargs == {
        "tenant_id": "acme:acme",
        "agents": ["search_agent", "routing_agent"],
        "trigger_dataset": "trig_ds",
    }


def test_triggered_failure_result_exits_1(monkeypatch):
    rec = _Recorder(_FAIL)
    monkeypatch.setattr(oc, "run_triggered_optimization", rec)
    code = _run_main(
        monkeypatch,
        [
            "--mode",
            "triggered",
            "--tenant-id",
            "acme",
            "--agents",
            "search_agent",
            "--trigger-dataset",
            "trig_ds",
        ],
    )
    assert code == 1
    assert rec.kwargs["tenant_id"] == "acme:acme"


def test_triggered_missing_args_exits_2(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_triggered_optimization", rec)
    # No --agents / --trigger-dataset: argparse.error exits 2 before dispatch.
    code = _run_main(monkeypatch, ["--mode", "triggered", "--tenant-id", "acme"])
    assert code == 2
    assert rec.calls == 0


def test_synthetic_default_optimizers_dispatch(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_synthetic_generation", rec)
    code = _run_main(monkeypatch, ["--mode", "synthetic", "--tenant-id", "acme"])
    assert code == 0
    assert rec.calls == 1
    assert rec.kwargs == {
        "tenant_id": "acme:acme",
        "optimizer_types": ["query_enhancement", "profile", "workflow"],
    }


def test_synthetic_agents_override_optimizers(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_synthetic_generation", rec)
    code = _run_main(
        monkeypatch,
        [
            "--mode",
            "synthetic",
            "--tenant-id",
            "acme",
            "--agents",
            "profile, workflow",
        ],
    )
    assert code == 0
    assert rec.kwargs == {
        "tenant_id": "acme:acme",
        "optimizer_types": ["profile", "workflow"],
    }


def test_synthetic_failure_result_exits_1(monkeypatch):
    rec = _Recorder(_FAIL)
    monkeypatch.setattr(oc, "run_synthetic_generation", rec)
    code = _run_main(monkeypatch, ["--mode", "synthetic", "--tenant-id", "acme"])
    assert code == 1
    assert rec.kwargs["tenant_id"] == "acme:acme"


def test_missing_tenant_for_lookback_mode_exits_2(monkeypatch):
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, "run_simba_optimization", rec)
    # simba requires --tenant-id: main() calls parser.error -> exit 2.
    code = _run_main(monkeypatch, ["--mode", "simba"])
    assert code == 2
    assert rec.calls == 0


def test_batch_shaped_failure_result_exits_1(monkeypatch):
    # Result with no top-level status but a failed per-agent entry -> exit 1
    # (the _run_failed batch branch, distinct from the status branch above).
    rec = _Recorder({"search_agent": {"status": "failed"}, "routing_agent": {"ok": 1}})
    monkeypatch.setattr(oc, "run_profile_optimization", rec)
    code = _run_main(
        monkeypatch,
        ["--mode", "profile", "--tenant-id", "acme", "--lookback-hours", "1"],
    )
    assert code == 1


def test_non_dict_result_exits_0(monkeypatch):
    # A non-dict worker result is not a failure signal -> exit 0.
    rec = _Recorder(["report-1", "report-2"])
    monkeypatch.setattr(oc, "run_online_evaluation", rec)
    code = _run_main(
        monkeypatch,
        ["--mode", "online-eval", "--tenant-id", "acme", "--lookback-hours", "1"],
    )
    assert code == 0
