"""optimization_cli.main() branch dispatch + exit-code contract.

Argo CronWorkflow steps key success/failure off main()'s process exit code, so
each --mode must (1) canonicalize the tenant id once before dispatch, (2) call
the one worker its branch selects with the parsed args, and (3) exit 1 when the
worker result reports failure else 0. Only rollback / ab-compare / egress-netpol
were driven through the real main() (via subprocess); these pin the other modes'
dispatch wiring with the workers stubbed so no Argo/Phoenix/Vespa/LM is touched.
"""

from __future__ import annotations

import inspect
import json
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


class _NoisyRecorder(_Recorder):
    """Worker stub that prints to stdout before returning a result."""

    def __init__(self, result, noise: str = "setup noise"):
        super().__init__(result)
        self.noise = noise

    async def _coro(self):
        print(self.noise)
        return self._result


class _NoisySyncRecorder:
    """Sync worker stub that prints to stdout before returning a result."""

    def __init__(self, result, noise: str = "setup noise"):
        self._result = result
        self.noise = noise
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        print(self.noise)
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


# Resolved at import, before any test monkeypatches a worker, so the
# expectation follows each worker's real signature instead of a transcribed
# list that drifts when a worker gains or loses the parameter.
_EMBEDDER_URL_WORKERS = frozenset(
    attr
    for _, attr in _LOOKBACK_MODES
    if "embedder_url" in inspect.signature(getattr(oc, attr)).parameters
)


def _expected_lookback_kwargs(worker_attr: str, embedder_url: str | None):
    expected = {
        "tenant_id": "acme:acme",
        "lookback_hours": 2.5,
        "telemetry_otlp_endpoint": None,
    }
    if worker_attr in _EMBEDDER_URL_WORKERS:
        expected["embedder_url"] = embedder_url
    return expected


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
    assert rec.kwargs == _expected_lookback_kwargs(worker_attr, None)


@pytest.mark.parametrize("mode,worker_attr", _LOOKBACK_MODES)
def test_lookback_mode_forwards_embedder_url(monkeypatch, mode, worker_attr):
    """--embedder-url must reach the worker: it gates trainset capping and the
    embed_fn used for training selection (optimization_cli.py:1294,1313)."""
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, worker_attr, rec)
    code = _run_main(
        monkeypatch,
        [
            "--mode",
            mode,
            "--tenant-id",
            "acme",
            "--lookback-hours",
            "2.5",
            "--embedder-url",
            "http://denseon.test:8000",
        ],
    )
    assert code == 0
    assert rec.calls == 1
    assert rec.args == ()
    assert rec.kwargs == _expected_lookback_kwargs(
        worker_attr, "http://denseon.test:8000"
    )


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
    assert rec.kwargs == {
        "output_dir": "/tmp/reports_x",
        "lookback_hours": 3.0,
        "telemetry_otlp_endpoint": None,
    }


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
        "telemetry_otlp_endpoint": None,
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
        "optimizer_types": [
            "query_enhancement",
            "profile",
            "routing",
            "entity_extraction",
        ],
        "telemetry_otlp_endpoint": None,
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
            "profile, routing",
        ],
    )
    assert code == 0
    assert rec.kwargs == {
        "tenant_id": "acme:acme",
        "optimizer_types": ["profile", "routing"],
        "telemetry_otlp_endpoint": None,
    }


@pytest.mark.parametrize(
    ("result", "expected_code"),
    [
        (
            {
                "status": "success",
                "results": {
                    "profile": {"status": "success"},
                    "workflow": {"status": "success"},
                },
            },
            0,
        ),
        (
            {
                "status": "success",
                "results": {
                    "profile": {"status": "success"},
                    "workflow": {"status": "no_data"},
                },
            },
            0,
        ),
        (
            {
                "status": "no_data",
                "results": {
                    "profile": {"status": "no_data"},
                    "workflow": {"status": "no_data"},
                },
            },
            0,
        ),
        (
            {
                "status": "success",
                "results": {
                    "profile": {"status": "success"},
                    "workflow": {"status": "failed", "error": "backend down"},
                },
            },
            1,
        ),
        (
            {
                "status": "success",
                "results": {
                    "profile": {"status": "success"},
                    "workflow": {"status": "error", "error": "invalid result"},
                },
            },
            1,
        ),
    ],
    ids=[
        "all-success",
        "success-and-no-data",
        "all-no-data",
        "nested-failed",
        "nested-error",
    ],
)
def test_synthetic_result_controls_process_exit(monkeypatch, result, expected_code):
    rec = _Recorder(result)
    monkeypatch.setattr(oc, "run_synthetic_generation", rec)
    code = _run_main(monkeypatch, ["--mode", "synthetic", "--tenant-id", "acme"])
    assert code == expected_code
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


@pytest.mark.parametrize(
    ("mode", "worker_attr", "argv", "recorder_cls"),
    [
        (
            "synthetic",
            "run_synthetic_generation",
            ["--mode", "synthetic", "--tenant-id", "acme"],
            _NoisyRecorder,
        ),
        (
            "profile",
            "run_profile_optimization",
            ["--mode", "profile", "--tenant-id", "acme"],
            _NoisyRecorder,
        ),
        (
            "cleanup",
            "run_cleanup",
            ["--mode", "cleanup"],
            _NoisyRecorder,
        ),
        (
            "monthly-reports",
            "run_monthly_reports",
            ["--mode", "monthly-reports"],
            _NoisyRecorder,
        ),
        (
            "egress-netpol",
            "run_egress_netpol",
            [
                "--mode",
                "egress-netpol",
                "--output-dir",
                "/tmp/cogniverse-egress-test",
                "--service-map",
                "vespa=cogniverse/vespa-service:8080",
            ],
            _NoisySyncRecorder,
        ),
    ],
    ids=[
        "synthetic",
        "profile",
        "cleanup",
        "monthly-reports",
        "egress-netpol",
    ],
)
def test_main_keeps_setup_stdout_off_stdout(
    monkeypatch, capfd, mode, worker_attr, argv, recorder_cls
):
    """Noisy setup output stays on stderr and stdout remains one JSON document."""
    rec = recorder_cls(_OK, noise=f"{mode} setup banner")
    monkeypatch.setattr(oc, worker_attr, rec)
    code = _run_main(monkeypatch, argv)
    assert code == 0

    captured = capfd.readouterr()
    assert json.loads(captured.out) == _OK
    assert captured.out.strip() == json.dumps(_OK, indent=2, default=str)
    assert f"{mode} setup banner" in captured.err


class TestConfigErrorExitsCleanly:
    """A configuration error (BACKEND_URL unset) exits 1 with a one-line
    ``Error:`` message — never a raw traceback."""

    def test_missing_backend_url_exits_with_clean_error(self):
        import os
        import subprocess
        import sys as _sys

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("BACKEND_URL", "BACKEND_PORT")
        }
        result = subprocess.run(
            [
                _sys.executable,
                "-m",
                "cogniverse_runtime.optimization_cli",
                "--mode",
                "cleanup",
                "--tenant-id",
                "acme:acme",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "Traceback" not in result.stderr


_DENSEON_URL = "http://cogniverse-denseon:8000"


@pytest.mark.parametrize("mode,worker_attr", _LOOKBACK_MODES)
def test_lookback_mode_derives_embedder_url_from_inference_service_urls(
    monkeypatch, mode, worker_attr
):
    """The scheduled (Argo) path passes no --embedder-url; main() must hand the
    denseon entry of INFERENCE_SERVICE_URLS to the worker, or any pool above
    trainset_cap fails with 'training selection requires --embedder-url'."""
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        json.dumps(
            {"denseon": _DENSEON_URL, "gliner": "http://cogniverse-gliner:8080"}
        ),
    )
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, worker_attr, rec)
    code = _run_main(
        monkeypatch, ["--mode", mode, "--tenant-id", "acme", "--lookback-hours", "2.5"]
    )
    assert code == 0
    assert rec.calls == 1
    assert rec.kwargs == _expected_lookback_kwargs(worker_attr, _DENSEON_URL)


@pytest.mark.parametrize("mode,worker_attr", _LOOKBACK_MODES)
def test_explicit_embedder_url_overrides_inference_service_urls(
    monkeypatch, mode, worker_attr
):
    monkeypatch.setenv("INFERENCE_SERVICE_URLS", json.dumps({"denseon": _DENSEON_URL}))
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, worker_attr, rec)
    code = _run_main(
        monkeypatch,
        [
            "--mode",
            mode,
            "--tenant-id",
            "acme",
            "--lookback-hours",
            "2.5",
            "--embedder-url",
            "http://denseon.test:8000",
        ],
    )
    assert code == 0
    assert rec.calls == 1
    assert rec.kwargs == _expected_lookback_kwargs(
        worker_attr, "http://denseon.test:8000"
    )


@pytest.mark.parametrize("mode,worker_attr", _LOOKBACK_MODES)
def test_inference_service_urls_without_denseon_leaves_embedder_url_unset(
    monkeypatch, mode, worker_attr
):
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        json.dumps({"gliner": "http://cogniverse-gliner:8080"}),
    )
    rec = _Recorder(_OK)
    monkeypatch.setattr(oc, worker_attr, rec)
    code = _run_main(
        monkeypatch, ["--mode", mode, "--tenant-id", "acme", "--lookback-hours", "2.5"]
    )
    assert code == 0
    assert rec.calls == 1
    assert rec.kwargs == _expected_lookback_kwargs(worker_attr, None)
