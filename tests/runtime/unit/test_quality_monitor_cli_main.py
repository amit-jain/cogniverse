"""quality_monitor_cli.main() branch dispatch + serving-loop retry contract.

Argo CronWorkflows key run success/failure off the one-shot exit codes, and
only --annotation-cycle was exercised before. These pin the --annotation-feedback
--argo-url guard, the result-driven codes (errored agents → 1), the --once
force-cycle codes (status ok → 0, else 1), and that the serving path keeps the
process alive when the monitor loop raises or returns unexpectedly.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpr
import pytest

from cogniverse_runtime import quality_monitor_cli as qm

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubMonitor:
    force_result = {"status": "ok"}
    run_exc: BaseException | None = None
    run_side_effects: list = []
    run_calls = 0
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        type(self).instances.append(self)

    async def force_optimization_cycle(self):
        return type(self).force_result

    async def run(self):
        type(self).run_calls += 1
        if type(self).run_side_effects:
            effect = type(self).run_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect
        if type(self).run_exc is not None:
            raise type(self).run_exc
        return None

    async def close(self):
        self.closed = True


class _StubTelemetry:
    def __init__(self):
        self.config = type("_Cfg", (), {"provider_config": {}})()


class _BreakLoop(BaseException):
    """Sentinel that forces the monitor runner to stop after a retry."""


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
    monkeypatch.setattr(
        qm, "_wait_for_runtime_search", lambda **kwargs: None, raising=False
    )
    _StubMonitor.force_result = {"status": "ok"}
    _StubMonitor.run_exc = None
    _StubMonitor.run_side_effects = []
    _StubMonitor.run_calls = 0
    _StubMonitor.instances = []
    return monkeypatch


_BASE = ["--tenant-id", "acme", "--llm-model", "gemma"]


def _main_exit(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["quality_monitor_cli.py", *argv])
    with pytest.raises(SystemExit) as exc:
        qm.main()
    return exc.value.code


def _main_run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["quality_monitor_cli.py", *argv])
    return qm.main()


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


def test_main_waits_for_runtime_search_before_constructing_monitor(patched):
    calls = []

    def wait_for_search(**kwargs):
        calls.append(kwargs)

    patched.setattr(qm, "_wait_for_runtime_search", wait_for_search, raising=False)

    assert _main_exit(patched, [*_BASE, "--once"]) == 0
    assert calls == [
        {
            "runtime_url": "http://localhost:28000",
            "tenant_id": "acme:acme",
            "golden_dataset_path": (
                "data/testset/evaluation/sample_videos_retrieval_queries.json"
            ),
            "timeout_seconds": 300.0,
            "poll_interval_seconds": 2.0,
        }
    ]


def test_main_uses_env_phoenix_url_for_monitor_and_provider(patched, monkeypatch):
    seen = {}

    def build_phoenix_provider(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setenv("TELEMETRY_HTTP_ENDPOINT", "http://phoenix-env:6006")
    patched.setattr(qm, "_build_phoenix_provider", build_phoenix_provider)

    assert _main_exit(patched, [*_BASE, "--once"]) == 0
    assert seen == {
        "tenant_id": "acme:acme",
        "http_endpoint": "http://phoenix-env:6006",
    }
    assert _StubMonitor.instances[-1].kwargs["phoenix_http_endpoint"] == (
        "http://phoenix-env:6006"
    )


def test_once_status_error_exits_1_and_closes(patched):
    _StubMonitor.force_result = {"status": "error", "reason": "eval failed"}
    code = _main_exit(patched, [*_BASE, "--once"])
    assert code == 1
    # close() ran in the same loop as the cycle even on the failure exit path.
    assert _StubMonitor.instances[-1].closed is True


def _silence_annotation_loop(monkeypatch):
    async def _noop_cycle(**kwargs):
        return {}

    monkeypatch.setattr(qm, "run_annotation_cycle", _noop_cycle)
    monkeypatch.setattr(qm, "_load_automation_rules", lambda tenant_id: None)


def test_default_monitor_run_error_retries_and_logs(patched, caplog):
    _silence_annotation_loop(patched)
    sleep_calls = []

    def fake_sleep(delay):
        sleep_calls.append(delay)

    patched.setattr(qm.time, "sleep", fake_sleep)
    _StubMonitor.run_side_effects = [RuntimeError("boom"), _BreakLoop()]
    with caplog.at_level("ERROR"):
        with pytest.raises(_BreakLoop):
            _main_run(patched, _BASE)
    assert sleep_calls == [1.0]
    assert len(_StubMonitor.instances) == 2
    assert [instance.closed for instance in _StubMonitor.instances] == [True, True]
    assert [record.levelname for record in caplog.records] == ["ERROR"]
    assert [record.message for record in caplog.records] == [
        "Quality monitor loop crashed; retrying in 1.0s"
    ]


def test_default_monitor_return_retries_and_logs(patched, caplog):
    _silence_annotation_loop(patched)
    sleep_calls = []

    def fake_sleep(delay):
        sleep_calls.append(delay)

    patched.setattr(qm.time, "sleep", fake_sleep)
    _StubMonitor.run_side_effects = [None, _BreakLoop()]
    with caplog.at_level("ERROR"):
        with pytest.raises(_BreakLoop):
            _main_run(patched, _BASE)
    assert sleep_calls == [1.0]
    assert len(_StubMonitor.instances) == 2
    assert [instance.closed for instance in _StubMonitor.instances] == [True, True]
    assert [record.levelname for record in caplog.records] == ["ERROR"]
    assert [record.message for record in caplog.records] == [
        "Quality monitor loop returned unexpectedly; retrying in 1.0s"
    ]


def test_default_monitor_keyboard_interrupt_exits_0(patched):
    _silence_annotation_loop(patched)
    _StubMonitor.run_exc = KeyboardInterrupt()
    code = _main_exit(patched, _BASE)
    assert code == 0
    assert _StubMonitor.instances[-1].closed is True


def test_startup_dependency_retry_isolated_across_concurrent_waiters():
    waiter_count = 8
    first_attempts = threading.Barrier(waiter_count)
    thread_state = threading.local()
    attempt_count = 0
    attempt_lock = threading.Lock()
    manager = object()

    def get_manager():
        nonlocal attempt_count
        with attempt_lock:
            attempt_count += 1
        thread_attempt = getattr(thread_state, "attempt", 0) + 1
        thread_state.attempt = thread_attempt
        if thread_attempt == 1:
            first_attempts.wait(timeout=5)
            raise httpr.ConnectError("Vespa is still starting")
        return manager

    def wait_for_manager():
        return qm._wait_for_telemetry_manager(
            get_manager=get_manager,
            timeout_seconds=1,
            poll_interval_seconds=0,
        )

    with ThreadPoolExecutor(max_workers=waiter_count) as pool:
        results = list(pool.map(lambda _: wait_for_manager(), range(waiter_count)))

    assert results == [manager] * waiter_count
    assert attempt_count == waiter_count * 2


def test_startup_dependency_retries_past_timeout_and_logs_exact_failure_once(
    caplog,
):
    expected_manager = object()
    attempts = 0

    def get_manager():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpr.ConnectError("Vespa config query refused")
        return expected_manager

    with caplog.at_level("WARNING"):
        manager = qm._wait_for_telemetry_manager(
            get_manager=get_manager,
            timeout_seconds=0,
            poll_interval_seconds=0,
        )

    assert manager is expected_manager
    assert attempts == 3
    assert [record.levelname for record in caplog.records] == [
        "ERROR",
        "WARNING",
    ]
    assert [record.message for record in caplog.records] == [
        "Telemetry configuration dependency was not ready after 1 attempt "
        "within 0.0s: ConnectError: Vespa config query refused; keeping the "
        "sidecar alive and retrying",
        "Telemetry configuration dependency is still not ready (attempt 2, "
        "retrying in 0.0s): ConnectError: Vespa config query refused",
    ]


def test_runtime_search_readiness_uses_real_request_and_exact_payload(tmp_path):
    golden = tmp_path / "golden.json"
    golden.write_text(
        '[{"query":"man lifting a barbell","expected_videos":["video-7"]}]',
        encoding="utf-8",
    )
    requests_seen = []

    class SearchHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            import json

            body = json.loads(
                self.rfile.read(int(self.headers["Content-Length"])).decode()
            )
            requests_seen.append({"path": self.path, "body": body})
            if len(requests_seen) == 1:
                payload = b'{"detail":"starting"}'
                self.send_response(503)
            elif len(requests_seen) == 2:
                payload = b'{"unexpected":[]}'
                self.send_response(200)
            else:
                payload = b'{"results":[{"source_id":"video-7","score":1.0}]}'
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SearchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = qm._wait_for_runtime_search(
            runtime_url=f"http://127.0.0.1:{server.server_port}",
            tenant_id="acme:production",
            golden_dataset_path=str(golden),
            timeout_seconds=2,
            poll_interval_seconds=0.001,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    expected_request = {
        "path": "/search/",
        "body": {
            "query": "man lifting a barbell",
            "profile": "video_colpali_smol500_mv_frame",
            "top_k": 1,
            "tenant_id": "acme:production",
        },
    }
    assert requests_seen == [expected_request, expected_request, expected_request]
    assert result == {"results": [{"source_id": "video-7", "score": 1.0}]}


def test_runtime_search_waiters_keep_independent_attempt_counts():
    waiter_count = 8
    first_attempts = threading.Barrier(waiter_count)
    local_state = threading.local()
    observed_attempts = []
    lock = threading.Lock()

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"results": []}

    def post(*args, **kwargs):
        attempt = getattr(local_state, "attempt", 0) + 1
        local_state.attempt = attempt
        if attempt == 1:
            first_attempts.wait(timeout=5)
            raise ConnectionError("runtime starting")
        with lock:
            observed_attempts.append(attempt)
        return Response()

    def wait():
        return qm._wait_for_runtime_search(
            runtime_url="http://runtime",
            tenant_id="acme:production",
            golden_queries=[{"query": "probe"}],
            timeout_seconds=2,
            poll_interval_seconds=0,
            post=post,
        )

    with ThreadPoolExecutor(max_workers=waiter_count) as pool:
        results = list(pool.map(lambda _: wait(), range(waiter_count)))

    assert results == [{"results": []}] * waiter_count
    assert observed_attempts == [2] * waiter_count


def test_runtime_search_keeps_retrying_past_startup_timeout():
    attempts = 0

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"results": [{"source_id": "video-7", "score": 1.0}]}

    def post(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError(f"runtime not ready attempt {attempts}")
        return Response()

    observed = {}
    try:
        observed["result"] = qm._wait_for_runtime_search(
            runtime_url="http://runtime",
            tenant_id="acme:production",
            golden_queries=[{"query": "probe"}],
            timeout_seconds=0,
            poll_interval_seconds=0,
            post=post,
        )
    except Exception as exc:
        observed["result"] = {
            "error": type(exc).__name__,
            "message": str(exc),
        }
    observed["attempts"] = attempts

    assert observed == {
        "result": {"results": [{"source_id": "video-7", "score": 1.0}]},
        "attempts": 3,
    }


class TestConfigErrorExitsCleanly:
    """A configuration error (BACKEND_URL unset) exits 1 with a one-line
    ``Error:`` message — never a raw traceback."""

    def test_missing_backend_url_exits_with_clean_error(self):
        import os
        import subprocess

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("BACKEND_URL", "BACKEND_PORT")
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cogniverse_runtime.quality_monitor_cli",
                "--tenant-id",
                "acme:acme",
                "--llm-model",
                "test-model",
                "--once",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "BACKEND_URL" in result.stderr
        assert "Traceback" not in result.stderr
