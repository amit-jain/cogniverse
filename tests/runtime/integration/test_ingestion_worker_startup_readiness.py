"""Ingestion-worker startup against a config store that is down at boot.

The worker's first config read happens before it touches Redis. When the
Vespa-backed store is unreachable that read raises
``ConfigStoreUnavailableError`` once the store's own retry budget (five
30-second visits) is spent; the worker must keep waiting through that and
serve once the store answers, the way the quality-monitor sidecar does.
A paused container is the outage shape seen in production: connections are
accepted and never answered, so every visit runs to its read timeout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import socket
import subprocess
import threading
import time

import pytest
import requests

from cogniverse_runtime.ingestion_worker import worker
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
from tests.utils.vespa_docker import VespaDockerManager

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

REDIS_CONTAINER = "redis-ingestion-startup-readiness"
WORKER_LOGGER = "cogniverse_runtime.ingestion_worker.worker"
STORE_VISIT_TIMEOUT_S = 30
STORE_VISIT_ATTEMPTS = 5
STORE_VISIT_BACKOFF_S = 0.25 + 0.5 + 1.0 + 2.0
STORE_RETRY_BUDGET_S = (
    STORE_VISIT_ATTEMPTS * STORE_VISIT_TIMEOUT_S + STORE_VISIT_BACKOFF_S
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=60, check=False
    )


@pytest.fixture(scope="module")
def outage_vespa():
    """A Vespa this module owns outright, so pausing it touches nothing else."""
    manager = VespaDockerManager()
    info = manager.start_container("test_ingestion_worker_startup_readiness")
    try:
        manager.wait_for_config_ready(info, timeout=180)
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=info["config_port"],
            schema_registry=None,
        ).upload_metadata_schemas(app_name="cogniverse", allow_schema_removal=False)
        manager.wait_for_application_ready(info, timeout=180)
        yield info
    finally:
        _docker("unpause", info["container_name"])
        manager.stop_container(info)


@pytest.fixture(scope="module")
def redis_url():
    port = _free_port()
    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"
    )
    _docker("rm", "-f", REDIS_CONTAINER)
    result = _docker(
        "run",
        "-d",
        "--name",
        REDIS_CONTAINER,
        "--label",
        f"cogniverse-test-owner-pid={os.getpid()}",
        "-p",
        f"{port}:6379",
        "--platform",
        docker_platform,
        "redis:7.4-alpine",
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Redis: {result.stderr}")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = _docker("exec", REDIS_CONTAINER, "redis-cli", "ping")
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.5)
    else:
        _docker("rm", "-f", REDIS_CONTAINER)
        pytest.fail("Redis did not become ready within 30s")
    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        _docker("rm", "-f", REDIS_CONTAINER)


@pytest.fixture
def worker_env(outage_vespa, redis_url, monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", str(outage_vespa["http_port"]))
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("INGEST_REAPER_ENABLED", "false")
    monkeypatch.setenv("INGEST_CONSUMER_ID", "startup-readiness-worker")
    monkeypatch.setenv("INGEST_STARTUP_GRACE_SECONDS", "1")
    monkeypatch.setenv("INGEST_CLAIM_BLOCK_MS", "200")
    return outage_vespa


class _WorkerThread:
    """``worker.run`` on its own event loop, driven from the test thread."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop: asyncio.Event | None = None
        self.outcome: str | None = None
        self.error: BaseException | None = None
        self._started = threading.Event()

        async def _never_called(job, **kwargs):
            raise AssertionError(f"processor must not run during startup: {job}")

        async def _main():
            self.loop = asyncio.get_running_loop()
            self.stop = asyncio.Event()
            self._started.set()
            await worker.run(stop=self.stop, processor=_never_called)

        def _target():
            try:
                asyncio.run(_main())
                self.outcome = "returned"
            except BaseException as exc:
                self.outcome = "raised"
                self.error = exc

        self.thread = threading.Thread(target=_target, name="worker-run", daemon=True)

    def start(self):
        self.thread.start()
        assert self._started.wait(timeout=30)

    def request_stop(self):
        self.loop.call_soon_threadsafe(self.stop.set)


def _records(caplog, level: str, logger_name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == level and r.name == logger_name]


def _wait_until(predicate, *, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_worker_waits_through_config_store_outage_then_serves(worker_env, caplog):
    info = worker_env
    caplog.set_level(logging.INFO, logger=WORKER_LOGGER)
    run = _WorkerThread()

    assert _docker("pause", info["container_name"]).returncode == 0
    paused_at = time.monotonic()
    try:
        run.start()
        # The first store visit sequence ends only after the store's whole
        # retry budget; the worker must still be alive when it does.
        budget_elapsed = _wait_until(
            lambda: run.outcome is not None or _records(caplog, "ERROR", WORKER_LOGGER),
            timeout=STORE_RETRY_BUDGET_S + 60,
        )
        assert budget_elapsed, "neither an ERROR nor an exit within the store budget"
        first_failure_at = time.monotonic() - paused_at
        assert run.outcome is None, (
            f"worker exited ({run.outcome}: {run.error!r}) {first_failure_at:.1f}s "
            f"after the store was paused instead of waiting for it"
        )
        assert run.thread.is_alive()
    finally:
        assert _docker("unpause", info["container_name"]).returncode == 0
    unpaused_at = time.monotonic() - paused_at

    served = _wait_until(
        lambda: (
            _records(caplog, "INFO", WORKER_LOGGER)
            and any(
                r.message.startswith("Worker startup-readiness-worker started")
                for r in _records(caplog, "INFO", WORKER_LOGGER)
            )
        ),
        timeout=STORE_RETRY_BUDGET_S + 60,
    )
    served_at = time.monotonic() - paused_at
    assert served, f"worker did not start serving after unpause (outcome={run.outcome})"

    run.request_stop()
    run.thread.join(timeout=30)
    assert run.outcome == "returned"

    errors = [r.message for r in _records(caplog, "ERROR", WORKER_LOGGER)]
    assert len(errors) == 1
    error_shape = re.fullmatch(
        r"Ingestion worker configuration dependency was not ready after 1 attempt "
        r"within 1\.0s: ConfigStoreUnavailableError: Failed to read Vespa config "
        r"visit after 5 attempts over (\d+\.\d{3})s: ReadTimeout: "
        r"HTTPConnectionPool\(host='localhost', port=(\d+)\): Read timed out\. "
        r"\(read timeout=30\); keeping the worker alive and retrying",
        errors[0],
    )
    assert error_shape, errors[0]
    assert int(error_shape.group(2)) == info["http_port"]
    assert (
        STORE_RETRY_BUDGET_S <= float(error_shape.group(1)) <= STORE_RETRY_BUDGET_S + 15
    )
    started_lines = [
        r.message
        for r in _records(caplog, "INFO", WORKER_LOGGER)
        if r.message.startswith("Worker startup-readiness-worker started")
    ]
    assert started_lines == [
        "Worker startup-readiness-worker started: group=ingestors "
        f"redis={os.environ['REDIS_URL']} reaper=off"
    ]
    print(
        f"TIMINGS paused=0.0s first_failure={first_failure_at:.1f}s "
        f"unpaused={unpaused_at:.1f}s served={served_at:.1f}s"
    )


def test_one_shot_startup_wait_fails_after_its_deadline(monkeypatch):
    """Without ``retry_forever`` the grace window is a deadline: the wait must
    raise with the store's typed error chained, not spin."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        dead_port = reserved.getsockname()[1]
    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", str(dead_port))

    started = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        worker._wait_for_startup_config(grace_s=0.0, retry_forever=False, abort=None)
    elapsed = time.monotonic() - started

    assert isinstance(excinfo.value.__cause__, ConfigStoreUnavailableError)
    assert isinstance(excinfo.value.__cause__.__cause__, requests.ConnectionError)
    shape = re.fullmatch(
        r"Ingestion worker configuration dependency was not ready after 1 attempt "
        r"within 0\.0s: ConfigStoreUnavailableError: Failed to read Vespa config "
        r"visit after 5 attempts over (\d+\.\d{3})s: ConnectionError: .*"
        rf"port={dead_port}\).*",
        str(excinfo.value),
        re.DOTALL,
    )
    assert shape, str(excinfo.value)
    assert STORE_VISIT_BACKOFF_S <= elapsed < STORE_VISIT_BACKOFF_S + 10


def test_startup_wait_ends_when_shutdown_is_requested(monkeypatch):
    """A retry-forever wait must still honour SIGTERM: once the stop flag is
    up the next retry raises instead of sleeping, so the pod exits within its
    grace period rather than being SIGKILLed mid-wait."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        dead_port = reserved.getsockname()[1]
    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", str(dead_port))
    stop = threading.Event()
    stop.set()
    from cogniverse_runtime.startup_wait import DependencyWaitAborted

    started = time.monotonic()
    with pytest.raises(DependencyWaitAborted) as excinfo:
        worker._wait_for_startup_config(
            grace_s=300.0, retry_forever=True, abort=stop.is_set
        )
    elapsed = time.monotonic() - started

    assert str(excinfo.value).startswith(
        "Ingestion worker configuration dependency wait aborted by shutdown "
        "request after 1 attempt: ConfigStoreUnavailableError: "
    )
    assert isinstance(excinfo.value.__cause__, ConfigStoreUnavailableError)
    assert STORE_VISIT_BACKOFF_S <= elapsed < STORE_VISIT_BACKOFF_S + 10
