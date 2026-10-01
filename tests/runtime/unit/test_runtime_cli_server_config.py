"""The runtime CLI reads uvicorn's own command line and UVICORN_* variables."""

from __future__ import annotations

import inspect

import pytest
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from uvicorn.main import main as uvicorn_cli

from cogniverse_runtime.runtime_cli import (
    APP,
    RuntimeWorkerSupervisor,
    uvicorn_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

IMAGE_ARGS = ["--host", "0.0.0.0", "--port", "8000"]


@pytest.fixture(autouse=True)
def _no_worker_env(monkeypatch):
    for name in ("UVICORN_WORKERS", "WEB_CONCURRENCY", "UVICORN_UDS"):
        monkeypatch.delenv(name, raising=False)


def test_the_image_arguments_and_chart_env_build_the_served_config(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "3")
    monkeypatch.setenv("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", "15")

    config = uvicorn_config(IMAGE_ARGS)

    assert (config.app, config.host, config.port) == (APP, "0.0.0.0", 8000)
    assert config.workers == 3
    assert config.timeout_graceful_shutdown == 15
    assert config.log_config == LOGGING_CONFIG
    assert config.headers == []
    assert config.should_reload is False


def test_one_worker_without_a_worker_setting():
    assert uvicorn_config(IMAGE_ARGS).workers == 1


def test_web_concurrency_sets_the_worker_count_as_uvicorn_reads_it(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    assert uvicorn_config(IMAGE_ARGS).workers == 2


def test_every_uvicorn_option_reaches_the_config():
    """An option uvicorn's command line gains must not be dropped silently:
    everything it parses is a Config parameter, except the import path it
    applies itself."""
    parsed = set(uvicorn_cli.make_context(uvicorn_cli.name, [APP]).params)
    accepted = set(inspect.signature(uvicorn.Config.__init__).parameters)
    assert parsed - accepted == {"app_dir"}


@pytest.mark.parametrize(
    "args", [["--uds", "/tmp/runtime.sock"], ["--fd", "3"]], ids=["uds", "fd"]
)
def test_workers_refuse_a_listener_they_would_not_bind(monkeypatch, args):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    with pytest.raises(ValueError) as refused:
        RuntimeWorkerSupervisor(uvicorn_config(args))
    assert str(refused.value) == (
        "Runtime workers each listen on --host/--port; --uds and --fd serve one process"
    )
