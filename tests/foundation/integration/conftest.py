"""Self-managed semantic-router stack for gateway-routing integration tests.

Launches Envoy + vLLM Semantic Router + a reflecting stub upstream as three
docker containers on a private network and tears them down afterwards. This is
the ``shared_vespa`` idiom (``docker run``, unique per-process names, a
health-wait loop, ``docker rm -f`` cleanup in ``finally``) — the test owns its
infrastructure. There is no docker-compose file, no pre-started service, and no
manual environment variable: running the module launches the stack, and the
suite skips cleanly when the Docker daemon is absent.

The container config (``_sr_stack/{envoy.yaml,sr-config.yaml,stub_upstream.py}``)
addresses peers by the docker network aliases ``semantic-router`` and
``stub-upstream``, so only the container names and the published Envoy port vary
per process.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from tests.utils.markers import is_docker_available

_STACK_DIR = Path(__file__).resolve().parent / "_sr_stack"
_ENVOY_IMAGE = "envoyproxy/envoy:v1.31-latest"
_SR_IMAGE = "ghcr.io/vllm-project/semantic-router:latest"
_STUB_IMAGE = "python:3.12-slim"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _docker_daemon_up() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(scope="module")
def semantic_router_stack():
    """Yield ``{"base_url", "host_port"}`` for a live Envoy->SR->stub chain."""
    if not is_docker_available() or not _docker_daemon_up():
        pytest.skip("Docker daemon unavailable; semantic-router stack cannot launch")

    uid = f"{os.getpid()}-{int(time.time() * 1000)}"
    net = f"cog-sr-net-{uid}"
    stub = f"cog-sr-stub-{uid}"
    router = f"cog-sr-router-{uid}"
    envoy = f"cog-sr-envoy-{uid}"
    host_port = _free_port()
    base_url = f"http://localhost:{host_port}/v1"

    # localhost must bypass any outbound HTTPS proxy the environment sets.
    prev_no_proxy = (os.environ.get("NO_PROXY"), os.environ.get("no_proxy"))
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    created: list[tuple[str, str]] = []
    try:
        r = _docker("network", "create", net)
        if r.returncode != 0:
            pytest.skip(f"cannot create docker network: {r.stderr.strip()}")
        created.append(("network", net))

        # Reflecting OpenAI-compatible stub backend.
        r = _docker(
            "run",
            "-d",
            "--name",
            stub,
            "--network",
            net,
            "--network-alias",
            "stub-upstream",
            "-v",
            f"{_STACK_DIR / 'stub_upstream.py'}:/app/stub.py:ro",
            _STUB_IMAGE,
            "python",
            "/app/stub.py",
        )
        if r.returncode != 0:
            pytest.fail(f"stub upstream failed to start: {r.stderr}")
        created.append(("container", stub))

        # Semantic router (ext_proc gRPC on :50051). Pulls ~1.5GB classifiers
        # on first run, hence the generous readiness deadline below.
        r = _docker(
            "run",
            "-d",
            "--name",
            router,
            "--network",
            net,
            "--network-alias",
            "semantic-router",
            "-e",
            "CONFIG_FILE=/app/config/config.yaml",
            "-v",
            f"{_STACK_DIR / 'sr-config.yaml'}:/app/config/config.yaml:ro",
            _SR_IMAGE,
            timeout=300,
        )
        if r.returncode != 0:
            pytest.fail(f"semantic-router failed to start: {r.stderr}")
        created.append(("container", router))

        # Envoy front proxy — the OpenAI-compatible entry point.
        r = _docker(
            "run",
            "-d",
            "--name",
            envoy,
            "--network",
            net,
            "-p",
            f"{host_port}:8801",
            "-v",
            f"{_STACK_DIR / 'envoy.yaml'}:/etc/envoy/envoy.yaml:ro",
            _ENVOY_IMAGE,
            "-c",
            "/etc/envoy/envoy.yaml",
        )
        if r.returncode != 0:
            pytest.fail(f"envoy failed to start: {r.stderr}")
        created.append(("container", envoy))

        # Poll the full chain until the stub answers through Envoy + ext_proc.
        deadline = time.time() + 300
        last_err = "no response"
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"http://localhost:{host_port}/v1/models", timeout=3
                )
                if resp.status_code == 200:
                    break
            except requests.RequestException as exc:
                last_err = str(exc)
            time.sleep(3)
        else:
            logs = _docker("logs", "--tail", "40", router).stdout
            pytest.fail(
                f"semantic-router stack not ready in 300s (last: {last_err})\n"
                f"router logs:\n{logs}"
            )

        yield {"base_url": base_url, "host_port": host_port}
    finally:
        for kind, name in reversed(created):
            if kind == "container":
                _docker("rm", "-f", name, timeout=30)
            else:
                _docker("network", "rm", name, timeout=30)
        for var, val in zip(("NO_PROXY", "no_proxy"), prev_no_proxy):
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
