"""Self-managed semantic-router stack for semantic-routing integration tests.

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

The router's classifier bundle and embedding model are cached in persistent
named volumes (``cog-sr-models`` / ``cog-sr-hf-cache``) so the multi-GB download
happens once per host; the readiness budget adapts to whether the caches are
already warm.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

_STACK_DIR = Path(__file__).resolve().parent / "_sr_stack"
_ENVOY_IMAGE = "envoyproxy/envoy:v1.31-latest"
_SR_IMAGE = "ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
_STUB_IMAGE = "python:3.12-slim"

# Persistent caches so the router's classifier bundle (/app/models) and any
# HuggingFace-hosted embedding model download ONCE per host instead of every
# run: docker rm -f otherwise discards them, and a cold ~GB download blows the
# readiness budget (the reason this suite errored on a cold runner). Named
# volumes mirror the face-embed-cache idiom — the test still owns its infra.
_SR_MODELS_VOLUME = "cog-sr-models"
_SR_HF_VOLUME = "cog-sr-hf-cache"
# Classifier bundle marker relative to the volume root (mounted at
# /app/models); present ⇒ the bundle is cached and the router starts warm.
_SR_CLASSIFIER_MARKER = "mmbert32k-intent-classifier-merged/category_mapping.json"


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


def _sr_models_cached() -> bool:
    """True when the persistent volume already holds the classifier bundle,
    so the router starts warm and the readiness budget can be short."""
    probe = _docker(
        "run",
        "--rm",
        "-v",
        f"{_SR_MODELS_VOLUME}:/models",
        _STUB_IMAGE,
        "test",
        "-f",
        f"/models/{_SR_CLASSIFIER_MARKER}",
        timeout=60,
    )
    return probe.returncode == 0


@pytest.fixture(scope="module")
def semantic_router_stack():
    """Yield ``{"base_url", "host_port"}`` for a live Envoy->SR->stub chain."""
    uid = f"{os.getpid()}-{int(time.time() * 1000)}"
    net = f"cog-sr-net-{uid}"
    stub = f"cog-sr-stub-{uid}"
    router = f"cog-sr-router-{uid}"
    envoy = f"cog-sr-envoy-{uid}"
    host_port = _free_port()
    base_url = f"http://localhost:{host_port}/v1"

    # Reap stack containers whose owning pytest was SIGKILLed before the
    # finally-teardown could run — an orphaned router holds its classifier
    # models in host RAM indefinitely.
    from tests.utils.vllm_sidecar import OWNER_LABEL, reap_dead_owner_containers

    reap_dead_owner_containers()
    owner_label = f"{OWNER_LABEL}={os.getpid()}"

    # Provision persistent model caches before starting the router so the
    # download happens once per host, not every run.
    _docker("volume", "create", _SR_MODELS_VOLUME)
    _docker("volume", "create", _SR_HF_VOLUME)
    warm_start = _sr_models_cached()

    # localhost must bypass any outbound HTTPS proxy the environment sets.
    prev_no_proxy = (os.environ.get("NO_PROXY"), os.environ.get("no_proxy"))
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    created: list[tuple[str, str]] = []
    try:
        r = _docker("network", "create", net)
        if r.returncode != 0:
            pytest.fail(f"cannot create docker network: {r.stderr.strip()}")
        created.append(("network", net))

        # Reflecting OpenAI-compatible stub backend.
        r = _docker(
            "run",
            "-d",
            "--name",
            stub,
            "--label",
            owner_label,
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

        # Semantic router (ext_proc gRPC on :50051). The image entrypoint reads
        # its config from the CMD arg, defaulting to /app/config.yaml; mount
        # there (CONFIG_FILE env is not honored). Downloads its classifier
        # bundle on first run, hence the generous readiness deadline below.
        r = _docker(
            "run",
            "-d",
            "--name",
            router,
            "--label",
            owner_label,
            "--network",
            net,
            "--network-alias",
            "semantic-router",
            "-v",
            f"{_STACK_DIR / 'sr-config.yaml'}:/app/config.yaml:ro",
            "-v",
            f"{_SR_MODELS_VOLUME}:/app/models",
            "-v",
            f"{_SR_HF_VOLUME}:/root/.cache/huggingface",
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
            "--label",
            owner_label,
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

        # Wait for the router's classifier runtime to finish loading. It serves
        # a placeholder classifier while downloading its bundle (~GB on first
        # run); requests routed during that window skip signal evaluation and
        # fall through to the tier default — so polling Envoy's /v1/models (up
        # far earlier) would race the tests. ``startup_complete`` marks the real
        # classifier ready.
        #
        # A cold first run downloads the classifier bundle + embedding model
        # into the persistent volumes; a warm run reuses them. Size the wait to
        # whichever path this run takes so a genuine hang still fails promptly.
        budget_s = 300 if warm_start else 1800
        deadline = time.time() + budget_s
        while time.time() < deadline:
            out = _docker("logs", router)
            if "startup_complete" in (out.stdout + out.stderr):
                break
            time.sleep(3)
        else:
            logs = _docker("logs", "--tail", "40", router).stdout
            pytest.fail(
                f"semantic-router classifier runtime not ready in {budget_s}s "
                f"(warm_start={warm_start})\nrouter logs:\n{logs}"
            )

        # Then confirm the Envoy -> ext_proc -> stub path answers end to end.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if (
                    requests.get(
                        f"http://localhost:{host_port}/v1/models", timeout=3
                    ).status_code
                    < 500
                ):
                    break
            except requests.RequestException:
                pass
            time.sleep(3)

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
