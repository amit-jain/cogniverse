"""Self-managed semantic-router stack for semantic-routing integration tests.

Launches Envoy + vLLM Semantic Router + a reflecting stub upstream as three
docker containers on a private network and tears them down afterwards. This is
the ``shared_vespa`` idiom (``docker run``, unique per-process names, a
health-wait loop, ``docker rm -f`` cleanup in ``finally``) — the test owns its
infrastructure. There is no docker-compose file, no pre-started service and no
manual environment variable: running the module launches the stack.

Envoy runs the chart's own data plane, rendered for the local peers by
``tests.utils.semantic_router_stack``; the router and stub answer on the docker
network aliases that rendering addresses. The router's config
(``_sr_stack/sr-config.yaml``) and the stub (``_sr_stack/stub_upstream.py``)
are the stack's own, so only the container names and the published Envoy port
vary per process.

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
import yaml

from tests.utils.semantic_router_stack import (
    CHART_VALUES,
    ROUTER_ALIAS,
    UPSTREAM_ALIAS,
    envoy_listener_port,
    render_envoy_config,
    wait_for_routed_chat,
)

_STACK_DIR = Path(__file__).resolve().parent / "_sr_stack"
_STUB_IMAGE = "python:3.12-slim"

# The tier the readiness probe presents and the model the router must rewrite
# ``auto`` to for it, per the ``free-default`` decision in ``sr-config.yaml``.
_PROBE_TENANT = "readiness-probe-tenant"
_PROBE_TIER = "free"
_PROBE_MODEL = "basic-chat"


def _shipped_image(*path: str) -> str:
    """The image ref the chart deploys, read from ``charts/cogniverse/values.yaml``.

    The stack runs the SAME build production runs: a router whose request
    re-serialization drops ``response_format.json_schema`` passes a stack
    pinned to some other tag while every served structured call 400s.
    """
    node = yaml.safe_load(CHART_VALUES.read_text())
    for key in path:
        node = node[key]
    digest = node.get("digest")
    if digest:
        return f"{node['repository']}@{digest}"
    return f"{node['repository']}:{node['tag']}"


_ENVOY_IMAGE = _shipped_image("semanticRouter", "envoy", "image")
_SR_IMAGE = _shipped_image("semanticRouter", "router", "image")

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


def _remove_network(name: str, *, attempts: int = 10, pause_s: float = 1.0) -> None:
    """Remove a stack network, retrying while docker releases its endpoints.

    ``network rm`` races the endpoint teardown of the containers just removed
    and fails with the network still present; an unremoved network carries the
    owner label, so the next session's reaper collects whatever survives this.
    """
    for attempt in range(attempts):
        if _docker("network", "rm", name, timeout=30).returncode == 0:
            return
        listing = _docker("network", "ls", "--format", "{{.Name}}", timeout=30)
        if name not in listing.stdout.split():
            return
        if attempt < attempts - 1:
            time.sleep(pause_s)


@pytest.fixture(scope="module")
def semantic_router_stack(tmp_path_factory):
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
    from tests.utils.vllm_sidecar import (
        OWNER_LABEL,
        reap_dead_owner_containers,
        reap_dead_owner_networks,
    )

    reap_dead_owner_containers()
    reap_dead_owner_networks()
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
        r = _docker("network", "create", "--label", owner_label, net)
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
            UPSTREAM_ALIAS,
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
            ROUTER_ALIAS,
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

        # Envoy front proxy — the OpenAI-compatible entry point, running the
        # chart's data plane with the local peers substituted in.
        envoy_config = tmp_path_factory.mktemp("sr-envoy") / "envoy.yaml"
        envoy_config.write_text(render_envoy_config())
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
            f"{host_port}:{envoy_listener_port()}",
            "-v",
            f"{envoy_config}:/etc/envoy/envoy.yaml:ro",
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

        # Then wait on a real routed completion. Envoy answers /v1/models
        # without ever carrying a body to the router, so that endpoint is up
        # while every completion still times out — the state this refuses to
        # hand to the tests. A routed chat takes 0.30s here (measured), so a
        # stack that has not served one in 60s is not slow, it is broken.
        try:
            wait_for_routed_chat(
                base_url,
                tenant_id=_PROBE_TENANT,
                tenant_tier=_PROBE_TIER,
                expected_model=_PROBE_MODEL,
                budget_s=60,
            )
        except RuntimeError as error:
            pytest.fail(
                f"{error}\nenvoy log:\n{_docker('logs', '--tail', '20', envoy).stdout}"
                f"{_docker('logs', '--tail', '20', envoy).stderr}"
                f"\nrouter log:\n{_docker('logs', '--tail', '20', router).stdout}"
            )

        yield {"base_url": base_url, "host_port": host_port}
    finally:
        for kind, name in reversed(created):
            if kind == "container":
                _docker("rm", "-f", name, timeout=30)
            else:
                _remove_network(name)
        for var, val in zip(("NO_PROXY", "no_proxy"), prev_no_proxy):
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
