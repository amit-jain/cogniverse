"""Real semantic-router e2e — drives cogniverse's own routing path against
the gateway that ``cogniverse up`` deploys into the cluster.

The chart deploys Envoy + the vLLM Semantic Router in front of the LLM
backend; the runtime routes every agent's LLM call through it. This test
exercises the actual deployed boundary: it builds a routed endpoint with
``apply_semantic_routing`` + ``create_dspy_lm`` (exactly as the agents do),
points it at the deployed Envoy, and sends a real completion — proving the
Envoy -> ext_proc(SR) -> LLM path works end to end in the cluster
(``failure_mode_allow: false`` means a broken SR would fail the call).

Requires:
- ``cogniverse up`` running with ``semanticRouter.enabled=true`` (the default).
- ``kubectl`` reachable in PATH.

Service exposure: the Envoy Service is ClusterIP (no NodePort), so the
fixture opens a session-scoped ``kubectl port-forward`` to a local port.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time

import httpx
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import apply_semantic_routing
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from tests.e2e.conftest import skip_if_no_runtime

pytestmark = [pytest.mark.e2e, skip_if_no_runtime]

_NAMESPACE = "cogniverse"
_ENVOY_SVC = "cogniverse-semantic-router-envoy"
_ENVOY_PORT = 8801


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kubectl_names(component: str) -> list[str]:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            _NAMESPACE,
            "get",
            "pods",
            "-l",
            f"app.kubernetes.io/component={component}",
            "--field-selector=status.phase=Running",
            "-o",
            "name",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.fixture(scope="session")
def sr_envoy_url():
    """Port-forward ``svc/cogniverse-semantic-router-envoy`` so the test can
    route completions through the deployed gateway. Skips when kubectl is
    missing or the service isn't deployed."""
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not in PATH; cannot reach the deployed semantic router")

    probe = subprocess.run(
        ["kubectl", "-n", _NAMESPACE, "get", "svc", _ENVOY_SVC, "-o", "name"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"{_ENVOY_SVC} service not present in '{_NAMESPACE}' namespace; "
            "deploy with `cogniverse up` first"
        )

    local_port = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "-n",
            _NAMESPACE,
            "port-forward",
            f"svc/{_ENVOY_SVC}",
            f"{local_port}:{_ENVOY_PORT}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=2):
                    break
            except OSError:
                time.sleep(1)
        else:
            proc.terminate()
            pytest.fail(
                f"port-forward to {_ENVOY_SVC} not reachable on :{local_port} in 30s"
            )
        yield f"http://127.0.0.1:{local_port}/v1"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_semantic_router_pods_running():
    """Both gateway pods (the router and its Envoy data plane) must be up."""
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not in PATH")
    assert _kubectl_names("semantic-router"), "no Running semantic-router pod"
    assert _kubectl_names("semantic-router-envoy"), "no Running envoy pod"


def _route(sr_envoy_url: str, tenant_id: str, tier: str):
    """Build the routed LM exactly as an agent would and return the dspy.LM."""
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    config = SemanticRouterConfig(
        enabled=True,
        semantic_router_url=sr_envoy_url,
        tenant_tiers={tenant_id: tier},
    )
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=config,
        tenant_id=tenant_id,
    )
    # The routed endpoint must carry the gateway api_base + the two authz
    # headers the router requires — this is the contract the agents rely on.
    assert routed.api_base == sr_envoy_url
    assert routed.extra_headers == {
        "x-authz-user-id": tenant_id,
        "x-authz-user-groups": tier,
    }
    lm = create_dspy_lm(routed)
    lm.cache = False
    return lm


def test_completion_routes_through_deployed_semantic_router(sr_envoy_url):
    """A real completion sent through the deployed Envoy -> SR -> LLM returns
    a non-empty answer, proving the whole routed path is live."""
    lm = _route(sr_envoy_url, tenant_id="pro-tenant", tier="pro")
    out = lm("Reply with the single word: pong")
    text = out[0] if isinstance(out, list) else out
    assert isinstance(text, str) and text.strip(), (
        "deployed semantic router returned no completion — the "
        "Envoy -> ext_proc -> LLM path is broken"
    )


def test_router_metrics_reachable():
    """The router exposes Prometheus metrics — a cheap liveness signal that
    the ext_proc service (not just Envoy) is actually up."""
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not in PATH")
    if not _kubectl_names("semantic-router"):
        pytest.skip("semantic-router pod not running")
    local_port = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "-n",
            _NAMESPACE,
            "port-forward",
            "svc/cogniverse-semantic-router",
            f"{local_port}:9190",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        last_status = None
        while time.time() < deadline:
            try:
                last_status = httpx.get(
                    f"http://127.0.0.1:{local_port}/metrics", timeout=2.0
                ).status_code
                if last_status == 200:
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                pass
            time.sleep(1)
        assert last_status == 200, f"router /metrics not 200 (got {last_status})"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
