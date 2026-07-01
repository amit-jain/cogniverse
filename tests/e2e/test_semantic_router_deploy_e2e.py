"""Deployment e2e: real routing through the `cogniverse up`-deployed semantic router.

This does NOT just check that a completion comes back — it asserts the actual
routing DECISION the router made, read from the router's own Prometheus counter
``llm_decision_match_total{decision_name=...}``. Sending a request as a given
tenant tier + content through the deployed Envoy must increment exactly the
decision the router should pick:

  * free tier              -> ``free-default``          (basic model)
  * pro tier + technical   -> ``pro-technical-keyword`` (reasoning model)
  * pro tier + non-technical -> ``pro-default``         (no reasoning)

Requires ``cogniverse up`` with ``semanticRouter.enabled=true`` (the default)
and ``kubectl`` in PATH. Both the Envoy entry and the router's metrics port are
ClusterIP, so the fixture opens ``kubectl port-forward`` for each.
"""

from __future__ import annotations

import re
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
_ROUTER_SVC = "cogniverse-semantic-router"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _svc_present(name: str) -> bool:
    r = subprocess.run(
        ["kubectl", "-n", _NAMESPACE, "get", "svc", name, "-o", "name"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return r.returncode == 0


def _running_pods(component: str) -> list[str]:
    r = subprocess.run(
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
    return (
        [ln for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else []
    )


def _port_forward(svc: str, remote_port: int, ready_path: str | None):
    """kubectl port-forward svc/<svc> <local>:<remote>; wait until reachable."""
    local = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "-n",
            _NAMESPACE,
            "port-forward",
            f"svc/{svc}",
            f"{local}:{remote_port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", local), timeout=2):
                if ready_path is None:
                    return proc, local
                if (
                    httpx.get(
                        f"http://127.0.0.1:{local}{ready_path}", timeout=2
                    ).status_code
                    < 500
                ):
                    return proc, local
        except (OSError, httpx.HTTPError):
            pass
        time.sleep(1)
    proc.terminate()
    pytest.fail(f"port-forward to {svc}:{remote_port} not reachable in 30s")


@pytest.fixture(scope="session")
def sr_endpoints():
    """Port-forward the deployed Envoy entry + router metrics; yield both URLs."""
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not in PATH; cannot reach the deployed semantic router")
    for svc in (_ENVOY_SVC, _ROUTER_SVC):
        if not _svc_present(svc):
            pytest.skip(f"{svc} not deployed in '{_NAMESPACE}'; run `cogniverse up`")

    envoy_proc, envoy_port = _port_forward(_ENVOY_SVC, 8801, None)
    metrics_proc, metrics_port = _port_forward(_ROUTER_SVC, 9190, "/metrics")
    try:
        yield {
            "envoy_url": f"http://127.0.0.1:{envoy_port}/v1",
            "metrics_url": f"http://127.0.0.1:{metrics_port}/metrics",
        }
    finally:
        for p in (envoy_proc, metrics_proc):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def _decision_count(metrics_url: str, decision_name: str) -> int:
    """Read llm_decision_match_total for one decision (0 if not yet present)."""
    body = httpx.get(metrics_url, timeout=5).text
    m = re.search(
        rf'^llm_decision_match_total\{{decision_name="{re.escape(decision_name)}"\}}\s+([0-9.]+)',
        body,
        re.MULTILINE,
    )
    return int(float(m.group(1))) if m else 0


def _route_completion(envoy_url: str, tenant_id: str, tier: str, prompt: str) -> str:
    """Route a real completion through the deployed router; return the answer text."""
    endpoint = LLMEndpointConfig(model="openai/auto", api_base="http://unused:1/v1")
    routed = apply_semantic_routing(
        endpoint=endpoint,
        config=SemanticRouterConfig(
            enabled=True,
            semantic_router_url=envoy_url,
            tenant_tiers={tenant_id: tier},
        ),
        tenant_id=tenant_id,
    )
    assert routed.extra_headers == {
        "x-authz-user-id": tenant_id,
        "x-authz-user-groups": tier,
    }
    lm = create_dspy_lm(routed)
    lm.cache = False
    out = lm(prompt)
    item = out[0] if isinstance(out, list) else out
    if isinstance(item, dict):
        return item.get("text") or item.get("content") or ""
    return item


def test_semantic_router_pods_running():
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not in PATH")
    assert _running_pods("semantic-router"), "no Running semantic-router pod"
    assert _running_pods("semantic-router-envoy"), "no Running envoy pod"


# (tenant_id, tier, prompt, expected decision the router must match)
_CASES = [
    ("free-tenant", "free", "hello there", "free-default"),
    (
        "pro-tenant",
        "pro",
        "design a recursive algorithm and analyze its time complexity",
        "pro-technical-keyword",
    ),
    ("pro-tenant", "pro", "what is a good movie to watch tonight?", "pro-default"),
]


@pytest.mark.parametrize("tenant_id,tier,prompt,expected_decision", _CASES)
def test_tier_and_content_route_to_expected_decision(
    sr_endpoints, tenant_id, tier, prompt, expected_decision
):
    """A real request as this tenant+content must increment exactly the
    decision the router is supposed to pick — proving tier gating + content
    routing work end to end against the deployed stack."""
    metrics_url = sr_endpoints["metrics_url"]
    before = _decision_count(metrics_url, expected_decision)

    answer = _route_completion(sr_endpoints["envoy_url"], tenant_id, tier, prompt)
    assert answer.strip(), "deployed router returned no completion"

    # Metrics update asynchronously; poll briefly for the counter to move.
    deadline = time.time() + 20
    after = before
    while time.time() < deadline:
        after = _decision_count(metrics_url, expected_decision)
        if after > before:
            break
        time.sleep(2)
    assert after > before, (
        f"router did not match decision {expected_decision!r} for tenant "
        f"{tenant_id!r} (tier={tier!r}); counter stayed at {before}"
    )
