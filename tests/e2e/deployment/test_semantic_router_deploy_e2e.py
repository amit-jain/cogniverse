"""Deployment e2e: real routing through the semantic router the chart deploys.

This stands up its OWN isolated k3d cluster and helm-installs the chart with a
lean profile (``sr_only_values.yaml``): just the semantic router + its Envoy +
a CPU Ollama LLM for the router to proxy — every other component is off, so the
cluster needs no GPU, no host mounts, and no locally-built application image.
Semantic routing is default-on, so the router config the chart renders is the
real one under test.

The test asserts the actual routing DECISION the deployed router makes, read
from its own Prometheus counter ``llm_decision_match_total{decision_name=...}``.
Sending a real completion as a given tenant tier + content through the deployed
Envoy must increment exactly the decision the router should pick:

  * free tier                -> ``free-default``          (basic model)
  * pro tier + technical     -> ``pro-technical-keyword`` (reasoning model)
  * pro tier + non-technical -> ``pro-default``           (no reasoning)

The router forwards to the in-cluster Ollama the chart wires it to (the
``srUpstream*`` helpers derive from ``primaryLLMEndpoint``), so a non-empty
completion also proves the Envoy -> ext_proc -> LLM path end to end.
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import apply_semantic_routing
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CHART_PATH = _PROJECT_ROOT / "charts" / "cogniverse"
_K3S_VALUES = _CHART_PATH / "values.k3s.yaml"
_SR_VALUES = Path(__file__).resolve().parent / "sr_only_values.yaml"

SR_CLUSTER = "cogniverse-sr-test"
SR_NAMESPACE = "cogniverse-sr-test"
OLLAMA_POD = "cogniverse-llm-0"
OLLAMA_MODEL = "qwen2.5:0.5b"

_ENVOY_SVC = "cogniverse-semantic-router-envoy"
_ROUTER_SVC = "cogniverse-semantic-router"

# Third-party images the lean profile runs (chart pins), imported into the k3d
# cluster up front so pods start instantly instead of racing a slow in-cluster
# pull at pod-start.
_THIRD_PARTY_IMAGES = [
    "ollama/ollama:0.20.5",
    "ghcr.io/vllm-project/semantic-router/vllm-sr:latest",
    "envoyproxy/envoy:v1.31-latest",
]


def _import_image_to_node(img: str, node: str) -> None:
    """Load a host docker image into the k3d node's containerd (k8s.io ns).

    ``k3d image import`` silently fails on images exported by docker's newer
    image store (logs ``ctr: content digest ... not found`` but still reports
    success), so the pod ends up pulling from the registry anyway. Going
    through a legacy ``docker save`` tarball + ``ctr images import`` lands the
    image reliably, so ``imagePullPolicy: IfNotPresent`` finds it.
    """
    subprocess.run(["docker", "pull", img], check=True, timeout=1800)
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "img.tar"
        subprocess.run(["docker", "save", img, "-o", str(tar)], check=True, timeout=600)
        subprocess.run(
            ["docker", "cp", str(tar), f"{node}:/tmp/img.tar"], check=True, timeout=300
        )
    subprocess.run(
        [
            "docker",
            "exec",
            node,
            "ctr",
            "-n",
            "k8s.io",
            "images",
            "import",
            "/tmp/img.tar",
        ],
        check=True,
        timeout=600,
    )


def _kc(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "-n", SR_NAMESPACE, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _running_pods(component: str) -> list[str]:
    r = _kc(
        "get",
        "pods",
        "-l",
        f"app.kubernetes.io/component={component}",
        "--field-selector=status.phase=Running",
        "-o",
        "name",
    )
    return (
        [ln for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else []
    )


def _wait_pod_running(pod: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = _kc("get", "pod", pod, "-o", "jsonpath={.status.phase}", timeout=20)
        if r.returncode == 0 and r.stdout.strip() == "Running":
            return
        time.sleep(5)
    logs = _kc("describe", "pod", pod).stdout[-2000:]
    pytest.fail(f"pod {pod} not Running in {timeout_s}s\n{logs}")


def _dump_cluster_state() -> None:
    print("\n========== SR CLUSTER STATE ==========", file=sys.stdout)
    sys.stdout.flush()
    subprocess.run(
        ["kubectl", "-n", SR_NAMESPACE, "get", "pods", "-o", "wide"],
        check=False,
        timeout=30,
    )
    subprocess.run(
        ["kubectl", "-n", SR_NAMESPACE, "get", "events", "--sort-by=.lastTimestamp"],
        check=False,
        timeout=30,
    )
    print("======================================\n", file=sys.stdout)
    sys.stdout.flush()


@pytest.fixture(scope="session")
def sr_cluster():
    """Create an isolated k3d cluster, helm-install the lean SR profile, and
    pull the tiny Ollama model the router proxies. Tears the cluster down."""
    if shutil.which("k3d") is None or shutil.which("helm") is None:
        pytest.skip("k3d/helm not installed; cannot stand up the SR test cluster")

    from cogniverse_cli.cluster import create_cluster, delete_cluster
    from cogniverse_cli.deploy import helm_install

    # Fresh cluster — drop any leftover from a prior interrupted run.
    subprocess.run(
        ["k3d", "cluster", "delete", SR_CLUSTER],
        capture_output=True,
        timeout=120,
        check=False,
    )
    create_cluster(
        name=SR_CLUSTER,
        ports=[],
        workspace_path=None,
        share_hf_cache=False,
        share_host_storage=False,
    )

    node = f"k3d-{SR_CLUSTER}-server-0"
    for img in _THIRD_PARTY_IMAGES:
        _import_image_to_node(img, node)

    try:
        try:
            helm_install(
                _CHART_PATH,
                [_K3S_VALUES, _SR_VALUES],
                namespace=SR_NAMESPACE,
                timeout="10m",
            )
        except RuntimeError:
            _dump_cluster_state()
            raise

        # Ollama has no auto-pull; fetch the tiny test model into the pod.
        _wait_pod_running(OLLAMA_POD, timeout_s=300)
        deadline = time.time() + 600
        while time.time() < deadline:
            r = _kc(
                "exec", OLLAMA_POD, "--", "ollama", "pull", OLLAMA_MODEL, timeout=600
            )
            if r.returncode == 0:
                break
            time.sleep(10)
        else:
            pytest.fail(f"ollama pull {OLLAMA_MODEL} failed:\n{r.stderr[-1500:]}")

        yield SR_NAMESPACE
    finally:
        delete_cluster(SR_CLUSTER)


@pytest.fixture(scope="session")
def sr_endpoints(sr_cluster):
    """Wait for the deployed router's classifier, then port-forward its Envoy
    entry + metrics port."""
    # The router serves a placeholder classifier while it downloads its
    # bundle (~GB) on first boot; requests routed during that window skip
    # signal evaluation. ``startup_complete`` in the router log marks the
    # real classifier ready.
    deadline = time.time() + 900
    while time.time() < deadline:
        logs = _kc("logs", f"deployment/{_ROUTER_SVC}", "--tail", "200", timeout=30)
        if "startup_complete" in (logs.stdout + logs.stderr):
            break
        time.sleep(5)
    else:
        tail = _kc("logs", f"deployment/{_ROUTER_SVC}", "--tail", "40").stdout
        pytest.fail(f"deployed semantic-router classifier not ready in 900s\n{tail}")

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


def _port_forward(svc: str, remote_port: int, ready_path: str | None):
    """kubectl port-forward svc/<svc> <local>:<remote>; wait until reachable."""
    local = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "-n",
            SR_NAMESPACE,
            "port-forward",
            f"svc/{svc}",
            f"{local}:{remote_port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 60
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
        time.sleep(2)
    proc.terminate()
    pytest.fail(f"port-forward to {svc}:{remote_port} not reachable in 60s")


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
    """Route a real completion through the deployed router; return the answer."""
    endpoint = LLMEndpointConfig(
        model="openai/auto",
        api_base="http://unused:1/v1",
        max_tokens=64,
        request_timeout=180,
        num_retries=2,
    )
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


def test_semantic_router_pods_running(sr_cluster):
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
