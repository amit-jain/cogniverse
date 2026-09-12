"""A tenant's stored tier steers the deployed router on the production path.

Set the tier through the deployed runtime's admin route, send one real query
through the dispatch route, and read the decision the deployed semantic router
recorded for it. The runtime resolves the tier itself from its config store, so
this exercises the whole chain the cluster runs: admin write -> the runtime's
per-tenant reader (and its in-process invalidation) -> the router's decision.

On this chart both catalog models resolve to the same upstream and the same
``provider_model_id`` (``charts/cogniverse/files/semantic-router/config.yaml``
lines 25-44), so promoting a tenant does NOT change which model answers. What
it changes is the decision the router matches and the reasoning flag it sends,
which is what this reads.
"""

from __future__ import annotations

import re
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
import yaml

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.unified_config import ROUTER_TIERS
from tests.e2e.conftest import (
    KUBECTL_CONTEXT,
    RUNTIME,
    register_tenant_and_wait,
    unique_id,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

# The release namespace the e2e cluster deploys into, as every other kubectl
# read in tests/e2e/conftest.py spells it.
NAMESPACE = "cogniverse"
_ROUTER_SVC = "cogniverse-semantic-router"
_ROUTER_METRICS_PORT = 9190
CHART_ROUTER_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "charts"
    / "cogniverse"
    / "files"
    / "semantic-router"
    / "config.yaml"
)
QUERY = "summarise what this tenant has ingested"
AGENT = "summarizer_agent"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _decisions_by_tier() -> dict[str, str]:
    """tier -> the name of the decision the deployed router binds for it.

    Read from the chart's own role bindings and decisions: the Go templating in
    the model blocks is irrelevant to the routing section, which is literal.
    """
    text = "\n".join(
        line
        for line in CHART_ROUTER_CONFIG.read_text().splitlines()
        if "{{" not in line
    )
    routing = yaml.safe_load(text)["routing"]
    role_by_group = {
        subject["name"]: binding["role"]
        for binding in routing["signals"]["role_bindings"]
        for subject in binding["subjects"]
        if subject["kind"] == "Group"
    }
    by_tier: dict[str, str] = {}
    for decision in routing["decisions"]:
        conditions = decision["rules"]["conditions"]
        roles = {c["name"] for c in conditions if c["type"] == "authz"}
        if len(conditions) != 1 or not roles:
            continue
        (role,) = roles
        by_tier[next(g for g, r in role_by_group.items() if r == role)] = decision[
            "name"
        ]
    return by_tier


@pytest.fixture(scope="module")
def router_metrics_url():
    """kubectl port-forward to the deployed router's metrics endpoint."""
    local = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "port-forward",
            f"svc/{_ROUTER_SVC}",
            f"{local}:{_ROUTER_METRICS_PORT}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{local}/metrics"
    deadline = time.time() + 60
    try:
        while True:
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    break
            except (OSError, httpx.HTTPError):
                pass
            if time.time() >= deadline:
                proc.terminate()
                pytest.fail(
                    f"port-forward to {_ROUTER_SVC}:{_ROUTER_METRICS_PORT} did not "
                    f"serve /metrics in 60s"
                )
            time.sleep(2)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _decision_count(metrics_url: str, decision_name: str) -> int:
    body = httpx.get(metrics_url, timeout=10).text
    match = re.search(
        rf'^llm_decision_match_total\{{decision_name="{re.escape(decision_name)}"\}}'
        rf"\s+([0-9.]+)",
        body,
        re.MULTILINE,
    )
    return int(float(match.group(1))) if match else 0


def _set_tier(tenant_id: str, tier: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        response = client.put(
            f"{RUNTIME}/admin/tenants/{tenant_id}/tier", json={"tier": tier}
        )
    assert response.status_code == 200, response.text
    return response.json()


def _dispatch_one_query(tenant_id: str) -> None:
    """One real query through the production dispatch route.

    Body shape is the route's ``AgentTask``: agent_name + query, tenant inside
    ``context``.
    """
    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            f"{RUNTIME}/agents/{AGENT}/process",
            json={
                "agent_name": AGENT,
                "query": QUERY,
                "context": {"tenant_id": tenant_id},
            },
        )
    assert response.status_code == 200, response.text[:600]


def test_the_stored_tier_steers_the_deployed_router(router_metrics_url):
    """Each tier set through the admin route changes the decision the deployed
    router matches for the next query on the production dispatch path."""
    decision_by_tier = _decisions_by_tier()
    assert set(decision_by_tier) == set(ROUTER_TIERS)

    tenant_id = unique_id("tier")
    register_tenant_and_wait(tenant_id)

    walk = sorted(ROUTER_TIERS)
    walk.append(walk[0])

    observed = []
    for tier in walk:
        decision = decision_by_tier[tier]
        before = _decision_count(router_metrics_url, decision)
        assert _set_tier(tenant_id, tier) == {
            "tenant_id": canonical_tenant_id(tenant_id),
            "tier": tier,
        }
        _dispatch_one_query(tenant_id)
        observed.append(_decision_count(router_metrics_url, decision) - before)

    assert observed == [1] * len(walk), (
        f"router decision counters for {tenant_id!r} moved by {observed} over "
        f"tiers {walk}; each tier must match its own decision exactly once"
    )
