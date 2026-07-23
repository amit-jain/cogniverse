"""The runtime Deployment must carry a startupProbe that outlasts cold start.

uvicorn runs the FastAPI lifespan (wait-for-Vespa, then on a fresh backend a
metadata-schema bootstrap + deploy convergence + config re-probe) BEFORE it
binds port 8000. That cold start can take ~810s worst case; the liveness budget
alone (60 + 24*30 = 780s) killed a legitimately converging pod into a
crash-loop. A startupProbe gates liveness/readiness until the socket answers, so
its budget must comfortably exceed the worst-case cold start.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

# Worst-case cold start the startup budget must cover (see main.py lifespan:
# _wait_for_backend_ready ~300s + fresh-backend bootstrap ~330s + deploy
# convergence + 12-attempt re-probe ~110s). Pin the floor the probe must clear.
WORST_CASE_COLD_START_S = 810

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ci_fast,
    pytest.mark.skipif(
        shutil.which("helm") is None,
        reason="helm CLI not installed — chart tests require helm",
    ),
]


def _render_chart(*set_args: str) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for s in set_args:
        args += ["--set", s]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _runtime_container(manifests: list) -> dict:
    deployments = [
        m
        for m in manifests
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-runtime"
    ]
    assert len(deployments) == 1, (
        f"Expected exactly one cogniverse-runtime Deployment, got {len(deployments)}"
    )
    containers = deployments[0]["spec"]["template"]["spec"]["containers"]
    runtime = [c for c in containers if c["name"] == "runtime"]
    assert len(runtime) == 1, "runtime container missing from the Deployment"
    return runtime[0]


def _budget_seconds(probe: dict) -> int:
    return (
        probe.get("initialDelaySeconds", 0)
        + probe["failureThreshold"] * probe["periodSeconds"]
    )


def test_runtime_has_startup_probe():
    probe = _runtime_container(_render_chart()).get("startupProbe")
    assert probe is not None, (
        "runtime container has no startupProbe — without it the liveness probe "
        "counts down during the ~810s cold start and crash-loops the pod while "
        "Vespa converges"
    )


def test_startup_probe_targets_the_same_tcp_socket_as_liveness():
    """It must probe the port that binds late (8000) over TCP — an HTTP probe
    stalls behind uvicorn workers exactly as the liveness comment describes."""
    container = _runtime_container(_render_chart())
    startup = container["startupProbe"]
    liveness = container["livenessProbe"]
    assert startup["tcpSocket"]["port"] == 8000
    assert startup["tcpSocket"]["port"] == liveness["tcpSocket"]["port"]


def test_startup_budget_exceeds_worst_case_cold_start():
    """The startup budget must clear the worst-case cold start with margin — a
    budget below it reintroduces the crash-loop the probe exists to prevent."""
    container = _runtime_container(_render_chart())
    startup_budget = _budget_seconds(container["startupProbe"])
    assert startup_budget > WORST_CASE_COLD_START_S, (
        f"startupProbe budget {startup_budget}s does not exceed the worst-case "
        f"cold start {WORST_CASE_COLD_START_S}s — the pod crash-loops before "
        f"the socket binds"
    )


def test_liveness_budget_alone_would_not_cover_cold_start():
    """Pins WHY the startupProbe is required: liveness on its own is shorter
    than the cold start, so removing the startupProbe brings the bug back."""
    container = _runtime_container(_render_chart())
    liveness_budget = _budget_seconds(container["livenessProbe"])
    assert liveness_budget < WORST_CASE_COLD_START_S, (
        "liveness budget now covers cold start on its own — re-derive whether "
        "the startupProbe is still load-bearing before relaxing this test"
    )


def test_startup_probe_can_be_disabled_by_operator():
    """Setting runtime.startupProbe to null removes the block (guarded render),
    so an operator with a fast backend can opt out without editing the chart."""
    manifests = _render_chart("runtime.startupProbe=null")
    container = _runtime_container(manifests)
    assert "startupProbe" not in container
