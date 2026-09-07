"""The startupProbe protects the entrypoint's backend grace and startup stages.

The backend wait retries forever. Its grace window determines when the runtime
logs an ERROR; kubelet owns the eventual restart. Derive the minimum probe
window from production constants across every shipped values stack.
"""

import inspect
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from cogniverse_runtime import backend_startup
from cogniverse_runtime import main as runtime_main

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

# Every values stack a deployment ships with; the e2e cluster deploys the last
# one. An overlay may override the base probe, so each is rendered.
OVERLAY_STACKS = {
    "base": (),
    "k3s": ("values.k3s.yaml",),
    "k3s+rocm": ("values.k3s.yaml", "values.rocm.yaml"),
    "k3s+rocm+modal-llm": (
        "values.k3s.yaml",
        "values.rocm.yaml",
        "values.modal-llm.yaml",
    ),
}

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ci_fast,
    pytest.mark.skipif(
        shutil.which("helm") is None,
        reason="helm CLI not installed — chart tests require helm",
    ),
]


def _render_chart(*set_args: str, overlays: tuple[str, ...] = ()) -> list:
    args = ["helm", "template", "cogniverse", str(CHART_PATH)]
    for overlay in overlays:
        args += ["-f", str(CHART_PATH / overlay)]
    args += ["--set", "runtime.qualityMonitor.tenantId=test-tenant"]
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


def _window_seconds(probe: dict) -> int:
    """Seconds from container start until this probe can fail the container."""
    return (
        probe.get("initialDelaySeconds", 0)
        + probe["failureThreshold"] * probe["periodSeconds"]
    )


def _backend_wait_worst_case_s() -> float:
    """Grace plus an attempt's status, feed, and config-server probes."""
    return (
        backend_startup.BACKEND_STARTUP_WAIT_BUDGET_S
        + 3 * backend_startup.BACKEND_STARTUP_PROBE_TIMEOUT_S
    )


def _fresh_install_stages_s() -> float:
    """Nominal cost of the stages a fresh backend adds before the socket
    binds: the config-server TCP wait and the post-bootstrap config re-probe,
    both derived from their production defaults."""
    config_server_wait = inspect.signature(
        backend_startup._wait_for_config_server
    ).parameters
    return (
        config_server_wait["max_attempts"].default
        * config_server_wait["interval"].default
        + runtime_main.CONFIG_STORE_REPROBE_ATTEMPTS
        * runtime_main.CONFIG_STORE_REPROBE_INTERVAL_S
        + backend_startup.BACKEND_STARTUP_RETRY_INTERVAL_S
    )


def _worst_case_cold_start_s() -> float:
    return _backend_wait_worst_case_s() + _fresh_install_stages_s()


def test_backend_wait_budget_carries_double_the_longest_observed_recovery():
    """The wait must outlast the longest Vespa restart the code records, with
    a 2x margin for a cluster that has not pruned since an upgrade."""
    assert (
        backend_startup.BACKEND_STARTUP_WAIT_BUDGET_S
        == 2 * backend_startup.BACKEND_RECOVERY_WORST_CASE_S
    )
    assert backend_startup.BACKEND_RECOVERY_WORST_CASE_S == 32 * 60


@pytest.mark.parametrize("stack", sorted(OVERLAY_STACKS))
def test_runtime_has_startup_probe_in_every_shipped_stack(stack):
    container = _runtime_container(_render_chart(overlays=OVERLAY_STACKS[stack]))
    assert "startupProbe" in container, (
        f"{stack}: runtime container has no startupProbe — liveness then counts "
        f"down during the backend wait and kills a pod that is correctly waiting"
    )


@pytest.mark.parametrize("stack", sorted(OVERLAY_STACKS))
def test_startup_probe_targets_the_same_tcp_socket_as_liveness(stack):
    """It must probe the port that binds late (8000) over TCP — an HTTP probe
    stalls behind uvicorn workers exactly as the liveness comment describes."""
    container = _runtime_container(_render_chart(overlays=OVERLAY_STACKS[stack]))
    startup = container["startupProbe"]
    liveness = container["livenessProbe"]
    assert startup["tcpSocket"]["port"] == 8000
    assert startup["tcpSocket"]["port"] == liveness["tcpSocket"]["port"]


@pytest.mark.parametrize("stack", sorted(OVERLAY_STACKS))
def test_startup_window_exceeds_the_entrypoints_backend_grace(stack):
    """The probe allows the grace window and fresh-install stages to finish."""
    container = _runtime_container(_render_chart(overlays=OVERLAY_STACKS[stack]))
    window = _window_seconds(container["startupProbe"])
    assert window > _worst_case_cold_start_s(), (
        f"{stack}: startupProbe window {window}s does not exceed the protected "
        f"startup allowance {_worst_case_cold_start_s():.0f}s "
        f"(backend wait {_backend_wait_worst_case_s():.0f}s + fresh-install "
        f"stages {_fresh_install_stages_s():.0f}s) — the kubelet kills the pod "
        f"while the entrypoint is still inside its backend grace"
    )


@pytest.mark.parametrize("stack", sorted(OVERLAY_STACKS))
def test_liveness_cannot_fire_before_the_backend_wait_ends(stack):
    """Liveness is disabled until the startupProbe succeeds, so its own window
    only matters if the startupProbe is removed — pin that it would then kill
    inside the backend wait (why the startupProbe is load-bearing), and that
    the startupProbe gating it is present on the same container."""
    container = _runtime_container(_render_chart(overlays=OVERLAY_STACKS[stack]))
    liveness_window = _window_seconds(container["livenessProbe"])
    assert liveness_window < _backend_wait_worst_case_s(), (
        f"{stack}: liveness window {liveness_window}s now covers the backend "
        f"wait on its own — re-derive whether the startupProbe is still "
        f"load-bearing before relaxing this test"
    )
    assert "startupProbe" in container


def test_startup_probe_can_be_disabled_by_operator():
    """Setting runtime.startupProbe to null removes the block (guarded render),
    so an operator with a fast backend can opt out without editing the chart."""
    manifests = _render_chart("runtime.startupProbe=null")
    container = _runtime_container(manifests)
    assert "startupProbe" not in container


def test_termination_grace_covers_the_blob_write_drain():
    """Shutdown drains accepted admin blob writes before teardown; the pod's
    grace period must cover that drain plus the rest of teardown, or SIGKILL
    lands mid-drain and the accepted write is lost."""
    from cogniverse_runtime.routers.admin import drain_blob_writes

    deployments = [
        m
        for m in _render_chart()
        if m.get("kind") == "Deployment"
        and m.get("metadata", {}).get("name") == "cogniverse-runtime"
    ]
    assert len(deployments) == 1
    grace = deployments[0]["spec"]["template"]["spec"].get(
        "terminationGracePeriodSeconds"
    )
    assert grace == 90
    drain_budget = inspect.signature(drain_blob_writes).parameters["timeout_s"].default
    assert grace - drain_budget >= 15, (
        f"grace {grace}s leaves {grace - drain_budget}s after the {drain_budget}s "
        f"blob-write drain for the rest of teardown"
    )
