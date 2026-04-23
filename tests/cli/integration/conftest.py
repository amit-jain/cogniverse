"""Session-scoped fixtures for CLI integration tests.

Creates a real k3d cluster once per test session with isolated ports.
Teardown deletes the cluster after all tests complete.

Requires: docker, k3d, kubectl, helm installed.
"""

import subprocess
import time

import pytest

CLUSTER_NAME = "cogniverse-test"
NAMESPACE = "cogniverse-test"

# High ephemeral ports to avoid collision with production cluster or common services
PORTS = {
    "vespa_http": 51080,
    "vespa_config": 51071,
    "runtime": 51000,
    "dashboard": 51501,
    "phoenix": 51006,
    "otel_grpc": 51317,
    "llm": 51434,
}


def _cmd(
    args: list[str], *, timeout: int = 120, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=check
    )


def _cluster_exists() -> bool:
    result = _cmd(["k3d", "cluster", "list", CLUSTER_NAME], check=False)
    return result.returncode == 0 and CLUSTER_NAME in result.stdout


@pytest.fixture(scope="session")
def k3d_cluster():
    """Create an isolated k3d test cluster with offset ports."""
    # Clean up any leftover test cluster
    if _cluster_exists():
        _cmd(["k3d", "cluster", "delete", CLUSTER_NAME], check=False, timeout=60)

    # Create cluster with offset port mappings
    cmd = ["k3d", "cluster", "create", CLUSTER_NAME]
    for port in PORTS.values():
        cmd.extend(["-p", f"{port}:{port}@loadbalancer"])

    result = _cmd(cmd, timeout=120, check=False)
    if result.returncode != 0:
        pytest.fail(f"k3d cluster creation failed: {result.stderr.strip()[:300]}")

    yield {
        "cluster_name": CLUSTER_NAME,
        "namespace": NAMESPACE,
        "ports": PORTS,
    }

    # Teardown — always delete the test cluster
    _cmd(["k3d", "cluster", "delete", CLUSTER_NAME], check=False, timeout=60)


@pytest.fixture(scope="session")
def deployed_stack(k3d_cluster):
    """Deploy the full cogniverse stack via Helm into the test cluster."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    chart_path = project_root / "charts" / "cogniverse"
    values_file = chart_path / "values.k3s.yaml"

    assert chart_path.exists(), f"Chart not found: {chart_path}"
    assert values_file.exists(), f"Values not found: {values_file}"

    # Build and import images
    for dockerfile, tag in [
        ("libs/runtime/Dockerfile", "cogniverse/runtime:dev"),
        ("libs/dashboard/Dockerfile", "cogniverse/dashboard:dev"),
    ]:
        _cmd(
            ["docker", "build", "-f", dockerfile, "-t", tag, "."],
            timeout=600,
        )
    _cmd(
        [
            "k3d",
            "image",
            "import",
            "cogniverse/runtime:dev",
            "cogniverse/dashboard:dev",
            "-c",
            CLUSTER_NAME,
        ],
        timeout=300,
    )

    # Install Argo CRDs (chart references CronWorkflow resources)
    from cogniverse_cli.argo import install_argo_controller

    try:
        install_argo_controller(namespace="argo")
    except Exception as e:
        pytest.fail(f"Argo controller install failed: {e}")

    # Helm install
    _cmd(
        [
            "helm",
            "install",
            "cogniverse",
            str(chart_path),
            "--namespace",
            NAMESPACE,
            "--create-namespace",
            "-f",
            str(values_file),
            "--timeout",
            "10m",
        ],
        timeout=660,
    )

    # Wait for pods
    _cmd(
        [
            "kubectl",
            "wait",
            "--for=condition=ready",
            "pod",
            "-l",
            "app.kubernetes.io/instance=cogniverse",
            "-n",
            NAMESPACE,
            "--timeout=300s",
        ],
        check=False,
        timeout=310,
    )

    # Port-forward with offset ports
    port_forwards = []
    pf_specs = [
        ("svc/cogniverse-vespa", f"{PORTS['vespa_http']}:8080"),
        ("svc/cogniverse-vespa", f"{PORTS['vespa_config']}:19071"),
        ("svc/cogniverse-runtime", f"{PORTS['runtime']}:8000"),
        ("svc/cogniverse-dashboard", f"{PORTS['dashboard']}:8501"),
        ("svc/cogniverse-phoenix", f"{PORTS['phoenix']}:6006"),
        ("svc/cogniverse-phoenix", f"{PORTS['otel_grpc']}:4317"),
        ("svc/cogniverse-llm", f"{PORTS['llm']}:11434"),
    ]
    for svc, ports in pf_specs:
        proc = subprocess.Popen(
            ["kubectl", "port-forward", svc, ports, "-n", NAMESPACE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        port_forwards.append(proc)

    time.sleep(5)  # let port-forwards establish

    yield {
        "runtime_url": f"http://localhost:{PORTS['runtime']}",
        "dashboard_url": f"http://localhost:{PORTS['dashboard']}",
        "vespa_url": f"http://localhost:{PORTS['vespa_http']}",
        "phoenix_url": f"http://localhost:{PORTS['phoenix']}",
        "llm_url": f"http://localhost:{PORTS['llm']}",
    }

    # Cleanup port-forwards
    for proc in port_forwards:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
