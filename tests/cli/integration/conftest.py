"""Session-scoped fixtures for CLI integration tests.

Creates a real k3d cluster once per test session. All integration tests
share this cluster. Teardown deletes the cluster after all tests complete.

Requires: docker, k3d, kubectl, helm installed.
"""

import subprocess
import time

import pytest

CLUSTER_NAME = "cogniverse-test"
NAMESPACE = "cogniverse"


def _cmd(args: list[str], *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


def _cluster_exists() -> bool:
    result = _cmd(["k3d", "cluster", "list", CLUSTER_NAME], check=False)
    return result.returncode == 0


def _k3d_available() -> bool:
    """Check if k3d is installed and Docker is running."""
    for tool in ["docker", "k3d", "kubectl", "helm"]:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            return False
    # Verify Docker daemon is actually running
    result = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=10
    )
    return result.returncode == 0


@pytest.fixture(scope="session")
def k3d_cluster():
    """Create a k3d cluster for the test session, tear down after."""
    # Skip if prerequisites missing
    for tool in ["docker", "k3d", "kubectl", "helm"]:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            pytest.skip(f"{tool} not installed")

    # Clean up any leftover cluster
    if _cluster_exists():
        _cmd(["k3d", "cluster", "delete", CLUSTER_NAME], check=False, timeout=60)

    # Create cluster with port mappings
    ports = [8080, 19071, 8000, 8501, 6006, 4317, 11434, 2746]
    cmd = ["k3d", "cluster", "create", CLUSTER_NAME]
    for p in ports:
        cmd.extend(["-p", f"{p}:{p}@loadbalancer"])

    result = _cmd(cmd, timeout=120, check=False)
    if result.returncode != 0:
        pytest.skip(
            f"k3d cluster creation failed (ports in use or Docker issue): "
            f"{result.stderr.strip()[:200]}"
        )

    yield {
        "cluster_name": CLUSTER_NAME,
        "namespace": NAMESPACE,
        "ports": ports,
    }

    # Teardown
    _cmd(["k3d", "cluster", "delete", CLUSTER_NAME], check=False, timeout=60)


@pytest.fixture(scope="session")
def deployed_stack(k3d_cluster):
    """Deploy the full cogniverse stack via Helm into the test cluster."""
    from pathlib import Path

    # Find chart and values
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
        ["k3d", "image", "import",
         "cogniverse/runtime:dev", "cogniverse/dashboard:dev",
         "-c", CLUSTER_NAME],
        timeout=300,
    )

    # Helm install
    _cmd([
        "helm", "install", "cogniverse", str(chart_path),
        "--namespace", NAMESPACE, "--create-namespace",
        "-f", str(values_file),
        "--timeout", "10m",
    ], timeout=660)

    # Wait for pods
    _cmd([
        "kubectl", "wait", "--for=condition=ready", "pod",
        "-l", "app.kubernetes.io/instance=cogniverse",
        "-n", NAMESPACE, "--timeout=300s",
    ], check=False, timeout=310)

    # Start port-forwards
    port_forwards = []
    pf_specs = [
        ("svc/cogniverse-vespa", NAMESPACE, "8080:8080"),
        ("svc/cogniverse-vespa", NAMESPACE, "19071:19071"),
        ("svc/cogniverse-runtime", NAMESPACE, "8000:8000"),
        ("svc/cogniverse-dashboard", NAMESPACE, "8501:8501"),
        ("svc/cogniverse-phoenix", NAMESPACE, "6006:6006"),
        ("svc/cogniverse-phoenix", NAMESPACE, "4317:4317"),
        ("svc/cogniverse-llm", NAMESPACE, "11434:11434"),
    ]
    for svc, ns, ports in pf_specs:
        proc = subprocess.Popen(
            ["kubectl", "port-forward", svc, ports, "-n", ns],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        port_forwards.append(proc)

    time.sleep(5)  # let port-forwards establish

    yield {
        "runtime_url": "http://localhost:8000",
        "dashboard_url": "http://localhost:8501",
        "vespa_url": "http://localhost:8080",
        "phoenix_url": "http://localhost:6006",
        "llm_url": "http://localhost:11434",
    }

    # Cleanup port-forwards
    for proc in port_forwards:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
