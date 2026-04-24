"""Session-scoped fixtures for deployment-lifecycle e2e tests.

Creates its own isolated k3d cluster with offset ports (51xxx) so the
tests stand up a real deployment and verify it end-to-end, independent
of any ``cogniverse up`` cluster the developer may already have
running.

The parent ``tests/e2e/conftest.py`` has a session-scoped autouse
``e2e_stack`` fixture that expects a pre-existing cluster — we override
it here with a no-op so this subsuite can manage its own lifecycle
without being short-circuited.

Requires: docker, k3d, kubectl, helm installed.
"""

import subprocess
import time

import pytest


@pytest.fixture(scope="session", autouse=True)
def e2e_stack():
    """Override the parent ``tests/e2e/conftest.py`` autouse ``e2e_stack``.

    The parent fixture assumes a running ``cogniverse up`` stack. Tests in
    this directory create their own k3d cluster via ``deployed_stack``
    below, so the parent check would either skip them incorrectly or
    collide with the self-managed cluster. Yielding a no-op disables the
    parent's check for this subsuite only."""
    yield


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
    """Run a command with captured output.

    On failure, prints the captured stdout+stderr tails directly to
    stdout so pytest's captured-output buffer (and thus the CI log)
    surfaces the real reason — ``CalledProcessError.stderr`` kwargs
    don't make it into ``--tb=long`` tracebacks, so without printing
    we'd only see "returned non-zero exit status 1".
    """
    import sys

    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
    except subprocess.CalledProcessError as exc:
        stderr_tail = (exc.stderr or "").strip()
        stdout_tail = (exc.stdout or "").strip()
        # Separator so the tail stands out in pytest's "Captured stdout"
        # section next to the usual docstring/traceback noise.
        print(
            f"\n========== FAILED: {' '.join(args[:5])}... "
            f"(exit {exc.returncode}) ==========",
            file=sys.stdout,
        )
        if stderr_tail:
            print(f"--- stderr (last 3000 chars) ---\n{stderr_tail[-3000:]}")
        if stdout_tail:
            print(f"--- stdout (last 1500 chars) ---\n{stdout_tail[-1500:]}")
        print("=" * 60)
        raise


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

    # Build and import images. Tests only verify deployment lifecycle
    # (pod health, helm release) — no agent queries — so CI runs with
    # CPU-only torch (no CUDA libs) and skip model pre-download to fit
    # in GHA's 14 GB disk. A developer running this locally can override
    # by unsetting these env vars.
    import os as _os

    build_args: list[str] = []
    if _os.environ.get("CPU_ONLY", "false").lower() in ("1", "true", "yes"):
        build_args += ["--build-arg", "CPU_ONLY=true"]
    if _os.environ.get("PREDOWNLOAD_MODELS", "true").lower() in ("0", "false", "no"):
        build_args += ["--build-arg", "PREDOWNLOAD_MODELS=false"]

    for dockerfile, tag in [
        ("libs/runtime/Dockerfile", "cogniverse/runtime:dev"),
        ("libs/dashboard/Dockerfile", "cogniverse/dashboard:dev"),
    ]:
        _cmd(
            ["docker", "build", "-f", dockerfile, *build_args, "-t", tag, "."],
            timeout=900,
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

    # Argo CRD chicken-and-egg: the main cogniverse chart references
    # CronWorkflow / WorkflowTemplate (argoproj.io/v1alpha1) resources.
    # Helm validates ALL manifests before any install step, so the
    # bundled argo-workflows sub-chart's CRDs aren't "live" when Helm
    # checks the main chart's CronWorkflow templates. Result:
    #   Error: resource mapping not found for kind "CronWorkflow"
    #   in version "argoproj.io/v1alpha1" — ensure CRDs are installed first.
    #
    # Solution: install the Argo CRDs before ``helm install``, and tell
    # the sub-chart not to install them itself (which would otherwise
    # fail with a release-ownership conflict).
    from cogniverse_cli.argo import install_argo_controller

    try:
        install_argo_controller(namespace="argo")
    except Exception as e:
        pytest.fail(f"Argo controller install failed: {e}")

    # Helm install (with argo-workflows.crds.install=false so the
    # sub-chart doesn't reinstall the CRDs we just laid down)
    _cmd(
        [
            "helm",
            "install",
            "cogniverse",
            str(chart_path),
            "--set",
            "argo-workflows.crds.install=false",
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
