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


def _runtime_already_up() -> bool:
    """Return True if a cogniverse runtime is already reachable.

    Probes the default ``cogniverse up`` NodePort (localhost:28000). If the
    developer has a stack running, the deployment-lifecycle tests have
    nothing to prove — they'd either collide with the existing cluster or
    duplicate coverage the regular e2e suite already provides.
    """
    import httpx

    try:
        r = httpx.get("http://localhost:28000/health/live", timeout=2.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout, OSError):
        return False


def refresh_workload_pods_if_devmode(
    namespace: str = "cogniverse", timeout_s: int = 240
) -> bool:
    """Restart runtime + dashboard pods so devMode bind-mounted code is reloaded.

    k3d + devMode mounts the laptop's ``libs/`` into the pod at ``/app/libs``,
    so file edits are immediately visible on disk — but the pod's Python
    interpreter imported the old modules at startup and won't pick up edits
    without a process restart. A test suite that runs against a long-lived
    ``cogniverse up`` stack therefore gets stale behaviour unless we force
    a reload here.

    Uses ``kubectl delete pod`` rather than ``kubectl rollout restart``
    because rolling updates require the node to fit two pods simultaneously,
    and the single-node k3d laptop setup is memory-constrained.

    No-op (returns True) if:
      * ``COGNIVERSE_SKIP_POD_REFRESH`` env var is set truthy (debug / fast
        dev cycles where you know code hasn't changed);
      * ``kubectl`` is unavailable or the k3d-cogniverse context isn't set;
      * no pods have a ``src-libs`` hostPath volume (production mode —
        code is baked into the image and a restart wouldn't change anything).

    Returns True on success, False if the runtime didn't come back healthy
    within ``timeout_s`` seconds.
    """
    import os as _os
    import time as _time

    import httpx

    if _os.environ.get("COGNIVERSE_SKIP_POD_REFRESH", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return True

    def _kc(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", "--context=k3d-cogniverse", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    probe = _kc("get", "ns", namespace, "-o", "name", timeout=10)
    if probe.returncode != 0:
        return (
            True  # no k3d context → production k8s or no cluster → nothing to refresh
        )

    components_json = _kc(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        "app.kubernetes.io/instance=cogniverse",
        "-o",
        'jsonpath={range .items[*]}{.metadata.name}|{.spec.volumes[?(@.hostPath.path=="/cogniverse-src/libs")].name}\n{end}',
        timeout=15,
    )
    if components_json.returncode != 0:
        return True

    devmode_pods = [
        line.split("|", 1)[0]
        for line in components_json.stdout.strip().splitlines()
        if "|" in line and line.split("|", 1)[1].strip()
    ]
    if not devmode_pods:
        return True  # production mode — no bind-mounts, no stale-code risk

    print(f"Refreshing {len(devmode_pods)} devMode pod(s) to pick up latest code...")
    for pod in devmode_pods:
        _kc("delete", "pod", pod, "-n", namespace, "--grace-period=5", timeout=30)

    # Wait for runtime to be fully ready. /health/live returns 200 as soon as
    # uvicorn binds (before routers / agents finish loading), so probe the
    # richer /health endpoint — it resolves the agent registry, dispatches
    # a backend ping, and only returns 200 once POST traffic is safe.
    for _ in range(timeout_s):
        _time.sleep(1)
        try:
            r = httpx.get("http://localhost:28000/health", timeout=10.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, OSError):
            pass
    return False


@pytest.fixture(scope="session", autouse=True)
def e2e_stack():
    """Override the parent ``tests/e2e/conftest.py`` autouse ``e2e_stack``.

    The parent fixture assumes a running ``cogniverse up`` stack. Tests in
    this directory create their own k3d cluster via ``deployed_stack``
    below, so the parent check would either skip them incorrectly or
    collide with the self-managed cluster.

    Behaviour:
      * If ``cogniverse up`` is already running (runtime reachable at
        localhost:28000), skip this whole subsuite — the deployment
        lifecycle is what ``cogniverse up`` just did, and the regular
        e2e suite covers the running-stack path.
      * Otherwise yield so ``k3d_cluster`` / ``deployed_stack`` can
        bring up their own isolated test cluster.
    """
    if _runtime_already_up():
        pytest.skip(
            "cogniverse runtime already reachable at localhost:28000 — "
            "deployment-lifecycle tests are a no-op when a stack is up. "
            "Run 'cogniverse down' first if you want to exercise the "
            "fresh-install path."
        )
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
    "whisper": 51998,
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

    # Create cluster with offset port mappings. The main chart declares
    # NodePort services with fixed ports in the 26xxx range (see
    # ``values.k3s.yaml``), but Kubernetes' default NodePort range is
    # 30000-32767 — so Helm install fails with "provided port is not in
    # the valid range" unless we widen the range via k3s-arg. This
    # matches ``cogniverse_cli.cluster.create_cluster``.
    cmd = [
        "k3d",
        "cluster",
        "create",
        CLUSTER_NAME,
        "--k3s-arg",
        "--service-node-port-range=1-65535@server:0",
    ]
    for port in PORTS.values():
        cmd.extend(["-p", f"{port}:{port}@loadbalancer"])

    # GPU passthrough — k3d nodes are docker containers, so the host's
    # GPU device files have to be mounted in for any in-cluster device
    # plugin to find them. NVIDIA has --gpus=all; AMD has no equivalent
    # so we mount /dev/kfd + /dev/dri explicitly when the build host
    # detected rocm. CPU build host: no mounts.
    from cogniverse_cli.images import detect_torch_backend

    host_backend = detect_torch_backend()
    if host_backend == "cuda":
        cmd.append("--gpus=all")
    elif host_backend == "rocm":
        cmd += [
            "--volume",
            "/dev/kfd:/dev/kfd",
            "--volume",
            "/dev/dri:/dev/dri",
        ]

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

    # Build and import images matching the host's torch backend. Each
    # backend ships its own image (cogniverse/runtime-cpu / -cuda / -rocm,
    # same for dashboard); the chart picks the matching tag via
    # ``runtime.imagesByBackend[backend]``. PREDOWNLOAD_MODELS is on by
    # default so the agent pipeline starts fast; flip the env var off to
    # save image build time when the tests don't exercise inference.
    import os as _os

    from cogniverse_cli.images import (
        DASHBOARD_TAGS_BY_BACKEND,
        RUNTIME_TAGS_BY_BACKEND,
        detect_torch_backend,
    )

    backend = detect_torch_backend()
    runtime_tag = RUNTIME_TAGS_BY_BACKEND[backend]
    dashboard_tag = DASHBOARD_TAGS_BY_BACKEND[backend]

    build_args: list[str] = ["--build-arg", f"TORCH_BACKEND={backend}"]
    if _os.environ.get("PREDOWNLOAD_MODELS", "true").lower() in ("0", "false", "no"):
        build_args += ["--build-arg", "PREDOWNLOAD_MODELS=false"]

    for dockerfile, tag in [
        ("libs/runtime/Dockerfile", runtime_tag),
        ("libs/dashboard/Dockerfile", dashboard_tag),
    ]:
        # Cold build runs the full workspace install (uv sync —
        # transformers, dspy, ColPali, VideoPrism, etc.), then the
        # backend-specific torch reinstall, then GLiNER predownload
        # (1.5 GB). Easily 15-25 min on the first run; cached
        # subsequent runs finish in seconds via docker layer cache.
        _cmd(
            ["docker", "build", "-f", dockerfile, *build_args, "-t", tag, "."],
            timeout=1800,
        )

    # Inference sidecars deployed by the chart when their `enabled` flag
    # flips to true. The whisper sidecar is gated behind ``whisper.enabled``
    # in values.k3s.yaml; build the image unconditionally so a deploy with
    # the flag on doesn't fail with ImagePullBackOff. The whisper Dockerfile
    # doesn't consume TORCH_BACKEND.
    _cmd(
        [
            "docker",
            "build",
            "-f",
            "deploy/whisper/Dockerfile",
            "-t",
            "cogniverse/whisper-fw:dev",
            "deploy/whisper",
        ],
        timeout=600,
    )

    _cmd(
        [
            "k3d",
            "image",
            "import",
            runtime_tag,
            dashboard_tag,
            "cogniverse/whisper-fw:dev",
            "-c",
            CLUSTER_NAME,
        ],
        timeout=300,
    )

    # No AMD device plugin install — runtime mounts /dev/kfd and
    # /dev/dri via hostPath when backend=rocm (chart's
    # ``$runtimeBackend == "rocm"`` branch). Skips the device-plugin
    # readiness wait that was timing the helm post-install hook out.
    # NVIDIA still routes through k3s's built-in nvidia.com/gpu
    # support when --gpus=all is set on cluster create.

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

    # Helm install. argo-workflows.crds.install=false so the sub-chart
    # doesn't reinstall the CRDs we just laid down. runtime.backend /
    # dashboard.backend pin the chart's image-by-backend selection to
    # the variant we just built (matches host's torch wheel).
    # Helm post-install hook (the schema-deployment Job in
    # templates/init-jobs.yaml) curl-waits for runtime /health, which
    # itself blocks on Vespa's config-server. Vespa's bundle-load + ZK
    # replay alone is ~5 min cold; runtime startup adds 1-2 min;
    # schema deploy itself another 1-2 min. The default helm
    # --timeout=10m doesn't leave enough headroom — bumping to 20m so
    # the hook chain reliably completes. The outer subprocess timeout
    # tracks helm's window plus a small buffer for argo etc.
    _cmd(
        [
            "helm",
            "install",
            "cogniverse",
            str(chart_path),
            "--set",
            "argo-workflows.crds.install=false",
            "--set",
            f"runtime.backend={backend}",
            "--set",
            f"dashboard.backend={backend}",
            "--namespace",
            NAMESPACE,
            "--create-namespace",
            "-f",
            str(values_file),
            "--timeout",
            "20m",
        ],
        timeout=1320,
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
        ("svc/cogniverse-whisper", f"{PORTS['whisper']}:7998"),
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
        "whisper_url": f"http://localhost:{PORTS['whisper']}",
    }

    # Cleanup port-forwards
    for proc in port_forwards:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
