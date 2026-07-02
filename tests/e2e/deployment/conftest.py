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
import sys
import time

import httpx
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

    # kubectl rollout status blocks until readyReplicas == replicas, which
    # /health probes alone don't guarantee — they can land on a pod from
    # the old rollout that's about to terminate.
    for deploy in ("cogniverse-runtime", "cogniverse-dashboard"):
        rollout = _kc(
            "rollout",
            "status",
            f"deployment/{deploy}",
            "-n",
            namespace,
            f"--timeout={timeout_s}s",
            timeout=timeout_s + 30,
        )
        if rollout.returncode != 0:
            return False

    # Two consecutive 200s — guards against /health hitting an
    # about-to-terminate pod just before the LB endpoints flip.
    for _ in range(timeout_s):
        _time.sleep(1)
        try:
            r = httpx.get("http://localhost:28000/health", timeout=10.0)
            if r.status_code == 200:
                _time.sleep(2)
                r2 = httpx.get("http://localhost:28000/health", timeout=10.0)
                if r2.status_code == 200:
                    return True
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            OSError,
        ):
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


CLUSTER_NAME = "cogniverse-deploy-test"
NAMESPACE = "cogniverse-deploy-test"

# High ephemeral ports to avoid collision with production cluster or common services
PORTS = {
    "vespa_http": 51080,
    "vespa_config": 51071,
    "runtime": 51000,
    "dashboard": 51501,
    "phoenix": 51006,
    "otel_grpc": 51317,
    "llm": 51434,
    "vllm_asr": 51998,
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
    """Create an isolated k3d test cluster provisioned like ``cogniverse up``.

    ``values.k3s.yaml`` uses hostStorage (hostPath ``/host-hf-cache`` +
    ``/host-data``) and schedules GPU inference pods with a
    ``amd.com/gpu.present`` / ``nvidia.com/gpu.present`` nodeSelector, so the
    cluster must bind-mount those host paths AND label the node — otherwise the
    GPU pods stay Pending and the runtime's hf-cache mount fails. ``ports=[]``
    skips the loadbalancer mappings because service access goes through
    ``kubectl port-forward`` to the ``PORTS`` in the deployed_stack fixture."""
    if _cluster_exists():
        _cmd(["k3d", "cluster", "delete", CLUSTER_NAME], check=False, timeout=60)

    import os

    from cogniverse_cli.cluster import create_cluster
    from cogniverse_cli.images import detect_torch_backend

    # The hostPath source for /host-hf-cache must exist for the bind-mount.
    os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)

    try:
        create_cluster(
            name=CLUSTER_NAME,
            ports=[],
            workspace_path=None,
            share_hf_cache=True,
            share_host_storage=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"k3d cluster creation failed: {(exc.stderr or '').strip()[:300] or exc}"
        )

    # Label the node so GPU inference pods schedule, as `cogniverse up` does.
    backend = detect_torch_backend()
    if backend == "rocm":
        _cmd(
            [
                "kubectl",
                "label",
                "node",
                "--all",
                "amd.com/gpu.present=true",
                "--overwrite",
            ],
            check=False,
        )
    elif backend == "cuda":
        _cmd(
            [
                "kubectl",
                "label",
                "node",
                "--all",
                "nvidia.com/gpu.present=true",
                "--overwrite",
            ],
            check=False,
        )

    yield {
        "cluster_name": CLUSTER_NAME,
        "namespace": NAMESPACE,
        "ports": PORTS,
    }

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

    from cogniverse_cli.config import get_device_values_file
    from cogniverse_cli.images import (
        build_images,
        detect_torch_backend,
        import_images,
    )

    backend = detect_torch_backend()
    device_values_file = get_device_values_file(backend, project_root=project_root)

    # Build the canonical image set the chart's k3s values enable —
    # runtime + dashboard for the host's torch backend, plus the
    # locally-built inference sidecars (pylate / colpali). vLLM-served
    # ASR (inference.vllm_asr) pulls vllm/vllm-openai-cpu directly from
    # docker hub at pod-start, so no local build is needed for it.
    # build_images() owns the docker-build invocations + the matching
    # --build-arg TORCH_BACKEND wiring; import_images() lifts them into
    # the k3d cluster so pods with pullPolicy=Never can find them.
    built_tags = build_images(project_root, torch_backend=backend)
    import_images(CLUSTER_NAME, built_tags)

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

    # Sync the HF token into the test namespace before install — gated models
    # (e.g. inference.vllm_llm_student → google/gemma-4-e4b-it) reference the
    # hf-token Secret and otherwise crash with CreateContainerConfigError.
    # cogniverse up does this via sync_hf_token_to_cluster, but that targets
    # the fixed "cogniverse" namespace, so replicate it into NAMESPACE here.
    from cogniverse_cli.secrets import HF_TOKEN_SECRET, _read_hf_token

    _hf_token = _read_hf_token()
    if _hf_token:
        subprocess.run(
            ["kubectl", "create", "namespace", NAMESPACE],
            capture_output=True,
            timeout=30,
            check=False,
        )
        _rendered = subprocess.run(
            [
                "kubectl",
                "create",
                "secret",
                "generic",
                HF_TOKEN_SECRET,
                "-n",
                NAMESPACE,
                f"--from-literal=HF_TOKEN={_hf_token}",
                "--dry-run=client",
                "-o",
                "yaml",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=_rendered.stdout,
            capture_output=True,
            text=True,
            timeout=30,
        )

    # Helm timeout bumped to 20m for cold-start clusters — Vespa
    # bundle-load + ZK replay alone is ~5 min, runtime startup adds
    # 1-2 min, and the schema-deployment post-install hook another
    # 1-2 min. devMode + sandbox forced off because the test cluster
    # has no /cogniverse-src bind-mount or openshell-mtls secret.
    from cogniverse_cli.deploy import helm_install

    helm_values = [values_file]
    if device_values_file:
        helm_values.append(device_values_file)
    helm_set_overrides = {
        "argo-workflows.crds.install": "false",
        "runtime.backend": backend,
        "dashboard.backend": backend,
        "devMode.enabled": "false",
        "runtime.sandbox.enabled": "false",
    }

    def _dump_pod_state() -> None:
        """Snapshot cluster state to pytest's captured stdout — runs on
        any helm-install failure so the next teardown doesn't take the
        evidence with it."""
        import sys

        print("\n========== POD STATE ON HELM FAILURE ==========", file=sys.stdout)
        for diag in [
            ["kubectl", "get", "pods", "-n", NAMESPACE, "-o", "wide"],
            ["kubectl", "get", "events", "-n", NAMESPACE, "--sort-by=.lastTimestamp"],
            ["kubectl", "describe", "pods", "-n", NAMESPACE],
        ]:
            print(f"\n--- {' '.join(diag)} ---", file=sys.stdout)
            sys.stdout.flush()
            subprocess.run(diag, check=False, timeout=60)
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-o",
                "jsonpath={range .items[?(@.status.phase!='Running')]}"
                "{.metadata.name}{'\\n'}{end}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        for pod in (result.stdout or "").split():
            if not pod:
                continue
            print(f"\n--- kubectl logs {pod} (last 100 lines) ---", file=sys.stdout)
            sys.stdout.flush()
            subprocess.run(
                ["kubectl", "logs", "-n", NAMESPACE, pod, "--tail=100"],
                check=False,
                timeout=30,
            )
        print("================================================\n", file=sys.stdout)

    try:
        helm_install(
            chart_path,
            helm_values,
            set_values=helm_set_overrides,
            namespace=NAMESPACE,
            timeout="20m",
        )
    except RuntimeError:
        _dump_pod_state()
        raise

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

    # Port-forward with offset ports. Capture stderr (instead of
    # silencing) so a dead port-forward can surface its reason later.
    port_forwards: list[tuple[subprocess.Popen, str]] = []
    pf_specs = [
        ("svc/cogniverse-vespa", f"{PORTS['vespa_http']}:8080"),
        ("svc/cogniverse-vespa", f"{PORTS['vespa_config']}:19071"),
        ("svc/cogniverse-runtime", f"{PORTS['runtime']}:8000"),
        ("svc/cogniverse-dashboard", f"{PORTS['dashboard']}:8501"),
        ("svc/cogniverse-phoenix", f"{PORTS['phoenix']}:6006"),
        ("svc/cogniverse-phoenix", f"{PORTS['otel_grpc']}:4317"),
        ("svc/cogniverse-llm", f"{PORTS['llm']}:11434"),
        ("svc/cogniverse-vllm-asr", f"{PORTS['vllm_asr']}:8000"),
    ]
    for svc, ports in pf_specs:
        pf_log = open(f"/tmp/pf_{svc.replace('/', '_')}_{ports}.log", "w")
        proc = subprocess.Popen(
            ["kubectl", "port-forward", svc, ports, "-n", NAMESPACE],
            stdout=pf_log,
            stderr=subprocess.STDOUT,
        )
        port_forwards.append((proc, pf_log.name))

    # Poll until the runtime port-forward responds, then give peers a
    # short grace window. Replaces a 5-second sleep that wasn't enough
    # under load — kubectl port-forward needs ~1s per endpoint and the
    # upstream pod's first /health response can lag a few seconds more.
    runtime_url = f"http://localhost:{PORTS['runtime']}/health"
    for attempt in range(30):
        try:
            r = httpx.get(runtime_url, timeout=2)
            if r.status_code < 500:
                break
        except httpx.RequestError:
            pass
        time.sleep(2)
    else:
        # Surface port-forward stderr if the runtime never became
        # reachable. The session fixture would otherwise yield a stack
        # of "Server disconnected" errors with no breadcrumbs.
        print("\n========== PORT-FORWARD STATE ==========", file=sys.stdout)
        for proc, log_path in port_forwards:
            print(f"\n--- {log_path} (pid={proc.pid}, alive={proc.poll() is None}) ---")
            try:
                with open(log_path) as fh:
                    print(fh.read()[-2000:])
            except OSError:
                pass
        print("=========================================\n", file=sys.stdout)
        sys.stdout.flush()
    time.sleep(2)  # let other port-forwards settle

    yield {
        "runtime_url": f"http://localhost:{PORTS['runtime']}",
        "dashboard_url": f"http://localhost:{PORTS['dashboard']}",
        "vespa_url": f"http://localhost:{PORTS['vespa_http']}",
        "phoenix_url": f"http://localhost:{PORTS['phoenix']}",
        "llm_url": f"http://localhost:{PORTS['llm']}",
        "vllm_asr_url": f"http://localhost:{PORTS['vllm_asr']}",
    }

    # Cleanup port-forwards
    for proc, _log_path in port_forwards:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
