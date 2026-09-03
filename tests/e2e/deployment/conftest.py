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

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from tests.e2e.conftest import KUBECTL_CONTEXT


def _probe_existing_runtime() -> httpx.Response | None:
    """Return the default-stack response when a runtime is already reachable.

    Probes the default ``cogniverse up`` NodePort (localhost:28000). If the
    developer has a stack running, the deployment-lifecycle tests have
    to fail explicitly because their isolated-cluster contract cannot be
    established safely.
    """
    try:
        response = httpx.get("http://localhost:28000/health/live", timeout=2.0)
        return response if response.status_code == 200 else None
    except (httpx.ConnectError, httpx.ReadTimeout, OSError):
        return None


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
            ["kubectl", "--context", KUBECTL_CONTEXT, *args],
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
def e2e_stack(resolved_inference_endpoints):
    """Override the parent ``tests/e2e/conftest.py`` autouse ``e2e_stack``.

    The parent fixture assumes a running ``cogniverse up`` stack. Tests in
    this directory create their own k3d cluster via ``deployed_stack``
    below, so the parent check would either skip them incorrectly or
    collide with the self-managed cluster.

    Behaviour:
      * If ``cogniverse up`` is already running (runtime reachable at
        localhost:28000), fail with the conflicting endpoint details so the
        isolated deployment is never silently left untested.
      * Otherwise yield so ``k3d_cluster`` / ``deployed_stack`` can
        bring up their own isolated test cluster.
    """
    response = _probe_existing_runtime()
    if response is not None:
        pytest.fail(
            "deployment-lifecycle isolation prerequisite failed because an "
            "existing runtime answered the default-stack probe; method='GET'; "
            "url='http://localhost:28000/health/live'; timeout=2.0s; "
            f"status={response.status_code}; body={response.text[:500]!r}; "
            "required_state='default endpoint unreachable'; action=\"run "
            "'cogniverse down' before exercising the fresh-install path\"",
            pytrace=False,
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


def _cluster_exists(cluster_name: str = CLUSTER_NAME) -> bool:
    result = _cmd(["k3d", "cluster", "list", cluster_name], check=False)
    return result.returncode == 0 and cluster_name in result.stdout


def create_test_cluster(
    cluster_name: str,
    *,
    ports: list[int] | None,
    share_host_storage: bool,
) -> None:
    """Create an isolated k3d test cluster provisioned like ``cogniverse up``.

    ``values.k3s.yaml`` uses hostStorage (hostPath ``/host-hf-cache`` +
    ``/host-data``) and schedules GPU inference pods with a
    ``amd.com/gpu.present`` / ``nvidia.com/gpu.present`` nodeSelector, so the
    cluster must bind-mount the hf-cache AND label the node — otherwise the
    GPU pods stay Pending and the runtime's hf-cache mount fails.

    ``ports=[]`` skips the loadbalancer mappings (service access via
    ``kubectl port-forward``); ``ports=None`` maps the canonical
    ``cogniverse up`` NodePorts; ``"host:node"`` string entries map offset
    host ports onto chart NodePorts (the main e2e suite's scheme).
    ``share_host_storage=False`` keeps /host-data node-local
    (DirectoryOrCreate) so the test cluster's Vespa/Phoenix data is fresh
    and cannot touch the dev cluster's persisted state.
    """
    if _cluster_exists(cluster_name):
        pytest.fail(
            f"Refusing to replace existing deployment-test cluster "
            f"{cluster_name!r}. Delete it explicitly with "
            f"`k3d cluster delete {cluster_name}`, then rerun."
        )

    from cogniverse_cli.cluster import create_cluster
    from cogniverse_cli.images import detect_torch_backend

    # The hostPath source for /host-hf-cache must exist for the bind-mount.
    os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)

    try:
        create_cluster(
            name=cluster_name,
            ports=ports,
            workspace_path=None,
            share_hf_cache=True,
            share_host_storage=share_host_storage,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"k3d cluster creation failed: {(exc.stderr or '').strip()[:300] or exc}"
        )

    # create_cluster already pins CoreDNS upstreams; re-assert here so a
    # cluster that raced the configmap's creation still converges (the pin
    # is idempotent and cheap when already applied).
    from cogniverse_cli.cluster import pin_coredns_upstreams

    pin_coredns_upstreams(cluster_name)

    if not share_host_storage:
        # /host-data is node-local here; the chart's DirectoryOrCreate
        # hostPath volumes would create root-owned dirs, and Phoenix runs
        # as a non-root uid — it exits 1 within a second when its working
        # dir isn't writable (and the runtime then crash-loops on the dead
        # telemetry backend). Pre-create the tree world-writable.
        _cmd(
            [
                "docker",
                "exec",
                f"k3d-{cluster_name}-server-0",
                "sh",
                "-c",
                "mkdir -p /host-data/phoenix /host-data/vespa"
                " && chmod -R 0777 /host-data",
            ],
            check=False,
        )

    # Label the node so GPU inference pods schedule, as `cogniverse up` does.
    backend = detect_torch_backend()
    if backend == "rocm":
        _cmd(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
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
                "--context",
                KUBECTL_CONTEXT,
                "label",
                "node",
                "--all",
                "nvidia.com/gpu.present=true",
                "--overwrite",
            ],
            check=False,
        )


def delete_test_cluster(cluster_name: str) -> None:
    _cmd(["k3d", "cluster", "delete", cluster_name], check=False, timeout=120)


@pytest.fixture(scope="session")
def k3d_cluster():
    """Isolated cluster for the deployment-lifecycle tests (port-forward
    access, so no loadbalancer mappings).

    The cluster stays available after the focused run. Stop it explicitly with
    ``k3d cluster stop cogniverse-deploy-test`` when the run has finished.
    """
    force_fresh = os.environ.get("E2E_FRESH", "").lower() in ("1", "true", "yes")
    if _cluster_exists(CLUSTER_NAME):
        if not force_fresh:
            pytest.fail(
                f"existing {CLUSTER_NAME} cluster was left intact; inspect it, then "
                "rerun with E2E_FRESH=1 to replace it explicitly"
            )
        delete_test_cluster(CLUSTER_NAME)
    create_test_cluster(CLUSTER_NAME, ports=[], share_host_storage=True)
    yield {
        "cluster_name": CLUSTER_NAME,
        "namespace": NAMESPACE,
        "ports": PORTS,
    }


def _render_release(project_root, helm_values, helm_set_overrides) -> list[dict]:
    """The manifests this deploy is about to apply."""
    command = [
        "helm",
        "template",
        "cogniverse",
        str(project_root / "charts" / "cogniverse"),
    ]
    for values in helm_values:
        command.extend(["-f", str(values)])
    for key, value in helm_set_overrides.items():
        command.extend(["--set", f"{key}={value}"])
    rendered = subprocess.run(command, capture_output=True, text=True, check=False)
    if rendered.returncode != 0:
        raise RuntimeError(
            f"helm template failed while budgeting the release:\n{rendered.stderr}"
        )
    return [d for d in yaml.safe_load_all(rendered.stdout) if d]


def rollout_timeout_minutes(documents: list[dict]) -> int:
    """How long helm must allow, taken from the chart's own rollout deadline.

    values.rocm.yaml paces sidecar startup by making each one wait on the
    previous, and the gate deadline grows with position in that sequence, so
    the tail of the chain declares a progressDeadlineSeconds far beyond any
    single model's load. A helm --timeout shorter than that gives up while the
    chain is still pacing exactly as designed, and reports a bare timeout that
    names no model. Deriving it keeps the two budgets from disagreeing when
    the sequence or the per-link pacing changes.
    """
    deadlines = [
        int(d["spec"]["progressDeadlineSeconds"])
        for d in documents
        if d.get("kind") == "Deployment" and d["spec"].get("progressDeadlineSeconds")
    ]
    if not deadlines:
        raise RuntimeError(
            "no Deployment declared progressDeadlineSeconds, so the deploy "
            "timeout cannot be derived from the chart"
        )
    # Round up to the next minute and add one, so helm outlives the rollout it
    # is waiting on rather than racing it.
    return max(deadlines) // 60 + 1


def _refuse_release_larger_than_the_node(documents: list[dict]) -> None:
    """Fail before deploying a release whose pods cannot all be scheduled.

    Kubernetes places pods on requests, so a release asking for more than the
    node has does not degrade -- it wedges: the pods that fit hold their
    reservations and the rest stay Pending. Compared against MemTotal, the
    physical ceiling, since anything above it cannot schedule regardless of
    how the node is carved up.
    """
    import re as _re

    def _gib(value):
        if not value:
            return 0.0
        text = str(value)
        for suffix, scale in (("Gi", 1.0), ("Mi", 1 / 1024), ("Ki", 1 / 1024**2)):
            if text.endswith(suffix):
                return float(text[: -len(suffix)]) * scale
        return float(text) / 1024**3

    requested = 0.0
    for document in documents:
        if not document or document.get("kind") not in {
            "Deployment",
            "StatefulSet",
            "DaemonSet",
        }:
            continue
        spec = document["spec"]["template"]["spec"]
        replicas = document["spec"].get("replicas", 1) or 1
        running = sum(
            _gib(((c.get("resources") or {}).get("requests") or {}).get("memory"))
            for c in spec.get("containers", [])
        )
        # A pod reserves max(largest init container, sum of run containers):
        # init containers finish before the others start.
        init = max(
            [
                _gib(((c.get("resources") or {}).get("requests") or {}).get("memory"))
                for c in spec.get("initContainers", [])
            ]
            or [0.0]
        )
        requested += max(running, init) * replicas

    meminfo = Path("/proc/meminfo").read_text()
    total_kb = int(_re.search(r"MemTotal:\s+(\d+) kB", meminfo).group(1))
    node_gib = total_kb / 1024**2
    if requested > node_gib:
        raise RuntimeError(
            f"this release requests {requested:.2f}Gi of memory but the node has "
            f"{node_gib:.2f}Gi, so it cannot schedule. Serving the chat models "
            "from Modal returns 44Gi and is how this host fits: rerun with "
            "COGNIVERSE_LLM_SERVING=modal."
        )


def deployment_helm_inputs(
    project_root,
    *,
    extra_set: dict[str, str] | None = None,
) -> dict:
    """Resolve the exact backend, overlays, image tags, and Helm overrides."""
    from cogniverse_cli.config import (
        LLM_SERVING_LOCAL,
        get_device_values_file,
        get_llm_serving_values_file,
    )
    from cogniverse_cli.images import (
        RUNTIME_REPOS_BY_BACKEND,
        detect_torch_backend,
        dev_image_set_values,
        dev_version,
    )

    chart_path = project_root / "charts" / "cogniverse"
    values_file = chart_path / "values.k3s.yaml"
    assert chart_path.exists(), f"Chart not found: {chart_path}"
    assert values_file.exists(), f"Values not found: {values_file}"

    backend = detect_torch_backend()
    device_values_file = get_device_values_file(backend, project_root=project_root)
    image_version = dev_version(project_root)
    helm_values = [values_file]
    if device_values_file:
        helm_values.append(device_values_file)
    serving_values_file = get_llm_serving_values_file(
        os.environ.get("COGNIVERSE_LLM_SERVING", LLM_SERVING_LOCAL),
        project_root=project_root,
    )
    if serving_values_file:
        helm_values.append(serving_values_file)
    helm_set_overrides = {
        "argo-workflows.crds.install": "false",
        "runtime.backend": backend,
        "dashboard.backend": backend,
        "devMode.enabled": "false",
    }
    helm_set_overrides.update(
        # Same overlays helm is about to apply: the tag overrides are emitted
        # per ENABLED sidecar, so computing them from chart defaults while
        # helm enables more (the device overlay turns on code_colbert_pylate)
        # leaves those deployments on the static placeholder tag that was
        # never built — ErrImageNeverPull on a Never-pull cluster.
        dev_image_set_values(
            project_root,
            torch_backend=backend,
            values_files=helm_values,
            version=image_version,
        )
    )
    if extra_set:
        helm_set_overrides.update(extra_set)
    return {
        "backend": backend,
        "image_version": image_version,
        "image_repository": RUNTIME_REPOS_BY_BACKEND[backend],
        "helm_values": helm_values,
        "helm_set_overrides": helm_set_overrides,
    }


def dump_pod_state(namespace: str) -> None:
    """Snapshot cluster state to pytest's captured stdout — runs on
    any helm-install failure so the next teardown doesn't take the
    evidence with it."""
    import sys

    print("\n========== POD STATE ON HELM FAILURE ==========", file=sys.stdout)
    for diag in [
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "pods",
            "-n",
            namespace,
            "-o",
            "wide",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "events",
            "-n",
            namespace,
            "--sort-by=.lastTimestamp",
        ],
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "describe",
            "pods",
            "-n",
            namespace,
        ],
    ]:
        print(f"\n--- {' '.join(diag)} ---", file=sys.stdout)
        sys.stdout.flush()
        subprocess.run(diag, check=False, timeout=60)
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "pods",
            "-n",
            namespace,
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
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "logs",
                "-n",
                namespace,
                pod,
                "--tail=100",
            ],
            check=False,
            timeout=30,
        )
    print("================================================\n", file=sys.stdout)


def deploy_stack(
    cluster_name: str,
    namespace: str,
    *,
    extra_set: dict[str, str] | None = None,
) -> None:
    """Build images from the working tree and Helm-install the full stack.

    ``devMode`` is always off: the deployed pods run the code BAKED INTO the
    freshly built images, never a bind-mounted tree with a stale interpreter.
    """
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    chart_path = project_root / "charts" / "cogniverse"
    deployment_inputs = deployment_helm_inputs(project_root, extra_set=extra_set)
    from cogniverse_cli.images import (
        build_images,
        import_images,
        prune_superseded_images,
        verify_local_images_cover_deploy,
    )

    backend = deployment_inputs["backend"]
    # One deploy-input-derived version for the built tags AND the helm
    # overrides — without the override the chart falls back to its static
    # ``0.1.0-dev`` placeholder and every first-party pod dies with
    # ErrImageNeverPull (pullPolicy=Never can only see the imported, tagged
    # images).
    image_version = deployment_inputs["image_version"]

    # Build from the same overlays helm receives: the enabled set is derived
    # from those values, so building without them misses every sidecar an
    # overlay turns on. verify_local_images_cover_deploy fails here rather
    # than leaving the pod on ErrImageNeverPull.
    built_tags = build_images(
        project_root,
        torch_backend=backend,
        values_files=deployment_inputs["helm_values"],
        version=image_version,
    )
    import_images(cluster_name, built_tags)
    verify_local_images_cover_deploy(
        project_root,
        deployment_inputs["helm_values"],
        built_tags=built_tags,
        version=image_version,
    )

    # Reclaim superseded generations (host + k3d node containerd) like
    # ``cogniverse up`` does — keeps the current build + one prior and drops
    # the rest. Without this, each e2e rebuild leaves ~24GB of stale
    # cogniverse/* tags, and repeated runs fill the host disk until Vespa
    # trips its 80% feed-block and the runtime crash-loops on NO_SPACE.
    try:
        prune_superseded_images(
            image_version, node_container=f"k3d-{cluster_name}-server-0"
        )
    except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
        print(f"Superseded-image prune skipped: {exc}", file=sys.stderr)

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
        install_argo_controller()
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
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "create",
                "namespace",
                namespace,
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
        _rendered = subprocess.run(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "create",
                "secret",
                "generic",
                HF_TOKEN_SECRET,
                "-n",
                namespace,
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
            ["kubectl", "--context", KUBECTL_CONTEXT, "apply", "-f", "-"],
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

    helm_values = deployment_inputs["helm_values"]
    helm_set_overrides = deployment_inputs["helm_set_overrides"]

    # Every sidecar helm is about to enable must be pinned to a tag that was
    # actually built and imported. Anything left on the chart's static
    # placeholder cannot be pulled (pullPolicy=Never) and surfaces minutes
    # later as ErrImageNeverPull on a pod, far from its cause.
    from cogniverse_cli.images import enabled_sidecars

    sidecars = enabled_sidecars(project_root, helm_values)
    unpinned = [
        svc
        for svc in sidecars
        if f"inference.{svc}.image.tag" not in helm_set_overrides
    ]
    assert not unpinned, (
        f"enabled sidecars {unpinned} have no image.tag override; they would "
        f"deploy on the chart's placeholder tag, which was never built. "
        f"Overrides cover: "
        f"{sorted(k for k in helm_set_overrides if k.endswith('.image.tag'))}"
    )

    # Budgeted here rather than while resolving inputs: this is the first
    # point that necessarily has the real chart on disk.
    documents = _render_release(project_root, helm_values, helm_set_overrides)
    _refuse_release_larger_than_the_node(documents)

    try:
        helm_install(
            chart_path,
            helm_values,
            set_values=helm_set_overrides,
            namespace=namespace,
            # Derived from the chart, not fixed: the sidecar pacing chain's
            # tail declares a rollout deadline well past 20m, so a hardcoded
            # 20m gave up mid-pacing and reported a timeout naming no model.
            timeout=f"{rollout_timeout_minutes(documents)}m",
        )
    except RuntimeError:
        dump_pod_state(namespace)
        raise

    wait_for_stack_ready(namespace)


INFERENCE_COMPONENT_PREFIX = "inference-"


# One budget for the whole stack, not one per workload: a per-workload budget
# multiplies by however many workloads the chart happens to render.
STACK_READY_BUDGET_S = 300


def stack_workloads(namespace: str) -> list[str]:
    """Non-inference Deployments and StatefulSets, as ``kind/name`` targets.

    Inference workloads are sequenced GPU model loads gated separately by the
    session fixture's 2400s deployment-available wait; including them here
    turns every deploy that restarts a model into a timeout.
    """
    listing = _cmd(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "deploy,statefulset",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/instance=cogniverse",
            "-o",
            'jsonpath={range .items[*]}{.kind}{"/"}{.metadata.name}{" "}'
            '{.metadata.labels.app\\.kubernetes\\.io/component}{"\\n"}{end}',
        ]
    )
    targets = []
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        target, component = parts
        if component.startswith(INFERENCE_COMPONENT_PREFIX):
            continue
        targets.append(target.lower())
    return sorted(set(targets))


def rollout_wait_args(namespace: str, target: str, *, timeout_s: int) -> list[str]:
    return [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "rollout",
        "status",
        target,
        "-n",
        namespace,
        f"--timeout={timeout_s}s",
    ]


def wait_for_stack_ready(namespace: str) -> None:
    """Wait for each workload's rollout, not for a snapshot of its pods.

    ``kubectl wait`` resolves its pod set once and then watches those exact
    pods, so a rolling update deletes them underneath it and the wait fails
    with "Error from server (NotFound): pods ... not found" -- on precisely
    the deploys that changed an image, which is the case it exists to cover.
    ``rollout status`` tracks the workload, so a replaced pod is the success
    path rather than the failure.
    """
    targets = stack_workloads(namespace)
    deadline = time.monotonic() + STACK_READY_BUDGET_S
    try:
        for target in targets:
            remaining = max(1, int(deadline - time.monotonic()))
            _cmd(
                rollout_wait_args(namespace, target, timeout_s=remaining),
                timeout=remaining + 10,
            )
    except subprocess.CalledProcessError:
        dump_pod_state(namespace)
        raise


@pytest.fixture(scope="session")
def deployed_stack(k3d_cluster):
    """Deploy the full cogniverse stack via Helm into the test cluster."""
    deploy_stack(CLUSTER_NAME, NAMESPACE)

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
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "port-forward",
                svc,
                ports,
                "-n",
                NAMESPACE,
            ],
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
