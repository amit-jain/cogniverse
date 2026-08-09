"""Test-owned ephemeral Kubernetes API server.

Boots a k3s server in Docker with the agent disabled — an API server plus
datastore only, no kubelet/scheduler — which is everything Secret and
CustomResource round-trips need. The container publishes the API on a free
localhost port; :func:`start_k8s_api_server` extracts the generated
kubeconfig, rewrites it for the published port, creates the ``cogniverse``
namespace, and installs the minimal Argo CronWorkflow CRD (the same
preserve-unknown-fields flavor Argo ships) so CLI/chart tests can point
``kubectl`` at a cluster the session owns instead of mutating a
developer's k3d cluster.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from tests.utils.vllm_sidecar import OWNER_LABEL, _free_port

K3S_IMAGE = "rancher/k3s:v1.30.6-k3s1"
NAMESPACE = "cogniverse"
_IN_CONTAINER_KUBECONFIG = "/etc/rancher/k3s/k3s.yaml"

# Argo's minimal CRD flavor: the API server validates only that ``spec``
# is an object and preserves unknown fields, exactly enough for kubectl
# get/apply/patch round-trips on CronWorkflow objects.
CRONWORKFLOW_CRD = {
    "apiVersion": "apiextensions.k8s.io/v1",
    "kind": "CustomResourceDefinition",
    "metadata": {"name": "cronworkflows.argoproj.io"},
    "spec": {
        "group": "argoproj.io",
        "names": {
            "kind": "CronWorkflow",
            "listKind": "CronWorkflowList",
            "plural": "cronworkflows",
            "singular": "cronworkflow",
            "shortNames": ["cwf", "cronwf"],
        },
        "scope": "Namespaced",
        "versions": [
            {
                "name": "v1alpha1",
                "served": True,
                "storage": True,
                "schema": {
                    "openAPIV3Schema": {
                        "type": "object",
                        "properties": {
                            "spec": {
                                "type": "object",
                                "x-kubernetes-preserve-unknown-fields": True,
                            },
                            "status": {
                                "type": "object",
                                "x-kubernetes-preserve-unknown-fields": True,
                            },
                        },
                        "required": ["spec"],
                    }
                },
            }
        ],
    },
}


def _kubectl(
    kubeconfig: Path,
    *args: str,
    input_text: str | None = None,
    timeout: float = 30,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _await_kubeconfig(container: str, deadline: float) -> str:
    """Poll the container for its generated kubeconfig until the deadline."""
    last_err = ""
    while time.monotonic() < deadline:
        out = subprocess.run(
            ["docker", "exec", container, "cat", _IN_CONTAINER_KUBECONFIG],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
        last_err = (out.stderr or "").strip()
        time.sleep(1)
    raise TimeoutError(
        f"k3s container {container} never produced {_IN_CONTAINER_KUBECONFIG} "
        f"(last error: {last_err})"
    )


def _await_ready(kubeconfig: Path, container: str, deadline: float) -> None:
    last = ""
    while time.monotonic() < deadline:
        probe = _kubectl(kubeconfig, "get", "--raw", "/readyz", timeout=10)
        if probe.returncode == 0:
            return
        last = (probe.stderr or probe.stdout or "").strip()
        time.sleep(1)
    raise TimeoutError(
        f"k3s API server in {container} never answered /readyz (last: {last})"
    )


def start_k8s_api_server(
    work_dir: Path, *, timeout_s: float = 180.0
) -> dict[str, str | int]:
    """Boot an agentless k3s API server and prepare it for the test suites.

    Returns ``{"container": name, "kubeconfig": path, "port": port}``. The
    kubeconfig is written under ``work_dir``. Raises with the container's
    recent log output when any provisioning step fails; the container is
    removed on failure so a broken boot never leaks.
    """
    for binary in ("docker", "kubectl"):
        if shutil.which(binary) is None:
            raise RuntimeError(
                f"{binary} is required to run the test-owned Kubernetes API "
                "server; install it on this host"
            )
    port = _free_port()
    # Port-qualified name: concurrent instances in one process (the session
    # fixture plus a direct call in the same sweep) never collide, so the
    # stale-name pre-clean below can only ever remove a leftover of itself.
    container = f"cogniverse-test-k8s-{os.getpid()}-{port}"
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--privileged",
            "--name",
            container,
            "--label",
            f"{OWNER_LABEL}={os.getpid()}",
            "-p",
            f"127.0.0.1:{port}:6443",
            K3S_IMAGE,
            "server",
            "--disable-agent",
            "--disable=traefik,servicelb,metrics-server",
            "--disable-cloud-controller",
            "--disable-network-policy",
            "--disable-helm-controller",
            "--write-kubeconfig-mode=644",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    deadline = time.monotonic() + timeout_s
    kubeconfig = work_dir / f"{container}-kubeconfig.yaml"
    try:
        raw = _await_kubeconfig(container, deadline)
        kubeconfig.write_text(
            raw.replace("https://127.0.0.1:6443", f"https://127.0.0.1:{port}")
        )
        _await_ready(kubeconfig, container, deadline)
        ns = _kubectl(kubeconfig, "create", "namespace", NAMESPACE)
        if ns.returncode != 0:
            raise RuntimeError(f"namespace creation failed: {ns.stderr.strip()}")
        crd = _kubectl(
            kubeconfig,
            "apply",
            "-f",
            "-",
            input_text=json.dumps(CRONWORKFLOW_CRD),
        )
        if crd.returncode != 0:
            raise RuntimeError(f"CronWorkflow CRD apply failed: {crd.stderr.strip()}")
        established = _kubectl(
            kubeconfig,
            "wait",
            "--for=condition=Established",
            "crd/cronworkflows.argoproj.io",
            "--timeout=60s",
            timeout=70,
        )
        if established.returncode != 0:
            raise RuntimeError(
                f"CronWorkflow CRD never established: {established.stderr.strip()}"
            )
    except Exception as exc:
        logs = subprocess.run(
            ["docker", "logs", "--tail", "40", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        stop_k8s_api_server(container)
        raise RuntimeError(
            f"test-owned k8s API server {container} failed to provision: {exc}\n"
            f"container log tail:\n{(logs.stderr or logs.stdout)[-2000:]}"
        ) from exc
    return {"container": container, "kubeconfig": str(kubeconfig), "port": port}


def stop_k8s_api_server(container: str) -> None:
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
