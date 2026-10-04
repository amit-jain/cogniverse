"""A real Argo API server over a fixture-owned Kubernetes datastore.

No workflow controller runs: Workflows are stored and served exactly as Argo
serves them, and a test sets the phases it needs with ``set_workflow_status``.
The optimization and job WorkflowTemplates the runtime references are applied.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import requests

from tests.utils.k8s_api_server import (
    CRONWORKFLOW_CRD,
    _kubectl,
    start_k8s_api_server,
    stop_k8s_api_server,
)


@contextmanager
def argo_api_server(workdir: Path) -> Iterator[dict]:
    """Yields ``{"url", "kubeconfig"}`` for an Argo API in namespace
    ``cogniverse``."""
    cluster = start_k8s_api_server(workdir)
    kubeconfig = Path(cluster["kubeconfig"])
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    name = f"cogniverse-test-argo-{os.getpid()}-{port}"
    try:
        for plural, kind, namespaced in (
            ("workflows", "Workflow", True),
            ("workflowtemplates", "WorkflowTemplate", True),
            ("clusterworkflowtemplates", "ClusterWorkflowTemplate", False),
        ):
            crd = copy.deepcopy(CRONWORKFLOW_CRD)
            crd["metadata"]["name"] = f"{plural}.argoproj.io"
            crd["spec"]["scope"] = "Namespaced" if namespaced else "Cluster"
            crd["spec"]["names"] = {
                "kind": kind,
                "listKind": f"{kind}List",
                "plural": plural,
                "singular": plural[:-1],
            }
            applied = _kubectl(
                kubeconfig, "apply", "-f", "-", input_text=json.dumps(crd)
            )
            assert applied.returncode == 0, applied.stderr
            ready = _kubectl(
                kubeconfig,
                "wait",
                "--for=condition=Established",
                f"crd/{plural}.argoproj.io",
                "--timeout=60s",
                timeout=70,
            )
            assert ready.returncode == 0, ready.stderr
        # argo-server reads the controller's ConfigMap on startup and exits
        # fatally without it; the defaults are what this fixture needs.
        applied = _kubectl(
            kubeconfig,
            "apply",
            "-f",
            "-",
            input_text=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": "workflow-controller-configmap",
                        "namespace": "cogniverse",
                    },
                }
            ),
        )
        assert applied.returncode == 0, applied.stderr
        for template_name, entrypoint in (
            ("cogniverse-job-runner", "job"),
            ("cogniverse-optimization-runner", "run-optimizer"),
        ):
            template = {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "WorkflowTemplate",
                "metadata": {"name": template_name, "namespace": "cogniverse"},
                "spec": {
                    "entrypoint": entrypoint,
                    "templates": [
                        {
                            "name": entrypoint,
                            "container": {
                                "image": "alpine:3.20",
                                "command": ["true"],
                            },
                        }
                    ],
                },
            }
            applied = _kubectl(
                kubeconfig, "apply", "-f", "-", input_text=json.dumps(template)
            )
            assert applied.returncode == 0, applied.stderr
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--label",
                f"cogniverse-test-owner-pid={os.getpid()}",
                "--network",
                "host",
                "--user",
                "0:0",
                "-v",
                f"{kubeconfig}:/kubeconfig:ro",
                "quay.io/argoproj/argocli:v3.7.3",
                "server",
                "--kubeconfig",
                "/kubeconfig",
                "--namespace",
                "cogniverse",
                "--namespaced",
                "--auth-mode",
                "server",
                "--secure=false",
                "--port",
                str(port),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{url}/api/v1/info", timeout=2)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.2)
        else:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True
            )
            pytest.fail(
                f"Argo server did not become ready: {logs.stdout}\n{logs.stderr}"
            )
        yield {"url": url, "kubeconfig": kubeconfig}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        stop_k8s_api_server(cluster["container"])


def apply_manifest(kubeconfig, manifest) -> None:
    applied = _kubectl(kubeconfig, "apply", "-f", "-", input_text=json.dumps(manifest))
    assert applied.returncode == 0, applied.stderr


def set_workflow_status(kubeconfig, name: str, status: dict) -> None:
    patched = _kubectl(
        kubeconfig,
        "patch",
        "workflow.argoproj.io",
        name,
        "-n",
        "cogniverse",
        "--type=merge",
        "-p",
        json.dumps({"status": status}),
    )
    assert patched.returncode == 0, patched.stderr
