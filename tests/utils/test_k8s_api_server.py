"""The test-owned Kubernetes API server provisions what its consumers use.

Boots the real k3s container and drives real kubectl round-trips: the
cogniverse namespace, the CronWorkflow CRD (create → patch → read back the
exact suspend value), and a Secret (write → read back the exact token) —
the two object kinds the secrets-sync and cron-suspension suites depend
on. Also pins the failure contract (raise with the container's log tail,
no leaked container) and that a failing second instance cannot take down
a live one.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.utils.k8s_api_server import (
    NAMESPACE,
    _kubectl,
    start_k8s_api_server,
    stop_k8s_api_server,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


def _container_names(prefix: str) -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return sorted(line for line in out.stdout.splitlines() if line)


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    info = start_k8s_api_server(tmp_path_factory.mktemp("k8s-api-server-selftest"))
    yield info
    stop_k8s_api_server(info["container"])
    assert _container_names(info["container"]) == [], (
        "stop_k8s_api_server must remove the container"
    )


def test_server_provisions_namespace_crd_and_object_round_trips(live_server):
    kubeconfig = Path(live_server["kubeconfig"])
    assert f"https://127.0.0.1:{live_server['port']}" in kubeconfig.read_text()

    ns = _kubectl(
        kubeconfig, "get", "namespace", NAMESPACE, "-o", "jsonpath={.metadata.name}"
    )
    assert (ns.returncode, ns.stdout) == (0, NAMESPACE)

    established = _kubectl(
        kubeconfig,
        "get",
        "crd",
        "cronworkflows.argoproj.io",
        "-o",
        'jsonpath={.status.conditions[?(@.type=="Established")].status}',
    )
    assert (established.returncode, established.stdout) == (0, "True")

    cron_name = f"selftest-cron-{uuid.uuid4().hex[:8]}"
    manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "CronWorkflow",
        "metadata": {"name": cron_name, "namespace": NAMESPACE},
        "spec": {"schedule": "0 0 31 2 *", "suspend": False},
    }
    applied = _kubectl(
        kubeconfig, "apply", "-n", NAMESPACE, "-f", "-", input_text=json.dumps(manifest)
    )
    assert applied.returncode == 0, applied.stderr
    before = _kubectl(
        kubeconfig,
        "get",
        "cronworkflow",
        cron_name,
        "-n",
        NAMESPACE,
        "-o",
        "jsonpath={.spec.suspend}",
    )
    assert (before.returncode, before.stdout) == (0, "false")
    patched = _kubectl(
        kubeconfig,
        "patch",
        "cronworkflow",
        cron_name,
        "-n",
        NAMESPACE,
        "--type",
        "merge",
        "-p",
        '{"spec":{"suspend":true}}',
    )
    assert patched.returncode == 0, patched.stderr
    after = _kubectl(
        kubeconfig,
        "get",
        "cronworkflow",
        cron_name,
        "-n",
        NAMESPACE,
        "-o",
        "jsonpath={.spec.suspend}",
    )
    assert (after.returncode, after.stdout) == (0, "true")

    token = f"selftest:{uuid.uuid4().hex}"
    created = _kubectl(
        kubeconfig,
        "create",
        "secret",
        "generic",
        "selftest-secret",
        "-n",
        NAMESPACE,
        f"--from-literal=telegram-bot-token={token}",
    )
    assert created.returncode == 0, created.stderr
    stored = _kubectl(
        kubeconfig,
        "get",
        "secret",
        "selftest-secret",
        "-n",
        NAMESPACE,
        "-o",
        "jsonpath={.data.telegram-bot-token}",
    )
    assert stored.returncode == 0, stored.stderr
    assert base64.b64decode(stored.stdout).decode() == token


def test_provisioning_timeout_raises_with_context_and_removes_container(tmp_path):
    """An instance that cannot become ready must raise, not half-provision.

    The zero deadline forces the kubeconfig wait to expire immediately after
    ``docker run``; the raised error must name the container, carry the
    timeout as its cause, and the failed container must be removed —
    the container set for this process is identical before and after.
    """
    prefix = f"cogniverse-test-k8s-{os.getpid()}"
    before = _container_names(prefix)
    with pytest.raises(
        RuntimeError,
        match=rf"{prefix}-\d+ failed to provision",
    ) as excinfo:
        start_k8s_api_server(tmp_path, timeout_s=0.0)
    assert isinstance(excinfo.value.__cause__, TimeoutError)
    assert "never produced" in str(excinfo.value.__cause__)
    assert _container_names(prefix) == before


def test_failed_second_instance_leaves_a_live_one_untouched(live_server, tmp_path):
    """Instances are port-qualified: a second start in the same process must
    neither pre-clean nor failure-clean any other instance's container.

    The container set for this process is compared before and after, rather
    than to this module's own instance alone — a full sweep also holds the
    session-scoped ``ephemeral_k8s_cluster`` under the same pid prefix, and
    that one must survive untouched too.
    """
    prefix = f"cogniverse-test-k8s-{os.getpid()}"
    before = _container_names(prefix)
    assert live_server["container"] in before

    with pytest.raises(RuntimeError, match="failed to provision"):
        start_k8s_api_server(tmp_path, timeout_s=0.0)

    assert _container_names(prefix) == before, (
        "a failed start must add and remove exactly its own container"
    )
    ready = _kubectl(Path(live_server["kubeconfig"]), "get", "--raw", "/readyz")
    assert (ready.returncode, ready.stdout) == (0, "ok")
