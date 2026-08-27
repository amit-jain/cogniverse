"""The stack-ready wait after helm install must select only pods that can
become ready and must fail the deploy when they do not."""

import subprocess

import pytest

import tests.e2e.deployment.conftest as deploy_conftest


def test_ready_pod_wait_excludes_completed_hook_pods():
    assert deploy_conftest.ready_pod_wait_args("cogniverse") == [
        "kubectl",
        "wait",
        "--for=condition=ready",
        "pod",
        "-l",
        "app.kubernetes.io/instance=cogniverse",
        "--field-selector=status.phase!=Succeeded",
        "-n",
        "cogniverse",
        "--timeout=300s",
    ]


def test_wait_for_stack_ready_raises_when_pods_never_become_ready(monkeypatch):
    calls: list[tuple[list[str], int]] = []
    dumps: list[str] = []

    def failing_cmd(args, *, timeout=120, check=True):
        calls.append((list(args), timeout))
        assert check is True
        raise subprocess.CalledProcessError(
            1, args, stderr="timed out waiting for the condition on pods/x"
        )

    monkeypatch.setattr(deploy_conftest, "_cmd", failing_cmd)
    monkeypatch.setattr(
        deploy_conftest, "dump_pod_state", lambda namespace: dumps.append(namespace)
    )
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        deploy_conftest.wait_for_stack_ready("cogniverse")
    assert excinfo.value.stderr == "timed out waiting for the condition on pods/x"
    assert calls == [(deploy_conftest.ready_pod_wait_args("cogniverse"), 310)]
    assert dumps == ["cogniverse"]
