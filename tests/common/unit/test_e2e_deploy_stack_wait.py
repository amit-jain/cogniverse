"""The stack-ready wait after helm install must select only pods that can
become ready within its budget and must fail the deploy when they do not.

Inference pods are sequenced GPU model loads gated separately by the
session fixture's 2400s deployment-available wait; selecting them here
turns every deploy that restarts a model into a timeout. Which inference
pods exist is a cluster fact (helm --set overrides disable services the
values files still enable), so the exclusion set is read from the
deployed deployments' component labels, never derived from config."""

import subprocess
from pathlib import Path

import pytest

import tests.e2e.deployment.conftest as deploy_conftest

CHART_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "charts"
    / "cogniverse"
    / "templates"
    / "all-resources.yaml"
)


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_inference_component_prefix_matches_the_chart_label_form():
    assert deploy_conftest.INFERENCE_COMPONENT_PREFIX == "inference-"
    template = CHART_TEMPLATE.read_text()
    assert "app.kubernetes.io/component: inference-{{ $name }}" in template, (
        "chart no longer labels inference pods as inference-<name>"
    )


def test_deployed_inference_components_reads_deployment_labels(monkeypatch):
    calls: list[list[str]] = []

    def fake_cmd(args, *, timeout=120, check=True):
        calls.append(list(args))
        return _completed(
            "dashboard\ninference-vllm_asr\ningestor\ninference-gliner\n"
            "runtime\ninference-vllm_asr\n\n"
        )

    monkeypatch.setattr(deploy_conftest, "_cmd", fake_cmd)
    assert deploy_conftest.deployed_inference_components("cogniverse") == [
        "inference-gliner",
        "inference-vllm_asr",
    ]
    assert calls == [
        [
            "kubectl",
            "get",
            "deploy",
            "-n",
            "cogniverse",
            "-l",
            "app.kubernetes.io/instance=cogniverse",
            "-o",
            "jsonpath={range .items[*]}"
            '{.metadata.labels.app\\.kubernetes\\.io/component}{"\\n"}{end}',
        ]
    ]


def test_deployed_inference_components_raises_when_kubectl_fails(monkeypatch):
    def failing_cmd(args, *, timeout=120, check=True):
        raise subprocess.CalledProcessError(1, args, stderr="connection refused")

    monkeypatch.setattr(deploy_conftest, "_cmd", failing_cmd)
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        deploy_conftest.deployed_inference_components("cogniverse")
    assert excinfo.value.stderr == "connection refused"


def test_ready_pod_wait_excludes_completed_hook_pods_and_inference_pods():
    assert deploy_conftest.ready_pod_wait_args(
        "cogniverse", inference_components=["inference-gliner", "inference-vllm_asr"]
    ) == [
        "kubectl",
        "wait",
        "--for=condition=ready",
        "pod",
        "-l",
        "app.kubernetes.io/instance=cogniverse,"
        "app.kubernetes.io/component notin (inference-gliner,inference-vllm_asr)",
        "--field-selector=status.phase!=Succeeded",
        "-n",
        "cogniverse",
        "--timeout=300s",
    ]


def test_ready_pod_wait_without_inference_pods_keeps_the_plain_selector():
    assert deploy_conftest.ready_pod_wait_args(
        "cogniverse", inference_components=[]
    ) == [
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


def test_wait_for_stack_ready_excludes_the_deployed_inference_pods(monkeypatch):
    calls: list[tuple[list[str], int]] = []

    def fake_cmd(args, *, timeout=120, check=True):
        calls.append((list(args), timeout))
        if args[1] == "get":
            return _completed(
                "runtime\ninference-vllm_llm_student\ninference-denseon\n"
            )
        return _completed("")

    monkeypatch.setattr(deploy_conftest, "_cmd", fake_cmd)
    deploy_conftest.wait_for_stack_ready("cogniverse")
    assert [args for args, _ in calls][1] == deploy_conftest.ready_pod_wait_args(
        "cogniverse",
        inference_components=["inference-denseon", "inference-vllm_llm_student"],
    )
    assert [timeout for _, timeout in calls] == [120, 310]


def test_wait_for_stack_ready_raises_when_pods_never_become_ready(monkeypatch):
    dumps: list[str] = []

    def cmd(args, *, timeout=120, check=True):
        assert check is True
        if args[1] == "get":
            return _completed("runtime\ninference-gliner\n")
        raise subprocess.CalledProcessError(
            1, args, stderr="timed out waiting for the condition on pods/x"
        )

    monkeypatch.setattr(deploy_conftest, "_cmd", cmd)
    monkeypatch.setattr(
        deploy_conftest, "dump_pod_state", lambda namespace: dumps.append(namespace)
    )
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        deploy_conftest.wait_for_stack_ready("cogniverse")
    assert excinfo.value.stderr == "timed out waiting for the condition on pods/x"
    assert dumps == ["cogniverse"]
