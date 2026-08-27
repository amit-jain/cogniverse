"""The stack-ready wait after helm install must select only pods that can
become ready within its budget and must fail the deploy when they do not.

Inference pods are sequenced GPU model loads gated separately by the
session fixture's 2400s deployment-available wait; selecting them here
turns every deploy that restarts a model into a timeout."""

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


def test_inference_component_labels_match_the_chart_label_form():
    assert deploy_conftest.inference_component_labels(["vllm_asr", "gliner"]) == [
        "inference-vllm_asr",
        "inference-gliner",
    ]
    template = CHART_TEMPLATE.read_text()
    assert "app.kubernetes.io/component: inference-{{ $name }}" in template, (
        "chart no longer labels inference pods as inference-<name>"
    )


def test_ready_pod_wait_excludes_completed_hook_pods_and_inference_pods():
    assert deploy_conftest.ready_pod_wait_args(
        "cogniverse", inference_components=["inference-vllm_asr", "inference-gliner"]
    ) == [
        "kubectl",
        "wait",
        "--for=condition=ready",
        "pod",
        "-l",
        "app.kubernetes.io/instance=cogniverse,"
        "app.kubernetes.io/component notin (inference-vllm_asr,inference-gliner)",
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
    components = ["inference-vllm_llm_student"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        deploy_conftest.wait_for_stack_ready(
            "cogniverse", inference_components=components
        )
    assert excinfo.value.stderr == "timed out waiting for the condition on pods/x"
    assert calls == [
        (
            deploy_conftest.ready_pod_wait_args(
                "cogniverse", inference_components=components
            ),
            310,
        )
    ]
    assert dumps == ["cogniverse"]
