"""The live-cluster offline-startup check reads Deployments and pods with kubectl."""

from __future__ import annotations

import copy
import json
import subprocess

import pytest

from scripts.check_offline_model_pods import check, main

_SELECTOR = {
    "app.kubernetes.io/component": "inference-denseon",
    "app.kubernetes.io/name": "cogniverse",
}


def _deployment(name: str, container: dict, component: str) -> dict:
    return {
        "metadata": {"name": name},
        "spec": {
            "selector": {
                "matchLabels": {**_SELECTOR, "app.kubernetes.io/component": component}
            },
            "template": {
                "metadata": {"labels": {"app.kubernetes.io/component": component}},
                "spec": {"containers": [container]},
            },
        },
    }


_OFFLINE_SERVER = {
    "name": "vllm-embed",
    "command": ["vllm"],
    "args": ["serve", "lightonai/DenseOn"],
    "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}],
}
_PIP_SERVER = {
    "name": "vllm-transcription",
    "command": ["sh", "-c"],
    "args": ["pip install soundfile librosa || exit 1\nexec vllm serve m"],
    "env": [],
}


def _pod(name: str, *, ready: str = "True", restarts: int = 0) -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "conditions": [{"type": "Ready", "status": ready}],
            "containerStatuses": [{"name": "server", "restartCount": restarts}],
        },
    }


class FakeKubectl:
    """Answers ``kubectl get`` from fixed objects and records each call."""

    def __init__(self, deployments: list[dict], pods: dict[str, list[dict]]) -> None:
        self.deployments = deployments
        self.pods = pods
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **_: object) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        args = argv[argv.index("get") + 1 : -2]
        if args == ["deployments"]:
            body: dict = {"items": self.deployments}
        elif args[0] == "deployment":
            body = next(d for d in self.deployments if d["metadata"]["name"] == args[1])
        else:
            selector = args[2]
            component = dict(kv.split("=") for kv in selector.split(","))[
                "app.kubernetes.io/component"
            ]
            body = {"items": self.pods.get(component, [])}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(body), stderr="")


def _cluster() -> FakeKubectl:
    return FakeKubectl(
        [
            _deployment("cogniverse-denseon", _OFFLINE_SERVER, "inference-denseon"),
            _deployment("cogniverse-vllm-asr", _PIP_SERVER, "inference-vllm_asr"),
            _deployment("cogniverse-runtime", _PIP_SERVER, "runtime"),
        ],
        {
            "inference-denseon": [_pod("denseon-1")],
            "inference-vllm_asr": [_pod("asr-1", ready="False", restarts=7)],
        },
    )


def test_every_inference_deployment_is_checked_and_offenders_are_named() -> None:
    kubectl = _cluster()
    assert check([], context="k3d-x", namespace="ns", run=kubectl) == {
        "cogniverse-denseon": [],
        "cogniverse-vllm-asr": [
            "container vllm-transcription installs packages at start",
            "container vllm-transcription starts through a shell script",
            "container vllm-transcription does not set HF_HUB_OFFLINE=1",
            "pod asr-1 is not Ready",
            "pod asr-1 container server restarted 7 times",
        ],
    }
    assert kubectl.calls[1] == [
        "kubectl",
        "--context",
        "k3d-x",
        "-n",
        "ns",
        "get",
        "pods",
        "-l",
        "app.kubernetes.io/component=inference-denseon,app.kubernetes.io/name=cogniverse",
        "-o",
        "json",
    ]


def test_a_named_deployment_with_no_pods_fails() -> None:
    kubectl = _cluster()
    kubectl.pods.pop("inference-denseon")
    assert check(
        ["cogniverse-denseon"], context="k3d-x", namespace="ns", run=kubectl
    ) == {"cogniverse-denseon": ["no pods"]}


def test_main_prints_each_deployment_and_exits_nonzero_on_a_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--context", "k3d-x", "--namespace", "ns"], run=_cluster()) == 1
    assert capsys.readouterr().out == (
        "cogniverse-denseon: ok\n"
        "cogniverse-vllm-asr: FAIL\n"
        "  - container vllm-transcription installs packages at start\n"
        "  - container vllm-transcription starts through a shell script\n"
        "  - container vllm-transcription does not set HF_HUB_OFFLINE=1\n"
        "  - pod asr-1 is not Ready\n"
        "  - pod asr-1 container server restarted 7 times\n"
    )


def test_main_exits_zero_when_every_pod_starts_offline() -> None:
    kubectl = _cluster()
    asr = copy.deepcopy(_OFFLINE_SERVER)
    asr["name"] = "vllm-transcription"
    kubectl.deployments[1] = _deployment(
        "cogniverse-vllm-asr", asr, "inference-vllm_asr"
    )
    kubectl.pods["inference-vllm_asr"] = [_pod("asr-2")]
    assert main([], run=kubectl) == 0


def test_kubectl_failure_raises_with_its_stderr() -> None:
    def failing(argv: list[str], **_: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no context x")

    with pytest.raises(RuntimeError) as excinfo:
        check([], context="x", namespace="ns", run=failing)
    assert str(excinfo.value) == "kubectl get deployments failed: no context x"


def test_a_namespace_without_inference_deployments_is_refused() -> None:
    kubectl = FakeKubectl([], {})
    with pytest.raises(RuntimeError) as excinfo:
        check([], context="x", namespace="ns", run=kubectl)
    assert str(excinfo.value) == "no inference Deployments in namespace ns"
