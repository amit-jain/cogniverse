"""GPU model startup pacing rendered by the Helm chart."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


MODAL_LLM_VALUES = ("-f", str(CHART_PATH / "values.modal-llm.yaml"))
"""The serving overlay this host deploys: both chat models move to Modal."""


def _render(*extra: str, rocm: bool = True) -> list[dict]:
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    if rocm:
        command.extend(["-f", str(CHART_PATH / "values.rocm.yaml")])
    command.extend(extra)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _inference_deployments(*extra: str, rocm: bool = True) -> dict[str, dict]:
    deployments = {}
    for document in _render(*extra, rocm=rocm):
        if document.get("kind") != "Deployment":
            continue
        component = document["metadata"]["labels"].get(
            "app.kubernetes.io/component", ""
        )
        if component.startswith("inference-"):
            deployments[component.removeprefix("inference-")] = document
    return deployments


def _gate(deployment: dict) -> dict | None:
    inits = deployment["spec"]["template"]["spec"].get("initContainers", [])
    for container in inits:
        if container["name"] == "startup-gate":
            return container
    return None


def _gate_env(deployment: dict) -> dict[str, str]:
    gate = _gate(deployment)
    assert gate is not None
    return {entry["name"]: entry["value"] for entry in gate["env"]}


def test_rocm_chain_gates_each_model_on_its_predecessor():
    deployments = _inference_deployments(*MODAL_LLM_VALUES)

    chain = {
        name: (
            _gate_env(deployment)["GATE_URL"],
            _gate_env(deployment)["GATE_DEADLINE_SECONDS"],
        )
        for name, deployment in deployments.items()
        if _gate(deployment) is not None
    }

    # Both chat models are served from Modal, so the sequence re-forms behind
    # the encoders and each surviving position keeps its own budget.
    assert chain == {
        "denseon": ("http://cogniverse-vllm-asr:8000/health", "2400"),
        "colbert_pylate": ("http://cogniverse-denseon:8000/health", "3000"),
        "code_colbert_pylate": ("http://cogniverse-colbert-pylate:8000/health", "3600"),
    }


def test_chain_head_and_non_sequenced_services_start_immediately():
    deployments = _inference_deployments(*MODAL_LLM_VALUES)

    ungated = {name for name, d in deployments.items() if _gate(d) is None}

    # vllm_asr is ungated because its predecessor (the student) is served from
    # Modal: a disabled predecessor releases its successor rather than
    # stranding it behind a health check that can never answer.
    assert ungated == {"vllm_colpali", "vllm_asr", "gliner"}


def test_gated_deployments_extend_the_rollout_progress_deadline():
    deployments = _inference_deployments(*MODAL_LLM_VALUES)

    deadlines = {
        name: deployment["spec"].get("progressDeadlineSeconds")
        for name, deployment in deployments.items()
    }

    assert deadlines == {
        "gliner": None,
        "vllm_colpali": None,
        "vllm_asr": None,
        "denseon": 3300,
        "colbert_pylate": 3900,
        "code_colbert_pylate": 4500,
    }


def test_weight_download_runs_before_the_gate_so_only_gpu_load_serializes():
    deployments = _inference_deployments(
        *MODAL_LLM_VALUES,
        "--set",
        "hfCache.enabled=false",
        "--set",
        "hfCache.persistence.enabled=true",
    )

    gated = deployments["denseon"]["spec"]["template"]["spec"]
    chain_head = deployments["vllm_colpali"]["spec"]["template"]["spec"]

    assert [c["name"] for c in gated["initContainers"]] == [
        "model-warm",
        "startup-gate",
    ]
    assert [c["name"] for c in chain_head["initContainers"]] == ["model-warm"]


def test_gating_leaves_the_readiness_contract_untouched():
    deployments = _inference_deployments(*MODAL_LLM_VALUES)

    probes = {
        name: (
            deployment["spec"]["template"]["spec"]["containers"][0]["readinessProbe"][
                "initialDelaySeconds"
            ],
            deployment["spec"]["template"]["spec"]["containers"][0]["livenessProbe"][
                "initialDelaySeconds"
            ],
        )
        for name, deployment in deployments.items()
    }

    assert probes == {
        "vllm_colpali": (0, 600),
        "vllm_asr": (0, 600),
        "denseon": (0, 90),
        "colbert_pylate": (0, 120),
        "code_colbert_pylate": (0, 60),
        "gliner": (0, 60),
    }


def test_disabled_predecessor_releases_its_successor_instead_of_stranding_it():
    deployments = _inference_deployments(
        "--set", "inference.vllm_colpali.enabled=false"
    )

    gated = {
        name: _gate_env(d)["GATE_URL"]
        for name, d in deployments.items()
        if _gate(d) is not None
    }

    assert "vllm_colpali" not in deployments
    assert gated == {
        "vllm_asr": "http://cogniverse-vllm-llm-student:8000/health",
        "denseon": "http://cogniverse-vllm-asr:8000/health",
        "colbert_pylate": "http://cogniverse-denseon:8000/health",
        "code_colbert_pylate": "http://cogniverse-colbert-pylate:8000/health",
    }
    assert _gate(deployments["vllm_llm_student"]) is None
    assert (
        deployments["vllm_llm_student"]["spec"].get("progressDeadlineSeconds") is None
    )


def test_default_values_pace_nothing():
    deployments = _inference_deployments(rocm=False)

    assert deployments != {}
    assert all(_gate(d) is None for d in deployments.values())
    assert all(
        d["spec"].get("progressDeadlineSeconds") is None for d in deployments.values()
    )


def test_gate_polls_until_the_predecessor_answers_then_stops_waiting():
    gate = _gate(_inference_deployments()["vllm_llm_student"])

    assert gate["command"] == ["sh", "-c"]
    script = gate["args"][0]
    assert "urllib.request.urlopen(url, timeout=5)" in script
    assert "if response.status == 200:" in script
    assert "if time.monotonic() >= deadline:" in script
    assert "time.sleep(5)" in script
