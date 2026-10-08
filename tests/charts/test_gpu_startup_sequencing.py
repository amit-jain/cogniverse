"""GPU model startup pacing rendered by the Helm chart."""

import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

    # Both chat models are served from Modal, so the sequence re-forms over
    # the pods that run: vllm_asr waits on vllm_colpali, the nearest earlier
    # entry with a pod, and each position keeps its own budget.
    assert chain == {
        "vllm_asr": ("http://cogniverse-vllm-colpali:8000/health", "1800"),
        "denseon": ("http://cogniverse-vllm-asr:8000/health", "2400"),
        "colbert_pylate": ("http://cogniverse-denseon:8000/health", "3000"),
        "code_colbert_pylate": ("http://cogniverse-colbert-pylate:8000/health", "3600"),
    }


def test_chain_head_and_non_sequenced_services_start_immediately():
    deployments = _inference_deployments(*MODAL_LLM_VALUES)

    ungated = {name for name, d in deployments.items() if _gate(d) is None}

    # vllm_colpali is the first sequenced entry with a pod here (the teacher
    # is served from Modal); gliner is not in the sequence.
    assert ungated == {"vllm_colpali", "gliner"}


def test_gated_deployments_extend_the_rollout_progress_deadline():
    deployments = _inference_deployments(*MODAL_LLM_VALUES)

    deadlines = {
        name: deployment["spec"].get("progressDeadlineSeconds")
        for name, deployment in deployments.items()
    }

    assert deadlines == {
        "gliner": None,
        "vllm_colpali": None,
        "vllm_asr": 2700,
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


def test_disabled_entry_mid_sequence_is_skipped_not_ungating_its_successor():
    """A disabled entry has no Service to wait on, so its successor waits on
    the nearest earlier entry that runs a pod instead of starting at once
    and reserving GPU memory alongside it."""
    deployments = _inference_deployments(
        "--set",
        "inference.vllm_colpali.enabled=false",
        "--set",
        "config.defaultProfiles.video=",
    )

    gated = {
        name: _gate_env(d)["GATE_URL"]
        for name, d in deployments.items()
        if _gate(d) is not None
    }

    assert "vllm_colpali" not in deployments
    assert gated == {
        "vllm_llm_student": "http://cogniverse-vllm-llm-teacher:8000/health",
        "vllm_asr": "http://cogniverse-vllm-llm-student:8000/health",
        "denseon": "http://cogniverse-vllm-asr:8000/health",
        "colbert_pylate": "http://cogniverse-denseon:8000/health",
        "code_colbert_pylate": "http://cogniverse-colbert-pylate:8000/health",
    }
    assert _gate(deployments["vllm_llm_teacher"]) is None


def test_consecutive_disabled_entries_are_all_skipped():
    deployments = _inference_deployments(
        *MODAL_LLM_VALUES,
        "--set",
        "inference.vllm_asr.enabled=false",
        "--set",
        "inference.denseon.enabled=false",
        "--set",
        "config.defaultProfiles.video=",
    )

    gated = {
        name: _gate_env(d)["GATE_URL"]
        for name, d in deployments.items()
        if _gate(d) is not None
    }

    assert gated == {
        "colbert_pylate": "http://cogniverse-vllm-colpali:8000/health",
        "code_colbert_pylate": "http://cogniverse-colbert-pylate:8000/health",
    }


def test_default_values_pace_nothing():
    deployments = _inference_deployments(rocm=False)

    assert deployments != {}
    assert all(_gate(d) is None for d in deployments.values())
    assert all(
        d["spec"].get("progressDeadlineSeconds") is None for d in deployments.values()
    )


CURL_IMAGE = "curlimages/curl:8.14.1"
"""The small image the chart's Jobs already run curl from."""

PROD_SECRETS = (
    "--set",
    "minio.rootPassword=overlay-secret",
    "--set",
    "openshell.server.sshHandshakeSecret=overlay-secret",
    "--set",
    "phoenix.postgres.auth.password=overlay-secret",
    "--set",
    "redis.auth.password=overlay-secret",
)

STACKS = {
    "default": (),
    "rocm": ("-f", str(CHART_PATH / "values.rocm.yaml")),
    "k3s+rocm": (
        "-f",
        str(CHART_PATH / "values.k3s.yaml"),
        "-f",
        str(CHART_PATH / "values.rocm.yaml"),
    ),
    "k3s+rocm+modal": (
        "-f",
        str(CHART_PATH / "values.k3s.yaml"),
        "-f",
        str(CHART_PATH / "values.rocm.yaml"),
        *MODAL_LLM_VALUES,
    ),
    "prod": ("-f", str(CHART_PATH / "values.prod.yaml"), *PROD_SECRETS),
}


def _render_deployment_texts(stack: str, runtime_tag: str) -> dict[str, tuple]:
    """Each rendered Deployment's component and exact text, by name, with
    every runtime image tag the chart can resolve set to ``runtime_tag``."""
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        "--set",
        "runtime.backend=rocm",
        *STACKS[stack],
    ]
    for key in (
        "runtime.image.tag",
        "runtime.imagesByBackend.rocm.tag",
        "runtime.imagesByBackend.cuda.tag",
        "runtime.imagesByBackend.cpu.tag",
    ):
        command.extend(["--set", f"{key}={runtime_tag}"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )
    texts = {}
    for chunk in result.stdout.split("\n---\n"):
        document = yaml.safe_load(chunk)
        if document and document.get("kind") == "Deployment":
            component = document["metadata"]["labels"]["app.kubernetes.io/component"]
            texts[document["metadata"]["name"]] = (component, chunk)
    return texts


@pytest.mark.parametrize("stack", sorted(STACKS))
def test_a_runtime_release_leaves_every_inference_deployment_byte_identical(stack):
    """A new runtime image must never change a model pod's template: the
    change would restart it and reload its weights onto the GPU."""
    before = _render_deployment_texts(stack, "release-a")
    after = _render_deployment_texts(stack, "release-b")

    # The tag change took effect: the runtime itself renders differently.
    assert before["cogniverse-runtime"][1] != after["cogniverse-runtime"][1]
    inference = sorted(
        name
        for name, (component, _) in before.items()
        if component.startswith("inference-")
    )
    assert inference != []
    assert [name for name in inference if before[name] != after[name]] == []


def test_the_release_check_covers_gated_model_pods():
    """The invariant above is only worth its name if gated pods are in it."""
    gated = {
        stack: sorted(
            name
            for name, deployment in _inference_deployments(*STACKS[stack]).items()
            if _gate(deployment) is not None
        )
        for stack in ("rocm", "k3s+rocm", "k3s+rocm+modal")
    }

    assert gated == {
        "rocm": [
            "code_colbert_pylate",
            "colbert_pylate",
            "denseon",
            "vllm_asr",
            "vllm_colpali",
            "vllm_llm_student",
        ],
        "k3s+rocm": [
            "code_colbert_pylate",
            "colbert_pylate",
            "denseon",
            "vllm_asr",
            "vllm_colpali",
            "vllm_llm_student",
        ],
        "k3s+rocm+modal": [
            "code_colbert_pylate",
            "colbert_pylate",
            "denseon",
            "vllm_asr",
        ],
    }


def _merged_sequence(stack_args: tuple[str, ...]) -> list[str]:
    values: dict = yaml.safe_load((CHART_PATH / "values.yaml").read_text())
    files = [stack_args[i + 1] for i, arg in enumerate(stack_args) if arg == "-f"]
    sequence = (values.get("inferenceStartup") or {}).get("sequence") or []
    for path in files:
        overlay = yaml.safe_load(Path(path).read_text()) or {}
        if "sequence" in (overlay.get("inferenceStartup") or {}):
            sequence = overlay["inferenceStartup"]["sequence"]
    return list(sequence)


@pytest.mark.parametrize("stack", sorted(STACKS))
def test_every_running_sequenced_pod_after_the_first_waits_on_the_previous_one(
    stack,
):
    """No sequenced pod starts ungated just because the entry before it in the
    list runs no pod: the running entries form one chain, in list order."""
    deployments = _inference_deployments(*STACKS[stack])
    running = [name for name in _merged_sequence(STACKS[stack]) if name in deployments]

    waits_on = {
        name: (
            _gate_env(deployments[name])["GATE_URL"]
            if _gate(deployments[name]) is not None
            else None
        )
        for name in running
    }

    assert waits_on == {
        name: (
            None
            if position == 0
            else f"http://cogniverse-{running[position - 1].replace('_', '-')}:"
            f"{deployments[running[position - 1]]['spec']['template']['spec']['containers'][0]['ports'][0]['containerPort']}/health"
        )
        for position, name in enumerate(running)
    }


def test_the_gate_runs_from_the_curl_image_the_jobs_already_pull():
    documents = _render(*MODAL_LLM_VALUES)
    schema_job = next(
        d
        for d in documents
        if d["kind"] == "Job"
        and d["metadata"]["name"] == "cogniverse-schema-deployment"
    )
    gates = {
        name: (gate["image"], gate["imagePullPolicy"], gate["command"])
        for name, deployment in _inference_deployments(*MODAL_LLM_VALUES).items()
        if (gate := _gate(deployment)) is not None
    }

    assert schema_job["spec"]["template"]["spec"]["containers"][0]["image"] == (
        CURL_IMAGE
    )
    assert gates == {
        name: (CURL_IMAGE, "IfNotPresent", ["sh", "-c"])
        for name in ("vllm_asr", "denseon", "colbert_pylate", "code_colbert_pylate")
    }


def test_the_air_gap_mirror_lists_the_gate_image():
    """mirror-third-party.yml copies every quoted ``image:`` of the k3s render
    that is not a cogniverse image; a gated model pod cannot start where the
    gate's image was never mirrored."""
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "-f",
            str(CHART_PATH / "values.k3s.yaml"),
            "--set",
            "argo-workflows.crds.install=false",
            "--set",
            "runtime.qualityMonitor.tenantId=mirror",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    mirrored = {
        image
        for image in re.findall(r'image: "([^"]+)"', rendered)
        if not image.startswith("cogniverse/")
    }

    assert CURL_IMAGE in mirrored


class _Predecessor(BaseHTTPRequestHandler):
    """A predecessor whose /health answers each status in turn, then 200."""

    statuses: list[int] = []
    seen: list[str] = []

    def do_GET(self):
        type(self).seen.append(self.path)
        status = type(self).statuses.pop(0) if type(self).statuses else 200
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def _run_gate(gate_url: str, deadline_seconds: int) -> subprocess.CompletedProcess:
    """Run the rendered gate exactly as the kubelet would: its image, its
    command as the entrypoint, its args and its environment."""
    gate = _gate(_inference_deployments(*MODAL_LLM_VALUES)["denseon"])
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "--entrypoint",
            gate["command"][0],
            "-e",
            f"GATE_URL={gate_url}",
            "-e",
            f"GATE_DEADLINE_SECONDS={deadline_seconds}",
            gate["image"],
            *gate["command"][1:],
            *gate["args"],
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.requires_docker
def test_the_gate_starts_the_model_on_an_exact_200_from_its_predecessor():
    _Predecessor.statuses = [503, 204]
    _Predecessor.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Predecessor)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/health"
    try:
        gate = _run_gate(url, deadline_seconds=600)
    finally:
        server.shutdown()

    assert gate.returncode == 0, gate.stderr
    assert gate.stdout.splitlines() == [
        f"waiting on {url}: HTTP 503",
        f"waiting on {url}: HTTP 204",
        f"{url} is serving; starting model load",
    ]
    assert _Predecessor.seen == ["/health", "/health", "/health"]


@pytest.mark.requires_docker
def test_the_gate_starts_the_model_at_its_deadline_when_nothing_answers():
    probe = ThreadingHTTPServer(("127.0.0.1", 0), _Predecessor)
    url = f"http://127.0.0.1:{probe.server_address[1]}/health"
    probe.server_close()  # nothing listens on this port any more

    gate = _run_gate(url, deadline_seconds=1)

    assert gate.returncode == 0, gate.stderr
    assert gate.stdout.splitlines() == [
        f"waiting on {url}: HTTP 000",
        f"waiting on {url}: HTTP 000",
        f"{url} did not answer before the pacing deadline",
    ]
