"""ROCm inference startup budgets rendered by the Helm chart."""

import shlex
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


def _rocm_inference_containers() -> dict[str, dict]:
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "-f",
            str(CHART_PATH / "values.rocm.yaml"),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )

    containers = {}
    for document in yaml.safe_load_all(result.stdout):
        if document is None or document.get("kind") != "Deployment":
            continue
        component = document["metadata"]["labels"].get(
            "app.kubernetes.io/component", ""
        )
        if component.startswith("inference-"):
            containers[component.removeprefix("inference-")] = document["spec"][
                "template"
            ]["spec"]["containers"][0]
    return containers


def test_tomoro_rocm_startup_profiles_one_sequence():
    container = _rocm_inference_containers()["vllm_colpali"]

    assert container["args"] == [
        "serve",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "--revision",
        "bf790bd8780b098b86453444632a184bb770be1a",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "4096",
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
        "--kv-cache-memory-bytes",
        "1G",
        "--mm-processor-kwargs",
        '{"max_pixels":1048576}',
        "--gpu-memory-utilization",
        "0.45",
        "--max-num-seqs",
        "1",
    ]


def test_whisper_rocm_startup_caps_sequences_and_batched_tokens():
    container = _rocm_inference_containers()["vllm_asr"]
    command = container["args"][0]
    serve_command = command[command.index("exec ") + len("exec ") :].replace(
        "\\\n", " "
    )

    assert shlex.split(serve_command) == [
        "vllm",
        "serve",
        "openai/whisper-large-v3-turbo",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--revision",
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "--runner",
        "generate",
        "--max-model-len",
        "448",
        "--gpu-memory-utilization",
        "0.16",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
    ]
