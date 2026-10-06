"""Inference model pods start with no network access.

A node that comes back from a reboot without a working DNS upstream must still
bring every model pod to readiness from what is already on the node: the image
and the model cache. So no container of an inference pod may install packages
or reach an outside host when it starts, the model server itself must keep the
Hugging Face libraries offline, and the only Hub download allowed is the
model-warm init container's fetch of a revision the cache does not hold yet.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.check_offline_model_pods import offline_startup_violations

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)

_ALL_SERVICES_ON = (
    "inference.clap_embed.enabled=true",
    "inference.face_embed.enabled=true",
    "inference.video_embed.enabled=true",
    "inference.code_colbert_pylate.enabled=true",
    "inference.vllm_llm_student.enabled=true",
    "inference.vllm_llm_teacher.enabled=true",
    "inference.vllm_colpali.enabled=true",
)

# Every way the chart backs the model cache, on every device overlay.
SCENARIOS = {
    "defaults": ((), ()),
    "k3s-rocm": (("values.k3s.yaml", "values.rocm.yaml"), ()),
    "k3s-cpu": (("values.k3s.yaml", "values.cpu.yaml"), ()),
    "cuda": (("values.cuda.yaml",), ()),
    "pvc-minio": (
        (),
        (
            "hfCache.persistence.enabled=true",
            "hfCache.persistence.minio.enabled=true",
            "hfCache.persistence.minio.endpoint=http://cogniverse-minio:9000",
            "hfCache.persistence.minio.existingSecret=cogniverse-minio",
        ),
    ),
}


def _render(values_files: tuple[str, ...], set_args: tuple[str, ...]) -> list[dict]:
    cmd = ["helm", "template", "cogniverse", str(CHART_PATH)]
    for values_file in values_files:
        cmd.extend(["-f", str(CHART_PATH / values_file)])
    for arg in (
        "runtime.qualityMonitor.tenantId=test-tenant",
        *_ALL_SERVICES_ON,
        *set_args,
    ):
        cmd.extend(["--set", arg])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _inference_pods(docs: list[dict]) -> dict[str, dict]:
    pods = {}
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        template = doc["spec"]["template"]
        component = template["metadata"]["labels"].get(
            "app.kubernetes.io/component", ""
        )
        if component.startswith("inference-"):
            pods[component.removeprefix("inference-")] = template["spec"]
    return pods


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_inference_pod_starts_without_network(scenario: str) -> None:
    values_files, set_args = SCENARIOS[scenario]
    pods = _inference_pods(_render(values_files, set_args))
    assert set(pods) == {
        "clap_embed",
        "code_colbert_pylate",
        "colbert_pylate",
        "denseon",
        "face_embed",
        "gliner",
        "video_embed",
        "vllm_asr",
        "vllm_colpali",
        "vllm_llm_student",
        "vllm_llm_teacher",
    }
    violations = {
        svc: problems
        for svc, pod in sorted(pods.items())
        if (problems := offline_startup_violations(pod))
    }
    assert violations == {}


def test_hub_weights_are_warmed_at_the_revision_the_server_loads() -> None:
    pods = _inference_pods(_render(("values.k3s.yaml", "values.rocm.yaml"), ()))
    for svc, pod in pods.items():
        warm = [c for c in pod.get("initContainers", []) if c["name"] == "model-warm"]
        server = pod["containers"][0]
        env = {e["name"]: e.get("value") for e in server.get("env", [])}
        args = server.get("args", [])
        if svc in {"gliner", "face_embed"}:
            # Both serve artifacts baked into their images.
            assert warm == [], svc
            continue
        assert len(warm) == 1, svc
        warm_env = {e["name"]: e.get("value") for e in warm[0]["env"]}
        if server.get("command") == ["vllm"]:
            loaded = (
                args[1],
                args[args.index("--revision") + 1] if "--revision" in args else "",
            )
        elif "MODEL_NAME" in env:
            loaded = (env["MODEL_NAME"], env["MODEL_REVISION"])
        else:
            prefix = {"clap_embed": "CLAP_EMBED", "video_embed": "VIDEO_EMBED"}[svc]
            loaded = (env[f"{prefix}_MODEL"], env[f"{prefix}_MODEL_REVISION"])
        assert (warm_env["MODEL"], warm_env["REVISION"]) == loaded, svc


class TestDetector:
    def _pod(self, **container: object) -> dict:
        main = {
            "name": "server",
            "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}],
            **container,
        }
        return {"containers": [main]}

    def test_shell_pip_install_before_serve_is_refused(self) -> None:
        pod = self._pod(
            command=["sh", "-c"],
            args=[
                "pip install --no-cache-dir soundfile librosa || exit 1\nexec vllm serve m"
            ],
        )
        assert offline_startup_violations(pod) == [
            "container server installs packages at start",
            "container server starts through a shell script",
        ]

    def test_apt_install_in_an_init_container_is_refused(self) -> None:
        pod = self._pod(command=["python3", "server.py"])
        pod["initContainers"] = [
            {"name": "deps", "command": ["sh", "-c", "apt-get -y install ffmpeg"]}
        ]
        assert offline_startup_violations(pod) == [
            "init deps installs packages at start"
        ]

    def test_server_without_hub_offline_is_refused(self) -> None:
        pod = {
            "containers": [
                {"name": "vllm", "command": ["vllm"], "args": ["serve", "m"]}
            ]
        }
        assert offline_startup_violations(pod) == [
            "container vllm does not set HF_HUB_OFFLINE=1"
        ]

    def test_unconditional_hub_download_in_init_is_refused(self) -> None:
        pod = self._pod(command=["vllm"])
        pod["initContainers"] = [
            {
                "name": "model-warm",
                "command": ["sh", "-c"],
                "args": ["python -c 'snapshot_download(repo_id=\"m\")'"],
            }
        ]
        assert offline_startup_violations(pod) == [
            "init model-warm downloads from the Hub without a cache check"
        ]

    def test_cache_first_warm_and_in_cluster_gate_are_clean(self) -> None:
        pod = self._pod(command=["vllm"])
        pod["initContainers"] = [
            {
                "name": "model-warm",
                "command": ["python3", "-c"],
                "args": [
                    "snapshot_download(repo_id=m, local_files_only=True)\n"
                    "snapshot_download(repo_id=m)"
                ],
            },
            {
                "name": "startup-gate",
                "command": ["sh", "-c"],
                "args": ['curl -sS "$GATE_URL"'],
                "env": [
                    {
                        "name": "GATE_URL",
                        "value": "http://cogniverse-denseon:8000/health",
                    }
                ],
            },
        ]
        assert offline_startup_violations(pod) == []

    def test_outside_url_is_refused(self) -> None:
        pod = self._pod(
            command=["sh", "-c"],
            args=["curl -fsSL https://example.com/get.sh -o /tmp/x"],
        )
        assert "container server reaches outside host https://example.com/get.sh" in (
            offline_startup_violations(pod)
        )
