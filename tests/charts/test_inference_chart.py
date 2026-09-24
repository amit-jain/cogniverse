"""Chart tests for the generic inference services.

The chart supports N parallel inference services under ``inference`` — each
entry deploys one pod. Keys are logical tags (e.g. ``colbert_pylate`` for
the LateOn text multi-vector pod, ``denseon`` for the DenseOn dense pod).
Each service has an ``engine`` that selects the container template
(``pylate`` for exact PyLate per-token multi-vector, ``vllm_token_embed``
for vLLM per-token multi-vector, ``vllm_embed`` for dense single-vector,
``vllm_chat``, ``vllm_transcription``, ``gliner``, ``fastapi``, …) and a
``type`` (``multi_vector`` or ``single_vector``).

The runtime receives one ``INFERENCE_SERVICE_URLS`` JSON env var containing
{service_key: url} for every enabled service. Profiles pick a service by key.
"""

import contextlib
import copy
import inspect
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import yaml
from cogniverse_cli.config import (
    LLM_SERVING_LOCAL,
    LLM_SERVING_MODAL,
    compose_values_files,
)
from cogniverse_cli.deploy import helm_install
from cogniverse_cli.images import SIDECAR_BUILDS
from fastapi import Body, FastAPI, HTTPException

from cogniverse_core.registries.schema_deploy_lease import (
    DEFAULT_WAIT_SECONDS,
    MAX_TOTAL_HOLD_SECONDS,
)
from cogniverse_foundation.config.utils import resolve_default_profile
from cogniverse_foundation.inference_specs import (
    INFERENCE_SERVICE_SPECS,
    get_inference_service_spec,
)
from cogniverse_runtime.admin.profile_models import SchemaDeploymentResponse
from cogniverse_vespa.backend import SCHEMA_CONVERGENCE_TIMEOUT_S
from cogniverse_vespa.vespa_schema_manager import DEPLOY_REQUEST_TIMEOUT_S

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*set_args: str, values: str | tuple[str, ...] | None = None) -> list[dict]:
    cmd = ["helm", "template", "cogniverse", str(CHART_PATH)]
    for values_file in (values,) if isinstance(values, str) else (values or ()):
        cmd.extend(["-f", str(CHART_PATH / values_file)])
    # The chart fail-fasts if qualityMonitor.tenantId is empty; supply a
    # placeholder so inference wiring is the only variable under test.
    cmd.extend(["--set", "runtime.qualityMonitor.tenantId=test-tenant"])
    for arg in set_args:
        cmd.extend(["--set", arg])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n{result.stderr}"
        )
    return [d for d in yaml.safe_load_all(result.stdout) if d is not None]


def _values(name: str) -> dict:
    return yaml.safe_load((CHART_PATH / name).read_text())


def _inference_deployments(docs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in docs:
        if d.get("kind") != "Deployment":
            continue
        component = (
            d.get("metadata", {})
            .get("labels", {})
            .get("app.kubernetes.io/component", "")
        )
        if component.startswith("inference-"):
            out[component.removeprefix("inference-")] = d
    return out


def _inference_services(docs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for document in docs:
        if document.get("kind") != "Service":
            continue
        component = (
            document.get("metadata", {})
            .get("labels", {})
            .get("app.kubernetes.io/component", "")
        )
        if component.startswith("inference-"):
            out[component.removeprefix("inference-")] = document
    return out


def _inference_env(deps: dict[str, dict], key: str) -> dict[str, str]:
    container = deps[key]["spec"]["template"]["spec"]["containers"][0]
    return {e["name"]: e.get("value") for e in container.get("env", [])}


_TUNABLEOP_VARS = {
    "PYTORCH_TUNABLEOP_ENABLED",
    "PYTORCH_TUNABLEOP_TUNING",
    "PYTORCH_TUNABLEOP_FILENAME",
}


def _runtime_env(docs: list[dict]) -> dict[str, str]:
    for d in docs:
        if (
            d.get("kind") == "Deployment"
            and d.get("metadata", {})
            .get("labels", {})
            .get("app.kubernetes.io/component")
            == "runtime"
        ):
            container = d["spec"]["template"]["spec"]["containers"][0]
            return {e["name"]: e.get("value") for e in container.get("env", [])}
    raise AssertionError("runtime Deployment not found")


def _service_urls(docs: list[dict]) -> dict[str, str]:
    env = _runtime_env(docs)
    raw = env.get("INFERENCE_SERVICE_URLS", "").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _teacher_api_base(docs: list[dict]) -> str:
    cm = next(
        d
        for d in docs
        if d.get("kind") == "ConfigMap" and "config.json" in (d.get("data") or {})
    )
    return json.loads(cm["data"]["config.json"])["llm_config"]["teacher"]["api_base"]


def test_default_runs_colbert_pylate_and_denseon_services():
    """Default-enabled inference services: colbert_pylate (LateOn text
    multi-vector), denseon (DenseOn dense single-vector), gliner
    (zero-shot NER), and vllm_asr (Whisper transcription). Mem0 needs
    denseon for memory embeddings, the slim runtime image excludes
    torch+gliner, and the default video-ingestion profiles hard-require
    vllm_asr for transcription, so all four ship enabled by default. The
    teacher (vllm_llm_teacher) also renders by default, at scale-to-zero,
    because the semantic router serves pro-reasoning from it."""
    deps = _inference_deployments(_render())
    assert set(deps.keys()) == {
        "colbert_pylate",
        "denseon",
        "gliner",
        "vllm_asr",
        "vllm_llm_teacher",
    }
    assert deps["colbert_pylate"]["metadata"]["name"] == "cogniverse-colbert-pylate"
    assert deps["denseon"]["metadata"]["name"] == "cogniverse-denseon"
    assert deps["gliner"]["metadata"]["name"] == "cogniverse-gliner"
    assert deps["vllm_asr"]["metadata"]["name"] == "cogniverse-vllm-asr"
    assert deps["vllm_llm_teacher"]["metadata"]["name"] == "cogniverse-vllm-llm-teacher"
    assert deps["vllm_llm_teacher"]["spec"]["replicas"] == 0


def test_default_gliner_deployment_uses_the_pinned_production_model():
    docs = _render()
    deps = _inference_deployments(docs)

    assert _inference_env(deps, "gliner")["MODEL_NAME"] == ("urchade/gliner_large-v2.1")


def test_video_embed_sidecar_env_matches_the_shipped_spec():
    """The chart must serve the checkpoint the spec declares.

    The sidecar asserts its embedding width on every response, so a chart
    that ships a different model or width fails each request instead of
    writing a mis-shaped vector. Compared against the production constant,
    not restated literals, so a spec change cannot pass unnoticed.
    """
    spec = INFERENCE_SERVICE_SPECS["video_embed"]
    deps = _inference_deployments(_render("inference.video_embed.enabled=true"))
    env = _inference_env(deps, "video_embed")

    assert env["VIDEO_EMBED_MODEL"] == spec.model_id
    assert env["VIDEO_EMBED_MODEL_REVISION"] == spec.model_revision
    assert env["VIDEO_EMBED_DIM"] == str(spec.output_dimension)
    assert env["VIDEO_EMBED_NUM_FRAMES"] == "8"

    container = deps["video_embed"]["spec"]["template"]["spec"]["containers"][0]
    base = _values("values.yaml")["inference"]["video_embed"]
    assert container["image"] == (
        f"{SIDECAR_BUILDS['video_embed'][0]}:{base['image']['tag']}"
    )
    # Model weights live in RAM for the pod's life; a request below the limit
    # makes the pod Burstable and evictable out of the shared pool.
    assert (
        container["resources"]["requests"]["memory"]
        == (container["resources"]["limits"]["memory"])
    )


def test_k3s_overlay_serves_video_embed_from_the_locally_built_image():
    """The local stack's text-to-video retrieval runs through the X-CLIP
    sidecar the deploy builds itself: the overlay enables the pod on the image
    family ``cogniverse up`` produces, serves the spec's checkpoint at the
    spec's width, exposes the fixed NodePort, and hands the runtime the
    in-cluster URL. Memory request equals the limit so the pod holding the
    weights cannot be evicted out of the shared pool."""
    spec = get_inference_service_spec("video_embed")
    base = _values("values.yaml")["inference"]["video_embed"]
    overlay = _values("values.k3s.yaml")["inference"]["video_embed"]
    docs = _render(values="values.k3s.yaml")
    deps = _inference_deployments(docs)
    container = deps["video_embed"]["spec"]["template"]["spec"]["containers"][0]

    assert container["image"] == (
        f"{SIDECAR_BUILDS['video_embed'][0]}:{overlay['image']['tag']}"
    )
    assert container["imagePullPolicy"] == "Never"

    env = _inference_env(deps, "video_embed")
    assert env["VIDEO_EMBED_MODEL"] == spec.model_id
    assert env["VIDEO_EMBED_MODEL_REVISION"] == spec.model_revision
    assert env["VIDEO_EMBED_DIM"] == str(spec.output_dimension)
    assert env["VIDEO_EMBED_DIM"] == base["env"]["VIDEO_EMBED_DIM"]
    assert env["VIDEO_EMBED_NUM_FRAMES"] == base["env"]["VIDEO_EMBED_NUM_FRAMES"]
    assert env["PORT"] == str(base["service"]["port"])

    services = _inference_services(docs)
    assert services["video_embed"]["spec"]["type"] == "NodePort"
    assert services["video_embed"]["spec"]["ports"] == [
        {
            "name": "http",
            "nodePort": base["service"]["nodePort"],
            "port": base["service"]["port"],
            "protocol": "TCP",
            "targetPort": "http",
        }
    ]
    node_ports = {
        key: service["spec"]["ports"][0]["nodePort"]
        for key, service in services.items()
        if service["spec"]["type"] == "NodePort"
    }
    assert node_ports["video_embed"] == base["service"]["nodePort"]
    assert len(set(node_ports.values())) == len(node_ports)

    assert _service_urls(docs)["video_embed"] == (
        f"http://cogniverse-video-embed:{base['service']['port']}"
    )

    resources = container["resources"]
    assert resources["requests"]["memory"] == resources["limits"]["memory"]


def test_gliner_and_face_nodeports_are_distinct_when_both_are_exposed():
    services = _inference_services(
        _render(
            "inference.gliner.service.type=NodePort",
            "inference.face_embed.enabled=true",
        )
    )

    assert services["gliner"]["spec"]["ports"] == [
        {
            "name": "http",
            "nodePort": 29007,
            "port": 8080,
            "protocol": "TCP",
            "targetPort": "http",
        }
    ]
    assert services["face_embed"]["spec"]["ports"] == [
        {
            "name": "http",
            "nodePort": 29009,
            "port": 8000,
            "protocol": "TCP",
            "targetPort": "http",
        }
    ]


@pytest.mark.parametrize(
    ("service", "port", "readiness_timing", "liveness_timing"),
    [
        (
            "gliner",
            8080,
            {
                "failureThreshold": 30,
                "initialDelaySeconds": 0,
                "periodSeconds": 15,
                "timeoutSeconds": 5,
            },
            {
                "failureThreshold": 10,
                "initialDelaySeconds": 60,
                "periodSeconds": 30,
                "timeoutSeconds": 5,
            },
        ),
        (
            "clap_embed",
            8000,
            {
                "failureThreshold": 5,
                "initialDelaySeconds": 0,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
            },
            {
                "failureThreshold": 5,
                "initialDelaySeconds": 30,
                "periodSeconds": 30,
                "timeoutSeconds": 5,
            },
        ),
        (
            "face_embed",
            8000,
            {
                "failureThreshold": 5,
                "initialDelaySeconds": 0,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
            },
            {
                "failureThreshold": 5,
                "initialDelaySeconds": 30,
                "periodSeconds": 30,
                "timeoutSeconds": 5,
            },
        ),
        (
            "video_embed",
            8000,
            {
                "failureThreshold": 5,
                "initialDelaySeconds": 0,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
            },
            {
                "failureThreshold": 5,
                "initialDelaySeconds": 30,
                "periodSeconds": 30,
                "timeoutSeconds": 5,
            },
        ),
    ],
)
def test_model_sidecars_separate_model_readiness_from_process_liveness(
    service: str,
    port: int,
    readiness_timing: dict[str, int],
    liveness_timing: dict[str, int],
):
    deps = _inference_deployments(
        _render(
            "inference.clap_embed.enabled=true",
            "inference.face_embed.enabled=true",
            "inference.video_embed.enabled=true",
        )
    )
    container = deps[service]["spec"]["template"]["spec"]["containers"][0]

    assert container["readinessProbe"] == {
        "httpGet": {"path": "/health", "port": port},
        **readiness_timing,
    }
    assert container["livenessProbe"] == {
        "tcpSocket": {"port": port},
        **liveness_timing,
    }


def test_default_colbert_pylate_serves_lateon_via_pylate():
    """Default colbert_pylate service runs the PyLate sidecar image with the
    pinned LateOn revision. LateOn needs PyLate's exact encode (query
    expansion over masked padding), which stock vLLM cannot reproduce, so
    the pod must not carry a vLLM launch command."""
    deps = _inference_deployments(_render())
    container = deps["colbert_pylate"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "cogniverse/pylate:0.1.0"
    assert "command" not in container
    assert "args" not in container
    env = {e["name"]: e.get("value") for e in container.get("env", [])}
    assert env == {
        "MODEL_NAME": "lightonai/LateOn",
        "MODEL_REVISION": "c01907b70557ee5c7753680d4819a5cce1674b83",
        "DEVICE": "cpu",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "MAX_INPUT_ITEMS": "256",
        "MAX_INPUT_CHARS": "2000000",
        "ENCODE_BATCH_SIZE": "32",
        "HF_HOME": "/root/.cache/huggingface",
    }
    assert container["ports"] == [{"name": "http", "containerPort": 8000}]


def test_enabled_code_colbert_pylate_pins_lateon_code_edge():
    deps = _inference_deployments(_render("inference.code_colbert_pylate.enabled=true"))
    container = deps["code_colbert_pylate"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "cogniverse/pylate:0.1.0"
    env = {e["name"]: e.get("value") for e in container.get("env", [])}
    assert env["MODEL_NAME"] == "lightonai/LateOn-Code-edge"
    assert env["MODEL_REVISION"] == "07ef20f406c86badca122464808f4cac2f6e4b25"


def test_default_inference_service_urls_contains_colbert_pylate_and_denseon():
    urls = _service_urls(_render())
    assert urls == {
        "colbert_pylate": "http://cogniverse-colbert-pylate:8000",
        "denseon": "http://cogniverse-denseon:8000",
        "gliner": "http://cogniverse-gliner:8080",
        "vllm_asr": "http://cogniverse-vllm-asr:8000",
        "vllm_llm_teacher": "http://cogniverse-vllm-llm-teacher:8000",
    }


def test_enabling_code_runs_three_parallel_services():
    """code_colbert_pylate adds a third pod alongside the defaults."""
    docs = _render("inference.code_colbert_pylate.enabled=true")
    deps = _inference_deployments(docs)
    assert set(deps.keys()) == {
        "colbert_pylate",
        "denseon",
        "gliner",
        "vllm_asr",
        "vllm_llm_teacher",
        "code_colbert_pylate",
    }
    assert deps["colbert_pylate"]["metadata"]["name"] == "cogniverse-colbert-pylate"
    assert (
        deps["code_colbert_pylate"]["metadata"]["name"]
        == "cogniverse-code-colbert-pylate"
    )


def test_enabling_code_adds_to_url_map():
    urls = _service_urls(_render("inference.code_colbert_pylate.enabled=true"))
    assert urls == {
        "colbert_pylate": "http://cogniverse-colbert-pylate:8000",
        "code_colbert_pylate": "http://cogniverse-code-colbert-pylate:8000",
        "denseon": "http://cogniverse-denseon:8000",
        "gliner": "http://cogniverse-gliner:8080",
        "vllm_asr": "http://cogniverse-vllm-asr:8000",
        "vllm_llm_teacher": "http://cogniverse-vllm-llm-teacher:8000",
    }


def test_overriding_one_service_model_does_not_affect_another():
    """Overriding one service's model must not bleed into a sibling pod."""
    docs = _render(
        "inference.code_colbert_pylate.enabled=true",
        "inference.colbert_pylate.model=lightonai/Reason-ModernColBERT",
    )
    deps = _inference_deployments(docs)

    def _env(service: str) -> dict:
        container = deps[service]["spec"]["template"]["spec"]["containers"][0]
        return {e["name"]: e.get("value") for e in container.get("env", [])}

    assert _env("colbert_pylate")["MODEL_NAME"] == "lightonai/Reason-ModernColBERT"
    assert _env("code_colbert_pylate")["MODEL_NAME"] == "lightonai/LateOn-Code-edge"


def test_default_denseon_serves_via_vllm_embed():
    """Default denseon service serves DenseOn via vLLM's dense embed runner."""
    deps = _inference_deployments(_render())
    container = deps["denseon"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].startswith("vllm/vllm-openai")
    args = container["args"]
    assert "lightonai/DenseOn" in args
    assert "serve" in args
    assert "--convert" in args and args[args.index("--convert") + 1] == "embed"
    assert "--hf-overrides" not in args  # dense, no multi-vector arch override
    env = {e["name"]: e.get("value") for e in container.get("env", [])}
    assert "MODEL_NAME" not in env


def test_disabling_colbert_pylate_drops_service_and_url():
    docs = _render("inference.colbert_pylate.enabled=false")
    deps = _inference_deployments(docs)
    assert "colbert_pylate" not in deps
    assert "colbert_pylate" not in _service_urls(docs)


def test_vllm_colpali_serves_tomoro_token_embed():
    docs = _render("inference.vllm_colpali.enabled=true")
    dep = _inference_deployments(docs)["vllm_colpali"]
    c = dep["spec"]["template"]["spec"]["containers"][0]
    # Pinned image, not a floating ``latest``: ColQwen3 support landed in
    # vLLM 0.21, and a stale cached ``latest`` silently serves 0.20 which
    # fails to load the model.
    assert c["image"] == "vllm/vllm-openai-cpu:v0.23.0"
    args = c["args"]
    assert "TomoroAI/tomoro-colqwen3-embed-4b" in args
    assert args[args.index("--runner") + 1] == "pooling"
    assert args[args.index("--convert") + 1] == "embed"
    # qwen3_vl's ViT tower OOMs vLLM's startup profiler on a worst-case
    # video buffer unless video multimodal input is disabled.
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"video":0,"image":1}'


def test_inference_readiness_has_no_fixed_cold_start_delay():
    docs = _render(
        "inference.vllm_colpali.enabled=true",
        "inference.code_colbert_pylate.enabled=true",
        "inference.vllm_llm_student.enabled=true",
        "inference.vllm_llm_teacher.enabled=true",
        "inference.clap_embed.enabled=true",
        "inference.face_embed.enabled=true",
    )
    deployments = _inference_deployments(docs)

    assert deployments
    for name, deployment in deployments.items():
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        probe = container["readinessProbe"]
        assert probe["initialDelaySeconds"] == 0, name
        assert probe["httpGet"]["path"] == "/health", name


def test_vllm_llm_student_allows_keyframe_images():
    """The answer/student LLM accepts up to 4 still images per prompt — the
    keyframes the multimodal generation agents attach — while keeping video at
    0. Profiling the video encoder cache is what blows startup memory; a few
    still images is bounded. image must stay >= the agents' max_keyframes_to_llm."""
    docs = _render("inference.vllm_llm_student.enabled=true")
    c = _inference_deployments(docs)["vllm_llm_student"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    args = c["args"]
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"video":0,"image":4}'


def test_vllm_asr_enabled_by_default():
    """vllm_asr ships enabled in base values.yaml because the default
    video-ingestion profiles hard-require transcription. Operators that
    never ingest video can disable it explicitly."""
    deps = _inference_deployments(_render())
    assert "vllm_asr" in deps
    assert _service_urls(_render())["vllm_asr"] == "http://cogniverse-vllm-asr:8000"


def test_disabling_vllm_asr_drops_service_and_url():
    docs = _render("inference.vllm_asr.enabled=false")
    deps = _inference_deployments(docs)
    assert "vllm_asr" not in deps
    assert "vllm_asr" not in _service_urls(docs)


def test_vllm_asr_serves_whisper_turbo_transcription():
    """When enabled, vllm_asr serves openai/whisper-large-v3-turbo via the
    transcription runner and gets a resolvable URL in the service map."""
    docs = _render("inference.vllm_asr.enabled=true")
    dep = _inference_deployments(docs)["vllm_asr"]
    assert dep["metadata"]["name"] == "cogniverse-vllm-asr"
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["image"].startswith("vllm/vllm-openai")
    # The transcription engine renders a single shell command string that
    # pip-installs the audio extras then execs `vllm serve <model>`.
    cmd = " ".join(c["args"])
    assert "vllm serve 'openai/whisper-large-v3-turbo'" in cmd
    assert "'--runner' \\\n  'generate'" in cmd
    urls = _service_urls(docs)
    assert urls["vllm_asr"] == "http://cogniverse-vllm-asr:8000"


def test_denseon_uses_vllm_embed_engine():
    docs = _render(
        "inference.denseon.engine=vllm_embed",
        "inference.denseon.model=lightonai/DenseOn",
    )
    c = _inference_deployments(docs)["denseon"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert c["image"].startswith("vllm/vllm-openai")
    args = " ".join(c["args"])
    assert "lightonai/DenseOn" in args and "serve" in args
    assert "--hf-overrides" not in args  # dense, no arch override


def test_service_keys_in_url_map_match_deployment_names():
    """Every deployed service has a matching URL entry."""
    docs = _render(
        "inference.code_colbert_pylate.enabled=true",
        "inference.vllm_colpali.enabled=true",
    )
    deps = _inference_deployments(docs)
    urls = _service_urls(docs)
    assert set(deps.keys()) == set(urls.keys())
    for key in deps:
        # cogniverse-<key-kebabcased>
        kebab = key.replace("_", "-")
        assert urls[key].startswith(f"http://cogniverse-{kebab}")


def test_k3s_exposes_every_stateless_inference_service_on_a_unique_node_port():
    docs = _render(
        "inference.vllm_colpali.enabled=true",
        "inference.code_colbert_pylate.enabled=true",
        "inference.vllm_llm_student.enabled=true",
        "inference.clap_embed.enabled=true",
        "inference.face_embed.enabled=true",
        values="values.k3s.yaml",
    )
    services = _inference_services(docs)
    expected_ports = {
        "vllm_colpali": 29001,
        "colbert_pylate": 29002,
        "code_colbert_pylate": 29004,
        "vllm_asr": 29005,
        "denseon": 29006,
        "gliner": 29007,
        "clap_embed": 29008,
        "face_embed": 29009,
        "vllm_llm_student": 29010,
    }

    actual_ports = {}
    for key in expected_ports:
        service = services[key]
        assert service["spec"]["type"] == "NodePort", key
        actual_ports[key] = service["spec"]["ports"][0]["nodePort"]

    assert actual_ports == expected_ports
    assert len(set(actual_ports.values())) == len(actual_ports)


def _rendered_chart_config() -> dict:
    """Parse the config.json the chart renders into the runtime ConfigMap."""
    docs = _render("runtime.qualityMonitor.tenantId=test-tenant")
    cm = next(
        d
        for d in docs
        if d.get("kind") == "ConfigMap" and "config.json" in (d.get("data") or {})
    )
    return json.loads(cm["data"]["config.json"])


def _normalize_profiles(profiles: dict) -> dict:
    """Strip the deploy-specific VLM endpoint before comparing. The chart's
    config.json is ``tpl``-rendered: ``vlm_endpoint`` is injected as the
    in-cluster vLLM ``/v1`` URL, while local leaves it empty for the operator
    to set. That field differs by design; every other field (models,
    inference_services, all other strategies) stays strict, so real
    profile/model drift is still caught."""
    normalized = copy.deepcopy(profiles)
    for profile in normalized.values():
        params = (profile.get("strategies", {}).get("description", {}) or {}).get(
            "params", {}
        )
        params.pop("vlm_endpoint", None)
    return normalized


def test_chart_config_profiles_match_local_config():
    """The chart-bundled config.json (what the deployed runtime reads) must
    carry the SAME backend.profiles as configs/config.json (what local/tests
    use), modulo deploy-specific VLM endpoint injection. Drift here ships a
    stale model to the cluster and crashes the runtime's
    validate_inference_services on startup — the colpali-v1.3 vs Tomoro
    mismatch this test guards against."""
    local = json.loads((REPO_ROOT / "configs" / "config.json").read_text())
    chart = _rendered_chart_config()
    assert _normalize_profiles(chart["backend"]["profiles"]) == _normalize_profiles(
        local["backend"]["profiles"]
    )


def test_chart_visual_profiles_serve_tomoro():
    """Every col* visual profile in the deployed config must bind vllm_colpali
    to Tomoro ColQwen3 — the model the chart actually serves."""
    chart = _rendered_chart_config()
    visual = {
        "video_colpali_smol500_mv_frame",
        "image_colpali_mv",
        "document_visual_colpali",
        "video_colqwen_omni_mv_chunk_30s",
    }
    profiles = chart["backend"]["profiles"]
    for name in visual:
        p = profiles[name]
        assert p["embedding_model"] == "TomoroAI/tomoro-colqwen3-embed-4b", name
        assert p["inference_services"]["embedding"] == "vllm_colpali", name


def test_shipped_video_chunk_profile_has_one_exact_colqwen3_contract():
    profile_name = "video_colqwen_omni_mv_chunk_30s"
    local = json.loads((REPO_ROOT / "configs" / "config.json").read_text())
    example = json.loads(
        (REPO_ROOT / "configs" / "examples" / "config.example.json").read_text()
    )
    chart = _rendered_chart_config()

    profiles = tuple(
        config["backend"]["profiles"][profile_name]
        for config in (local, chart, example)
    )
    for profile in profiles:
        assert profile["description"] == (
            "ColQwen3 visual document retrieval served by the Cogniverse ColPali "
            "service. 320-dim per-patch multi-vector embeddings."
        )
        assert profile["embedding_model"] == "TomoroAI/tomoro-colqwen3-embed-4b"
        assert profile["model_config"] == {"token_pool_factor": 3}
        assert profile["model_loader"] == "colqwen"
        assert profile["inference_services"] == {
            "embedding": "vllm_colpali",
            "transcription": "vllm_asr",
        }
        assert profile["schema_config"] == {
            "schema_name": profile_name,
            "model_name": "ColQwen3",
            "num_patches": 1024,
            "embedding_dim": 320,
            "binary_dim": 40,
        }


def _chart_config(docs: list[dict]) -> dict:
    cm = next(
        d
        for d in docs
        if d.get("kind") == "ConfigMap" and "config.json" in (d.get("data") or {})
    )
    return json.loads(cm["data"]["config.json"])


def _shipped_default_video_selection() -> dict:
    local = json.loads((REPO_ROOT / "configs" / "config.json").read_text())
    return local["backend"]["default_profiles"]["video"]


def test_base_values_select_no_video_profile():
    """The base values are the CPU defaults, and no visual embedder is enabled
    there, so they select no video profile rather than one nothing serves."""
    config = _chart_config(_render())

    assert config["backend"]["default_profiles"] == {}
    assert "active_video_profile" not in config
    assert resolve_default_profile(config) is None


def test_rocm_overlay_selects_the_shipped_default_video_profile():
    """The selection names the profile only. Search resolves the ranking from
    the profile's schema when default_profiles carries no strategy."""
    config = _chart_config(_render(values="values.rocm.yaml"))

    assert _shipped_default_video_selection()["profile"] == (
        "video_colpali_smol500_mv_frame"
    )
    assert config["backend"]["default_profiles"] == {
        "video": {"profile": "video_colpali_smol500_mv_frame"}
    }
    assert config["active_video_profile"] == "video_colpali_smol500_mv_frame"
    assert resolve_default_profile(config) == "video_colpali_smol500_mv_frame"


_SELECTED_VIDEO_PROFILE = "video_colpali_smol500_mv_frame"
_STUDENT_API_BASE = "https://student.example.com/v1"
_PROD_SECRETS = (
    "minio.rootPassword=test-minio",
    "openshell.server.sshHandshakeSecret=test-handshake",
    "phoenix.postgres.auth.password=test-postgres",
    "redis.auth.password=test-redis",
)
_VISUAL_SERVICES = {"vllm_colpali", "vllm_asr", "vllm_llm_student"}


def _cli_values_stack(
    backend: str | None, *, use_k3d: bool, serving: str = LLM_SERVING_LOCAL
) -> tuple[str, ...]:
    """The values files ``cogniverse up`` composes, in its order."""
    return tuple(
        path.name
        for path in compose_values_files(
            use_k3d=use_k3d, backend=backend, serving=serving
        )
    )


def test_cli_values_stacks_name_the_composed_files():
    """An existing cluster gets values.prod.yaml alone whatever the host's
    device and serving mode; k3d layers both overlays on values.k3s.yaml."""
    assert _cli_values_stack("rocm", use_k3d=False, serving=LLM_SERVING_MODAL) == (
        "values.prod.yaml",
    )
    assert _cli_values_stack("rocm", use_k3d=True, serving=LLM_SERVING_MODAL) == (
        "values.k3s.yaml",
        "values.rocm.yaml",
        "values.modal-llm.yaml",
    )
    assert _cli_values_stack("cpu", use_k3d=True) == (
        "values.k3s.yaml",
        "values.cpu.yaml",
    )
    assert _cli_values_stack("mps", use_k3d=True) == ("values.k3s.yaml",)


def _composition_failure(values: tuple[str, ...], *set_args: str) -> str:
    """The message the configmap's validation refuses a composition with."""
    cmd = ["helm", "template", "cogniverse", str(CHART_PATH)]
    for values_file in values:
        cmd.extend(["-f", str(CHART_PATH / values_file)])
    cmd.extend(["--set", "runtime.qualityMonitor.tenantId=test-tenant"])
    for arg in set_args:
        cmd.extend(["--set", arg])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, (
        "chart rendered instead of refusing:\n" + result.stdout[:2000]
    )
    first_line = result.stderr.strip().splitlines()[0]
    prefix = "Error: execution error at (cogniverse/templates/configmap.yaml:2:4): "
    assert first_line.startswith(prefix), result.stderr
    return first_line.removeprefix(prefix)


def _video_description_endpoint(config: dict, profile: str) -> str:
    strategies = config["backend"]["profiles"][profile]["strategies"]
    assert strategies["description"]["class"] == "VLMDescriptionStrategy"
    return strategies["description"]["params"]["vlm_endpoint"]


def _served_model_arg(deployment: dict) -> str:
    """The model id ``vllm serve`` is given, in list or shell-script form."""
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    tokens = [
        token
        for part in [*container.get("command", []), *container.get("args", [])]
        for token in shlex.split(part.replace("\\\n", " "))
    ]
    return tokens[tokens.index("serve") + 1]


@pytest.mark.parametrize(
    "values",
    [
        _cli_values_stack("rocm", use_k3d=True, serving=LLM_SERVING_MODAL),
        # Operator helm composition; the CLI never layers overlays on prod.
        ("values.prod.yaml", "values.rocm.yaml", "values.modal-llm.yaml"),
    ],
    ids=["cli-k3d-rocm-modal", "operator-prod-rocm-modal"],
)
def test_rocm_composition_with_the_external_student_serves_its_selected_profile(
    values: tuple[str, ...],
):
    """The deployed composition: ROCm embedders in-cluster, student external."""
    docs = _render(*_PROD_SECRETS, values=values)
    config = _chart_config(docs)
    deployments = _inference_deployments(docs)
    profile = config["backend"]["profiles"][_SELECTED_VIDEO_PROFILE]

    assert resolve_default_profile(config) == _SELECTED_VIDEO_PROFILE
    assert profile["inference_services"] == {
        "embedding": "vllm_colpali",
        "transcription": "vllm_asr",
    }
    assert {key: _service_urls(docs)[key] for key in ("vllm_colpali", "vllm_asr")} == {
        "vllm_colpali": "http://cogniverse-vllm-colpali:8000",
        "vllm_asr": "http://cogniverse-vllm-asr:8000",
    }
    assert _VISUAL_SERVICES & set(deployments) == {"vllm_colpali", "vllm_asr"}
    assert {
        key: _served_model_arg(deployments[key]) for key in ("vllm_colpali", "vllm_asr")
    } == {
        "vllm_colpali": profile["embedding_model"],
        "vllm_asr": profile["strategies"]["transcription"]["params"]["model"],
    }
    assert (
        _video_description_endpoint(config, _SELECTED_VIDEO_PROFILE)
        == _values("values.modal-llm.yaml")["runtime"]["primaryLLM"]["apiBase"]
    )


@pytest.mark.parametrize(
    "values",
    [
        _cli_values_stack("cpu", use_k3d=True),
        _cli_values_stack(None, use_k3d=False),
        # Operator helm composition; the CLI never layers overlays on prod.
        ("values.prod.yaml", "values.cpu.yaml"),
    ],
    ids=["cli-k3d-cpu", "cli-prod", "operator-prod-cpu"],
)
def test_cpu_composition_omits_visual_ingestion(values: tuple[str, ...]):
    docs = _render(*_PROD_SECRETS, values=values)
    config = _chart_config(docs)

    assert config["backend"]["default_profiles"] == {}
    assert "active_video_profile" not in config
    assert resolve_default_profile(config) is None
    assert "vllm_colpali" not in _service_urls(docs)
    assert _VISUAL_SERVICES & set(_inference_deployments(docs)) == {"vllm_asr"}


_CUDA_STACKS = pytest.mark.parametrize(
    "values",
    [
        _cli_values_stack("cuda", use_k3d=True),
        # Operator helm composition; the CLI never layers overlays on prod.
        ("values.prod.yaml", "values.cuda.yaml"),
    ],
    ids=["cli-k3d-cuda", "operator-prod-cuda"],
)


@_CUDA_STACKS
def test_cuda_composition_without_a_student_endpoint_is_refused(
    values: tuple[str, ...],
):
    stderr = _composition_failure(values, *_PROD_SECRETS)

    assert stderr == (
        f"config.defaultProfiles.video={_SELECTED_VIDEO_PROFILE} describes frames "
        "with VLMDescriptionStrategy at http://cogniverse-vllm-llm-student:8000/v1, "
        "the in-cluster vllm_llm_student Service: set "
        "inference.vllm_llm_student.enabled=true or point runtime.primaryLLM.apiBase "
        "at a served endpoint"
    ), stderr


@_CUDA_STACKS
def test_cuda_composition_with_an_explicit_student_endpoint_renders(
    values: tuple[str, ...],
):
    docs = _render(
        *_PROD_SECRETS,
        f"runtime.primaryLLM.apiBase={_STUDENT_API_BASE}",
        values=values,
    )
    config = _chart_config(docs)
    deployments = _inference_deployments(docs)

    assert resolve_default_profile(config) == _SELECTED_VIDEO_PROFILE
    assert {key: _service_urls(docs)[key] for key in ("vllm_colpali", "vllm_asr")} == {
        "vllm_colpali": "http://cogniverse-vllm-colpali:8000",
        "vllm_asr": "http://cogniverse-vllm-asr:8000",
    }
    assert _VISUAL_SERVICES & set(deployments) == {"vllm_colpali", "vllm_asr"}
    assert [
        deployments[key]["spec"]["template"]["spec"]["nodeSelector"]
        for key in ("vllm_colpali", "vllm_asr")
    ] == [{"nvidia.com/gpu.present": "true"}] * 2
    assert (
        _video_description_endpoint(config, _SELECTED_VIDEO_PROFILE)
        == _STUDENT_API_BASE
    )


def test_fully_external_composition_renders_without_local_model_pods():
    """The CLI's existing-cluster composition with every selected key
    off-cluster and its ``--llm external`` overrides: no local Deployment
    serves any of them."""
    docs = _render(
        *_PROD_SECRETS,
        f"config.defaultProfiles.video={_SELECTED_VIDEO_PROFILE}",
        "inference.vllm_colpali.externalUrl=https://colpali.example.com",
        "inference.vllm_asr.enabled=false",
        "inference.vllm_asr.externalUrl=https://asr.example.com",
        f"runtime.primaryLLM.apiBase={_STUDENT_API_BASE}",
        "llm.engine=external",
        "llm.builtin.enabled=false",
        "llm.external.enabled=true",
        "llm.external.url=https://llm.example.com/v1",
        values=_cli_values_stack(None, use_k3d=False),
    )
    config = _chart_config(docs)

    assert resolve_default_profile(config) == _SELECTED_VIDEO_PROFILE
    assert {key: _service_urls(docs)[key] for key in ("vllm_colpali", "vllm_asr")} == {
        "vllm_colpali": "https://colpali.example.com",
        "vllm_asr": "https://asr.example.com",
    }
    assert _VISUAL_SERVICES & set(_inference_deployments(docs)) == set()
    assert (
        _video_description_endpoint(config, _SELECTED_VIDEO_PROFILE)
        == _STUDENT_API_BASE
    )


@pytest.mark.parametrize(
    ("set_args", "role", "key"),
    [
        ((), "embedding", "vllm_colpali"),
        (
            (
                "inference.vllm_colpali.enabled=true",
                "inference.vllm_asr.enabled=false",
            ),
            "transcription",
            "vllm_asr",
        ),
    ],
    ids=["colpali-disabled", "asr-disabled"],
)
def test_a_selected_profile_bound_to_an_undeployed_service_is_refused(
    set_args: tuple[str, ...], role: str, key: str
):
    stderr = _composition_failure(
        _cli_values_stack(None, use_k3d=False),
        *_PROD_SECRETS,
        f"config.defaultProfiles.video={_SELECTED_VIDEO_PROFILE}",
        f"runtime.primaryLLM.apiBase={_STUDENT_API_BASE}",
        *set_args,
    )

    assert stderr == (
        f"config.defaultProfiles.video={_SELECTED_VIDEO_PROFILE} binds "
        f"inference_services.{role} to {key}: set inference.{key}.enabled=true "
        f"or inference.{key}.externalUrl"
    ), stderr


def test_a_selected_profile_bound_to_an_undefined_service_is_refused():
    stderr = _composition_failure(("values.rocm.yaml",), "inference.vllm_colpali=null")

    assert stderr == (
        f"config.defaultProfiles.video={_SELECTED_VIDEO_PROFILE} binds "
        "inference_services.embedding to vllm_colpali, which is not a service "
        "under inference"
    ), stderr


def test_an_unknown_selected_profile_is_refused():
    stderr = _composition_failure(
        ("values.rocm.yaml",), "config.defaultProfiles.video=video_missing"
    )

    assert stderr == (
        "config.defaultProfiles.video=video_missing is not a profile in "
        "backend.profiles"
    ), stderr


def test_a_selected_vlm_profile_without_a_student_endpoint_is_refused():
    stderr = _composition_failure(
        ("values.rocm.yaml",), "inference.vllm_llm_student.enabled=false"
    )

    assert stderr == (
        f"config.defaultProfiles.video={_SELECTED_VIDEO_PROFILE} describes frames "
        "with VLMDescriptionStrategy at http://cogniverse-vllm-llm-student:8000/v1, "
        "the in-cluster vllm_llm_student Service: set "
        "inference.vllm_llm_student.enabled=true or point runtime.primaryLLM.apiBase "
        "at a served endpoint"
    ), stderr


@pytest.mark.parametrize(
    "api_base",
    [
        "http://cogniverse-vllm-llm-student:8000/v1",
        "http://cogniverse-vllm-llm-student:8000/v1/",
        "http://cogniverse-vllm-llm-student.default:8000/v1",
        "http://cogniverse-vllm-llm-student.default.svc:8000/v1",
        "http://cogniverse-vllm-llm-student.default.svc.cluster.local:8000/v1",
        "http://cogniverse-vllm-llm-student.default.svc.cluster.local.:8000/v1/",
        "HTTP://Cogniverse-VLLM-LLM-Student:8000/v1",
    ],
    ids=["short", "trailing-slash", "ns", "svc", "fqdn", "fqdn-root-dot", "case"],
)
def test_a_primary_llm_api_base_naming_the_undeployed_student_is_refused(
    api_base: str,
):
    """An apiBase is not an endpoint by being set: one naming the in-cluster
    student Service, in any spelling the cluster resolves, backs nothing
    while that service is disabled."""
    stderr = _composition_failure(
        ("values.rocm.yaml",),
        "inference.vllm_llm_student.enabled=false",
        f"runtime.primaryLLM.apiBase={api_base}",
    )

    assert stderr == (
        f"config.defaultProfiles.video={_SELECTED_VIDEO_PROFILE} describes frames "
        f"with VLMDescriptionStrategy at {api_base}, "
        "the in-cluster vllm_llm_student Service: set "
        "inference.vllm_llm_student.enabled=true or point runtime.primaryLLM.apiBase "
        "at a served endpoint"
    ), stderr


@pytest.mark.parametrize(
    "api_base",
    [
        "http://cogniverse-vllm-llm-student.example.com:8000/v1",
        "http://cogniverse-vllm-llm-student.other.svc.cluster.local:8000/v1",
        "http://cogniverse-vllm-llm-student:8001/v1",
    ],
    ids=["external-host", "other-namespace", "other-port"],
)
def test_an_api_base_naming_another_endpoint_renders_without_the_student(
    api_base: str,
):
    config = _chart_config(
        _render(
            "inference.vllm_llm_student.enabled=false",
            f"runtime.primaryLLM.apiBase={api_base}",
            values="values.rocm.yaml",
        )
    )

    assert _video_description_endpoint(config, _SELECTED_VIDEO_PROFILE) == api_base


def _schema_deployment_calls(docs: list[dict]) -> list[tuple[str, str]]:
    """(tenant, profile) for each deploy call the schema-deployment job makes."""
    (job,) = [
        d
        for d in docs
        if d.get("kind") == "Job"
        and d["metadata"]["name"] == "cogniverse-schema-deployment"
    ]
    script = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    return re.findall(
        r'/admin/profiles/([^/"]+)/deploy" \\\n.*\n\s*-d \'\{"tenant_id": "([^"]+)"',
        script,
    )


@pytest.mark.parametrize(
    ("values", "set_args", "expected"),
    [
        (_cli_values_stack("cpu", use_k3d=True), (), []),
        (
            _cli_values_stack("rocm", use_k3d=True, serving=LLM_SERVING_MODAL),
            (),
            [(_SELECTED_VIDEO_PROFILE, "default")],
        ),
        (
            _cli_values_stack("rocm", use_k3d=True, serving=LLM_SERVING_MODAL),
            ("config.defaultProfiles.video=video_colqwen_omni_mv_chunk_30s",),
            [("video_colqwen_omni_mv_chunk_30s", "default")],
        ),
    ],
    ids=["cli-k3d-cpu", "cli-k3d-rocm-modal", "cli-k3d-rocm-modal-chunk-profile"],
)
def test_schema_deployment_job_deploys_only_the_selected_video_profile(
    values: tuple[str, ...], set_args: tuple[str, ...], expected: list
):
    docs = _render(*set_args, values=values)

    assert _schema_deployment_calls(docs) == expected


def test_the_removed_schema_deployment_profiles_key_fails_the_render():
    """An override of the removed list would otherwise be ignored silently."""
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
            "--set",
            "initJobs.schemaDeployment.profiles[0]=image_colpali_mv",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout[:2000]
    first_line = result.stderr.strip().splitlines()[0]
    assert first_line.startswith(
        "Error: execution error at (cogniverse/templates/init-jobs.yaml:"
    ), result.stderr
    assert first_line.split("): ", 1)[1] == (
        "initJobs.schemaDeployment.profiles is removed: the schema-deployment "
        "job deploys config.defaultProfiles.video"
    ), result.stderr


_SCHEMA_JOB_TENANTS = ("acme", "beta")
_RUNTIME_URL_LINE = 'RUNTIME_URL="http://cogniverse-runtime:8000"'


def _schema_deployment_script() -> str:
    tenants = [
        f"config.tenants[{index}].id={tenant}"
        for index, tenant in enumerate(_SCHEMA_JOB_TENANTS)
    ]
    docs = _render(
        *tenants,
        values=_cli_values_stack("rocm", use_k3d=True, serving=LLM_SERVING_MODAL),
    )
    (job,) = [
        d
        for d in docs
        if d.get("kind") == "Job"
        and d["metadata"]["name"] == "cogniverse-schema-deployment"
    ]
    return job["spec"]["template"]["spec"]["containers"][0]["command"][-1]


# The deploy route's longest legitimate answer: the lease wait for a live
# holder, the heartbeat's cap on one legitimate lease body, and the
# convergence wait after the lease.
_LONGEST_DEPLOY_ANSWER_SECONDS = (
    DEFAULT_WAIT_SECONDS + MAX_TOTAL_HOLD_SECONDS + SCHEMA_CONVERGENCE_TIMEOUT_S
)

# One deploy's Vespa requests at their bounds: five conflict attempts of at
# most three requests (the schema manager's session create, prepare and
# activate; the backend's prepareandactivate is one), the backoff between
# attempts, and the schema listing before the package is built.
_DEPLOY_ATTEMPTS = 5
_REQUESTS_PER_ATTEMPT = 3
_BACKOFF_SECONDS = 0.5 + 1 + 2 + 4
_SCHEMA_LISTING_SECONDS = 20


def test_schema_deployment_request_outlasts_the_longest_deploy_answer():
    """curl must not give up on a deploy the runtime is still legitimately
    running: every deploy's --max-time is the longest answer the lease bounds
    allow, which covers one deploy's own worst-case Vespa requests."""
    script = _schema_deployment_script()
    timeouts = re.findall(r"--max-time (\d+) -X POST \"\$RUNTIME_URL/admin/", script)

    assert _LONGEST_DEPLOY_ANSWER_SECONDS == 9240
    assert [float(value) for value in timeouts] == [
        _LONGEST_DEPLOY_ANSWER_SECONDS
    ] * len(_SCHEMA_JOB_TENANTS), script
    assert MAX_TOTAL_HOLD_SECONDS >= (
        _DEPLOY_ATTEMPTS * _REQUESTS_PER_ATTEMPT * sum(DEPLOY_REQUEST_TIMEOUT_S)
        + _BACKOFF_SECONDS
        + _SCHEMA_LISTING_SECONDS
    )


def test_the_cli_helm_timeout_governs_the_schema_deployment_hook():
    """``cogniverse up`` waits helm's --timeout for the hook; it is the shorter
    bound, so a slow schema deploy fails the install rather than being cut
    off by curl."""
    default = inspect.signature(helm_install).parameters["timeout"].default

    assert default == "10m"
    assert 10 * 60 < _LONGEST_DEPLOY_ANSWER_SECONDS


def test_chart_validation_fires_on_the_sources_the_schema_job_tests_import():
    """The schema-deployment tests derive their bounds from runtime modules,
    so editing those modules must re-run this suite."""
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "chart-validation.yml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    sources = sorted(
        str(Path(sys.modules[name].__file__).resolve().relative_to(REPO_ROOT))
        for name in {
            "cogniverse_core.registries.schema_deploy_lease",
            SchemaDeploymentResponse.__module__,
            helm_install.__module__,
            "cogniverse_vespa.backend",
            "cogniverse_vespa.vespa_schema_manager",
        }
    )

    for trigger in ("push", "pull_request"):
        assert [
            path for path in sources if path not in triggers[trigger]["paths"]
        ] == []


@contextlib.contextmanager
def _deploy_endpoint(outcomes: dict[str, str | int]):
    """The runtime's deploy route answering each tenant with a fixed outcome:
    a ``deployment_status`` string, or an HTTP error status."""
    app = FastAPI()
    requests: list[tuple[str, dict]] = []

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy"}

    @app.post(
        "/admin/profiles/{profile_name}/deploy",
        response_model=SchemaDeploymentResponse,
    )
    def deploy(profile_name: str, body: dict = Body(...)) -> SchemaDeploymentResponse:
        requests.append((profile_name, body))
        outcome = outcomes[body["tenant_id"]]
        if isinstance(outcome, int):
            raise HTTPException(status_code=outcome, detail="deploy error")
        return SchemaDeploymentResponse(
            profile_name=profile_name,
            tenant_id=body["tenant_id"],
            schema_name="video_colpali_smol500_mv_frame",
            tenant_schema_name="",
            deployment_status=outcome,
            deployed_at="2026-09-24T00:00:00+00:00",
            error_message="lease wait timed out" if outcome == "failed" else None,
        )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "deploy endpoint did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        server.should_exit = True
        thread.join(timeout=30)


@pytest.mark.parametrize(
    ("outcomes", "exit_code", "attempted"),
    [
        ({"acme": "success", "beta": "already_deployed"}, 0, ["acme", "beta"]),
        ({"acme": "failed", "beta": "success"}, 1, ["acme"]),
        ({"acme": "success", "beta": "failed"}, 1, ["acme", "beta"]),
        ({"acme": 500, "beta": "success"}, 22, ["acme"]),
    ],
    ids=["deployed", "first-failed", "second-failed", "server-error"],
)
def test_schema_deployment_job_fails_unless_every_tenant_is_deployed(
    outcomes: dict, exit_code: int, attempted: list[str]
):
    """A ``failed`` deploy answers HTTP 200; the job must still fail so the
    Job's backoffLimit retries it instead of the hook reporting success."""
    script = _schema_deployment_script()
    assert script.count(_RUNTIME_URL_LINE) == 1, script

    with _deploy_endpoint(outcomes) as (url, requests):
        env = {
            name: value
            for name, value in os.environ.items()
            if name.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
        }
        result = subprocess.run(
            [
                "/bin/sh",
                "-c",
                script.replace(_RUNTIME_URL_LINE, f'RUNTIME_URL="{url}"'),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    assert result.returncode == exit_code, result.stdout + result.stderr
    assert requests == [
        (_SELECTED_VIDEO_PROFILE, {"tenant_id": tenant, "force": False})
        for tenant in attempted
    ]
    assert ("Schema deployment completed!" in result.stdout) is (exit_code == 0), (
        result.stdout
    )


def _is_rocm(dep: dict) -> bool:
    vols = dep["spec"]["template"]["spec"].get("volumes", [])
    return any(v.get("name") == "kfd" for v in vols)


def test_rocm_overlay_wires_tunableop_env_on_rocm_pods_only():
    """The ROCm overlay sets runtime.tunableOp, so every rocm-device
    inference pod that does not opt out gets PyTorch TunableOp pointed at a
    per-service results file inside the persistent model-cache mount. The
    PyLate sidecars opt out explicitly, and CPU sidecars in the same render
    (e.g. gliner) carry none."""
    deps = _inference_deployments(_render(values="values.rocm.yaml"))
    rocm_pods = [k for k, d in deps.items() if _is_rocm(d)]
    assert set(rocm_pods) == {
        "vllm_colpali",
        "vllm_asr",
        "vllm_llm_student",
        "vllm_llm_teacher",
        "denseon",
        "colbert_pylate",
        "code_colbert_pylate",
    }, rocm_pods
    for key in (
        "vllm_colpali",
        "vllm_asr",
        "vllm_llm_student",
        "vllm_llm_teacher",
        "denseon",
    ):
        env = _inference_env(deps, key)
        assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1", key
        assert env["PYTORCH_TUNABLEOP_TUNING"] == "1", key
        assert (
            env["PYTORCH_TUNABLEOP_FILENAME"]
            == f"/root/.cache/huggingface/tunableop_{key.replace('_', '-')}_%d.csv"
        ), key
    for key in ("colbert_pylate", "code_colbert_pylate"):
        assert not (set(_inference_env(deps, key)) & _TUNABLEOP_VARS), key
    for key, dep in deps.items():
        if not _is_rocm(dep):
            assert not (set(_inference_env(deps, key)) & _TUNABLEOP_VARS), key


def test_tunableop_env_absent_by_default():
    """Default (non-rocm) render carries no TunableOp env on any pod."""
    for key, dep in _inference_deployments(_render()).items():
        names = {
            e["name"]
            for e in dep["spec"]["template"]["spec"]["containers"][0].get("env", [])
        }
        assert not (names & _TUNABLEOP_VARS), key


def test_tunableop_requires_both_rocm_device_and_toggle():
    """Both conditions are necessary: a rocm pod with the toggle off, and a
    cpu pod with the toggle on, each carry no TunableOp env."""
    rocm_no_toggle = _inference_env(
        _inference_deployments(_render("inference.denseon.device=rocm")), "denseon"
    )
    assert not (set(rocm_no_toggle) & _TUNABLEOP_VARS)

    toggle_no_rocm = _inference_env(
        _inference_deployments(_render("runtime.tunableOp=true")), "denseon"
    )
    assert not (set(toggle_no_rocm) & _TUNABLEOP_VARS)


def test_cpu_overlay_keeps_tunableop_env_off_even_when_global_toggle_is_on():
    deps = _inference_deployments(
        _render("runtime.tunableOp=true", values="values.cpu.yaml")
    )

    for key, deployment in deps.items():
        names = {
            e["name"]
            for e in deployment["spec"]["template"]["spec"]["containers"][0].get(
                "env", []
            )
        }
        assert not (names & _TUNABLEOP_VARS), key


# Chart-served vLLM services whose artifact is pinned in INFERENCE_SERVICE_SPECS.
# Helm cannot read the Python spec map, so values.yaml holds a second copy of
# each model id and sha; test_chart_model_pins_match_inference_service_specs
# fails as soon as the two copies disagree.
REVISION_PINNED_SERVICES = ("vllm_colpali", "vllm_llm_student", "denseon", "vllm_asr")


def test_chart_model_pins_match_inference_service_specs():
    """Both copies of every pinned model id and revision agree."""
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text())["inference"]
    assert {
        name: (values[name]["model"], values[name]["revision"])
        for name in REVISION_PINNED_SERVICES
    } == {
        name: (
            INFERENCE_SERVICE_SPECS[name].model_id,
            INFERENCE_SERVICE_SPECS[name].model_revision,
        )
        for name in REVISION_PINNED_SERVICES
    }


def test_vllm_token_embed_serve_args_pin_the_revision():
    """vllm_colpali serves the pinned ColQwen3 artifact, so /v1/models reports
    the revision the identity gate demands."""
    docs = _render("inference.vllm_colpali.enabled=true")
    c = _inference_deployments(docs)["vllm_colpali"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert c["args"][:8] == [
        "serve",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "--revision",
        "bf790bd8780b098b86453444632a184bb770be1a",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_vllm_chat_serve_args_pin_the_revision():
    """vllm_llm_student serves the pinned Gemma artifact."""
    docs = _render("inference.vllm_llm_student.enabled=true")
    c = _inference_deployments(docs)["vllm_llm_student"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert c["args"][:8] == [
        "serve",
        "google/gemma-4-e4b-it",
        "--revision",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_vllm_embed_serve_args_pin_the_revision():
    """denseon serves the pinned DenseOn artifact; the revision precedes the
    engine's own conversion flags."""
    c = _inference_deployments(_render())["denseon"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert c["args"] == [
        "serve",
        "lightonai/DenseOn",
        "--revision",
        "cb9947ebccb33862d24e3c7ca2edb25e51acd887",
        "--convert",
        "embed",
        "--dtype",
        "float32",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_vllm_transcription_serve_script_pins_the_revision():
    """The transcription engine renders a shell script rather than an argv
    list; the pinned whisper revision lands on the exec'd serve line."""
    docs = _render("inference.vllm_asr.enabled=true")
    c = _inference_deployments(docs)["vllm_asr"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert (
        "exec vllm serve 'openai/whisper-large-v3-turbo' \\\n"
        "  --host 0.0.0.0 --port 8000 \\\n"
        "  --revision '41f01f3fe87f28c78e2fbf8b568835947dd65ed9' \\\n"
    ) in "".join(c["args"])


def test_service_without_a_pinned_revision_renders_no_revision_flag():
    """The chart values keep the teacher unpinned even after the spec exists,
    so the in-cluster Deployment still renders without a revision flag."""
    docs = _render("inference.vllm_llm_teacher.enabled=true")
    c = _inference_deployments(docs)["vllm_llm_teacher"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert "vllm_llm_teacher" in INFERENCE_SERVICE_SPECS
    assert "--revision" not in c["args"]
    assert c["args"][:2] == ["serve", "Qwen/Qwen3-14B-AWQ"]


def test_cpu_overlay_swaps_whisper_without_inheriting_the_turbo_revision():
    """The CPU overlay serves whisper-tiny, whose repo has no such sha, so it
    carries the swapped model and no revision at all."""
    docs = _render("inference.vllm_asr.enabled=true", values="values.cpu.yaml")
    c = _inference_deployments(docs)["vllm_asr"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    script = "".join(c["args"])
    assert "exec vllm serve 'openai/whisper-tiny' \\\n" in script
    assert "--revision" not in script
    assert "41f01f3fe87f28c78e2fbf8b568835947dd65ed9" not in script


def _runtime_container(docs: list[dict]) -> dict:
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        if doc["metadata"]["labels"].get("app.kubernetes.io/component") == "runtime":
            return doc["spec"]["template"]["spec"]["containers"][0]
    raise AssertionError("no runtime Deployment rendered")


def _env(container: dict, name: str) -> str:
    for entry in container["env"]:
        if entry["name"] == name:
            return entry["value"]
    raise AssertionError(f"{name} not in rendered runtime env")


def test_k3s_rocm_runtime_receives_the_clap_embed_url():
    """audio_analysis_agent encodes its query with CLAP. The runtime image
    carries no torch, so the acoustic path works only through the sidecar,
    and the sidecar URL reaches the runtime only when clap_embed is enabled."""
    docs = _render(values=("values.k3s.yaml", "values.rocm.yaml"))
    urls = json.loads(_env(_runtime_container(docs), "INFERENCE_SERVICE_URLS"))

    assert urls["clap_embed"] == "http://cogniverse-clap-embed:8000"


def test_k3s_rocm_clap_embed_pod_is_cpu_only():
    """clap_embed must not draw on the GPU pool the vLLM pods share — its
    fractions are budgeted exactly and an unbudgeted GPU claim freezes the host."""
    docs = _render(values=("values.k3s.yaml", "values.rocm.yaml"))
    container = _inference_deployments(docs)["clap_embed"]["spec"]["template"]["spec"][
        "containers"
    ][0]

    resources = container["resources"]
    assert resources == {
        "limits": {"cpu": "2", "memory": "6Gi"},
        # Memory request equals the limit so the scheduler reserves what the
        # pod may take; cpu stays burstable because exceeding it throttles.
        "requests": {"cpu": "500m", "memory": "6Gi"},
    }
    assert {e["name"]: e["value"] for e in container["env"]} == {
        "CLAP_EMBED_MODEL": "laion/clap-htsat-unfused",
        "CLAP_EMBED_SAMPLE_RATE": "48000",
        "HF_HOME": "/root/.cache/huggingface",
        "HOST": "0.0.0.0",
        "PORT": "8000",
    }
    assert "--gpu-memory-utilization" not in container.get("args", [])


def _ingestor_env(docs: list[dict]) -> dict[str, str]:
    for d in docs:
        if (
            d.get("kind") == "Deployment"
            and d.get("metadata", {}).get("name") == "cogniverse-ingestor"
        ):
            container = d["spec"]["template"]["spec"]["containers"][0]
            return {e["name"]: e.get("value") for e in container.get("env", [])}
    raise AssertionError("ingestor Deployment not found")


_COLBERT_MODAL_URL = "https://amit--cogniverse-colbert-pylate.modal.run"


def test_external_url_replaces_the_cluster_internal_url_in_both_url_maps():
    """Strategy B: a Modal-hosted service keeps its key in the URL map but
    resolves to the external endpoint, for the runtime and the ingestor."""
    docs = _render(f"inference.colbert_pylate.externalUrl={_COLBERT_MODAL_URL}")
    expected = {
        "colbert_pylate": _COLBERT_MODAL_URL,
        "denseon": "http://cogniverse-denseon:8000",
        "gliner": "http://cogniverse-gliner:8080",
        "vllm_asr": "http://cogniverse-vllm-asr:8000",
        "vllm_llm_teacher": "http://cogniverse-vllm-llm-teacher:8000",
    }
    assert _service_urls(docs) == expected
    assert json.loads(_ingestor_env(docs)["INFERENCE_SERVICE_URLS"]) == expected


def test_external_url_skips_the_local_deployment_and_service():
    docs = _render(f"inference.colbert_pylate.externalUrl={_COLBERT_MODAL_URL}")
    assert set(_inference_deployments(docs)) == {
        "denseon",
        "gliner",
        "vllm_asr",
        "vllm_llm_teacher",
    }
    assert set(_inference_services(docs)) == {
        "denseon",
        "gliner",
        "vllm_asr",
        "vllm_llm_teacher",
    }


def test_no_external_url_renders_the_default_env_byte_identical():
    """Strategy A: without externalUrl the rendered env value is exactly the
    pre-externalUrl form, byte for byte, in both consumers."""
    docs = _render()
    expected = (
        "{\n"
        '  "colbert_pylate": "http://cogniverse-colbert-pylate:8000",\n'
        '  "denseon": "http://cogniverse-denseon:8000",\n'
        '  "gliner": "http://cogniverse-gliner:8080",\n'
        '  "vllm_asr": "http://cogniverse-vllm-asr:8000",\n'
        '  "vllm_llm_teacher": "http://cogniverse-vllm-llm-teacher:8000"}'
    )
    assert _runtime_env(docs)["INFERENCE_SERVICE_URLS"] == expected
    assert _ingestor_env(docs)["INFERENCE_SERVICE_URLS"] == expected


def test_external_url_skips_the_model_cache_pvc():
    docs = _render(
        "hfCache.persistence.enabled=true",
        f"inference.colbert_pylate.externalUrl={_COLBERT_MODAL_URL}",
    )
    pvcs = {
        d["metadata"]["name"] for d in docs if d.get("kind") == "PersistentVolumeClaim"
    }
    assert "cogniverse-colbert-pylate-model-cache" not in pvcs
    assert "cogniverse-denseon-model-cache" in pvcs


def test_external_url_predecessor_does_not_gate_its_successor():
    """In the rocm startup chain denseon waits on vllm_asr; a Modal-hosted
    vllm_asr deploys no local Service, so denseon must start ungated instead
    of waiting on a /health that can never answer."""
    docs = _render(
        "inference.vllm_asr.externalUrl=https://amit--cogniverse-vllm-asr.modal.run",
        values="values.rocm.yaml",
    )
    deps = _inference_deployments(docs)
    assert "vllm_asr" not in deps
    inits = deps["denseon"]["spec"]["template"]["spec"].get("initContainers", [])
    assert [c["name"] for c in inits if c["name"] == "startup-gate"] == []


def test_external_url_on_denseon_redirects_the_semantic_embed_url():
    """COGNIVERSE_SEMANTIC_EMBED_URL is derived from the same service; a
    Modal-hosted denseon must not leave it pointing at the skipped pod."""
    modal_url = "https://amit--cogniverse-denseon.modal.run"
    docs = _render(f"inference.denseon.externalUrl={modal_url}")
    assert _runtime_env(docs)["COGNIVERSE_SEMANTIC_EMBED_URL"] == modal_url


def test_teacher_endpoint_uses_the_in_cluster_url_when_not_external():
    docs = _render("inference.vllm_llm_teacher.enabled=true")

    assert _service_urls(docs)["vllm_llm_teacher"] == (
        "http://cogniverse-vllm-llm-teacher:8000"
    )
    assert _teacher_api_base(docs) == ("http://cogniverse-vllm-llm-teacher:8000/v1")


def test_teacher_endpoint_uses_the_modal_url_when_external():
    modal_url = "https://amit--cogniverse-vllm-llm-teacher.modal.run"
    docs = _render(
        "inference.vllm_llm_teacher.enabled=true",
        f"inference.vllm_llm_teacher.externalUrl={modal_url}",
    )

    assert _service_urls(docs)["vllm_llm_teacher"] == modal_url
    assert _teacher_api_base(docs) == f"{modal_url}/v1"


@pytest.mark.parametrize(
    "external_url",
    (
        "https://amit--cogniverse-vllm-llm-teacher.modal.run/v1",
        "https://amit--cogniverse-vllm-llm-teacher.modal.run/",
    ),
)
def test_teacher_external_url_rejects_trailing_slash_or_v1_suffix(external_url):
    expected = (
        "inference.vllm_llm_teacher.externalUrl must be the service root URL "
        "(no trailing / or /v1)"
    )

    with pytest.raises(AssertionError, match=re.escape(expected)):
        _render(
            "inference.vllm_llm_teacher.enabled=true",
            f"inference.vllm_llm_teacher.externalUrl={external_url}",
        )


def test_external_url_on_the_student_llm_fails_the_render():
    """The primary/teacher LLM endpoints are derived by their own helpers
    with their own override (runtime.primaryLLM.*); externalUrl on the LLM
    services would skip the pod while LLM_ENDPOINT still points at it."""
    with pytest.raises(AssertionError, match="runtime.primaryLLM"):
        _render(
            "inference.vllm_llm_student.enabled=true",
            "inference.vllm_llm_student.externalUrl=https://amit--llm.modal.run",
        )
    # Invalid even while the service is disabled: the key would silently
    # start lying the moment the service is enabled.
    with pytest.raises(AssertionError, match="runtime.primaryLLM"):
        _render("inference.vllm_llm_student.externalUrl=https://amit--llm.modal.run")


def test_every_inference_service_can_be_pointed_off_cluster():
    """A service with no off-cluster hook cannot be moved to Modal at all.

    Every service uses ``externalUrl`` except the student, whose endpoint is
    derived from ``runtime.primaryLLM.apiBase`` — the chart fails the render if
    ``inference.vllm_llm_student.externalUrl`` is set, so it must not have one.
    """
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text())["inference"]
    services = {
        name: block
        for name, block in values.items()
        if isinstance(block, dict) and "enabled" in block
    }

    without_hook = sorted(n for n, b in services.items() if "externalUrl" not in b)

    assert without_hook == ["vllm_llm_student"], without_hook


def test_external_url_survives_disabling_the_local_pod():
    """Moving a service to Modal means: point the client there AND stop running
    the pod. ``enabled`` governs the pod, ``externalUrl`` governs the client, so
    disabling one must not silently discard the other.
    """
    docs = _render(
        "inference.vllm_llm_teacher.enabled=false",
        "inference.vllm_llm_teacher.externalUrl=https://teacher.example.modal.run",
    )

    assert "cogniverse-vllm-llm-teacher" not in _inference_deployments(docs)
    assert _teacher_api_base(docs) == "https://teacher.example.modal.run/v1"


def test_disabled_external_service_still_consumes_the_bearer_secret():
    docs = _render(
        "inference.vllm_llm_teacher.enabled=false",
        "inference.vllm_llm_teacher.externalUrl=https://teacher.example.modal.run",
    )
    runtime = next(
        d
        for d in docs
        if d.get("kind") == "Deployment"
        and d["metadata"]["name"] == "cogniverse-runtime"
    )
    container = next(
        c
        for c in runtime["spec"]["template"]["spec"]["containers"]
        if c["name"] == "runtime"
    )
    env = {e["name"]: e for e in container["env"]}

    assert env["COGNIVERSE_INFERENCE_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "cogniverse-inference-api-key",
        "key": "COGNIVERSE_INFERENCE_API_KEY",
        "optional": False,
    }


class TestLlmServingOverlay:
    """Where the chat LLMs are served is orthogonal to the local GPU vendor.

    values.rocm.yaml means 'this host has AMD GPUs'. Folding Modal endpoints
    into it would hand Modal to every ROCm deployment, including ones with no
    Modal account. The redirection is a separate, opt-in, backend-agnostic
    overlay composed on top.
    """

    ROCM = "values.rocm.yaml"
    MODAL = "values.modal-llm.yaml"

    def _runtime_env(self, docs: list[dict]) -> dict:
        runtime = next(
            d
            for d in docs
            if d.get("kind") == "Deployment"
            and d["metadata"]["name"] == "cogniverse-runtime"
        )
        container = next(
            c
            for c in runtime["spec"]["template"]["spec"]["containers"]
            if c["name"] == "runtime"
        )
        return {e["name"]: e for e in container["env"]}

    def test_device_overlay_alone_keeps_the_chat_models_local(self):
        docs = _render(values=self.ROCM)
        env = self._runtime_env(docs)

        deployments = _inference_deployments(docs)
        assert "vllm_llm_student" in deployments
        assert "vllm_llm_teacher" in deployments
        assert (
            env["LLM_ENDPOINT"]["value"] == "http://cogniverse-vllm-llm-student:8000/v1"
        )
        assert (
            env["COGNIVERSE_INFERENCE_API_KEY"]["value"] == "placeholder-no-auth-needed"
        )

    def test_serving_overlay_moves_both_chat_models_off_cluster(self):
        docs = _render(values=(self.ROCM, self.MODAL))
        env = self._runtime_env(docs)

        deployments = _inference_deployments(docs)
        assert "vllm_llm_student" not in deployments
        assert "vllm_llm_teacher" not in deployments
        assert env["LLM_ENDPOINT"]["value"] == (
            "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
        )
        assert _teacher_api_base(docs) == (
            "https://amit-jain--cogniverse-vllm-llm-teacher-inference.modal.run/v1"
        )
        assert env["COGNIVERSE_INFERENCE_API_KEY"]["valueFrom"]["secretKeyRef"] == {
            "name": "cogniverse-inference-api-key",
            "key": "COGNIVERSE_INFERENCE_API_KEY",
            "optional": False,
        }

    def test_serving_overlay_leaves_the_embedders_on_the_local_gpu(self):
        """It redirects the chat models only; the embedders stay where the
        device overlay put them."""
        local = _inference_deployments(_render(values=self.ROCM))
        composed = _inference_deployments(_render(values=(self.ROCM, self.MODAL)))

        assert sorted(set(local) - set(composed)) == [
            "vllm_llm_student",
            "vllm_llm_teacher",
        ]
        assert sorted(set(composed) - set(local)) == []


# Every engine ``inference.<svc>.engine`` accepts, mapped to the container
# name its branch renders. ``fastapi`` names the container after the service
# key; the rest name it after the engine. Anything outside this map is
# refused at render time rather than served as a plain vLLM pod.
_ENGINE_CONTAINER_NAMES = {
    "fastapi": "denseon",
    "gliner": "gliner",
    "pylate": "pylate",
    "vllm": "vllm",
    "vllm_chat": "vllm-chat",
    "vllm_embed": "vllm-embed",
    "vllm_token_embed": "vllm-token-embed",
    "vllm_transcription": "vllm-transcription",
}

# The service whose engine the refusal and branch tests override. Enabled by
# every profile, and its key kebabcases to a name no engine branch produces,
# so the ``fastapi`` container name cannot be confused with an engine's.
_ENGINE_CARRIER = "denseon"


def _render_failure(*set_args: str) -> str:
    """Stderr of a render the chart must refuse."""
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for arg in set_args:
        cmd.extend(["--set", arg])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, (
        "chart rendered instead of refusing:\n" + result.stdout[:2000]
    )
    return result.stderr


def _shipped_inference_engines() -> dict[str, str]:
    """``{service key: engine}`` over every values file the chart ships."""
    engines: dict[str, str] = {}
    for values_file in sorted(CHART_PATH.glob("values*.yaml")):
        inference = yaml.safe_load(values_file.read_text()).get("inference") or {}
        for key, cfg in inference.items():
            if isinstance(cfg, dict) and "engine" in cfg:
                engines[key] = cfg["engine"]
    return engines


def test_each_engine_renders_the_container_its_branch_names():
    for engine, container_name in _ENGINE_CONTAINER_NAMES.items():
        deps = _inference_deployments(
            _render(f"inference.{_ENGINE_CARRIER}.engine={engine}")
        )
        containers = deps[_ENGINE_CARRIER]["spec"]["template"]["spec"]["containers"]
        assert [c["name"] for c in containers] == [container_name], (
            f"engine {engine!r} rendered the wrong container"
        )


def test_every_engine_the_shipped_values_select_is_one_the_chart_implements():
    """A shipped profile may only name an engine that has a branch.

    ``vllm`` is the default and the one implemented engine no shipped service
    selects by name, so it is the only member of the map that is absent here.
    """
    shipped = _shipped_inference_engines()

    assert shipped == {
        "clap_embed": "fastapi",
        "code_colbert_pylate": "pylate",
        "colbert_pylate": "pylate",
        "denseon": "vllm_embed",
        "face_embed": "fastapi",
        "gliner": "gliner",
        "video_embed": "fastapi",
        "vllm_asr": "vllm_transcription",
        "vllm_colpali": "vllm_token_embed",
        "vllm_llm_student": "vllm_chat",
        "vllm_llm_teacher": "vllm_chat",
    }
    assert set(shipped.values()) == set(_ENGINE_CONTAINER_NAMES) - {"vllm"}


def test_the_colpali_native_engine_is_refused():
    """ColPali is served by ``vllm_colpali`` on the ``vllm_token_embed``
    engine. Nothing in the image tooling produces a standalone ColPali sidecar
    image, so the name must fail the render rather than schedule a pod that can
    only ImagePullBackOff."""
    stderr = _render_failure(f"inference.{_ENGINE_CARRIER}.engine=colpali_native")

    assert "colpali_native" in stderr
    assert f"inference.{_ENGINE_CARRIER}.engine" in stderr


def test_an_unrecognised_engine_is_refused_rather_than_served_as_plain_vllm():
    """Generic over the engine name: a typo must fail the render rather than
    quietly serve a plain vLLM pod from whatever image the service names."""
    stderr = _render_failure(f"inference.{_ENGINE_CARRIER}.engine=vllm_token_embeb")

    assert "vllm_token_embeb" in stderr
    assert f"inference.{_ENGINE_CARRIER}.engine" in stderr


def test_shipped_profiles_render_exactly_these_inference_containers():
    """The engine each shipped profile selects, read off the rendered pods."""
    per_profile = {
        profile: {
            key: dep["spec"]["template"]["spec"]["containers"][0]["name"]
            for key, dep in _inference_deployments(_render(values=values)).items()
        }
        for profile, values in {
            "default": (),
            "k3s": ("values.k3s.yaml",),
            "rocm": ("values.rocm.yaml",),
            "modal-llm": ("values.rocm.yaml", "values.modal-llm.yaml"),
        }.items()
    }
    per_profile["devMode"] = {
        key: dep["spec"]["template"]["spec"]["containers"][0]["name"]
        for key, dep in _inference_deployments(
            _render("devMode.enabled=true", "devMode.hostPath=/cogniverse-src")
        ).items()
    }

    assert per_profile == {
        "default": {
            "colbert_pylate": "pylate",
            "denseon": "vllm-embed",
            "gliner": "gliner",
            "vllm_asr": "vllm-transcription",
            "vllm_llm_teacher": "vllm-chat",
        },
        "devMode": {
            "colbert_pylate": "pylate",
            "denseon": "vllm-embed",
            "gliner": "gliner",
            "vllm_asr": "vllm-transcription",
            "vllm_llm_teacher": "vllm-chat",
        },
        "k3s": {
            "clap_embed": "clap-embed",
            "colbert_pylate": "pylate",
            "denseon": "vllm-embed",
            "gliner": "gliner",
            "video_embed": "video-embed",
            "vllm_asr": "vllm-transcription",
            "vllm_llm_teacher": "vllm-chat",
        },
        "modal-llm": {
            "code_colbert_pylate": "pylate",
            "colbert_pylate": "pylate",
            "denseon": "vllm-embed",
            "gliner": "gliner",
            "vllm_asr": "vllm-transcription",
            "vllm_colpali": "vllm-token-embed",
        },
        "rocm": {
            "code_colbert_pylate": "pylate",
            "colbert_pylate": "pylate",
            "denseon": "vllm-embed",
            "gliner": "gliner",
            "vllm_asr": "vllm-transcription",
            "vllm_colpali": "vllm-token-embed",
            "vllm_llm_student": "vllm-chat",
            "vllm_llm_teacher": "vllm-chat",
        },
    }
