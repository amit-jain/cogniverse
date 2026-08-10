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

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from cogniverse_cli.modal_inference_config import INFERENCE_SERVICE_SPECS

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*set_args: str, values: str | None = None) -> list[dict]:
    cmd = ["helm", "template", "cogniverse", str(CHART_PATH)]
    if values is not None:
        cmd.extend(["-f", str(CHART_PATH / values)])
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


def test_default_runs_colbert_pylate_and_denseon_services():
    """Default-enabled inference services: colbert_pylate (LateOn text
    multi-vector), denseon (DenseOn dense single-vector), gliner
    (zero-shot NER), and vllm_asr (Whisper transcription). Mem0 needs
    denseon for memory embeddings, the slim runtime image excludes
    torch+gliner, and the default video-ingestion profiles hard-require
    vllm_asr for transcription, so all four ship enabled by default."""
    deps = _inference_deployments(_render())
    assert set(deps.keys()) == {"colbert_pylate", "denseon", "gliner", "vllm_asr"}
    assert deps["colbert_pylate"]["metadata"]["name"] == "cogniverse-colbert-pylate"
    assert deps["denseon"]["metadata"]["name"] == "cogniverse-denseon"
    assert deps["gliner"]["metadata"]["name"] == "cogniverse-gliner"
    assert deps["vllm_asr"]["metadata"]["name"] == "cogniverse-vllm-asr"


def test_default_gliner_deployment_uses_the_pinned_production_model():
    docs = _render()
    deps = _inference_deployments(docs)

    assert _inference_env(deps, "gliner")["MODEL_NAME"] == ("urchade/gliner_large-v2.1")


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
        "inference.videoprism_jax.enabled=true",
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
        "inference.videoprism_jax.enabled=true",
        "inference.vllm_llm_student.enabled=true",
        "inference.clap_embed.enabled=true",
        "inference.face_embed.enabled=true",
        values="values.k3s.yaml",
    )
    services = _inference_services(docs)
    expected_ports = {
        "vllm_colpali": 29001,
        "colbert_pylate": 29002,
        "videoprism_jax": 29003,
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
    """Strip the deploy-specific VLM description fields before comparing. The
    chart's config.json is ``tpl``-rendered: ``vlm_endpoint`` is injected as the
    in-cluster vLLM URL and ``auto_start`` is false (the cluster runs the VLM as
    its own pod), while local uses an empty endpoint + ``auto_start`` true
    (auto-start the local sidecar). Those two fields differ by design; every
    other field (models, inference_services, all other strategies) stays strict,
    so real profile/model drift is still caught."""
    normalized = copy.deepcopy(profiles)
    for profile in normalized.values():
        params = (profile.get("strategies", {}).get("description", {}) or {}).get(
            "params", {}
        )
        params.pop("vlm_endpoint", None)
        params.pop("auto_start", None)
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


def _is_rocm(dep: dict) -> bool:
    vols = dep["spec"]["template"]["spec"].get("volumes", [])
    return any(v.get("name") == "kfd" for v in vols)


def test_rocm_overlay_wires_tunableop_env_on_rocm_pods_only():
    """The ROCm overlay sets runtime.tunableOp, so every rocm-device
    inference pod gets PyTorch TunableOp pointed at a per-service results
    file inside the persistent model-cache mount. CPU sidecars in the same
    render (e.g. gliner) carry none."""
    deps = _inference_deployments(_render(values="values.rocm.yaml"))
    rocm_pods = [k for k, d in deps.items() if _is_rocm(d)]
    assert set(rocm_pods) >= {"vllm_colpali", "denseon", "colbert_pylate"}, rocm_pods
    for key in rocm_pods:
        env = _inference_env(deps, key)
        assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1", key
        assert env["PYTORCH_TUNABLEOP_TUNING"] == "1", key
        assert (
            env["PYTORCH_TUNABLEOP_FILENAME"]
            == f"/root/.cache/huggingface/tunableop_{key.replace('_', '-')}_%d.csv"
        ), key
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
    """vllm_llm_teacher has no entry in the spec map and no pin in values, so
    the serve args carry no revision rather than an empty one."""
    docs = _render("inference.vllm_llm_teacher.enabled=true")
    c = _inference_deployments(docs)["vllm_llm_teacher"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert "vllm_llm_teacher" not in INFERENCE_SERVICE_SPECS
    assert "--revision" not in c["args"]
    assert c["args"][:2] == ["serve", "cyankiwi/Qwen3.6-27B-AWQ-INT4"]


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
