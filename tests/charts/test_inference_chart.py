"""Chart tests for the generic inference services.

The chart supports N parallel inference services under ``inference`` — each
entry deploys one pod. Keys are logical tags (e.g. ``colbert_pylate`` for
the LateOn text multi-vector pod, ``denseon`` for the DenseOn dense pod).
Each service has an ``engine`` that selects the container template
(``vllm_token_embed`` for per-token multi-vector, ``vllm_embed`` for dense
single-vector, ``vllm_chat``, ``vllm_transcription``, ``gliner``,
``fastapi``, …) and a ``type`` (``multi_vector`` or ``single_vector``).

The runtime receives one ``INFERENCE_SERVICE_URLS`` JSON env var containing
{service_key: url} for every enabled service. Profiles pick a service by key.
"""

import json
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


def _render(*set_args: str) -> list[dict]:
    cmd = ["helm", "template", "cogniverse", str(CHART_PATH)]
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


def test_default_colbert_pylate_serves_lateon_via_vllm():
    """Default colbert_pylate service serves LateOn via vLLM's
    token-embed runner with the ColBERTModernBertModel arch override."""
    deps = _inference_deployments(_render())
    container = deps["colbert_pylate"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].startswith("vllm/vllm-openai")
    args = container["args"]
    assert "lightonai/LateOn" in args
    assert "--hf-overrides" in args
    joined = " ".join(args)
    assert "ColBERTModernBertModel" in joined
    # The hf-overrides value is one valid-JSON arg, not split across list items.
    override = args[args.index("--hf-overrides") + 1]
    assert json.loads(override) == {"architectures": ["ColBERTModernBertModel"]}
    env = {e["name"]: e.get("value") for e in container.get("env", [])}
    assert "MODEL_NAME" not in env


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
    colbert_args = deps["colbert_pylate"]["spec"]["template"]["spec"]["containers"][0][
        "args"
    ]
    code_args = deps["code_colbert_pylate"]["spec"]["template"]["spec"]["containers"][
        0
    ]["args"]
    assert "lightonai/Reason-ModernColBERT" in colbert_args
    assert "lightonai/LateOn" not in colbert_args
    assert "lightonai/LateOn-Code-edge" in code_args
    assert "lightonai/Reason-ModernColBERT" not in code_args


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


def _rendered_chart_config() -> dict:
    """Parse the config.json the chart renders into the runtime ConfigMap."""
    docs = _render("runtime.qualityMonitor.tenantId=test-tenant")
    cm = next(
        d
        for d in docs
        if d.get("kind") == "ConfigMap" and "config.json" in (d.get("data") or {})
    )
    return json.loads(cm["data"]["config.json"])


def test_chart_config_profiles_match_local_config():
    """The chart-bundled config.json (what the deployed runtime reads) must
    carry the SAME backend.profiles as configs/config.json (what local/tests
    use). Drift here ships a stale model to the cluster and crashes the
    runtime's validate_inference_services on startup — the colpali-v1.3 vs
    Tomoro mismatch this test guards against."""
    local = json.loads((REPO_ROOT / "configs" / "config.json").read_text())
    chart = _rendered_chart_config()
    assert chart["backend"]["profiles"] == local["backend"]["profiles"]


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
