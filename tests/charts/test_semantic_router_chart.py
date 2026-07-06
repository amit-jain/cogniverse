"""Chart tests for the semantic-router upstream wiring.

The router is a transparent proxy in front of the SAME LLM the runtime calls,
so its backend endpoint must track ``cogniverse.primaryLLMEndpoint`` for every
engine — not a separate engine switch that can drift from it. A prior bug in
``srUpstreamHost``/``srUpstreamPort`` pointed the router at the non-existent
``-llm`` service for the vllm engine (whose LLM actually lives on the
``-vllm-llm-student`` service), so completions would never reach a backend.
These render-time assertions pin the endpoint per engine so that can't regress.
"""

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
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        "--set",
        "semanticRouter.enabled=true",
    ]
    for arg in set_args:
        cmd.extend(["--set", arg])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n{result.stderr}"
        )
    return [d for d in yaml.safe_load_all(result.stdout) if d is not None]


def _sr_config(docs: list[dict]) -> dict:
    for d in docs:
        if (
            d.get("kind") == "ConfigMap"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-config"
        ):
            return yaml.safe_load(d["data"]["config.yaml"])
    raise AssertionError("semantic-router-config ConfigMap not rendered")


def _backend_endpoints(cfg: dict) -> set[str]:
    endpoints: set[str] = set()
    for model in cfg["providers"]["models"]:
        for ref in model["backend_refs"]:
            endpoints.add(ref["endpoint"])
    return endpoints


def _envoy_upstream(docs: list[dict]) -> str:
    """host:port of the ``llm_upstream`` cluster in the rendered Envoy config."""
    for d in docs:
        if (
            d.get("kind") == "ConfigMap"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-envoy"
        ):
            envoy = yaml.safe_load(d["data"]["envoy.yaml"])
            for cluster in envoy["static_resources"]["clusters"]:
                if cluster["name"] == "llm_upstream":
                    sock = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][
                        0
                    ]["endpoint"]["address"]["socket_address"]
                    return f"{sock['address']}:{sock['port_value']}"
    raise AssertionError("llm_upstream cluster not found in envoy config")


def test_vllm_engine_routes_to_student_service():
    cfg = _sr_config(_render("llm.engine=vllm"))
    assert _backend_endpoints(cfg) == {"cogniverse-vllm-llm-student:8000"}


def test_ollama_engine_routes_to_llm_service():
    cfg = _sr_config(_render("llm.engine=ollama"))
    assert _backend_endpoints(cfg) == {"cogniverse-llm:11434"}


def test_external_engine_parses_configured_url():
    cfg = _sr_config(
        _render("llm.engine=external", "llm.external.url=http://my-llm:9000/v1")
    )
    assert _backend_endpoints(cfg) == {"my-llm:9000"}


def test_envoy_upstream_matches_sr_backend_for_vllm():
    docs = _render("llm.engine=vllm")
    assert _envoy_upstream(docs) == "cogniverse-vllm-llm-student:8000"
    assert _backend_endpoints(_sr_config(docs)) == {"cogniverse-vllm-llm-student:8000"}


def test_provider_model_id_is_bare_served_model_for_vllm():
    cfg = _sr_config(_render("llm.engine=vllm"))
    model_ids = {m["provider_model_id"] for m in cfg["providers"]["models"]}
    assert model_ids == {"google/gemma-4-e4b-it"}


def _router_image(docs: list[dict]) -> str:
    for d in docs:
        if (
            d.get("kind") == "Deployment"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router"
        ):
            for c in d["spec"]["template"]["spec"]["containers"]:
                if c["name"] == "semantic-router":
                    return c["image"]
    raise AssertionError("semantic-router Deployment/container not rendered")


def test_semantic_cache_enabled_with_memory_backend():
    cache = _sr_config(_render("llm.engine=vllm"))["global"]["stores"]["semantic_cache"]
    assert cache["enabled"] is True
    assert cache["backend_type"] == "memory"
    assert cache["similarity_threshold"] == 0.95
    assert cache["max_entries"] == 1024
    assert cache["ttl_seconds"] == 3600
    assert cache["eviction_policy"] == "lru"
    assert cache["embedding_model"] == "mmbert"


def test_semantic_cache_embedding_runtime_configured():
    # Without mmbert_model_path + preload the embedding runtime never reaches
    # ready and the router silently bypasses the cache.
    semantic = _sr_config(_render("llm.engine=vllm"))["global"]["model_catalog"][
        "embeddings"
    ]["semantic"]
    assert semantic["mmbert_model_path"] == "models/mmbert-embed-32k-2d-matryoshka"
    assert semantic["embedding_config"]["model_type"] == "mmbert"
    assert semantic["embedding_config"]["preload_embeddings"] is True


def test_every_decision_enables_semantic_cache_plugin():
    # The cache is gated per-decision: with decisions present but no
    # semantic-cache plugin, every request bypasses the cache.
    decisions = _sr_config(_render("llm.engine=vllm"))["routing"]["decisions"]
    assert decisions, "expected routing decisions"
    for decision in decisions:
        plugins = decision.get("plugins", [])
        cache_plugins = [p for p in plugins if p.get("type") == "semantic-cache"]
        assert len(cache_plugins) == 1, f"{decision['name']} missing semantic-cache"
        assert cache_plugins[0]["configuration"]["enabled"] is True


def test_router_image_pinned_by_digest():
    # A moving `latest` left an older image cached whose embedding runtime never
    # reached ready; the digest pin makes the deployed router reproducible.
    image = _router_image(_render("llm.engine=vllm"))
    assert "@sha256:" in image, f"router image not digest-pinned: {image}"


def test_router_image_falls_back_to_tag_when_digest_cleared():
    image = _router_image(
        _render("llm.engine=vllm", "semanticRouter.router.image.digest=")
    )
    assert image.endswith(":latest"), f"expected tag fallback, got {image}"
