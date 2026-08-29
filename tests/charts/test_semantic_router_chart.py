"""Chart tests for the semantic-router upstream wiring.

The router is a transparent proxy in front of the SAME LLM the runtime calls,
so its backend endpoint must track ``cogniverse.primaryLLMEndpoint`` for every
engine — not a separate engine switch that can drift from it. A prior bug in
``srUpstreamHost``/``srUpstreamPort`` pointed the router at the non-existent
``-llm`` service for the vllm engine (whose LLM actually lives on the
``-vllm-llm-student`` service), so completions would never reach a backend.
These render-time assertions pin the endpoint per engine so that can't regress.
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


def _render_with_values(*values_files: str) -> list[dict]:
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
    for values_file in values_files:
        cmd.extend(["-f", str(CHART_PATH / values_file)])
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


def _container_env_entries(
    docs: list[dict], deployment_name: str, container_name: str
) -> list[dict]:
    for d in docs:
        if (
            d.get("kind") == "Deployment"
            and d.get("metadata", {}).get("name") == deployment_name
        ):
            containers = d["spec"]["template"]["spec"]["containers"]
            container = next(c for c in containers if c["name"] == container_name)
            return container.get("env", [])
    raise AssertionError(f"{deployment_name}/{container_name} not rendered")


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


def _envoy_service(docs: list[dict]) -> dict:
    for d in docs:
        if (
            d.get("kind") == "Service"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-envoy"
        ):
            return d
    raise AssertionError("semantic-router envoy Service not rendered")


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


def test_router_cold_download_has_thirty_minute_startup_budget():
    docs = _render("llm.engine=vllm")
    deployment = next(
        doc
        for doc in docs
        if doc.get("kind") == "Deployment"
        and doc["metadata"]["name"] == "cogniverse-semantic-router"
    )
    container = next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == "semantic-router"
    )
    probe = container["startupProbe"]

    assert probe["httpGet"] == {"path": "/metrics", "port": "metrics"}
    assert probe["periodSeconds"] == 10
    assert probe["failureThreshold"] == 180


def test_router_receives_the_optional_hf_token_secret():
    entries = _container_env_entries(
        _render(), "cogniverse-semantic-router", "semantic-router"
    )
    matches = [e for e in entries if e["name"] == "HF_TOKEN"]
    assert matches == [
        {
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "hf-token",
                    "key": "HF_TOKEN",
                    "optional": True,
                }
            },
        }
    ]


def test_envoy_does_not_receive_the_hf_token_env():
    entries = _container_env_entries(
        _render(), "cogniverse-semantic-router-envoy", "envoy"
    )
    assert [e for e in entries if e["name"] == "HF_TOKEN"] == []


def test_k3s_overlay_exposes_semantic_router_envoy_on_a_nodeport():
    service = _envoy_service(_render_with_values("values.k3s.yaml"))
    assert service["spec"]["type"] == "NodePort"
    assert service["spec"]["ports"] == [
        {
            "name": "http",
            "nodePort": 28081,
            "port": 8801,
            "protocol": "TCP",
            "targetPort": "http",
        }
    ]


def _router_container(docs: list[dict]) -> dict:
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        name = doc["metadata"]["name"]
        if "semantic-router" in name and "envoy" not in name:
            return doc["spec"]["template"]["spec"]
    raise AssertionError("semantic-router Deployment not rendered")


def test_router_model_directory_is_backed_by_the_persistent_claim():
    """The weights land in the claim, so a restart resumes instead of restarting.

    The router's downloader writes to models/<name> relative to its workdir and
    ignores HF_HOME, so a claim mounted only at HF_HOME leaves the weights on the
    container's ephemeral layer: every restart discards a partial download and the
    pod can never finish one.
    """
    pod = _router_container(_render("semanticRouter.router.persistence.enabled=true"))
    container = pod["containers"][0]

    assert [
        (mount["name"], mount["mountPath"], mount.get("subPath"))
        for mount in container["volumeMounts"]
    ] == [
        ("router-config", "/app/config.yaml", "config.yaml"),
        ("models-cache", "/models-cache", "hf"),
        ("models-cache", "/app/models", "models"),
    ]

    claims = {
        volume["name"]: volume["persistentVolumeClaim"]["claimName"]
        for volume in pod["volumes"]
        if "persistentVolumeClaim" in volume
    }
    assert claims == {"models-cache": "cogniverse-semantic-router-models"}


def _backend_protocols(cfg: dict) -> set[str]:
    return {
        ref["protocol"]
        for model in cfg["providers"]["models"]
        for ref in model["backend_refs"]
    }


class TestUpstreamScheme:
    """The router dials the upstream itself, so the scheme decides both the
    default port and the protocol it speaks.

    An https endpoint with no explicit port defaulted to :80 and protocol http.
    Envoy answers that with 'no healthy upstream' and the runtime surfaces
    litellm.ServiceUnavailableError on every LLM call.
    """

    HTTPS = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
    HTTP = "http://cogniverse-vllm-llm-student:8000/v1"

    def test_https_upstream_without_a_port_uses_443_and_https(self):
        cfg = _sr_config(_render(f"runtime.primaryLLM.apiBase={self.HTTPS}"))

        assert _backend_endpoints(cfg) == {
            "amit-jain--cogniverse-vllm-llm-student-inference.modal.run:443"
        }
        assert _backend_protocols(cfg) == {"https"}

    def test_http_upstream_keeps_its_explicit_port_and_http(self):
        cfg = _sr_config(_render(f"runtime.primaryLLM.apiBase={self.HTTP}"))

        assert _backend_endpoints(cfg) == {"cogniverse-vllm-llm-student:8000"}
        assert _backend_protocols(cfg) == {"http"}


def _llm_upstream_cluster(docs: list[dict]) -> dict:
    for d in docs:
        if (
            d.get("kind") == "ConfigMap"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-envoy"
        ):
            envoy = yaml.safe_load(d["data"]["envoy.yaml"])
            for cluster in envoy["static_resources"]["clusters"]:
                if cluster["name"] == "llm_upstream":
                    return cluster
    raise AssertionError("llm_upstream cluster not rendered")


class TestEnvoyUpstreamTls:
    """Envoy terminates the runtime's LLM traffic and re-dials the upstream.

    Pointing it at :443 without a TLS transport socket connects in plaintext to
    a TLS listener, which Envoy reports as 'no healthy upstream'.
    """

    HTTPS = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
    HTTP = "http://cogniverse-vllm-llm-student:8000/v1"

    def test_https_upstream_gets_a_tls_socket_with_sni(self):
        cluster = _llm_upstream_cluster(
            _render(f"runtime.primaryLLM.apiBase={self.HTTPS}")
        )

        socket = cluster["transport_socket"]
        assert socket["name"] == "envoy.transport_sockets.tls"
        assert socket["typed_config"]["@type"] == (
            "type.googleapis.com/envoy.extensions.transport_sockets.tls.v3"
            ".UpstreamTlsContext"
        )
        assert socket["typed_config"]["sni"] == (
            "amit-jain--cogniverse-vllm-llm-student-inference.modal.run"
        )

    def test_http_upstream_has_no_tls_socket(self):
        cluster = _llm_upstream_cluster(
            _render(f"runtime.primaryLLM.apiBase={self.HTTP}")
        )

        assert "transport_socket" not in cluster


def _runtime_config(docs: list[dict]) -> dict:
    for d in docs:
        if (
            d.get("kind") == "ConfigMap"
            and d.get("metadata", {}).get("name") == "cogniverse-config"
        ):
            return json.loads(d["data"]["config.json"])
    raise AssertionError("cogniverse-config ConfigMap not rendered")


class TestStudentEndpointFollowsTheOverride:
    """Every consumer of the student model must reach it where it is served.

    ``llmStudentEndpoint`` built the in-cluster Service URL unconditionally, so
    with the student on Modal the ingestion pipeline's VLM description strategy
    dialled a Service that no longer exists and failed with
    NameResolutionError on cogniverse-vllm-llm-student.
    """

    MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"

    STUDENT_KEYS = ("vlm_endpoint", "base_url")

    def _student_urls(self, cfg: dict) -> set[str]:
        """Every endpoint any consumer would use to reach the student model.

        Walked recursively rather than by a fixed path, so a new consumer added
        anywhere in the config is covered instead of silently missed.
        """
        found: set[str] = set()

        def walk(node, model_hint=None):
            if isinstance(node, dict):
                hint = node.get("model", model_hint)
                for key, value in node.items():
                    if key in self.STUDENT_KEYS and isinstance(value, str):
                        if "student" in value or value == self.MODAL:
                            found.add(value)
                    else:
                        walk(value, hint)
            elif isinstance(node, list):
                for item in node:
                    walk(item, model_hint)

        walk(cfg)
        return found

    def test_override_reaches_every_student_consumer(self):
        cfg = _runtime_config(_render(f"runtime.primaryLLM.apiBase={self.MODAL}"))

        assert self._student_urls(cfg) == {self.MODAL}

    def test_without_an_override_they_stay_in_cluster(self):
        cfg = _runtime_config(_render())

        assert self._student_urls(cfg) == {"http://cogniverse-vllm-llm-student:8000/v1"}
