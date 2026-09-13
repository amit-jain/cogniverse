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


# The router serves pro-reasoning from the teacher, so a render with the
# router on needs a served teacher; the in-cluster one is the plain case.
TEACHER_SERVED = "inference.vllm_llm_teacher.enabled=true"


def _helm_template(*set_args: str) -> subprocess.CompletedProcess:
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
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _render(*set_args: str) -> list[dict]:
    result = _helm_template(TEACHER_SERVED, *set_args)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n{result.stderr}"
        )
    return [d for d in yaml.safe_load_all(result.stdout) if d is not None]


def _chart_values(*values_files: str) -> dict:
    merged: dict = {}
    for name in ("values.yaml", *values_files):
        _deep_merge(merged, yaml.safe_load((CHART_PATH / name).read_text()))
    return merged


def _deep_merge(into: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _deep_merge(into[key], value)
        else:
            into[key] = value


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


def _backend_endpoints(cfg: dict) -> dict[str, str]:
    """Catalog model name -> the one backend endpoint it is bound to."""
    endpoints: dict[str, str] = {}
    for model in cfg["providers"]["models"]:
        (ref,) = model["backend_refs"]
        endpoints[model["name"]] = ref["endpoint"]
    return endpoints


def _provider_model_ids(cfg: dict) -> dict[str, str]:
    return {m["name"]: m["provider_model_id"] for m in cfg["providers"]["models"]}


TEACHER_IN_CLUSTER = "cogniverse-vllm-llm-teacher:8000"


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
    assert _backend_endpoints(cfg) == {
        "basic-chat": "cogniverse-vllm-llm-student:8000",
        "pro-reasoning": TEACHER_IN_CLUSTER,
    }


def test_ollama_engine_routes_to_llm_service():
    cfg = _sr_config(_render("llm.engine=ollama"))
    assert _backend_endpoints(cfg) == {
        "basic-chat": "cogniverse-llm:11434",
        "pro-reasoning": TEACHER_IN_CLUSTER,
    }


def test_external_engine_parses_configured_url():
    cfg = _sr_config(
        _render("llm.engine=external", "llm.external.url=http://my-llm:9000/v1")
    )
    assert _backend_endpoints(cfg) == {
        "basic-chat": "my-llm:9000",
        "pro-reasoning": TEACHER_IN_CLUSTER,
    }


def test_envoy_upstream_matches_sr_backend_for_vllm():
    docs = _render("llm.engine=vllm")
    assert _envoy_upstream(docs) == "cogniverse-vllm-llm-student:8000"
    assert _backend_endpoints(_sr_config(docs))["basic-chat"] == (
        "cogniverse-vllm-llm-student:8000"
    )


def test_each_catalog_model_serves_its_own_shipped_model_id():
    """basic-chat is the student the chart serves and pro-reasoning the
    teacher, both read from the values the pods themselves are rendered
    from, so a model bump in values moves the router with it."""
    cfg = _sr_config(_render("llm.engine=vllm"))
    inference = _chart_values()["inference"]
    assert _provider_model_ids(cfg) == {
        "basic-chat": inference["vllm_llm_student"]["model"],
        "pro-reasoning": inference["vllm_llm_teacher"]["model"],
    }
    assert (
        inference["vllm_llm_student"]["model"]
        != (inference["vllm_llm_teacher"]["model"])
    )


class TestTheTeacherHasItsOwnEnvoyCluster:
    """Envoy has one static cluster per backend and routes on the model the
    router selected; before this every catalog model reached the student's
    cluster whatever the router config said."""

    def test_the_teacher_route_precedes_the_catch_all(self):
        docs = _render("llm.engine=vllm")
        routes = _envoy_routes(docs)
        assert [route["route"]["cluster"] for route in routes] == [
            "llm_teacher",
            "llm_upstream",
        ]
        (header,) = routes[0]["match"]["headers"]
        assert header == {
            "name": "x-selected-model",
            "string_match": {"exact": "pro-reasoning"},
        }
        assert "headers" not in routes[1]["match"]
        assert [route["match"]["prefix"] for route in routes] == ["/", "/"]
        assert [route["route"]["timeout"] for route in routes] == ["300s", "300s"]

    def test_the_matched_value_is_the_model_every_pro_decision_serves(self):
        docs = _render("llm.engine=vllm")
        (header,) = _envoy_routes(docs)[0]["match"]["headers"]
        cfg = _sr_config(docs)
        pro_models = {
            ref["model"]
            for decision in cfg["routing"]["decisions"]
            if decision["name"].startswith("pro-")
            for ref in decision["modelRefs"]
        }
        assert pro_models == {header["string_match"]["exact"]}
        assert header["string_match"]["exact"] in {
            m["name"] for m in cfg["providers"]["models"]
        }

    def test_in_cluster_teacher_is_plain_http_with_no_host_rewrite(self):
        docs = _render("llm.engine=vllm")
        cluster = _envoy_cluster(docs, "llm_teacher")
        assert _cluster_address(cluster) == TEACHER_IN_CLUSTER
        assert "transport_socket" not in cluster
        assert "auto_host_rewrite" not in _envoy_routes(docs)[0]["route"]
        assert _backend_endpoints(_sr_config(docs))["pro-reasoning"] == (
            TEACHER_IN_CLUSTER
        )

    def test_modal_teacher_gets_tls_sni_and_a_host_rewrite(self):
        docs = _render_with_values("values.k3s.yaml", "values.modal-llm.yaml")
        values = _chart_values("values.k3s.yaml", "values.modal-llm.yaml")
        teacher_host = values["inference"]["vllm_llm_teacher"]["externalUrl"].split(
            "://", 1
        )[1]
        student_host = (
            values["runtime"]["primaryLLM"]["apiBase"]
            .split("://", 1)[1]
            .removesuffix("/v1")
        )
        assert teacher_host != student_host
        cluster = _envoy_cluster(docs, "llm_teacher")
        assert _cluster_address(cluster) == f"{teacher_host}:443"
        assert cluster["transport_socket"]["typed_config"]["sni"] == teacher_host
        assert _cluster_address(_envoy_cluster(docs, "llm_upstream")) == (
            f"{student_host}:443"
        )
        routes = _envoy_routes(docs)
        assert [route["route"]["auto_host_rewrite"] for route in routes] == [
            True,
            True,
        ]
        assert _backend_endpoints(_sr_config(docs)) == {
            "basic-chat": f"{student_host}:443",
            "pro-reasoning": f"{teacher_host}:443",
        }
        assert _backend_protocols(_sr_config(docs)) == {"https"}


class TestAnUnservedTeacherFailsTheRender:
    """The router sends every pro decision to the teacher, so a chart with the
    router on and nothing serving the teacher is refused at render time rather
    than answering 'no healthy upstream' on the first pro request."""

    UNSERVED = (
        "inference.vllm_llm_teacher.enabled=false",
        "inference.vllm_llm_teacher.externalUrl=",
    )

    def test_router_on_and_teacher_unserved_is_refused_naming_both_values(self):
        result = _helm_template(*self.UNSERVED)
        assert result.returncode != 0
        assert "inference.vllm_llm_teacher.enabled" in result.stderr
        assert "inference.vllm_llm_teacher.externalUrl" in result.stderr
        assert "semanticRouter.enabled" in result.stderr

    def test_an_external_teacher_renders(self):
        result = _helm_template(
            "inference.vllm_llm_teacher.enabled=false",
            "inference.vllm_llm_teacher.externalUrl=https://teacher.example.modal.run",
        )
        assert result.returncode == 0, result.stderr

    def test_router_off_renders_without_a_teacher(self):
        result = _helm_template(*self.UNSERVED, "semanticRouter.enabled=false")
        assert result.returncode == 0, result.stderr


def _envoy_config(docs: list[dict]) -> dict:
    for d in docs:
        if (
            d.get("kind") == "ConfigMap"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-envoy"
        ):
            return yaml.safe_load(d["data"]["envoy.yaml"])
    raise AssertionError("semantic-router envoy ConfigMap not rendered")


def _envoy_routes(docs: list[dict]) -> list[dict]:
    listener = _envoy_config(docs)["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    (vhost,) = hcm["route_config"]["virtual_hosts"]
    return vhost["routes"]


def _envoy_cluster(docs: list[dict], name: str) -> dict:
    (cluster,) = [
        c
        for c in _envoy_config(docs)["static_resources"]["clusters"]
        if c["name"] == name
    ]
    return cluster


def _cluster_address(cluster: dict) -> str:
    (endpoint,) = cluster["load_assignment"]["endpoints"]
    (lb,) = endpoint["lb_endpoints"]
    sock = lb["endpoint"]["address"]["socket_address"]
    return f"{sock['address']}:{sock['port_value']}"


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


# The policy every decision carries. ``mode: exact`` keys a hit on a SHA-256
# over the whole normalized request, so two prompts differing by one character
# are two entries; ``scope: user`` partitions by the tenant identity the
# runtime sends. ``max-age`` is the only client directive, and it can only
# narrow the entry age a request will accept.
_EXACT_CACHE_PLUGIN = {
    "type": "response_cache",
    "configuration": {
        "enabled": True,
        "mode": "exact",
        "scope": "user",
        "request_controls": {
            "enabled": True,
            "header": "x-vsr-cache-control",
            "allowed": ["max-age"],
        },
    },
}


def test_response_cache_store_is_memory_backed_with_the_configured_bounds():
    """Bounds come from values.yaml, never restated here."""
    bounds = yaml.safe_load((CHART_PATH / "values.yaml").read_text())["semanticRouter"][
        "router"
    ]["responseCache"]
    stores = _sr_config(_render("llm.engine=vllm"))["global"]["stores"]
    assert stores == {
        "response_cache": {
            "backend_type": "memory",
            "enabled": True,
            "max_entries": bounds["maxEntries"],
            "ttl_seconds": bounds["ttlSeconds"],
        }
    }
    # Both bounds read 0 as "unlimited" in the router: a ttl of 0 is an entry
    # that outlives the process, a max_entries of 0 is a map that grows until
    # the pod is OOM-killed.
    assert bounds["ttlSeconds"] not in (0, None)
    assert bounds["maxEntries"] not in (0, None)


def test_the_agent_side_cache_reads_the_same_bounds():
    """The LM cache in the agents inherits these, so the two must agree."""
    from cogniverse_foundation.config.unified_config import SemanticRouterConfig

    bounds = yaml.safe_load((CHART_PATH / "values.yaml").read_text())["semanticRouter"][
        "router"
    ]["responseCache"]
    declared = SemanticRouterConfig()

    assert (bounds["ttlSeconds"], bounds["maxEntries"]) == (
        declared.response_cache_ttl_seconds,
        declared.response_cache_max_entries,
    )


def test_semantic_cache_embedding_runtime_configured():
    # Without mmbert_model_path + preload the embedding runtime never reaches
    # ready and the router silently bypasses the cache.
    semantic = _sr_config(_render("llm.engine=vllm"))["global"]["model_catalog"][
        "embeddings"
    ]["semantic"]
    assert semantic["mmbert_model_path"] == "models/mmbert-embed-32k-2d-matryoshka"
    assert semantic["embedding_config"]["model_type"] == "mmbert"
    assert semantic["embedding_config"]["preload_embeddings"] is True


def test_every_decision_caches_on_exact_request_identity_only():
    """The cache is gated per-decision: a decision without the plugin never
    caches, and a decision whose plugin omits ``mode`` falls back to similarity
    matching at 0.8 - which answers one query out of another query's entry."""
    decisions = _sr_config(_render("llm.engine=vllm"))["routing"]["decisions"]
    assert [decision["name"] for decision in decisions] == [
        "pro-technical-keyword",
        "pro-technical-domain",
        "pro-default",
        "free-default",
        "base-default",
    ]
    assert [decision.get("plugins") for decision in decisions] == [
        [_EXACT_CACHE_PLUGIN]
    ] * len(decisions)


def test_decisions_declare_their_model_reasoning_and_admitting_tier_exactly():
    """Which catalog model each decision serves, whether it reasons, and the
    condition that admits it - a tier change is a deliberate edit here."""
    cfg = _sr_config(_render("llm.engine=vllm"))
    decisions = cfg["routing"]["decisions"]
    assert [decision["priority"] for decision in decisions] == [300, 250, 200, 100, 50]
    assert [
        [ref["model"] for ref in decision["modelRefs"]] for decision in decisions
    ] == [
        ["pro-reasoning"],
        ["pro-reasoning"],
        ["pro-reasoning"],
        ["basic-chat"],
        ["basic-chat"],
    ]
    assert [
        [ref["use_reasoning"] for ref in decision["modelRefs"]]
        for decision in decisions
    ] == [[True], [True], [False], [False], [False]]
    assert [
        [(cond["type"], cond["name"]) for cond in decision["rules"]["conditions"]]
        for decision in decisions
    ] == [
        [("authz", "pro_tier"), ("keyword", "technical")],
        [("authz", "pro_tier"), ("domain", "technical")],
        [("authz", "pro_tier")],
        [("authz", "free_tier")],
        [("authz", "base_tier")],
    ]
    assert sorted(cfg["global"]["stores"]) == ["response_cache"]


def test_the_classification_entrypoint_selects_the_tier_only_recipe():
    """A bounded-output call names this virtual model; the recipe it selects
    must test the tenant tier and nothing else. One ``domain`` or ``keyword``
    condition anywhere in it puts the classifier back on the request, which is
    the cost the entrypoint exists to remove."""
    cfg = _sr_config(_render("llm.engine=vllm"))
    assert cfg["entrypoints"] == [
        {"model_names": ["cogniverse-classification"], "recipe": "classification"}
    ]
    assert [recipe["name"] for recipe in cfg["recipes"]] == ["classification"]
    decisions = cfg["recipes"][0]["routing"]["decisions"]
    assert [decision["name"] for decision in decisions] == [
        "classification-pro",
        "classification-free",
        "classification-base",
    ]
    assert [decision["priority"] for decision in decisions] == [200, 100, 50]
    assert [
        [(cond["type"], cond["name"]) for cond in decision["rules"]["conditions"]]
        for decision in decisions
    ] == [
        [("authz", "pro_tier")],
        [("authz", "free_tier")],
        [("authz", "base_tier")],
    ]
    assert [
        [(ref["model"], ref["use_reasoning"]) for ref in decision["modelRefs"]]
        for decision in decisions
    ] == [
        [("basic-chat", False)],
        [("basic-chat", False)],
        [("basic-chat", False)],
    ]


def test_the_classification_recipe_caches_on_exact_request_identity_only():
    """The recipe's decisions are named, so their plugins run - which is what
    naming a catalog model directly would have given up."""
    recipe = _sr_config(_render("llm.engine=vllm"))["recipes"][0]
    assert [decision["plugins"] for decision in recipe["routing"]["decisions"]] == [
        [_EXACT_CACHE_PLUGIN]
    ] * 3


def test_no_decision_in_any_routing_profile_matches_on_similarity():
    """Swept over both profiles, not just the default one: on the shipped
    embedding model the highest similarity between requests about different
    content (0.9932) exceeds the lowest between equivalent requests (0.9867),
    so any similarity threshold answers one request out of another's entry."""
    cfg = _sr_config(_render("llm.engine=vllm"))
    profiles = [cfg["routing"]] + [recipe["routing"] for recipe in cfg["recipes"]]
    modes = {
        plugin["configuration"]["mode"]
        for profile in profiles
        for decision in profile["decisions"]
        for plugin in decision["plugins"]
        if plugin["type"] == "response_cache"
    }
    assert modes == {"exact"}
    thresholds = [
        plugin["configuration"].get("semantic")
        for profile in profiles
        for decision in profile["decisions"]
        for plugin in decision["plugins"]
    ]
    assert thresholds == [None] * len(thresholds)


def test_the_rewrite_budget_fires_before_the_ext_proc_message_timeout():
    """Whichever deadline fires first owns the failure. The rewrite's own
    budget must be the smaller one, or a slow router surfaces as Envoy
    abandoning the ext_proc stream instead of as a degraded rewrite."""
    from cogniverse_agents.search_agent import QUERY_REWRITE_BUDGET_S

    envoys = [
        yaml.safe_load(d["data"]["envoy.yaml"])
        for d in _render("llm.engine=vllm")
        if d.get("kind") == "ConfigMap"
        and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-envoy"
    ]
    assert len(envoys) == 1
    message_timeouts = [
        http_filter["typed_config"]["message_timeout"]
        for listener in envoys[0]["static_resources"]["listeners"]
        for chain in listener["filter_chains"]
        for filt in chain["filters"]
        for http_filter in filt["typed_config"]["http_filters"]
        if "message_timeout" in http_filter.get("typed_config", {})
    ]
    assert message_timeouts == ["30s"]
    assert QUERY_REWRITE_BUDGET_S == 3.5
    assert QUERY_REWRITE_BUDGET_S < float(message_timeouts[0].rstrip("s"))


def test_router_image_pinned_by_digest():
    # A moving `latest` left an older image cached whose embedding runtime never
    # reached ready; the digest pin makes the deployed router reproducible.
    image = _router_image(_render("llm.engine=vllm"))
    assert "@sha256:" in image, f"router image not digest-pinned: {image}"


def test_router_image_falls_back_to_tag_when_digest_cleared():
    # Derived from values.yaml, never a literal: a tag bump would otherwise
    # have to be restated here, and the copy that drifts is the one nobody
    # reads.
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text())
    image = values["semanticRouter"]["router"]["image"]
    fallen_back = _router_image(
        _render("llm.engine=vllm", "semanticRouter.router.image.digest=")
    )
    assert fallen_back == f"{image['repository']}:{image['tag']}"


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

        assert _backend_endpoints(cfg)["basic-chat"] == (
            "amit-jain--cogniverse-vllm-llm-student-inference.modal.run:443"
        )
        assert _backend_protocols(cfg) == {"http", "https"}

    def test_http_upstream_keeps_its_explicit_port_and_http(self):
        cfg = _sr_config(_render(f"runtime.primaryLLM.apiBase={self.HTTP}"))

        assert (
            _backend_endpoints(cfg)["basic-chat"] == "cogniverse-vllm-llm-student:8000"
        )
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


class TestConfigChangeRollsThePods:
    """Envoy and the router read their ConfigMaps at startup only.

    Their images are pinned, so a pure config change produced an identical pod
    spec, kubectl saw nothing to roll, and the pods kept serving the config they
    booted with. A TLS/port fix sat correct in the ConfigMap for two runs while
    a 19-hour-old Envoy kept answering 'no healthy upstream'.
    """

    ROUTER = "cogniverse-semantic-router"
    ENVOY = "cogniverse-semantic-router-envoy"

    def _annotations(self, docs: list[dict], name: str) -> dict:
        for d in docs:
            if d.get("kind") == "Deployment" and d["metadata"]["name"] == name:
                return d["spec"]["template"]["metadata"].get("annotations") or {}
        raise AssertionError(f"{name} Deployment not rendered")

    def test_both_router_pods_carry_a_config_checksum(self):
        docs = _render()

        for name in (self.ROUTER, self.ENVOY):
            annotations = self._annotations(docs, name)
            assert "cogniverse.io/router-config-checksum" in annotations, name
            assert len(annotations["cogniverse.io/router-config-checksum"]) == 64, name

    def test_a_config_change_changes_the_checksum(self):
        before = _render()
        after = _render(
            "runtime.primaryLLM.apiBase=https://changed.example.modal.run/v1"
        )

        for name in (self.ROUTER, self.ENVOY):
            assert (
                self._annotations(before, name)["cogniverse.io/router-config-checksum"]
                != self._annotations(after, name)[
                    "cogniverse.io/router-config-checksum"
                ]
            ), name

    def test_an_unrelated_change_leaves_the_checksum_alone(self):
        before = _render()
        after = _render("runtime.replicaCount=2")

        for name in (self.ROUTER, self.ENVOY):
            assert (
                self._annotations(before, name)["cogniverse.io/router-config-checksum"]
                == self._annotations(after, name)[
                    "cogniverse.io/router-config-checksum"
                ]
            ), name


def _llm_route(docs: list[dict]) -> dict:
    for d in docs:
        if (
            d.get("kind") == "ConfigMap"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-envoy"
        ):
            envoy = yaml.safe_load(d["data"]["envoy.yaml"])
            for listener in envoy["static_resources"]["listeners"]:
                for chain in listener["filter_chains"]:
                    for filt in chain["filters"]:
                        cfg = filt["typed_config"]
                        for vhost in cfg["route_config"]["virtual_hosts"]:
                            for route in vhost["routes"]:
                                if route["route"].get("cluster") == "llm_upstream":
                                    return route["route"]
    raise AssertionError("llm_upstream route not rendered")


class TestUpstreamHostHeader:
    """Modal routes by Host header, not by path.

    Envoy forwarded the client's authority (the in-cluster envoy Service name),
    which Modal answered with 'modal-http: invalid function call' - surfaced as
    litellm NotFoundError on every agent call.
    """

    HTTPS = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
    HTTP = "http://cogniverse-vllm-llm-student:8000/v1"

    def test_external_upstream_rewrites_the_host_header(self):
        route = _llm_route(_render(f"runtime.primaryLLM.apiBase={self.HTTPS}"))

        assert route["auto_host_rewrite"] is True

    def test_in_cluster_upstream_leaves_the_host_alone(self):
        route = _llm_route(_render(f"runtime.primaryLLM.apiBase={self.HTTP}"))

        assert "auto_host_rewrite" not in route


def _envoy_hcm(docs: list[dict]) -> dict:
    for d in docs:
        if (
            d.get("kind") == "ConfigMap"
            and d.get("metadata", {}).get("name") == "cogniverse-semantic-router-envoy"
        ):
            envoy = yaml.safe_load(d["data"]["envoy.yaml"])
            listener = envoy["static_resources"]["listeners"][0]
            return listener["filter_chains"][0]["filters"][0]["typed_config"]
    raise AssertionError("envoy HCM not found")


def test_envoy_access_log_names_the_reason_for_every_local_reply():
    """A proxy-emitted status (413, 504, 503) is attributable from the pod log
    alone: the access log carries Envoy's own reason string and both byte
    counts, so a rejection never has to be reconstructed from the client."""
    hcm = _envoy_hcm(_render())
    logs = hcm["access_log"]
    assert len(logs) == 1, logs
    typed = logs[0]["typed_config"]
    assert typed["@type"].endswith("StdoutAccessLog"), typed
    fields = typed["log_format"]["json_format"]
    assert fields == {
        "ts": "%START_TIME%",
        "method": "%REQ(:METHOD)%",
        "path": "%REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%",
        "status": "%RESPONSE_CODE%",
        "reason": "%RESPONSE_CODE_DETAILS%",
        "flags": "%RESPONSE_FLAGS%",
        "bytes_received": "%BYTES_RECEIVED%",
        "bytes_sent": "%BYTES_SENT%",
        "duration_ms": "%DURATION%",
        "upstream": "%UPSTREAM_HOST%",
        "request_id": "%REQ(X-REQUEST-ID)%",
    }, fields


@pytest.mark.parametrize("ttl,capacity", [(3600, 1024), (17, 9)])
def test_rendered_lm_cache_bounds_follow_chart_values(ttl, capacity):
    docs = _render(
        f"semanticRouter.router.responseCache.ttlSeconds={ttl}",
        f"semanticRouter.router.responseCache.maxEntries={capacity}",
    )
    bounds = _runtime_config(docs)["semantic_router"]
    assert bounds == {
        "response_cache_ttl_seconds": ttl,
        "response_cache_max_entries": capacity,
    }
