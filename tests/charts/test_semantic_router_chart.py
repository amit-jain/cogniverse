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
