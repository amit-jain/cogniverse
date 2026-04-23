"""Chart tests for the generic inference services.

The chart supports N parallel inference services under ``inference`` — each
entry deploys one pod. Keys are logical tags the deployer chooses
(``general``, ``code``, ``image``, ``embed``...). Each service has an
``engine`` (``vllm`` or ``pylate``) and a ``type`` (``multi_vector`` or
``single_vector``).

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
        component = d.get("metadata", {}).get("labels", {}).get(
            "app.kubernetes.io/component", ""
        )
        if component.startswith("inference-"):
            out[component.removeprefix("inference-")] = d
    return out


def _runtime_env(docs: list[dict]) -> dict[str, str]:
    for d in docs:
        if (
            d.get("kind") == "Deployment"
            and d.get("metadata", {}).get("labels", {}).get(
                "app.kubernetes.io/component"
            )
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


def test_default_runs_only_general_service():
    deps = _inference_deployments(_render())
    assert set(deps.keys()) == {"general"}
    assert deps["general"]["metadata"]["name"] == "cogniverse-general"


def test_default_general_uses_pylate_with_lateon():
    """Default general service serves LateOn via pylate; vllm+ModernColBERT is not the default."""
    deps = _inference_deployments(_render())
    container = deps["general"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].startswith("cogniverse/pylate")
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["MODEL_NAME"] == "lightonai/LateOn"


def test_default_inference_service_urls_contains_general_only():
    urls = _service_urls(_render())
    assert urls == {"general": "http://cogniverse-general:8000"}


def test_enabling_code_runs_two_parallel_services():
    """code and general coexist as independent pods."""
    docs = _render("inference.code.enabled=true")
    deps = _inference_deployments(docs)
    assert set(deps.keys()) == {"general", "code"}
    assert deps["general"]["metadata"]["name"] == "cogniverse-general"
    assert deps["code"]["metadata"]["name"] == "cogniverse-code"


def test_enabling_code_adds_to_url_map():
    urls = _service_urls(_render("inference.code.enabled=true"))
    assert urls == {
        "general": "http://cogniverse-general:8000",
        "code": "http://cogniverse-code:8000",
    }


def test_switching_general_to_vllm_does_not_affect_code():
    """Flipping one service's engine must not affect another."""
    docs = _render(
        "inference.code.enabled=true",
        "inference.general.engine=vllm",
        "inference.general.model=lightonai/Reason-ModernColBERT",
    )
    deps = _inference_deployments(docs)
    general_image = deps["general"]["spec"]["template"]["spec"]["containers"][0]["image"]
    code_image = deps["code"]["spec"]["template"]["spec"]["containers"][0]["image"]
    assert general_image.startswith("vllm/vllm-openai-cpu")
    assert code_image.startswith("cogniverse/pylate")


def test_pylate_container_env_pins_model_and_device():
    docs = _render("inference.general.model=lightonai/LateOn-Code-edge")
    deps = _inference_deployments(docs)
    container = deps["general"]["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["MODEL_NAME"] == "lightonai/LateOn-Code-edge"
    assert env["DEVICE"] == "cpu"
    assert env["PORT"] == "8000"


def test_disabling_general_drops_service_and_url():
    docs = _render("inference.general.enabled=false")
    deps = _inference_deployments(docs)
    assert "general" not in deps
    assert "general" not in _service_urls(docs)


def test_service_keys_in_url_map_match_deployment_names():
    """Every deployed service has a matching URL entry."""
    docs = _render(
        "inference.code.enabled=true",
        "inference.image.enabled=true",
    )
    deps = _inference_deployments(docs)
    urls = _service_urls(docs)
    assert set(deps.keys()) == set(urls.keys())
    for key in deps:
        assert urls[key].startswith(f"http://cogniverse-{key}")
