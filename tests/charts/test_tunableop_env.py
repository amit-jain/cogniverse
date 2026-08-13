"""PyTorch TunableOp env rendered per inference service."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

TUNABLEOP_KEYS = (
    "PYTORCH_TUNABLEOP_ENABLED",
    "PYTORCH_TUNABLEOP_TUNING",
    "PYTORCH_TUNABLEOP_FILENAME",
)

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*extra: str, rocm: bool = True) -> list[dict]:
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    if rocm:
        command.extend(["-f", str(CHART_PATH / "values.rocm.yaml")])
    command.extend(extra)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _tunableop_env(*extra: str, rocm: bool = True) -> dict[str, dict[str, str]]:
    """Map deployment name -> {tunableop env var: value} for every Deployment."""
    found: dict[str, dict[str, str]] = {}
    for document in _render(*extra, rocm=rocm):
        if document.get("kind") != "Deployment":
            continue
        name = document["metadata"]["name"]
        env_vars: dict[str, str] = {}
        for container in document["spec"]["template"]["spec"].get("containers", []):
            for entry in container.get("env", []) or []:
                if entry.get("name") in TUNABLEOP_KEYS:
                    env_vars[entry["name"]] = entry.get("value", "")
        found[name] = env_vars
    return found


ENABLE_BOTH_PYLATES = (
    "--set",
    "runtime.tunableOp=true",
    "--set",
    "inference.code_colbert_pylate.enabled=true",
)


def test_pylate_services_render_no_tunableop_env():
    """The pylate server faults the GPU while TunableOp tunes an untuned shape."""
    env = _tunableop_env(*ENABLE_BOTH_PYLATES)

    assert env["cogniverse-colbert-pylate"] == {}
    assert env["cogniverse-code-colbert-pylate"] == {}


def test_non_pylate_rocm_services_keep_tunableop_env():
    env = _tunableop_env(*ENABLE_BOTH_PYLATES)

    for name in (
        "cogniverse-vllm-asr",
        "cogniverse-vllm-colpali",
        "cogniverse-denseon",
    ):
        assert env[name]["PYTORCH_TUNABLEOP_ENABLED"] == "1", name
        assert env[name]["PYTORCH_TUNABLEOP_TUNING"] == "1", name
        assert env[name]["PYTORCH_TUNABLEOP_FILENAME"].endswith("_%d.csv"), name


def test_explicit_override_re_enables_tunableop_for_a_pylate_service():
    """The engine-derived default is overridable per service."""
    env = _tunableop_env(
        *ENABLE_BOTH_PYLATES,
        "--set",
        "inference.colbert_pylate.tunableOp=true",
    )

    assert env["cogniverse-colbert-pylate"]["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["cogniverse-colbert-pylate"]["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert env["cogniverse-code-colbert-pylate"] == {}


def test_explicit_override_disables_tunableop_for_a_non_pylate_service():
    env = _tunableop_env(
        *ENABLE_BOTH_PYLATES,
        "--set",
        "inference.denseon.tunableOp=false",
    )

    assert env["cogniverse-denseon"] == {}
    assert env["cogniverse-vllm-asr"]["PYTORCH_TUNABLEOP_ENABLED"] == "1"


def test_global_toggle_off_renders_no_tunableop_env_anywhere():
    env = _tunableop_env(
        "--set",
        "runtime.tunableOp=false",
        "--set",
        "inference.code_colbert_pylate.enabled=true",
    )

    assert all(values == {} for values in env.values()), {
        name: values for name, values in env.items() if values
    }


def test_non_rocm_overlay_renders_no_tunableop_env():
    env = _tunableop_env("--set", "runtime.tunableOp=true", rocm=False)

    assert all(values == {} for values in env.values()), {
        name: values for name, values in env.items() if values
    }
