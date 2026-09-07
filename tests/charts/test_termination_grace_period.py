"""Vespa StatefulSet grace period covers the searchnode prepareRestart flush."""

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

PREPARE_RESTART_RPC_BUDGET_S = 600
CONTAINER_PREPARE_STOP_RPC_TIMEOUT_S = 370


def _vespa_statefulset(*extra: str) -> dict:
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        *extra,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )
    matches = [
        d
        for d in yaml.safe_load_all(result.stdout)
        if d
        and d.get("kind") == "StatefulSet"
        and d["metadata"]["name"] == "cogniverse-vespa"
    ]
    assert len(matches) == 1, [m["metadata"]["name"] for m in matches]
    return matches[0]


def test_default_grace_period_is_1200s():
    spec = _vespa_statefulset()["spec"]["template"]["spec"]
    assert spec["terminationGracePeriodSeconds"] == 1200


def test_default_grace_period_exceeds_stop_budget():
    spec = _vespa_statefulset()["spec"]["template"]["spec"]
    assert spec["terminationGracePeriodSeconds"] > (
        CONTAINER_PREPARE_STOP_RPC_TIMEOUT_S + PREPARE_RESTART_RPC_BUDGET_S
    )


def test_grace_period_is_value_driven():
    spec = _vespa_statefulset("--set", "vespa.terminationGracePeriodSeconds=1234")[
        "spec"
    ]["template"]["spec"]
    assert spec["terminationGracePeriodSeconds"] == 1234
