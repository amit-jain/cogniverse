"""Every pod running cogniverse code resolves the remote semantic embedder.

The semantic embedder has no in-process fallback: a process with neither
``COGNIVERSE_SEMANTIC_EMBED_URL`` nor a ``denseon`` entry in
``INFERENCE_SERVICE_URLS`` fails on its first embedding. Each container the
chart renders on the runtime or dashboard image must therefore resolve the
runtime's embedder URL, read from the same render.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

# Every container rendered on a cogniverse application image. Pinned so a pod
# added on that image is checked here rather than failing on its first
# embedding.
EXPECTED_APP_CONTAINERS = frozenset(
    {
        "Deployment:cogniverse-runtime/runtime",
        "Deployment:cogniverse-quality-monitor/quality-monitor",
        "Deployment:cogniverse-dashboard/dashboard",
        "Deployment:cogniverse-ingestor/ingestor",
        "CronWorkflow:cogniverse-daily-cleanup/cleanup",
        "CronWorkflow:cogniverse-synthetic-generation/generate-synthetic",
        "CronWorkflow:cogniverse-scheduled-distillation/scheduled-distillation",
        "CronWorkflow:cogniverse-annotation-cycle/annotation-cycle",
        "CronWorkflow:cogniverse-annotation-feedback/annotation-feedback",
        "CronWorkflow:cogniverse-monthly-reports/generate-reports",
        "WorkflowTemplate:cogniverse-job-runner/run-job",
        "WorkflowTemplate:cogniverse-optimization-runner/check-profile-ground-truth",
        "WorkflowTemplate:cogniverse-optimization-runner/run-optimizer",
    }
)
APP_IMAGES = ("cogniverse/runtime", "cogniverse/dashboard")

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*set_args: str) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
        "--set",
        "hostStorage.backup.enabled=true",
    ]
    for value in set_args:
        args += ["--set", value]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _containers(manifests: list) -> dict:
    """``"<kind>:<name>/<container>" -> container`` for every rendered pod."""
    found = {}
    for doc in manifests:
        kind = doc.get("kind")
        if kind in ("Deployment", "StatefulSet", "Job"):
            pod = doc["spec"]["template"]["spec"]
            for container in pod.get("containers", []) + pod.get("initContainers", []):
                found[f"{kind}:{doc['metadata']['name']}/{container['name']}"] = (
                    container
                )
        if kind in ("CronWorkflow", "WorkflowTemplate"):
            spec = doc["spec"].get("workflowSpec", doc["spec"])
            for template in spec.get("templates", []):
                container = template.get("container") or template.get("script")
                if container is not None:
                    found[f"{kind}:{doc['metadata']['name']}/{template['name']}"] = (
                        container
                    )
    return found


def _embedder_url(container: dict):
    """The URL the entrypoint resolves: the explicit setting, else the
    service map's ``denseon`` entry; ``None`` when neither is rendered."""
    env = {entry["name"]: entry for entry in container.get("env", [])}
    explicit = env.get("COGNIVERSE_SEMANTIC_EMBED_URL", {}).get("value")
    if explicit:
        return explicit
    services = env.get("INFERENCE_SERVICE_URLS", {}).get("value")
    return json.loads(services).get("denseon") if services else None


@pytest.mark.parametrize(
    "set_args",
    [(), ("inference.denseon.externalUrl=https://denseon.example.modal.run",)],
    ids=["in-cluster", "external"],
)
def test_every_app_container_resolves_the_runtimes_embedder_url(set_args):
    containers = _containers(_render(*set_args))
    app = {
        name: container
        for name, container in containers.items()
        if container.get("image", "").startswith(APP_IMAGES)
    }
    runtime_url = _embedder_url(app["Deployment:cogniverse-runtime/runtime"])

    assert set(app) == EXPECTED_APP_CONTAINERS
    assert runtime_url.startswith(("http://cogniverse-denseon", "https://denseon"))
    assert {name: _embedder_url(c) for name, c in app.items()} == dict.fromkeys(
        EXPECTED_APP_CONTAINERS, runtime_url
    )
