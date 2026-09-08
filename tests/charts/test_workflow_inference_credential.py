"""Every Argo pod resolves the inference bearer the way the runtime does.

The credential reaches an optimization pod two ways, and both are pinned here:

* chart-rendered ``CronWorkflow`` / ``WorkflowTemplate`` containers carry the
  entry ``cogniverse.inferenceApiKeyEnv`` renders;
* workflows a running pod submits to the Argo API get their container spec
  from ``_workflow_pod_spec_from_env``, which turns
  ``OPTIMIZATION_INFERENCE_API_KEY_SECRET`` into a ``secretKeyRef`` — so the
  reference travels and the bearer itself never lands in a Workflow manifest.

Expectations are read out of the same render (the runtime Deployment's own
entry, the Secret that entry names), never restated, so a change to the values
key behind the helper moves both sides together or fails here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

# Every container template the chart renders into an Argo workflow. Pinned so
# a workflow added without the credential fails here instead of at 3 AM.
EXPECTED_WORKFLOW_CONTAINERS = frozenset(
    {
        "cogniverse-agent-optimization/restart-deployment",
        "cogniverse-annotation-cycle/annotation-cycle",
        "cogniverse-annotation-feedback/annotation-feedback",
        "cogniverse-backup-phoenix/dump",
        "cogniverse-backup-phoenix/upload",
        "cogniverse-backup-vespa/dump",
        "cogniverse-backup-vespa/upload",
        "cogniverse-daily-cleanup/cleanup",
        "cogniverse-job-runner/run-job",
        "cogniverse-monthly-reports/generate-reports",
        "cogniverse-monthly-reports/upload-reports",
        "cogniverse-optimization-runner/run-optimizer",
        "cogniverse-scheduled-distillation/scheduled-distillation",
        "cogniverse-synthetic-generation/generate-synthetic",
    }
)

# Pods that submit workflows of their own; each mirrors its wiring onto the
# pod it spawns via the OPTIMIZATION_* contract.
EXPECTED_SUBMITTERS = frozenset(
    {
        "cogniverse-annotation-feedback/annotation-feedback",
        "cogniverse-quality-monitor/quality-monitor",
        "cogniverse-scheduled-distillation/scheduled-distillation",
    }
)

CREDENTIAL = "COGNIVERSE_INFERENCE_API_KEY"
SECRET_NAME_ENV = "OPTIMIZATION_INFERENCE_API_KEY_SECRET"

# Backups are off by default; the two data-backup CronWorkflows run the same
# runtime image and need the same credential, so every render here enables
# them rather than leaving four workflows unchecked.
BACKUPS_ON = "hostStorage.backup.enabled=true"
EXTERNAL_INFERENCE = "inference.vllm_llm_teacher.externalUrl=https://example.modal.run"

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
    ]
    for value in set_args:
        args += ["--set", value]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _env_map(container: dict) -> dict:
    return {entry["name"]: entry for entry in container.get("env", [])}


def _credential_source(container: dict) -> dict:
    """The ``value`` / ``valueFrom`` body of a container's bearer entry.

    An empty dict when the container has no such entry, so a missing
    credential reads as a value difference naming the container rather than a
    KeyError from the collection step.
    """
    entry = _env_map(container).get(CREDENTIAL, {})
    return {key: value for key, value in entry.items() if key != "name"}


def _secret_name_handed_on(container: dict) -> str:
    """The Secret a submitter names for the pods it spawns, "" when it names
    none — so a submitter that lost the wiring shows up as a value difference
    under its own key instead of a KeyError.
    """
    return _env_map(container).get(SECRET_NAME_ENV, {}).get("value", "")


def _workflow_containers(manifests: list) -> dict:
    """``"<workflow>/<template>" -> container`` for every rendered Argo pod."""
    containers = {}
    for doc in manifests:
        if doc.get("kind") not in ("CronWorkflow", "WorkflowTemplate"):
            continue
        spec = doc["spec"].get("workflowSpec", doc["spec"])
        for template in spec.get("templates", []):
            container = template.get("container")
            if container is not None:
                name = f"{doc['metadata']['name']}/{template['name']}"
                containers[name] = container
    return containers


def _runtime_container(manifests: list) -> dict:
    for doc in manifests:
        if (
            doc.get("kind") == "Deployment"
            and doc["metadata"]["name"] == "cogniverse-runtime"
        ):
            for container in doc["spec"]["template"]["spec"]["containers"]:
                if container["name"] == "runtime":
                    return container
    raise AssertionError("cogniverse-runtime Deployment not rendered")


def _submitter_containers(manifests: list) -> dict:
    """Containers wired to submit optimization workflows of their own."""
    submitters = {}
    candidates = dict(_workflow_containers(manifests))
    for doc in manifests:
        if doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            candidates[f"{doc['metadata']['name']}/{container['name']}"] = container
    for name, container in candidates.items():
        if "OPTIMIZATION_WORKFLOW_IMAGE" in _env_map(container):
            submitters[name] = container
    return submitters


def test_in_cluster_workflows_share_the_runtime_credential_entry():
    """With every inference service in-cluster there is no Secret to read, so
    the runtime carries the no-auth placeholder inline — and so must every
    workflow pod, or its optimizer steps abort on a missing bearer."""
    manifests = _render(BACKUPS_ON)
    runtime_source = _credential_source(_runtime_container(manifests))

    assert set(runtime_source) == {"value"}

    found = {
        name: _credential_source(container)
        for name, container in _workflow_containers(manifests).items()
    }
    assert found == dict.fromkeys(EXPECTED_WORKFLOW_CONTAINERS, runtime_source)


def test_external_inference_workflows_share_the_runtime_secret_reference():
    """One external inference endpoint switches the runtime onto the synced
    Secret. Every workflow pod must switch with it, to the same Secret and the
    same key — a workflow left on the placeholder authenticates as nobody."""
    manifests = _render(BACKUPS_ON, EXTERNAL_INFERENCE)
    runtime_source = _credential_source(_runtime_container(manifests))
    secret_ref = runtime_source["valueFrom"]["secretKeyRef"]

    assert set(runtime_source) == {"valueFrom"}
    assert secret_ref["key"] == CREDENTIAL
    assert secret_ref["optional"] is False

    found = {
        name: _credential_source(container)
        for name, container in _workflow_containers(manifests).items()
    }
    assert found == dict.fromkeys(EXPECTED_WORKFLOW_CONTAINERS, runtime_source)


def test_submitters_pass_the_runtime_secret_name_to_spawned_pods():
    """A submitted Workflow's container spec is built in Python from this env
    var. It names the Secret, never the bearer: the value stays out of the
    Workflow object the Argo API stores and serves."""
    manifests = _render(BACKUPS_ON, EXTERNAL_INFERENCE)
    secret_name = _credential_source(_runtime_container(manifests))["valueFrom"][
        "secretKeyRef"
    ]["name"]

    submitters = _submitter_containers(manifests)
    assert set(submitters) == EXPECTED_SUBMITTERS

    named = {
        name: _secret_name_handed_on(container)
        for name, container in submitters.items()
    }
    assert named == dict.fromkeys(EXPECTED_SUBMITTERS, secret_name)


def test_in_cluster_submitters_forward_their_own_placeholder():
    """No Secret exists in a fully in-cluster render, so no submitter may name
    one; the spawned pod inherits the submitter's own placeholder entry, which
    is the runtime's."""
    manifests = _render(BACKUPS_ON)
    runtime_source = _credential_source(_runtime_container(manifests))

    submitters = _submitter_containers(manifests)
    assert set(submitters) == EXPECTED_SUBMITTERS

    named = {
        name: _secret_name_handed_on(container)
        for name, container in submitters.items()
    }
    assert named == dict.fromkeys(EXPECTED_SUBMITTERS, "")

    own = {
        name: _credential_source(container) for name, container in submitters.items()
    }
    assert own == dict.fromkeys(EXPECTED_SUBMITTERS, runtime_source)
