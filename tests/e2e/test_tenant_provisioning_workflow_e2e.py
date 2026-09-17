"""The shipped tenant-provisioning workflow runs on the live cluster.

``workflows/tenant-provisioning.yaml`` is a WorkflowTemplate the deploy CLI
installs; the e2e cluster carries only the chart's own templates, so this
module renders a one-off Workflow from the shipped file and submits that —
the same spec, launched the same way the template would launch it.

The workflow creates a Kubernetes namespace, so the test owns the namespace
it names and deletes it, its Workflow object and its tenant on teardown.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from cogniverse_core.memory.provenance_store import PROVENANCE_BASE_SCHEMA
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_runtime import provision_tenant
from cogniverse_runtime.memory_init import MEMORY_BASE_SCHEMA
from tests.e2e.conftest import (
    KUBECTL_CONTEXT,
    SAMPLE_VIDEO_PATH,
    _deployed_schema_names_strict,
    _ensure_sample_content_ingested,
    _sample_video_media_type,
    _search_sample_content,
    _tenant_schema_name,
    _tenant_schema_names_in_vespa,
    register_tenant_and_wait,
    unique_id,
)

pytestmark = [pytest.mark.e2e]

NAMESPACE = "cogniverse"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROVISIONING_WORKFLOW = REPO_ROOT / "workflows" / "tenant-provisioning.yaml"
CONFIG_PATH = REPO_ROOT / "configs" / "config.json"
STANDALONE_SCRIPT = "scripts/provision_tenant.py"

# Every container step pulls the runtime image and the schema step recompiles
# the whole Vespa application package; measured tenant deploys on this cluster
# run 35-90 s each and the workflow performs one.
PROVISIONING_TIMEOUT_S = 1800.0
POLL_INTERVAL_S = 5.0

# The entry point the runtime image installs. The image ships the packages and
# ``configs/`` only, so a step invoking anything else cannot run in it.
PROVISION_ENTRYPOINT = ["python", "-m", provision_tenant.__name__]


def _kubectl(*args: str, timeout: int = 60, stdin: str | None = None):
    return subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _require_kubectl(result, what: str):
    if result.returncode != 0:
        pytest.fail(
            f"{what} failed: exit={result.returncode}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
            pytrace=False,
        )
    return result


def _workflow_template_spec() -> dict:
    """The WorkflowTemplate spec the repository ships."""
    documents = [
        document
        for document in yaml.safe_load_all(PROVISIONING_WORKFLOW.read_text())
        if document and document.get("kind") == "WorkflowTemplate"
    ]
    assert len(documents) == 1, (
        f"{PROVISIONING_WORKFLOW} must declare exactly one WorkflowTemplate; "
        f"found {len(documents)}"
    )
    return documents[0]["spec"]


def _declared_parameters(spec: dict) -> dict[str, str]:
    return {p["name"]: p.get("value", "") for p in spec["arguments"]["parameters"]}


def _declared_steps(spec: dict) -> list[str]:
    """The pipeline's step names, in the order the entrypoint declares them."""
    entrypoint = next(t for t in spec["templates"] if t["name"] == spec["entrypoint"])
    return [step["name"] for group in entrypoint["steps"] for step in group]


def _deployed_runtime_image() -> str:
    """The runtime image the cluster already runs, so no step pulls a new one."""
    read = _require_kubectl(
        _kubectl(
            "-n",
            NAMESPACE,
            "get",
            "deployment/cogniverse-runtime",
            "-o",
            "jsonpath={.spec.template.spec.containers[0].image}",
        ),
        "reading the deployed runtime image",
    )
    image = read.stdout.strip()
    assert image, "the runtime Deployment declares no container image"
    return image


def _profile_schema_names(profiles: list[str]) -> list[str]:
    """Each profile's base schema name from the cluster's shipped catalog."""
    catalog = json.loads(CONFIG_PATH.read_text())["backend"]["profiles"]
    return [catalog[name]["schema_name"] for name in profiles]


def _memory_step_schema_names(spec: dict) -> list[str]:
    """Base schemas the workflow's memory step deploys for the tenant.

    The step runs ``provision_tenant --step memory``, which initialises Mem0
    with its memory schema and deploys the provenance schema beside it.
    """
    memory_steps = [
        template["name"]
        for template in spec["templates"]
        if template.get("container", {}).get("args", [])[:2] == ["--step", "memory"]
    ]
    assert memory_steps == ["initialize-memory"], memory_steps
    assert "initialize-memory" in _declared_steps(spec), _declared_steps(spec)
    return [MEMORY_BASE_SCHEMA, PROVENANCE_BASE_SCHEMA]


def _workflow_status(name: str) -> dict:
    read = _kubectl("-n", NAMESPACE, "get", "workflow", name, "-o", "json")
    if read.returncode != 0:
        return {}
    return json.loads(read.stdout).get("status", {}) or {}


def _workflow_logs(name: str) -> str:
    read = _kubectl(
        "logs",
        "-n",
        NAMESPACE,
        "-l",
        f"workflows.argoproj.io/workflow={name}",
        "--all-containers",
        "--tail=-1",
        timeout=120,
    )
    return read.stdout or read.stderr or ""


def _step_phases(status: dict, declared_steps: list[str]) -> dict[str, str]:
    """Phase of each declared pipeline step, keyed by the step's own name."""
    phases: dict[str, str] = {}
    for node in (status.get("nodes") or {}).values():
        name = node.get("displayName", "")
        if name in declared_steps:
            phases[name] = node.get("phase", "")
    return phases


@pytest.fixture(scope="module")
def provisioned_tenant():
    """Run the shipped provisioning workflow for a tenant this module owns.

    Yields the submitted Workflow's terminal status together with everything
    the assertions derive from: the tenant, its profiles and the schema names
    the profiles resolve to.
    """
    spec = _workflow_template_spec()
    parameters = _declared_parameters(spec)
    profiles = [name for name in parameters["profiles"].split(",") if name]
    assert profiles, (
        f"{PROVISIONING_WORKFLOW} declares no default profiles to provision"
    )

    tenant = unique_id("prode2eclients")
    canonical = canonical_tenant_id(tenant)
    register_tenant_and_wait(tenant, created_by="e2e")
    schemas_before = _tenant_schema_names_in_vespa(
        canonical, _deployed_schema_names_strict()
    )

    submitted = dict(spec)
    submitted["arguments"] = {
        "parameters": [
            {
                "name": name,
                "value": {
                    "tenant-id": tenant,
                    "tenant-name": tenant,
                    "runtime-image": _deployed_runtime_image(),
                }.get(name, value),
            }
            for name, value in parameters.items()
        ]
    }
    manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": "tenant-provisioning-e2e-",
            "namespace": NAMESPACE,
        },
        "spec": submitted,
    }
    created = _require_kubectl(
        _kubectl(
            "create",
            "-n",
            NAMESPACE,
            "-f",
            "-",
            "-o",
            "json",
            stdin=json.dumps(manifest),
        ),
        "submitting the tenant-provisioning workflow",
    )
    workflow_name = json.loads(created.stdout)["metadata"]["name"]
    # A namespace name allows no underscores; the workflow maps the tenant
    # id's underscores to hyphens.
    tenant_namespace = f"cogniverse-{tenant.replace('_', '-')}"

    try:
        deadline = time.monotonic() + PROVISIONING_TIMEOUT_S
        status: dict = {}
        while time.monotonic() < deadline:
            status = _workflow_status(workflow_name)
            if status.get("phase") in {"Succeeded", "Failed", "Error"}:
                break
            time.sleep(POLL_INTERVAL_S)
        yield {
            "spec": spec,
            "workflow": workflow_name,
            "status": status,
            "logs": _workflow_logs(workflow_name),
            "tenant": tenant,
            "canonical": canonical,
            "namespace": tenant_namespace,
            "profiles": profiles,
            "schemas_before": schemas_before,
        }
    finally:
        _kubectl(
            "delete",
            "namespace",
            tenant_namespace,
            "--ignore-not-found",
            "--wait=false",
        )
        _kubectl("delete", "workflow", workflow_name, "-n", NAMESPACE, "--wait=false")


def test_every_container_step_runs_the_installed_provisioning_entry_point():
    """No step shells out to the checkout; the image ships packages only.

    The image's final stage carries the installed packages and ``configs/``
    and nothing else — no ``uv``, no ``scripts/`` — so a step invoking either
    fails before it reaches Vespa.
    """
    spec = _workflow_template_spec()
    runtime_steps = {
        template["name"]: template["container"]
        for template in spec["templates"]
        if "container" in template
        and template["container"]["image"] == "{{workflow.parameters.runtime-image}}"
    }
    assert runtime_steps, "the pipeline declares no step running the runtime image"

    # Keyed by the provisioning step each container asks for, so the pin is
    # that the CLI's whole step vocabulary is launched, each through the entry
    # point the image installs.
    by_step: dict[str, list[str]] = {}
    for container in runtime_steps.values():
        args = container.get("args") or []
        if "--step" in args:
            by_step[args[args.index("--step") + 1]] = container["command"]
    assert by_step == {
        step: PROVISION_ENTRYPOINT
        for step in provision_tenant._STEPS  # noqa: SLF001
    }, by_step

    rendered = PROVISIONING_WORKFLOW.read_text()
    assert STANDALONE_SCRIPT not in rendered, (
        f"{PROVISIONING_WORKFLOW} still runs {STANDALONE_SCRIPT}, which the "
        "runtime image does not ship"
    )
    assert "uv run" not in rendered, (
        f"{PROVISIONING_WORKFLOW} still invokes uv, which the runtime image "
        "does not ship"
    )


def test_the_provisioning_workflow_completes_every_declared_step(provisioned_tenant):
    """Each step of the shipped pipeline reaches Succeeded, none is skipped."""
    status = provisioned_tenant["status"]
    declared = _declared_steps(provisioned_tenant["spec"])
    phases = _step_phases(status, declared)

    assert status.get("phase") == "Succeeded", (
        f"workflow {provisioned_tenant['workflow']} phase="
        f"{status.get('phase')!r} message={status.get('message')!r}\n"
        f"step phases={phases}\n--- logs ---\n"
        f"{provisioned_tenant['logs'][-6000:]}"
    )
    assert phases == {step: "Succeeded" for step in declared}, phases


def test_the_schema_step_deploys_each_profiles_tenant_schema(provisioned_tenant):
    """The profiles land in Vespa under their tenant-scoped names.

    The schema step resolves each profile's ``schema_name`` through the
    tenant's merged catalog and deploys it into the live application package,
    and the memory step deploys the memory and provenance schemas, so the
    tenant gains exactly those schemas and keeps the ones it had.
    """
    canonical = provisioned_tenant["canonical"]
    base_names = _profile_schema_names(provisioned_tenant["profiles"])
    memory_names = _memory_step_schema_names(_workflow_template_spec())
    assert memory_names == ["agent_memories", "provenance"]
    expected = {
        _tenant_schema_name(base, canonical) for base in base_names + memory_names
    }

    deployed = _tenant_schema_names_in_vespa(canonical, _deployed_schema_names_strict())
    assert deployed == provisioned_tenant["schemas_before"] | expected, (
        f"tenant schemas after provisioning: {sorted(deployed)}; "
        f"before: {sorted(provisioned_tenant['schemas_before'])}; "
        f"expected to gain: {sorted(expected)}"
    )


def test_the_verify_step_reports_every_schema_it_queried(provisioned_tenant):
    """The verify step names the tenant schemas it proved queryable.

    Its line is the only place the workflow reports what it checked, so a
    verify that registered a row without querying Vespa, or that ran against
    the profile name instead of the schema name, changes it.
    """
    canonical = provisioned_tenant["canonical"]
    base_names = _profile_schema_names(provisioned_tenant["profiles"])
    reported = ", ".join(_tenant_schema_name(base, canonical) for base in base_names)

    logs = provisioned_tenant["logs"]
    assert f"Provisioned verify for tenant {canonical}: {reported}" in logs, (
        f"the verify step did not report {reported!r}\n--- logs ---\n{logs[-6000:]}"
    )
    assert "uv: not found" not in logs, logs[-4000:]
    assert STANDALONE_SCRIPT not in logs, logs[-4000:]


def test_the_workflow_creates_the_tenants_namespace_with_its_labels(
    provisioned_tenant,
):
    """The namespace step creates the tenant namespace the manifest declares."""
    read = _require_kubectl(
        _kubectl(
            "get",
            "namespace",
            provisioned_tenant["namespace"],
            "-o",
            "jsonpath={.metadata.labels}",
        ),
        f"reading namespace {provisioned_tenant['namespace']}",
    )
    labels = json.loads(read.stdout)
    assert labels["tenant"] == provisioned_tenant["tenant"]
    assert labels["managed-by"] == "cogniverse"


def test_the_provisioned_profile_serves_the_content_ingested_into_it(
    provisioned_tenant,
):
    """A search on the provisioned schema returns the exact content fed to it."""
    tenant = provisioned_tenant["canonical"]
    profile = provisioned_tenant["profiles"][0]
    media_type = _sample_video_media_type(SAMPLE_VIDEO_PATH)

    content_id = _ensure_sample_content_ingested(
        SAMPLE_VIDEO_PATH,
        profile=profile,
        media_type=media_type,
        tenant_id=tenant,
    )
    matches, error = _search_sample_content(
        content_id=content_id,
        tenant_id=tenant,
        profile=profile,
        suffix=SAMPLE_VIDEO_PATH.suffix,
        media_type=media_type,
    )
    assert error is None, error
    assert matches, (
        f"the provisioned {profile} schema served no document for {content_id}"
    )
    assert {match["source_id"] for match in matches} == {content_id}
