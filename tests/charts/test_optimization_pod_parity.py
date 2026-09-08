"""One definition for the optimization pod: the shared WorkflowTemplate.

Every optimization pod in the release comes from
``charts/cogniverse/templates/optimization-workflow-template.yaml`` --- the
manual ``POST /admin/tenant/{id}/optimize`` route, the scheduled
CronWorkflows, and the Workflows the quality monitor and the annotation
feedback cron submit at runtime. A second hand-built copy loses whatever the
copy forgot, silently: no per-tenant mutex means two triggers for one tenant
stack pods instead of queueing, and no resource block means BestEffort QoS,
first out under memory pressure.

The pins here are the template's side of that contract (mutex, env, resources,
config mount, the parameter set a triggered compile needs) plus the wiring
that lets a submitting pod name it. Expected env is read out of the sibling
WorkflowTemplate in the same render rather than restated, so a values key that
moves takes both templates with it or fails here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

TEMPLATE_NAME = "cogniverse-optimization-runner"
PEER_TEMPLATE_NAME = "cogniverse-job-runner"
RUNNER_TEMPLATE = "run-optimizer"

# The env every optimizer pod needs. BACKEND_* reaches Vespa, TELEMETRY_*
# reaches Phoenix, INFERENCE_SERVICE_URLS + COGNIVERSE_INFERENCE_API_KEY reach
# the embedding services, LLM_* reaches the teacher/student model.
EXPECTED_ENV_NAMES = frozenset(
    {
        "BACKEND_URL",
        "BACKEND_PORT",
        "TELEMETRY_HTTP_ENDPOINT",
        "TELEMETRY_OTLP_ENDPOINT",
        "INFERENCE_SERVICE_URLS",
        "COGNIVERSE_INFERENCE_API_KEY",
        "LLM_ENDPOINT",
        "LLM_ENGINE",
        "LLM_MODEL",
    }
)

EXPECTED_RESOURCES = {
    "requests": {"cpu": "2", "memory": "4Gi"},
    "limits": {"cpu": "4", "memory": "8Gi"},
}

EXPECTED_CONFIG_MOUNT = {
    "name": "config",
    "mountPath": "/app/configs/config.json",
    "subPath": "config.json",
    "readOnly": True,
}

# The env var a submitting pod reads to name the template above. Kept in step
# with cogniverse_runtime.config_loader.OPTIMIZATION_WORKFLOW_TEMPLATE_ENV by
# test_optimization_workflow_delegation.py, which asserts the runtime reads
# this exact name.
TEMPLATE_NAME_ENV = "OPTIMIZATION_WORKFLOW_TEMPLATE"

# Pods that submit optimization Workflows of their own and therefore need the
# template's name. Pinned so a new submitter without the wiring fails here.
EXPECTED_SUBMITTERS = frozenset(
    {
        "cogniverse-annotation-feedback/annotation-feedback",
        "cogniverse-quality-monitor/quality-monitor",
        "cogniverse-scheduled-distillation/scheduled-distillation",
        "cogniverse-runtime/runtime",
    }
)

# Env names that fed the hand-built pod spec the shared template replaces. A
# submitter carrying one again means a second definition came back.
RETIRED_POD_SPEC_ENV = frozenset(
    {
        "OPTIMIZATION_WORKFLOW_IMAGE",
        "OPTIMIZATION_CONFIG_MAP",
        "OPTIMIZATION_DEV_HOSTPATH",
        "OPTIMIZATION_INFERENCE_API_KEY_SECRET",
    }
)


def _render(*set_args: str) -> list:
    helm = shutil.which("helm")
    assert helm is not None, (
        "helm is required to render the chart; the chart suite has no "
        "meaning without it"
    )
    args = [
        helm,
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for value in set_args:
        args += ["--set", value]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _workflow_template(docs: list, name: str) -> dict:
    for doc in docs:
        if doc.get("kind") == "WorkflowTemplate" and doc["metadata"]["name"] == name:
            return doc
    rendered = sorted(
        d["metadata"]["name"] for d in docs if d.get("kind") == "WorkflowTemplate"
    )
    raise AssertionError(f"WorkflowTemplate {name!r} not rendered; got {rendered}")


def _named_template(workflow_template: dict, name: str) -> dict:
    for template in workflow_template["spec"]["templates"]:
        if template["name"] == name:
            return template
    names = [t["name"] for t in workflow_template["spec"]["templates"]]
    raise AssertionError(f"template {name!r} not in {names}")


def _env_map(container: dict) -> dict:
    return {
        entry["name"]: {k: v for k, v in entry.items() if k != "name"}
        for entry in container.get("env", [])
    }


def _all_submitter_candidates(docs: list) -> dict:
    """``"<workload>/<container>" -> container`` for every rendered pod."""
    candidates: dict[str, dict] = {}
    for doc in docs:
        kind = doc.get("kind")
        name = doc["metadata"]["name"]
        if kind in ("CronWorkflow", "WorkflowTemplate"):
            spec = doc["spec"].get("workflowSpec", doc["spec"])
            for template in spec.get("templates", []):
                container = template.get("container")
                if container is not None:
                    candidates[f"{name}/{template['name']}"] = container
        elif kind == "Deployment":
            for container in doc["spec"]["template"]["spec"]["containers"]:
                candidates[f"{name}/{container['name']}"] = container
    return candidates


def test_shared_template_serialises_optimizations_per_tenant():
    """Two triggers for one tenant queue on the mutex instead of stacking two
    optimizer pods; two tenants never block each other. The mutex therefore
    has to interpolate the template's own tenant-id parameter."""
    docs = _render()
    template = _workflow_template(docs, TEMPLATE_NAME)

    declared = [p["name"] for p in template["spec"]["arguments"]["parameters"]]
    tenant_param = "tenant-id"
    assert tenant_param in declared, declared

    mutex = template["spec"]["synchronization"]["mutex"]["name"]
    assert mutex == "optimize-{{workflow.parameters.%s}}" % tenant_param

    # Substituting two tenants must give two different lock names, which a
    # constant mutex (one global optimization at a time) would not.
    def _resolved(tenant: str) -> str:
        return mutex.replace("{{workflow.parameters.%s}}" % tenant_param, tenant)

    assert _resolved("acme:prod") == "optimize-acme:prod"
    assert _resolved("globex:prod") == "optimize-globex:prod"


def test_shared_template_env_matches_the_sibling_runner():
    """Both WorkflowTemplates run the runtime image for background work and
    are wired identically; the expected map is read from the peer in the same
    render so a values key that moves moves both or fails here."""
    docs = _render()
    optimizer = _named_template(
        _workflow_template(docs, TEMPLATE_NAME), RUNNER_TEMPLATE
    )["container"]
    peer = _named_template(_workflow_template(docs, PEER_TEMPLATE_NAME), "run-job")[
        "container"
    ]

    assert set(_env_map(optimizer)) == EXPECTED_ENV_NAMES
    assert _env_map(optimizer) == _env_map(peer)


def test_shared_template_reserves_cpu_and_memory():
    """No resource block means BestEffort QoS: the kubelet evicts the
    optimizer first under memory pressure, mid-compile."""
    docs = _render()
    container = _named_template(
        _workflow_template(docs, TEMPLATE_NAME), RUNNER_TEMPLATE
    )["container"]

    assert container["resources"] == EXPECTED_RESOURCES


def test_shared_template_mounts_the_release_config():
    docs = _render()
    workflow_template = _workflow_template(docs, TEMPLATE_NAME)
    container = _named_template(workflow_template, RUNNER_TEMPLATE)["container"]

    assert container["volumeMounts"] == [EXPECTED_CONFIG_MOUNT]
    assert workflow_template["spec"]["volumes"] == [
        {"name": "config", "configMap": {"name": "cogniverse-config"}}
    ]


def test_shared_template_accepts_a_triggered_compile():
    """A quality-drop or annotation-feedback trigger compiles a named agent
    set against a stored dataset, so ``agents`` and ``trigger-dataset`` are
    part of the template's parameter set; every other mode leaves them empty
    and optimization_cli ignores them."""
    docs = _render()
    workflow_template = _workflow_template(docs, TEMPLATE_NAME)
    runner = _named_template(workflow_template, RUNNER_TEMPLATE)

    assert workflow_template["spec"]["arguments"]["parameters"] == [
        {"name": "mode"},
        {"name": "tenant-id"},
        {"name": "lookback-hours", "value": "48"},
        {"name": "agents", "value": ""},
        {"name": "trigger-dataset", "value": ""},
    ]
    assert runner["inputs"]["parameters"] == [
        {"name": "mode"},
        {"name": "tenant-id"},
        {"name": "lookback-hours"},
        {"name": "agents", "value": ""},
        {"name": "trigger-dataset", "value": ""},
    ]
    assert runner["container"]["command"] == [
        "python",
        "-m",
        "cogniverse_runtime.optimization_cli",
    ]
    assert runner["container"]["args"] == [
        "--mode",
        "{{inputs.parameters.mode}}",
        "--tenant-id",
        "{{inputs.parameters.tenant-id}}",
        "--lookback-hours",
        "{{inputs.parameters.lookback-hours}}",
        "--agents",
        "{{inputs.parameters.agents}}",
        "--trigger-dataset",
        "{{inputs.parameters.trigger-dataset}}",
    ]


def test_every_submitter_is_told_which_template_to_reference():
    """A submitter without the name has nothing to reference and would have to
    build a pod spec of its own."""
    docs = _render()
    rendered_name = _workflow_template(docs, TEMPLATE_NAME)["metadata"]["name"]

    named = {
        key: _env_map(container).get(TEMPLATE_NAME_ENV, {}).get("value")
        for key, container in _all_submitter_candidates(docs).items()
        if TEMPLATE_NAME_ENV in _env_map(container)
    }

    assert named == dict.fromkeys(EXPECTED_SUBMITTERS, rendered_name)


def test_no_pod_carries_the_retired_hand_built_spec_wiring():
    """The image / config-map / hostPath / bearer-secret names existed only to
    let a submitter assemble its own container spec."""
    docs = _render("devMode.enabled=true", "devMode.hostPath=/cogniverse-src")

    offenders = {
        key: sorted(RETIRED_POD_SPEC_ENV & set(_env_map(container)))
        for key, container in _all_submitter_candidates(docs).items()
        if RETIRED_POD_SPEC_ENV & set(_env_map(container))
    }

    assert offenders == {}
