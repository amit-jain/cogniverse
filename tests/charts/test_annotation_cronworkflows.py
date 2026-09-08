"""Chart wiring for the annotation-identification and feedback CronWorkflows.

Renders the chart and pins:
1. both CronWorkflows exist and invoke ``quality_monitor_cli`` with the right
   one-shot flag;
2. the cron schedules mirror the ``IntervalConfig`` defaults
   (``annotation_interval_minutes`` / ``feedback_interval_minutes``) — the
   config knob and the chart value must not drift apart silently;
3. every pod that submits optimization workflows is told the name of the
   shared optimization WorkflowTemplate, which owns the spawned pod's
   container spec, env, resources and per-tenant mutex.
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


def _render_chart(extra_sets: list | None = None) -> list:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for value in extra_sets or []:
        args += ["--set", value]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _cronworkflow(manifests, name):
    for doc in manifests:
        if (
            doc.get("kind") == "CronWorkflow"
            and doc.get("metadata", {}).get("name") == name
        ):
            return doc
    raise AssertionError(f"CronWorkflow {name} not rendered")


def _container_args(cron):
    return cron["spec"]["workflowSpec"]["templates"][0]["container"]["args"]


def _env_map(container):
    return {e["name"]: e.get("value") for e in container.get("env", [])}


# Env names that fed the hand-built pod spec the shared template replaces.
RETIRED_POD_SPEC_ENV = frozenset(
    {
        "OPTIMIZATION_WORKFLOW_IMAGE",
        "OPTIMIZATION_CONFIG_MAP",
        "OPTIMIZATION_DEV_HOSTPATH",
        "OPTIMIZATION_INFERENCE_API_KEY_SECRET",
    }
)


def _workflow_template_name(manifests):
    """The rendered WorkflowTemplate every submitter must name."""
    for doc in manifests:
        if doc.get("kind") == "WorkflowTemplate" and doc["metadata"]["name"].endswith(
            "-optimization-runner"
        ):
            return doc["metadata"]["name"]
    raise AssertionError("no optimization-runner WorkflowTemplate rendered")


def _quality_monitor_container(manifests):
    for doc in manifests:
        if (
            doc.get("kind") == "Deployment"
            and doc.get("metadata", {}).get("name") == "cogniverse-quality-monitor"
        ):
            containers = doc["spec"]["template"]["spec"]["containers"]
            assert [c["name"] for c in containers] == ["quality-monitor"]
            return containers[0]
    raise AssertionError("cogniverse-quality-monitor Deployment not rendered")


def test_annotation_cycle_cronworkflow_invokes_the_flag():
    cron = _cronworkflow(_render_chart(), "cogniverse-annotation-cycle")
    args = _container_args(cron)
    assert "--annotation-cycle" in args
    assert "--runtime-url" in args
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"


def test_annotation_feedback_cronworkflow_invokes_the_flag():
    cron = _cronworkflow(_render_chart(), "cogniverse-annotation-feedback")
    args = _container_args(cron)
    assert "--annotation-feedback" in args
    assert "--argo-url" in args


def test_every_cron_processes_the_configured_tenant():
    """Every tenant-carrying CronWorkflow must run for
    runtime.qualityMonitor.tenantId — a hardcoded tenant silently processes
    the wrong tenant on every deployment that sets a different one. Sweeps
    both carriers: container ``--tenant-id`` args and workflow
    ``tenant-id`` parameters."""
    tenants = {}
    for doc in _render_chart():
        if doc.get("kind") != "CronWorkflow":
            continue
        spec = doc["spec"]["workflowSpec"]
        values = [
            param.get("value")
            for param in spec.get("arguments", {}).get("parameters", [])
            if param.get("name") == "tenant-id"
        ]
        for template in spec.get("templates", []):
            args = template.get("container", {}).get("args", [])
            if "--tenant-id" in args:
                values.append(args[args.index("--tenant-id") + 1])
        if values:
            tenants[doc["metadata"]["name"]] = values
    assert len(tenants) >= 6, f"tenant-carrying crons went missing: {tenants}"
    # __system__ is the deliberate cross-tenant maintenance identity (the
    # daily gateway pipeline runs against the system tenant), not a
    # hardcoded per-tenant value.
    for name, values in tenants.items():
        for value in values:
            assert value in ("test-tenant", "__system__"), f"{name}: {values}"


def test_argo_subchart_disabled_while_crons_render():
    """The in-release argo-workflows subchart duplicates the standalone Argo
    install every workflow URL points at (argo-server.argo.svc); rendering
    it just ships permanently-broken duplicate pods. argo.enabled must keep
    gating the CronWorkflows without dragging the subchart in."""
    manifests = _render_chart()
    subchart_workloads = [
        doc["metadata"]["name"]
        for doc in manifests
        if doc.get("kind") in ("Deployment", "StatefulSet")
        and "argo-workflows" in doc.get("metadata", {}).get("name", "")
    ]
    assert subchart_workloads == [], subchart_workloads
    _cronworkflow(manifests, "cogniverse-annotation-feedback")


def test_schedules_mirror_interval_config_defaults():
    """The cron cadence and the IntervalConfig knob are one contract: a change
    to either without the other silently de-syncs the loop's documented
    behavior from its actual schedule."""
    from cogniverse_agents.routing.config import IntervalConfig

    intervals = IntervalConfig()
    manifests = _render_chart()

    cycle = _cronworkflow(manifests, "cogniverse-annotation-cycle")
    assert (
        cycle["spec"]["schedule"]
        == f"*/{intervals.annotation_interval_minutes} * * * *"
    )

    feedback = _cronworkflow(manifests, "cogniverse-annotation-feedback")
    assert (
        feedback["spec"]["schedule"]
        == f"*/{intervals.feedback_interval_minutes} * * * *"
    )


def test_feedback_cron_names_the_shared_workflow_template():
    """The feedback cron submits per-agent compile workflows; each references
    the shared template rather than carrying a container spec."""
    manifests = _render_chart()
    cron = _cronworkflow(manifests, "cogniverse-annotation-feedback")
    container = cron["spec"]["workflowSpec"]["templates"][0]["container"]
    env = _env_map(container)

    assert env["OPTIMIZATION_WORKFLOW_TEMPLATE"] == _workflow_template_name(manifests)
    assert RETIRED_POD_SPEC_ENV & set(env) == set()
    # Submitting a Workflow needs the RBAC the chart binds to this account.
    assert cron["spec"]["workflowSpec"]["serviceAccountName"] == "cogniverse"
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"
    # The cron pod's own wiring: quality_monitor_cli reads spans and writes
    # trigger datasets before it submits anything.
    assert env["BACKEND_URL"] == "http://cogniverse-vespa"
    assert env["BACKEND_PORT"] == "8080"
    assert env["TELEMETRY_HTTP_ENDPOINT"] == "http://cogniverse-phoenix:6006"
    assert env["TELEMETRY_OTLP_ENDPOINT"] == "cogniverse-phoenix:4317"


def test_devmode_does_not_change_what_a_submitter_hands_on():
    """devMode source mounts belong to the shared template, so the submitter
    still passes one name and nothing else."""
    manifests = _render_chart(
        ["devMode.enabled=true", "devMode.hostPath=/cogniverse-src"]
    )
    cron = _cronworkflow(manifests, "cogniverse-annotation-feedback")
    env = _env_map(cron["spec"]["workflowSpec"]["templates"][0]["container"])

    assert env["OPTIMIZATION_WORKFLOW_TEMPLATE"] == _workflow_template_name(manifests)
    assert RETIRED_POD_SPEC_ENV & set(env) == set()
    # devMode mounts belong to the referenced template, so the name a
    # submitter hands on is the same one it hands on without devMode.
    assert _workflow_template_name(manifests) == _workflow_template_name(
        _render_chart()
    )


def test_every_optimization_submitter_names_one_template():
    """One template name across every submitting pod: a second name would be
    a second pod definition."""
    manifests = _render_chart()
    named = {}
    for doc in manifests:
        kind = doc.get("kind")
        if kind == "Deployment":
            pods = [
                (c["name"], c) for c in doc["spec"]["template"]["spec"]["containers"]
            ]
        elif kind == "CronWorkflow":
            pods = [
                (t["name"], t["container"])
                for t in doc["spec"]["workflowSpec"]["templates"]
                if "container" in t
            ]
        else:
            continue
        for name, container in pods:
            env = _env_map(container)
            if "OPTIMIZATION_WORKFLOW_TEMPLATE" in env:
                named[f"{doc['metadata']['name']}/{name}"] = env[
                    "OPTIMIZATION_WORKFLOW_TEMPLATE"
                ]

    assert set(named) == {
        "cogniverse-annotation-feedback/annotation-feedback",
        "cogniverse-quality-monitor/quality-monitor",
        "cogniverse-runtime/runtime",
        "cogniverse-scheduled-distillation/scheduled-distillation",
    }
    assert set(named.values()) == {_workflow_template_name(manifests)}


def test_scheduled_distillation_names_the_shared_workflow_template():
    manifests = _render_chart(
        [
            "runtime.imagesByBackend.cuda.repository=registry.local/cogniverse-runtime",
            "runtime.imagesByBackend.cuda.tag=e2e-exact",
            "devMode.enabled=true",
            "devMode.hostPath=/cogniverse-src",
        ]
    )
    cron = _cronworkflow(manifests, "cogniverse-scheduled-distillation")
    container = cron["spec"]["workflowSpec"]["templates"][0]["container"]
    env = _env_map(container)

    assert container["image"] == "registry.local/cogniverse-runtime:e2e-exact"
    assert env["OPTIMIZATION_WORKFLOW_TEMPLATE"] == _workflow_template_name(manifests)
    assert RETIRED_POD_SPEC_ENV & set(env) == set()
    assert env["BACKEND_URL"] == "http://cogniverse-vespa"
    assert env["BACKEND_PORT"] == "8080"
    assert env["TELEMETRY_HTTP_ENDPOINT"] == "http://cogniverse-phoenix:6006"
    assert env["TELEMETRY_OTLP_ENDPOINT"] == "cogniverse-phoenix:4317"


def test_annotation_crons_fall_back_to_default_tenant():
    """With qualityMonitor.enabled=false its tenantId fail-guard never runs,
    so an empty tenantId (the chart default) must fall back to "default"
    like the sibling crons do — an empty --tenant-id makes both annotation
    crons no-op against an empty-string tenant project."""
    manifests = _render_chart(
        [
            "runtime.qualityMonitor.enabled=false",
            "runtime.qualityMonitor.tenantId=",
        ]
    )
    for name in ("cogniverse-annotation-cycle", "cogniverse-annotation-feedback"):
        args = _container_args(_cronworkflow(manifests, name))
        tenant = args[args.index("--tenant-id") + 1]
        assert tenant == "default", f"{name}: --tenant-id {tenant!r}"


def test_quality_monitor_names_the_shared_workflow_template():
    """The monitor's quality-drop trigger submits workflows too — same
    contract as the feedback cron."""
    manifests = _render_chart(
        ["devMode.enabled=true", "devMode.hostPath=/cogniverse-src"]
    )
    container = _quality_monitor_container(manifests)
    env = _env_map(container)

    assert env["OPTIMIZATION_WORKFLOW_TEMPLATE"] == _workflow_template_name(manifests)
    assert RETIRED_POD_SPEC_ENV & set(env) == set()
    assert env["BACKEND_URL"] == "http://cogniverse-vespa"
    assert env["BACKEND_PORT"] == "8080"
    assert env["TELEMETRY_HTTP_ENDPOINT"] == "http://cogniverse-phoenix:6006"
    assert env["TELEMETRY_OTLP_ENDPOINT"] == "cogniverse-phoenix:4317"
