"""Chart wiring for the annotation-identification and feedback CronWorkflows.

Renders the chart and pins:
1. both CronWorkflows exist and invoke ``quality_monitor_cli`` with the right
   one-shot flag;
2. the cron schedules mirror the ``IntervalConfig`` defaults
   (``annotation_interval_minutes`` / ``feedback_interval_minutes``) — the
   config knob and the chart value must not drift apart silently;
3. every pod that submits optimization workflows carries the OPTIMIZATION_*
   env contract ``_workflow_pod_spec_from_env`` reads, so spawned pods run
   the same image/config/source as their submitter instead of the bare
   fallback manifest.
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


def _quality_monitor_sidecar(manifests):
    for doc in manifests:
        if (
            doc.get("kind") == "Deployment"
            and doc.get("metadata", {}).get("name") == "cogniverse-runtime"
        ):
            for container in doc["spec"]["template"]["spec"]["containers"]:
                if container["name"] == "quality-monitor":
                    return container
    raise AssertionError("quality-monitor sidecar not rendered")


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


def test_feedback_cron_carries_workflow_pod_env():
    """The feedback cron submits per-agent compile workflows; the spawned pod
    must inherit the submitter's image and config, not the bare fallback."""
    cron = _cronworkflow(_render_chart(), "cogniverse-annotation-feedback")
    container = cron["spec"]["workflowSpec"]["templates"][0]["container"]
    env = _env_map(container)

    assert env["OPTIMIZATION_WORKFLOW_IMAGE"] == container["image"]
    assert env["OPTIMIZATION_CONFIG_MAP"] == "cogniverse-config"
    assert "OPTIMIZATION_DEV_HOSTPATH" not in env
    # All four passthrough vars _workflow_pod_spec_from_env forwards must be
    # present, or the spawned pod silently loses that backend endpoint.
    assert env["BACKEND_URL"] == "http://cogniverse-vespa"
    assert env["BACKEND_PORT"]
    assert env["TELEMETRY_HTTP_ENDPOINT"]
    assert env["TELEMETRY_OTLP_ENDPOINT"] == "cogniverse-phoenix:4317"


def test_feedback_cron_dev_hostpath_follows_devmode():
    cron = _cronworkflow(
        _render_chart(["devMode.enabled=true", "devMode.hostPath=/cogniverse-src"]),
        "cogniverse-annotation-feedback",
    )
    env = _env_map(cron["spec"]["workflowSpec"]["templates"][0]["container"])
    assert env["OPTIMIZATION_DEV_HOSTPATH"] == "/cogniverse-src"


def test_quality_monitor_sidecar_carries_workflow_pod_env():
    """The sidecar's quality-drop trigger submits workflows too — same env
    contract as the feedback cron."""
    container = _quality_monitor_sidecar(
        _render_chart(["devMode.enabled=true", "devMode.hostPath=/cogniverse-src"])
    )
    env = _env_map(container)

    assert env["OPTIMIZATION_WORKFLOW_IMAGE"] == container["image"]
    assert env["OPTIMIZATION_CONFIG_MAP"] == "cogniverse-config"
    assert env["OPTIMIZATION_DEV_HOSTPATH"] == "/cogniverse-src"
    assert env["BACKEND_URL"] == "http://cogniverse-vespa"
    assert env["BACKEND_PORT"]
    assert env["TELEMETRY_HTTP_ENDPOINT"]
    assert env["TELEMETRY_OTLP_ENDPOINT"] == "cogniverse-phoenix:4317"
