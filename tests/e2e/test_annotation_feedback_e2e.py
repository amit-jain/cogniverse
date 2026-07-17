"""End-to-end annotation-feedback loop against the live cluster — real Argo.

No capture clients anywhere: a fresh per-run tenant is provisioned through the
runtime's admin API (tenant + video profile + schema deploy), real human
annotations are seeded in the in-cluster Phoenix, the chart's
``annotation-feedback`` CronWorkflow (rendered for that tenant) is submitted
as a one-off Workflow to the real Argo server, the workflow pod runs the real
CLI against the in-cluster Phoenix + Vespa, the CLI's own Argo submission
creates a REAL second workflow (the gateway-thresholds recompile), and the
loop's cooldown state lands in the in-cluster Vespa config store.

The isolated tenant makes the optimization's OUTPUT exactly assertable: the
calibration window holds only this run's seeds, so the persisted analysis,
both recomputed thresholds, and the values the live gateway then serves (read
back from the probe's routing span) are all pinned to exact numbers.

Requires the deployed k3d stack (runtime/phoenix/vespa NodePorts) and the
Argo install — the same environment every test in this tier targets.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id

NAMESPACE = "cogniverse"
PHOENIX_HTTP = "http://localhost:26006"  # phoenix.service nodePort
PHOENIX_GRPC = "localhost:4317"  # phoenix OTLP nodePort
VESPA_URL = "http://localhost"  # vespa.service nodePort
VESPA_PORT = 8080
RUNTIME_URL = "http://localhost:28000"  # runtime.service nodePort
# Per-run tenant: seeds, calibration windows, artifacts, and loop state are
# isolated from real tenants and prior runs, so every measurement the compile
# persists is exactly derivable from this run's seeds. The runtime keys
# everything by the canonical form; seeds and readers must match it.
RUN_ID = uuid4().hex[:8]
TENANT = f"e2e_{RUN_ID}"
CANONICAL_TENANT = canonical_tenant_id(TENANT)
FEEDBACK_CRON = "cogniverse-annotation-feedback"
SPAWNED_LABEL = "trigger=annotation-feedback"
POLL_INTERVAL_S = 5.0
WORKFLOW_TIMEOUT_S = 600.0

pytestmark = pytest.mark.skipif(
    shutil.which("kubectl") is None or shutil.which("helm") is None,
    reason="kubectl/helm CLI not installed — cluster e2e requires both",
)


def _kubectl(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=60,
        input=input_text,
    )


def _deployed_runtime_settings() -> tuple[str | None, str | None]:
    """devMode source hostPath and image of the deployed runtime, if any.

    The rendered CronWorkflows must match both: without the src mount the
    workflow pod runs the (possibly stale) baked image instead of the
    checked-out code, and with the chart-default image tag the pod points
    at an image the cluster never imported (ImagePullBackOff). Workflow
    pods reuse the runtime image by design, so the deployment is the
    source of truth for both settings.
    """
    out = _kubectl(
        "get", "deployment", "cogniverse-runtime", "-n", NAMESPACE, "-o", "json"
    )
    if out.returncode != 0:
        return None, None
    pod_spec = json.loads(out.stdout)["spec"]["template"]["spec"]
    hostpath = None
    for volume in pod_spec.get("volumes", []):
        host_path = volume.get("hostPath", {}).get("path", "")
        if volume.get("name") == "src-libs" and host_path.endswith("/libs"):
            hostpath = host_path.removesuffix("/libs")
    image = pod_spec["containers"][0].get("image")
    return hostpath, image


def _ensure_annotation_crons_applied() -> None:
    """Apply the chart's two annotation CronWorkflows to the cluster.

    The deployed release predates them; rendering from the LOCAL chart and
    applying is idempotent and keeps the e2e self-managing. devMode settings
    are mirrored from the deployed runtime so the workflow pods run the same
    code as the rest of the cluster.
    """
    helm_args = [
        "helm",
        "template",
        "cogniverse",
        "charts/cogniverse",
        "--namespace",
        NAMESPACE,
        "--set",
        f"runtime.qualityMonitor.tenantId={TENANT}",
    ]
    dev_hostpath, runtime_image = _deployed_runtime_settings()
    if dev_hostpath:
        helm_args += [
            "--set",
            "devMode.enabled=true",
            "--set",
            f"devMode.hostPath={dev_hostpath}",
        ]
    if runtime_image and ":" in runtime_image:
        repository, tag = runtime_image.rsplit(":", 1)
        helm_args += [
            "--set",
            "runtime.imagesByBackend=null",
            "--set",
            f"runtime.image.repository={repository}",
            "--set",
            f"runtime.image.tag={tag}",
        ]
    rendered = subprocess.run(
        helm_args,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert rendered.returncode == 0, rendered.stderr
    import yaml

    wanted = {"cogniverse-annotation-cycle", "cogniverse-annotation-feedback"}
    docs = [
        d
        for d in yaml.safe_load_all(rendered.stdout)
        if d
        and d.get("kind") == "CronWorkflow"
        and d.get("metadata", {}).get("name") in wanted
    ]
    assert len(docs) == 2, f"expected 2 annotation CronWorkflows, got {len(docs)}"
    applied = _kubectl(
        "apply", "-n", NAMESPACE, "-f", "-", input_text=yaml.dump_all(docs)
    )
    assert applied.returncode == 0, applied.stderr


PROFILE_NAME = "video_colpali_smol500_mv_frame"


def _provision_run_tenant() -> None:
    """Create this run's tenant and deploy its video profile.

    The serving probe routes a real query into search; without the tenant's
    video schema deployed, Vespa rejects the source ref and the gateway
    dispatch returns 500 before the calibration ever matters.
    """
    import httpx

    created = httpx.post(
        f"{RUNTIME_URL}/admin/tenants",
        json={"tenant_id": CANONICAL_TENANT, "created_by": "annotation-feedback-e2e"},
        timeout=120.0,
    )
    assert created.status_code in (200, 201, 409), (
        f"tenant creation failed: {created.status_code} {created.text[:300]}"
    )

    profile_name = PROFILE_NAME
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "config.json").read_text()
    )
    profile_def = config["backend"]["profiles"][profile_name]
    payload = {
        "profile_name": profile_name,
        "tenant_id": CANONICAL_TENANT,
        "type": profile_def.get("type", "video"),
        "description": profile_def.get("description", ""),
        "schema_name": profile_def.get("schema_name", profile_name),
        "embedding_model": profile_def.get("embedding_model", ""),
        "pipeline_config": profile_def.get("pipeline_config", {}),
        "strategies": profile_def.get("strategies", {}),
        "embedding_type": profile_def.get("embedding_type", "multi_vector"),
        "schema_config": profile_def.get("schema_config", {}),
        "model_specific": profile_def.get("model_specific"),
        "deploy_schema": True,
    }
    create = httpx.post(f"{RUNTIME_URL}/admin/profiles", json=payload, timeout=300.0)
    if create.status_code != 201:
        # The create route reports an existing profile as a 400 validation
        # error; anything else is a real failure.
        assert create.status_code in (400, 409) and "already exists" in create.text, (
            f"profile registration failed: {create.status_code} {create.text[:300]}"
        )
        deploy = httpx.post(
            f"{RUNTIME_URL}/admin/profiles/{profile_name}/deploy",
            json={"tenant_id": CANONICAL_TENANT},
            timeout=300.0,
        )
        assert deploy.status_code == 200, (
            f"schema deploy failed: {deploy.status_code} {deploy.text[:300]}"
        )


def _submit_workflow_from_cron(cron_name: str) -> str:
    out = _kubectl("get", "cronworkflow", cron_name, "-n", NAMESPACE, "-o", "json")
    assert out.returncode == 0, out.stderr
    spec = json.loads(out.stdout)["spec"]["workflowSpec"]
    workflow = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"{cron_name}-e2e-",
            "namespace": NAMESPACE,
            "labels": {"cogniverse.test/source": "e2e-annotation-feedback"},
        },
        "spec": spec,
    }
    created = _kubectl(
        "create",
        "-n",
        NAMESPACE,
        "-f",
        "-",
        "-o",
        "json",
        input_text=json.dumps(workflow),
    )
    assert created.returncode == 0, created.stderr
    return json.loads(created.stdout)["metadata"]["name"]


def _workflow_status(name: str) -> dict:
    out = _kubectl("get", "workflow", name, "-n", NAMESPACE, "-o", "json")
    if out.returncode != 0:
        return {}
    return json.loads(out.stdout).get("status", {}) or {}


def _wait_terminal(name: str, timeout_s: float = WORKFLOW_TIMEOUT_S) -> str:
    deadline = time.monotonic() + timeout_s
    phase = "Pending"
    while time.monotonic() < deadline:
        phase = _workflow_status(name).get("phase") or "Pending"
        if phase in {"Succeeded", "Failed", "Error"}:
            return phase
        time.sleep(POLL_INTERVAL_S)
    return phase


def _workflow_logs(name: str) -> str:
    out = _kubectl(
        "logs",
        "-n",
        NAMESPACE,
        "-l",
        f"workflows.argoproj.io/workflow={name}",
        "--all-containers",
        "--tail=500",
    )
    return out.stdout or out.stderr or ""


def _delete_workflow(name: str) -> None:
    _kubectl("delete", "workflow", name, "-n", NAMESPACE, "--wait=false")


def _spawned_feedback_workflows() -> set[str]:
    out = _kubectl(
        "get",
        "workflows",
        "-n",
        NAMESPACE,
        "-l",
        SPAWNED_LABEL,
        "-o",
        "jsonpath={.items[*].metadata.name}",
    )
    return set((out.stdout or "").split())


def _cluster_config_manager():
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(backend_url=VESPA_URL, backend_port=VESPA_PORT)
    return ConfigManager(store=store)


@pytest.fixture()
def cluster_telemetry():
    """Real TelemetryManager against the in-cluster Phoenix NodePorts."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    config = TelemetryConfig(
        otlp_endpoint=PHOENIX_GRPC,
        provider_config={
            "http_endpoint": PHOENIX_HTTP,
            "grpc_endpoint": PHOENIX_GRPC,
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


async def _seed_annotated_routing_spans(manager, count: int) -> None:
    from cogniverse_agents.routing.annotation_storage import AnnotationStorage
    from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel
    from cogniverse_foundation.telemetry.span_contract import record_span_io

    span_ids = []
    for i in range(count):
        with manager.span(
            name="cogniverse.routing", tenant_id=CANONICAL_TENANT
        ) as span:
            record_span_io(
                span,
                input_value=f"e2e feedback query {i}",
                output={"chosen_agent": "search_agent", "confidence": 0.4},
                operation="routing",
            )
            span_ids.append(format(span.get_span_context().span_id, "016x"))
    # The spawned gateway-thresholds compile reads cogniverse.gateway spans.
    for i in range(6):
        with manager.span(
            name="cogniverse.gateway", tenant_id=CANONICAL_TENANT
        ) as span:
            record_span_io(
                span,
                input_value=f"e2e gateway query {i}",
                output={
                    "complexity": "simple" if i % 2 else "complex",
                    "modality": "video",
                    "generation_type": "raw_results",
                    "routed_to": "search_agent",
                    "confidence": 0.55 + i * 0.05,
                },
                operation="gateway",
            )
    manager.force_flush(timeout_millis=10000)

    storage = AnnotationStorage(tenant_id=TENANT)
    # Wait for span indexing, then annotate each span as a human reviewer.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await storage.provider.traces.get_spans(
            project=storage.project_name,
            start_time=end - timedelta(minutes=30),
            end_time=end,
            filters={"name": "cogniverse.routing"},
            limit=10000,
        )
        if (
            spans is not None
            and not spans.empty
            and set(span_ids) <= set(spans.get("context.span_id", []))
        ):
            break
        await asyncio.sleep(3)

    for span_id in span_ids:
        assert await storage.store_human_annotation(
            span_id, AnnotationLabel.WRONG_ROUTING, "e2e seeded review"
        )

    # Wait until the feedback cycle's own read path sees them all.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        rows = await storage.query_annotated_spans(
            start_time=end - timedelta(hours=1),
            end_time=end,
            only_human_reviewed=True,
        )
        if len(rows) >= count:
            return
        await asyncio.sleep(3)
    raise AssertionError("seeded annotations never became queryable")


@pytest.mark.asyncio
async def test_feedback_workflow_runs_and_spawns_real_recompile(cluster_telemetry):
    from cogniverse_runtime.quality_monitor_cli import (
        _load_loop_state,
        _save_loop_state,
    )

    _ensure_annotation_crons_applied()
    _provision_run_tenant()

    # Reset the loop state so cooldown/poll gates from prior runs can't mask
    # this run (also what makes the test idempotent across reruns).
    config_manager = _cluster_config_manager()
    _save_loop_state(config_manager, CANONICAL_TENANT, {})

    # 12 human 'wrong_routing' reviews >= min_annotations_for_update (10).
    await _seed_annotated_routing_spans(cluster_telemetry, count=12)

    # Overwrite the thresholds artifact with a sentinel so the outcome
    # assertion below can only pass if the spawned compile actually wrote
    # a fresh calibration — "Succeeded" alone proves nothing about output.
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    artifacts = ArtifactManager(
        cluster_telemetry.get_provider(tenant_id=TENANT), TENANT
    )
    await artifacts.save_blob(
        kind="config",
        key="gateway_thresholds",
        content=json.dumps({"e2e_sentinel": True}),
    )

    before = _spawned_feedback_workflows()
    feedback_wf = _submit_workflow_from_cron(FEEDBACK_CRON)
    spawned_wf = None
    try:
        phase = _wait_terminal(feedback_wf)
        logs = _workflow_logs(feedback_wf)
        assert phase == "Succeeded", (
            f"feedback workflow {feedback_wf} ended {phase}; logs:\n{logs[-3000:]}"
        )
        # Fresh per-run tenant: the lookback window contains exactly this
        # run's 12 reviews, which sit in the thresholds_refresh tier
        # (>= min_annotations_for_update 10, < min_annotations_for_optimization 50).
        routing_action = re.search(
            r"'routing': \{'annotations': (\d+), 'action': '(\w+)'\}", logs
        )
        assert routing_action, logs[-3000:]
        assert int(routing_action.group(1)) == 12, routing_action.group(0)
        assert routing_action.group(2) == "thresholds_refresh", routing_action.group(0)

        # The CLI's own Argo submission created a REAL second workflow.
        new_spawned = _spawned_feedback_workflows() - before
        assert len(new_spawned) == 1, (
            f"expected exactly one spawned recompile workflow, got {new_spawned}; "
            f"logs:\n{logs[-3000:]}"
        )
        spawned_wf = new_spawned.pop()
        assert spawned_wf.startswith("annotation-feedback-routing-")

        # The spawned pod inherited the submitter's wiring end to end
        # (chart env -> CLI -> manifest -> Argo): same image as the deployed
        # runtime, plus the config mount it needs to reach the backends.
        dev_hostpath, deployed_image = _deployed_runtime_settings()
        spawned = json.loads(
            _kubectl(
                "get", "workflow", spawned_wf, "-n", NAMESPACE, "-o", "json"
            ).stdout
        )
        spawned_template = spawned["spec"]["templates"][0]
        assert spawned_template["container"]["image"] == deployed_image
        spawned_volumes = {v["name"] for v in spawned_template["volumes"]}
        assert "config" in spawned_volumes
        if dev_hostpath:
            assert {"src-libs", "src-scripts"} <= spawned_volumes

        # And that recompile (gateway-thresholds — no LM needed) runs to
        # completion against the in-cluster Vespa + Phoenix.
        spawned_phase = _wait_terminal(spawned_wf)
        assert spawned_phase == "Succeeded", (
            f"spawned workflow {spawned_wf} ended {spawned_phase}; logs:\n"
            f"{_workflow_logs(spawned_wf)[-3000:]}"
        )

        # What the optimization ACHIEVED: it replaced the sentinel with a
        # calibrated thresholds artifact (the exact blob GatewayAgent loads
        # at serve time). The calibration emits one of exactly three
        # threshold values, and gliner_threshold is a pure function of the
        # persisted p25 — both hold regardless of other cluster traffic in
        # the lookback window.
        blob = await artifacts.load_blob("config", "gateway_thresholds")
        assert blob is not None, "no gateway_thresholds artifact after compile"
        thresholds = json.loads(blob)
        assert "e2e_sentinel" not in thresholds, (
            "spawned compile succeeded but never wrote the thresholds artifact"
        )
        assert set(thresholds) == {
            "fast_path_confidence_threshold",
            "gliner_threshold",
            "analysis",
        }, thresholds
        analysis = thresholds["analysis"]
        assert set(analysis) == {
            "total_spans",
            "simple_count",
            "complex_count",
            "simple_error_rate",
            "complex_error_rate",
            "mean_confidence",
            "p25_confidence",
        }, analysis
        # Fresh per-run tenant: the window holds exactly the 6 seeded
        # gateway spans (3 simple / 3 complex, confidences 0.55..0.80, no
        # errors), so every persisted measurement and both recomputed knobs
        # are exact: mean 0.675, p25 0.6125; no-error/low-mean keeps the
        # fast path at the 0.4 default; gliner = round(0.6125 * 0.8, 3).
        assert analysis == {
            "total_spans": 6,
            "simple_count": 3,
            "complex_count": 3,
            "simple_error_rate": 0.0,
            "complex_error_rate": 0.0,
            "mean_confidence": 0.675,
            "p25_confidence": 0.6125,
        }, analysis
        assert thresholds["fast_path_confidence_threshold"] == 0.4, thresholds
        assert thresholds["gliner_threshold"] == 0.49, thresholds

        # The calibration is only achieved once the live gateway SERVES
        # with it. The dispatcher re-runs _load_artifact on every dispatch,
        # and every routing decision records the thresholds it ran under on
        # its cogniverse.routing span — so drive one real request and read
        # the served calibration back from the in-cluster Phoenix.
        import httpx

        from cogniverse_foundation.telemetry.span_contract import read_span_io

        probe_query = (
            f"calibration pickup probe {datetime.now(timezone.utc).timestamp():.0f}: "
            "find robot assembly videos"
        )
        drive = httpx.post(
            f"{RUNTIME_URL}/agents/gateway_agent/process",
            json={
                "agent_name": "gateway_agent",
                "query": probe_query,
                "context": {"tenant_id": TENANT},
            },
            timeout=600.0,
        )
        assert drive.status_code == 200, (
            f"gateway dispatch failed: {drive.status_code} {drive.text[:300]}"
        )
        provider = cluster_telemetry.get_provider(tenant_id=CANONICAL_TENANT)
        project = cluster_telemetry.config.get_project_name(CANONICAL_TENANT)
        probe_out = None
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and probe_out is None:
            end = datetime.now(timezone.utc)
            spans = await provider.traces.get_spans(
                project=project,
                start_time=end - timedelta(minutes=10),
                end_time=end,
                limit=1000,
            )
            if spans is not None and not spans.empty and "name" in spans.columns:
                for _, row in spans[spans["name"] == "cogniverse.routing"].iterrows():
                    io = read_span_io(row)
                    if io["input"] == probe_query:
                        probe_out = io["output"]
                        break
            if probe_out is None:
                await asyncio.sleep(2)
        assert probe_out is not None, "probe routing span never indexed in Phoenix"
        assert (
            probe_out["fast_path_confidence_threshold"]
            == thresholds["fast_path_confidence_threshold"]
        ), probe_out
        assert probe_out["gliner_threshold"] == thresholds["gliner_threshold"], (
            probe_out
        )

        # The cooldown state landed durably in the in-cluster Vespa config
        # store — a rerun within min_days_between_optimizations would skip.
        state = _load_loop_state(config_manager, CANONICAL_TENANT)
        assert "routing" in state.get("last_optimization_at", {}), state
    finally:
        _delete_workflow(feedback_wf)
        if spawned_wf:
            _delete_workflow(spawned_wf)
        # Drop this run's schema so repeated runs don't accumulate content
        # clusters in the shared Vespa (best-effort; the run is already
        # judged by the assertions above).
        import contextlib

        import httpx

        with contextlib.suppress(Exception):
            httpx.delete(
                f"{RUNTIME_URL}/admin/profiles/{PROFILE_NAME}",
                params={"tenant_id": CANONICAL_TENANT, "delete_schema": "true"},
                timeout=180.0,
            )
