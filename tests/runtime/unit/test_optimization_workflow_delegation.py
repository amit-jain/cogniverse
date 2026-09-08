"""Quality-triggered optimization spawns the chart's pod, not one of its own.

``POST /admin/tenant/{id}/optimize`` submits a Workflow whose whole spec is a
``workflowTemplateRef`` at the chart's optimization runner, so the pod it
spawns inherits the per-tenant mutex, the Vespa/Phoenix/inference/LLM env and
the cpu+memory requests and limits that template declares (pinned in
``tests/charts/test_optimization_pod_parity.py``).

The quality monitor and the annotation-feedback cron submit optimization
Workflows too. If either assembles a container spec of its own, whatever that
copy omits is silently absent from the pod: no mutex means two triggers for
one tenant stack pods instead of queueing, no resource block means BestEffort
QoS and first eviction under memory pressure, and a partial env list means the
compile cannot reach the services it scores against.

So the pin is delegation itself --- the submitted Workflow carries a reference
and arguments and nothing else --- plus the identity of the reference with the
one the manual route uses.
"""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_evaluation.quality_monitor import (
    OPTIMIZATION_WORKFLOW_PARAMETER_NAMES,
    AgentType,
    OptimizationTrigger,
    QualityMonitor,
    submit_argo_optimization_workflow,
)
from cogniverse_runtime.config_loader import (
    OPTIMIZATION_WORKFLOW_TEMPLATE_ENV,
    WorkflowSettings,
    get_workflow_settings,
)
from cogniverse_runtime.routers.tenant import _build_optimization_workflow_manifest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TEMPLATE = "cogniverse-optimization-runner"
NAMESPACE = "cogniverse"
TENANT = "acme:prod"

# Everything the pod spec needs and a submitted Workflow must NOT restate.
# Each key here is a whole pod-spec definition: an inline ``templates`` list
# carries the container (image, command, env, resources, volume mounts), an
# ``entrypoint`` names it, and ``volumes`` backs its mounts.
POD_SPEC_KEYS = frozenset({"templates", "entrypoint", "volumes", "podSpecPatch"})


@pytest.fixture(autouse=True)
def workflow_settings():
    get_workflow_settings._instance = WorkflowSettings(
        api_url="https://argo-server.argo.svc.cluster.local:2746",
        namespace=NAMESPACE,
        service_account="cogniverse",
        job_template="cogniverse-job-runner",
        optimization_template=TEMPLATE,
    )
    yield
    del get_workflow_settings._instance


class _ArgoCapture:
    """Records the manifests posted to the Argo API."""

    def __init__(self):
        self.manifests = []

    async def post(self, url, json=None, headers=None):
        self.manifests.append(json["workflow"])

        class _Response:
            status_code = 201
            text = ""

            @staticmethod
            def json():
                return {"metadata": {"name": "wf-1"}}

        return _Response()


def _trigger(tenant: str = TENANT) -> OptimizationTrigger:
    from datetime import datetime, timezone

    return OptimizationTrigger(
        timestamp=datetime(2026, 9, 8, 3, 0, tzinfo=timezone.utc),
        tenant_id=tenant,
        agents_to_optimize=[AgentType.SEARCH, AgentType.SUMMARY],
        golden_eval=None,
        live_eval=None,
        low_scoring_examples={},
        high_scoring_examples={},
        misrouted_queries=[],
    )


def _monitor(tenant: str = TENANT, template: str | None = TEMPLATE) -> QualityMonitor:
    return QualityMonitor(
        tenant_id=tenant,
        runtime_url="http://cogniverse-runtime:28000",
        phoenix_http_endpoint="http://cogniverse-phoenix:6006",
        llm_base_url="http://cogniverse-llm:11434/v1",
        llm_model="gemma3:4b",
        golden_dataset_path="/nonexistent.json",
        argo_api_url="https://argo-server.argo.svc.cluster.local:2746",
        argo_namespace=NAMESPACE,
        workflow_template=template,
    )


async def _submit_quality_trigger(tenant: str = TENANT) -> dict:
    monitor = _monitor(tenant)
    capture = _ArgoCapture()
    monitor._argo_client = capture
    submitted = await monitor.submit_optimization(
        _trigger(tenant), trigger_dataset="opt-trigger-ds"
    )
    assert submitted is True
    assert len(capture.manifests) == 1
    return capture.manifests[0]


@pytest.mark.asyncio
async def test_quality_trigger_carries_no_pod_spec_of_its_own():
    """An inline container spec is a second definition of the optimizer pod:
    it silently drops the per-tenant mutex, the resource requests/limits and
    every env entry the chart template declares."""
    spec = (await _submit_quality_trigger())["spec"]

    assert set(spec) == {"workflowTemplateRef", "arguments"}, (
        "the quality-triggered Workflow builds its own pod spec "
        f"({sorted(POD_SPEC_KEYS & set(spec))}), so the pod runs without the "
        "per-tenant optimize mutex, without cpu/memory requests and limits, "
        "and without the INFERENCE_SERVICE_URLS / LLM_ENDPOINT / LLM_ENGINE / "
        "LLM_MODEL env the shared WorkflowTemplate supplies"
    )


@pytest.mark.asyncio
async def test_quality_trigger_references_the_manual_routes_template():
    """One template, so a change to the pod reaches both paths at once."""
    manual = _build_optimization_workflow_manifest(
        TENANT, "gateway-thresholds", NAMESPACE
    )
    triggered = await _submit_quality_trigger()

    assert triggered["spec"]["workflowTemplateRef"] == {"name": TEMPLATE}
    assert (
        triggered["spec"]["workflowTemplateRef"]
        == manual["spec"]["workflowTemplateRef"]
    )
    assert manual["spec"]["workflowTemplateRef"]["name"] == (
        get_workflow_settings().optimization_template
    )


@pytest.mark.asyncio
async def test_quality_trigger_supplies_every_template_parameter():
    """The template's entrypoint binds all five inputs; an argument the
    submitter omits makes Argo reject the Workflow at admission."""
    manifest = await _submit_quality_trigger()

    assert manifest["spec"]["arguments"]["parameters"] == [
        {"name": "mode", "value": "triggered"},
        {"name": "tenant-id", "value": TENANT},
        {"name": "lookback-hours", "value": "4.0"},
        {"name": "agents", "value": "search,summary"},
        {"name": "trigger-dataset", "value": "opt-trigger-ds"},
    ]
    assert [p["name"] for p in manifest["spec"]["arguments"]["parameters"]] == list(
        OPTIMIZATION_WORKFLOW_PARAMETER_NAMES
    )


@pytest.mark.asyncio
async def test_quality_trigger_labels_are_k8s_safe_while_the_parameter_is_exact():
    manifest = await _submit_quality_trigger()

    assert manifest["metadata"]["labels"] == {
        "app": "cogniverse",
        "trigger": "quality-monitor",
        "tenant": "acme-prod",
    }
    assert manifest["metadata"]["namespace"] == NAMESPACE
    assert manifest["metadata"]["generateName"] == (
        "quality-triggered-optimization-20260908-030000-"
    )


@pytest.mark.asyncio
async def test_submit_refuses_when_no_template_is_configured():
    """Fault contract: with the chart's env var unset there is nothing to
    reference. Submitting anyway would spawn a pod with no mutex, no resources
    and no endpoints, and report success for it."""
    capture = _ArgoCapture()
    with pytest.raises(ValueError) as exc:
        await submit_argo_optimization_workflow(
            http_client=capture,
            argo_api_url="https://argo:2746",
            argo_namespace=NAMESPACE,
            tenant_id=TENANT,
            name_prefix="quality-triggered-optimization-20260908-030000",
            trigger_label="quality-monitor",
            workflow_template="",
            mode="triggered",
            lookback_hours="4.0",
            agents="search",
            trigger_dataset="opt-trigger-ds",
        )

    assert OPTIMIZATION_WORKFLOW_TEMPLATE_ENV in str(exc.value)
    assert capture.manifests == []


@pytest.mark.asyncio
async def test_concurrent_triggers_do_not_mix_tenants():
    """One shared Argo client, four tenants submitting at once: each Workflow
    keeps its own tenant in the parameter Argo interpolates into the mutex, so
    no tenant serialises behind another's lock."""
    tenants = ["acme:prod", "globex:prod", "initech:prod", "hooli:prod"]
    capture = _ArgoCapture()
    started = asyncio.Event()
    ready = asyncio.Semaphore(0)

    async def _submit(tenant: str):
        monitor = _monitor(tenant)
        monitor._argo_client = capture
        ready.release()
        await started.wait()
        return await monitor.submit_optimization(
            _trigger(tenant), trigger_dataset=f"ds-{tenant}"
        )

    tasks = [asyncio.create_task(_submit(t)) for t in tenants]
    for _ in tenants:
        await ready.acquire()
    started.set()
    assert await asyncio.gather(*tasks) == [True, True, True, True]

    submitted = {
        m["metadata"]["labels"]["tenant"]: {
            p["name"]: p["value"] for p in m["spec"]["arguments"]["parameters"]
        }
        for m in capture.manifests
    }
    assert set(submitted) == {"acme-prod", "globex-prod", "initech-prod", "hooli-prod"}
    for tenant in tenants:
        label = tenant.replace(":", "-")
        assert submitted[label]["tenant-id"] == tenant
        assert submitted[label]["trigger-dataset"] == f"ds-{tenant}"
