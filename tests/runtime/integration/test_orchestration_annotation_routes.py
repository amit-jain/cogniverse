"""The orchestration annotation routes against real Phoenix.

Workflow spans are emitted by the orchestrator's own span writer. The runtime
reads and annotates them through a forwarding proxy in front of Phoenix's
HTTP API, so a test can fail or hold the annotation write itself; spans are
exported to Phoenix directly. Every review is read back from Phoenix.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotationStorage,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.routers import orchestration_annotations
from tests.utils.approval_review import run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

SEARCH_THEN_SUMMARIZE = {
    "workflow_id": "wf-lecture-report",
    "query": "summarize the lecture and write a report",
    "agent_sequence": ["search_agent", "summarizer_agent"],
    "execution_order": ["search_agent", "summarizer_agent"],
    "execution_time": 3.5,
    "success": True,
    "tasks_completed": 2,
    "pattern": "sequential",
}
FAILED_SEARCH = {
    "workflow_id": "wf-missing-clip",
    "query": "find the keynote intro clip",
    "agent_sequence": ["search_agent"],
    "execution_order": ["search_agent"],
    "execution_time": 1.25,
    "success": False,
    "tasks_completed": 0,
    "pattern": "parallel",
    "error_summary": "search_agent timed out",
}
REVIEW = {
    "annotator": "reviewer@example.com",
    "quality_label": "poor",
    "quality_score": 0.3,
    "pattern_is_optimal": False,
    "suggested_pattern": "parallel",
    "pattern_feedback": "the two steps are independent",
    "agents_are_correct": False,
    "missing_agents": ["report_agent"],
    "unnecessary_agents": ["summarizer_agent"],
    "execution_order_is_optimal": True,
    "improvement_notes": "use the report agent for reports",
}


@pytest.fixture(scope="module")
def phoenix_proxy(phoenix_container):
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def telemetry(phoenix_container, phoenix_proxy):
    """The global telemetry manager: spans export to Phoenix, reads and
    annotation writes go through ``phoenix_proxy``."""
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": f"http://127.0.0.1:{phoenix_proxy.port}",
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture()
def app(phoenix_proxy):
    app = FastAPI()
    app.include_router(orchestration_annotations.router, prefix="/admin/tenant")
    yield app
    phoenix_proxy.intercept = None


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://runtime",
        timeout=120,
    )


def _emit(telemetry, tenant_id, workflow):
    """Emit ``workflow`` through the orchestrator's span writer."""
    emitter = SimpleNamespace(telemetry_manager=telemetry)
    run_in_own_loop(
        OrchestratorAgent._emit_orchestration_span(
            emitter, tenant_id=tenant_id, **workflow
        )
    )
    telemetry.force_flush(timeout_millis=10000)


async def _workflows_until(client, tenant, predicate, timeout=60.0):
    """The listed workflows once ``predicate`` holds for them."""
    deadline = time.monotonic() + timeout
    while True:
        response = await client.get(
            f"/admin/tenant/{tenant}/orchestration-workflows?lookback_hours=1"
        )
        assert response.status_code == 200, response.text
        workflows = response.json()["workflows"]
        if predicate(workflows) or time.monotonic() > deadline:
            return workflows
        await asyncio.sleep(2)


def _listed(workflow):
    return {
        "query": workflow["query"],
        "workflow_id": workflow["workflow_id"],
        "pattern": workflow["pattern"],
        "agent_sequence": workflow["agent_sequence"],
        "execution_order": workflow["execution_order"],
        "execution_time": workflow["execution_time"],
        "tasks_completed": workflow["tasks_completed"],
        "success": workflow["success"],
        "error_summary": workflow.get("error_summary"),
    }


def _tenant():
    return canonical_tenant_id(f"orchrev{uuid4().hex[:8]}")


async def test_workflows_list_newest_first_with_what_the_orchestrator_recorded(
    telemetry, app
):
    tenant = _tenant()
    _emit(telemetry, tenant, SEARCH_THEN_SUMMARIZE)
    _emit(telemetry, tenant, FAILED_SEARCH)
    async with _client(app) as client:
        workflows = await _workflows_until(client, tenant, lambda w: len(w) == 2)
    assert [
        {
            key: value
            for key, value in workflow.items()
            if key not in {"span_id", "start_time"}
        }
        for workflow in workflows
    ] == [
        {**_listed(FAILED_SEARCH), "review": None},
        {**_listed(SEARCH_THEN_SUMMARIZE), "review": None},
    ]
    starts = [datetime.fromisoformat(w["start_time"]) for w in workflows]
    assert starts == sorted(starts, reverse=True)


async def test_a_review_is_stored_on_the_span_and_listed_with_it(telemetry, app):
    tenant = _tenant()
    _emit(telemetry, tenant, SEARCH_THEN_SUMMARIZE)
    async with _client(app) as client:
        (workflow,) = await _workflows_until(client, tenant, lambda w: len(w) == 1)
        response = await client.post(
            f"/admin/tenant/{tenant}/orchestration-workflows/"
            f"{workflow['span_id']}/annotation",
            json={**REVIEW, "start_time": workflow["start_time"]},
        )
        review = {
            "annotator": "reviewer@example.com",
            "label": "poor",
            "score": 0.3,
            "annotation_source": "human",
            "pattern_is_optimal": False,
            "agents_are_correct": False,
            "execution_order_is_optimal": True,
            "improvement_notes": "use the report agent for reports",
        }
        assert (response.status_code, response.json()) == (
            200,
            {**workflow, "review": review},
        )
        (listed,) = await _workflows_until(
            client, tenant, lambda w: w[0]["review"] is not None
        )
    assert listed == {**workflow, "review": review}

    storage = OrchestrationAnnotationStorage(tenant_id=tenant)
    end = datetime.now(timezone.utc)
    (stored,) = await storage.query_annotated_spans(
        start_time=end - timedelta(hours=1), end_time=end
    )
    metadata = stored["annotations"][0]["metadata"]
    assert (
        stored["span_id"],
        stored["annotations"][0]["result"],
        {
            key: metadata[key]
            for key in (
                "workflow_id",
                "query",
                "actual_pattern",
                "actual_agents",
                "suggested_pattern",
                "missing_agents",
                "unnecessary_agents",
                "suggested_agents",
                "workflow_succeeded",
                "annotator_id",
                "annotation_source",
            )
        },
    ) == (
        workflow["span_id"],
        {"label": "poor", "score": 0.3},
        {
            "workflow_id": "wf-lecture-report",
            "query": "summarize the lecture and write a report",
            "actual_pattern": "sequential",
            "actual_agents": "search_agent,summarizer_agent",
            "suggested_pattern": "parallel",
            "missing_agents": "report_agent",
            "unnecessary_agents": "summarizer_agent",
            "suggested_agents": "search_agent,report_agent",
            "workflow_succeeded": True,
            "annotator_id": "reviewer@example.com",
            "annotation_source": "human",
        },
    )


async def test_a_review_the_span_cannot_take_is_refused_and_stores_nothing(
    telemetry, app, phoenix_proxy
):
    tenant = _tenant()
    _emit(telemetry, tenant, SEARCH_THEN_SUMMARIZE)
    async with _client(app) as client:
        (workflow,) = await _workflows_until(client, tenant, lambda w: len(w) == 1)
        url = f"/admin/tenant/{tenant}/orchestration-workflows"
        phoenix_proxy.requests.clear()
        unknown = await client.post(
            f"{url}/0000000000000000/annotation",
            json={**REVIEW, "start_time": workflow["start_time"]},
        )
        naive = await client.post(
            f"{url}/{workflow['span_id']}/annotation",
            json={**REVIEW, "start_time": "2026-10-04T12:00:00"},
        )
        out_of_range = await client.post(
            f"{url}/{workflow['span_id']}/annotation",
            json={**REVIEW, "quality_score": 1.5, "start_time": workflow["start_time"]},
        )
        bad_label = await client.post(
            f"{url}/{workflow['span_id']}/annotation",
            json={
                **REVIEW,
                "quality_label": "great",
                "start_time": workflow["start_time"],
            },
        )
        (after,) = await _workflows_until(client, tenant, lambda w: len(w) == 1)
    assert (unknown.status_code, unknown.json()) == (
        404,
        {
            "detail": "No orchestration workflow span 0000000000000000 started at "
            f"{workflow['start_time']} for tenant {tenant}."
        },
    )
    assert (naive.status_code, naive.json()) == (
        422,
        {"detail": "start_time must include a timezone."},
    )
    assert [
        (response.status_code, [error["loc"] for error in response.json()["detail"]])
        for response in (out_of_range, bad_label)
    ] == [(422, [["body", "quality_score"]]), (422, [["body", "quality_label"]])]
    assert after["review"] is None
    assert [
        path for method, path, _ in phoenix_proxy.requests if method == "POST"
    ] == []


async def test_two_reviews_stored_at_once_each_land_on_their_own_workflow(
    telemetry, app, phoenix_proxy
):
    tenant = _tenant()
    _emit(telemetry, tenant, SEARCH_THEN_SUMMARIZE)
    _emit(telemetry, tenant, FAILED_SEARCH)
    barrier = threading.Barrier(2, timeout=30)
    held = []

    def hold_annotation_writes(method, path, body):
        # Both annotation writes reach Phoenix together.
        if method == "POST" and "annotations" in path:
            held.append(json.loads(body))
            barrier.wait()
        return None

    async with _client(app) as client:
        failed, succeeded = await _workflows_until(
            client, tenant, lambda w: len(w) == 2
        )
        phoenix_proxy.intercept = hold_annotation_writes
        labels = {failed["span_id"]: "failed", succeeded["span_id"]: "excellent"}
        responses = await asyncio.gather(
            *(
                client.post(
                    f"/admin/tenant/{tenant}/orchestration-workflows/"
                    f"{workflow['span_id']}/annotation",
                    json={
                        **REVIEW,
                        "quality_label": labels[workflow["span_id"]],
                        "start_time": workflow["start_time"],
                    },
                )
                for workflow in (failed, succeeded)
            )
        )
        phoenix_proxy.intercept = None
        listed = await _workflows_until(
            client, tenant, lambda w: all(item["review"] for item in w)
        )
    assert [response.status_code for response in responses] == [200, 200]
    assert len(held) == 2
    assert {
        workflow["span_id"]: (workflow["workflow_id"], workflow["review"]["label"])
        for workflow in listed
    } == {
        failed["span_id"]: ("wf-missing-clip", "failed"),
        succeeded["span_id"]: ("wf-lecture-report", "excellent"),
    }


async def test_a_failed_annotation_write_answers_502_and_stores_nothing(
    telemetry, app, phoenix_proxy
):
    tenant = _tenant()
    _emit(telemetry, tenant, SEARCH_THEN_SUMMARIZE)
    async with _client(app) as client:
        (workflow,) = await _workflows_until(client, tenant, lambda w: len(w) == 1)
        phoenix_proxy.intercept = lambda method, path, body: (
            (503, {"detail": "unavailable"})
            if method == "POST" and "annotations" in path
            else None
        )
        response = await client.post(
            f"/admin/tenant/{tenant}/orchestration-workflows/"
            f"{workflow['span_id']}/annotation",
            json={**REVIEW, "start_time": workflow["start_time"]},
        )
        phoenix_proxy.intercept = None
        (after,) = await _workflows_until(client, tenant, lambda w: len(w) == 1)
    detail = response.json()["detail"]
    assert (
        response.status_code,
        {key: detail[key] for key in ("error", "message", "tenant_id")},
    ) == (
        502,
        {
            "error": "annotation_not_stored",
            "message": f"The review of workflow span {workflow['span_id']} was not stored.",
            "tenant_id": tenant,
        },
    )
    assert after["review"] is None


async def test_an_unreachable_telemetry_backend_lists_as_an_outage(
    telemetry, app, phoenix_proxy
):
    tenant = _tenant()
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
    async with _client(app) as client:
        response = await client.get(
            f"/admin/tenant/{tenant}/orchestration-workflows?lookback_hours=1"
        )
    detail = response.json()["detail"]
    assert (
        response.status_code,
        {key: detail[key] for key in ("error", "message", "tenant_id")},
    ) == (
        502,
        {
            "error": "telemetry_unavailable",
            "message": f"Could not read the orchestration workflows of tenant {tenant}.",
            "tenant_id": tenant,
        },
    )
