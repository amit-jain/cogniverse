"""Real-Phoenix round-trip for OrchestrationAnnotationStorage.

Stores orchestration-quality annotations via the public API, then reads them
back via query_annotated_spans — no mocks. Pins two contracts the read path
silently broke:

  * the default constructor queries the CANONICAL per-tenant project (the one
    the runtime emits orchestration spans to), not a literal "cogniverse"; and
  * only_human_reviewed splits on annotation_source in the annotation metadata,
    NOT Phoenix's annotator_kind (add_annotation stamps "HUMAN" for both a human
    review and an llm_auto one), so a real human annotation is returned rather
    than dropped to [].
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests

from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotation,
    OrchestrationAnnotationStorage,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_ORCHESTRATION,
    BatchExportConfig,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry

pytestmark = pytest.mark.integration


def _annotation(span_id: str, source: str, label: str, score: float):
    return OrchestrationAnnotation(
        workflow_id=f"wf-{span_id}",
        span_id=span_id,
        query="summarise the incident",
        orchestration_pattern="sequential",
        agents_used=["search_agent", "summarizer_agent"],
        execution_order=["search_agent", "summarizer_agent"],
        execution_time=1.2,
        pattern_is_optimal=True,
        agents_are_correct=True,
        execution_order_is_optimal=True,
        workflow_quality_label=label,
        quality_score=score,
        annotator_id="reviewer-1",
        annotation_source=source,
    )


async def _emit_orchestration_span(real_telemetry, tenant_id: str) -> str:
    with real_telemetry.span(
        name=SPAN_NAME_ORCHESTRATION,
        tenant_id=tenant_id,
    ) as span:
        return format(span.get_span_context().span_id, "016x")


async def _wait_for_span(storage, project, span_id) -> bool:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await storage.provider.traces.get_spans(
            project=project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=10000,
        )
        if (
            spans is not None
            and not spans.empty
            and "context.span_id" in spans.columns
            and (spans["context.span_id"] == span_id).any()
        ):
            return True
        await asyncio.sleep(2)
    return False


async def _query(storage, only_human_reviewed):
    end = datetime.now(timezone.utc)
    return await storage.query_annotated_spans(
        start_time=end - timedelta(hours=1),
        end_time=end,
        only_human_reviewed=only_human_reviewed,
    )


@pytest.mark.asyncio
async def test_human_annotation_round_trips_and_auto_is_filtered(real_telemetry):
    tenant_id = canonical_tenant_id(f"orchann{uuid4().hex[:8]}")
    project = real_telemetry.config.get_project_name(tenant_id)

    human_span = await _emit_orchestration_span(real_telemetry, tenant_id)
    auto_span = await _emit_orchestration_span(real_telemetry, tenant_id)
    real_telemetry.force_flush(timeout_millis=10000)

    # Default constructor — resolves the canonical per-tenant project, the same
    # one the spans above landed in. A literal "cogniverse" default would find
    # nothing here.
    storage = OrchestrationAnnotationStorage(tenant_id=tenant_id)
    assert storage.project_name == project
    assert await _wait_for_span(storage, project, human_span)
    assert await _wait_for_span(storage, project, auto_span)

    assert await storage.store_annotation(_annotation(human_span, "human", "good", 0.9))
    assert await storage.store_annotation(
        _annotation(auto_span, "llm_auto", "acceptable", 0.6)
    )

    # only_human_reviewed=True returns ONLY the human annotation, even though
    # Phoenix stamped annotator_kind="HUMAN" on both.
    human_only = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        human_only = await _query(storage, only_human_reviewed=True)
        if human_only:
            break
        await asyncio.sleep(2)
    assert [r["span_id"] for r in human_only] == [human_span]
    ann = human_only[0]["annotations"][0]
    assert ann["annotation_source"] == "human"
    assert ann["result"]["label"] == "good"
    assert ann["result"]["score"] == pytest.approx(0.9)

    # With the flag off, both annotated spans come back.
    everything = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        everything = await _query(storage, only_human_reviewed=False)
        if len(everything) >= 2:
            break
        await asyncio.sleep(2)
    assert {r["span_id"] for r in everything} == {human_span, auto_span}


def test_orchestration_span_failure_propagates_when_real_phoenix_is_down(
    phoenix_container,
):
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=False),
        )
    )
    assert manager.config.batch_config.use_sync_export is False
    tenant_id = canonical_tenant_id(f"orchdown{uuid4().hex[:8]}")
    emitter = SimpleNamespace(
        telemetry_manager=manager,
        _current_tenant_id=tenant_id,
    )
    container_name = phoenix_container["container_name"]
    subprocess.run(
        ["docker", "stop", container_name],
        check=True,
        capture_output=True,
        timeout=30,
    )
    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to persist orchestration telemetry: "
                f"tenant={tenant_id} workflow=wf-phoenix-down"
            ),
        ):
            OrchestratorAgent._emit_orchestration_span(
                emitter,
                tenant_id=tenant_id,
                workflow_id="wf-phoenix-down",
                query="find the incident recording",
                agent_sequence=["search_agent"],
                execution_time=1.25,
                success=True,
                tasks_completed=1,
                pattern="sequential",
                execution_order=["search_agent"],
            )
    finally:
        TelemetryManager.reset()
        get_telemetry_registry().clear_cache()
        subprocess.run(
            ["docker", "start", container_name],
            check=True,
            capture_output=True,
            timeout=30,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if (
                    requests.get(
                        f"{phoenix_container['http_endpoint']}/healthz",
                        timeout=2,
                    ).status_code
                    == 200
                ):
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            pytest.fail("Phoenix did not recover after the telemetry fault test")
