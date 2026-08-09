"""Lossless evaluator batching against real Phoenix orchestration spans."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_agents.routing.orchestration_evaluator import OrchestrationEvaluator
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION

pytestmark = pytest.mark.integration


class _Recorder:
    def __init__(self) -> None:
        self.workflow_ids: list[str] = []

    async def record_execution(self, execution) -> None:
        self.workflow_ids.append(execution.workflow_id)


async def _wait_for_workflows(provider, project: str, expected: set[str]) -> None:
    from cogniverse_foundation.telemetry.span_contract import read_span_io

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        end_time = datetime.now(timezone.utc)
        spans = await provider.traces.get_all_spans(
            project=project,
            start_time=end_time - timedelta(hours=1),
            end_time=end_time,
            filters={"name": SPAN_NAME_ORCHESTRATION},
        )
        observed = {
            read_span_io(row)["output"].get("workflow_id")
            for _, row in spans.iterrows()
        }
        if expected <= observed:
            return
        await asyncio.sleep(0.25)
    pytest.fail(f"Phoenix did not expose exact workflows {sorted(expected)}")


@pytest.mark.asyncio
async def test_real_phoenix_batches_return_every_workflow_once(real_telemetry):
    tenant_id = canonical_tenant_id(f"orchpage{uuid4().hex[:8]}")
    workflow_ids = [f"wf-page-{index}" for index in range(5)]
    emitter = SimpleNamespace(
        telemetry_manager=real_telemetry,
        _current_tenant_id=tenant_id,
        _validated_agent_observations=OrchestratorAgent._validated_agent_observations,
    )
    for workflow_id in workflow_ids:
        await asyncio.to_thread(
            OrchestratorAgent._emit_orchestration_span,
            emitter,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            query=f"query for {workflow_id}",
            agent_sequence=["search_agent"],
            execution_time=1.0,
            success=True,
            tasks_completed=1,
            pattern="sequential",
            execution_order=["search_agent"],
        )
        await asyncio.sleep(0.002)

    provider = real_telemetry.get_provider(tenant_id=tenant_id)
    project = real_telemetry.config.get_project_name(tenant_id)
    await _wait_for_workflows(provider, project, set(workflow_ids))

    recorder = _Recorder()
    evaluator = OrchestrationEvaluator(recorder, tenant_id=tenant_id)
    results = [
        await evaluator.evaluate_orchestration_spans(batch_size=2) for _ in range(4)
    ]

    assert recorder.workflow_ids == workflow_ids
    assert len(recorder.workflow_ids) == len(set(recorder.workflow_ids)) == 5
    assert [result["workflows_extracted"] for result in results] == [2, 2, 1, 0]
    assert [result["spans_processed"] for result in results] == [2, 2, 1, 0]
