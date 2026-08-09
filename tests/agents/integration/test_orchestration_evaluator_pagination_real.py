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
from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
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


@pytest.mark.asyncio
async def test_real_cli_drains_and_persists_profiles_and_template(
    real_telemetry,
    workflow_state_redis_url,
):
    from cogniverse_core.registries import WorkflowStoreRegistry
    from cogniverse_runtime.optimization_cli import run_workflow_optimization

    tenant_id = canonical_tenant_id(f"orchlearn{uuid4().hex[:8]}")
    query = "find exact aurora video"
    workflow_ids = [f"wf-learn-{index:02d}" for index in range(55)]
    emitter = SimpleNamespace(
        telemetry_manager=real_telemetry,
        _current_tenant_id=tenant_id,
        _validated_agent_observations=OrchestratorAgent._validated_agent_observations,
    )
    for index, workflow_id in enumerate(workflow_ids):
        failed = index == 54
        await asyncio.to_thread(
            OrchestratorAgent._emit_orchestration_span,
            emitter,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            query=query,
            agent_sequence=["search_agent"],
            execution_time=1.25,
            success=not failed,
            tasks_completed=0 if failed else 1,
            pattern="sequential",
            execution_order=["search_agent"],
            error_summary="ReadTimeout: Vespa query exceeded 30s" if failed else None,
            agent_observations=[
                {
                    "agent_name": "search_agent",
                    "execution_time": 1.5 if failed else 0.75,
                    "success": not failed,
                    "confidence": 0.1 if failed else 0.9,
                }
            ],
        )

    provider = real_telemetry.get_provider(tenant_id=tenant_id)
    project = real_telemetry.config.get_project_name(tenant_id)
    await _wait_for_workflows(provider, project, set(workflow_ids))

    WorkflowStoreRegistry.clear_cache()
    WorkflowStoreRegistry.get(
        name="telemetry",
        config={
            "telemetry_provider": provider,
            "redis_url": workflow_state_redis_url,
        },
    )
    result = await run_workflow_optimization(tenant_id=tenant_id, lookback_hours=1)
    fresh_intelligence = WorkflowIntelligence(tenant_id=tenant_id)
    await fresh_intelligence.load_historical_data()

    assert result == {
        "status": "success",
        "spans_found": 55,
        "workflows_extracted": 55,
        "execution_demos_saved": 55,
        "agent_profiles_saved": 1,
        "workflow_templates_saved": 1,
    }
    assert len(fresh_intelligence.workflow_history) == 55
    failed_executions = [
        execution
        for execution in fresh_intelligence.workflow_history
        if not execution.success
    ]
    assert [execution.workflow_id for execution in failed_executions] == ["wf-learn-54"]
    assert failed_executions[0].error_details == (
        "ReadTimeout: Vespa query exceeded 30s"
    )

    profile = fresh_intelligence.agent_performance["search_agent"]
    assert (
        profile.total_executions,
        profile.successful_executions,
        profile.average_execution_time,
        profile.average_confidence,
        profile.error_rate,
        profile.preferred_query_types,
    ) == (
        55,
        54,
        pytest.approx(42 / 55),
        pytest.approx(48.7 / 55),
        1 / 55,
        ["sequential_query"],
    )

    template = fresh_intelligence._find_matching_template(query)
    assert template is not None
    assert template.query_patterns == [query]
    assert template.task_sequence == [
        {"agent": "search_agent", "task": "process", "dependencies": []}
    ]
    assert template.expected_execution_time == 1.25
    assert template.success_rate == 54 / 55
