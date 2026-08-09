"""Lossless deterministic batching for orchestration span evaluation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cogniverse_agents.routing.orchestration_evaluator import OrchestrationEvaluator

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Recorder:
    def __init__(self) -> None:
        self.workflow_ids: list[str] = []

    async def record_execution(self, execution) -> None:
        self.workflow_ids.append(execution.workflow_id)


def _span(span_id: str, workflow_id: str, start_time: datetime) -> dict:
    return {
        "name": "cogniverse.orchestration",
        "context.span_id": span_id,
        "start_time": start_time,
        "attributes.input.value": f"query for {workflow_id}",
        "attributes.output.value": {
            "workflow_id": workflow_id,
            "pattern": "sequential",
            "agent_sequence": ["search_agent"],
            "execution_order": ["search_agent"],
            "execution_time": 1.0,
            "success": True,
            "tasks_completed": 1,
        },
        "status_message": None,
    }


def _evaluator(get_all_spans, recorder: _Recorder | None = None):
    evaluator = object.__new__(OrchestrationEvaluator)
    evaluator.project_name = "proj"
    evaluator.tenant_id = "acme:unit"
    evaluator.workflow_intelligence = recorder or _Recorder()
    evaluator._evaluation_cursor = None
    evaluator._evaluation_lock = asyncio.Lock()

    class _Traces:
        pass

    traces = _Traces()
    traces.get_all_spans = get_all_spans
    evaluator.provider = type("P", (), {"traces": traces})()
    return evaluator


@pytest.mark.asyncio
async def test_repeated_batches_process_exact_ids_once_in_timestamp_and_id_order():
    first_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    second_time = first_time + timedelta(seconds=1)
    spans = pd.DataFrame(
        [
            _span("span-c", "wf-c", first_time),
            _span("span-e", "wf-e", second_time),
            _span("span-a", "wf-a", first_time),
            _span("span-d", "wf-d", second_time),
            _span("span-b", "wf-b", first_time),
        ]
    )
    query_windows = []

    async def get_all_spans(project, start_time, end_time, filters):
        assert project == "proj"
        assert filters == {"name": "cogniverse.orchestration"}
        query_windows.append((start_time, end_time))
        return spans.copy(deep=True)

    recorder = _Recorder()
    evaluator = _evaluator(get_all_spans, recorder)

    evaluation_end = second_time + timedelta(minutes=1)
    results = [
        await evaluator.evaluate_orchestration_spans(
            batch_size=2,
            evaluation_end_time=evaluation_end,
        )
        for _ in range(4)
    ]

    assert recorder.workflow_ids == ["wf-a", "wf-b", "wf-c", "wf-d", "wf-e"]
    assert [result["workflows_extracted"] for result in results] == [2, 2, 1, 0]
    assert [result["spans_processed"] for result in results] == [2, 2, 1, 0]
    assert [window[0] for window in query_windows[1:]] == [
        first_time,
        second_time,
        second_time,
    ]
    assert {window[1] for window in query_windows} == {evaluation_end}
    assert evaluator._evaluation_cursor == (second_time, "span-e")


@pytest.mark.asyncio
async def test_record_failure_retries_failed_span_without_replaying_prior_success():
    first_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    spans = pd.DataFrame(
        [
            _span("span-a", "wf-a", first_time),
            _span("span-b", "wf-b", first_time + timedelta(seconds=1)),
            _span("span-c", "wf-c", first_time + timedelta(seconds=2)),
        ]
    )

    async def get_spans(**kwargs):
        return spans.copy(deep=True)

    class _FailOnceRecorder(_Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def record_execution(self, execution) -> None:
            if execution.workflow_id == "wf-b" and not self.failed:
                self.failed = True
                raise TimeoutError("workflow store timed out")
            await super().record_execution(execution)

    recorder = _FailOnceRecorder()
    evaluator = _evaluator(get_spans, recorder)

    with pytest.raises(
        RuntimeError,
        match="Failed to record orchestration span span-b",
    ) as exc_info:
        await evaluator.evaluate_orchestration_spans(batch_size=3)
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    second = await evaluator.evaluate_orchestration_spans(batch_size=3)

    assert second["workflows_extracted"] == 2
    assert second["errors"] == []
    assert recorder.workflow_ids == ["wf-a", "wf-b", "wf-c"]
    assert evaluator._evaluation_cursor == (
        first_time + timedelta(seconds=2),
        "span-c",
    )


@pytest.mark.asyncio
async def test_failed_query_does_not_advance_cursor():
    calls = 0

    async def get_spans(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("telemetry down")

    evaluator = _evaluator(get_spans)

    with pytest.raises(
        RuntimeError,
        match="Failed to query orchestration telemetry",
    ) as exc_info:
        await evaluator.evaluate_orchestration_spans(lookback_hours=2)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "telemetry down"
    assert calls == 3
    assert evaluator._evaluation_cursor is None


@pytest.mark.asyncio
async def test_query_timeout_retries_with_one_fixed_window(monkeypatch):
    import cogniverse_agents.routing.orchestration_evaluator as evaluator_module

    calls = []

    async def get_spans(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            await asyncio.sleep(1)
        return pd.DataFrame()

    monkeypatch.setattr(evaluator_module, "_SPAN_QUERY_TIMEOUT_S", 0.01)
    monkeypatch.setattr(evaluator_module, "_SPAN_QUERY_RETRY_DELAY_S", 0.0)
    evaluation_end = datetime(2026, 8, 5, 12, 30, tzinfo=timezone.utc)
    evaluator = _evaluator(get_spans)

    result = await evaluator.evaluate_orchestration_spans(
        lookback_hours=2,
        evaluation_end_time=evaluation_end,
    )

    assert result == {
        "spans_processed": 0,
        "workflows_extracted": 0,
        "errors": [],
        "evaluation_time": evaluation_end.isoformat(),
        "has_more": False,
    }
    assert len(calls) == 3
    assert {call["end_time"] for call in calls} == {evaluation_end}
    assert {call["start_time"] for call in calls} == {
        evaluation_end - timedelta(hours=2)
    }


@pytest.mark.asyncio
async def test_malformed_batch_rejects_without_recording_valid_prefix():
    first_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    valid = _span("span-a", "wf-a", first_time)
    malformed = _span("span-b", "wf-b", first_time + timedelta(seconds=1))
    malformed["attributes.output.value"]["tasks_completed"] = "1"
    spans = pd.DataFrame([valid, malformed])

    async def get_spans(**kwargs):
        return spans.copy(deep=True)

    recorder = _Recorder()
    evaluator = _evaluator(get_spans, recorder)

    with pytest.raises(
        ValueError,
        match="orchestration span tasks_completed must be a non-negative int",
    ):
        await evaluator.evaluate_orchestration_spans(batch_size=2)

    assert recorder.workflow_ids == []
    assert evaluator._evaluation_cursor is None


def test_task_count_is_planned_sequence_and_completed_count_remains_observed():
    now = datetime.now(timezone.utc)
    span = _span("span-failed", "wf-failed", now)
    span["attributes.output.value"].update(
        {
            "agent_sequence": ["router", "search", "summarizer"],
            "execution_order": ["router"],
            "success": False,
            "tasks_completed": 1,
            "error_summary": "TimeoutError: summarizer timed out after 30s",
        }
    )
    span["status_code"] = "ERROR"
    span["status_message"] = "TimeoutError: summarizer timed out after 30s"
    evaluator = _evaluator(lambda **kwargs: None)

    execution = evaluator._extract_workflow_execution(pd.Series(span))

    assert execution.task_count == 3
    assert execution.success is False
    assert execution.metadata["tasks_completed"] == 1
    assert execution.error_details == "TimeoutError: summarizer timed out after 30s"


def test_per_agent_observations_round_trip_without_workflow_metric_substitution():
    span = _span("span-agents", "wf-agents", datetime.now(timezone.utc))
    span["attributes.output.value"].update(
        {
            "agent_sequence": ["entity_agent", "search_agent"],
            "execution_order": ["entity_agent", "search_agent"],
            "execution_time": 4.5,
            "tasks_completed": 2,
            "agent_observations": [
                {
                    "agent_name": "entity_agent",
                    "execution_time": 0.25,
                    "success": True,
                    "confidence": 0.97,
                },
                {
                    "agent_name": "search_agent",
                    "execution_time": 3.75,
                    "success": True,
                    "confidence": 0.81,
                },
            ],
        }
    )
    evaluator = _evaluator(lambda **kwargs: None)

    execution = evaluator._extract_workflow_execution(pd.Series(span))

    assert execution.execution_time == 4.5
    assert execution.metadata["agent_observations"] == [
        {
            "agent_name": "entity_agent",
            "execution_time": 0.25,
            "success": True,
            "confidence": 0.97,
        },
        {
            "agent_name": "search_agent",
            "execution_time": 3.75,
            "success": True,
            "confidence": 0.81,
        },
    ]


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            {
                "agent_name": "unknown_agent",
                "execution_time": 0.2,
                "success": True,
            },
            "references an agent outside agent_sequence",
        ),
        (
            {
                "agent_name": "search_agent",
                "execution_time": -0.2,
                "success": True,
            },
            "execution_time must be a non-negative finite float",
        ),
        (
            {
                "agent_name": "search_agent",
                "execution_time": 0.2,
                "success": True,
                "confidence": "0.9",
            },
            "confidence must be a finite float between 0 and 1",
        ),
    ],
)
def test_malformed_agent_observation_is_rejected(observation, message):
    span = _span("span-agent-invalid", "wf-agent-invalid", datetime.now(timezone.utc))
    span["attributes.output.value"]["agent_observations"] = [observation]
    evaluator = _evaluator(lambda **kwargs: None)

    with pytest.raises(ValueError, match=message):
        evaluator._extract_workflow_execution(pd.Series(span))


@pytest.mark.parametrize(
    ("status_code", "error_summary", "message"),
    [
        ("OK", "TimeoutError: search timed out", "must have ERROR status"),
        ("ERROR", "", "error_summary must be a non-empty str"),
        ("ERROR", "x" * 513, "error_summary cannot exceed 512 characters"),
    ],
)
def test_failed_span_requires_error_status_and_bounded_summary(
    status_code,
    error_summary,
    message,
):
    span = _span("span-failed", "wf-failed", datetime.now(timezone.utc))
    span["attributes.output.value"].update(
        {
            "success": False,
            "tasks_completed": 0,
            "error_summary": error_summary,
        }
    )
    span["status_code"] = status_code
    evaluator = _evaluator(lambda **kwargs: None)

    with pytest.raises(ValueError, match=message):
        evaluator._extract_workflow_execution(pd.Series(span))


@pytest.mark.asyncio
async def test_real_workflow_intelligence_accepts_evaluator_outcome_metadata():
    from cogniverse_agents.workflow.intelligence import WorkflowIntelligence

    span = _span(
        "span-real-intelligence", "wf-real-intelligence", datetime.now(timezone.utc)
    )
    spans = pd.DataFrame([span])

    async def get_spans(**kwargs):
        return spans.copy(deep=True)

    intelligence = WorkflowIntelligence(tenant_id="acme:unit")
    evaluator = _evaluator(get_spans)
    evaluator.workflow_intelligence = intelligence

    result = await evaluator.evaluate_orchestration_spans()

    assert result["workflows_extracted"] == 1
    assert [execution.workflow_id for execution in intelligence.workflow_history] == [
        "wf-real-intelligence"
    ]


def test_ctor_canonicalizes_tenant_and_initializes_empty_cursor():
    manager = MagicMock()
    manager.get_provider.return_value = MagicMock()
    manager.config.get_project_name.side_effect = lambda tenant: f"cogniverse-{tenant}"

    with patch(
        "cogniverse_agents.routing.orchestration_evaluator.get_telemetry_manager",
        return_value=manager,
    ):
        evaluator = OrchestrationEvaluator(MagicMock(), tenant_id="acme")

    assert evaluator.tenant_id == "acme:acme"
    manager.get_provider.assert_called_once_with(tenant_id="acme:acme")
    assert evaluator.project_name == "cogniverse-acme:acme"
    assert evaluator._evaluation_cursor is None


@pytest.mark.asyncio
async def test_concurrent_runs_serialize_the_evaluation():
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    calls = []

    async def get_all_spans(project, start_time, end_time, filters):
        calls.append((start_time, end_time))
        if len(calls) == 1:
            first_entered.set()
            await release_first.wait()
        else:
            second_entered.set()
        return pd.DataFrame()

    ev = _evaluator(get_all_spans)
    first = asyncio.create_task(ev.evaluate_orchestration_spans(lookback_hours=1))
    await first_entered.wait()
    second = asyncio.create_task(ev.evaluate_orchestration_spans(lookback_hours=1))

    for _ in range(10):
        await asyncio.sleep(0)
    assert second_entered.is_set() is False

    release_first.set()
    await asyncio.gather(first, second)

    assert len(calls) == 2
