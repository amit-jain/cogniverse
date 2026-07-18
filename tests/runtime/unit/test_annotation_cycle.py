"""The scheduled annotation-identification cycle feeds the review worklist.

``run_annotation_cycle`` walks each agent type, identifies spans needing
human review, drops spans that already carry an annotation, caps the batch,
and POSTs the remainder to the runtime's queue-enqueue endpoint (exercised
here through the real mounted router, not a stub).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
)
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_runtime.quality_monitor_cli import run_annotation_cycle
from cogniverse_runtime.routers import agents as agents_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _request(span_id, agent_type="routing"):
    return AnnotationRequest(
        span_id=span_id,
        timestamp=datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc),
        query="find robot videos",
        chosen_agent="video_search",
        routing_confidence=0.3,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=AnnotationPriority.HIGH,
        reason="low confidence",
        context={},
        agent_type=agent_type,
    )


class _StubAnnotationAgent:
    """Identifies a fixed set of spans per agent type."""

    by_type = {
        "routing": ["r1", "r2"],
        "query_enhancement": ["q1"],
    }

    def __init__(self, tenant_id, automation_rules=None, **kwargs):
        self.tenant_id = tenant_id

    async def identify_spans_needing_annotation(
        self, lookback_hours=None, agent_type="routing"
    ):
        return [_request(sid, agent_type) for sid in self.by_type.get(agent_type, [])]


class _StubStorage:
    """Reports span r2 as already annotated (must not be re-enqueued)."""

    def __init__(self, tenant_id, agent_type="routing"):
        self.agent_type = agent_type

    async def fetch_project_spans(self, start_time, end_time):
        return None

    async def query_annotated_spans(
        self, start_time, end_time, only_human_reviewed, spans_df=None
    ):
        if self.agent_type == "routing":
            return [{"span_id": "r2"}]
        return []


@pytest.mark.asyncio
async def test_cycle_enqueues_unannotated_spans_with_tenant():
    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    fresh_queue_patch = patch.object(
        agents_router,
        "_annotation_queue",
        __import__(
            "cogniverse_agents.routing.annotation_queue",
            fromlist=["AnnotationQueue"],
        ).AnnotationQueue(),
    )

    with (
        fresh_queue_patch,
        patch(
            "cogniverse_agents.routing.annotation_agent.AnnotationAgent",
            _StubAnnotationAgent,
        ),
        patch(
            "cogniverse_agents.routing.annotation_storage.AnnotationStorage",
            _StubStorage,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            result = await run_annotation_cycle(
                tenant_id="acme:acme",
                runtime_url="http://runtime",
                agent_types=["routing", "query_enhancement"],
                http_client=client,
            )

        queue = agents_router.get_annotation_queue()
        # r2 was already annotated → only r1 + q1 enqueued.
        assert result["identified"] == 3
        assert result["already_annotated"] == 1
        assert result["enqueued"] == 2
        assert queue.get("r1") is not None
        assert queue.get("r2") is None
        assert queue.get("q1") is not None
        # Requests carry the tenant so completion can persist durably.
        assert queue.get("r1").tenant_id == "acme:acme"
        assert queue.get("q1").agent_type == "query_enhancement"


@pytest.mark.asyncio
async def test_cycle_caps_total_at_max_annotations_per_cycle():
    from cogniverse_agents.routing.config import (
        AutomationRulesConfig,
        OptimizationTriggersConfig,
    )

    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    rules = AutomationRulesConfig(
        optimization_triggers=OptimizationTriggersConfig(max_annotations_per_cycle=1)
    )

    with (
        patch.object(
            agents_router,
            "_annotation_queue",
            __import__(
                "cogniverse_agents.routing.annotation_queue",
                fromlist=["AnnotationQueue"],
            ).AnnotationQueue(),
        ),
        patch(
            "cogniverse_agents.routing.annotation_agent.AnnotationAgent",
            _StubAnnotationAgent,
        ),
        patch(
            "cogniverse_agents.routing.annotation_storage.AnnotationStorage",
            _StubStorage,
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            result = await run_annotation_cycle(
                tenant_id="acme:acme",
                runtime_url="http://runtime",
                agent_types=["routing", "query_enhancement"],
                http_client=client,
                automation_rules=rules,
            )

        assert result["enqueued"] == 1
        assert agents_router.get_annotation_queue().statistics()["total"] == 1


class _CountingTraceStore:
    """Counts whole-project span pulls and serves one span row."""

    def __init__(self):
        self.get_spans_calls = 0

    async def get_spans(
        self, project, start_time=None, end_time=None, filters=None, limit=1000
    ):
        import pandas as pd

        self.get_spans_calls += 1
        return pd.DataFrame([{"context.span_id": "r1", "name": "cogniverse.routing"}])


class _EmptyAnnotationStore:
    async def get_annotations(self, spans_df=None, project=None, annotation_names=None):
        import pandas as pd

        return pd.DataFrame()


class _StubTelemetryManager:
    """Hands the real AnnotationStorage a counting provider."""

    def __init__(self, provider):
        from types import SimpleNamespace

        self._provider = provider
        self.config = SimpleNamespace(get_project_name=lambda tid: f"cogniverse-{tid}")

    def get_provider(self, tenant_id):
        return self._provider


@pytest.mark.asyncio
async def test_cycle_pulls_project_spans_once_for_all_agent_types():
    """One whole-project span pull serves every agent type in a cycle.

    The per-agent query_annotated_spans calls share an identical time window,
    so the real AnnotationStorage must receive the pre-fetched frame instead
    of re-pulling the project once per agent type."""
    from types import SimpleNamespace

    _StubAnnotationAgent.by_type = {
        "routing": ["r1"],
        "query_enhancement": ["q1"],
        "search": ["s1"],
    }
    trace_store = _CountingTraceStore()
    provider = SimpleNamespace(traces=trace_store, annotations=_EmptyAnnotationStore())

    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    try:
        with (
            patch.object(
                agents_router,
                "_annotation_queue",
                __import__(
                    "cogniverse_agents.routing.annotation_queue",
                    fromlist=["AnnotationQueue"],
                ).AnnotationQueue(),
            ),
            patch(
                "cogniverse_agents.routing.annotation_agent.AnnotationAgent",
                _StubAnnotationAgent,
            ),
            patch(
                "cogniverse_agents.routing.annotation_storage.get_telemetry_manager",
                return_value=_StubTelemetryManager(provider),
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://runtime"
            ) as client:
                result = await run_annotation_cycle(
                    tenant_id="acme:acme",
                    runtime_url="http://runtime",
                    agent_types=["routing", "query_enhancement", "search"],
                    http_client=client,
                )
    finally:
        _StubAnnotationAgent.by_type = {
            "routing": ["r1", "r2"],
            "query_enhancement": ["q1"],
        }

    assert result["identified"] == 3
    assert result["already_annotated"] == 0
    assert result["enqueued"] == 3
    assert trace_store.get_spans_calls == 1
