"""Synthetic schema timestamps must be timezone-aware (UTC), not naive."""

from __future__ import annotations

from cogniverse_synthetic.schemas import (
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)


def test_routing_schema_timestamp_is_tz_aware():
    s = RoutingExperienceSchema(
        query="q",
        enhanced_query="q",
        chosen_agent="search_agent",
        routing_confidence=0.5,
        search_quality=0.5,
        agent_success=True,
    )
    assert s.timestamp.tzinfo is not None


def test_workflow_schema_timestamp_is_tz_aware():
    s = WorkflowExecutionSchema(
        workflow_id="wf1",
        query="q",
        query_type="video",
        execution_time=1.0,
        success=True,
        agent_sequence=["search_agent"],
        task_count=1,
        parallel_efficiency=0.5,
        confidence_score=0.5,
    )
    assert s.timestamp.tzinfo is not None
