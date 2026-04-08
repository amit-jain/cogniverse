"""
Unit tests for WorkflowIntelligence with DSPy integration and adaptive learning
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from cogniverse_agents.workflow.intelligence import (
    OptimizationStrategy,
    WorkflowIntelligence,
)
from cogniverse_agents.workflow_types import WorkflowPlan, WorkflowStatus, WorkflowTask


def _make_intelligence(**kwargs) -> WorkflowIntelligence:
    """Create a WorkflowIntelligence instance with a mock telemetry_provider."""
    defaults = dict(
        telemetry_provider=Mock(),
        tenant_id="test_tenant",
    )
    defaults.update(kwargs)
    return WorkflowIntelligence(**defaults)


@pytest.mark.unit
class TestWorkflowIntelligence:
    """Test cases for WorkflowIntelligence class"""

    def test_workflow_intelligence_initialization(self):
        """Test WorkflowIntelligence initializes with required telemetry_provider and tenant_id"""
        intelligence = _make_intelligence(
            max_history_size=1000,
            optimization_strategy=OptimizationStrategy.BALANCED,
        )

        assert intelligence._artifact_manager is not None
        assert intelligence.max_history_size == 1000
        assert intelligence.optimization_strategy == OptimizationStrategy.BALANCED

    def test_workflow_intelligence_requires_tenant_id(self):
        """Empty tenant_id must raise ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            WorkflowIntelligence(telemetry_provider=Mock(), tenant_id="")

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_record_workflow_execution(self):
        """Test recording workflow execution"""
        intelligence = _make_intelligence()

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="find AI videos",
            status=WorkflowStatus.COMPLETED,
            tasks=[
                WorkflowTask(
                    task_id="task1", agent_name="video_search", query="find AI videos"
                ),
                WorkflowTask(
                    task_id="task2", agent_name="summarizer", query="summarize results"
                ),
            ],
        )
        workflow_plan.start_time = datetime.now()
        workflow_plan.end_time = datetime.now()

        assert len(intelligence.workflow_history) == 0
        assert len(intelligence.agent_performance) == 0

        await intelligence.record_workflow_execution(workflow_plan)

        assert len(intelligence.workflow_history) == 1
        recorded_execution = intelligence.workflow_history[0]
        assert recorded_execution.workflow_id == "test-workflow"
        assert recorded_execution.query == "find AI videos"

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_optimization_workflow_methods(self):
        """Test workflow optimization functionality"""
        intelligence = _make_intelligence()

        assert hasattr(intelligence, "optimize_workflow_plan")
        assert callable(getattr(intelligence, "optimize_workflow_plan"))

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="test query",
            tasks=[
                WorkflowTask(task_id="task1", agent_name="agent1", query="test query")
            ],
        )

        optimized_plan = await intelligence.optimize_workflow_plan(
            "test query", workflow_plan
        )
        assert isinstance(optimized_plan, WorkflowPlan)

    @pytest.mark.asyncio
    async def test_get_agent_performance_metrics(self):
        """Test getting agent performance metrics"""
        intelligence = _make_intelligence()

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="find AI videos",
            status=WorkflowStatus.COMPLETED,
            tasks=[
                WorkflowTask(
                    task_id="task1", agent_name="video_search", query="find AI videos"
                )
            ],
        )
        workflow_plan.start_time = datetime.now()
        workflow_plan.end_time = datetime.now()

        await intelligence.record_workflow_execution(workflow_plan)

        report = intelligence.get_agent_performance_report()
        assert isinstance(report, dict)

        if "video_search" in intelligence.agent_performance:
            video_perf = intelligence.agent_performance["video_search"]
            assert video_perf.agent_name == "video_search"

    def test_query_type_classification(self):
        """Test query type classification functionality"""
        intelligence = _make_intelligence()

        assert hasattr(intelligence, "_classify_query_type")
        assert callable(getattr(intelligence, "_classify_query_type"))

        query_type = intelligence._classify_query_type(
            "find videos about machine learning"
        )
        assert isinstance(query_type, str)
        assert len(query_type) > 0
        assert query_type == "video_search"


@pytest.mark.unit
class TestWorkflowIntelligenceEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_max_history_size_limit(self):
        """Test that workflow history respects max size limit"""
        intelligence = _make_intelligence(max_history_size=3)

        for i in range(5):
            workflow_plan = WorkflowPlan(
                workflow_id=f"workflow-{i}",
                original_query=f"query {i}",
                status=WorkflowStatus.COMPLETED,
                tasks=[
                    WorkflowTask(
                        task_id=f"task-{i}", agent_name="agent1", query=f"query {i}"
                    )
                ],
            )
            workflow_plan.start_time = datetime.now()
            workflow_plan.end_time = datetime.now()

            await intelligence.record_workflow_execution(workflow_plan)

        assert len(intelligence.workflow_history) == 3
        assert intelligence.workflow_history[-1].workflow_id == "workflow-4"
        assert intelligence.workflow_history[0].workflow_id == "workflow-2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
