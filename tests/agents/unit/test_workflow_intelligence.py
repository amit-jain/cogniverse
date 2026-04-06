"""
Unit tests for WorkflowIntelligence with DSPy integration and adaptive learning
"""

from datetime import datetime

import pytest

from cogniverse_agents.workflow_intelligence import (
    OptimizationStrategy,
    WorkflowIntelligence,
)
from cogniverse_agents.workflow_types import WorkflowPlan, WorkflowStatus, WorkflowTask


@pytest.mark.unit
class TestWorkflowIntelligence:
    """Test cases for WorkflowIntelligence class"""

    def test_workflow_intelligence_initialization_with_persistence(
        self, workflow_store
    ):
        """Test WorkflowIntelligence initialization with persistence"""
        intelligence = WorkflowIntelligence(
            max_history_size=1000,
            enable_persistence=True,
            optimization_strategy=OptimizationStrategy.BALANCED,
            workflow_store=workflow_store,
        )

        assert intelligence.enable_persistence is True
        # With WorkflowStore backend, persistence is handled by the store
        assert intelligence.workflow_store is not None

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_record_workflow_execution(self):
        """Test recording workflow execution"""
        intelligence = WorkflowIntelligence(enable_persistence=False)

        # Create a workflow plan instead of execution
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

        # Initial state
        assert len(intelligence.workflow_history) == 0
        assert len(intelligence.agent_performance) == 0

        # Record execution
        await intelligence.record_workflow_execution(workflow_plan)

        # Check history
        assert len(intelligence.workflow_history) == 1
        recorded_execution = intelligence.workflow_history[0]
        assert recorded_execution.workflow_id == "test-workflow"
        assert recorded_execution.query == "find AI videos"

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_optimization_workflow_methods(self):
        """Test workflow optimization functionality"""
        intelligence = WorkflowIntelligence(enable_persistence=False)

        # Test that optimization method exists
        assert hasattr(intelligence, "optimize_workflow_plan")
        assert callable(getattr(intelligence, "optimize_workflow_plan"))

        # Create test workflow plan
        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="test query",
            tasks=[
                WorkflowTask(task_id="task1", agent_name="agent1", query="test query")
            ],
        )

        # Test optimization (will use fallback module)
        optimized_plan = await intelligence.optimize_workflow_plan(
            "test query", workflow_plan
        )
        assert isinstance(optimized_plan, WorkflowPlan)

    @pytest.mark.asyncio
    async def test_get_agent_performance_metrics(self):
        """Test getting agent performance metrics"""
        intelligence = WorkflowIntelligence(enable_persistence=False)

        # Create and record workflow execution
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

        # Test getting agent performance report (actual method name)
        report = intelligence.get_agent_performance_report()
        assert isinstance(report, dict)

        # Test accessing agent performance directly
        if "video_search" in intelligence.agent_performance:
            video_perf = intelligence.agent_performance["video_search"]
            assert video_perf.agent_name == "video_search"

    def test_query_type_classification(self):
        """Test query type classification functionality"""
        intelligence = WorkflowIntelligence(enable_persistence=False)

        # Test that classification method exists (it's private _classify_query_type)
        assert hasattr(intelligence, "_classify_query_type")
        assert callable(getattr(intelligence, "_classify_query_type"))

        # Test basic classification
        query_type = intelligence._classify_query_type(
            "find videos about machine learning"
        )
        assert isinstance(query_type, str)
        assert len(query_type) > 0
        assert query_type == "video_search"  # Should classify as video search


@pytest.mark.unit
class TestWorkflowIntelligenceEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_max_history_size_limit(self):
        """Test that workflow history respects max size limit"""
        intelligence = WorkflowIntelligence(
            max_history_size=3, enable_persistence=False
        )

        # Add more executions than max size
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

        # Should only keep the last 3
        assert len(intelligence.workflow_history) == 3
        assert intelligence.workflow_history[-1].workflow_id == "workflow-4"
        assert intelligence.workflow_history[0].workflow_id == "workflow-2"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
