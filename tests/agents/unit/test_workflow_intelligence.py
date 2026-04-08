"""
Unit tests for WorkflowIntelligence — read-only template loader
"""

from unittest.mock import Mock

import pytest

from cogniverse_agents.workflow.intelligence import (
    OptimizationStrategy,
    WorkflowIntelligence,
    WorkflowTemplate,
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
    async def test_record_workflow_execution_appends_to_history(self):
        """record_workflow_execution converts a plan and appends to history."""
        intelligence = _make_intelligence()

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="find AI videos",
            status=WorkflowStatus.COMPLETED,
            tasks=[
                WorkflowTask(
                    task_id="task1", agent_name="video_search", query="find AI videos"
                ),
            ],
        )

        assert len(intelligence.workflow_history) == 0
        await intelligence.record_workflow_execution(workflow_plan)
        assert len(intelligence.workflow_history) == 1
        recorded = intelligence.workflow_history[0]
        assert recorded.workflow_id == "test-workflow"
        assert recorded.query == "find AI videos"
        assert recorded.success is True
        assert recorded.agent_sequence == ["video_search"]

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
        """Test getting agent performance report (read-only from loaded data)"""
        intelligence = _make_intelligence()

        report = intelligence.get_agent_performance_report()
        assert isinstance(report, dict)
        # No data loaded, so report is empty
        assert len(report) == 0

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
class TestSimplifiedWorkflowIntelligence:
    """Tests for the simplified read-only template loader."""

    def test_initialization_without_optimization(self):
        intelligence = _make_intelligence()
        assert intelligence is not None
        # No DSPy modules should be present
        assert not hasattr(intelligence, "workflow_optimizer")
        assert not hasattr(intelligence, "template_generator")

    @pytest.mark.asyncio
    async def test_load_templates(self):
        intelligence = _make_intelligence()
        templates = intelligence.get_workflow_templates()
        assert isinstance(templates, list)
        assert len(templates) == 0  # Nothing loaded

    @pytest.mark.asyncio
    async def test_find_matching_template(self):
        intelligence = _make_intelligence()
        template = WorkflowTemplate(
            template_id="t1",
            name="multi_modal_search",
            description="Multi-modal search workflow",
            query_patterns=["find videos and documents"],
            task_sequence=[
                {"agent": "search_agent"},
                {"agent": "document_agent"},
            ],
            expected_execution_time=3.0,
            success_rate=0.9,
        )
        intelligence.workflow_templates["t1"] = template

        match = intelligence._find_matching_template(
            "find videos and documents about AI"
        )
        assert match is not None
        assert match.template_id == "t1"

    def test_get_agent_performance_report(self):
        intelligence = _make_intelligence()
        report = intelligence.get_agent_performance_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_record_workflow_execution_appends(self):
        """record_workflow_execution converts plan and appends to history."""
        intelligence = _make_intelligence()
        plan = WorkflowPlan(
            workflow_id="test-wf",
            original_query="test",
            status=WorkflowStatus.COMPLETED,
            tasks=[],
        )
        await intelligence.record_workflow_execution(plan)
        assert len(intelligence.workflow_history) == 1
        assert intelligence.workflow_history[0].workflow_id == "test-wf"

    @pytest.mark.asyncio
    async def test_record_execution_appends_to_history(self):
        """record_execution appends a WorkflowExecution to history."""
        from cogniverse_agents.workflow.intelligence import WorkflowExecution

        intelligence = _make_intelligence()
        execution = WorkflowExecution(
            workflow_id="wf-1",
            query="test",
            query_type="general",
            execution_time=1.0,
            success=True,
            agent_sequence=["agent1"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.9,
        )
        await intelligence.record_execution(execution)
        assert len(intelligence.workflow_history) == 1
        assert intelligence.workflow_history[0].workflow_id == "wf-1"
        assert intelligence.workflow_history[0].success is True

    @pytest.mark.asyncio
    async def test_record_execution_respects_max_history_size(self):
        """record_execution should respect max_history_size limit."""
        from cogniverse_agents.workflow.intelligence import WorkflowExecution

        intelligence = _make_intelligence(max_history_size=2)
        for i in range(3):
            execution = WorkflowExecution(
                workflow_id=f"wf-{i}",
                query="test",
                query_type="general",
                execution_time=1.0,
                success=True,
                agent_sequence=["agent1"],
                task_count=1,
                parallel_efficiency=1.0,
                confidence_score=0.9,
            )
            await intelligence.record_execution(execution)
        assert len(intelligence.workflow_history) == 2
        assert intelligence.workflow_history[0].workflow_id == "wf-1"
        assert intelligence.workflow_history[1].workflow_id == "wf-2"

    @pytest.mark.asyncio
    async def test_optimize_from_ground_truth_is_noop(self):
        """optimize_from_ground_truth returns skip status."""
        intelligence = _make_intelligence()
        result = await intelligence.optimize_from_ground_truth()
        assert result["status"] == "skipped"
        assert result["reason"] == "use_argo_batch_jobs"

    @pytest.mark.asyncio
    async def test_optimize_plan_without_templates_uses_strategy(self):
        """Without templates, optimize_workflow_plan falls through to strategy."""
        intelligence = _make_intelligence()
        plan = WorkflowPlan(
            workflow_id="wf-1",
            original_query="test query",
            tasks=[
                WorkflowTask(task_id="t1", agent_name="agent1", query="test query")
            ],
        )
        result = await intelligence.optimize_workflow_plan("test query", plan)
        assert isinstance(result, WorkflowPlan)
        assert intelligence.optimization_stats["successful_optimizations"] == 1

    def test_get_workflow_templates_returns_values(self):
        """get_workflow_templates returns list of template objects."""
        intelligence = _make_intelligence()
        template = WorkflowTemplate(
            template_id="t1",
            name="test",
            description="test template",
            query_patterns=["test"],
            task_sequence=[],
            expected_execution_time=1.0,
            success_rate=0.9,
        )
        intelligence.workflow_templates["t1"] = template
        templates = intelligence.get_workflow_templates()
        assert len(templates) == 1
        assert templates[0].template_id == "t1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
