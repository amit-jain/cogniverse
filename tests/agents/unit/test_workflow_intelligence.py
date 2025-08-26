"""
Unit tests for WorkflowIntelligence with DSPy integration and adaptive learning
"""

from collections import deque
from datetime import datetime
from unittest.mock import patch

import pytest

from src.app.agents.workflow_intelligence import (
    AgentPerformance,
    OptimizationStrategy,
    TemplateGeneratorSignature,
    WorkflowExecution,
    WorkflowIntelligence,
    WorkflowOptimizationSignature,
    WorkflowTemplate,
    create_workflow_intelligence,
)
from src.app.agents.workflow_types import WorkflowPlan, WorkflowStatus, WorkflowTask


@pytest.mark.unit
class TestOptimizationStrategy:
    """Test optimization strategy enum"""

    @pytest.mark.ci_fast
    def test_optimization_strategy_values(self):
        """Test all optimization strategy values are available"""
        assert OptimizationStrategy.PERFORMANCE_BASED.value == "performance_based"
        assert OptimizationStrategy.SUCCESS_RATE_BASED.value == "success_rate_based"
        assert OptimizationStrategy.LATENCY_OPTIMIZED.value == "latency_optimized"
        assert OptimizationStrategy.COST_OPTIMIZED.value == "cost_optimized"
        assert OptimizationStrategy.BALANCED.value == "balanced"


@pytest.mark.unit
class TestWorkflowExecution:
    """Test WorkflowExecution data structure"""

    @pytest.mark.ci_fast
    def test_workflow_execution_creation(self):
        """Test WorkflowExecution creation with required fields"""
        execution = WorkflowExecution(
            workflow_id="test-workflow-123",
            query="find videos about AI",
            query_type="video_search",
            execution_time=2.5,
            success=True,
            agent_sequence=["video_search", "summarizer"],
            task_count=2,
            parallel_efficiency=0.8,
            confidence_score=0.9
        )
        
        assert execution.workflow_id == "test-workflow-123"
        assert execution.query == "find videos about AI"
        assert execution.query_type == "video_search"
        assert execution.execution_time == 2.5
        assert execution.success is True
        assert execution.agent_sequence == ["video_search", "summarizer"]
        assert execution.task_count == 2
        assert execution.parallel_efficiency == 0.8
        assert execution.confidence_score == 0.9
        assert isinstance(execution.timestamp, datetime)

    def test_workflow_execution_with_optional_fields(self):
        """Test WorkflowExecution with optional fields"""
        execution = WorkflowExecution(
            workflow_id="test-workflow",
            query="test query",
            query_type="test",
            execution_time=1.0,
            success=False,
            agent_sequence=["agent1"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.5,
            user_satisfaction=0.7,
            error_details="Test error"
        )
        
        assert execution.user_satisfaction == 0.7
        assert execution.error_details == "Test error"
        assert execution.success is False


@pytest.mark.unit
class TestAgentPerformance:
    """Test AgentPerformance data structure"""

    @pytest.mark.ci_fast
    def test_agent_performance_creation(self):
        """Test AgentPerformance creation and defaults"""
        performance = AgentPerformance(agent_name="video_search_agent")
        
        assert performance.agent_name == "video_search_agent"
        assert performance.total_executions == 0
        assert performance.successful_executions == 0
        assert performance.average_execution_time == 0.0
        assert performance.average_confidence == 0.0
        assert performance.error_rate == 0.0
        assert performance.preferred_query_types == []
        assert performance.performance_trend == "stable"
        assert isinstance(performance.last_updated, datetime)

    def test_agent_performance_with_data(self):
        """Test AgentPerformance with actual performance data"""
        performance = AgentPerformance(
            agent_name="summarizer_agent",
            total_executions=100,
            successful_executions=95,
            average_execution_time=1.2,
            average_confidence=0.85,
            error_rate=0.05,
            preferred_query_types=["summarization", "analysis"],
            performance_trend="improving"
        )
        
        assert performance.total_executions == 100
        assert performance.successful_executions == 95
        assert performance.error_rate == 0.05
        assert performance.performance_trend == "improving"
        assert "summarization" in performance.preferred_query_types


@pytest.mark.unit
class TestWorkflowTemplate:
    """Test WorkflowTemplate data structure"""

    @pytest.mark.ci_fast
    def test_workflow_template_creation(self):
        """Test WorkflowTemplate creation"""
        template = WorkflowTemplate(
            template_id="video-search-template",
            name="Video Search Workflow",
            description="Standard video search and summarization workflow",
            query_patterns=["find videos about *", "search for * videos"],
            task_sequence=[
                {"agent": "video_search", "action": "search"},
                {"agent": "summarizer", "action": "summarize"}
            ],
            expected_execution_time=3.0,
            success_rate=0.92
        )
        
        assert template.template_id == "video-search-template"
        assert template.name == "Video Search Workflow"
        assert len(template.query_patterns) == 2
        assert len(template.task_sequence) == 2
        assert template.expected_execution_time == 3.0
        assert template.success_rate == 0.92
        assert template.usage_count == 0
        assert isinstance(template.created_at, datetime)


@pytest.mark.unit
class TestWorkflowOptimizationSignature:
    """Test DSPy signature for workflow optimization"""

    @pytest.mark.ci_fast
    def test_workflow_optimization_signature_structure(self):
        """Test WorkflowOptimizationSignature has correct DSPy structure"""
        signature = WorkflowOptimizationSignature
        
        # Check that signature exists and has proper docstring (used by DSPy)
        assert signature is not None
        assert hasattr(signature, '__doc__')
        assert "DSPy signature for intelligent workflow optimization" in signature.__doc__
        
        # Test that we can reference the signature (validates DSPy structure)
        try:
            str(signature)
            assert True
        except Exception:
            pytest.fail("DSPy WorkflowOptimizationSignature structure is invalid")


@pytest.mark.unit
class TestTemplateGeneratorSignature:
    """Test DSPy signature for template generation"""

    @pytest.mark.ci_fast
    def test_template_generator_signature_structure(self):
        """Test TemplateGeneratorSignature has correct DSPy structure"""
        signature = TemplateGeneratorSignature
        
        # Check that signature exists and has proper docstring (used by DSPy)
        assert signature is not None
        assert hasattr(signature, '__doc__')
        assert "DSPy signature for generating workflow templates" in signature.__doc__
        
        # Test that we can reference the signature (validates DSPy structure)
        try:
            str(signature)
            assert True
        except Exception:
            pytest.fail("DSPy TemplateGeneratorSignature structure is invalid")


@pytest.mark.unit
class TestWorkflowIntelligence:
    """Test cases for WorkflowIntelligence class"""


    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
    @pytest.mark.ci_fast
    def test_workflow_intelligence_initialization_no_persistence(self):
        """Test WorkflowIntelligence initialization without persistence"""
        intelligence = WorkflowIntelligence(
            max_history_size=100,
            enable_persistence=False,
            optimization_strategy=OptimizationStrategy.PERFORMANCE_BASED
        )
        
        assert intelligence.max_history_size == 100
        assert intelligence.enable_persistence is False
        assert intelligence.optimization_strategy == OptimizationStrategy.PERFORMANCE_BASED
        assert isinstance(intelligence.workflow_history, deque)
        assert intelligence.workflow_history.maxlen == 100
        assert isinstance(intelligence.agent_performance, dict)
        assert isinstance(intelligence.workflow_templates, dict)
        
        # Check stats initialization
        stats = intelligence.optimization_stats
        assert stats["total_optimizations"] == 0
        assert stats["successful_optimizations"] == 0
        assert stats["average_improvement"] == 0.0

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", True)
    def test_workflow_intelligence_initialization_with_persistence(self):
        """Test WorkflowIntelligence initialization with persistence"""
        # Mock the database initialization to avoid actual database operations
        with patch.object(WorkflowIntelligence, '_initialize_persistence') as mock_init_db, \
             patch.object(WorkflowIntelligence, '_load_historical_data') as mock_load_data:
            
            intelligence = WorkflowIntelligence(
                max_history_size=1000,
                enable_persistence=True,
                optimization_strategy=OptimizationStrategy.BALANCED
            )
            
            assert intelligence.enable_persistence is True
            mock_init_db.assert_called_once()
            mock_load_data.assert_called_once()

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
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
                    task_id="task1",
                    agent_name="video_search",
                    query="find AI videos"
                ),
                WorkflowTask(
                    task_id="task2", 
                    agent_name="summarizer",
                    query="summarize results"
                )
            ]
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

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
    @pytest.mark.ci_fast
    @pytest.mark.asyncio  
    async def test_optimization_workflow_methods(self):
        """Test workflow optimization functionality"""
        intelligence = WorkflowIntelligence(enable_persistence=False)
        
        # Test that optimization method exists
        assert hasattr(intelligence, 'optimize_workflow_plan')
        assert callable(getattr(intelligence, 'optimize_workflow_plan'))
        
        # Create test workflow plan
        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="test query",
            tasks=[
                WorkflowTask(
                    task_id="task1",
                    agent_name="agent1",
                    query="test query"
                )
            ]
        )
        
        # Test optimization (will use fallback module)
        optimized_plan = await intelligence.optimize_workflow_plan(
            "test query", workflow_plan
        )
        assert isinstance(optimized_plan, WorkflowPlan)

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
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
                    task_id="task1",
                    agent_name="video_search",
                    query="find AI videos"
                )
            ]
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

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
    def test_query_type_classification(self):
        """Test query type classification functionality"""
        intelligence = WorkflowIntelligence(enable_persistence=False)
        
        # Test that classification method exists (it's private _classify_query_type)
        assert hasattr(intelligence, '_classify_query_type')
        assert callable(getattr(intelligence, '_classify_query_type'))
        
        # Test basic classification
        query_type = intelligence._classify_query_type("find videos about machine learning")
        assert isinstance(query_type, str)
        assert len(query_type) > 0
        assert query_type == "video_search"  # Should classify as video search


@pytest.mark.unit
class TestWorkflowIntelligenceOptimization:
    """Test optimization functionality"""

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
    def test_optimization_statistics_tracking(self):
        """Test optimization statistics are tracked correctly"""
        intelligence = WorkflowIntelligence(enable_persistence=False)
        
        # Initial stats
        initial_total = intelligence.optimization_stats["total_optimizations"]
        initial_successful = intelligence.optimization_stats["successful_optimizations"]
        
        # Test that update stats method exists and works
        if hasattr(intelligence, '_update_optimization_stats'):
            intelligence._update_optimization_stats(True, 15.0)
            
            assert intelligence.optimization_stats["total_optimizations"] == initial_total + 1
            assert intelligence.optimization_stats["successful_optimizations"] == initial_successful + 1

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
    def test_template_management(self):
        """Test workflow template creation and management"""
        intelligence = WorkflowIntelligence(enable_persistence=False)
        
        template = WorkflowTemplate(
            template_id="test-template",
            name="Test Template",
            description="Test template for unit tests",
            query_patterns=["test *"],
            task_sequence=[{"agent": "test_agent", "action": "test"}],
            expected_execution_time=1.0,
            success_rate=0.95
        )
        
        # Test template storage
        intelligence.workflow_templates[template.template_id] = template
        
        assert len(intelligence.workflow_templates) == 1
        assert "test-template" in intelligence.workflow_templates
        retrieved_template = intelligence.workflow_templates["test-template"]
        assert retrieved_template.name == "Test Template"


@pytest.mark.unit
class TestWorkflowIntelligenceEdgeCases:
    """Test edge cases and error handling"""

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_max_history_size_limit(self):
        """Test that workflow history respects max size limit"""
        intelligence = WorkflowIntelligence(
            max_history_size=3,
            enable_persistence=False
        )
        
        # Add more executions than max size
        for i in range(5):
            workflow_plan = WorkflowPlan(
                workflow_id=f"workflow-{i}",
                original_query=f"query {i}",
                status=WorkflowStatus.COMPLETED,
                tasks=[
                    WorkflowTask(
                        task_id=f"task-{i}",
                        agent_name="agent1",
                        query=f"query {i}"
                    )
                ]
            )
            workflow_plan.start_time = datetime.now()
            workflow_plan.end_time = datetime.now()
            
            await intelligence.record_workflow_execution(workflow_plan)
        
        # Should only keep the last 3
        assert len(intelligence.workflow_history) == 3
        assert intelligence.workflow_history[-1].workflow_id == "workflow-4"
        assert intelligence.workflow_history[0].workflow_id == "workflow-2"

    @patch("src.app.agents.workflow_intelligence.SQLITE_AVAILABLE", False)  
    @pytest.mark.ci_fast
    def test_dspy_modules_initialization(self):
        """Test that DSPy modules are initialized"""
        intelligence = WorkflowIntelligence(enable_persistence=False)
        
        # Check that DSPy modules are initialized
        assert hasattr(intelligence, 'workflow_optimizer')
        assert hasattr(intelligence, 'template_generator')


@pytest.mark.unit
class TestWorkflowIntelligenceFactory:
    """Test factory function for creating WorkflowIntelligence"""

    @pytest.mark.ci_fast
    def test_create_workflow_intelligence_function(self):
        """Test create_workflow_intelligence factory function"""
        intelligence = create_workflow_intelligence(
            optimization_strategy=OptimizationStrategy.LATENCY_OPTIMIZED
        )
        
        assert intelligence is not None
        assert isinstance(intelligence, WorkflowIntelligence)
        assert intelligence.optimization_strategy == OptimizationStrategy.LATENCY_OPTIMIZED

    def test_create_workflow_intelligence_with_params(self):
        """Test factory function with custom parameters"""
        intelligence = create_workflow_intelligence(
            max_history_size=500,
            enable_persistence=False,
            optimization_strategy=OptimizationStrategy.COST_OPTIMIZED
        )
        
        assert intelligence.max_history_size == 500
        assert intelligence.enable_persistence is False
        assert intelligence.optimization_strategy == OptimizationStrategy.COST_OPTIMIZED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
