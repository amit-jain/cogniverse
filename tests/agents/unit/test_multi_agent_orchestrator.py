"""
Unit tests for MultiAgentOrchestrator with DSPy integration
"""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.app.agents.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    ResultAggregatorSignature,
    WorkflowPlannerSignature,
)
from src.app.agents.workflow_types import WorkflowStatus


@pytest.mark.unit
class TestWorkflowPlannerSignature:
    """Test DSPy signature for workflow planning"""

    @pytest.mark.ci_fast
    def test_workflow_planner_signature_structure(self):
        """Test WorkflowPlannerSignature has correct DSPy structure"""
        signature = WorkflowPlannerSignature
        
        # Check that signature exists and has proper docstring (used by DSPy)
        assert signature is not None
        assert hasattr(signature, '__doc__')
        assert "DSPy signature for workflow planning" in signature.__doc__
        
        # Test that we can reference the signature (validates DSPy structure)
        try:
            str(signature)
            assert True
        except Exception:
            pytest.fail("DSPy WorkflowPlannerSignature structure is invalid")


@pytest.mark.unit 
class TestResultAggregatorSignature:
    """Test DSPy signature for result aggregation"""

    @pytest.mark.ci_fast
    def test_result_aggregator_signature_structure(self):
        """Test ResultAggregatorSignature has correct DSPy structure"""
        signature = ResultAggregatorSignature
        
        # Check that signature exists and has proper docstring (used by DSPy)
        assert signature is not None
        assert hasattr(signature, '__doc__')
        assert "DSPy signature for aggregating multi-agent results" in signature.__doc__
        
        # Test that we can reference the signature (validates DSPy structure)
        try:
            str(signature)
            assert True
        except Exception:
            pytest.fail("DSPy ResultAggregatorSignature structure is invalid")


@pytest.mark.unit
class TestMultiAgentOrchestrator:
    """Test cases for MultiAgentOrchestrator class"""

    @pytest.fixture
    def mock_routing_agent(self):
        """Mock routing agent"""
        mock_agent = Mock()
        mock_agent.analyze_and_route = AsyncMock(return_value={
            "workflow_type": "multi_agent",
            "agents_to_call": ["video_search", "summarizer"],
            "confidence": 0.9
        })
        return mock_agent

    @pytest.fixture
    def sample_agents_config(self):
        """Sample agent configuration"""
        return {
            "video_search_agent": {
                "capabilities": ["video_content_search", "visual_query_analysis"],
                "url": "http://localhost:8002",
                "timeout": 30
            },
            "summarizer_agent": {
                "capabilities": ["text_summarization", "content_synthesis"],
                "url": "http://localhost:8003", 
                "timeout": 45
            }
        }

    @patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent")
    @patch("src.app.agents.multi_agent_orchestrator.A2AClient")
    @patch("src.app.agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.ci_fast
    def test_orchestrator_initialization_default(self, mock_workflow_intel, mock_a2a, mock_routing):
        """Test MultiAgentOrchestrator initialization with defaults"""
        mock_routing_instance = Mock()
        mock_routing.return_value = mock_routing_instance
        mock_a2a_instance = Mock()
        mock_a2a.return_value = mock_a2a_instance
        mock_workflow_intel_instance = Mock()
        mock_workflow_intel.return_value = mock_workflow_intel_instance

        orchestrator = MultiAgentOrchestrator()

        # Check initialization
        assert orchestrator.routing_agent == mock_routing_instance
        assert orchestrator.max_parallel_tasks == 3
        assert orchestrator.workflow_timeout == timedelta(minutes=15)
        assert orchestrator.enable_workflow_intelligence is True
        assert orchestrator.workflow_intelligence == mock_workflow_intel_instance
        assert orchestrator.a2a_client == mock_a2a_instance
        assert orchestrator.active_workflows == {}
        
        # Check statistics initialization
        stats = orchestrator.orchestration_stats
        assert stats["total_workflows"] == 0
        assert stats["completed_workflows"] == 0
        assert stats["failed_workflows"] == 0
        assert stats["average_execution_time"] == 0.0

    @patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent")
    @patch("src.app.agents.multi_agent_orchestrator.A2AClient")
    @pytest.mark.ci_fast
    def test_orchestrator_initialization_custom_config(self, mock_a2a, mock_routing, sample_agents_config):
        """Test MultiAgentOrchestrator with custom configuration"""
        mock_routing_instance = Mock()
        mock_routing.return_value = mock_routing_instance
        
        orchestrator = MultiAgentOrchestrator(
            available_agents=sample_agents_config,
            max_parallel_tasks=5,
            workflow_timeout_minutes=20,
            enable_workflow_intelligence=False
        )

        assert orchestrator.available_agents == sample_agents_config
        assert orchestrator.max_parallel_tasks == 5
        assert orchestrator.workflow_timeout == timedelta(minutes=20)
        assert orchestrator.enable_workflow_intelligence is False
        assert orchestrator.workflow_intelligence is None

    @patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent")
    @patch("src.app.agents.multi_agent_orchestrator.A2AClient")
    @patch("src.app.agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.ci_fast
    def test_get_default_agents(self, mock_workflow_intel, mock_a2a, mock_routing):
        """Test default agent configuration"""
        orchestrator = MultiAgentOrchestrator()
        
        default_agents = orchestrator._get_default_agents()
        
        assert "video_search_agent" in default_agents
        assert "summarizer_agent" in default_agents
        assert "detailed_report_agent" in default_agents
        
        # Check video search agent config
        video_agent = default_agents["video_search_agent"]
        assert "video_content_search" in video_agent["capabilities"]
        assert "endpoint" in video_agent
        assert "timeout_seconds" in video_agent

    @patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent")
    @patch("src.app.agents.multi_agent_orchestrator.A2AClient") 
    @patch("src.app.agents.multi_agent_orchestrator.create_workflow_intelligence")
    def test_initialize_dspy_modules(self, mock_workflow_intel, mock_a2a, mock_routing):
        """Test DSPy module initialization"""
        orchestrator = MultiAgentOrchestrator()
        
        # DSPy modules should be initialized
        assert hasattr(orchestrator, 'workflow_planner')
        assert hasattr(orchestrator, 'result_aggregator')

    @patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent")
    @patch("src.app.agents.multi_agent_orchestrator.A2AClient")
    @patch("src.app.agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.asyncio
    async def test_process_complex_query_basic(self, mock_workflow_intel, mock_a2a, mock_routing, mock_routing_agent):
        """Test basic complex query processing"""
        mock_routing.return_value = mock_routing_agent
        
        orchestrator = MultiAgentOrchestrator()
        
        # Mock workflow intelligence
        mock_workflow_intel_instance = Mock()
        mock_workflow_intel_instance.should_optimize_workflow = Mock(return_value=False)
        orchestrator.workflow_intelligence = mock_workflow_intel_instance
        
        # Mock workflow planning
        orchestrator.workflow_planner = Mock()
        orchestrator.workflow_planner.return_value = Mock(
            workflow_tasks=[{"agent": "video_search_agent", "task": "search videos"}],
            execution_strategy="sequential",
            expected_outcome="video search results",
            reasoning="Simple video search workflow"
        )
        
        query = "Find videos about machine learning"
        
        # Mock _execute_workflow to avoid complex execution
        orchestrator._execute_workflow = AsyncMock(return_value={
            "status": "completed",
            "results": {"video_search_agent": {"videos": [], "count": 0}},
            "execution_time": 1.5
        })
        
        result = await orchestrator.process_complex_query(query)
        
        assert result is not None
        assert "status" in result


@pytest.mark.unit
class TestMultiAgentOrchestratorWorkflowExecution:
    """Test workflow execution functionality"""

    @pytest.fixture
    def orchestrator_with_mocks(self):
        """Create orchestrator with mocked dependencies"""
        with patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent"), \
             patch("src.app.agents.multi_agent_orchestrator.A2AClient"), \
             patch("src.app.agents.multi_agent_orchestrator.create_workflow_intelligence"):
            
            orchestrator = MultiAgentOrchestrator(enable_workflow_intelligence=False)
            return orchestrator

    def test_create_fallback_workflow_plan_structure(self, orchestrator_with_mocks):
        """Test fallback workflow plan creation"""
        orchestrator = orchestrator_with_mocks
        
        # Test creating a workflow plan with the actual method signature
        workflow_plan = orchestrator._create_fallback_workflow_plan(
            workflow_id="test-workflow-123",
            query="test query",
            context="test context",
            user_id="test-user"
        )
        
        assert workflow_plan is not None
        assert workflow_plan.status == WorkflowStatus.PENDING
        assert len(workflow_plan.tasks) >= 1  # Should have at least one task
        assert workflow_plan.original_query == "test query"
        assert workflow_plan.workflow_id == "test-workflow-123"

    def test_workflow_statistics_update(self, orchestrator_with_mocks):
        """Test workflow statistics tracking"""
        orchestrator = orchestrator_with_mocks
        
        # Create a mock workflow plan
        from datetime import datetime

        from src.app.agents.workflow_types import WorkflowPlan
        
        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="test query",
            status=WorkflowStatus.COMPLETED,
            tasks=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Initial stats
        initial_completed = orchestrator.orchestration_stats["completed_workflows"]
        
        # Update stats
        orchestrator._update_orchestration_stats(workflow_plan, True)
        
        # Check updates
        assert orchestrator.orchestration_stats["completed_workflows"] == initial_completed + 1


@pytest.mark.unit
class TestMultiAgentOrchestratorEdgeCases:
    """Test edge cases and error handling"""

    @patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent")
    @patch("src.app.agents.multi_agent_orchestrator.A2AClient")
    def test_orchestrator_with_disabled_workflow_intelligence(self, mock_a2a, mock_routing):
        """Test orchestrator when workflow intelligence is disabled"""
        orchestrator = MultiAgentOrchestrator(enable_workflow_intelligence=False)
        
        assert orchestrator.enable_workflow_intelligence is False
        assert orchestrator.workflow_intelligence is None

    @patch("src.app.agents.multi_agent_orchestrator.EnhancedRoutingAgent")
    @patch("src.app.agents.multi_agent_orchestrator.A2AClient")
    @pytest.mark.ci_fast
    def test_orchestrator_agent_utilization_tracking(self, mock_a2a, mock_routing):
        """Test agent utilization statistics"""
        orchestrator = MultiAgentOrchestrator(enable_workflow_intelligence=False)
        
        # Test that agent utilization dict is initialized
        stats = orchestrator.orchestration_stats["agent_utilization"]
        assert isinstance(stats, dict)
        assert len(stats) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
