"""
Unit tests for MultiAgentOrchestrator with DSPy integration
"""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from cogniverse_agents.multi_agent_orchestrator import (
    FusionStrategy,
    MultiAgentOrchestrator,
    ResultAggregatorSignature,
    WorkflowPlannerSignature,
)
from cogniverse_agents.workflow_types import TaskStatus, WorkflowStatus, WorkflowTask


@pytest.mark.unit
class TestWorkflowPlannerSignature:
    """Test DSPy signature for workflow planning"""

    @pytest.mark.ci_fast
    def test_workflow_planner_signature_structure(self, telemetry_manager_without_phoenix):
        """Test WorkflowPlannerSignature has correct DSPy structure"""
        signature = WorkflowPlannerSignature

        # Check that signature exists and has proper docstring (used by DSPy)
        assert signature is not None
        assert hasattr(signature, "__doc__")
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
    def test_result_aggregator_signature_structure(self, telemetry_manager_without_phoenix):
        """Test ResultAggregatorSignature has correct DSPy structure"""
        signature = ResultAggregatorSignature

        # Check that signature exists and has proper docstring (used by DSPy)
        assert signature is not None
        assert hasattr(signature, "__doc__")
        assert "DSPy signature for cross-modal fusion" in signature.__doc__

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
        mock_agent.route_query = AsyncMock(
            return_value={
                "workflow_type": "multi_agent",
                "agents_to_call": ["video_search", "summarizer"],
                "confidence": 0.9,
            }
        )
        return mock_agent

    @pytest.fixture
    def sample_agents_config(self):
        """Sample agent configuration"""
        return {
            "video_search_agent": {
                "capabilities": ["video_content_search", "visual_query_analysis"],
                "url": "http://localhost:8002",
                "timeout": 30,
            },
            "summarizer_agent": {
                "capabilities": ["text_summarization", "content_synthesis"],
                "url": "http://localhost:8003",
                "timeout": 45,
            },
        }

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.ci_fast
    def test_orchestrator_initialization_default(
        self, mock_workflow_intel, mock_a2a, mock_routing, telemetry_manager_without_phoenix
    ):
        """Test MultiAgentOrchestrator initialization with defaults"""
        mock_routing_instance = Mock()
        mock_routing.return_value = mock_routing_instance
        mock_a2a_instance = Mock()
        mock_a2a.return_value = mock_a2a_instance
        mock_workflow_intel_instance = Mock()
        mock_workflow_intel.return_value = mock_workflow_intel_instance

        orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config)

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

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @pytest.mark.ci_fast
    def test_orchestrator_initialization_custom_config(
        self, mock_a2a, mock_routing, sample_agents_config, telemetry_manager_without_phoenix
    ):
        """Test MultiAgentOrchestrator with custom configuration"""
        mock_routing_instance = Mock()
        mock_routing.return_value = mock_routing_instance

        orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config, available_agents=sample_agents_config,
            max_parallel_tasks=5,
            workflow_timeout_minutes=20,
            enable_workflow_intelligence=False,
        )

        assert orchestrator.available_agents == sample_agents_config
        assert orchestrator.max_parallel_tasks == 5
        assert orchestrator.workflow_timeout == timedelta(minutes=20)
        assert orchestrator.enable_workflow_intelligence is False
        assert orchestrator.workflow_intelligence is None

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.ci_fast
    def test_get_default_agents(self, mock_workflow_intel, mock_a2a, mock_routing, telemetry_manager_without_phoenix):
        """Test default agent configuration"""
        orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config)

        default_agents = orchestrator._get_default_agents()

        assert "video_search_agent" in default_agents
        assert "summarizer_agent" in default_agents
        assert "detailed_report_agent" in default_agents

        # Check video search agent config
        video_agent = default_agents["video_search_agent"]
        assert "video_content_search" in video_agent["capabilities"]
        assert "endpoint" in video_agent
        assert "timeout_seconds" in video_agent

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence")
    def test_initialize_dspy_modules(self, mock_workflow_intel, mock_a2a, mock_routing, telemetry_manager_without_phoenix):
        """Test DSPy module initialization"""
        orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config)

        # DSPy modules should be initialized
        assert hasattr(orchestrator, "workflow_planner")
        assert hasattr(orchestrator, "result_aggregator")

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.asyncio
    async def test_process_complex_query_basic(
        self, mock_workflow_intel, mock_a2a, mock_routing, mock_routing_agent, telemetry_manager_without_phoenix
    ):
        """Test basic complex query processing"""
        mock_routing.return_value = mock_routing_agent

        orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config)

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
            reasoning="Simple video search workflow",
        )

        query = "Find videos about machine learning"

        # Mock _execute_workflow to avoid complex execution
        orchestrator._execute_workflow = AsyncMock(
            return_value={
                "status": "completed",
                "results": {"video_search_agent": {"videos": [], "count": 0}},
                "execution_time": 1.5,
            }
        )

        result = await orchestrator.process_complex_query(query)

        assert result is not None
        assert "status" in result


@pytest.mark.unit
class TestMultiAgentOrchestratorWorkflowExecution:
    """Test workflow execution functionality"""

    @pytest.fixture
    def orchestrator_with_mocks(self, telemetry_manager_without_phoenix):
        """Create orchestrator with mocked dependencies"""
        with (
            patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent"),
            patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
            patch(
                "cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"
            ),
        ):

            orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config, enable_workflow_intelligence=False)
            return orchestrator

    def test_create_fallback_workflow_plan_structure(self, orchestrator_with_mocks):
        """Test fallback workflow plan creation"""
        orchestrator = orchestrator_with_mocks

        # Test creating a workflow plan with the actual method signature
        workflow_plan = orchestrator._create_fallback_workflow_plan(
            workflow_id="test-workflow-123",
            query="test query",
            context="test context",
            user_id="test-user",
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

        from cogniverse_agents.workflow_types import WorkflowPlan

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="test query",
            status=WorkflowStatus.COMPLETED,
            tasks=[],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Initial stats
        initial_completed = orchestrator.orchestration_stats["completed_workflows"]

        # Update stats
        orchestrator._update_orchestration_stats(workflow_plan, True)

        # Check updates
        assert (
            orchestrator.orchestration_stats["completed_workflows"]
            == initial_completed + 1
        )


@pytest.mark.unit
class TestMultiAgentOrchestratorEdgeCases:
    """Test edge cases and error handling"""

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    def test_orchestrator_with_disabled_workflow_intelligence(
        self, mock_a2a, mock_routing, telemetry_manager_without_phoenix
    ):
        """Test orchestrator when workflow intelligence is disabled"""
        orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config, enable_workflow_intelligence=False)

        assert orchestrator.enable_workflow_intelligence is False
        assert orchestrator.workflow_intelligence is None

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @pytest.mark.ci_fast
    def test_orchestrator_agent_utilization_tracking(self, mock_a2a, mock_routing, telemetry_manager_without_phoenix):
        """Test agent utilization statistics"""
        orchestrator = MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config, enable_workflow_intelligence=False)

        # Test that agent utilization dict is initialized
        stats = orchestrator.orchestration_stats["agent_utilization"]
        assert isinstance(stats, dict)
        assert len(stats) == 0


@pytest.mark.unit
class TestCrossModalFusion:
    """Test cross-modal fusion functionality"""

    @pytest.fixture
    def orchestrator(self, telemetry_manager_without_phoenix):
        """Create orchestrator for testing fusion"""
        with (
            patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent"),
            patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
        ):
            return MultiAgentOrchestrator(tenant_id="test_tenant", telemetry_config=telemetry_manager_without_phoenix.config, enable_workflow_intelligence=False)

    @pytest.fixture
    def sample_task_results(self):
        """Sample task results from different modalities"""
        return {
            "task_video": {
                "agent": "video_search_agent",
                "modality": "video",
                "query": "Find videos about machine learning",
                "result": {"videos": [{"title": "ML Basics", "score": 0.9}]},
                "execution_time": 1.5,
                "confidence": 0.9,
            },
            "task_image": {
                "agent": "image_search_agent",
                "modality": "image",
                "query": "Find images about machine learning",
                "result": {
                    "images": [{"title": "Neural Network Diagram", "score": 0.85}]
                },
                "execution_time": 1.2,
                "confidence": 0.85,
            },
            "task_text": {
                "agent": "text_search_agent",
                "modality": "text",
                "query": "Find documents about machine learning",
                "result": {
                    "documents": [{"title": "Introduction to ML", "score": 0.8}]
                },
                "execution_time": 0.9,
                "confidence": 0.8,
            },
        }

    @pytest.mark.ci_fast
    def test_detect_agent_modality(self, orchestrator):
        """Test modality detection from agent names"""
        assert orchestrator._detect_agent_modality("video_search_agent") == "video"
        assert orchestrator._detect_agent_modality("image_search_agent") == "image"
        assert orchestrator._detect_agent_modality("audio_analysis_agent") == "audio"
        assert orchestrator._detect_agent_modality("document_agent") == "document"
        assert orchestrator._detect_agent_modality("text_search_agent") == "text"
        assert orchestrator._detect_agent_modality("unknown_agent") == "text"  # Default

    @pytest.mark.ci_fast
    def test_select_fusion_strategy_temporal(self, orchestrator):
        """Test temporal fusion strategy selection"""
        query = "Show me the timeline of events in this sequence"
        agent_modalities = {
            "task1": "video",
            "task2": "image",
        }

        strategy = orchestrator._select_fusion_strategy(query, agent_modalities)
        assert strategy == FusionStrategy.TEMPORAL

    @pytest.mark.ci_fast
    def test_select_fusion_strategy_hierarchical(self, orchestrator):
        """Test hierarchical fusion strategy selection"""
        query = "Compare video results versus image results"
        agent_modalities = {
            "task1": "video",
            "task2": "image",
        }

        strategy = orchestrator._select_fusion_strategy(query, agent_modalities)
        assert strategy == FusionStrategy.HIERARCHICAL

    @pytest.mark.ci_fast
    def test_select_fusion_strategy_semantic(self, orchestrator):
        """Test semantic fusion strategy selection"""
        query = "Explain the concept of machine learning across different sources"
        agent_modalities = {
            "task1": "video",
            "task2": "document",
        }

        strategy = orchestrator._select_fusion_strategy(query, agent_modalities)
        assert strategy == FusionStrategy.SEMANTIC

    @pytest.mark.ci_fast
    def test_select_fusion_strategy_score_based(self, orchestrator):
        """Test score-based fusion strategy selection"""
        query = "Find all content about robotics"
        agent_modalities = {
            "task1": "video",
            "task2": "image",
            "task3": "document",
        }

        strategy = orchestrator._select_fusion_strategy(query, agent_modalities)
        assert strategy == FusionStrategy.SCORE_BASED

    @pytest.mark.ci_fast
    def test_select_fusion_strategy_simple(self, orchestrator):
        """Test simple fusion for single modality"""
        query = "Find videos about robotics"
        agent_modalities = {
            "task1": "video",
        }

        strategy = orchestrator._select_fusion_strategy(query, agent_modalities)
        assert strategy == FusionStrategy.SIMPLE

    @pytest.mark.ci_fast
    def test_fuse_by_score(self, orchestrator, sample_task_results):
        """Test score-based fusion"""
        result = orchestrator._fuse_by_score(sample_task_results)

        assert "content" in result
        assert "confidence" in result
        assert result["confidence"] > 0
        # Should be weighted by confidence scores
        assert "VIDEO" in result["content"]  # Highest confidence
        assert "IMAGE" in result["content"]
        assert "TEXT" in result["content"]

    @pytest.mark.ci_fast
    def test_fuse_by_score_empty_results(self, orchestrator):
        """Test score-based fusion with empty results"""
        result = orchestrator._fuse_by_score({})

        assert result["content"] == ""
        assert result["confidence"] == 0.0

    @pytest.mark.ci_fast
    def test_fuse_by_temporal_alignment(self, orchestrator, sample_task_results):
        """Test temporal fusion"""
        result = orchestrator._fuse_by_temporal_alignment(sample_task_results)

        assert "content" in result
        assert "confidence" in result
        # Should be ordered by execution time
        content = result["content"]
        text_pos = content.find("TEXT")
        image_pos = content.find("IMAGE")
        video_pos = content.find("VIDEO")
        # TEXT (0.9s) < IMAGE (1.2s) < VIDEO (1.5s)
        assert text_pos < image_pos < video_pos

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_fuse_by_semantic_similarity(self, orchestrator, sample_task_results):
        """Test semantic fusion"""
        query = "machine learning"
        result = await orchestrator._fuse_by_semantic_similarity(
            sample_task_results, query
        )

        assert "content" in result
        assert "confidence" in result
        # Should contain relevance scores
        assert "relevance:" in result["content"]

    @pytest.mark.ci_fast
    def test_fuse_hierarchically(self, orchestrator, sample_task_results):
        """Test hierarchical fusion"""
        agent_modalities = {
            "task_video": "video",
            "task_image": "image",
            "task_text": "text",
        }

        result = orchestrator._fuse_hierarchically(
            sample_task_results, agent_modalities
        )

        assert "content" in result
        assert "confidence" in result
        # Should have modality headers
        assert "## VIDEO RESULTS" in result["content"]
        assert "## IMAGE RESULTS" in result["content"]
        assert "## TEXT RESULTS" in result["content"]
        # Should show confidence per modality
        assert "Confidence:" in result["content"]

    @pytest.mark.ci_fast
    def test_fuse_simple(self, orchestrator, sample_task_results):
        """Test simple fusion"""
        result = orchestrator._fuse_simple(sample_task_results)

        assert "content" in result
        assert "confidence" in result
        # Simple concatenation
        assert len(result["content"]) > 0

    @pytest.mark.ci_fast
    def test_check_cross_modal_consistency_single_modality(self, orchestrator):
        """Test cross-modal consistency with single modality"""
        task_results = {
            "task1": {
                "modality": "video",
                "result": "test result",
                "confidence": 0.9,
            }
        }

        consistency = orchestrator._check_cross_modal_consistency(task_results)

        assert consistency["consistency_score"] == 1.0
        assert "Single modality" in consistency["note"]

    @pytest.mark.ci_fast
    def test_check_cross_modal_consistency_multiple_modalities(
        self, orchestrator, sample_task_results
    ):
        """Test cross-modal consistency with multiple modalities"""
        consistency = orchestrator._check_cross_modal_consistency(sample_task_results)

        assert "consistency_score" in consistency
        assert "confidence_variance" in consistency
        assert "conflicts" in consistency
        assert "agreements" in consistency
        assert consistency["modality_count"] == 3

    @pytest.mark.ci_fast
    def test_calculate_fusion_quality(self, orchestrator, sample_task_results):
        """Test fusion quality metrics calculation"""
        fused_result = {
            "content": "Combined results",
            "confidence": 0.85,
        }
        consistency_metrics = {
            "consistency_score": 0.7,
            "confidence_variance": 0.05,
        }

        quality = orchestrator._calculate_fusion_quality(
            sample_task_results, fused_result, consistency_metrics
        )

        assert "overall_quality" in quality
        assert "coverage" in quality
        assert "consistency" in quality
        assert "coherence" in quality
        assert "redundancy" in quality
        assert "complementarity" in quality
        assert "modality_count" in quality
        assert quality["modality_count"] == 3
        assert 0.0 <= quality["overall_quality"] <= 1.0

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_aggregate_results_with_fusion(self, orchestrator):
        """Test _aggregate_results with cross-modal fusion"""
        from datetime import datetime

        from cogniverse_agents.workflow_types import WorkflowPlan

        # Create mock workflow plan with completed tasks
        tasks = []
        for agent_name in ["video_search_agent", "image_search_agent"]:
            task = WorkflowTask(
                task_id=f"task_{agent_name}",
                agent_name=agent_name,
                query="Find content about robotics",
                dependencies=set(),
            )
            task.status = TaskStatus.COMPLETED
            task.start_time = datetime.now()
            task.end_time = datetime.now()
            task.result = {
                "content": f"Results from {agent_name}",
                "confidence": 0.85,
            }
            tasks.append(task)

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="Find content about robotics",
            status=WorkflowStatus.COMPLETED,
            tasks=tasks,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        result = await orchestrator._aggregate_results(workflow_plan)

        assert "aggregated_content" in result
        assert "confidence" in result
        assert "fusion_strategy" in result
        assert "fusion_quality" in result
        assert "cross_modal_consistency" in result
        assert "modality_coverage" in result
        assert "video" in result["modality_coverage"]
        assert "image" in result["modality_coverage"]

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_aggregate_results_no_completed_tasks(self, orchestrator):
        """Test _aggregate_results with no completed tasks"""
        from datetime import datetime

        from cogniverse_agents.workflow_types import WorkflowPlan

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="Test query",
            status=WorkflowStatus.FAILED,
            tasks=[],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        result = await orchestrator._aggregate_results(workflow_plan)

        assert "error" in result
        assert result["error"] == "No completed tasks to aggregate"


@pytest.mark.unit
class TestFusionStrategyEnum:
    """Test FusionStrategy enum"""

    @pytest.mark.ci_fast
    def test_fusion_strategy_values(self, telemetry_manager_without_phoenix):
        """Test FusionStrategy enum values"""
        assert FusionStrategy.SCORE_BASED.value == "score"
        assert FusionStrategy.TEMPORAL.value == "temporal"
        assert FusionStrategy.SEMANTIC.value == "semantic"
        assert FusionStrategy.HIERARCHICAL.value == "hierarchical"
        assert FusionStrategy.SIMPLE.value == "simple"

    @pytest.mark.ci_fast
    def test_fusion_strategy_members(self, telemetry_manager_without_phoenix):
        """Test FusionStrategy enum has all expected members"""
        strategies = [s.value for s in FusionStrategy]
        assert "score" in strategies
        assert "temporal" in strategies
        assert "semantic" in strategies
        assert "hierarchical" in strategies
        assert "simple" in strategies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
