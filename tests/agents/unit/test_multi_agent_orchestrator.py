"""
Unit tests for MultiAgentOrchestrator with DSPy integration
"""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from cogniverse_agents.multi_agent_orchestrator import (
    FusionStrategy,
    MultiAgentOrchestrator,
    ResultAggregatorSignature,
    WorkflowPlannerSignature,
)
from cogniverse_agents.workflow_types import (
    TaskStatus,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowTask,
)


@pytest.mark.unit
class TestWorkflowPlannerSignature:
    """Test DSPy signature for workflow planning"""

    @pytest.mark.ci_fast
    def test_workflow_planner_signature_structure(
        self, telemetry_manager_without_phoenix
    ):
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
    def test_result_aggregator_signature_structure(
        self, telemetry_manager_without_phoenix
    ):
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
        self,
        mock_workflow_intel,
        mock_a2a,
        mock_routing,
        telemetry_manager_without_phoenix,
    ):
        """Test MultiAgentOrchestrator initialization with defaults"""
        mock_routing_instance = Mock()
        mock_routing.return_value = mock_routing_instance
        mock_a2a_instance = Mock()
        mock_a2a.return_value = mock_a2a_instance
        mock_workflow_intel_instance = Mock()
        mock_workflow_intel.return_value = mock_workflow_intel_instance

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
        )

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
        self,
        mock_a2a,
        mock_routing,
        sample_agents_config,
        telemetry_manager_without_phoenix,
    ):
        """Test MultiAgentOrchestrator with custom configuration"""
        mock_routing_instance = Mock()
        mock_routing.return_value = mock_routing_instance

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
            available_agents=sample_agents_config,
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
    def test_get_default_agents(
        self,
        mock_workflow_intel,
        mock_a2a,
        mock_routing,
        telemetry_manager_without_phoenix,
    ):
        """Test default agent configuration"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
        )

        default_agents = orchestrator._get_default_agents()

        assert "search_agent" in default_agents
        assert "summarizer_agent" in default_agents
        assert "detailed_report_agent" in default_agents

        # Check search agent config
        search_agent = default_agents["search_agent"]
        assert "video_content_search" in search_agent["capabilities"]
        assert "endpoint" in search_agent
        assert "timeout_seconds" in search_agent

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence")
    def test_initialize_dspy_modules(
        self,
        mock_workflow_intel,
        mock_a2a,
        mock_routing,
        telemetry_manager_without_phoenix,
    ):
        """Test DSPy module initialization"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
        )

        # DSPy modules should be initialized
        assert hasattr(orchestrator, "workflow_planner")
        assert hasattr(orchestrator, "result_aggregator")

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.asyncio
    async def test_process_complex_query_basic(
        self,
        mock_workflow_intel,
        mock_a2a,
        mock_routing,
        mock_routing_agent,
        telemetry_manager_without_phoenix,
    ):
        """Test basic complex query processing"""
        mock_routing.return_value = mock_routing_agent

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
        )

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
            orchestrator = MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                enable_workflow_intelligence=False,
            )
            return orchestrator

    @pytest.mark.asyncio
    async def test_plan_workflow_raises_runtime_error_on_failure(
        self, orchestrator_with_mocks
    ):
        """Test that _plan_workflow raises RuntimeError instead of silently falling back"""
        orchestrator = orchestrator_with_mocks

        # Force DSPy planner to raise an exception
        orchestrator.workflow_planner = Mock(
            side_effect=Exception("DSPy planner unavailable")
        )

        with pytest.raises(RuntimeError, match="Workflow planning failed"):
            await orchestrator._plan_workflow(
                workflow_id="test-workflow-123",
                query="test query",
                context="test context",
                user_id="test-user",
                preferences=None,
            )

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
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
            enable_workflow_intelligence=False,
        )

        assert orchestrator.enable_workflow_intelligence is False
        assert orchestrator.workflow_intelligence is None

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @pytest.mark.ci_fast
    def test_orchestrator_agent_utilization_tracking(
        self, mock_a2a, mock_routing, telemetry_manager_without_phoenix
    ):
        """Test agent utilization statistics"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
            enable_workflow_intelligence=False,
        )

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
            return MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                enable_workflow_intelligence=False,
            )

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


@pytest.mark.unit
class TestOrchestratorTelemetrySpan:
    """Test orchestration telemetry span instrumentation"""

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence")
    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_span_attributes_set_on_success(
        self,
        mock_workflow_intel,
        mock_a2a,
        mock_routing,
        telemetry_manager_without_phoenix,
    ):
        """Test that cogniverse.orchestration span attributes are set correctly"""
        mock_routing.return_value = Mock()
        mock_span = Mock()
        mock_span.set_attribute = Mock()

        from contextlib import contextmanager

        @contextmanager
        def fake_span(*args, **kwargs):
            yield mock_span

        telemetry_manager_without_phoenix.span = fake_span

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
        )

        # Mock internals to avoid real execution
        orchestrator._plan_workflow = AsyncMock(
            return_value=Mock(
                tasks=[
                    Mock(
                        agent_name="search_agent",
                        status=TaskStatus.COMPLETED,
                        dependencies=set(),
                    ),
                    Mock(
                        agent_name="summarizer_agent",
                        status=TaskStatus.COMPLETED,
                        dependencies={"search"},
                    ),
                ],
                end_time=None,
                start_time=None,
                metadata={},
            )
        )
        orchestrator._execute_workflow = AsyncMock()
        orchestrator._aggregate_results = AsyncMock(return_value={"content": "results"})
        orchestrator._update_orchestration_stats = Mock()

        # Set end_time/start_time after plan
        plan_mock = orchestrator._plan_workflow.return_value
        from datetime import datetime

        plan_mock.start_time = datetime.now()
        plan_mock.end_time = datetime.now()

        result = await orchestrator.process_complex_query("test multi-step query")

        assert result["status"] == "completed"

        # Verify span attributes were set
        attr_calls = {
            call.args[0]: call.args[1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert attr_calls["orchestration.query"] == "test multi-step query"
        assert "orchestration.workflow_id" in attr_calls
        assert attr_calls["orchestration.pattern"] == "mixed"
        assert "orchestration.execution_time" in attr_calls
        assert attr_calls["orchestration.tasks_completed"] == 2
        assert "orchestration.agents_used" in attr_calls
        assert "orchestration.execution_order" in attr_calls

    @patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent")
    @patch("cogniverse_agents.multi_agent_orchestrator.A2AClient")
    @pytest.mark.ci_fast
    def test_determine_execution_pattern(
        self,
        mock_a2a,
        mock_routing,
        telemetry_manager_without_phoenix,
    ):
        """Test _determine_execution_pattern classifies correctly"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
            enable_workflow_intelligence=False,
        )

        # No tasks = sequential
        assert orchestrator._determine_execution_pattern([]) == "sequential"

        # All tasks with no deps = parallel
        t1 = Mock(dependencies=set())
        t2 = Mock(dependencies=set())
        assert orchestrator._determine_execution_pattern([t1, t2]) == "parallel"

        # All tasks with deps = sequential
        t3 = Mock(dependencies={"t1"})
        t4 = Mock(dependencies={"t2"})
        assert orchestrator._determine_execution_pattern([t3, t4]) == "sequential"

        # Mix = mixed
        assert orchestrator._determine_execution_pattern([t1, t3]) == "mixed"


@pytest.mark.unit
class TestOrchestrationPipelineChain:
    """Tests that exercise the internal method chain with mocked system boundaries.

    All internal methods (_plan_workflow, _execute_workflow, _execute_task,
    _prepare_task_context, _aggregate_results) run with REAL code.
    httpx.AsyncClient is mocked — no real TCP connections are made.
    The DSPy modules (workflow_planner, result_aggregator) are replaced with
    controllable mocks since no LLM is available in tests.
    """

    @pytest.fixture
    def agents_config(self):
        """Agent configuration for pipeline tests."""
        return {
            "search_agent": {
                "capabilities": ["video_content_search", "multimodal_retrieval"],
                "endpoint": "http://localhost:8000",
                "timeout_seconds": 30,
                "parallel_capacity": 2,
            },
            "summarizer_agent": {
                "capabilities": ["content_summarization", "report_generation"],
                "endpoint": "http://localhost:8000",
                "timeout_seconds": 30,
                "parallel_capacity": 2,
            },
        }

    def _make_planner_result(self, tasks_data, strategy="sequential"):
        """Create a Mock mimicking DSPy workflow_planner.forward() output."""
        result = Mock()
        result.workflow_tasks = tasks_data
        result.execution_strategy = strategy
        result.expected_outcome = "results"
        result.reasoning = "test"
        return result

    @pytest.fixture
    def pipeline_orchestrator(self, telemetry_manager_without_phoenix, agents_config):
        """Construct a real MultiAgentOrchestrator with controlled DSPy mocks."""
        with (
            patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent") as mock_routing_cls,
            patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
            patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"),
        ):
            mock_routing_instance = Mock()
            mock_routing_instance.route_query = AsyncMock()
            mock_routing_cls.return_value = mock_routing_instance

            orchestrator = MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                available_agents=agents_config,
                enable_workflow_intelligence=False,
            )

            # Replace DSPy modules with controllable mocks
            orchestrator.workflow_planner = Mock()
            orchestrator.result_aggregator = Mock()

            return orchestrator

    @pytest.fixture
    def mock_httpx_success(self):
        """Patch httpx.AsyncClient to return successful responses."""
        mock_response = Mock()
        mock_response.json.return_value = {"result": "data", "confidence": 0.8}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_client_cm = Mock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "cogniverse_agents.multi_agent_orchestrator.httpx.AsyncClient",
            return_value=mock_client_cm,
        ):
            yield mock_client

    @pytest.mark.asyncio
    async def test_execute_workflow_abort_sets_end_time(
        self, pipeline_orchestrator, agents_config
    ):
        """When all tasks fail and workflow aborts, no TypeError on end_time."""
        orchestrator = pipeline_orchestrator
        orchestrator.workflow_planner.forward = Mock(
            return_value=self._make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "q1", "dependencies": []},
                {"task_id": "t2", "agent": "search_agent", "query": "q2", "dependencies": []},
            ])
        )

        # httpx always raises → both tasks fail → ≥50% → abort
        error_response = Mock()
        error_response.status_code = 500
        error_response.request = Mock(url="http://test")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server error", request=Mock(url="http://test"), response=error_response
            )
        )
        mock_client_cm = Mock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "cogniverse_agents.multi_agent_orchestrator.httpx.AsyncClient",
            return_value=mock_client_cm,
        ):
            # Set max_retries=0 on tasks after planning to avoid backoff
            original_plan = orchestrator._plan_workflow

            async def plan_then_zero_retries(*args, **kwargs):
                plan = await original_plan(*args, **kwargs)
                for task in plan.tasks:
                    task.max_retries = 0
                return plan

            orchestrator._plan_workflow = plan_then_zero_retries

            # Should not crash with TypeError about NoneType - datetime
            result = await orchestrator.process_complex_query("test")

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_fallback_passes_tenant_id(self, pipeline_orchestrator):
        """When planning fails, fallback calls route_query with tenant_id."""
        orchestrator = pipeline_orchestrator

        # Force planning to fail → triggers fallback
        orchestrator.workflow_planner.forward = Mock(
            side_effect=RuntimeError("planning failed")
        )

        # Mock route_query to return a valid RoutingOutput
        from cogniverse_agents.routing_agent import RoutingOutput

        mock_routing_output = RoutingOutput(
            query="test",
            recommended_agent="search_agent",
            confidence=0.8,
            reasoning="fallback",
            enhanced_query="test",
        )
        orchestrator.routing_agent.route_query = AsyncMock(return_value=mock_routing_output)

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "failed"
        assert "fallback_result" in result
        orchestrator.routing_agent.route_query.assert_called_once()
        call_kwargs = orchestrator.routing_agent.route_query.call_args
        assert call_kwargs.kwargs.get("tenant_id") == "test_tenant" or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "test_tenant"
        )

    @pytest.mark.asyncio
    async def test_execute_task_uses_httpx_to_correct_endpoint(
        self, pipeline_orchestrator, mock_httpx_success
    ):
        """_execute_task calls httpx with correct URL, headers, and JSON body."""
        orchestrator = pipeline_orchestrator
        orchestrator.workflow_planner.forward = Mock(
            return_value=self._make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "find videos", "dependencies": []},
            ])
        )

        result = await orchestrator.process_complex_query("find videos")

        assert result["status"] == "completed"

        # Verify httpx was called correctly
        mock_httpx_success.post.assert_called_once()
        call_args = mock_httpx_success.post.call_args

        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
        assert "/agents/search_agent/process" in url

        json_body = call_args.kwargs.get("json", {})
        assert "agent_name" in json_body
        assert "query" in json_body
        assert "context" in json_body

        headers = call_args.kwargs.get("headers", {})
        assert headers.get("Content-Type") == "application/json"

    @pytest.mark.asyncio
    async def test_plan_workflow_resolves_hallucinated_agent_names(
        self, pipeline_orchestrator, mock_httpx_success
    ):
        """Planner proposes unknown agent names → resolved to registered agents."""
        orchestrator = pipeline_orchestrator
        orchestrator.workflow_planner.forward = Mock(
            return_value=self._make_planner_result([
                {"task_id": "t1", "agent": "VideoSearchAgent", "query": "search", "dependencies": []},
                {"task_id": "t2", "agent": "ContentSummarizer", "query": "summarize", "dependencies": ["t1"]},
            ])
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"

        # Verify httpx calls used resolved agent names, not hallucinated ones
        calls = mock_httpx_success.post.call_args_list
        urls = [
            c.args[0] if c.args else c.kwargs.get("url", "")
            for c in calls
        ]
        for url in urls:
            assert "VideoSearchAgent" not in url
            assert "ContentSummarizer" not in url
        # Should have resolved to valid registered agents
        url_str = " ".join(urls)
        assert "search_agent" in url_str or "summarizer_agent" in url_str

    @pytest.mark.asyncio
    async def test_workflow_intelligence_post_optimization_validates_agents(
        self, telemetry_manager_without_phoenix, agents_config
    ):
        """After workflow intelligence optimization, invalid agents are re-resolved."""
        with (
            patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent") as mock_routing_cls,
            patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
            patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence") as mock_wi_factory,
        ):
            mock_routing_cls.return_value = Mock()

            # Create mock workflow intelligence that corrupts agent names
            mock_wi = AsyncMock()

            async def corrupt_plan(query, plan, ctx):
                """Simulate optimizer returning an invalid agent name."""
                for task in plan.tasks:
                    task.agent_name = "InvalidAgent"
                return plan

            mock_wi.optimize_workflow_plan = AsyncMock(side_effect=corrupt_plan)
            mock_wi_factory.return_value = mock_wi

            orchestrator = MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                available_agents=agents_config,
                enable_workflow_intelligence=True,
            )

            orchestrator.workflow_planner = Mock()
            orchestrator.workflow_planner.forward = Mock(
                return_value=self._make_planner_result([
                    {"task_id": "t1", "agent": "search_agent", "query": "test", "dependencies": []},
                ])
            )

            # Patch httpx to capture URLs
            mock_response = Mock()
            mock_response.json.return_value = {"result": "data", "confidence": 0.8}
            mock_response.raise_for_status = Mock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_cm = Mock()
            mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "cogniverse_agents.multi_agent_orchestrator.httpx.AsyncClient",
                return_value=mock_client_cm,
            ):
                result = await orchestrator.process_complex_query("test")

            assert result["status"] == "completed"
            # Verify InvalidAgent was NOT used in the URL
            call_url = mock_client.post.call_args.args[0]
            assert "InvalidAgent" not in call_url

    @pytest.mark.asyncio
    async def test_task_execution_uses_configured_endpoint_not_default_port(
        self, telemetry_manager_without_phoenix
    ):
        """_execute_task uses agent's configured endpoint, not a default."""
        custom_agents = {
            "search_agent": {
                "capabilities": ["video_content_search"],
                "endpoint": "http://myhost:9999",
                "timeout_seconds": 30,
                "parallel_capacity": 1,
            },
        }

        with (
            patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent") as mock_routing_cls,
            patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
            patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"),
        ):
            mock_routing_cls.return_value = Mock()
            orchestrator = MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                available_agents=custom_agents,
                enable_workflow_intelligence=False,
            )
            orchestrator.workflow_planner = Mock()
            orchestrator.workflow_planner.forward = Mock(
                return_value=self._make_planner_result([
                    {"task_id": "t1", "agent": "search_agent", "query": "test", "dependencies": []},
                ])
            )

            mock_response = Mock()
            mock_response.json.return_value = {"result": "data", "confidence": 0.8}
            mock_response.raise_for_status = Mock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_cm = Mock()
            mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "cogniverse_agents.multi_agent_orchestrator.httpx.AsyncClient",
                return_value=mock_client_cm,
            ):
                result = await orchestrator.process_complex_query("test")

            assert result["status"] == "completed"
            call_url = mock_client.post.call_args.args[0]
            assert call_url.startswith("http://myhost:9999/agents/search_agent/process")

    @pytest.mark.asyncio
    async def test_dependency_context_passed_as_string_not_dict_unpack(
        self, pipeline_orchestrator
    ):
        """Task 2's context.dependency_context is a string, not a dict."""
        orchestrator = pipeline_orchestrator
        orchestrator.workflow_planner.forward = Mock(
            return_value=self._make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "search data", "dependencies": []},
                {"task_id": "t2", "agent": "summarizer_agent", "query": "summarize", "dependencies": ["t1"]},
            ])
        )

        call_bodies = []

        first_response = Mock()
        first_response.json.return_value = {"result": "search data found", "confidence": 0.9}
        first_response.raise_for_status = Mock()

        second_response = Mock()
        second_response.json.return_value = {"result": "summary", "confidence": 0.85}
        second_response.raise_for_status = Mock()

        call_count = 0

        async def capture_post(url, json=None, headers=None):
            nonlocal call_count
            call_bodies.append({"url": url, "json": json})
            call_count += 1
            if call_count == 1:
                return first_response
            return second_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=capture_post)

        mock_client_cm = Mock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "cogniverse_agents.multi_agent_orchestrator.httpx.AsyncClient",
            return_value=mock_client_cm,
        ):
            result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"
        assert len(call_bodies) == 2

        # Second call should have dependency_context as a string
        second_body = call_bodies[1]["json"]
        dep_context = second_body["context"].get("dependency_context")
        assert dep_context is not None
        assert isinstance(dep_context, str), f"Expected str, got {type(dep_context)}"
        assert "search data found" in dep_context

    @pytest.mark.asyncio
    async def test_aggregate_results_handles_null_workflow_timestamps(
        self, pipeline_orchestrator
    ):
        """_aggregate_results with None start_time/end_time does not TypeError."""
        from datetime import datetime

        orchestrator = pipeline_orchestrator

        # Construct a WorkflowPlan with None timestamps
        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="test query",
            status=WorkflowStatus.COMPLETED,
            tasks=[],
            start_time=None,
            end_time=None,
        )

        # Add 2 completed tasks with valid timestamps
        for i, agent in enumerate(["search_agent", "summarizer_agent"]):
            task = WorkflowTask(
                task_id=f"task_{i}",
                agent_name=agent,
                query="test",
                dependencies=set(),
            )
            task.status = TaskStatus.COMPLETED
            task.start_time = datetime.now()
            task.end_time = datetime.now()
            task.result = {"content": f"result from {agent}", "confidence": 0.8}
            workflow_plan.tasks.append(task)

        # Should not raise TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
        try:
            result = await orchestrator._aggregate_results(workflow_plan)
            # If it returns successfully, verify execution_time is a number
            exec_time = result.get("workflow_metadata", {}).get("execution_time", 0)
            assert isinstance(exec_time, (int, float))
        except TypeError as e:
            if "NoneType" in str(e):
                pytest.fail(
                    f"_aggregate_results crashed on null timestamps: {e}. "
                    "Line 783 needs a None guard for workflow_plan.end_time - workflow_plan.start_time"
                )
            raise

    @pytest.mark.asyncio
    async def test_full_pipeline_plan_execute_aggregate_chain(
        self, pipeline_orchestrator
    ):
        """Full pipeline: plan → execute → aggregate with 2 dependent tasks."""
        orchestrator = pipeline_orchestrator
        orchestrator.workflow_planner.forward = Mock(
            return_value=self._make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "find videos", "dependencies": []},
                {"task_id": "t2", "agent": "summarizer_agent", "query": "summarize findings", "dependencies": ["t1"]},
            ])
        )

        call_order = []

        search_response = Mock()
        search_response.json.return_value = {"result": "video results", "confidence": 0.9}
        search_response.raise_for_status = Mock()

        summary_response = Mock()
        summary_response.json.return_value = {"result": "summary of videos", "confidence": 0.85}
        summary_response.raise_for_status = Mock()

        async def ordered_post(url, json=None, headers=None):
            call_order.append(url)
            if "search_agent" in url:
                return search_response
            return summary_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ordered_post)

        mock_client_cm = Mock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "cogniverse_agents.multi_agent_orchestrator.httpx.AsyncClient",
            return_value=mock_client_cm,
        ):
            result = await orchestrator.process_complex_query("find videos and summarize")

        assert result["status"] == "completed"
        summary = result["execution_summary"]
        assert summary["total_tasks"] == 2
        assert summary["completed_tasks"] == 2
        assert summary["execution_time"] > 0
        agents_used = summary["agents_used"]
        assert "search_agent" in agents_used
        assert "summarizer_agent" in agents_used

        # httpx called exactly 2 times, search_agent before summarizer_agent
        assert len(call_order) == 2
        assert "search_agent" in call_order[0]
        assert "summarizer_agent" in call_order[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
