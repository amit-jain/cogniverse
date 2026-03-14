"""
Integration tests for Orchestration Optimization

Tests the complete flow with real Phoenix telemetry:
1. Create real orchestration spans in Phoenix
2. Query spans with OrchestrationEvaluator
3. Store annotations with OrchestrationAnnotationStorage
4. Process annotations with OrchestrationFeedbackLoop
5. Trigger unified optimization
"""

import time

import pytest

from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.orchestration_annotation_storage import (
    OrchestrationAnnotation,
    OrchestrationAnnotationStorage,
)
from cogniverse_agents.routing.orchestration_evaluator import (
    OrchestrationEvaluator,
)
from cogniverse_agents.routing.orchestration_feedback_loop import (
    OrchestrationFeedbackLoop,
)
from cogniverse_agents.routing.unified_optimizer import UnifiedOptimizer
from cogniverse_agents.workflow_intelligence import (
    WorkflowExecution,
    WorkflowIntelligence,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from tests.utils.async_polling import wait_for_phoenix_processing

_TEST_TENANT = "orchestration_opt_test"


@pytest.fixture
def real_telemetry_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)


@pytest.fixture
def test_tenant_id():
    """Unique tenant ID for test isolation"""
    return f"orch_opt_{int(time.time())}"


@pytest.mark.integration
class TestOrchestrationOptimizationIntegration:
    """Integration tests for orchestration optimization with real Phoenix"""

    @pytest.fixture
    def workflow_intelligence(self):
        """Create real WorkflowIntelligence instance"""
        return WorkflowIntelligence(
            max_history_size=1000,
            enable_persistence=False,
        )

    @pytest.fixture
    def routing_optimizer(self, real_telemetry_provider):
        """Create real AdvancedRoutingOptimizer instance"""
        return AdvancedRoutingOptimizer(
            tenant_id=_TEST_TENANT,
            llm_config=LLMEndpointConfig(model="ollama/gemma3:4b", api_base="http://localhost:11434"),
            telemetry_provider=real_telemetry_provider,
        )

    @pytest.mark.asyncio
    async def test_phoenix_evaluator_extracts_workflows_from_spans(
        self,
        workflow_intelligence,
        telemetry_manager_with_phoenix,
        test_tenant_id,
    ):
        """Test that OrchestrationEvaluator extracts workflows from real Phoenix spans"""
        evaluator = OrchestrationEvaluator(
            workflow_intelligence=workflow_intelligence,
            tenant_id=test_tenant_id,
        )

        with telemetry_manager_with_phoenix.span(
            name="cogniverse.orchestration",
            tenant_id=test_tenant_id,
            attributes={
                "orchestration.workflow_id": "wf-123",
                "orchestration.query": "Find videos and documents about AI",
                "orchestration.pattern": "parallel",
                "orchestration.agents_used": "video_search_agent,text_search_agent",
                "orchestration.execution_order": "video_search_agent,text_search_agent",
                "orchestration.execution_time": 2.5,
                "orchestration.tasks_completed": 2,
                "routing.confidence": 0.9,
            },
        ):
            pass

        with telemetry_manager_with_phoenix.span(
            name="cogniverse.orchestration",
            tenant_id=test_tenant_id,
            attributes={
                "orchestration.workflow_id": "wf-456",
                "orchestration.query": "Failed query",
                "orchestration.pattern": "sequential",
                "orchestration.agents_used": "agent1",
                "orchestration.execution_order": "agent1",
                "orchestration.execution_time": 1.0,
                "orchestration.tasks_completed": 0,
                "routing.confidence": 0.5,
            },
        ):
            pass

        telemetry_manager_with_phoenix.force_flush(timeout_millis=5000)
        wait_for_phoenix_processing(delay=2)

        result = await evaluator.evaluate_orchestration_spans(lookback_hours=1)

        assert result["spans_processed"] == 2
        assert result["workflows_extracted"] == 2

        assert len(workflow_intelligence.workflow_history) == 2

        # Phoenix may return spans in any order, so look up by workflow_id
        workflows_by_id = {
            w.workflow_id: w for w in workflow_intelligence.workflow_history
        }
        assert "wf-123" in workflows_by_id
        assert "wf-456" in workflows_by_id

        successful_workflow = workflows_by_id["wf-123"]
        assert successful_workflow.query == "Find videos and documents about AI"
        assert successful_workflow.success is True
        assert successful_workflow.execution_time == 2.5
        assert len(successful_workflow.agent_sequence) == 2

    @pytest.mark.asyncio
    async def test_annotation_storage_stores_and_retrieves_annotations(
        self, telemetry_manager_with_phoenix, test_tenant_id
    ):
        """Test that OrchestrationAnnotationStorage stores and retrieves annotations"""
        storage = OrchestrationAnnotationStorage(tenant_id=test_tenant_id)

        with telemetry_manager_with_phoenix.span(
            name="cogniverse.orchestration",
            tenant_id=test_tenant_id,
            attributes={
                "orchestration.workflow_id": "wf-789",
                "orchestration.query": "Test query for annotation",
                "orchestration.pattern": "parallel",
            },
        ) as span:
            span_context = getattr(span, "context", None)
            if span_context and hasattr(span_context, "span_id"):
                test_span_id = str(span_context.span_id)
            else:
                test_span_id = "span-789"

        telemetry_manager_with_phoenix.force_flush(timeout_millis=5000)
        wait_for_phoenix_processing(delay=2)

        annotation = OrchestrationAnnotation(
            workflow_id="wf-789",
            span_id=test_span_id,
            query="Test query for annotation",
            orchestration_pattern="parallel",
            agents_used=["video_search_agent", "text_search_agent"],
            execution_order=["video_search_agent", "text_search_agent"],
            execution_time=2.0,
            pattern_is_optimal=False,
            suggested_pattern="sequential",
            agents_are_correct=True,
            execution_order_is_optimal=True,
            workflow_quality_label="good",
            quality_score=0.8,
            annotator_id="test-annotator",
        )

        result = await storage.store_annotation(annotation)

        assert result is True

    @pytest.mark.asyncio
    async def test_unified_optimizer_integrates_orchestration_with_routing(
        self, workflow_intelligence, routing_optimizer
    ):
        """Test that UnifiedOptimizer integrates orchestration outcomes with routing optimization"""
        unified_optimizer = UnifiedOptimizer(
            routing_optimizer=routing_optimizer,
            workflow_intelligence=workflow_intelligence,
        )

        workflows = [
            WorkflowExecution(
                workflow_id="wf-success-1",
                query="Multi-modal search query",
                query_type="multi_modal_search",
                execution_time=2.5,
                success=True,
                agent_sequence=["video_search_agent", "text_search_agent"],
                task_count=2,
                parallel_efficiency=1.8,
                confidence_score=0.9,
                user_satisfaction=0.85,
                metadata={"orchestration_pattern": "parallel"},
            ),
            WorkflowExecution(
                workflow_id="wf-success-2",
                query="Sequential analysis query",
                query_type="sequential_report",
                execution_time=3.0,
                success=True,
                agent_sequence=["agent1", "agent2", "agent3"],
                task_count=3,
                parallel_efficiency=0.0,
                confidence_score=0.95,
                user_satisfaction=0.9,
                metadata={"orchestration_pattern": "sequential"},
            ),
            WorkflowExecution(
                workflow_id="wf-low-quality",
                query="Failed query",
                query_type="test",
                execution_time=1.0,
                success=False,
                agent_sequence=["agent1"],
                task_count=1,
                parallel_efficiency=0.0,
                confidence_score=0.5,
                user_satisfaction=0.3,
                metadata={},
            ),
        ]

        result = await unified_optimizer.integrate_orchestration_outcomes(workflows)

        assert result["workflows_processed"] == 3
        assert result["routing_experiences_created"] == 2
        assert result["patterns_learned"]["parallel"] == 1
        assert result["patterns_learned"]["sequential"] == 1

        assert len(routing_optimizer.experiences) == 2

        routing_exp_1 = routing_optimizer.experiences[0]
        assert routing_exp_1.query == "Multi-modal search query"
        assert routing_exp_1.chosen_agent == "video_search_agent"
        assert routing_exp_1.search_quality == 0.85
        assert routing_exp_1.metadata["source"] == "orchestration_workflow"
        assert routing_exp_1.metadata["orchestration_pattern"] == "parallel"
        assert routing_exp_1.metadata["multi_agent_synergy"] is True

    @pytest.mark.asyncio
    async def test_unified_optimizer_runs_complete_optimization_cycle(
        self, workflow_intelligence, routing_optimizer
    ):
        """Test that UnifiedOptimizer runs complete unified optimization cycle"""
        successful_workflow = WorkflowExecution(
            workflow_id="wf-1",
            query="Test query",
            query_type="multi_modal_search",
            execution_time=2.0,
            success=True,
            agent_sequence=["video_search_agent"],
            task_count=1,
            parallel_efficiency=0.0,
            confidence_score=0.9,
            user_satisfaction=0.85,
            metadata={"orchestration_pattern": "parallel"},
        )
        workflow_intelligence.workflow_history.append(successful_workflow)

        unified_optimizer = UnifiedOptimizer(
            routing_optimizer=routing_optimizer,
            workflow_intelligence=workflow_intelligence,
        )

        result = await unified_optimizer.optimize_unified_policy()

        assert "workflow_optimization" in result
        assert "routing_optimization" in result
        assert "integration" in result

        assert result["workflow_optimization"]["status"] == "skipped"

        assert result["routing_optimization"]["status"] == "success"

        assert result["integration"]["workflows_processed"] == 1
        assert result["integration"]["routing_experiences_created"] == 1

        assert len(routing_optimizer.experiences) == 1

    @pytest.mark.asyncio
    async def test_end_to_end_orchestration_optimization_flow(
        self,
        workflow_intelligence,
        routing_optimizer,
        telemetry_manager_with_phoenix,
        test_tenant_id,
    ):
        """Test complete end-to-end orchestration optimization flow with real Phoenix"""
        evaluator = OrchestrationEvaluator(
            workflow_intelligence=workflow_intelligence,
            tenant_id=test_tenant_id,
        )

        with telemetry_manager_with_phoenix.span(
            name="cogniverse.orchestration",
            tenant_id=test_tenant_id,
            attributes={
                "orchestration.workflow_id": "wf-e2e-1",
                "orchestration.query": "Find videos and documents",
                "orchestration.pattern": "parallel",
                "orchestration.agents_used": "video_search_agent,text_search_agent",
                "orchestration.execution_order": "video_search_agent,text_search_agent",
                "orchestration.execution_time": 2.0,
                "orchestration.tasks_completed": 2,
                "routing.confidence": 0.9,
            },
        ):
            pass

        telemetry_manager_with_phoenix.force_flush(timeout_millis=5000)
        wait_for_phoenix_processing(delay=2)

        eval_result = await evaluator.evaluate_orchestration_spans()

        assert eval_result["workflows_extracted"] == 1
        assert len(workflow_intelligence.workflow_history) == 1

        feedback_loop = OrchestrationFeedbackLoop(
            workflow_intelligence=workflow_intelligence,
            tenant_id=test_tenant_id,
            min_annotations_for_update=1,
        )

        feedback_result = await feedback_loop.process_new_annotations()

        assert "annotations_found" in feedback_result
        assert "workflows_learned" in feedback_result

        unified_optimizer = UnifiedOptimizer(
            routing_optimizer=routing_optimizer,
            workflow_intelligence=workflow_intelligence,
        )

        unified_result = await unified_optimizer.optimize_unified_policy()

        assert "workflow_optimization" in unified_result
        assert "routing_optimization" in unified_result
        assert unified_result["integration"]["workflows_processed"] >= 1
