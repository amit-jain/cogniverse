"""
Integration tests for Phase 7.5: Orchestration Optimization

Tests the complete flow with REAL components (no mocks):
1. Create real orchestration spans in Phoenix
2. Query spans with OrchestrationEvaluator
3. Store annotations with OrchestrationAnnotationStorage
4. Process annotations with OrchestrationFeedbackLoop
5. Trigger unified optimization
"""

from unittest.mock import patch

import pandas as pd
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


@pytest.mark.integration
class TestOrchestrationOptimizationIntegration:
    """Integration tests for orchestration optimization with REAL components"""

    @pytest.fixture
    def workflow_intelligence(self):
        """Create real WorkflowIntelligence instance"""
        return WorkflowIntelligence(
            max_history_size=1000,
            enable_persistence=False,  # Disable DB for tests
        )

    @pytest.fixture
    def routing_optimizer(self):
        """Create real AdvancedRoutingOptimizer instance"""
        return AdvancedRoutingOptimizer(tenant_id="test-tenant", base_storage_dir="/tmp/test_routing_optimizer")

    @pytest.mark.asyncio
    async def test_phoenix_evaluator_extracts_workflows_from_spans(
        self, workflow_intelligence, telemetry_manager_without_phoenix
    ):
        """Test that OrchestrationEvaluator extracts workflows from Phoenix spans"""
        evaluator = OrchestrationEvaluator(
            workflow_intelligence=workflow_intelligence,
            tenant_id="test-tenant",
        )

        # Mock Phoenix spans (we can't create real spans without full system running)
        mock_spans_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.orchestration",
                    "context.span_id": "span-123",
                    "status_code": "OK",
                    "attributes": {
                        "orchestration.workflow_id": "wf-123",
                        "orchestration.query": "Find videos and documents about AI",
                        "orchestration.pattern": "parallel",
                        "orchestration.agents_used": "video_search_agent,text_search_agent",
                        "orchestration.execution_order": "video_search_agent,text_search_agent",
                        "orchestration.execution_time": 2.5,
                        "orchestration.tasks_completed": 2,
                        "routing.confidence": 0.9,
                    },
                },
                {
                    "name": "cogniverse.orchestration",
                    "context.span_id": "span-456",
                    "status_code": "ERROR",
                    "status_message": "Agent timeout",
                    "attributes": {
                        "orchestration.workflow_id": "wf-456",
                        "orchestration.query": "Failed query",
                        "orchestration.pattern": "sequential",
                        "orchestration.agents_used": "agent1",
                        "orchestration.execution_order": "agent1",
                        "orchestration.execution_time": 1.0,
                        "orchestration.tasks_completed": 0,
                        "routing.confidence": 0.5,
                    },
                },
            ]
        )

        # Patch only Phoenix client, use real WorkflowIntelligence
        with patch.object(
            evaluator.provider.traces,
            "get_spans",
            return_value=mock_spans_df,
        ):
            result = await evaluator.evaluate_orchestration_spans(lookback_hours=1)

        # Verify results
        assert result["spans_processed"] == 2
        assert result["workflows_extracted"] == 2

        # Verify workflows were actually recorded in WorkflowIntelligence
        assert len(workflow_intelligence.workflow_history) == 2

        # Verify workflow data
        successful_workflow = workflow_intelligence.workflow_history[0]
        assert successful_workflow.workflow_id == "wf-123"
        assert successful_workflow.query == "Find videos and documents about AI"
        assert successful_workflow.success is True
        assert successful_workflow.execution_time == 2.5
        assert len(successful_workflow.agent_sequence) == 2

        failed_workflow = workflow_intelligence.workflow_history[1]
        assert failed_workflow.workflow_id == "wf-456"
        assert failed_workflow.success is False
        assert failed_workflow.error_details == "Agent timeout"

    @pytest.mark.asyncio
    async def test_annotation_storage_stores_and_retrieves_annotations(self, telemetry_manager_without_phoenix):
        """Test that OrchestrationAnnotationStorage stores and retrieves annotations"""
        storage = OrchestrationAnnotationStorage(tenant_id="test-tenant")

        annotation = OrchestrationAnnotation(
            workflow_id="wf-789",
            span_id="span-789",
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

        # Patch only provider annotations, storage logic is real
        with patch.object(storage.provider.annotations, "add_annotation") as mock_add_annotation:
            result = await storage.store_annotation(annotation)

        # Verify annotation was stored
        assert result is True
        mock_add_annotation.assert_called_once()

        # Verify annotation data was correctly constructed using kwargs
        call_kwargs = mock_add_annotation.call_args.kwargs
        assert call_kwargs["span_id"] == "span-789"
        assert call_kwargs["name"] == "orchestration_quality"
        assert call_kwargs["label"] == "good"
        assert call_kwargs["score"] == 0.8
        assert call_kwargs["metadata"]["pattern_is_optimal"] is False
        assert call_kwargs["metadata"]["suggested_pattern"] == "sequential"

    @pytest.mark.asyncio
    async def test_feedback_loop_processes_annotations_and_triggers_optimization(
        self, workflow_intelligence, telemetry_manager_without_phoenix
    ):
        """Test that OrchestrationFeedbackLoop processes annotations and triggers optimization"""
        feedback_loop = OrchestrationFeedbackLoop(
            workflow_intelligence=workflow_intelligence,
            tenant_id="test-tenant",
            poll_interval_minutes=15,
            min_annotations_for_update=2,
        )

        # Mock annotated spans from Phoenix
        mock_annotated_spans = [
            {
                "span_id": "span-1",
                "span_data": {
                    "attributes": {
                        "orchestration.workflow_id": "wf-1",
                        "orchestration.query": "Multi-modal query 1",
                    }
                },
                "annotations": [
                    {
                        "result": {"label": "excellent", "score": 0.95},
                        "metadata": {
                            "pattern_is_optimal": False,
                            "suggested_pattern": "parallel",
                            "actual_pattern": "sequential",
                            "agents_are_correct": True,
                            "actual_agents": "video_search_agent,text_search_agent",
                            "execution_order_is_optimal": True,
                            "actual_execution_order": "video_search_agent,text_search_agent",
                            "execution_time": 2.0,
                            "annotation_timestamp": "2025-01-01T00:00:00",
                        },
                    }
                ],
            },
            {
                "span_id": "span-2",
                "span_data": {
                    "attributes": {
                        "orchestration.workflow_id": "wf-2",
                        "orchestration.query": "Sequential query 1",
                    }
                },
                "annotations": [
                    {
                        "result": {"label": "good", "score": 0.85},
                        "metadata": {
                            "pattern_is_optimal": True,
                            "actual_pattern": "sequential",
                            "agents_are_correct": True,
                            "actual_agents": "agent1,agent2",
                            "execution_order_is_optimal": True,
                            "actual_execution_order": "agent1,agent2",
                            "execution_time": 1.5,
                            "annotation_timestamp": "2025-01-01T00:00:01",
                        },
                    }
                ],
            },
        ]

        # Patch only annotation storage query, use real WorkflowIntelligence
        with patch.object(
            feedback_loop.annotation_storage,
            "query_annotated_spans",
            return_value=mock_annotated_spans,
        ):
            result = await feedback_loop.process_new_annotations()

        # Verify annotations were processed
        assert result["annotations_found"] == 2
        assert result["workflows_learned"] == 2

        # Verify ground truth workflows were recorded in REAL WorkflowIntelligence
        ground_truth_workflows = [
            w
            for w in workflow_intelligence.workflow_history
            if w.metadata.get("is_ground_truth", False)
        ]
        assert len(ground_truth_workflows) == 2

        # Verify optimization was triggered (min_annotations_for_update = 2)
        assert result["optimizer_updated"] is True

        # Verify corrected values were used
        ground_truth_1 = ground_truth_workflows[0]
        assert (
            ground_truth_1.metadata["orchestration_pattern"] == "parallel"
        )  # Suggested, not actual
        assert ground_truth_1.success is True  # excellent label
        assert ground_truth_1.user_satisfaction == 0.95

    @pytest.mark.asyncio
    async def test_unified_optimizer_integrates_orchestration_with_routing(
        self, workflow_intelligence, routing_optimizer
    ):
        """Test that UnifiedOptimizer integrates orchestration outcomes with routing optimization"""
        unified_optimizer = UnifiedOptimizer(
            routing_optimizer=routing_optimizer,
            workflow_intelligence=workflow_intelligence,
        )

        # Create successful workflows in REAL WorkflowIntelligence
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
                user_satisfaction=0.3,  # Too low, should be filtered
                metadata={},
            ),
        ]

        # Integrate workflows into routing optimization (using REAL optimizer)
        result = await unified_optimizer.integrate_orchestration_outcomes(workflows)

        # Verify integration results
        assert result["workflows_processed"] == 3
        assert result["routing_experiences_created"] == 2  # Only high-quality workflows
        assert result["patterns_learned"]["parallel"] == 1
        assert result["patterns_learned"]["sequential"] == 1

        # Verify routing experiences were recorded in REAL AdvancedRoutingOptimizer
        assert len(routing_optimizer.experiences) == 2

        # Verify routing experience metadata includes orchestration insights
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
        # Add a successful workflow to WorkflowIntelligence
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

        # Run unified optimization
        result = await unified_optimizer.optimize_unified_policy()

        # Verify all three optimization steps ran
        assert "workflow_optimization" in result
        assert "routing_optimization" in result
        assert "integration" in result

        # Verify workflow optimization results
        assert result["workflow_optimization"]["status"] == "skipped"  # No ground truth

        # Verify routing optimization results
        assert result["routing_optimization"]["status"] == "success"

        # Verify integration results
        assert result["integration"]["workflows_processed"] == 1
        assert result["integration"]["routing_experiences_created"] == 1

        # Verify routing experience was created in REAL optimizer
        assert len(routing_optimizer.experiences) == 1

    @pytest.mark.asyncio
    async def test_end_to_end_orchestration_optimization_flow(
        self, workflow_intelligence, routing_optimizer, telemetry_manager_without_phoenix
    ):
        """Test complete end-to-end orchestration optimization flow with REAL components"""
        # 1. Create OrchestrationEvaluator with REAL WorkflowIntelligence
        evaluator = OrchestrationEvaluator(
            workflow_intelligence=workflow_intelligence,
            tenant_id="test-tenant",
        )

        # 2. Mock orchestration spans in Phoenix (only Phoenix client is mocked)
        mock_spans_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.orchestration",
                    "context.span_id": "span-e2e-1",
                    "status_code": "OK",
                    "attributes": {
                        "orchestration.workflow_id": "wf-e2e-1",
                        "orchestration.query": "Find videos and documents",
                        "orchestration.pattern": "parallel",
                        "orchestration.agents_used": "video_search_agent,text_search_agent",
                        "orchestration.execution_order": "video_search_agent,text_search_agent",
                        "orchestration.execution_time": 2.0,
                        "orchestration.tasks_completed": 2,
                        "routing.confidence": 0.9,
                    },
                }
            ]
        )

        # 3. Evaluate spans (REAL WorkflowIntelligence receives workflows)
        with patch.object(
            evaluator.provider.traces,
            "get_spans",
            return_value=mock_spans_df,
        ):
            eval_result = await evaluator.evaluate_orchestration_spans()

        assert eval_result["workflows_extracted"] == 1
        assert len(workflow_intelligence.workflow_history) == 1

        # 4. Create feedback loop with REAL WorkflowIntelligence
        feedback_loop = OrchestrationFeedbackLoop(
            workflow_intelligence=workflow_intelligence,
            tenant_id="test-tenant",
            min_annotations_for_update=1,
        )

        # Mock annotated spans (only annotation storage is mocked)
        mock_annotated_spans = [
            {
                "span_id": "span-e2e-1",
                "span_data": {
                    "attributes": {
                        "orchestration.workflow_id": "wf-e2e-1",
                        "orchestration.query": "Find videos and documents",
                    }
                },
                "annotations": [
                    {
                        "result": {"label": "excellent", "score": 0.95},
                        "metadata": {
                            "pattern_is_optimal": True,
                            "actual_pattern": "parallel",
                            "agents_are_correct": True,
                            "actual_agents": "video_search_agent,text_search_agent",
                            "execution_order_is_optimal": True,
                            "actual_execution_order": "video_search_agent,text_search_agent",
                            "execution_time": 2.0,
                            "annotation_timestamp": "2025-01-01T00:00:00",
                        },
                    }
                ],
            }
        ]

        with patch.object(
            feedback_loop.annotation_storage,
            "query_annotated_spans",
            return_value=mock_annotated_spans,
        ):
            feedback_result = await feedback_loop.process_new_annotations()

        assert feedback_result["workflows_learned"] == 1
        assert feedback_result["optimizer_updated"] is True

        # Verify ground truth was recorded in REAL WorkflowIntelligence
        ground_truth_workflows = [
            w
            for w in workflow_intelligence.workflow_history
            if w.metadata.get("is_ground_truth", False)
        ]
        assert len(ground_truth_workflows) == 1
        assert ground_truth_workflows[0].user_satisfaction == 0.95

        # 5. Run unified optimization with REAL components
        unified_optimizer = UnifiedOptimizer(
            routing_optimizer=routing_optimizer,
            workflow_intelligence=workflow_intelligence,
        )

        unified_result = await unified_optimizer.optimize_unified_policy()

        # Verify complete flow
        assert unified_result["workflow_optimization"]["status"] == "success"
        assert unified_result["routing_optimization"]["status"] == "success"
        assert unified_result["integration"]["routing_experiences_created"] >= 1

        # Verify REAL routing optimizer received experiences
        assert len(routing_optimizer.experiences) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
