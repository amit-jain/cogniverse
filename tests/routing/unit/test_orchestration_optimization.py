"""
Unit tests for Phase 7.5: Orchestration Optimization Components

Tests:
- PhoenixOrchestrationEvaluator
- OrchestrationAnnotationStorage
- OrchestrationFeedbackLoop
- UnifiedOptimizer
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from cogniverse_agents.workflow_intelligence import WorkflowExecution
from cogniverse_agents.routing.orchestration_feedback_loop import OrchestrationFeedbackLoop
from cogniverse_agents.routing.phoenix_orchestration_evaluator import (
    PhoenixOrchestrationEvaluator,
)
from cogniverse_agents.routing.unified_optimizer import UnifiedOptimizer


class TestPhoenixOrchestrationEvaluator:
    """Test PhoenixOrchestrationEvaluator span extraction logic"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_workflow_intelligence = MagicMock()
        self.mock_workflow_intelligence.record_execution = AsyncMock()

        self.evaluator = PhoenixOrchestrationEvaluator(
            workflow_intelligence=self.mock_workflow_intelligence,
            tenant_id="test-tenant",
        )

    def test_extract_workflow_execution_success(self):
        """Test extracting WorkflowExecution from valid span data"""
        span_row = pd.Series(
            {
                "context.span_id": "span-123",
                "status_code": "OK",
                "attributes": {
                    "orchestration.workflow_id": "workflow-456",
                    "orchestration.query": "Find videos and documents about AI",
                    "orchestration.pattern": "parallel",
                    "orchestration.agents_used": "video_search_agent,text_search_agent",
                    "orchestration.execution_order": "video_search_agent,text_search_agent",
                    "orchestration.execution_time": 2.5,
                    "orchestration.tasks_completed": 2,
                    "routing.confidence": 0.9,
                },
            }
        )

        workflow = self.evaluator._extract_workflow_execution(span_row)

        assert workflow is not None
        assert workflow.workflow_id == "workflow-456"
        assert workflow.query == "Find videos and documents about AI"
        assert workflow.success is True
        assert workflow.execution_time == 2.5
        assert workflow.task_count == 2
        assert workflow.confidence_score == 0.9
        assert len(workflow.agent_sequence) == 2
        assert "video_search_agent" in workflow.agent_sequence

    def test_extract_workflow_execution_missing_required_fields(self):
        """Test that extraction returns None when required fields missing"""
        span_row = pd.Series(
            {
                "context.span_id": "span-123",
                "attributes": {
                    # Missing workflow_id
                    "orchestration.query": "Test query",
                },
            }
        )

        workflow = self.evaluator._extract_workflow_execution(span_row)

        assert workflow is None

    def test_compute_parallel_efficiency(self):
        """Test parallel efficiency calculation"""
        attributes = {
            "orchestration.agent_times": "video_search_agent:1.2,text_search_agent:1.5"
        }

        efficiency = self.evaluator._compute_parallel_efficiency(
            pattern="parallel", attributes=attributes, total_time=2.0
        )

        # (1.2 + 1.5) / 2.0 = 1.35
        assert efficiency == pytest.approx(1.35, rel=0.01)

    def test_compute_parallel_efficiency_non_parallel_pattern(self):
        """Test that non-parallel patterns return 0 efficiency"""
        efficiency = self.evaluator._compute_parallel_efficiency(
            pattern="sequential", attributes={}, total_time=2.0
        )

        assert efficiency == 0.0

    def test_classify_query_type_multi_modal(self):
        """Test query type classification for multi-modal queries"""
        query_type = self.evaluator._classify_query_type(
            "Find videos and documents about AI", "parallel"
        )

        assert query_type == "multi_modal_search"

    def test_classify_query_type_sequential_report(self):
        """Test query type classification for sequential reports"""
        query_type = self.evaluator._classify_query_type(
            "Provide detailed analysis of trends", "sequential"
        )

        assert query_type == "sequential_report"

    def test_classify_query_type_summarization(self):
        """Test query type classification for summarization"""
        query_type = self.evaluator._classify_query_type(
            "Summarize the findings", "parallel"
        )

        assert query_type == "summarization"

    @pytest.mark.asyncio
    async def test_evaluate_orchestration_spans_empty_dataframe(self):
        """Test evaluation with no spans found"""
        with patch.object(
            self.evaluator.phoenix_client, "get_spans_dataframe"
        ) as mock_get_spans:
            mock_get_spans.return_value = pd.DataFrame()

            result = await self.evaluator.evaluate_orchestration_spans(lookback_hours=1)

            assert result["spans_processed"] == 0
            assert result["workflows_extracted"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_orchestration_spans_processes_valid_spans(self):
        """Test evaluation processes valid spans correctly"""
        spans_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.orchestration",
                    "context.span_id": "span-1",
                    "status_code": "OK",
                    "attributes": {
                        "orchestration.workflow_id": "wf-1",
                        "orchestration.query": "Test query",
                        "orchestration.pattern": "parallel",
                        "orchestration.agents_used": "agent1,agent2",
                        "orchestration.execution_order": "agent1,agent2",
                        "orchestration.execution_time": 1.5,
                        "orchestration.tasks_completed": 2,
                        "routing.confidence": 0.8,
                    },
                }
            ]
        )

        with patch.object(
            self.evaluator.phoenix_client, "get_spans_dataframe"
        ) as mock_get_spans:
            mock_get_spans.return_value = spans_df

            result = await self.evaluator.evaluate_orchestration_spans()

            assert result["spans_processed"] == 1
            assert result["workflows_extracted"] == 1
            self.mock_workflow_intelligence.record_execution.assert_called_once()


class TestOrchestrationFeedbackLoop:
    """Test OrchestrationFeedbackLoop annotation processing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_workflow_intelligence = MagicMock()
        self.mock_workflow_intelligence.record_ground_truth_execution = AsyncMock()
        self.mock_workflow_intelligence.optimize_from_ground_truth = AsyncMock()

        self.feedback_loop = OrchestrationFeedbackLoop(
            workflow_intelligence=self.mock_workflow_intelligence,
            tenant_id="test-tenant",
            poll_interval_minutes=15,
            min_annotations_for_update=2,
        )

    def test_annotation_to_ground_truth_uses_corrected_values(self):
        """Test that annotation conversion uses human-corrected values"""
        span_data = {
            "span_id": "span-123",
            "span_data": {
                "attributes": {
                    "orchestration.workflow_id": "wf-123",
                    "orchestration.query": "Test query",
                }
            },
            "annotations": [
                {
                    "result": {"label": "excellent", "score": 0.95},
                    "metadata": {
                        "pattern_is_optimal": False,
                        "suggested_pattern": "sequential",
                        "actual_pattern": "parallel",
                        "agents_are_correct": False,
                        "suggested_agents": "agent1,agent2,agent3",
                        "actual_agents": "agent1,agent2",
                        "execution_order_is_optimal": True,
                        "actual_execution_order": "agent1,agent2",
                        "execution_time": 2.5,
                        "improvement_notes": "Should use sequential for better results",
                    },
                }
            ],
        }

        ground_truth = self.feedback_loop._annotation_to_ground_truth(span_data)

        assert ground_truth is not None
        assert ground_truth.query == "Test query"
        assert ground_truth.success is True  # excellent label = success
        assert ground_truth.user_satisfaction == 0.95
        # Should use suggested values, not actual
        assert ground_truth.metadata["orchestration_pattern"] == "sequential"
        assert len(ground_truth.agent_sequence) == 3  # suggested_agents
        assert ground_truth.metadata["source"] == "human_annotation"

    def test_annotation_to_ground_truth_uses_actual_when_optimal(self):
        """Test that annotation uses actual values when marked optimal"""
        span_data = {
            "span_id": "span-456",
            "span_data": {
                "attributes": {
                    "orchestration.workflow_id": "wf-456",
                    "orchestration.query": "Test query 2",
                }
            },
            "annotations": [
                {
                    "result": {"label": "good", "score": 0.85},
                    "metadata": {
                        "pattern_is_optimal": True,
                        "actual_pattern": "parallel",
                        "agents_are_correct": True,
                        "actual_agents": "agent1,agent2",
                        "execution_order_is_optimal": True,
                        "actual_execution_order": "agent1,agent2",
                        "execution_time": 1.5,
                    },
                }
            ],
        }

        ground_truth = self.feedback_loop._annotation_to_ground_truth(span_data)

        assert ground_truth is not None
        # Should use actual values when marked optimal
        assert ground_truth.metadata["orchestration_pattern"] == "parallel"
        assert len(ground_truth.agent_sequence) == 2

    def test_classify_query_type(self):
        """Test query type classification"""
        assert (
            self.feedback_loop._classify_query_type(
                "Find videos and documents", "parallel"
            )
            == "multi_modal_search"
        )
        assert (
            self.feedback_loop._classify_query_type("Detailed analysis", "sequential")
            == "sequential_report"
        )
        assert (
            self.feedback_loop._classify_query_type("Summarize this", "parallel")
            == "summarization"
        )

    @pytest.mark.asyncio
    async def test_process_new_annotations_triggers_optimization(self):
        """Test that sufficient annotations trigger optimization"""
        annotated_spans = [
            {
                "span_id": "span-1",
                "span_data": {
                    "attributes": {
                        "orchestration.workflow_id": "wf-1",
                        "orchestration.query": "Query 1",
                    }
                },
                "annotations": [
                    {
                        "result": {"label": "good", "score": 0.8},
                        "metadata": {
                            "pattern_is_optimal": True,
                            "actual_pattern": "parallel",
                            "agents_are_correct": True,
                            "actual_agents": "agent1",
                            "execution_order_is_optimal": True,
                            "actual_execution_order": "agent1",
                            "execution_time": 1.0,
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
                        "orchestration.query": "Query 2",
                    }
                },
                "annotations": [
                    {
                        "result": {"label": "excellent", "score": 0.9},
                        "metadata": {
                            "pattern_is_optimal": True,
                            "actual_pattern": "sequential",
                            "agents_are_correct": True,
                            "actual_agents": "agent2",
                            "execution_order_is_optimal": True,
                            "actual_execution_order": "agent2",
                            "execution_time": 2.0,
                            "annotation_timestamp": "2025-01-01T00:00:01",
                        },
                    }
                ],
            },
        ]

        with patch.object(
            self.feedback_loop.annotation_storage, "query_annotated_spans"
        ) as mock_query:
            mock_query.return_value = annotated_spans

            result = await self.feedback_loop.process_new_annotations()

            assert result["annotations_found"] == 2
            assert result["workflows_learned"] == 2
            # Should trigger optimization (min_annotations_for_update = 2)
            assert result["optimizer_updated"] is True
            self.mock_workflow_intelligence.optimize_from_ground_truth.assert_called_once()


class TestUnifiedOptimizer:
    """Test UnifiedOptimizer integration logic"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_routing_optimizer = MagicMock()
        self.mock_routing_optimizer.record_routing_experience = AsyncMock()
        self.mock_routing_optimizer.optimize_routing_policy = AsyncMock(
            return_value={"status": "success"}
        )

        self.mock_workflow_intelligence = MagicMock()
        self.mock_workflow_intelligence.optimize_from_ground_truth = AsyncMock(
            return_value={"status": "success"}
        )
        self.mock_workflow_intelligence.get_successful_workflows = MagicMock(
            return_value=[]
        )

        self.unified_optimizer = UnifiedOptimizer(
            routing_optimizer=self.mock_routing_optimizer,
            workflow_intelligence=self.mock_workflow_intelligence,
        )

    def test_workflow_to_routing_experience_conversion(self):
        """Test converting WorkflowExecution to RoutingExperience"""
        workflow = WorkflowExecution(
            workflow_id="wf-123",
            query="Test multi-modal query",
            query_type="multi_modal_search",
            execution_time=2.5,
            success=True,
            agent_sequence=["video_search_agent", "text_search_agent"],
            task_count=2,
            parallel_efficiency=1.8,
            confidence_score=0.9,
            user_satisfaction=0.85,
            metadata={"orchestration_pattern": "parallel"},
        )

        routing_exp = self.unified_optimizer._workflow_to_routing_experience(workflow)

        assert routing_exp is not None
        assert routing_exp.query == "Test multi-modal query"
        assert routing_exp.chosen_agent == "video_search_agent"  # First in sequence
        assert routing_exp.search_quality == 0.85  # user_satisfaction
        assert routing_exp.agent_success is True
        assert routing_exp.processing_time == 2.5
        assert routing_exp.metadata["source"] == "orchestration_workflow"
        assert routing_exp.metadata["orchestration_pattern"] == "parallel"
        assert routing_exp.metadata["multi_agent_synergy"] is True

    def test_workflow_to_routing_experience_failed_workflow(self):
        """Test conversion of failed workflow"""
        workflow = WorkflowExecution(
            workflow_id="wf-456",
            query="Failed query",
            query_type="test",
            execution_time=1.0,
            success=False,
            agent_sequence=["agent1"],
            task_count=1,
            parallel_efficiency=0.0,
            confidence_score=0.5,
            user_satisfaction=None,
            metadata={},
        )

        routing_exp = self.unified_optimizer._workflow_to_routing_experience(workflow)

        assert routing_exp is not None
        assert routing_exp.agent_success is False
        assert routing_exp.search_quality == 0.3  # Failed = low quality

    @pytest.mark.asyncio
    async def test_integrate_orchestration_outcomes_filters_low_quality(self):
        """Test that only high-quality workflows are integrated"""
        workflows = [
            WorkflowExecution(
                workflow_id="wf-good",
                query="Good query",
                query_type="test",
                execution_time=1.0,
                success=True,
                agent_sequence=["agent1"],
                task_count=1,
                parallel_efficiency=0.0,
                confidence_score=0.9,
                user_satisfaction=0.8,
                metadata={"orchestration_pattern": "parallel"},
            ),
            WorkflowExecution(
                workflow_id="wf-bad",
                query="Bad query",
                query_type="test",
                execution_time=1.0,
                success=False,
                agent_sequence=["agent1"],
                task_count=1,
                parallel_efficiency=0.0,
                confidence_score=0.5,
                user_satisfaction=0.3,  # Too low
                metadata={},
            ),
        ]

        result = await self.unified_optimizer.integrate_orchestration_outcomes(
            workflows
        )

        assert result["workflows_processed"] == 2
        assert result["routing_experiences_created"] == 1  # Only the good one
        assert self.mock_routing_optimizer.record_routing_experience.call_count == 1

    @pytest.mark.asyncio
    async def test_optimize_unified_policy_runs_all_steps(self):
        """Test that unified optimization runs all optimization steps"""
        # Mock successful workflows
        self.mock_workflow_intelligence.get_successful_workflows.return_value = [
            WorkflowExecution(
                workflow_id="wf-1",
                query="Test",
                query_type="test",
                execution_time=1.0,
                success=True,
                agent_sequence=["agent1"],
                task_count=1,
                parallel_efficiency=0.0,
                confidence_score=0.9,
                user_satisfaction=0.8,
                metadata={"orchestration_pattern": "parallel"},
            )
        ]

        result = await self.unified_optimizer.optimize_unified_policy()

        # Should run all three optimization steps
        assert "workflow_optimization" in result
        assert "routing_optimization" in result
        assert "integration" in result

        self.mock_workflow_intelligence.optimize_from_ground_truth.assert_called_once()
        self.mock_routing_optimizer.optimize_routing_policy.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
