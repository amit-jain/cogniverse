"""
Integration tests for annotation workflow

Tests the complete flow:
1. AnnotationAgent identifies spans needing review
2. LLMAutoAnnotator generates initial annotations
3. AnnotationStorage stores annotations
4. AnnotationFeedbackLoop feeds to optimizer
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.app.routing.advanced_optimizer import AdvancedRoutingOptimizer
from src.app.routing.annotation_agent import AnnotationAgent, AnnotationPriority
from src.app.routing.annotation_feedback_loop import AnnotationFeedbackLoop
from src.app.routing.annotation_storage import AnnotationStorage
from src.app.routing.llm_auto_annotator import AnnotationLabel, LLMAutoAnnotator


@pytest.fixture
def mock_phoenix_client():
    """Mock Phoenix client for testing"""
    client = Mock()

    # Create mock span data
    spans_data = {
        "name": ["cogniverse.routing", "cogniverse.routing"],
        "context.span_id": ["span_001", "span_002"],
        "start_time": [
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(hours=2)
        ],
        "status": ["ERROR", "OK"],
        "attributes.routing": [
            {
                "query": "What are the best restaurants in Paris?",
                "chosen_agent": "video_search",
                "confidence": 0.4,
                "context": {}
            },
            {
                "query": "Show me nature documentaries",
                "chosen_agent": "detailed_report",
                "confidence": 0.55,
                "context": {}
            }
        ]
    }

    client.get_spans_dataframe.return_value = pd.DataFrame(spans_data)
    return client


@pytest.fixture
def mock_routing_evaluator():
    """Mock RoutingEvaluator for testing"""
    from src.evaluation.evaluators.routing_evaluator import RoutingOutcome

    evaluator = Mock()
    evaluator._classify_routing_outcome = Mock(
        return_value=(RoutingOutcome.FAILURE, {"reason": "downstream_error"})
    )
    return evaluator


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM completion response for testing"""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = '{"label": "wrong_routing", "confidence": 0.8, "reasoning": "Query is about restaurants, not videos", "suggested_correct_agent": "web_search", "requires_human_review": false}'
    return response


class TestAnnotationAgent:
    """Test AnnotationAgent span identification"""

    @patch("src.app.routing.annotation_agent.px.Client")
    @patch("src.app.routing.annotation_agent.RoutingEvaluator")
    def test_identify_spans_needing_annotation(
        self, mock_evaluator_class, mock_client_class, mock_phoenix_client, mock_routing_evaluator
    ):
        """Test that AnnotationAgent identifies low-quality spans"""
        # Setup mocks
        mock_client_class.return_value = mock_phoenix_client
        mock_evaluator_class.return_value = mock_routing_evaluator

        # Initialize agent
        agent = AnnotationAgent(
            tenant_id="test",
            confidence_threshold=0.6,
            max_annotations_per_run=10
        )

        # Identify spans
        requests = agent.identify_spans_needing_annotation(lookback_hours=24)

        # Assertions
        assert len(requests) > 0, "Should find spans needing annotation"

        # Check that low confidence span was identified
        low_conf_request = next(
            (r for r in requests if r.routing_confidence < 0.6), None
        )
        assert low_conf_request is not None, "Should identify low confidence span"
        assert low_conf_request.priority in [
            AnnotationPriority.HIGH,
            AnnotationPriority.MEDIUM
        ], "Low confidence should be high/medium priority"

    @patch("src.app.routing.annotation_agent.px.Client")
    @patch("src.app.routing.annotation_agent.RoutingEvaluator")
    def test_prioritization(
        self, mock_evaluator_class, mock_client_class, mock_phoenix_client, mock_routing_evaluator
    ):
        """Test that requests are properly prioritized"""
        from src.evaluation.evaluators.routing_evaluator import RoutingOutcome

        # Setup mocks with different outcomes
        mock_client_class.return_value = mock_phoenix_client

        def classify_outcome(span_row):
            conf = span_row.get("attributes.routing", {}).get("confidence", 1.0)
            if conf < 0.5:
                return (RoutingOutcome.FAILURE, {"reason": "low_confidence"})
            return (RoutingOutcome.AMBIGUOUS, {"reason": "unclear"})

        mock_routing_evaluator._classify_routing_outcome = classify_outcome
        mock_evaluator_class.return_value = mock_routing_evaluator

        # Initialize agent
        agent = AnnotationAgent(tenant_id="test", confidence_threshold=0.6)

        # Identify spans
        requests = agent.identify_spans_needing_annotation(lookback_hours=24)

        # Check prioritization
        if len(requests) > 1:
            # High priority should come before medium/low
            priorities = [r.priority for r in requests]
            high_indices = [
                i for i, p in enumerate(priorities) if p == AnnotationPriority.HIGH
            ]
            medium_indices = [
                i for i, p in enumerate(priorities) if p == AnnotationPriority.MEDIUM
            ]

            if high_indices and medium_indices:
                assert max(high_indices) < min(
                    medium_indices
                ), "HIGH priority should come before MEDIUM"


class TestLLMAutoAnnotator:
    """Test LLM-based auto-annotation"""

    @patch("src.app.routing.llm_auto_annotator.completion")
    def test_annotate(self, mock_completion, mock_litellm_response):
        """Test LLM annotation generation"""
        from src.evaluation.evaluators.routing_evaluator import RoutingOutcome
        from src.app.routing.annotation_agent import AnnotationRequest

        # Setup mock
        mock_completion.return_value = mock_litellm_response

        # Initialize annotator
        annotator = LLMAutoAnnotator()

        # Create annotation request
        request = AnnotationRequest(
            span_id="span_001",
            timestamp=datetime.now(),
            query="What are the best restaurants in Paris?",
            chosen_agent="video_search",
            routing_confidence=0.4,
            outcome=RoutingOutcome.FAILURE,
            priority=AnnotationPriority.HIGH,
            reason="Failure with low confidence",
            context={}
        )

        # Generate annotation
        annotation = annotator.annotate(request)

        # Assertions
        assert annotation.span_id == "span_001"
        assert annotation.label == AnnotationLabel.WRONG_ROUTING
        assert annotation.confidence > 0.0
        assert len(annotation.reasoning) > 0
        assert annotation.suggested_correct_agent == "web_search"

    @patch("src.app.routing.llm_auto_annotator.completion")
    def test_batch_annotate(self, mock_completion, mock_litellm_response):
        """Test batch annotation"""
        from src.evaluation.evaluators.routing_evaluator import RoutingOutcome
        from src.app.routing.annotation_agent import AnnotationRequest

        # Setup mock
        mock_completion.return_value = mock_litellm_response

        # Initialize annotator
        annotator = LLMAutoAnnotator()

        # Create multiple requests
        requests = [
            AnnotationRequest(
                span_id=f"span_{i:03d}",
                timestamp=datetime.now(),
                query=f"Test query {i}",
                chosen_agent="video_search",
                routing_confidence=0.4,
                outcome=RoutingOutcome.FAILURE,
                priority=AnnotationPriority.HIGH,
                reason="Test",
                context={}
            )
            for i in range(3)
        ]

        # Generate annotations
        annotations = annotator.batch_annotate(requests)

        # Assertions
        assert len(annotations) == 3
        assert all(ann.span_id.startswith("span_") for ann in annotations)


class TestAnnotationStorage:
    """Test annotation storage in Phoenix"""

    @patch("src.app.routing.annotation_storage.px.Client")
    def test_store_llm_annotation(self, mock_client_class):
        """Test storing LLM annotation"""
        from src.app.routing.llm_auto_annotator import AutoAnnotation

        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Initialize storage
        storage = AnnotationStorage(tenant_id="test")

        # Create annotation
        annotation = AutoAnnotation(
            span_id="span_001",
            label=AnnotationLabel.WRONG_ROUTING,
            confidence=0.8,
            reasoning="Test reasoning",
            suggested_correct_agent="web_search",
            requires_human_review=False
        )

        # Store annotation
        success = storage.store_llm_annotation("span_001", annotation)

        # Assertions
        assert success is True

    @patch("src.app.routing.annotation_storage.px.Client")
    def test_store_human_annotation(self, mock_client_class):
        """Test storing human annotation"""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Initialize storage
        storage = AnnotationStorage(tenant_id="test")

        # Store annotation
        success = storage.store_human_annotation(
            span_id="span_001",
            label=AnnotationLabel.CORRECT_ROUTING,
            reasoning="Human verified this is correct",
            suggested_agent=None,
            annotator_id="user123"
        )

        # Assertions
        assert success is True


class TestAnnotationFeedbackLoop:
    """Test feedback loop to optimizer"""

    @pytest.mark.asyncio
    @patch("src.app.routing.annotation_feedback_loop.AnnotationStorage")
    async def test_process_new_annotations(self, mock_storage_class):
        """Test processing annotations and feeding to optimizer"""
        # Setup mock storage
        mock_storage = Mock()
        mock_storage.query_annotated_spans.return_value = [
            {
                "span_id": "span_001",
                "query": "Test query",
                "chosen_agent": "video_search",
                "routing_confidence": 0.4,
                "annotation_label": "wrong_routing",
                "annotation_confidence": 0.8,
                "annotation_reasoning": "Wrong agent chosen",
                "annotation_timestamp": datetime.now().isoformat(),
                "suggested_agent": "web_search",
                "context": {}
            }
        ]
        mock_storage_class.return_value = mock_storage

        # Initialize optimizer
        optimizer = AdvancedRoutingOptimizer()

        # Initialize feedback loop
        feedback_loop = AnnotationFeedbackLoop(
            optimizer=optimizer,
            tenant_id="test",
            poll_interval_minutes=15,
            min_annotations_for_update=1
        )

        # Process annotations
        result = await feedback_loop.process_new_annotations()

        # Assertions
        assert result["annotations_found"] == 1
        assert result["experiences_created"] == 1

    @pytest.mark.asyncio
    @patch("src.app.routing.annotation_feedback_loop.AnnotationStorage")
    async def test_annotation_to_experience_conversion(self, mock_storage_class):
        """Test that annotations are correctly converted to experiences"""
        # Setup mock storage
        mock_storage = Mock()
        mock_storage.query_annotated_spans.return_value = [
            {
                "span_id": "span_001",
                "query": "Test query",
                "chosen_agent": "video_search",
                "routing_confidence": 0.7,
                "annotation_label": "correct_routing",
                "annotation_confidence": 1.0,
                "annotation_reasoning": "Correct choice",
                "annotation_timestamp": datetime.now().isoformat(),
                "suggested_agent": None,
                "context": {}
            }
        ]
        mock_storage_class.return_value = mock_storage

        # Initialize optimizer
        optimizer = AdvancedRoutingOptimizer()

        # Initialize feedback loop
        feedback_loop = AnnotationFeedbackLoop(
            optimizer=optimizer,
            tenant_id="test"
        )

        # Process annotations
        result = await feedback_loop.process_new_annotations()

        # Assertions
        assert result["experiences_created"] == 1

        # Check that optimizer received experience
        # (In a real test, we'd mock the optimizer's record_routing_experience method)


class TestEndToEndAnnotationWorkflow:
    """Integration test for complete annotation workflow"""

    @pytest.mark.asyncio
    @patch("src.app.routing.annotation_agent.px.Client")
    @patch("src.app.routing.annotation_agent.RoutingEvaluator")
    @patch("src.app.routing.llm_auto_annotator.completion")
    @patch("src.app.routing.annotation_storage.px.Client")
    @patch("src.app.routing.annotation_feedback_loop.AnnotationStorage")
    async def test_complete_workflow(
        self,
        mock_feedback_storage_class,
        mock_storage_client_class,
        mock_completion,
        mock_evaluator_class,
        mock_agent_client_class,
        mock_phoenix_client,
        mock_routing_evaluator,
        mock_litellm_response
    ):
        """Test complete annotation workflow from identification to optimizer"""
        # Setup all mocks
        mock_agent_client_class.return_value = mock_phoenix_client
        mock_evaluator_class.return_value = mock_routing_evaluator
        mock_completion.return_value = mock_litellm_response

        mock_storage_client = Mock()
        mock_storage_client_class.return_value = mock_storage_client

        # Step 1: Identify spans needing annotation
        agent = AnnotationAgent(tenant_id="test", confidence_threshold=0.6)
        requests = agent.identify_spans_needing_annotation(lookback_hours=24)

        assert len(requests) > 0, "Should identify spans needing annotation"

        # Step 2: Generate LLM annotations
        annotator = LLMAutoAnnotator()
        annotations = annotator.batch_annotate(requests[:1])

        assert len(annotations) > 0, "Should generate LLM annotations"

        # Step 3: Store annotations
        storage = AnnotationStorage(tenant_id="test")
        for annotation in annotations:
            success = storage.store_llm_annotation(requests[0].span_id, annotation)
            assert success is True, "Should store annotations"

        # Step 4: Feed to optimizer via feedback loop
        mock_feedback_storage = Mock()
        mock_feedback_storage.query_annotated_spans.return_value = [
            {
                "span_id": requests[0].span_id,
                "query": requests[0].query,
                "chosen_agent": requests[0].chosen_agent,
                "routing_confidence": requests[0].routing_confidence,
                "annotation_label": annotations[0].label.value,
                "annotation_confidence": annotations[0].confidence,
                "annotation_reasoning": annotations[0].reasoning,
                "annotation_timestamp": datetime.now().isoformat(),
                "suggested_agent": annotations[0].suggested_correct_agent,
                "context": {}
            }
        ]
        mock_feedback_storage_class.return_value = mock_feedback_storage

        optimizer = AdvancedRoutingOptimizer()
        feedback_loop = AnnotationFeedbackLoop(
            optimizer=optimizer,
            tenant_id="test",
            min_annotations_for_update=1
        )

        result = await feedback_loop.process_new_annotations()

        # Assertions
        assert result["annotations_found"] == 1, "Should find stored annotation"
        assert result["experiences_created"] == 1, "Should create experience from annotation"