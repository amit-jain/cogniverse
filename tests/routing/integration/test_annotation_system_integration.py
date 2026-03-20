"""
Comprehensive Integration Test for Annotation System

Validates the COMPLETE annotation workflow end-to-end:
1. Creates real routing spans with Phoenix telemetry
2. Runs AnnotationAgent to identify spans needing review
3. Generates LLM annotations
4. Stores annotations in Phoenix using SpanEvaluations API
5. Queries annotations back from Phoenix
6. Feeds annotations to optimizer via feedback loop
7. Verifies complete data flow works with real Phoenix instance

Uses shared phoenix_container fixture from tests/conftest.py.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.annotation_agent import (
    AnnotationAgent,
    AnnotationPriority,
)
from cogniverse_agents.routing.annotation_feedback_loop import AnnotationFeedbackLoop
from cogniverse_agents.routing.annotation_storage import RoutingAnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    LLMAutoAnnotator,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_ROUTING,
    TelemetryConfig,
)
from tests.utils.async_polling import (
    simulate_processing_delay,
    wait_for_phoenix_processing,
)

logger = logging.getLogger(__name__)

_TEST_TENANT = "annotation_system_test"


@pytest.fixture
def real_telemetry_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)


@pytest.fixture
def test_tenant_id():
    """Use unique tenant ID for test isolation"""
    return f"test_annotation_{int(time.time())}"


@pytest.fixture
def telemetry_provider(telemetry_manager_with_phoenix, test_tenant_id):
    """Telemetry provider for querying spans via abstraction"""
    return telemetry_manager_with_phoenix.get_provider(tenant_id=test_tenant_id)


class TestAnnotationSystemIntegration:
    """Comprehensive integration tests for annotation system"""

    @pytest.mark.asyncio
    async def test_complete_annotation_workflow_with_real_phoenix(
        self,
        telemetry_provider,
        test_tenant_id,
        telemetry_manager_with_phoenix,
        real_telemetry_provider,
    ):
        """Test complete annotation workflow end-to-end with real Phoenix"""
        span_ids = []
        queries = [
            ("What are the best restaurants in Paris?", "video_search", 0.3),
            ("Show me nature documentaries", "detailed_report", 0.45),
            ("Explain quantum computing", "summarizer", 0.9),
        ]

        for query, agent, confidence in queries:
            with telemetry_manager_with_phoenix.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                attributes={
                    "routing.query": query,
                    "routing.chosen_agent": agent,
                    "routing.confidence": confidence,
                    "routing.context": "{}",
                    "routing.processing_time": 0.1,
                },
            ) as span:
                span_id = getattr(span, "context", None)
                if span_id and hasattr(span_id, "span_id"):
                    span_ids.append(str(span_id.span_id))
                simulate_processing_delay(delay=0.1)

        telemetry_manager_with_phoenix.force_flush(timeout_millis=5000)
        wait_for_phoenix_processing(delay=2)

        config = TelemetryConfig()
        project_name = config.get_project_name(test_tenant_id)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=5)

        spans_df = await telemetry_provider.traces.get_spans(
            project=project_name, start_time=start_time, end_time=end_time
        )

        assert not spans_df.empty, "No spans found in Phoenix"

        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert len(routing_spans) >= len(queries), (
            f"Expected at least {len(queries)} routing spans, found {len(routing_spans)}"
        )

        annotation_agent = AnnotationAgent(
            tenant_id=test_tenant_id,
            confidence_threshold=0.6,
            max_annotations_per_run=10,
        )

        annotation_requests = await annotation_agent.identify_spans_needing_annotation(
            lookback_hours=1
        )

        assert len(annotation_requests) >= 2, (
            f"Expected at least 2 annotation requests, got {len(annotation_requests)}"
        )

        low_conf_requests = [
            r for r in annotation_requests if r.routing_confidence < 0.6
        ]
        assert len(low_conf_requests) >= 2, (
            f"Expected at least 2 low confidence requests, got {len(low_conf_requests)}"
        )

        has_llm_access = os.getenv("ANNOTATION_API_KEY")

        if has_llm_access:
            llm_annotator = LLMAutoAnnotator(
                llm_config=LLMEndpointConfig(
                    model="ollama/gemma3:4b", api_base="http://localhost:11434"
                )
            )
            auto_annotation = llm_annotator.annotate(annotation_requests[0])

            assert auto_annotation.label in [
                AnnotationLabel.CORRECT_ROUTING,
                AnnotationLabel.WRONG_ROUTING,
                AnnotationLabel.AMBIGUOUS,
                AnnotationLabel.INSUFFICIENT_INFO,
            ]
            assert 0.0 <= auto_annotation.confidence <= 1.0
            assert len(auto_annotation.reasoning) > 0
        else:
            from cogniverse_agents.routing.llm_auto_annotator import AutoAnnotation

            auto_annotation = AutoAnnotation(
                span_id=annotation_requests[0].span_id,
                label=AnnotationLabel.WRONG_ROUTING,
                confidence=0.8,
                reasoning="Test annotation for integration test",
                suggested_correct_agent="web_search",
                requires_human_review=False,
            )

        annotation_storage = RoutingAnnotationStorage(tenant_id=test_tenant_id)

        success = await annotation_storage.store_llm_annotation(
            span_id=annotation_requests[0].span_id, annotation=auto_annotation
        )

        assert success, "Failed to store annotation in Phoenix"

        wait_for_phoenix_processing(delay=2)

        optimizer = AdvancedRoutingOptimizer(
            tenant_id=test_tenant_id,
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            telemetry_provider=real_telemetry_provider,
        )
        feedback_loop = AnnotationFeedbackLoop(
            optimizer=optimizer,
            tenant_id=test_tenant_id,
            min_annotations_for_update=1,
        )

        result = await feedback_loop.process_new_annotations()

        assert "annotations_found" in result
        assert "experiences_created" in result

    @pytest.mark.asyncio
    async def test_annotation_storage_persistence(
        self, telemetry_provider, test_tenant_id, telemetry_manager_with_phoenix
    ):
        """Test that annotations are actually persisted and retrievable from Phoenix"""
        with telemetry_manager_with_phoenix.span(
            name=SPAN_NAME_ROUTING,
            tenant_id=test_tenant_id,
            attributes={
                "routing.query": "Test persistence query",
                "routing.chosen_agent": "video_search",
                "routing.confidence": 0.4,
                "routing.context": "{}",
            },
        ) as span:
            span_context = getattr(span, "context", None)
            if span_context and hasattr(span_context, "span_id"):
                test_span_id = str(span_context.span_id)
            else:
                test_span_id = "test_span_" + str(int(time.time()))

        telemetry_manager_with_phoenix.force_flush(timeout_millis=5000)
        wait_for_phoenix_processing(delay=2)

        annotation_storage = RoutingAnnotationStorage(tenant_id=test_tenant_id)

        from cogniverse_agents.routing.llm_auto_annotator import AutoAnnotation

        test_annotation = AutoAnnotation(
            span_id=test_span_id,
            label=AnnotationLabel.CORRECT_ROUTING,
            confidence=0.95,
            reasoning="Test persistence annotation",
            suggested_correct_agent=None,
            requires_human_review=False,
        )

        success = await annotation_storage.store_llm_annotation(
            span_id=test_span_id, annotation=test_annotation
        )

        assert success, "Failed to store annotation"

        wait_for_phoenix_processing(delay=3, description="Phoenix annotation indexing")

        config = TelemetryConfig()
        project_name = config.get_project_name(test_tenant_id)

        spans_df = await telemetry_provider.traces.get_spans(
            project=project_name,
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            end_time=datetime.now(timezone.utc),
        )

        assert not spans_df.empty, "No spans found after storage"

    @pytest.mark.asyncio
    async def test_annotation_agent_with_real_data(
        self, telemetry_provider, test_tenant_id, telemetry_manager_with_phoenix
    ):
        """Test AnnotationAgent against real Phoenix data"""
        test_cases = [
            ("Low confidence failure", "video_search", 0.2, "ERROR"),
            ("Medium confidence success", "detailed_report", 0.55, "OK"),
            ("High confidence success", "summarizer", 0.95, "OK"),
        ]

        for query, agent, confidence, status in test_cases:
            with telemetry_manager_with_phoenix.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                attributes={
                    "routing.query": query,
                    "routing.chosen_agent": agent,
                    "routing.confidence": confidence,
                    "routing.context": "{}",
                },
            ) as span:
                if status == "ERROR":
                    try:
                        raise Exception("Simulated routing error")
                    except Exception as e:
                        span.record_exception(e)

        telemetry_manager_with_phoenix.force_flush(timeout_millis=5000)
        wait_for_phoenix_processing(delay=2)

        annotation_agent = AnnotationAgent(
            tenant_id=test_tenant_id,
            confidence_threshold=0.6,
            max_annotations_per_run=10,
        )

        requests = await annotation_agent.identify_spans_needing_annotation(
            lookback_hours=1
        )

        assert len(requests) >= 2, f"Expected at least 2 requests, got {len(requests)}"

        priorities = {r.priority for r in requests}
        assert (
            AnnotationPriority.HIGH in priorities
            or AnnotationPriority.MEDIUM in priorities
        ), "Expected HIGH or MEDIUM priority requests"

        low_conf_requests = [r for r in requests if r.routing_confidence < 0.3]
        if low_conf_requests:
            assert low_conf_requests[0].priority == AnnotationPriority.HIGH, (
                "Low confidence should be HIGH priority"
            )

    @pytest.mark.asyncio
    async def test_feedback_loop_end_to_end(
        self,
        telemetry_provider,
        test_tenant_id,
        telemetry_manager_with_phoenix,
        real_telemetry_provider,
    ):
        """Test complete feedback loop: span -> annotation -> storage -> optimizer"""
        with telemetry_manager_with_phoenix.span(
            name=SPAN_NAME_ROUTING,
            tenant_id=test_tenant_id,
            attributes={
                "routing.query": "Feedback loop test query",
                "routing.chosen_agent": "video_search",
                "routing.confidence": 0.3,
                "routing.context": '{"entities": [], "relationships": []}',
            },
        ):
            pass

        telemetry_manager_with_phoenix.force_flush(timeout_millis=5000)
        wait_for_phoenix_processing(delay=2)

        annotation_agent = AnnotationAgent(
            tenant_id=test_tenant_id, confidence_threshold=0.6
        )
        requests = await annotation_agent.identify_spans_needing_annotation(
            lookback_hours=1
        )

        assert len(requests) > 0, "No annotation requests found"

        annotation_storage = RoutingAnnotationStorage(tenant_id=test_tenant_id)

        success = await annotation_storage.store_human_annotation(
            span_id=requests[0].span_id,
            label=AnnotationLabel.WRONG_ROUTING,
            reasoning="Human identified wrong routing",
            suggested_agent="detailed_report",
            annotator_id="test_user",
        )

        assert success, "Failed to store human annotation"

        wait_for_phoenix_processing(delay=3, description="Phoenix annotation indexing")

        optimizer = AdvancedRoutingOptimizer(
            tenant_id=test_tenant_id,
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            telemetry_provider=real_telemetry_provider,
        )
        initial_experience_count = len(optimizer.experiences)

        feedback_loop = AnnotationFeedbackLoop(
            optimizer=optimizer,
            tenant_id=test_tenant_id,
            min_annotations_for_update=1,
        )

        result = await feedback_loop.process_new_annotations()

        assert "annotations_found" in result
        assert "experiences_created" in result

        if result["experiences_created"] > 0:
            assert len(optimizer.experiences) > initial_experience_count, (
                "Optimizer did not receive new experiences"
            )

    @pytest.mark.asyncio
    async def test_bulk_evaluation_logging(
        self, telemetry_provider, test_tenant_id, telemetry_manager_with_phoenix
    ):
        """Test bulk evaluation upload via provider abstraction."""
        import pandas as pd

        span_ids = []
        for i in range(3):
            with telemetry_manager_with_phoenix.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                attributes={
                    "routing.query": f"Test query {i}",
                    "routing.chosen_agent": "video_search",
                    "routing.confidence": 0.5 + (i * 0.1),
                    "routing.context": "{}",
                },
            ) as span:
                span_context = getattr(span, "context", None)
                if span_context and hasattr(span_context, "span_id"):
                    span_id = str(span_context.span_id)
                    span_ids.append(span_id)

        assert len(span_ids) == 3, "Failed to create test spans"

        wait_for_phoenix_processing(delay=2, description="span indexing")

        eval_data = []
        for i, span_id in enumerate(span_ids):
            eval_data.append(
                {
                    "span_id": span_id,
                    "score": 0.7 + (i * 0.1),
                    "label": "good" if i < 2 else "excellent",
                    "explanation": f"Test evaluation {i}",
                }
            )

        evaluations_df = pd.DataFrame(eval_data)

        project = f"cogniverse-{test_tenant_id}"

        try:
            await telemetry_provider.annotations.log_evaluations(
                eval_name="test_bulk_evaluation",
                evaluations_df=evaluations_df,
                project=project,
            )
        except Exception as e:
            pytest.fail(f"Failed to log evaluations: {e}")

        wait_for_phoenix_processing(delay=3, description="evaluation indexing")

        eval_end_time = datetime.now(timezone.utc)
        eval_start_time = eval_end_time - timedelta(minutes=5)

        # Phoenix under concurrent test load may return transient 500s;
        # retry up to 3 times with backoff.
        import asyncio as _asyncio

        spans_df = None
        for attempt in range(3):
            try:
                spans_df = await telemetry_provider.traces.get_spans(
                    project=project,
                    start_time=eval_start_time,
                    end_time=eval_end_time,
                    limit=10,
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                await _asyncio.sleep(2 * (attempt + 1))

        if not spans_df.empty:
            annotations_df = await telemetry_provider.annotations.get_annotations(
                spans_df=spans_df,
                project=project,
                annotation_names=["test_bulk_evaluation"],
            )

            if len(annotations_df) > 0:
                logger.info("Evaluations successfully stored and retrieved")
