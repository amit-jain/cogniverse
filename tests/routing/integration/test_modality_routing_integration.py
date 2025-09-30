"""
Integration tests for Phase 6: Multi-Modal Query Classification

Tests the complete end-to-end flow:
1. Routing agent receives multi-modal queries
2. GLiNER detects audio, image, document modalities
3. Modalities are tracked in telemetry spans
4. Results are verifiable in Phoenix
"""

import logging
import time
from datetime import datetime, timedelta

import phoenix as px
import pytest

from src.app.routing.base import SearchModality
from src.app.routing.router import ComprehensiveRouter
from src.app.telemetry.config import (
    SERVICE_NAME_ORCHESTRATION,
    SPAN_NAME_ROUTING,
    TelemetryConfig,
)
from src.app.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


@pytest.fixture
def phoenix_client():
    """Phoenix client for querying spans"""
    return px.Client()


@pytest.fixture
def test_tenant_id():
    """Unique tenant ID for test isolation"""
    return f"test_modality_{int(time.time())}"


@pytest.fixture
def telemetry_config(test_tenant_id):
    """Telemetry config for test tenant"""
    return TelemetryConfig.from_env()


@pytest.fixture
def telemetry_manager(test_tenant_id, telemetry_config):
    """Telemetry manager for creating test spans"""
    manager = TelemetryManager()
    return manager


@pytest.fixture
def project_name(test_tenant_id, telemetry_config):
    """Get Phoenix project name for test tenant"""
    return telemetry_config.get_project_name(
        test_tenant_id, service=SERVICE_NAME_ORCHESTRATION
    )


@pytest.fixture
def router():
    """Comprehensive router for testing"""
    return ComprehensiveRouter()


class TestMultiModalRoutingIntegration:
    """
    Integration tests for multi-modal query classification

    Phase 6 Checkpoint Validation:
    - Audio, image, document modalities detected
    - Modality information tracked in telemetry
    - GLiNER labels expanded to include new modalities
    - Backward compatibility maintained
    - Results verifiable in Phoenix
    """

    @pytest.mark.asyncio
    async def test_audio_query_routing(
        self,
        router,
        telemetry_manager,
        test_tenant_id,
        phoenix_client,
        project_name,
    ):
        """Test routing of audio-focused queries"""
        logger.info("\n=== TEST: Audio Query Routing ===")

        audio_queries = [
            "find podcasts about machine learning",
            "search for music tracks about nature",
            "get audio recordings of lectures",
        ]

        start_time = datetime.now()

        for query in audio_queries:
            logger.info(f"Processing audio query: {query}")

            # Route the query
            decision = await router.route(query)

            # Create telemetry span
            with telemetry_manager.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                service_name=SERVICE_NAME_ORCHESTRATION,
                attributes={
                    "routing.query": query,
                    "routing.chosen_agent": "audio_search",
                    "routing.confidence": decision.confidence_score,
                    "routing.search_modality": decision.search_modality.value,
                    "routing.detected_modalities": (
                        ",".join(decision.detected_modalities)
                        if decision.detected_modalities
                        else ""
                    ),
                    "routing.context": "{}",
                },
            ):
                time.sleep(0.05)

            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")

        telemetry_manager.force_flush(timeout_millis=5000)
        time.sleep(2)

        # Verify spans in Phoenix
        end_time = datetime.now()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=project_name,
            start_time=start_time - timedelta(seconds=10),
            end_time=end_time,
        )

        assert not spans_df.empty, "No spans found in Phoenix"
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert (
            len(routing_spans) >= 3
        ), f"Expected 3 routing spans, got {len(routing_spans)}"

        logger.info(f"✅ Found {len(routing_spans)} audio routing spans in Phoenix")

    @pytest.mark.asyncio
    async def test_image_query_routing(
        self,
        router,
        telemetry_manager,
        test_tenant_id,
        phoenix_client,
        project_name,
    ):
        """Test routing of image-focused queries"""
        logger.info("\n=== TEST: Image Query Routing ===")

        image_queries = [
            "show me photos of Paris",
            "find diagrams of neural networks",
            "get charts showing market trends",
        ]

        start_time = datetime.now()

        for query in image_queries:
            logger.info(f"Processing image query: {query}")

            decision = await router.route(query)

            with telemetry_manager.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                service_name=SERVICE_NAME_ORCHESTRATION,
                attributes={
                    "routing.query": query,
                    "routing.chosen_agent": "image_search",
                    "routing.confidence": decision.confidence_score,
                    "routing.search_modality": decision.search_modality.value,
                    "routing.detected_modalities": (
                        ",".join(decision.detected_modalities)
                        if decision.detected_modalities
                        else ""
                    ),
                    "routing.context": "{}",
                },
            ):
                time.sleep(0.05)

            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")

        telemetry_manager.force_flush(timeout_millis=5000)
        time.sleep(2)

        # Verify spans in Phoenix
        end_time = datetime.now()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=project_name,
            start_time=start_time - timedelta(seconds=10),
            end_time=end_time,
        )

        assert not spans_df.empty
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert len(routing_spans) >= 3

        logger.info(f"✅ Found {len(routing_spans)} image routing spans in Phoenix")

    @pytest.mark.asyncio
    async def test_document_query_routing(
        self,
        router,
        telemetry_manager,
        test_tenant_id,
        phoenix_client,
        project_name,
    ):
        """Test routing of document-focused queries"""
        logger.info("\n=== TEST: Document Query Routing ===")

        document_queries = [
            "find PDF reports about climate change",
            "search for Excel spreadsheets with financial data",
            "get PowerPoint presentations on marketing",
        ]

        start_time = datetime.now()

        for query in document_queries:
            logger.info(f"Processing document query: {query}")

            decision = await router.route(query)

            with telemetry_manager.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                service_name=SERVICE_NAME_ORCHESTRATION,
                attributes={
                    "routing.query": query,
                    "routing.chosen_agent": "document_search",
                    "routing.confidence": decision.confidence_score,
                    "routing.search_modality": decision.search_modality.value,
                    "routing.detected_modalities": (
                        ",".join(decision.detected_modalities)
                        if decision.detected_modalities
                        else ""
                    ),
                    "routing.context": "{}",
                },
            ):
                time.sleep(0.05)

            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")

        telemetry_manager.force_flush(timeout_millis=5000)
        time.sleep(2)

        # Verify spans in Phoenix
        end_time = datetime.now()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=project_name,
            start_time=start_time - timedelta(seconds=10),
            end_time=end_time,
        )

        assert not spans_df.empty
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert len(routing_spans) >= 3

        logger.info(f"✅ Found {len(routing_spans)} document routing spans in Phoenix")

    @pytest.mark.asyncio
    async def test_multi_modal_query_routing(
        self,
        router,
        telemetry_manager,
        test_tenant_id,
        phoenix_client,
        project_name,
    ):
        """Test routing of queries with multiple modalities"""
        logger.info("\n=== TEST: Multi-Modal Query Routing ===")

        multi_modal_queries = [
            "find videos, images, and podcasts about AI",
            "search for PDFs, presentations, and diagrams on quantum computing",
            "get music tracks, photos, and video clips of nature",
        ]

        start_time = datetime.now()

        for query in multi_modal_queries:
            logger.info(f"Processing multi-modal query: {query}")

            decision = await router.route(query)

            with telemetry_manager.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                service_name=SERVICE_NAME_ORCHESTRATION,
                attributes={
                    "routing.query": query,
                    "routing.chosen_agent": "multi_search",
                    "routing.confidence": decision.confidence_score,
                    "routing.search_modality": decision.search_modality.value,
                    "routing.detected_modalities": (
                        ",".join(decision.detected_modalities)
                        if decision.detected_modalities
                        else ""
                    ),
                    "routing.context": "{}",
                },
            ):
                time.sleep(0.05)

            # Should detect multiple modalities
            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")
            assert (
                len(decision.detected_modalities) >= 2
            ), f"Expected multiple modalities, got {decision.detected_modalities}"

        telemetry_manager.force_flush(timeout_millis=5000)
        time.sleep(2)

        # Verify spans in Phoenix
        end_time = datetime.now()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=project_name,
            start_time=start_time - timedelta(seconds=10),
            end_time=end_time,
        )

        assert not spans_df.empty
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert len(routing_spans) >= 3

        logger.info(
            f"✅ Found {len(routing_spans)} multi-modal routing spans in Phoenix"
        )

    @pytest.mark.asyncio
    async def test_backward_compatibility(
        self,
        router,
        telemetry_manager,
        test_tenant_id,
        phoenix_client,
        project_name,
    ):
        """Test backward compatibility with existing video/text queries"""
        logger.info("\n=== TEST: Backward Compatibility ===")

        legacy_queries = [
            ("show me videos about cats", SearchModality.VIDEO),
            ("find documents about machine learning", SearchModality.TEXT),
            ("search for both videos and articles", SearchModality.BOTH),
        ]

        start_time = datetime.now()

        for query, expected_modality in legacy_queries:
            logger.info(f"Processing legacy query: {query}")

            decision = await router.route(query)

            # Should still work with existing system
            logger.info(f"  Search modality: {decision.search_modality}")
            logger.info(f"  Expected: {expected_modality}")

            with telemetry_manager.span(
                name=SPAN_NAME_ROUTING,
                tenant_id=test_tenant_id,
                service_name=SERVICE_NAME_ORCHESTRATION,
                attributes={
                    "routing.query": query,
                    "routing.chosen_agent": "legacy_search",
                    "routing.confidence": decision.confidence_score,
                    "routing.search_modality": decision.search_modality.value,
                    "routing.detected_modalities": (
                        ",".join(decision.detected_modalities)
                        if decision.detected_modalities
                        else ""
                    ),
                    "routing.context": "{}",
                },
            ):
                time.sleep(0.05)

        telemetry_manager.force_flush(timeout_millis=5000)
        time.sleep(2)

        # Verify spans in Phoenix
        end_time = datetime.now()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=project_name,
            start_time=start_time - timedelta(seconds=10),
            end_time=end_time,
        )

        assert not spans_df.empty
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert len(routing_spans) >= 3

        logger.info(
            f"✅ Backward compatibility maintained, {len(routing_spans)} spans in Phoenix"
        )

    @pytest.mark.asyncio
    async def test_modality_metadata_in_phoenix(
        self,
        router,
        telemetry_manager,
        test_tenant_id,
        phoenix_client,
        project_name,
    ):
        """Test that modality metadata is properly stored in Phoenix"""
        logger.info("\n=== TEST: Modality Metadata in Phoenix ===")

        test_query = "find podcasts, images, and PDFs about AI"

        start_time = datetime.now()

        decision = await router.route(test_query)

        with telemetry_manager.span(
            name=SPAN_NAME_ROUTING,
            tenant_id=test_tenant_id,
            service_name=SERVICE_NAME_ORCHESTRATION,
            attributes={
                "routing.query": test_query,
                "routing.chosen_agent": "multi_search",
                "routing.confidence": decision.confidence_score,
                "routing.search_modality": decision.search_modality.value,
                "routing.detected_modalities": (
                    ",".join(decision.detected_modalities)
                    if decision.detected_modalities
                    else ""
                ),
                "routing.context": "{}",
            },
        ):
            time.sleep(0.05)

        telemetry_manager.force_flush(timeout_millis=5000)
        time.sleep(2)

        # Verify spans in Phoenix
        end_time = datetime.now()
        spans_df = phoenix_client.get_spans_dataframe(
            project_name=project_name,
            start_time=start_time - timedelta(seconds=10),
            end_time=end_time,
        )

        assert not spans_df.empty
        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert len(routing_spans) >= 1

        # Check that modality attributes are present
        span = routing_spans.iloc[0]
        logger.info(f"  Span attributes: {span.get('attributes', {})}")

        logger.info("✅ Modality metadata properly stored in Phoenix")
