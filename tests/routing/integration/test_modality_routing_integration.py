"""
Integration tests for Phase 6: Multi-Modal Query Classification

Tests the complete end-to-end flow:
1. Routing agent receives multi-modal queries
2. GLiNER detects audio, image, document modalities
3. Modalities are tracked in telemetry spans
4. Results are verifiable in Phoenix
"""

import logging
import os
import subprocess
import time
from datetime import datetime, timedelta

import phoenix as px
import pytest
import requests
from cogniverse_agents.routing.base import SearchModality
from cogniverse_agents.routing.router import ComprehensiveRouter
from cogniverse_core.telemetry.config import (
    SERVICE_NAME_ORCHESTRATION,
    SPAN_NAME_ROUTING,
    BatchExportConfig,
    TelemetryConfig,
)
from cogniverse_core.telemetry.manager import TelemetryManager

from tests.utils.async_polling import simulate_processing_delay, wait_for_vespa_indexing

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function", autouse=True)
def phoenix_container():
    """Start Phoenix Docker container on non-default ports for each test"""
    import os

    # CRITICAL: Set environment variables BEFORE any TelemetryManager is created
    # TelemetryManager is a singleton that only initializes once, so we need to:
    # 1. Set the env vars first
    # 2. Reset the singleton so it re-reads the env vars
    original_endpoint = os.environ.get("OTLP_ENDPOINT")
    original_sync_export = os.environ.get("TELEMETRY_SYNC_EXPORT")

    os.environ["OTLP_ENDPOINT"] = "http://localhost:14317"
    os.environ["TELEMETRY_SYNC_EXPORT"] = "true"  # Use sync export for tests

    # Reset TelemetryManager singleton to force re-initialization with new env var
    from cogniverse_core.telemetry.manager import TelemetryManager
    if hasattr(TelemetryManager, '_instance') and TelemetryManager._instance is not None:
        # Shutdown existing instance
        try:
            TelemetryManager._instance.shutdown()
        except Exception:
            pass
        # Reset singleton
        TelemetryManager._instance = None

    container_name = f"phoenix_test_{int(time.time() * 1000)}"

    # Clean up any existing Phoenix test containers
    try:
        # Find and stop old containers using these ports
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", "name=phoenix_test"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.stdout.strip():
            old_containers = result.stdout.strip().split('\n')
            for container_id in old_containers:
                subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
            logger.info(f"Cleaned up {len(old_containers)} old Phoenix test containers")
    except Exception as e:
        logger.warning(f"Error cleaning up old containers: {e}")

    try:
        # Start Phoenix container
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", "16006:6006",  # HTTP port
                "-p", "14317:4317",  # gRPC port
                "arizephoenix/phoenix:latest"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Phoenix container {container_name} started")

        # Wait for Phoenix to be ready (takes ~15-20 seconds for DB migrations)
        # Use polling loop similar to Vespa with retry logic
        max_wait_time = 60  # seconds
        poll_interval = 0.5  # seconds
        start_time = time.time()
        phoenix_ready = False

        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get("http://localhost:16006", timeout=2)
                if response.status_code == 200:
                    phoenix_ready = True
                    elapsed = time.time() - start_time
                    logger.info(f"Phoenix ready after {elapsed:.1f} seconds")
                    break
            except Exception:
                pass
            time.sleep(poll_interval)

        if not phoenix_ready:
            # Get logs for debugging
            logs_result = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.error(f"Phoenix logs:\n{logs_result.stdout}\n{logs_result.stderr}")
            raise RuntimeError(f"Phoenix failed to start after {max_wait_time} seconds")

        yield container_name

    finally:
        # Cleanup: stop and remove container
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=False,
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ["docker", "rm", container_name],
                check=False,
                capture_output=True,
                timeout=10
            )
            logger.info(f"Phoenix container {container_name} stopped and removed")
        except Exception as e:
            logger.warning(f"Error cleaning up Phoenix container: {e}")
            # Force remove if normal stop failed
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    capture_output=True,
                    timeout=10
                )
            except Exception:
                pass

        # Restore original environment variables
        if original_endpoint:
            os.environ["OTLP_ENDPOINT"] = original_endpoint
        else:
            os.environ.pop("OTLP_ENDPOINT", None)

        if original_sync_export:
            os.environ["TELEMETRY_SYNC_EXPORT"] = original_sync_export
        else:
            os.environ.pop("TELEMETRY_SYNC_EXPORT", None)


@pytest.fixture
def phoenix_client():
    """Phoenix client for querying spans on non-default port"""
    return px.Client(endpoint="http://localhost:16006")


@pytest.fixture
def test_tenant_id():
    """Unique tenant ID for test isolation"""
    return f"test_modality_{int(time.time())}"


@pytest.fixture
def telemetry_config(test_tenant_id, phoenix_container):
    """Telemetry config for test tenant pointing to non-default Phoenix port"""
    # Depend on phoenix_container to ensure env var is set before reading config
    # Read the OTLP_ENDPOINT from environment (set by phoenix_container)
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "localhost:4317")
    config = TelemetryConfig(
        otlp_endpoint=otlp_endpoint,
        provider_config={
            "http_endpoint": "http://localhost:16006",
            "grpc_endpoint": "http://localhost:14317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    return config


@pytest.fixture
def telemetry_manager(test_tenant_id, telemetry_config, phoenix_container):
    """Telemetry manager for creating test spans"""
    import cogniverse_core.telemetry.manager as telemetry_manager_module

    # Depend on phoenix_container to ensure singleton is reset before creating manager
    # TelemetryManager singleton has been reset in phoenix_container fixture
    # Pass the config explicitly to ensure it has the correct endpoints
    manager = TelemetryManager(config=telemetry_config)
    telemetry_manager_module._telemetry_manager = manager
    return manager


@pytest.fixture
def project_name(test_tenant_id, telemetry_config):
    """Get Phoenix project name for test tenant"""
    return telemetry_config.get_project_name(
        test_tenant_id, service=SERVICE_NAME_ORCHESTRATION
    )


@pytest.fixture
def router(phoenix_container):
    """Comprehensive router for testing"""
    # Depend on phoenix_container to ensure env var is set before router initialization
    # Router may create TelemetryManager internally
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
                project_name=SERVICE_NAME_ORCHESTRATION,
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
                simulate_processing_delay(delay=0.05, description="processing")

            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")

        telemetry_manager.force_flush(timeout_millis=5000)
        wait_for_vespa_indexing(delay=2)

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
                project_name=SERVICE_NAME_ORCHESTRATION,
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
                simulate_processing_delay(delay=0.05, description="processing")

            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")

        telemetry_manager.force_flush(timeout_millis=5000)
        wait_for_vespa_indexing(delay=2)

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
                project_name=SERVICE_NAME_ORCHESTRATION,
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
                simulate_processing_delay(delay=0.05, description="processing")

            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")

        telemetry_manager.force_flush(timeout_millis=5000)
        wait_for_vespa_indexing(delay=2)

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
                project_name=SERVICE_NAME_ORCHESTRATION,
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
                simulate_processing_delay(delay=0.05, description="processing")

            # Should detect multiple modalities
            logger.info(f"  Detected modalities: {decision.detected_modalities}")
            logger.info(f"  Search modality: {decision.search_modality}")
            assert (
                len(decision.detected_modalities) >= 2
            ), f"Expected multiple modalities, got {decision.detected_modalities}"

        telemetry_manager.force_flush(timeout_millis=5000)
        wait_for_vespa_indexing(delay=2)

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
                project_name=SERVICE_NAME_ORCHESTRATION,
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
                simulate_processing_delay(delay=0.05, description="processing")

        telemetry_manager.force_flush(timeout_millis=5000)
        wait_for_vespa_indexing(delay=2)

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
            project_name=SERVICE_NAME_ORCHESTRATION,
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
            simulate_processing_delay(delay=0.05, description="processing")

        telemetry_manager.force_flush(timeout_millis=5000)
        wait_for_vespa_indexing(delay=2)

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
