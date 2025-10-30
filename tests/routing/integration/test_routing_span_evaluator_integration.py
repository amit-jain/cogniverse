"""
Integration tests for RoutingSpanEvaluator

Tests the complete flow of:
1. Generating real routing spans via RoutingAgent
2. Querying spans from Phoenix
3. Extracting RoutingExperience objects
4. Feeding experiences to AdvancedRoutingOptimizer
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import time

import pytest
import requests
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.routing_span_evaluator import RoutingSpanEvaluator
from cogniverse_core.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def phoenix_container():
    """Start Phoenix Docker container on non-default ports for routing span evaluator tests"""
    import cogniverse_core.telemetry.manager as telemetry_manager_module
    from cogniverse_core.telemetry.config import BatchExportConfig, TelemetryConfig
    from cogniverse_core.telemetry.registry import get_telemetry_registry

    # Reset TelemetryManager singleton AND clear provider cache
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    # Initialize with test Phoenix config BEFORE any tests run
    telemetry_config = TelemetryConfig(
        otlp_endpoint="http://localhost:24317",
        provider_config={
            "http_endpoint": "http://localhost:26006",
            "grpc_endpoint": "http://localhost:24317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    telemetry_manager_module._telemetry_manager = TelemetryManager(config=telemetry_config)

    container_name = f"phoenix_routing_span_eval_test_{int(time.time() * 1000)}"

    # Clean up old containers AND data directories
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", "name=phoenix_routing_span_eval_test"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            old_containers = result.stdout.strip().split("\n")
            for container_id in old_containers:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True,
                    timeout=10,
                )
            logger.info(f"Cleaned up {len(old_containers)} old Phoenix test containers")
    except Exception as e:
        logger.warning(f"Error cleaning up old containers: {e}")

    # Clean up old data directories from previous test runs
    try:
        import glob
        import shutil
        tmp_dir = tempfile.gettempdir()
        pattern = os.path.join(tmp_dir, "phoenix_routing_span_eval_*")
        logger.info(f"🧹 Searching for old Phoenix data directories: {pattern}")
        old_data_dirs = glob.glob(pattern)
        logger.info(f"🧹 Found {len(old_data_dirs)} old Phoenix data directories to clean")
        for old_dir in old_data_dirs:
            logger.info(f"🧹 Attempting to remove: {old_dir}")
            try:
                shutil.rmtree(old_dir)
                logger.info(f"✅ Successfully cleaned up: {old_dir}")
            except Exception as e:
                logger.warning(f"❌ Failed to remove {old_dir}: {e}")
    except Exception as e:
        logger.warning(f"❌ Error cleaning up old data directories: {e}")

    try:
        # Create temporary directory for Phoenix data
        test_data_dir = os.path.join(
            tempfile.gettempdir(), f"phoenix_routing_span_eval_{int(time.time())}"
        )
        logger.info(f"📁 Creating new Phoenix data directory: {test_data_dir}")
        os.makedirs(test_data_dir, exist_ok=True)
        logger.info(f"✅ Created Phoenix data directory: {test_data_dir}")

        # Start Phoenix container
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "26006:6006",  # HTTP port
                "-p",
                "24317:4317",  # gRPC port
                "-v",
                f"{test_data_dir}:/phoenix_data",
                "-e",
                "PHOENIX_WORKING_DIR=/phoenix_data",
                "-e",
                "PHOENIX_SQL_DATABASE_URL=sqlite:////phoenix_data/phoenix.db",
                "arizephoenix/phoenix:latest",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Phoenix container {container_name} started")

        # Wait for Phoenix to be ready
        max_wait_time = 60
        poll_interval = 0.5
        start_time = time.time()
        phoenix_ready = False

        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get("http://localhost:26006", timeout=2)
                if response.status_code == 200:
                    phoenix_ready = True
                    elapsed = time.time() - start_time
                    logger.info(f"Phoenix ready after {elapsed:.1f} seconds")
                    break
            except Exception:
                pass
            time.sleep(poll_interval)

        if not phoenix_ready:
            logs_result = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            logger.error(f"Phoenix logs:\n{logs_result.stdout}\n{logs_result.stderr}")
            raise RuntimeError(f"Phoenix failed to start after {max_wait_time} seconds")

        yield container_name

    finally:
        # Stop and remove Phoenix container
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=False,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "rm", container_name],
                check=False,
                capture_output=True,
                timeout=10,
            )
            logger.info(f"Phoenix container {container_name} stopped and removed")
        except Exception as e:
            logger.warning(f"Error cleaning up Phoenix container: {e}")
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass

        # Clean up data directory
        try:
            import shutil
            logger.info(f"🧹 Attempting final cleanup of Phoenix data directory: {test_data_dir}")
            if 'test_data_dir' in locals() and os.path.exists(test_data_dir):
                shutil.rmtree(test_data_dir)
                logger.info(f"✅ Cleaned up Phoenix data directory: {test_data_dir}")
            else:
                logger.info("📭 No data directory to clean up (doesn't exist or not in locals)")
        except Exception as e:
            logger.warning(f"❌ Error cleaning up data directory: {e}")



@pytest.fixture
async def routing_agent_with_spans(phoenix_container):
    """Create routing agent and generate real routing spans"""
    from cogniverse_agents.routing_agent import RoutingAgent
    from cogniverse_core.telemetry.config import BatchExportConfig, TelemetryConfig

    telemetry_config = TelemetryConfig(
        otlp_endpoint="http://localhost:24317",
        provider_config={
            "http_endpoint": "http://localhost:26006",
            "grpc_endpoint": "http://localhost:24317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    agent = RoutingAgent(tenant_id="test-tenant", telemetry_config=telemetry_config)

    # Generate real routing spans by processing test queries
    test_queries = [
        ("show me videos of basketball", "video_search_agent"),
        ("summarize the game highlights", "summarizer_agent"),
        ("detailed analysis of the match", "detailed_report_agent"),
        ("find soccer highlights", "video_search_agent"),
    ]

    logger.info("🔄 Generating routing spans...")
    for query, expected_agent in test_queries:
        result = await agent.route_query(query, tenant_id="test-tenant")
        # result is a RoutingDecision object
        agent_name = result.recommended_agent if result else 'unknown'
        logger.info(
            f"✅ Processed query: '{query}', "
            f"agent: {agent_name}, "
            f"expected: {expected_agent}"
        )

    # Force flush telemetry spans
    success = agent.telemetry_manager.force_flush(timeout_millis=10000)
    if not success:
        logger.error("❌ Failed to flush telemetry spans")
    else:
        logger.info("✅ Telemetry spans flushed")

    # Give Phoenix time to process spans
    await asyncio.sleep(2)

    return agent


@pytest.fixture(scope="function")
def optimizer():
    """Create AdvancedRoutingOptimizer for testing - fresh for each test"""
    return AdvancedRoutingOptimizer(tenant_id="test-tenant")


@pytest.fixture(scope="function")
def span_evaluator(optimizer):
    """Create RoutingSpanEvaluator for testing - fresh for each test"""
    # Telemetry manager already initialized by phoenix_container fixture
    return RoutingSpanEvaluator(optimizer=optimizer, tenant_id="test-tenant")


class TestRoutingSpanEvaluatorIntegration:
    """Integration tests for RoutingSpanEvaluator"""

    @pytest.mark.asyncio
    async def test_query_real_routing_spans(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test querying real routing spans from Phoenix"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Query spans via telemetry provider
        logger.info(f"📊 Querying spans from project: {span_evaluator.project_name}")

        from cogniverse_core.telemetry.manager import get_telemetry_manager
        telemetry_manager = get_telemetry_manager()
        provider = telemetry_manager.get_provider(tenant_id="test-tenant")
        spans_df = await provider.traces.get_spans(
            project=span_evaluator.project_name
        )

        logger.info(f"📊 Total spans in project: {len(spans_df)}")

        # Filter for routing spans
        routing_spans = spans_df[spans_df["name"] == "cogniverse.routing"]
        logger.info(f"📊 Routing spans found: {len(routing_spans)}")

        assert (
            len(routing_spans) >= 4
        ), f"Expected at least 4 routing spans, found {len(routing_spans)}"

    @pytest.mark.asyncio
    async def test_extract_routing_experiences(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test extracting routing experiences from real spans"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate routing spans
        logger.info("🔍 Evaluating routing spans...")
        results = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        logger.info(f"📊 Evaluation results: {results}")

        assert (
            results["spans_processed"] >= 4
        ), f"Expected at least 4 spans processed, got {results['spans_processed']}"
        assert (
            results["experiences_created"] >= 4
        ), f"Expected at least 4 experiences created, got {results['experiences_created']}"

    @pytest.mark.asyncio
    async def test_feed_experiences_to_optimizer(self):
        """Test feeding extracted experiences to AdvancedRoutingOptimizer"""
        import tempfile

        from cogniverse_agents.routing_agent import RoutingAgent
        from cogniverse_core.telemetry.config import BatchExportConfig, TelemetryConfig

        # Create telemetry config
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )

        # Create fresh routing agent and generate unique spans
        agent = RoutingAgent(tenant_id="test-tenant", telemetry_config=telemetry_config)

        # Use unique queries to avoid span ID collisions with other tests
        unique_queries = [
            ("show me tennis matches", "video_search_agent"),
            ("summarize the tennis tournament", "summarizer_agent"),
            ("detailed report on tennis finals", "detailed_report_agent"),
            ("find tennis highlights", "video_search_agent"),
        ]

        logger.info("🔄 Generating unique routing spans for optimizer test...")
        for query, _ in unique_queries:
            result = await agent.route_query(query, tenant_id="test-tenant")
            agent_name = result.recommended_agent if result else 'unknown'
            logger.info(
                f"✅ Processed query: '{query}', agent: {agent_name}"
            )

        # Force flush telemetry spans
        success = agent.telemetry_manager.force_flush(timeout_millis=10000)
        if not success:
            logger.error("❌ Failed to flush telemetry spans")
        else:
            logger.info("✅ Telemetry spans flushed")
        await asyncio.sleep(2)

        # Create optimizer with temporary storage to avoid loading existing data
        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer = AdvancedRoutingOptimizer(tenant_id="test-tenant", base_storage_dir=temp_dir)

            # Verify optimizer starts empty (no loaded data)
            initial_count = len(optimizer.experience_replay)
            logger.info(f"📊 Initial experience count: {initial_count}")
            assert (
                initial_count == 0
            ), f"Expected empty optimizer, got {initial_count} experiences"

            # Create span evaluator with our optimizer
            evaluator = RoutingSpanEvaluator(optimizer=optimizer, tenant_id="test-tenant")

            # Evaluate routing spans
            logger.info(f"📊 Evaluating spans from project: {evaluator.project_name}")
            results = await evaluator.evaluate_routing_spans(lookback_hours=1)
            logger.info(f"📊 Evaluation results: {results}")

            # Check that experiences were added to optimizer
            final_count = len(evaluator.optimizer.experience_replay)
            logger.info(f"📊 Final experience count: {final_count}")

            experiences_added = final_count - initial_count
            logger.info(f"📊 Experiences added: {experiences_added}")

            assert (
                experiences_added >= 4
            ), f"Expected at least 4 experiences added to optimizer, got {experiences_added}"

    @pytest.mark.asyncio
    async def test_routing_experience_structure(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that extracted RoutingExperience objects have correct structure"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate routing spans
        await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        # Get experiences from optimizer
        experiences = span_evaluator.optimizer.experience_replay

        assert (
            len(experiences) >= 4
        ), f"Expected at least 4 experiences, got {len(experiences)}"

        # Validate structure of first experience
        exp = experiences[0]
        assert exp.query, "Experience should have query"
        assert exp.chosen_agent, "Experience should have chosen_agent"
        assert 0.0 <= exp.routing_confidence <= 1.0, "Confidence should be in [0,1]"
        assert 0.0 <= exp.search_quality <= 1.0, "Search quality should be in [0,1]"
        assert isinstance(exp.agent_success, bool), "agent_success should be boolean"

        logger.info(f"✅ Experience structure validated: {exp}")

    @pytest.mark.asyncio
    async def test_duplicate_span_filtering(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that duplicate spans are not processed twice"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate routing spans twice
        results1 = await span_evaluator.evaluate_routing_spans(lookback_hours=1)
        results2 = await span_evaluator.evaluate_routing_spans(lookback_hours=1)

        logger.info(
            f"📊 First evaluation: {results1['experiences_created']} experiences"
        )
        logger.info(
            f"📊 Second evaluation: {results2['experiences_created']} experiences"
        )

        # Second evaluation should create 0 new experiences (duplicates filtered)
        assert results2["experiences_created"] == 0, (
            f"Expected 0 new experiences on second run (duplicates), "
            f"got {results2['experiences_created']}"
        )

    @pytest.mark.asyncio
    async def test_span_extraction_with_batch_size(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test span extraction respects batch size limits"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Evaluate with small batch size
        results = await span_evaluator.evaluate_routing_spans(
            lookback_hours=1, batch_size=2
        )

        logger.info(f"📊 Batch size limited results: {results}")

        # Should process at most 2 spans
        assert (
            results["spans_processed"] <= 2
        ), f"Expected at most 2 spans with batch_size=2, got {results['spans_processed']}"
        assert results["experiences_created"] <= 2, (
            f"Expected at most 2 experiences with batch_size=2, "
            f"got {results['experiences_created']}"
        )

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_workflow(self, optimizer):
        """Test complete end-to-end workflow from span generation to experience creation"""
        from cogniverse_agents.routing_agent import RoutingAgent
        from cogniverse_core.telemetry.config import BatchExportConfig, TelemetryConfig

        # 1. Create telemetry config
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )

        # 2. Create fresh routing agent
        agent = RoutingAgent(tenant_id="test-tenant", telemetry_config=telemetry_config)

        # 2. Process a single query
        query = "show me basketball dunks"
        logger.info(f"🔄 Processing query: '{query}'")
        result = await agent.route_query(query, tenant_id="test-tenant")
        logger.info(f"✅ Result: {result}")

        # 3. Flush telemetry
        agent.telemetry_manager.force_flush(timeout_millis=10000)
        await asyncio.sleep(2)

        # 4. Create span evaluator
        evaluator = RoutingSpanEvaluator(optimizer=optimizer, tenant_id="test-tenant")

        # 5. Evaluate spans
        results = await evaluator.evaluate_routing_spans(lookback_hours=1)
        logger.info(f"📊 Evaluation results: {results}")

        # 6. Verify experience was created
        assert results["spans_processed"] >= 1, "Should process at least 1 span"
        assert (
            results["experiences_created"] >= 1
        ), "Should create at least 1 experience"

        # 7. Verify experience is in optimizer replay buffer
        assert (
            len(optimizer.experience_replay) >= 1
        ), "Optimizer should have experiences"

        # 8. Verify experience content - check that the query exists in any experience
        queries = [exp.query for exp in optimizer.experience_replay]
        assert (
            query in queries
        ), f"Expected query '{query}' in experiences, got: {queries}"

        # Get the matching experience
        exp = next(exp for exp in optimizer.experience_replay if exp.query == query)
        logger.info(
            f"✅ End-to-end workflow complete: {exp.chosen_agent} (confidence: {exp.routing_confidence})"
        )

    @pytest.mark.asyncio
    async def test_realistic_span_structure(
        self, routing_agent_with_spans, span_evaluator
    ):
        """Test that spans have the expected structure from Phase 1"""
        # Trigger fixture to generate spans
        _ = routing_agent_with_spans

        # Query telemetry directly to inspect span structure
        from cogniverse_core.telemetry.manager import get_telemetry_manager
        telemetry_manager = get_telemetry_manager()
        provider = telemetry_manager.get_provider(tenant_id="test-tenant")
        spans_df = await provider.traces.get_spans(
            project=span_evaluator.project_name
        )

        routing_spans = spans_df[spans_df["name"] == "cogniverse.routing"]

        assert (
            len(routing_spans) >= 4
        ), f"Expected at least 4 routing spans, found {len(routing_spans)}"

        # Inspect first routing span structure
        first_span = routing_spans.iloc[0]
        logger.info(f"📊 Span structure: {first_span.to_dict()}")

        # Check for expected attributes
        assert (
            first_span["name"] == "cogniverse.routing"
        ), "Span name should be cogniverse.routing"

        # Check for routing attributes (either flattened or nested)
        has_flattened = (
            "attributes.routing" in first_span and first_span["attributes.routing"]
        )
        has_nested = "attributes" in first_span and any(
            k.startswith("routing.") for k in first_span.get("attributes", {}).keys()
        )

        assert (
            has_flattened or has_nested
        ), "Span should have routing attributes in either flattened or nested format"

        if has_flattened:
            routing_attrs = first_span["attributes.routing"]
            logger.info(f"📊 Flattened routing attributes: {routing_attrs}")
            assert "chosen_agent" in routing_attrs, "Should have chosen_agent"
            assert "confidence" in routing_attrs, "Should have confidence"
        else:
            logger.info("📊 Using nested attribute format")

        logger.info("✅ Span structure matches Phase 1 expectations")
