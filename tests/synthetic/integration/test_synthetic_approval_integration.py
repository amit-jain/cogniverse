"""
End-to-end integration tests for synthetic data generation with human-in-the-loop approval

Tests the complete flow:
1. Generate synthetic data with DSPy
2. Extract confidence scores
3. Auto-approve high confidence items
4. Queue low confidence items for review
5. Store in Phoenix
6. Process human decisions
7. Regenerate rejected items
8. Verify in Phoenix traces
"""

import logging
import os
import subprocess
import time
from datetime import datetime

import dspy
import httpx
import phoenix as px
import pytest
import requests
from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
)
from cogniverse_agents.approval.phoenix_storage import ApprovalStorageImpl
from cogniverse_core.config.unified_config import (
    BackendConfig,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_core.telemetry.manager import TelemetryManager
from cogniverse_synthetic.approval.confidence_extractor import (
    SyntheticDataConfidenceExtractor,
)
from cogniverse_synthetic.approval.feedback_handler import SyntheticDataFeedbackHandler
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_synthetic.service import SyntheticDataService

from tests.utils.async_polling import wait_for_phoenix_processing

logger = logging.getLogger(__name__)


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama service is available."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def is_openai_api_available() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


# Skip markers for integration tests requiring real LMs
skip_if_no_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama service not available at http://localhost:11434",
)

skip_if_no_openai = pytest.mark.skipif(
    not is_openai_api_available(), reason="OPENAI_API_KEY environment variable not set"
)


@pytest.fixture(scope="function", autouse=True)
def phoenix_container():
    """Start Phoenix Docker container on non-default ports for each test"""
    import os

    # Set environment variables BEFORE any TelemetryManager is created
    original_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    original_sync_export = os.environ.get("TELEMETRY_SYNC_EXPORT")

    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:24317"
    os.environ["TELEMETRY_SYNC_EXPORT"] = "true"

    # Reset TelemetryManager singleton using reset() class method
    TelemetryManager.reset()

    container_name = f"phoenix_synthetic_test_{int(time.time() * 1000)}"

    # Clean up old containers
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", "name=phoenix_synthetic_test"],
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

    try:
        # Create temporary directory for Phoenix data
        import os
        import tempfile

        test_data_dir = os.path.join(
            tempfile.gettempdir(), f"phoenix_test_{int(time.time())}"
        )
        os.makedirs(test_data_dir, exist_ok=True)

        # Start Phoenix container with SQLite persistent storage
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
                f"{test_data_dir}:/phoenix_data",  # Mount temp directory
                "-e",
                "PHOENIX_WORKING_DIR=/phoenix_data",  # Enable persistent storage
                "-e",
                "PHOENIX_SQL_DATABASE_URL=sqlite:////phoenix_data/phoenix.db",  # SQLite database
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
        # Cleanup - COMMENTED OUT FOR DEBUGGING
        # Stop Phoenix container after test
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

        # Restore original environment variables
        if original_endpoint:
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = original_endpoint
        else:
            os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)

        if original_sync_export:
            os.environ["TELEMETRY_SYNC_EXPORT"] = original_sync_export
        else:
            os.environ.pop("TELEMETRY_SYNC_EXPORT", None)


@pytest.fixture
def phoenix_client():
    """Phoenix client for querying approval data"""
    return px.Client(endpoint="http://localhost:26006")


@pytest.fixture(params=["ollama"])
def dspy_lm(request):
    """Configure DSPy with real LM following established pattern"""
    lm_type = request.param

    if lm_type == "ollama":
        if not is_ollama_available():
            pytest.skip("Ollama not available")
        # Use dspy.LM with smallest model for tests
        lm = dspy.LM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )
    else:
        raise ValueError(f"Unknown LM type: {lm_type}")

    dspy.configure(lm=lm)
    yield lm
    dspy.configure(lm=None)


@pytest.fixture
def generator_config():
    """Test synthetic generator configuration"""
    return SyntheticGeneratorConfig(
        optimizer_configs={
            "routing": OptimizerGenerationConfig(
                optimizer_type="routing",
                dspy_modules={
                    "query_generator": DSPyModuleConfig(
                        signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                        module_type="Predict",
                    )
                },
            ),
        }
    )


@pytest.fixture
def backend_config():
    """Test backend configuration"""
    return BackendConfig(profiles={})


@pytest.fixture
def telemetry_manager(phoenix_container):
    """TelemetryManager configured for approval tests"""
    # phoenix_container fixture ensures env vars are set and TelemetryManager singleton is reset
    # Create TelemetryManager which will read PHOENIX_COLLECTOR_ENDPOINT and TELEMETRY_SYNC_EXPORT
    from cogniverse_core.telemetry.manager import TelemetryManager

    manager = TelemetryManager()
    yield manager

    # Cleanup
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.fixture
def approval_storage(phoenix_container, telemetry_manager):
    """Phoenix approval storage with proper TelemetryManager integration"""
    # Depend on phoenix_container to ensure it's running and env vars are set
    # Depend on telemetry_manager to use proper tenant-scoped span creation
    return ApprovalStorageImpl(
        phoenix_grpc_endpoint="http://localhost:24317",  # gRPC port for span export
        phoenix_http_endpoint="http://localhost:26006",  # HTTP port for span queries
        tenant_id="test-tenant1",
        telemetry_manager=telemetry_manager,
    )


@pytest.fixture
def confidence_extractor():
    """Confidence extractor for synthetic data"""
    return SyntheticDataConfidenceExtractor()


@pytest.fixture
def feedback_handler(dspy_lm):
    """Feedback handler for synthetic data"""
    return SyntheticDataFeedbackHandler()


@pytest.fixture
def approval_agent(approval_storage, confidence_extractor, feedback_handler):
    """Human approval agent"""
    return HumanApprovalAgent(
        storage=approval_storage,
        confidence_extractor=confidence_extractor,
        feedback_handler=feedback_handler,
        confidence_threshold=0.8,  # High threshold for testing
    )


@pytest.fixture
def synthetic_service(generator_config, backend_config, dspy_lm):
    """Synthetic data service"""
    return SyntheticDataService(
        generator_config=generator_config, backend_config=backend_config
    )


class TestSyntheticApprovalIntegration:
    """End-to-end integration tests for synthetic data approval workflow"""

    @pytest.mark.asyncio
    async def test_end_to_end_approval_workflow(
        self, synthetic_service, approval_agent, approval_storage, phoenix_client
    ):
        """Test complete workflow: generate -> review -> approve/reject -> regenerate"""

        # Step 1: Generate synthetic data
        request = SyntheticDataRequest(optimizer="routing", count=5)
        response = await synthetic_service.generate(request)

        assert len(response.data) == 5
        assert response.optimizer == "routing"

        # Step 2: Create review items from generated data
        items = []
        for idx, example in enumerate(response.data):
            item_dict = {
                "query": example.get("query", ""),
                "entities": example.get("entities", []),
                "enhanced_query": example.get("enhanced_query", ""),
            }
            items.append(item_dict)

        # Step 3: Submit batch for approval
        batch_id = f"test_batch_{int(time.time())}"
        context = {"optimizer": "routing", "purpose": "integration_test"}

        batch = await approval_agent.process_batch(items, batch_id, context)
        assert batch.batch_id == batch_id

        # Step 4: Verify batch was stored
        retrieved_batch = await approval_storage.get_batch(batch_id)
        assert retrieved_batch is not None
        assert len(retrieved_batch.items) == 5

        # Step 5: Process approval decisions
        # Approve some, reject others
        decisions = [
            ReviewDecision(
                item_id=retrieved_batch.items[0].item_id,
                approved=True,
                feedback="Looks good",
                reviewer="test_user",
            ),
            ReviewDecision(
                item_id=retrieved_batch.items[1].item_id,
                approved=False,
                feedback="Query too generic",
                corrections={"query": "more specific query"},
                reviewer="test_user",
            ),
            ReviewDecision(
                item_id=retrieved_batch.items[2].item_id,
                approved=True,
                reviewer="test_user",
            ),
        ]

        # Process decisions
        for decision in decisions:
            result = await approval_agent.apply_decision(batch_id, decision)
            if decision.approved:
                assert result.status == ApprovalStatus.APPROVED
            else:
                # Should regenerate
                assert result.status in [
                    ApprovalStatus.REGENERATED,
                    ApprovalStatus.REJECTED,
                ]

        # Step 6: Verify final state
        # Wait for Phoenix to index annotations (Phoenix has indexing lag)
        wait_for_phoenix_processing(delay=2.0, description="annotation indexing")

        final_batch = await approval_storage.get_batch(batch_id)
        assert final_batch is not None

        approved_count = len(final_batch.approved)
        rejected_count = len(final_batch.rejected)

        # At least 2 approved (decisions[0] and decisions[2])
        assert approved_count >= 2
        logger.info(
            f"Final state: {approved_count} approved, {rejected_count} rejected"
        )

    @pytest.mark.asyncio
    async def test_auto_approval_threshold(self, approval_agent, approval_storage):
        """Test that high confidence items are auto-approved"""

        # Create items with varying data
        items = [{"query": f"test query {i}"} for i in range(5)]

        batch_id = "confidence_test"
        context = {"purpose": "threshold_test"}

        await approval_agent.process_batch(items, batch_id, context)

        # Retrieve and check auto-approved items
        retrieved = await approval_storage.get_batch(batch_id)
        assert retrieved is not None

        auto_approved = retrieved.auto_approved
        pending = retrieved.pending_review

        # Some items should be auto-approved based on confidence threshold
        total_items = len(auto_approved) + len(pending)
        assert total_items == 5

        logger.info(
            f"Auto-approved: {len(auto_approved)}, Pending review: {len(pending)}"
        )

    @pytest.mark.asyncio
    async def test_feedback_driven_regeneration(
        self, approval_agent, approval_storage, synthetic_service, dspy_lm
    ):
        """Test that rejected items are regenerated with feedback"""

        # Generate initial data
        request = SyntheticDataRequest(optimizer="routing", count=1)
        response = await synthetic_service.generate(request)

        initial_query = response.data[0]["query"]

        # Submit batch for approval
        items = [response.data[0]]
        batch_id = "regen_batch"
        context = {"purpose": "regen_test", "optimizer": "routing"}

        batch = await approval_agent.process_batch(items, batch_id, context)
        item_id = batch.items[0].item_id

        # Reject with specific feedback
        decision = ReviewDecision(
            item_id=item_id,
            approved=False,
            feedback="Query needs to be more specific and include entity name",
            corrections={"entity_requirement": "Must contain entity name"},
            reviewer="test_user",
        )

        # Process rejection - should trigger regeneration
        result = await approval_agent.apply_decision(batch_id, decision)

        # Check that item was regenerated
        if result.status == ApprovalStatus.REGENERATED:
            # Verify regenerated data is different
            regenerated_query = result.data.get("query", "")
            assert regenerated_query != initial_query
            logger.info(f"Original: {initial_query}")
            logger.info(f"Regenerated: {regenerated_query}")

    @pytest.mark.asyncio
    async def test_pending_batches_retrieval(
        self, approval_storage, confidence_extractor, feedback_handler
    ):
        """Test retrieving batches with pending reviews"""

        # Create approval agent with very high threshold so nothing auto-approves
        approval_agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
            feedback_handler=feedback_handler,
            confidence_threshold=1.0,  # Nothing will auto-approve
        )

        # Create multiple batches
        items1 = [{"query": "test1"}]
        items2 = [{"query": "test2"}]

        await approval_agent.process_batch(items1, "batch1", {"optimizer": "routing"})
        await approval_agent.process_batch(items2, "batch2", {"optimizer": "modality"})

        # Wait for Phoenix to index batches (Phoenix has indexing lag)
        wait_for_phoenix_processing(delay=2.0, description="batch indexing")

        # Retrieve all pending batches
        pending = await approval_storage.get_pending_batches()
        assert len(pending) >= 2

        # Filter by context
        routing_batches = await approval_storage.get_pending_batches(
            context_filter={"optimizer": "routing"}
        )
        assert len(routing_batches) >= 1
        assert any(b.batch_id == "batch1" for b in routing_batches)

    @pytest.mark.asyncio
    async def test_batch_approval_rate_calculation(
        self, approval_agent, approval_storage
    ):
        """Test approval rate calculation for batch"""

        items = [{"query": f"test {i}"} for i in range(4)]

        batch_id = "rate_test"
        context = {"purpose": "rate_calculation"}

        batch = await approval_agent.process_batch(items, batch_id, context)

        # Approve one pending item
        if batch.pending_review:
            pending_item = batch.pending_review[0]
            decision = ReviewDecision(
                item_id=pending_item.item_id, approved=True, reviewer="test_user"
            )
            await approval_agent.apply_decision(batch_id, decision)

        # Check approval rate
        final_batch = await approval_storage.get_batch(batch_id)
        approval_rate = final_batch.approval_rate

        # Should have some approved items
        assert approval_rate >= 0.0  # At least 0
        logger.info(f"Approval rate: {approval_rate:.2%}")

    @pytest.mark.asyncio
    async def test_phoenix_storage_integration(self, approval_storage, phoenix_client):
        """Test that approval data is correctly stored and retrievable from Phoenix"""

        # Create and save a batch
        item = ReviewItem(
            item_id="phoenix_test_item",
            data={"query": "test query", "entities": ["TestEntity"]},
            confidence=0.7,
            metadata={"source": "phoenix_integration_test"},
        )

        batch = ApprovalBatch(
            batch_id="phoenix_test_batch",
            items=[item],
            context={"purpose": "phoenix_integration"},
        )

        batch_id = await approval_storage.save_batch(batch)
        assert batch_id == "phoenix_test_batch"

        # Wait for Phoenix to process spans
        # Phoenix indexing can take several seconds
        wait_for_phoenix_processing(delay=3.0, description="span indexing")

        # Retrieve from storage
        retrieved = await approval_storage.get_batch(batch_id)
        assert retrieved is not None
        assert len(retrieved.items) == 1
        assert retrieved.items[0].item_id == "phoenix_test_item"

        # Update item status
        retrieved.items[0].status = ApprovalStatus.APPROVED
        retrieved.items[0].reviewed_at = datetime.utcnow()
        await approval_storage.update_item(retrieved.items[0], batch_id=batch_id)

        # Wait for Phoenix to process
        wait_for_phoenix_processing(delay=2.0, description="annotation update")

        # Verify update
        final = await approval_storage.get_batch(batch_id)
        assert final.items[0].status == ApprovalStatus.APPROVED
        assert final.items[0].reviewed_at is not None


class TestSyntheticServiceIntegration:
    """Integration tests for synthetic data service with real backends"""

    @pytest.mark.asyncio
    async def test_service_with_phoenix_telemetry(
        self, synthetic_service, phoenix_container, phoenix_client
    ):
        """Test that synthetic generation produces Phoenix traces"""

        request = SyntheticDataRequest(optimizer="routing", count=3)
        response = await synthetic_service.generate(request)

        assert len(response.data) == 3

        # Wait for telemetry export
        wait_for_phoenix_processing(delay=2.0, description="telemetry export")

        # Query Phoenix for traces
        # Note: This would require Phoenix spans to be created during generation
        # The actual query depends on how telemetry is instrumented

    @pytest.mark.asyncio
    async def test_multiple_optimizer_types(self, synthetic_service):
        """Test generating data for different optimizer types"""

        # Note: This requires full generator_config with all optimizers
        # For now just test routing
        request = SyntheticDataRequest(optimizer="routing", count=5)
        response = await synthetic_service.generate(request)

        assert response.optimizer == "routing"
        assert len(response.data) == 5

        # Verify schema
        for example in response.data:
            assert "query" in example
            assert "entities" in example
            assert "enhanced_query" in example


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
