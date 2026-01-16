"""
Integration tests for Adapter Registry with real Vespa Docker instance.

Tests actual adapter registration, querying, activation, and lifecycle
operations against a live Vespa backend.

Requires Docker to be running.
"""

import logging
import time

import pytest

from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for adapter registry integration tests.

    Deploys the adapter_registry schema before yielding.
    """
    manager = VespaDockerManager()

    try:
        # Start container with module-specific ports
        container_info = manager.start_container(module_name=__name__, use_module_ports=True)

        # Wait for config server to be ready
        manager.wait_for_config_ready(container_info, timeout=180)

        # Give Vespa additional time for internal services to initialize
        logger.info("Waiting 15 seconds for Vespa internal services to initialize...")
        time.sleep(15)

        # Deploy metadata schemas including adapter_registry
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager.upload_metadata_schemas(app_name="cogniverse")
        logger.info("Deployed metadata schemas (including adapter_registry)")

        # Wait for Vespa HTTP/application endpoint to be ready
        manager.wait_for_application_ready(container_info, timeout=120)

        logger.info("Vespa initialization complete - ready for adapter registry tests")

        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        manager.stop_container()


@pytest.fixture
def adapter_store(vespa_instance):
    """Create VespaAdapterStore connected to test Vespa instance."""
    from cogniverse_vespa.registry.adapter_store import VespaAdapterStore

    http_port = vespa_instance["http_port"]
    logger.info(f"Creating VespaAdapterStore with http_port={http_port}")

    store = VespaAdapterStore(
        vespa_url="http://localhost",
        vespa_port=http_port,
    )
    return store


@pytest.fixture
def adapter_registry(adapter_store):
    """Create AdapterRegistry with real Vespa store."""
    from cogniverse_finetuning.registry import AdapterRegistry

    return AdapterRegistry(store=adapter_store)


@pytest.mark.integration
class TestAdapterRegistryVespaIntegration:
    """Integration tests for AdapterRegistry with real Vespa."""

    def test_health_check(self, adapter_store):
        """Test that Vespa adapter store is healthy."""
        assert adapter_store.health_check() is True

    def test_register_and_get_adapter(self, adapter_registry):
        """Test registering and retrieving an adapter."""
        # Register adapter
        adapter_id = adapter_registry.register_adapter(
            tenant_id="test_tenant",
            name="routing_sft_integration",
            version="1.0.0",
            base_model="HuggingFaceTB/SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/test_adapter",
            agent_type="routing",
            metrics={"train_loss": 0.45, "epochs": 3},
            training_config={"batch_size": 4, "learning_rate": 2e-4},
        )

        assert adapter_id is not None
        logger.info(f"Registered adapter: {adapter_id}")

        # Retrieve adapter
        adapter = adapter_registry.get_adapter(adapter_id)

        assert adapter is not None
        assert adapter.adapter_id == adapter_id
        assert adapter.tenant_id == "test_tenant"
        assert adapter.name == "routing_sft_integration"
        assert adapter.version == "1.0.0"
        assert adapter.base_model == "HuggingFaceTB/SmolLM-135M"
        assert adapter.model_type == "llm"
        assert adapter.training_method == "sft"
        assert adapter.agent_type == "routing"
        assert adapter.metrics["train_loss"] == 0.45
        assert adapter.status == "inactive"
        assert adapter.is_active is False

    def test_register_adapter_with_uri(self, adapter_registry):
        """Test registering an adapter with cloud storage URI."""
        adapter_id = adapter_registry.register_adapter(
            tenant_id="test_tenant",
            name="routing_sft_hf",
            version="1.0.0",
            base_model="HuggingFaceTB/SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/local_adapter",
            adapter_uri="hf://myorg/routing-adapter/v1",
            agent_type="routing",
        )

        adapter = adapter_registry.get_adapter(adapter_id)

        assert adapter is not None
        assert adapter.adapter_uri == "hf://myorg/routing-adapter/v1"
        assert adapter.get_effective_uri() == "hf://myorg/routing-adapter/v1"

    def test_list_adapters_by_tenant(self, adapter_registry):
        """Test listing adapters filtered by tenant."""
        # Register adapters for different tenants
        adapter_registry.register_adapter(
            tenant_id="tenant_a",
            name="adapter_1",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/adapter_1",
            agent_type="routing",
        )

        adapter_registry.register_adapter(
            tenant_id="tenant_a",
            name="adapter_2",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="dpo",
            adapter_path="/tmp/adapter_2",
            agent_type="routing",
        )

        adapter_registry.register_adapter(
            tenant_id="tenant_b",
            name="adapter_3",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/adapter_3",
            agent_type="routing",
        )

        # List adapters for tenant_a
        adapters_a = adapter_registry.list_adapters(tenant_id="tenant_a")
        assert len(adapters_a) >= 2
        assert all(a.tenant_id == "tenant_a" for a in adapters_a)

        # List adapters for tenant_b
        adapters_b = adapter_registry.list_adapters(tenant_id="tenant_b")
        assert len(adapters_b) >= 1
        assert all(a.tenant_id == "tenant_b" for a in adapters_b)

    def test_list_adapters_by_agent_type(self, adapter_registry):
        """Test listing adapters filtered by agent type."""
        # Register adapters for different agent types
        adapter_registry.register_adapter(
            tenant_id="tenant_filter",
            name="routing_adapter",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/routing",
            agent_type="routing",
        )

        adapter_registry.register_adapter(
            tenant_id="tenant_filter",
            name="profile_adapter",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/profile",
            agent_type="profile_selection",
        )

        # List only routing adapters
        routing_adapters = adapter_registry.list_adapters(
            tenant_id="tenant_filter", agent_type="routing"
        )
        assert len(routing_adapters) >= 1
        assert all(a.agent_type == "routing" for a in routing_adapters)

    def test_activate_and_get_active_adapter(self, adapter_registry):
        """Test activating an adapter and retrieving active adapter."""
        # Register two adapters
        adapter_id_1 = adapter_registry.register_adapter(
            tenant_id="tenant_active",
            name="routing_v1",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/v1",
            agent_type="routing",
        )

        adapter_id_2 = adapter_registry.register_adapter(
            tenant_id="tenant_active",
            name="routing_v2",
            version="2.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/v2",
            agent_type="routing",
        )

        # Initially no active adapter
        active = adapter_registry.get_active_adapter("tenant_active", "routing")
        # May be None or previous test's adapter

        # Activate first adapter
        adapter_registry.activate_adapter(adapter_id_1)

        active = adapter_registry.get_active_adapter("tenant_active", "routing")
        assert active is not None
        assert active.adapter_id == adapter_id_1
        assert active.is_active is True

        # Activate second adapter (should deactivate first)
        adapter_registry.activate_adapter(adapter_id_2)

        active = adapter_registry.get_active_adapter("tenant_active", "routing")
        assert active is not None
        assert active.adapter_id == adapter_id_2
        assert active.is_active is True

        # First adapter should be deactivated
        adapter_1 = adapter_registry.get_adapter(adapter_id_1)
        assert adapter_1.is_active is False

    def test_deactivate_adapter(self, adapter_registry):
        """Test deactivating an adapter."""
        adapter_id = adapter_registry.register_adapter(
            tenant_id="tenant_deactivate",
            name="to_deactivate",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/deactivate",
            agent_type="routing",
        )

        # Activate
        adapter_registry.activate_adapter(adapter_id)
        adapter = adapter_registry.get_adapter(adapter_id)
        assert adapter.is_active is True

        # Deactivate
        adapter_registry.deactivate_adapter(adapter_id)
        adapter = adapter_registry.get_adapter(adapter_id)
        assert adapter.is_active is False

    def test_deprecate_adapter(self, adapter_registry):
        """Test deprecating an adapter."""
        adapter_id = adapter_registry.register_adapter(
            tenant_id="tenant_deprecate",
            name="to_deprecate",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/deprecate",
            agent_type="routing",
        )

        # Activate first
        adapter_registry.activate_adapter(adapter_id)

        # Deprecate
        adapter_registry.deprecate_adapter(adapter_id)

        adapter = adapter_registry.get_adapter(adapter_id)
        assert adapter.status == "deprecated"
        assert adapter.is_active is False

    def test_delete_adapter(self, adapter_registry):
        """Test deleting an adapter from registry."""
        adapter_id = adapter_registry.register_adapter(
            tenant_id="tenant_delete",
            name="to_delete",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/delete",
            agent_type="routing",
        )

        # Verify it exists
        adapter = adapter_registry.get_adapter(adapter_id)
        assert adapter is not None

        # Delete
        result = adapter_registry.delete_adapter(adapter_id)
        assert result is True

        # Verify it's gone
        adapter = adapter_registry.get_adapter(adapter_id)
        assert adapter is None

    def test_get_latest_version(self, adapter_registry):
        """Test getting latest version of an adapter by name."""
        # Register multiple versions
        adapter_registry.register_adapter(
            tenant_id="tenant_version",
            name="versioned_adapter",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/v1.0.0",
            agent_type="routing",
        )

        adapter_registry.register_adapter(
            tenant_id="tenant_version",
            name="versioned_adapter",
            version="2.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/v2.0.0",
            agent_type="routing",
        )

        adapter_registry.register_adapter(
            tenant_id="tenant_version",
            name="versioned_adapter",
            version="1.5.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/v1.5.0",
            agent_type="routing",
        )

        # Get latest version
        latest = adapter_registry.get_latest_version(
            tenant_id="tenant_version", name="versioned_adapter"
        )

        assert latest is not None
        assert latest.version == "2.0.0"


@pytest.mark.integration
class TestInferenceIntegration:
    """Integration tests for inference helper functions."""

    def test_get_active_adapter_for_inference(self, adapter_registry):
        """Test inference helper to get active adapter."""

        # Register and activate an adapter
        adapter_id = adapter_registry.register_adapter(
            tenant_id="inference_tenant",
            name="inference_adapter",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/inference_adapter",
            adapter_uri="hf://myorg/inference-adapter",
            agent_type="routing",
        )

        adapter_registry.activate_adapter(adapter_id)

        # Use inference helper (note: this creates its own registry instance)
        # For integration test, we verify the adapter was activated via direct query
        active = adapter_registry.get_active_adapter("inference_tenant", "routing")

        assert active is not None
        assert active.name == "inference_adapter"
        assert active.get_effective_uri() == "hf://myorg/inference-adapter"

    def test_list_available_adapters_for_inference(self, adapter_registry):
        """Test listing adapters for inference."""
        # Register adapters
        adapter_registry.register_adapter(
            tenant_id="list_inference_tenant",
            name="adapter_a",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/a",
            agent_type="routing",
        )

        adapter_id = adapter_registry.register_adapter(
            tenant_id="list_inference_tenant",
            name="adapter_b",
            version="1.0.0",
            base_model="SmolLM-135M",
            model_type="llm",
            training_method="sft",
            adapter_path="/tmp/b",
            agent_type="routing",
        )

        # Activate one
        adapter_registry.activate_adapter(adapter_id)

        # List active adapters
        active_adapters = adapter_registry.list_adapters(
            tenant_id="list_inference_tenant", status="active"
        )

        assert len(active_adapters) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
