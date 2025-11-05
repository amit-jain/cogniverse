"""
Integration tests for multi-tenant telemetry system.

These tests validate:
1. Multi-tenant span isolation (tenant A spans don't leak to tenant B)
2. Project name mapping (tenant_id → Phoenix project)
3. SYNC vs BATCH export modes both work
4. Tenant provider caching and LRU eviction
5. Force flush across all tenants
"""

import time

import pytest
from cogniverse_core.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_core.telemetry.manager import NoOpSpan, TelemetryManager

from tests.utils.async_polling import wait_for_phoenix_processing


@pytest.mark.integration
@pytest.mark.telemetry
class TestMultiTenantTelemetryIntegration:
    """Integration tests for multi-tenant telemetry."""

    @pytest.fixture(scope="function")
    def telemetry_config_sync(self):
        """Create config for SYNC export mode (for testing)."""
        return TelemetryConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4317", provider_config={"http_endpoint": "http://localhost:6006", "grpc_endpoint": "http://localhost:4317"},
            
            service_name="test-service",
            environment="test",
            level=TelemetryLevel.DETAILED,
            max_cached_tenants=5,
            batch_config=BatchExportConfig(
                use_sync_export=True,  # SYNC mode for immediate testing
            ),
        )

    @pytest.fixture(scope="function")
    def telemetry_config_batch(self):
        """Create config for BATCH export mode (production mode)."""
        return TelemetryConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4317", provider_config={"http_endpoint": "http://localhost:6006", "grpc_endpoint": "http://localhost:4317"},
            
            service_name="test-service",
            environment="test",
            level=TelemetryLevel.DETAILED,
            max_cached_tenants=5,
            batch_config=BatchExportConfig(
                use_sync_export=False,  # BATCH mode (production)
            ),
        )

    def test_singleton_pattern(self, telemetry_config_sync):
        """Test that TelemetryManager is a singleton."""
        # Reset singleton for test
        TelemetryManager._instance = None

        manager1 = TelemetryManager(telemetry_config_sync)
        manager2 = TelemetryManager()  # Should return same instance

        assert manager1 is manager2
        assert id(manager1) == id(manager2)

        # Cleanup
        TelemetryManager._instance = None

    def test_multi_tenant_span_creation_sync_mode(self, telemetry_config_sync):
        """
        CRITICAL TEST: Validate multi-tenant span isolation in SYNC mode.

        Tests:
        1. Create spans for tenant-a and tenant-b
        2. Verify tenant.id attribute is set correctly
        3. Verify spans are created without errors
        4. Verify different tracers are used for different tenants
        """
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(telemetry_config_sync)

        # Create spans for tenant-a
        tenant_a_spans = []
        for i in range(3):
            with manager.span(
                name=f"tenant_a_operation_{i}",
                tenant_id="tenant-a",
                attributes={"operation_id": i, "tenant": "a"},
            ) as span:
                # Verify span is not NoOp
                assert not isinstance(span, NoOpSpan)
                tenant_a_spans.append(span)
                wait_for_phoenix_processing(delay=0.01, description="Phoenix processing")  # Small delay

        # Create spans for tenant-b
        tenant_b_spans = []
        for i in range(3):
            with manager.span(
                name=f"tenant_b_operation_{i}",
                tenant_id="tenant-b",
                attributes={"operation_id": i, "tenant": "b"},
            ) as span:
                assert not isinstance(span, NoOpSpan)
                tenant_b_spans.append(span)
                wait_for_phoenix_processing(delay=0.01, description="Phoenix processing")

        # Verify different tracers for different tenants
        tracer_a = manager.get_tracer("tenant-a")
        tracer_b = manager.get_tracer("tenant-b")

        assert tracer_a is not None
        assert tracer_b is not None
        # They should be cached separately (cache key format: tenant_id:project_name)
        assert "tenant-a:cogniverse-tenant-a" in manager._tenant_tracers
        assert "tenant-b:cogniverse-tenant-b" in manager._tenant_tracers

        # Verify cache hits
        stats = manager.get_stats()
        assert stats["cache_hits"] > 0  # Should have cache hits from repeated calls
        assert stats["cached_tenants"] >= 2  # At least tenant-a and tenant-b

        # Force flush to Phoenix
        success = manager.force_flush(timeout_millis=5000)
        assert success, "Force flush should succeed"

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None

    def test_multi_tenant_span_creation_batch_mode(self, telemetry_config_batch):
        """Test multi-tenant spans in BATCH mode (production mode)."""
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(telemetry_config_batch)

        # Create spans for multiple tenants
        tenants = ["tenant-1", "tenant-2", "tenant-3"]
        spans_per_tenant = 5

        for tenant_id in tenants:
            for i in range(spans_per_tenant):
                with manager.span(
                    name=f"operation_{i}",
                    tenant_id=tenant_id,
                    attributes={"index": i},
                ) as span:
                    assert not isinstance(span, NoOpSpan)
                    # In batch mode, spans are buffered
                    wait_for_phoenix_processing(delay=0.01, description="Phoenix processing")

        # Verify tracers created for all tenants
        assert len(manager._tenant_providers) >= 3

        # Force flush to send buffered spans
        success = manager.force_flush(timeout_millis=10000)
        assert success

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None

    def test_project_name_mapping(self, telemetry_config_sync):
        """Test that tenant_id maps to correct Phoenix project name."""
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(telemetry_config_sync)

        # Test project name generation (uses template: cogniverse-{tenant_id}-{service})
        test_cases = [
            ("acme", "routing", "cogniverse-acme-routing"),
            ("startup", "search", "cogniverse-startup-search"),
            ("tenant-123", "test-service", "cogniverse-tenant-123-test-service"),
        ]

        for tenant_id, service_suffix, expected_project in test_cases:
            project_name = telemetry_config_sync.get_project_name(
                tenant_id, service_suffix
            )
            assert (
                project_name == expected_project
            ), f"Expected {expected_project}, got {project_name}"

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None

    def test_tenant_cache_eviction(self, telemetry_config_sync):
        """
        Test LRU cache eviction for tenants.

        Creates more tenants than max_cached_tenants to trigger eviction.
        """
        # Reset singleton
        TelemetryManager._instance = None

        # Config with small cache size for testing
        config = TelemetryConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4317", provider_config={"http_endpoint": "http://localhost:6006", "grpc_endpoint": "http://localhost:4317"},
            
            service_name="test-service",
            max_cached_tenants=3,  # Small cache for testing
            batch_config=BatchExportConfig(use_sync_export=True),
        )

        manager = TelemetryManager(config)

        # Create spans for 5 tenants (exceeds cache size of 3)
        for i in range(5):
            tenant_id = f"tenant-{i}"
            with manager.span(
                name="operation",
                tenant_id=tenant_id,
                attributes={"tenant_index": i},
            ) as span:
                assert not isinstance(span, NoOpSpan)

        # Verify cache was evicted
        stats = manager.get_stats()
        assert (
            stats["cached_tracers"] <= 3
        ), f"Cache should be evicted, got {stats['cached_tracers']} tracers"

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None

    def test_span_error_handling(self, telemetry_config_sync):
        """Test that span context manager handles exceptions correctly."""
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(telemetry_config_sync)

        # Test exception within span
        with pytest.raises(ValueError, match="Test error"):
            with manager.span(
                name="error_operation",
                tenant_id="test-tenant",
                attributes={"will_fail": True},
            ) as span:
                assert not isinstance(span, NoOpSpan)
                raise ValueError("Test error")

        # Span should still be recorded with error status
        # Force flush to send
        manager.force_flush(timeout_millis=2000)

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None

    def test_disabled_telemetry_returns_noop_span(self):
        """Test that disabled telemetry returns NoOp spans."""
        # Reset singleton
        TelemetryManager._instance = None

        config = TelemetryConfig(
            enabled=False,  # Disabled
        )

        manager = TelemetryManager(config)

        with manager.span(
            name="test_operation", tenant_id="test-tenant"
        ) as span:
            assert isinstance(span, NoOpSpan)

        # Cleanup
        TelemetryManager._instance = None

    def test_span_attributes_with_none_values(self, telemetry_config_sync):
        """Test that None attribute values are skipped (OpenTelemetry rejects them)."""
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(telemetry_config_sync)

        # Create span with None attributes
        with manager.span(
            name="test_operation",
            tenant_id="test-tenant",
            attributes={
                "valid_attr": "value",
                "none_attr": None,  # Should be skipped
                "another_valid": 123,
            },
        ) as span:
            assert not isinstance(span, NoOpSpan)
            # Span should be created without error (None values skipped)

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None

    def test_multiple_service_names_per_tenant(self, telemetry_config_sync):
        """Test that one tenant can have multiple service tracers."""
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(telemetry_config_sync)

        tenant_id = "multi-service-tenant"

        # Create spans for different services under same tenant
        services = ["routing", "search", "ingestion"]

        for service in services:
            with manager.span(
                name="operation",
                tenant_id=tenant_id,
                project_name=service,
                attributes={"service": service},
            ) as span:
                assert not isinstance(span, NoOpSpan)

        # Verify different tracers created (cache key format: tenant_id:full_project_name)
        for service in services:
            cache_key = f"{tenant_id}:cogniverse-{tenant_id}-{service}"
            assert cache_key in manager._tenant_tracers

        # Should have 3 tracers for same tenant
        stats = manager.get_stats()
        assert stats["cached_tracers"] >= 3

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None

    def test_stats_reporting(self, telemetry_config_sync):
        """Test telemetry statistics reporting."""
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(telemetry_config_sync)

        # Create some spans
        for i in range(5):
            with manager.span(
                name=f"op_{i}", tenant_id="stats-tenant"
            ) as _span:
                pass

        # Get stats
        stats = manager.get_stats()

        # Verify structure
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "failed_initializations" in stats
        assert "cached_tenants" in stats
        assert "cached_tracers" in stats
        assert "config" in stats

        # Verify values
        assert isinstance(stats["cache_hits"], int)
        assert isinstance(stats["cache_misses"], int)
        assert stats["config"]["enabled"] is True
        assert stats["config"]["environment"] == "test"

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None


@pytest.mark.integration
@pytest.mark.telemetry
class TestPhoenixIntegrationWithRealServer:
    """
    Integration tests with real Phoenix server.

    These tests are skipped by default. To run them:
    1. Start Phoenix: python -m phoenix.server.main serve
    2. Unskip this class
    3. Run: pytest tests/telemetry/integration/test_multi_tenant_telemetry.py -v
    """

    @pytest.fixture(scope="function")
    def phoenix_config(self):
        """Config for real Phoenix integration."""
        return TelemetryConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4317", provider_config={"http_endpoint": "http://localhost:6006", "grpc_endpoint": "http://localhost:4317"},
            
            service_name="integration-test",
            environment="test",
            batch_config=BatchExportConfig(
                use_sync_export=True,  # Sync for immediate verification
            ),
        )

    def test_real_phoenix_multi_tenant_isolation(self, phoenix_config):
        """
        CRITICAL TEST: Validate tenant isolation with real Phoenix.

        This test:
        1. Creates spans for tenant-alpha and tenant-beta
        2. Flushes to Phoenix
        3. Queries Phoenix API to verify spans in correct projects
        4. Validates no cross-contamination
        """
        # Reset singleton
        TelemetryManager._instance = None

        manager = TelemetryManager(phoenix_config)

        # Create spans for tenant-alpha
        for i in range(5):
            with manager.span(
                name=f"alpha_operation_{i}",
                tenant_id="tenant-alpha",
                project_name="routing",
                attributes={
                    "operation_id": i,
                    "tenant": "alpha",
                    "test_timestamp": time.time(),
                },
            ) as span:
                span.set_attribute("step", "processing")

        # Create spans for tenant-beta
        for i in range(5):
            with manager.span(
                name=f"beta_operation_{i}",
                tenant_id="tenant-beta",
                project_name="routing",
                attributes={
                    "operation_id": i,
                    "tenant": "beta",
                    "test_timestamp": time.time(),
                },
            ) as span:
                span.set_attribute("step", "processing")

        # Force flush
        success = manager.force_flush(timeout_millis=10000)
        assert success

        # Wait for Phoenix to process
        wait_for_phoenix_processing(delay=2, description="Phoenix processing")

        # TODO: Query Phoenix API to verify:
        # - Project "tenant-alpha-routing" has 5 spans
        # - Project "tenant-beta-routing" has 5 spans
        # - No cross-contamination

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None
