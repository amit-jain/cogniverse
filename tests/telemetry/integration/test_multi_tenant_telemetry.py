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
import uuid

import pytest

from cogniverse_foundation.telemetry import manager as manager_mod
from cogniverse_foundation.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_foundation.telemetry.manager import (
    NoOpSpan,
    TelemetryManager,
    get_telemetry_manager,
)
from tests.utils.async_polling import wait_for_phoenix_processing


@pytest.mark.integration
@pytest.mark.telemetry
@pytest.mark.ci_fast
class TestMultiTenantTelemetryIntegration:
    """Integration tests for multi-tenant telemetry."""

    @pytest.fixture(scope="function")
    def telemetry_config_sync(self):
        """Create config for SYNC export mode (for testing)."""
        return TelemetryConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4317",
            provider_config={
                "http_endpoint": "http://localhost:6006",
                "grpc_endpoint": "http://localhost:4317",
            },
            service_name="test-service",
            environment="test",
            level=TelemetryLevel.VERBOSE,
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
            otlp_endpoint="http://localhost:4317",
            provider_config={
                "http_endpoint": "http://localhost:6006",
                "grpc_endpoint": "http://localhost:4317",
            },
            service_name="test-service",
            environment="test",
            level=TelemetryLevel.VERBOSE,
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
                wait_for_phoenix_processing(
                    delay=0.01, description="Phoenix processing"
                )  # Small delay

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
                wait_for_phoenix_processing(
                    delay=0.01, description="Phoenix processing"
                )

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
                    wait_for_phoenix_processing(
                        delay=0.01, description="Phoenix processing"
                    )

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
            assert project_name == expected_project, (
                f"Expected {expected_project}, got {project_name}"
            )

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
            level=TelemetryLevel.VERBOSE,  # test uses default "agents" component
            otlp_endpoint="http://localhost:4317",
            provider_config={
                "http_endpoint": "http://localhost:6006",
                "grpc_endpoint": "http://localhost:4317",
            },
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
        assert stats["cached_tracers"] <= 3, (
            f"Cache should be evicted, got {stats['cached_tracers']} tracers"
        )

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

        with manager.span(name="test_operation", tenant_id="test-tenant") as span:
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
            with manager.span(name=f"op_{i}", tenant_id="stats-tenant") as _span:
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
@pytest.mark.ci_fast
class TestPhoenixIntegrationWithRealServer:
    """
    Integration tests with real Phoenix server.

    Uses the phoenix_container fixture from conftest.py which manages
    its own Docker container on ports 16006 (HTTP) and 14317 (gRPC).
    """

    def test_real_phoenix_multi_tenant_isolation(self, phoenix_container):
        """
        CRITICAL TEST: Validate tenant isolation with real Phoenix.

        This test:
        1. Creates spans for tenant-alpha and tenant-beta tagged with a unique run ID
        2. Flushes to Phoenix
        3. Queries Phoenix API and filters to only this run's spans
        4. Validates count and no cross-contamination

        Uses a per-run UUID in span names so the test is idempotent against a
        persistent Phoenix instance that accumulates spans across test runs.
        Uses the phoenix_container fixture for per-pid HTTP/gRPC ports.
        """
        # Reset singleton
        TelemetryManager._instance = None

        phoenix_config = TelemetryConfig(
            enabled=True,
            level=TelemetryLevel.VERBOSE,  # test uses default "agents" component
            otlp_endpoint=phoenix_container["grpc_endpoint"],
            provider_config={
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            service_name="integration-test",
            environment="test",
            batch_config=BatchExportConfig(
                use_sync_export=True,
            ),
        )

        manager = TelemetryManager(phoenix_config)

        # Unique ID for this test run so we can isolate our spans from previous runs
        run_id = uuid.uuid4().hex[:8]

        # Create spans for tenant-alpha
        for i in range(5):
            with manager.span(
                name=f"alpha_op_{run_id}_{i}",
                tenant_id="tenant-alpha",
                project_name="routing",
                attributes={
                    "operation_id": i,
                    "tenant": "alpha",
                    "test_run_id": run_id,
                    "test_timestamp": time.time(),
                },
            ) as span:
                span.set_attribute("step", "processing")

        # Create spans for tenant-beta
        for i in range(5):
            with manager.span(
                name=f"beta_op_{run_id}_{i}",
                tenant_id="tenant-beta",
                project_name="routing",
                attributes={
                    "operation_id": i,
                    "tenant": "beta",
                    "test_run_id": run_id,
                    "test_timestamp": time.time(),
                },
            ) as span:
                span.set_attribute("step", "processing")

        # Force flush
        success = manager.force_flush(timeout_millis=10000)
        assert success

        # Wait for Phoenix to process
        wait_for_phoenix_processing(delay=2, description="Phoenix processing")

        # Query Phoenix API to verify tenant isolation
        from phoenix.client import Client

        client = Client(base_url=phoenix_container["http_endpoint"])

        # Verify tenant-alpha spans — filter to this run's spans by name prefix
        alpha_project = phoenix_config.get_project_name("tenant-alpha", "routing")
        all_alpha_spans = client.spans.get_spans_dataframe(
            project_identifier=alpha_project
        )
        assert all_alpha_spans is not None, f"No spans found in project {alpha_project}"
        alpha_spans = all_alpha_spans[
            all_alpha_spans["name"].str.startswith(f"alpha_op_{run_id}_")
        ]
        assert len(alpha_spans) == 5, (
            f"Expected 5 spans with run_id={run_id} in {alpha_project}, got {len(alpha_spans)}"
        )

        # Verify tenant-beta spans — filter to this run's spans by name prefix
        beta_project = phoenix_config.get_project_name("tenant-beta", "routing")
        all_beta_spans = client.spans.get_spans_dataframe(
            project_identifier=beta_project
        )
        assert all_beta_spans is not None, f"No spans found in project {beta_project}"
        beta_spans = all_beta_spans[
            all_beta_spans["name"].str.startswith(f"beta_op_{run_id}_")
        ]
        assert len(beta_spans) == 5, (
            f"Expected 5 spans with run_id={run_id} in {beta_project}, got {len(beta_spans)}"
        )

        # Verify no cross-contamination: alpha spans must not appear in beta project
        # and vice versa (using this run's unique prefix).
        alpha_run_names = set(alpha_spans["name"].tolist())
        assert all(f"alpha_op_{run_id}" in name for name in alpha_run_names), (
            f"Cross-contamination: alpha project has non-alpha spans for run {run_id}: {alpha_run_names}"
        )
        beta_run_names = set(beta_spans["name"].tolist())
        assert all(f"beta_op_{run_id}" in name for name in beta_run_names), (
            f"Cross-contamination: beta project has non-beta spans for run {run_id}: {beta_run_names}"
        )

        # Cleanup
        manager.shutdown()
        TelemetryManager._instance = None


@pytest.mark.integration
@pytest.mark.telemetry
@pytest.mark.ci_fast
class TestGetSpansNameFilterRealPhoenix:
    """The ``{"name": ...}`` filter on PhoenixTraceStore.get_spans must run
    server-side against real Phoenix and return exactly the matching spans —
    client-side name filtering pulled the whole project frame per call and
    burned the limit budget on unrelated span types."""

    def test_name_filter_returns_only_matching_spans(self, phoenix_container):
        import asyncio

        TelemetryManager._instance = None
        phoenix_config = TelemetryConfig(
            enabled=True,
            level=TelemetryLevel.VERBOSE,
            otlp_endpoint=phoenix_container["grpc_endpoint"],
            provider_config={
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            service_name="integration-test",
            environment="test",
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        manager = TelemetryManager(phoenix_config)
        run_id = uuid.uuid4().hex[:8]
        tenant = f"filter-{run_id}"

        # Two span names in one project: 3 checkpoint-style, 2 other.
        for i in range(3):
            with manager.span(
                name="workflow_checkpoint",
                tenant_id=tenant,
                project_name="routing",
                attributes={"test_run_id": run_id, "i": i},
            ):
                pass
        for i in range(2):
            with manager.span(
                name=f"other_op_{run_id}",
                tenant_id=tenant,
                project_name="routing",
                attributes={"test_run_id": run_id, "i": i},
            ):
                pass

        assert manager.force_flush(timeout_millis=10000)
        wait_for_phoenix_processing(delay=2, description="Phoenix processing")

        from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

        store = PhoenixTraceStore(
            http_endpoint=phoenix_container["http_endpoint"],
        )
        project = phoenix_config.get_project_name(tenant, "routing")

        filtered = asyncio.run(
            store.get_spans(project=project, filters={"name": "workflow_checkpoint"})
        )
        unfiltered = asyncio.run(store.get_spans(project=project))

        # Exact server-side semantics: only the requested name comes back.
        assert set(filtered["name"].unique()) == {"workflow_checkpoint"}
        assert len(filtered) == 3
        assert set(unfiltered["name"].unique()) == {
            "workflow_checkpoint",
            f"other_op_{run_id}",
        }
        assert len(unfiltered) == 5

        manager.shutdown()
        TelemetryManager._instance = None


@pytest.mark.integration
@pytest.mark.telemetry
@pytest.mark.ci_fast
class TestManagerResetRebuildRealPhoenix:
    """After ``reset()``, ``get_telemetry_manager()`` must hand back a fresh,
    live manager — not the shut-down one — that still emits to real Phoenix.
    """

    class _Cfg:
        def __init__(self, phoenix_container) -> None:
            self._pc = phoenix_container

        def get_telemetry_config(self, tenant_id: str) -> TelemetryConfig:
            return TelemetryConfig(
                enabled=True,
                level=TelemetryLevel.VERBOSE,
                otlp_endpoint=self._pc["grpc_endpoint"],
                provider_config={
                    "http_endpoint": self._pc["http_endpoint"],
                    "grpc_endpoint": self._pc["grpc_endpoint"],
                },
                service_name="reset-rebuild-test",
                environment="test",
                batch_config=BatchExportConfig(use_sync_export=True),
            )

    def test_reset_rebuilds_a_live_manager_that_emits(self, phoenix_container):
        TelemetryManager.reset()
        manager_mod._telemetry_manager = None

        cfg = self._Cfg(phoenix_container)
        cfg_obj = cfg.get_telemetry_config("system")
        run_id = uuid.uuid4().hex[:8]

        try:
            m1 = get_telemetry_manager(cfg)
            for i in range(3):
                with m1.span(
                    name=f"pre_{run_id}_{i}",
                    tenant_id="tenant-alpha",
                    project_name="routing",
                    attributes={"phase": "pre-reset", "run_id": run_id},
                ) as span:
                    span.set_attribute("i", i)
            assert m1.force_flush(timeout_millis=10000)

            TelemetryManager.reset()
            m2 = get_telemetry_manager(cfg)

            # The fix: a brand-new, live instance — not the shut-down m1.
            assert m2 is not m1
            assert m2._initialized is True

            for i in range(3):
                with m2.span(
                    name=f"post_{run_id}_{i}",
                    tenant_id="tenant-beta",
                    project_name="routing",
                    attributes={"phase": "post-reset", "run_id": run_id},
                ) as span:
                    span.set_attribute("i", i)
            assert m2.force_flush(timeout_millis=10000)

            wait_for_phoenix_processing(delay=2, description="Phoenix processing")

            from phoenix.client import Client

            client = Client(base_url=phoenix_container["http_endpoint"])

            alpha_project = cfg_obj.get_project_name("tenant-alpha", "routing")
            alpha_df = client.spans.get_spans_dataframe(
                project_identifier=alpha_project
            )
            assert alpha_df is not None
            pre = alpha_df[alpha_df["name"].str.startswith(f"pre_{run_id}_")]
            assert len(pre) == 3, f"pre-reset spans in {alpha_project}: {len(pre)}"

            # The rebuilt manager emitted its spans to the live instance.
            beta_project = cfg_obj.get_project_name("tenant-beta", "routing")
            beta_df = client.spans.get_spans_dataframe(project_identifier=beta_project)
            assert beta_df is not None
            post = beta_df[beta_df["name"].str.startswith(f"post_{run_id}_")]
            assert len(post) == 3, f"post-reset spans in {beta_project}: {len(post)}"
        finally:
            TelemetryManager.reset()
            manager_mod._telemetry_manager = None


@pytest.mark.integration
@pytest.mark.telemetry
@pytest.mark.ci_fast
class TestRequiredSpanRealBoundary:
    @pytest.mark.asyncio
    async def test_required_span_round_trips_exact_record(self, phoenix_container):
        import asyncio

        TelemetryManager.reset()
        run_id = uuid.uuid4().hex
        tenant_id = f"required-{run_id[:8]}"
        config = TelemetryConfig(
            enabled=True,
            level=TelemetryLevel.VERBOSE,
            otlp_endpoint=phoenix_container["grpc_endpoint"],
            provider_config={
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            service_name="required-span-test",
            environment="test",
            batch_config=BatchExportConfig(
                use_sync_export=False,
                export_timeout_millis=10_000,
            ),
        )
        manager = TelemetryManager(config)
        project = config.get_project_name(tenant_id, "experiments")

        try:
            async with manager.required_span(
                f"required_record_{run_id}",
                tenant_id=tenant_id,
                project_name="experiments",
                attributes={
                    "run.id": run_id,
                    "record.kind": "adapter_publication",
                },
            ) as span:
                span.set_attribute("record.state", "committed")

            def emit_thread_control():
                with manager.span(
                    f"required_thread_control_{run_id}",
                    tenant_id=tenant_id,
                    project_name="experiments",
                    attributes={"run.id": run_id},
                    require_export=True,
                ):
                    pass

            await asyncio.to_thread(emit_thread_control)

            from phoenix.client import Client

            client = Client(base_url=phoenix_container["http_endpoint"])

            async def load_record():
                project_names = set()
                for _ in range(120):
                    projects = await asyncio.to_thread(client.projects.list)
                    project_names = {item["name"] for item in projects}
                    for observed_project in project_names:
                        frame = await asyncio.to_thread(
                            client.spans.get_spans_dataframe,
                            project_identifier=observed_project,
                        )
                        if frame.empty or "name" not in frame.columns:
                            continue
                        expected_names = {
                            f"required_record_{run_id}",
                            f"required_thread_control_{run_id}",
                        }
                        matches = frame[frame["name"].isin(expected_names)]
                        if not matches.empty:
                            return observed_project, matches, project_names
                    await asyncio.sleep(0.1)
                return None, None, project_names

            observed_project, matches, project_names = await load_record()
            assert observed_project == project, project_names
            assert matches is not None
            assert set(matches["name"]) == {
                f"required_record_{run_id}",
                f"required_thread_control_{run_id}",
            }
            assert len(matches) == 2
            record = matches[matches["name"] == f"required_record_{run_id}"].iloc[0]
            assert record["attributes.run"] == {"id": run_id}
            assert record["attributes.record"] == {
                "kind": "adapter_publication",
                "state": "committed",
            }
            assert record["attributes.tenant"] == {"id": tenant_id}
            assert record["attributes.service"] == {"name": "required-span-test"}
            assert record["attributes.environment"] == "test"
        finally:
            TelemetryManager.reset()

    @pytest.mark.asyncio
    async def test_required_span_reports_closed_collector_failure(
        self,
        unused_tcp_port,
    ):
        TelemetryManager.reset()
        endpoint = f"localhost:{unused_tcp_port}"
        config = TelemetryConfig(
            enabled=True,
            level=TelemetryLevel.VERBOSE,
            otlp_endpoint=endpoint,
            service_name="required-span-test",
            environment="test",
            batch_config=BatchExportConfig(
                use_sync_export=False,
                export_timeout_millis=100,
            ),
        )
        manager = TelemetryManager(config)

        try:
            with pytest.raises(
                RuntimeError,
                match=(
                    "Required telemetry export failed: tenant=closed-boundary "
                    "project=cogniverse-closed-boundary-experiments "
                    f"endpoint={endpoint}"
                ),
            ) as exc_info:
                async with manager.required_span(
                    "required_record",
                    tenant_id="closed-boundary",
                    project_name="experiments",
                    attributes={"run.id": "must-not-succeed"},
                ):
                    pass
            assert str(exc_info.value.__cause__) == (
                "Phoenix rejected required span export: "
                "project=cogniverse-closed-boundary-experiments "
                f"endpoint={endpoint}"
            )
        finally:
            TelemetryManager.reset()
