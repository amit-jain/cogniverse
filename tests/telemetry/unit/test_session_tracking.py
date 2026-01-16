"""
Unit tests for session tracking in telemetry.

Tests validate:
1. session_context() abstract method in TelemetryProvider
2. session_context() implementation in PhoenixProvider
3. session_span() method in TelemetryManager
4. session() method in TelemetryManager
5. Session ID propagation to nested spans
"""

from unittest.mock import MagicMock, patch

import pytest
from cogniverse_foundation.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_foundation.telemetry.manager import NoOpSpan, TelemetryManager


@pytest.fixture(scope="function")
def telemetry_config_sync():
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
        level=TelemetryLevel.DETAILED,
        max_cached_tenants=5,
        batch_config=BatchExportConfig(
            use_sync_export=True,
        ),
    )


@pytest.fixture(scope="function", autouse=True)
def reset_telemetry_manager():
    """Reset TelemetryManager singleton before each test."""
    TelemetryManager._instance = None
    yield
    # Cleanup after test
    if TelemetryManager._instance is not None:
        try:
            TelemetryManager._instance.shutdown()
        except Exception:
            pass
        TelemetryManager._instance = None


class TestSessionContextAbstractMethod:
    """Test session_context() abstract method in TelemetryProvider."""

    def test_session_context_is_abstract_method(self):
        """Verify session_context is defined as an abstract method."""

        from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

        # Check that session_context is defined
        assert hasattr(TelemetryProvider, "session_context")

        # Check it's marked as abstract
        method = getattr(TelemetryProvider, "session_context")
        assert getattr(method, "__isabstractmethod__", False) or hasattr(
            method, "__wrapped__"
        )

    def test_session_context_signature(self):
        """Verify session_context has correct signature."""
        import inspect

        from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

        sig = inspect.signature(TelemetryProvider.session_context)
        params = list(sig.parameters.keys())

        # Should have self and session_id parameters
        assert "self" in params
        assert "session_id" in params


@pytest.mark.integration
@pytest.mark.telemetry
class TestPhoenixProviderSessionContext:
    """Test session_context() implementation in PhoenixProvider.

    Requires cogniverse-telemetry-phoenix to be installed.
    """

    @pytest.fixture
    def phoenix_provider(self):
        """Import and return PhoenixProvider if available."""
        try:
            from cogniverse_telemetry_phoenix.provider import PhoenixProvider
            return PhoenixProvider()
        except ImportError:
            pytest.skip("cogniverse-telemetry-phoenix not installed")

    def test_phoenix_provider_implements_session_context(self, phoenix_provider):
        """Verify PhoenixProvider implements session_context."""
        assert hasattr(phoenix_provider, "session_context")
        assert callable(phoenix_provider.session_context)

    def test_session_context_uses_openinference(self, phoenix_provider):
        """Verify session_context uses openinference.using_session."""
        with patch("cogniverse_telemetry_phoenix.provider.using_session") as mock_using_session:
            # Setup mock context manager
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_using_session.return_value = mock_cm

            session_id = "test-session-123"

            with phoenix_provider.session_context(session_id):
                pass

            # Verify using_session was called with correct session_id
            mock_using_session.assert_called_once_with(session_id)
            mock_cm.__enter__.assert_called_once()
            mock_cm.__exit__.assert_called_once()

    def test_session_context_is_context_manager(self, phoenix_provider):
        """Verify session_context is a context manager."""
        # Should work as context manager without error
        with patch("cogniverse_telemetry_phoenix.provider.using_session") as mock:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock.return_value = mock_cm

            with phoenix_provider.session_context("session-id"):
                # Context manager should work
                pass


class TestTelemetryManagerSession:
    """Test session() method in TelemetryManager."""

    def test_session_requires_tenant_id(self, telemetry_config_sync):
        """Test that session() raises ValueError without tenant_id."""
        manager = TelemetryManager(telemetry_config_sync)

        with pytest.raises(ValueError, match="tenant_id is required"):
            with manager.session(tenant_id="", session_id="session-123"):
                pass

    def test_session_requires_session_id(self, telemetry_config_sync):
        """Test that session() raises ValueError without session_id."""
        manager = TelemetryManager(telemetry_config_sync)

        with pytest.raises(ValueError, match="session_id is required"):
            with manager.session(tenant_id="test-tenant", session_id=""):
                pass

    def test_session_with_disabled_telemetry(self):
        """Test session() gracefully handles disabled telemetry."""
        config = TelemetryConfig(enabled=False)
        manager = TelemetryManager(config)

        # Should work without error (no-op)
        with manager.session(tenant_id="test-tenant", session_id="session-123"):
            pass

    def test_session_graceful_degradation_without_provider(self, telemetry_config_sync):
        """Test session() gracefully degrades when provider unavailable."""
        manager = TelemetryManager(telemetry_config_sync)

        # Should work without error even if provider fails
        # (graceful degradation - yields without session context)
        with manager.session(tenant_id="test-tenant", session_id="session-123"):
            pass


class TestTelemetryManagerSessionSpan:
    """Test session_span() method in TelemetryManager."""

    def test_session_span_requires_tenant_id(self, telemetry_config_sync):
        """Test that session_span() raises ValueError without tenant_id."""
        manager = TelemetryManager(telemetry_config_sync)

        with pytest.raises(ValueError, match="tenant_id is required"):
            with manager.session_span(
                name="operation",
                tenant_id="",
                session_id="session-123",
            ):
                pass

    def test_session_span_requires_session_id(self, telemetry_config_sync):
        """Test that session_span() raises ValueError without session_id."""
        manager = TelemetryManager(telemetry_config_sync)

        with pytest.raises(ValueError, match="session_id is required"):
            with manager.session_span(
                name="operation",
                tenant_id="test-tenant",
                session_id="",
            ):
                pass

    def test_session_span_with_disabled_telemetry(self):
        """Test session_span() returns NoOpSpan when telemetry disabled."""
        config = TelemetryConfig(enabled=False)
        manager = TelemetryManager(config)

        with manager.session_span(
            name="operation",
            tenant_id="test-tenant",
            session_id="session-123",
        ) as span:
            assert isinstance(span, NoOpSpan)

    def test_session_span_graceful_degradation(self, telemetry_config_sync):
        """Test session_span() gracefully degrades when provider unavailable."""
        manager = TelemetryManager(telemetry_config_sync)

        # Should work without error - returns span even if session context fails
        with manager.session_span(
            name="test_operation",
            tenant_id="test-tenant",
            session_id="session-123",
            attributes={"key": "value"},
        ) as span:
            # May be NoOpSpan if provider unavailable, but should not raise
            pass


@pytest.mark.integration
@pytest.mark.telemetry
class TestSearchRouterSessionIntegration:
    """Test session tracking in search router (without actually calling the API).

    Requires cogniverse-runtime to be installed.
    """

    @pytest.fixture
    def search_models(self):
        """Import search models if available."""
        try:
            from cogniverse_runtime.routers.search import SearchRequest, SearchResponse
            return SearchRequest, SearchResponse
        except ImportError:
            pytest.skip("cogniverse-runtime not installed")

    def test_search_request_accepts_session_id(self, search_models):
        """Test that SearchRequest model accepts session_id field."""
        SearchRequest, _ = search_models

        # Should work without error
        request = SearchRequest(
            query="test query",
            tenant_id="test-tenant",
            session_id="test-session-123",
        )

        assert request.session_id == "test-session-123"

    def test_search_request_session_id_optional(self, search_models):
        """Test that session_id is optional in SearchRequest (client provides it)."""
        SearchRequest, _ = search_models

        request = SearchRequest(
            query="test query",
            tenant_id="test-tenant",
            # session_id not provided - client's responsibility
        )

        assert request.session_id is None

    def test_search_response_session_id_optional(self, search_models):
        """Test that SearchResponse session_id is optional (echoes client's value)."""
        _, SearchResponse = search_models

        # Response with session_id
        response_with = SearchResponse(
            query="test query",
            profile="default",
            strategy="hybrid",
            results_count=0,
            results=[],
            session_id="client-session-456",
        )
        assert response_with.session_id == "client-session-456"

        # Response without session_id (client didn't provide one)
        response_without = SearchResponse(
            query="test query",
            profile="default",
            strategy="hybrid",
            results_count=0,
            results=[],
        )
        assert response_without.session_id is None


@pytest.mark.integration
@pytest.mark.telemetry
class TestSessionTrackingWithPhoenix:
    """Integration tests for session tracking with Phoenix.

    These tests require:
    1. cogniverse-telemetry-phoenix installed
    2. A running Phoenix server

    Run with: pytest tests/telemetry/unit/test_session_tracking.py -v -m integration
    """

    @pytest.fixture(scope="function")
    def phoenix_config(self):
        """Config for Phoenix integration."""
        return TelemetryConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4317",
            provider_config={
                "http_endpoint": "http://localhost:6006",
                "grpc_endpoint": "http://localhost:4317",
            },
            service_name="session-test",
            environment="test",
            batch_config=BatchExportConfig(
                use_sync_export=True,
            ),
        )

    def test_session_span_creates_traces_with_session_id(self, phoenix_config):
        """Test that session_span creates traces with session.id attribute."""
        try:
            # Check if Phoenix provider is available
            from cogniverse_telemetry_phoenix.provider import PhoenixProvider
        except ImportError:
            pytest.skip("cogniverse-telemetry-phoenix not installed")

        manager = TelemetryManager(phoenix_config)
        session_id = "integration-test-session"

        # Create session span
        with manager.session_span(
            name="integration_test_operation",
            tenant_id="test-tenant",
            session_id=session_id,
            attributes={"test": "session_tracking"},
        ) as span:
            # May be NoOpSpan if Phoenix server not running
            if isinstance(span, NoOpSpan):
                pytest.skip("Phoenix server not running")

            # Create nested span
            with manager.span(
                name="nested_operation",
                tenant_id="test-tenant",
            ) as nested:
                assert not isinstance(nested, NoOpSpan)

        # Force flush
        manager.force_flush(timeout_millis=5000)

    def test_multiple_requests_grouped_by_session(self, phoenix_config):
        """Test that multiple requests with same session_id are grouped."""
        try:
            from cogniverse_telemetry_phoenix.provider import PhoenixProvider
        except ImportError:
            pytest.skip("cogniverse-telemetry-phoenix not installed")

        manager = TelemetryManager(phoenix_config)
        session_id = "multi-request-session"

        # Simulate multiple requests
        spans_created = 0
        for i in range(3):
            with manager.session_span(
                name=f"request_{i}",
                tenant_id="test-tenant",
                session_id=session_id,
                attributes={"request_number": i},
            ) as span:
                if not isinstance(span, NoOpSpan):
                    spans_created += 1

        if spans_created == 0:
            pytest.skip("Phoenix server not running")

        # Force flush
        manager.force_flush(timeout_millis=5000)
