"""Session tracking against a real Phoenix server.

Moved out of tests/telemetry/unit/ where the integration marker made these
tests unreachable: the unit CI job deselects integration-marked tests and the
integration job only collects this directory. The managed phoenix_container
fixture replaces the old hardcoded localhost endpoints and the
"server not running" skip.
"""

import pytest

from cogniverse_foundation.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_foundation.telemetry.manager import NoOpSpan, TelemetryManager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.telemetry]


@pytest.fixture(scope="function")
def reset_telemetry_manager():
    TelemetryManager._instance = None
    yield
    if TelemetryManager._instance is not None:
        try:
            TelemetryManager._instance.shutdown()
        except Exception:
            pass
        TelemetryManager._instance = None


@pytest.fixture(scope="function")
def phoenix_config(phoenix_container, reset_telemetry_manager):
    """Config wired to the managed Phoenix container."""
    return TelemetryConfig(
        enabled=True,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        service_name="session-test",
        environment="test",
        level=TelemetryLevel.VERBOSE,
        batch_config=BatchExportConfig(
            use_sync_export=True,
        ),
    )


class TestSessionTrackingWithPhoenix:
    def test_session_span_creates_traces_with_session_id(self, phoenix_config):
        """session_span creates real (non-NoOp) spans carrying session.id."""
        manager = TelemetryManager(phoenix_config)
        session_id = "integration-test-session"

        with manager.session_span(
            name="integration_test_operation",
            tenant_id="test-tenant",
            session_id=session_id,
            attributes={"test": "session_tracking"},
        ) as span:
            assert not isinstance(span, NoOpSpan)

            # Nested span — pass component=search_service so the test runs
            # regardless of TelemetryConfig.level (the default 'agents' tier
            # would NoOp below VERBOSE).
            with manager.span(
                name="nested_operation",
                tenant_id="test-tenant",
                component="search_service",
            ) as nested:
                assert not isinstance(nested, NoOpSpan)

        manager.force_flush(timeout_millis=5000)

    def test_multiple_requests_grouped_by_session(self, phoenix_config):
        """Multiple session_span requests with the same session_id each
        propagate that session.id into the OTel context — the grouping key
        OpenInference stamps onto every span, so the traces correlate."""
        from openinference.instrumentation import get_attributes_from_context

        manager = TelemetryManager(phoenix_config)
        session_id = "multi-request-session"

        seen = []
        for i in range(3):
            with manager.session_span(
                name=f"request_{i}",
                tenant_id="test-tenant",
                session_id=session_id,
                attributes={"request_number": i},
            ) as span:
                assert not isinstance(span, NoOpSpan)
                seen.append(dict(get_attributes_from_context()).get("session.id"))

        # Every request carried the same session.id; none leaked after exit.
        assert seen == [session_id, session_id, session_id]
        assert "session.id" not in dict(get_attributes_from_context())
