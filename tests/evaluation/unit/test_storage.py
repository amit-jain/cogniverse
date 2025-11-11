"""
Comprehensive unit tests for Phoenix storage.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from cogniverse_evaluation.data.storage import (
    ConnectionConfig,
    ConnectionState,
    ExportMetrics,
    MonitoredSpanExporter,
    TelemetryStorage,
)
from opentelemetry.sdk.trace.export import SpanExportResult


class TestConnectionConfig:
    """Test connection configuration."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = ConnectionConfig()
        assert config.http_endpoint == "http://localhost:6006"
        assert config.otlp_endpoint == "localhost:4317"
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.enable_health_checks is True

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = ConnectionConfig(
            http_endpoint="http://custom:8080",
            otlp_endpoint="custom:4317",
            max_retries=5,
            enable_health_checks=False
        )
        assert config.http_endpoint == "http://custom:8080"
        assert config.otlp_endpoint == "custom:4317"
        assert config.max_retries == 5
        assert config.enable_health_checks is False


class TestExportMetrics:
    """Test export metrics tracking."""

    @pytest.mark.unit
    def test_record_success(self):
        """Test recording successful export."""
        metrics = ExportMetrics()
        metrics.record_success(10, 150.5)

        assert metrics.total_spans_sent == 10
        assert metrics.total_spans_failed == 0
        assert metrics.last_successful_export is not None
        assert len(metrics.export_latencies) == 1
        assert metrics.export_latencies[0] == 150.5

    @pytest.mark.unit
    def test_record_failure(self):
        """Test recording failed export."""
        metrics = ExportMetrics()
        metrics.record_failure(5)

        assert metrics.total_spans_sent == 0
        assert metrics.total_spans_failed == 5
        assert metrics.total_export_errors == 1
        assert metrics.last_failed_export is not None

    @pytest.mark.unit
    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = ExportMetrics()

        # No data
        assert metrics.get_success_rate() == 0.0

        # Some successes and failures
        metrics.record_success(8, 100)
        metrics.record_failure(2)
        assert metrics.get_success_rate() == 0.8

    @pytest.mark.unit
    def test_average_latency(self):
        """Test average latency calculation."""
        metrics = ExportMetrics()

        # No data
        assert metrics.get_avg_latency() == 0.0

        # Add some latencies
        metrics.record_success(1, 100)
        metrics.record_success(1, 200)
        metrics.record_success(1, 150)
        assert metrics.get_avg_latency() == 150.0

    @pytest.mark.unit
    def test_latency_buffer_limit(self):
        """Test that latency buffer respects max size."""
        metrics = ExportMetrics()

        # Add more than max size (100)
        for i in range(150):
            metrics.record_success(1, float(i))

        # Should only keep last 100
        assert len(metrics.export_latencies) == 100
        # Should have values 50-149
        assert metrics.export_latencies[0] == 50.0
        assert metrics.export_latencies[-1] == 149.0


class TestMonitoredSpanExporter:
    """Test monitored span exporter with retry logic."""

    @pytest.fixture
    def mock_exporter(self):
        """Create mock OTLP exporter."""
        exporter = Mock()
        exporter.export = Mock(return_value=SpanExportResult.SUCCESS)
        exporter.shutdown = Mock()
        return exporter

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ConnectionConfig(
            max_retries=3,
            retry_delay_seconds=0.01,  # Short delay for tests
            retry_backoff_factor=2.0,
        )

    @pytest.mark.unit
    def test_successful_export(self, mock_exporter, config):
        """Test successful span export."""
        metrics = ExportMetrics()
        monitored = MonitoredSpanExporter(mock_exporter, config, metrics)

        spans = [Mock(), Mock(), Mock()]
        result = monitored.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert metrics.total_spans_sent == 3
        assert metrics.total_spans_failed == 0
        mock_exporter.export.assert_called_once_with(spans)

    @pytest.mark.unit
    def test_export_with_retry(self, mock_exporter, config):
        """Test export with retry on failure."""
        metrics = ExportMetrics()
        monitored = MonitoredSpanExporter(mock_exporter, config, metrics)

        # Fail twice, then succeed
        mock_exporter.export.side_effect = [
            SpanExportResult.FAILURE,
            SpanExportResult.FAILURE,
            SpanExportResult.SUCCESS,
        ]

        spans = [Mock(), Mock()]
        result = monitored.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert mock_exporter.export.call_count == 3
        assert metrics.total_spans_sent == 2
        assert metrics.total_spans_failed == 0

    @pytest.mark.unit
    def test_export_all_retries_fail(self, mock_exporter, config):
        """Test export when all retries fail."""
        metrics = ExportMetrics()
        monitored = MonitoredSpanExporter(mock_exporter, config, metrics)

        # Always fail
        mock_exporter.export.return_value = SpanExportResult.FAILURE

        spans = [Mock(), Mock()]
        result = monitored.export(spans)

        assert result == SpanExportResult.FAILURE
        assert mock_exporter.export.call_count == 3  # max_retries
        assert metrics.total_spans_sent == 0
        assert metrics.total_spans_failed == 2
        assert metrics.total_export_errors == 1

    @pytest.mark.unit
    def test_export_with_exception(self, mock_exporter, config):
        """Test export with exception handling."""
        metrics = ExportMetrics()
        monitored = MonitoredSpanExporter(mock_exporter, config, metrics)

        # Raise exception twice, then succeed
        mock_exporter.export.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            SpanExportResult.SUCCESS,
        ]

        spans = [Mock()]
        result = monitored.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert mock_exporter.export.call_count == 3
        assert metrics.total_spans_sent == 1

    @pytest.mark.unit
    def test_shutdown(self, mock_exporter, config):
        """Test exporter shutdown."""
        metrics = ExportMetrics()
        monitored = MonitoredSpanExporter(mock_exporter, config, metrics)

        monitored.shutdown()
        mock_exporter.shutdown.assert_called_once()


class TestTelemetryStorage:
    """Test production telemetry storage."""

    @pytest.fixture
    def mock_phoenix_client(self):
        """Create mock Phoenix client (deprecated - use mock_provider)."""
        client = Mock()
        client.get_spans_dataframe = Mock(return_value=pd.DataFrame())
        client.upload_dataset = Mock(return_value=Mock(id="test_id"))
        client.get_dataset = Mock(return_value=Mock(id="test_id", name="test"))
        return client

    @pytest.fixture
    def mock_provider(self, mock_phoenix_client):
        """Create mock evaluator provider."""
        from unittest.mock import AsyncMock
        provider = Mock()
        # Create telemetry provider with traces interface
        provider.telemetry.traces.get_spans = AsyncMock(return_value=mock_phoenix_client.get_spans_dataframe())
        return provider

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ConnectionConfig(
            max_retries=2,
            retry_delay_seconds=0.01,
            health_check_interval_seconds=0.1,
            enable_health_checks=False,  # Disable for most tests
        )

    @pytest.mark.unit
    def test_initialization_success(self, mock_provider, config):
        """Test successful initialization."""
        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                assert storage.connection_state == ConnectionState.CONNECTED
                assert storage.provider is not None
                assert storage.tracer is not None

    @pytest.mark.unit
    def test_initialization_with_retry(self, config):
        """Test initialization with connection retry."""
        from unittest.mock import AsyncMock
        mock_provider = Mock()
        # Fail once, then succeed
        mock_provider.telemetry.traces.get_spans = AsyncMock(side_effect=[
            Exception("Connection failed"),
            pd.DataFrame(),
        ])

        with patch("cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                assert storage.connection_state == ConnectionState.CONNECTED
                assert mock_provider.telemetry.traces.get_spans.call_count == 2

    @pytest.mark.unit
    def test_initialization_failure(self, config):
        """Test initialization failure after retries."""
        from unittest.mock import AsyncMock
        mock_provider = Mock()
        mock_provider.telemetry.traces.get_spans = AsyncMock(side_effect=Exception("Connection failed"))

        with patch("cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider):
            with pytest.raises(ConnectionError) as exc_info:
                TelemetryStorage(config)

            assert "Failed to connect to telemetry provider" in str(exc_info.value)

    @pytest.mark.unit
    def test_log_experiment_results(self, mock_provider, config):
        """Test logging experiment results."""
        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                # Mock tracer
                mock_span = MagicMock()
                mock_span.get_span_context.return_value.trace_id = 123456
                storage.tracer = Mock()
                storage.tracer.start_as_current_span.return_value.__enter__ = Mock(
                    return_value=mock_span
                )
                storage.tracer.start_as_current_span.return_value.__exit__ = Mock(
                    return_value=None
                )

                experiment_id = storage.log_experiment_results(
                    experiment_name="test_exp",
                    profile="profile1",
                    strategy="strategy1",
                    results=[{"query": "test", "score": 0.8}],
                    metrics={"mrr": 0.75, "recall@1": 0.6},
                )

                assert experiment_id is not None
                assert "test_exp_profile1_strategy1" in experiment_id
                mock_span.set_attributes.assert_called()
                mock_span.add_event.assert_called()

    @pytest.mark.unit
    def test_log_experiment_when_disconnected(self, config):
        """Test logging when disconnected."""
        storage = TelemetryStorage.__new__(TelemetryStorage)
        storage.config = config
        storage.connection_state = ConnectionState.DISCONNECTED
        storage.metrics = ExportMetrics()

        result = storage.log_experiment_results(
            experiment_name="test", profile="p1", strategy="s1", results=[], metrics={}
        )

        assert result is None

    @pytest.mark.unit
    def test_get_traces_for_evaluation(self, mock_phoenix_client, mock_provider, config):
        """Test getting traces for evaluation."""
        from unittest.mock import AsyncMock
        test_df = pd.DataFrame(
            [
                {"trace_id": "trace1", "name": "test1"},
                {"trace_id": "trace2", "name": "test2"},
            ]
        )
        # Configure the provider's async mock to return test data
        mock_provider.telemetry.traces.get_spans = AsyncMock(return_value=test_df)

        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                # Test regular query
                df = storage.get_traces_for_evaluation(limit=10)
                assert len(df) == 2

                # Test with filter
                df = storage.get_traces_for_evaluation(
                    filter_condition="name == 'test1'", limit=10
                )
                # Verify the async mock was called with expected args
                assert mock_provider.telemetry.traces.get_spans.call_count >= 2  # init + actual calls

    @pytest.mark.unit
    def test_get_traces_with_trace_ids(self, mock_phoenix_client, mock_provider, config):
        """Test getting specific traces by ID."""
        from unittest.mock import AsyncMock
        trace1_df = pd.DataFrame([{"trace_id": "trace1", "name": "test1"}])
        trace2_df = pd.DataFrame([{"trace_id": "trace2", "name": "test2"}])

        # First call is for init connection test, then two for actual traces
        mock_provider.telemetry.traces.get_spans = AsyncMock(side_effect=[
            pd.DataFrame(),  # For init connection test
            trace1_df,
            trace2_df,
        ])

        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                df = storage.get_traces_for_evaluation(trace_ids=["trace1", "trace2"])

                assert len(df) == 2
                assert mock_provider.telemetry.traces.get_spans.call_count == 3  # 1 for init + 2 for traces

    @pytest.mark.unit
    def test_get_traces_when_disconnected(self, config):
        """Test get_traces_for_evaluation when disconnected."""
        storage = TelemetryStorage.__new__(TelemetryStorage)
        storage.config = config
        storage.connection_state = ConnectionState.DISCONNECTED
        storage.metrics = ExportMetrics()

        result = storage.get_traces_for_evaluation()

        assert result.empty

    @pytest.mark.unit
    def test_get_traces_with_error(self, mock_phoenix_client, mock_provider, config):
        """Test get_traces_for_evaluation error handling."""
        mock_phoenix_client.get_spans_dataframe.side_effect = [
            pd.DataFrame(),  # For init
            Exception("Query failed"),  # For actual query
        ]

        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                result = storage.get_traces_for_evaluation()

                assert result.empty

    @pytest.mark.unit
    def test_get_metrics(self, mock_provider, config):
        """Test getting storage metrics."""
        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                # Add some metrics
                storage.metrics.record_success(10, 100)
                storage.metrics.record_failure(2)

                metrics = storage.get_metrics()

                assert metrics["connection_state"] == "connected"
                assert metrics["total_spans_sent"] == 10
                assert metrics["total_spans_failed"] == 2
                assert metrics["success_rate"] == 10 / 12
                assert metrics["avg_latency_ms"] == 100.0

    @pytest.mark.unit
    def test_context_manager(self, mock_provider, config):
        """Test context manager functionality."""
        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                with TelemetryStorage(config) as storage:
                    assert storage.connection_state == ConnectionState.CONNECTED

                # After context exit
                assert storage.connection_state == ConnectionState.DISCONNECTED

    @pytest.mark.unit
    def test_health_check_error_handling(self, mock_phoenix_client, mock_provider, config):
        """Test health check handles errors gracefully."""
        from unittest.mock import AsyncMock
        # First successful init, then simulate connection loss
        mock_provider.telemetry.traces.get_spans = AsyncMock(side_effect=[
            pd.DataFrame(),  # For init
            Exception("Connection lost"),  # For health check
        ])

        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)
                assert storage.connection_state == ConnectionState.CONNECTED

                # Manually trigger health check which will fail
                storage._perform_health_check()

                # Should mark as disconnected
                assert storage.connection_state == ConnectionState.DISCONNECTED

                storage.shutdown()

    @pytest.mark.unit
    def test_health_check_reconnection_failure(self):
        """Test health check when reconnection fails."""
        config = ConnectionConfig(enable_health_checks=False)

        from unittest.mock import AsyncMock
        mock_provider = Mock()
        mock_provider.telemetry.traces.get_spans = AsyncMock(return_value=pd.DataFrame())

        with patch("cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                # Simulate disconnection
                storage.connection_state = ConnectionState.DISCONNECTED

                # Make reconnection fail
                with patch.object(
                    storage,
                    "_initialize_connection",
                    side_effect=Exception("Cannot reconnect"),
                ):
                    storage._perform_health_check()

                    # Should still be disconnected
                    assert storage.connection_state == ConnectionState.DISCONNECTED

    @pytest.mark.unit
    def test_shutdown(self, mock_provider, config):
        """Test graceful shutdown."""
        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                storage.shutdown()

                assert storage.connection_state == ConnectionState.DISCONNECTED
                assert storage.provider is None

    @pytest.mark.unit
    def test_span_creation_without_tracer(self, mock_provider, config):
        """Test _create_span when tracer is None."""
        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)
                storage.tracer = None

                with storage._create_span("test_span") as span:
                    assert span is None

    @pytest.mark.unit
    def test_span_creation_with_exception(self, mock_provider, config):
        """Test _create_span exception handling."""
        with patch(
            "cogniverse_core.evaluation.providers.get_evaluator_provider", return_value=mock_provider
        ):
            with patch("cogniverse_core.evaluation.data.storage.trace"):
                storage = TelemetryStorage(config)

                mock_span = MagicMock()
                mock_span.record_exception = Mock()
                mock_span.set_status = Mock()

                storage.tracer = Mock()
                storage.tracer.start_as_current_span.return_value.__enter__ = Mock(
                    return_value=mock_span
                )
                storage.tracer.start_as_current_span.return_value.__exit__ = Mock(
                    return_value=None
                )

                with pytest.raises(ValueError):
                    with storage._create_span("test_span"):
                        raise ValueError("Test error")

                mock_span.record_exception.assert_called()
