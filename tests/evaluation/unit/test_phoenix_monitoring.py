"""
Unit tests for Phoenix monitoring module.
"""

import threading
from unittest.mock import Mock, patch

import pytest

from cogniverse_telemetry_phoenix.evaluation.monitoring import (
    AlertThresholds,
    MetricWindow,
    RetrievalMonitor,
)
from tests.utils.async_polling import wait_for_phoenix_processing


class TestAlertThresholds:
    """Test alert thresholds configuration."""

    @pytest.mark.unit
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = AlertThresholds()

        assert thresholds.latency_p95_ms == 1000.0
        assert thresholds.error_rate == 0.05
        assert thresholds.mrr_drop == 0.1
        assert thresholds.throughput_drop == 0.3

    @pytest.mark.unit
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = AlertThresholds(
            latency_p95_ms=500.0, error_rate=0.01, mrr_drop=0.05, throughput_drop=0.2
        )

        assert thresholds.latency_p95_ms == 500.0
        assert thresholds.error_rate == 0.01
        assert thresholds.mrr_drop == 0.05
        assert thresholds.throughput_drop == 0.2


class TestMetricWindow:
    """Test metric window for aggregation."""

    @pytest.mark.unit
    def test_default_window(self):
        """Test default window initialization."""
        window = MetricWindow()

        assert window.window_size == 100
        assert len(window.values) == 0
        assert window.values.maxlen == 100

    @pytest.mark.unit
    def test_custom_window_size(self):
        """A custom window_size must size the underlying deque."""
        window = MetricWindow(window_size=50)

        assert window.window_size == 50
        assert window.values.maxlen == 50

    @pytest.mark.unit
    def test_add_values_respects_window_size(self):
        """The window must retain at most window_size most-recent values."""
        window = MetricWindow(window_size=5)

        for i in range(10):
            window.add(float(i))

        assert len(window.values) == 5
        assert list(window.values) == [5.0, 6.0, 7.0, 8.0, 9.0]

    @pytest.mark.unit
    def test_get_mean_empty(self):
        """Test mean calculation with empty window."""
        window = MetricWindow()

        assert window.get_mean() == 0.0

    @pytest.mark.unit
    def test_get_mean_with_values(self):
        """Test mean calculation with values."""
        window = MetricWindow()

        # Add values
        window.add(10.0)
        window.add(20.0)
        window.add(30.0)

        assert window.get_mean() == 20.0

    @pytest.mark.unit
    def test_get_p95_empty(self):
        """Test 95th percentile with empty window."""
        window = MetricWindow()

        assert window.get_p95() == 0.0

    @pytest.mark.unit
    def test_get_p95_with_values(self):
        """Test 95th percentile calculation."""
        window = MetricWindow()

        # Add 100 values (0-99)
        for i in range(100):
            window.add(float(i))

        # P95 should be around 95
        p95 = window.get_p95()
        assert 94 <= p95 <= 96

    @pytest.mark.unit
    def test_get_p95_small_sample(self):
        """Test 95th percentile with small sample."""
        window = MetricWindow()

        # Add 5 values
        for i in [1, 2, 3, 4, 5]:
            window.add(float(i))

        # P95 of [1,2,3,4,5] should be 5
        assert window.get_p95() == 5.0


class TestRetrievalMonitor:
    """Test retrieval monitoring functionality."""

    @pytest.fixture
    def mock_phoenix(self):
        """Mock Phoenix client."""
        with patch("cogniverse_telemetry_phoenix.evaluation.monitoring.px") as mock_px:
            mock_client = Mock()
            mock_px.Client.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def monitor(self, mock_phoenix):
        """Create monitor instance."""
        return RetrievalMonitor()

    @pytest.mark.unit
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = RetrievalMonitor()

        assert monitor.alert_thresholds is not None
        assert isinstance(monitor.alert_thresholds, AlertThresholds)
        assert monitor.phoenix_session is None
        assert monitor.latency_windows == {}
        assert monitor.error_windows == {}
        assert monitor.mrr_windows == {}
        assert monitor.active_alerts == {}
        assert list(monitor.metrics_buffer) == []
        assert monitor.metrics_buffer.maxlen == 10000
        assert monitor.monitoring_thread is None

    @pytest.mark.unit
    def test_start_monitoring(self):
        """Test starting monitoring."""
        with patch("cogniverse_telemetry_phoenix.evaluation.monitoring.px") as mock_px:
            mock_px.launch_app.return_value = Mock(url="http://localhost:6006")

            monitor = RetrievalMonitor()
            with patch.object(monitor, "_monitoring_loop"):
                monitor.start()

                assert monitor.phoenix_session is not None
                assert monitor.monitoring_thread is not None
                # Thread might not be alive immediately in test environment
                # Just check it was created
                assert isinstance(monitor.monitoring_thread, threading.Thread)

                # Stop monitoring
                monitor.stop()

    @pytest.mark.unit
    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        with patch("cogniverse_telemetry_phoenix.evaluation.monitoring.px") as mock_px:
            mock_px.launch_app.return_value = Mock()

            monitor = RetrievalMonitor()
            with patch.object(monitor, "_monitoring_loop"):
                monitor.start()
                wait_for_phoenix_processing(
                    delay=0.1, description="Phoenix processing"
                )  # Let thread start

                monitor.stop()

                assert monitor.stop_monitoring.is_set()
                # Thread should stop
                monitor.monitoring_thread.join(timeout=2)
                assert not monitor.monitoring_thread.is_alive()

    @pytest.mark.unit
    def test_log_retrieval_event(self):
        """Test logging retrieval events."""
        monitor = RetrievalMonitor()

        event = {
            "profile": "test_profile",
            "strategy": "test_strategy",
            "latency_ms": 100.0,
            "mrr": 0.8,
            "error": False,
            "query": "test query",
            "num_results": 10,
        }
        monitor.log_retrieval_event(event)

        # Check metrics were buffered
        assert len(monitor.metrics_buffer) == 1
        assert monitor.metrics_buffer[0]["profile"] == "test_profile"
        assert monitor.metrics_buffer[0]["latency_ms"] == 100.0
        assert monitor.metrics_buffer[0]["mrr"] == 0.8

    @pytest.mark.unit
    def test_log_retrieval_event_emits_span(self):
        """The event is emitted as a real OpenTelemetry span, captured here
        via an in-memory exporter."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with patch(
            "cogniverse_telemetry_phoenix.evaluation.monitoring._otel_trace.get_tracer",
            return_value=tracer,
        ):
            RetrievalMonitor().log_retrieval_event(
                {
                    "profile": "video_colpali",
                    "strategy": "binary_binary",
                    "latency_ms": 123.0,
                    "mrr": 0.75,
                    "num_results": 8,
                    "error": False,
                    "query": "robots playing soccer",
                }
            )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "retrieval"
        assert span.attributes["profile"] == "video_colpali"
        assert span.attributes["strategy"] == "binary_binary"
        assert span.attributes["query"] == "robots playing soccer"
        assert span.attributes["latency_ms"] == 123.0
        assert span.attributes["mrr"] == 0.75
        assert span.attributes["num_results"] == 8
        assert span.attributes["error"] is False

    @pytest.mark.unit
    def test_check_alerts_latency(self):
        """Test latency alert detection."""
        monitor = RetrievalMonitor()

        # Add latency values above threshold
        profile_key = "test_profile_test_strategy"
        monitor.latency_windows[profile_key] = MetricWindow()
        for _ in range(10):
            monitor.latency_windows[profile_key].add(1500.0)  # Above threshold

        # Check that high latency is detected
        assert (
            monitor.latency_windows[profile_key].get_p95()
            > monitor.alert_thresholds.latency_p95_ms
        )

    @pytest.mark.unit
    def test_check_alerts_error_rate(self):
        """Test error rate alert detection."""
        monitor = RetrievalMonitor()

        # Add error values above threshold
        profile_key = "test_profile_test_strategy"
        monitor.error_windows[profile_key] = MetricWindow()
        # Add mostly errors (80% error rate)
        for i in range(10):
            monitor.error_windows[profile_key].add(1.0 if i < 8 else 0.0)

        # Check that high error rate is detected
        error_rate = monitor.error_windows[profile_key].get_mean()
        assert error_rate > monitor.alert_thresholds.error_rate

    @pytest.mark.unit
    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        monitor = RetrievalMonitor()

        # Add metrics for different profiles
        profile_key = "profile1_strategy1"
        monitor.latency_windows[profile_key] = MetricWindow()
        monitor.latency_windows[profile_key].add(100.0)
        monitor.latency_windows[profile_key].add(200.0)

        monitor.error_windows[profile_key] = MetricWindow()
        monitor.error_windows[profile_key].add(0.0)

        monitor.mrr_windows[profile_key] = MetricWindow()
        monitor.mrr_windows[profile_key].add(0.8)
        monitor.mrr_windows[profile_key].add(0.9)

        summary = monitor.get_metrics_summary()

        assert profile_key in summary
        assert "latency_mean" in summary[profile_key]
        assert "latency_p95" in summary[profile_key]
        assert "error_rate" in summary[profile_key]
        assert "mrr_mean" in summary[profile_key]
