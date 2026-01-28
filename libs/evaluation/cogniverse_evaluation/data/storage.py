"""
Production-ready telemetry storage implementation with comprehensive error handling,
retry logic, health checks, and monitoring.
"""

import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

# Provider import moved to initialization method
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from cogniverse_core.common.utils.async_polling import wait_for_retry_backoff

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Telemetry connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class ConnectionConfig:
    """Configuration for telemetry connection."""

    http_endpoint: str = "http://localhost:6006"
    otlp_endpoint: str = "localhost:4317"
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    connection_timeout_seconds: float = 10.0
    health_check_interval_seconds: float = 30.0
    max_batch_size: int = 512
    export_timeout_millis: int = 30000
    enable_metrics: bool = True
    enable_health_checks: bool = True


@dataclass
class ExportMetrics:
    """Metrics for span export monitoring."""

    total_spans_sent: int = 0
    total_spans_failed: int = 0
    total_export_errors: int = 0
    last_successful_export: Optional[datetime] = None
    last_failed_export: Optional[datetime] = None
    export_latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_success(self, num_spans: int, latency_ms: float):
        """Record successful export."""
        self.total_spans_sent += num_spans
        self.last_successful_export = datetime.now()
        self.export_latencies.append(latency_ms)

    def record_failure(self, num_spans: int):
        """Record failed export."""
        self.total_spans_failed += num_spans
        self.total_export_errors += 1
        self.last_failed_export = datetime.now()

    def get_success_rate(self) -> float:
        """Get export success rate."""
        total = self.total_spans_sent + self.total_spans_failed
        if total == 0:
            return 0.0
        return self.total_spans_sent / total

    def get_avg_latency(self) -> float:
        """Get average export latency."""
        if not self.export_latencies:
            return 0.0
        return sum(self.export_latencies) / len(self.export_latencies)


class MonitoredSpanExporter(SpanExporter):
    """Wrapper around OTLP exporter with monitoring and retry logic."""

    def __init__(
        self,
        exporter: OTLPSpanExporter,
        config: ConnectionConfig,
        metrics: ExportMetrics,
    ):
        self.exporter = exporter
        self.config = config
        self.metrics = metrics

    def export(self, spans) -> SpanExportResult:
        """Export spans with retry logic and monitoring."""
        start_time = time.time()
        num_spans = len(spans)

        for attempt in range(self.config.max_retries):
            try:
                result = self.exporter.export(spans)

                if result == SpanExportResult.SUCCESS:
                    latency_ms = (time.time() - start_time) * 1000
                    self.metrics.record_success(num_spans, latency_ms)
                    return result

                # Export failed, retry with backoff
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_backoff_factor**attempt
                    )
                    logger.warning(
                        f"Span export failed, retrying in {delay}s (attempt {attempt + 1})"
                    )
                    wait_for_retry_backoff(
                        attempt,
                        base_delay=self.config.retry_delay_seconds,
                        max_delay=60.0,
                        description="span export retry",
                    )

            except Exception as e:
                logger.error(f"Error exporting spans (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_backoff_factor**attempt
                    )
                    wait_for_retry_backoff(
                        attempt,
                        base_delay=self.config.retry_delay_seconds,
                        max_delay=60.0,
                        description="span export error retry",
                    )

        # All retries failed
        self.metrics.record_failure(num_spans)
        logger.error(
            f"Failed to export {num_spans} spans after {self.config.max_retries} attempts"
        )
        return SpanExportResult.FAILURE

    def shutdown(self):
        """Shutdown exporter."""
        self.exporter.shutdown()


class TelemetryStorage:
    """
    Production-ready telemetry storage with comprehensive error handling,
    monitoring, and reliability features.
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        """Initialize telemetry storage with production configuration."""
        # Default to no health checks for tests
        self.config = config or ConnectionConfig(enable_health_checks=False)
        self.metrics = ExportMetrics()
        self.connection_state = ConnectionState.DISCONNECTED
        self.provider = None  # Will be initialized in _initialize_connection
        self.tracer: Optional[trace.Tracer] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        self._lock = threading.Lock()

        # Initialize connection
        self._initialize_connection()

        # Start health checks if enabled
        if self.config.enable_health_checks:
            self._start_health_checks()

    def _initialize_connection(self):
        """Initialize telemetry provider and OpenTelemetry with retry logic."""
        with self._lock:
            self.connection_state = ConnectionState.CONNECTING

            for attempt in range(self.config.max_retries):
                try:
                    # Register project config with telemetry manager
                    import asyncio

                    from cogniverse_evaluation.providers import (
                        get_evaluation_provider,
                        reset_evaluation_provider,
                    )
                    from cogniverse_foundation.telemetry.manager import (
                        get_telemetry_manager,
                    )

                    # Reset any cached provider first
                    reset_evaluation_provider()

                    # Register project with telemetry endpoints
                    telemetry_manager = get_telemetry_manager()
                    telemetry_manager.register_project(
                        tenant_id="default",
                        project_name="evaluation",
                        http_endpoint=self.config.http_endpoint,
                        grpc_endpoint=self.config.otlp_endpoint,
                    )

                    # Initialize evaluator provider (will use registered config)
                    self.provider = get_evaluation_provider(
                        tenant_id="default",
                        config={
                            "project_name": "evaluation",
                            "http_endpoint": self.config.http_endpoint,
                            "grpc_endpoint": self.config.otlp_endpoint,
                        },
                    )

                    # Test connection
                    asyncio.run(
                        self.provider.telemetry.traces.get_spans(
                            project="cogniverse-default", limit=1
                        )
                    )

                    # Configure OpenTelemetry
                    self._configure_opentelemetry()

                    self.connection_state = ConnectionState.CONNECTED
                    logger.info("Successfully connected to telemetry provider")
                    return

                except Exception as e:
                    logger.error(
                        f"Failed to connect to telemetry provider (attempt {attempt + 1}): {e}"
                    )
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay_seconds * (
                            self.config.retry_backoff_factor**attempt
                        )
                        wait_for_retry_backoff(
                            attempt,
                            base_delay=self.config.retry_delay_seconds,
                            max_delay=60.0,
                            description="Telemetry connection retry",
                        )

            self.connection_state = ConnectionState.FAILED
            raise ConnectionError(
                f"Failed to connect to telemetry provider after {self.config.max_retries} attempts"
            )

    def _configure_opentelemetry(self):
        """Configure OpenTelemetry with monitoring wrapper."""
        # Check if already configured
        current_provider = trace.get_tracer_provider()
        if not hasattr(current_provider, "_resource"):
            resource = Resource.create(
                {
                    "service.name": "evaluation",
                    "service.version": "1.0.0",
                    "deployment.environment": os.getenv("ENVIRONMENT", "production"),
                }
            )

            provider = TracerProvider(resource=resource)

            # Create monitored exporter
            base_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                timeout=self.config.export_timeout_millis,
                headers={"content-type": "application/json"},
            )

            monitored_exporter = MonitoredSpanExporter(
                base_exporter, self.config, self.metrics
            )

            processor = BatchSpanProcessor(
                monitored_exporter,
                max_queue_size=2048,
                max_export_batch_size=self.config.max_batch_size,
                export_timeout_millis=self.config.export_timeout_millis,
            )

            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

            logger.info(
                f"Configured OpenTelemetry with monitored exporter to {self.config.otlp_endpoint}"
            )

        self.tracer = trace.get_tracer(__name__, "1.0.0")

    def _start_health_checks(self):
        """Start background health check thread."""
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True, name="phoenix-health-check"
        )
        self._health_check_thread.start()
        logger.info("Started Telemetry health check thread")

    def _health_check_loop(self):
        """Background loop for health checks."""
        while not self._stop_health_check.is_set():
            try:
                self._perform_health_check()
            except Exception as e:
                logger.error(f"Error in health check: {e}")

            self._stop_health_check.wait(self.config.health_check_interval_seconds)

    def _perform_health_check(self):
        """Perform health check and reconnect if needed."""
        with self._lock:
            if self.connection_state != ConnectionState.CONNECTED:
                logger.info("Telemetry disconnected, attempting reconnection...")
                try:
                    self._initialize_connection()
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
                return

            # Test connection
            try:
                if self.provider:
                    import asyncio

                    # Handle nested event loop
                    try:
                        loop = asyncio.get_running_loop()
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            _ = pool.submit(
                                lambda: asyncio.run(
                                    self.provider.telemetry.traces.get_spans(
                                        project="cogniverse-default", limit=1
                                    )
                                )
                            ).result()
                    except RuntimeError:
                        _ = asyncio.run(
                            self.provider.telemetry.traces.get_spans(
                                project="cogniverse-default", limit=1
                            )
                        )
                    # Connection is healthy
                    return
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                self.connection_state = ConnectionState.DISCONNECTED

    @contextmanager
    def _create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a span with error handling."""
        if not self.tracer:
            logger.warning(f"Tracer not initialized, skipping span creation for {name}")
            yield None
            return

        span = self.tracer.start_as_current_span(name=name, kind=trace.SpanKind.CLIENT)

        with span as current_span:
            try:
                if attributes:
                    current_span.set_attributes(attributes)
                yield current_span
                current_span.set_status(Status(StatusCode.OK))
            except Exception as e:
                current_span.record_exception(e)
                current_span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def log_experiment_results(
        self,
        experiment_name: str,
        profile: str,
        strategy: str,
        results: List[Dict[str, Any]],
        metrics: Dict[str, float],
    ) -> Optional[str]:
        """
        Log experiment results with comprehensive error handling.

        Returns:
            Experiment ID if successful, None otherwise
        """
        if self.connection_state != ConnectionState.CONNECTED:
            logger.error("Cannot log experiment results: Telemetry not connected")
            return None

        try:
            with self._create_span(
                "experiment_results",
                attributes={
                    "experiment_name": experiment_name,
                    "profile": profile,
                    "strategy": strategy,
                    "num_results": len(results),
                },
            ) as span:
                # Format results for dashboard
                formatted_results = {
                    "profile": profile,
                    "strategy": strategy,
                    "aggregate_metrics": {
                        "mrr": {"mean": metrics.get("mrr", 0.0)},
                        "recall@1": {"mean": metrics.get("recall@1", 0.0)},
                        "recall@5": {"mean": metrics.get("recall@5", 0.0)},
                    },
                    "queries": results,
                    "timestamp": datetime.now().isoformat(),
                }

                # Add detailed metrics as events
                if span:
                    for metric_name, metric_value in metrics.items():
                        span.add_event(
                            f"metric_{metric_name}",
                            attributes={"value": str(metric_value)},
                        )

                    # Get trace ID for reference
                    trace_id = format(span.get_span_context().trace_id, "032x")
                    span.set_attribute("trace_id", trace_id)

                experiment_id = f"{experiment_name}_{profile}_{strategy}_{datetime.now().timestamp()}"

                logger.info(f"Logged experiment {experiment_id} to telemetry provider")
                return experiment_id

        except Exception as e:
            logger.error(f"Failed to log experiment results: {e}")
            return None

    def get_traces_for_evaluation(
        self,
        trace_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        filter_condition: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get traces with proper error handling and fallback.
        """
        if self.connection_state != ConnectionState.CONNECTED:
            logger.warning("Telemetry not connected, returning empty DataFrame")
            return pd.DataFrame()

        import asyncio
        import concurrent.futures

        def _fetch_spans(**kwargs):
            """Helper to fetch spans with async handling."""
            try:
                loop = asyncio.get_running_loop()
                # If we're here, loop is running - run in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        lambda: asyncio.run(
                            self.provider.telemetry.traces.get_spans(**kwargs)
                        )
                    ).result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self.provider.telemetry.traces.get_spans(**kwargs))

        try:
            if trace_ids:
                # Provider doesn't support bulk trace ID queries well
                # Fetch individually and combine
                dfs = []
                for trace_id in trace_ids[:limit]:  # Limit to prevent overload
                    try:
                        df = _fetch_spans(
                            project="cogniverse-default",
                            filter_condition=f"trace_id == '{trace_id}'",
                            root_spans_only=True,
                            limit=1,
                        )
                        if not df.empty:
                            dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to fetch trace {trace_id}: {e}")

                if dfs:
                    return pd.concat(dfs, ignore_index=True)
                return pd.DataFrame()

            # Regular query
            return _fetch_spans(
                project="cogniverse-default",
                filter_condition=filter_condition,
                start_time=start_time,
                root_spans_only=True,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            return pd.DataFrame()

    def get_metrics(self) -> Dict[str, Any]:
        """Get storage metrics for monitoring."""
        return {
            "connection_state": self.connection_state.value,
            "total_spans_sent": self.metrics.total_spans_sent,
            "total_spans_failed": self.metrics.total_spans_failed,
            "success_rate": self.metrics.get_success_rate(),
            "avg_latency_ms": self.metrics.get_avg_latency(),
            "last_successful_export": (
                self.metrics.last_successful_export.isoformat()
                if self.metrics.last_successful_export
                else None
            ),
            "last_failed_export": (
                self.metrics.last_failed_export.isoformat()
                if self.metrics.last_failed_export
                else None
            ),
        }

    def shutdown(self):
        """Gracefully shutdown storage."""
        logger.info("Shutting down Telemetry storage...")

        # Stop health checks
        if self._health_check_thread:
            self._stop_health_check.set()
            self._health_check_thread.join(timeout=5)

        # Clear provider reference
        if self.provider:
            self.provider = None

        self.connection_state = ConnectionState.DISCONNECTED
        logger.info("Telemetry storage shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
