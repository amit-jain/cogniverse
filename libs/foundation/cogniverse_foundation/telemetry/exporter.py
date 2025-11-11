"""
Batch span exporter with queue management and drop policies.
"""

import logging
import time
from threading import Lock
from typing import Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Span

from .config import BatchExportConfig

logger = logging.getLogger(__name__)


class QueueManagedBatchExporter(SpanExporter):
    """
    Batch exporter that drops spans when queue is full instead of blocking.

    This prevents telemetry from impacting application performance when
    the export destination is slow or unavailable.
    """

    def __init__(self, wrapped_exporter: SpanExporter, config: BatchExportConfig):
        self.wrapped_exporter = wrapped_exporter
        self.config = config

        # Drop tracking
        self._dropped_spans_count = 0
        self._last_drop_log_time = 0
        self._drop_log_count_since_last = 0
        self._lock = Lock()

        logger.info(f"Initialized QueueManagedBatchExporter with config: {config}")

    def export(self, spans: Sequence[Span]) -> SpanExportResult:
        """
        Export spans with queue management.

        Args:
            spans: Spans to export

        Returns:
            SpanExportResult indicating success/failure
        """
        if not spans:
            return SpanExportResult.SUCCESS

        # Check if we should drop spans due to queue pressure
        if self._should_drop_spans():
            self._handle_dropped_spans(len(spans))
            return SpanExportResult.SUCCESS  # Don't fail the batch processor

        try:
            # Delegate to wrapped exporter
            result = self.wrapped_exporter.export(spans)

            # Reset drop tracking on successful export
            if result == SpanExportResult.SUCCESS:
                with self._lock:
                    self._dropped_spans_count = 0

            return result

        except Exception as e:
            logger.warning(f"Export failed: {e}")
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Force flush wrapped exporter."""
        try:
            return self.wrapped_exporter.force_flush(timeout_millis)
        except Exception as e:
            logger.warning(f"Force flush failed: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown wrapped exporter."""
        try:
            self.wrapped_exporter.shutdown()
        except Exception as e:
            logger.warning(f"Shutdown failed: {e}")

    def _should_drop_spans(self) -> bool:
        """
        Determine if spans should be dropped.

        This is a simplified implementation. In production, you might:
        - Check actual queue sizes in the BatchSpanProcessor
        - Monitor export latency
        - Check memory pressure
        - Use circuit breaker patterns
        """
        # For now, simulate queue pressure by checking dropped span count
        with self._lock:
            # If we've dropped many spans recently, keep dropping
            return self._dropped_spans_count > self.config.max_queue_size * 0.8

    def _handle_dropped_spans(self, span_count: int):
        """Handle dropped spans with rate-limited logging."""
        with self._lock:
            self._dropped_spans_count += span_count
            current_time = time.time()

            # Rate limit drop logging
            if (
                self.config.log_dropped_spans
                and current_time - self._last_drop_log_time > 60
            ):  # 1 minute

                if (
                    self._drop_log_count_since_last
                    < self.config.max_drop_log_rate_per_minute
                ):
                    logger.warning(
                        f"Dropped {span_count} spans due to queue pressure. "
                        f"Total dropped: {self._dropped_spans_count}"
                    )
                    self._drop_log_count_since_last += 1

                # Reset rate limiting
                if current_time - self._last_drop_log_time > 60:
                    self._last_drop_log_time = current_time
                    self._drop_log_count_since_last = 0

    def get_stats(self) -> dict:
        """Get exporter statistics."""
        with self._lock:
            return {
                "dropped_spans_total": self._dropped_spans_count,
                "config": {
                    "max_queue_size": self.config.max_queue_size,
                    "drop_on_queue_full": self.config.drop_on_queue_full,
                    "log_dropped_spans": self.config.log_dropped_spans,
                },
            }
