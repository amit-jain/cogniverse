"""
Unit tests for ModalityMetricsTracker
"""

import time

import pytest

from cogniverse_agents.search.multi_modal_reranker import QueryModality
from cogniverse_core.telemetry.modality_metrics import ModalityMetricsTracker


class TestModalityMetricsTracker:
    """Test ModalityMetricsTracker functionality"""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance"""
        return ModalityMetricsTracker(window_size=100)

    def test_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker.window_size == 100
        for modality in QueryModality:
            assert modality in tracker.modality_latencies
            assert len(tracker.modality_latencies[modality]) == 0

    def test_record_successful_execution(self, tracker):
        """Test recording successful execution"""
        tracker.record_modality_execution(
            modality=QueryModality.VIDEO,
            latency_ms=150.5,
            success=True,
        )

        assert len(tracker.modality_latencies[QueryModality.VIDEO]) == 1
        assert tracker.modality_latencies[QueryModality.VIDEO][0] == 150.5
        assert tracker.modality_success[QueryModality.VIDEO] == 1
        assert tracker.modality_requests[QueryModality.VIDEO] == 1

    def test_record_failed_execution(self, tracker):
        """Test recording failed execution"""
        tracker.record_modality_execution(
            modality=QueryModality.VIDEO,
            latency_ms=50.0,
            success=False,
            error="Connection timeout",
        )

        assert tracker.modality_success[QueryModality.VIDEO] == 0
        assert tracker.modality_errors[QueryModality.VIDEO]["Connection timeout"] == 1
        assert tracker.modality_requests[QueryModality.VIDEO] == 1

    def test_record_multiple_executions(self, tracker):
        """Test recording multiple executions"""
        for i in range(10):
            tracker.record_modality_execution(
                modality=QueryModality.VIDEO,
                latency_ms=100.0 + i * 10,
                success=(i % 2 == 0),  # Alternate success/failure
            )

        assert len(tracker.modality_latencies[QueryModality.VIDEO]) == 10
        assert tracker.modality_success[QueryModality.VIDEO] == 5
        assert tracker.modality_requests[QueryModality.VIDEO] == 10

    def test_rolling_window_eviction(self, tracker):
        """Test rolling window maintains size limit"""
        small_tracker = ModalityMetricsTracker(window_size=5)

        # Record 10 executions
        for i in range(10):
            small_tracker.record_modality_execution(
                modality=QueryModality.VIDEO,
                latency_ms=float(i),
                success=True,
            )

        # Should only keep last 5
        latencies = small_tracker.modality_latencies[QueryModality.VIDEO]
        assert len(latencies) == 5
        assert latencies == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_get_modality_stats_empty(self, tracker):
        """Test getting stats with no data"""
        stats = tracker.get_modality_stats(QueryModality.VIDEO)

        assert stats["total_requests"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["p50_latency"] == 0.0

    def test_get_modality_stats_with_data(self, tracker):
        """Test getting stats with data"""
        # Record 10 executions
        for i in range(10):
            tracker.record_modality_execution(
                modality=QueryModality.VIDEO,
                latency_ms=100.0 + i * 10,
                success=(i < 8),  # 8 successes, 2 failures
            )

        stats = tracker.get_modality_stats(QueryModality.VIDEO)

        assert stats["modality"] == "video"
        assert stats["total_requests"] == 10
        assert stats["success_count"] == 8
        assert stats["error_count"] == 2
        assert stats["success_rate"] == 0.8
        assert 140 < stats["p50_latency"] < 150  # Median
        assert stats["p95_latency"] > stats["p50_latency"]
        assert stats["p99_latency"] >= stats["p95_latency"]
        assert stats["min_latency"] == 100.0
        assert stats["max_latency"] == 190.0

    def test_error_breakdown(self, tracker):
        """Test error breakdown tracking"""
        tracker.record_modality_execution(QueryModality.VIDEO, 100, False, "timeout")
        tracker.record_modality_execution(
            QueryModality.VIDEO, 100, False, "connection_error"
        )
        tracker.record_modality_execution(QueryModality.VIDEO, 100, False, "timeout")

        stats = tracker.get_modality_stats(QueryModality.VIDEO)

        assert stats["error_breakdown"]["timeout"] == 2
        assert stats["error_breakdown"]["connection_error"] == 1

    def test_throughput_calculation(self, tracker):
        """Test throughput (QPS) calculation"""
        # Record executions over time
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)
        time.sleep(0.1)  # 100ms delay
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)

        stats = tracker.get_modality_stats(QueryModality.VIDEO)

        # Should have throughput > 0
        assert stats["throughput_qps"] > 0

    def test_get_all_stats(self, tracker):
        """Test getting stats for all modalities"""
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)
        tracker.record_modality_execution(QueryModality.DOCUMENT, 50, True)

        all_stats = tracker.get_all_stats()

        # Should only include modalities with activity
        assert "video" in all_stats
        assert "document" in all_stats
        assert len(all_stats) == 2  # Only these two have data

    def test_get_summary_stats(self, tracker):
        """Test getting summary statistics"""
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)
        tracker.record_modality_execution(QueryModality.VIDEO, 150, False)
        tracker.record_modality_execution(QueryModality.DOCUMENT, 50, True)

        summary = tracker.get_summary_stats()

        assert summary["total_requests"] == 3
        assert summary["overall_success_rate"] == 2 / 3
        assert summary["active_modalities"] == 2
        assert "avg_latency_p95" in summary
        assert summary["modality_breakdown"]["video"] == 2
        assert summary["modality_breakdown"]["document"] == 1

    def test_get_summary_stats_empty(self, tracker):
        """Test summary stats with no data"""
        summary = tracker.get_summary_stats()

        assert summary["total_requests"] == 0
        assert summary["overall_success_rate"] == 0.0
        assert summary["active_modalities"] == 0

    def test_get_slowest_modalities(self, tracker):
        """Test getting slowest modalities"""
        # VIDEO: slow
        tracker.record_modality_execution(QueryModality.VIDEO, 500, True)
        tracker.record_modality_execution(QueryModality.VIDEO, 600, True)

        # DOCUMENT: fast
        tracker.record_modality_execution(QueryModality.DOCUMENT, 50, True)
        tracker.record_modality_execution(QueryModality.DOCUMENT, 60, True)

        # AUDIO: medium
        tracker.record_modality_execution(QueryModality.AUDIO, 200, True)

        slowest = tracker.get_slowest_modalities(top_k=2)

        assert len(slowest) == 2
        assert slowest[0]["modality"] == "video"  # Slowest
        assert slowest[0]["p95_latency"] > slowest[1]["p95_latency"]

    def test_get_error_prone_modalities(self, tracker):
        """Test getting error-prone modalities"""
        # VIDEO: high error rate (50%)
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)
        tracker.record_modality_execution(QueryModality.VIDEO, 100, False, "error1")

        # DOCUMENT: low error rate (10%)
        for i in range(9):
            tracker.record_modality_execution(QueryModality.DOCUMENT, 50, True)
        tracker.record_modality_execution(QueryModality.DOCUMENT, 50, False, "error2")

        error_prone = tracker.get_error_prone_modalities(min_error_rate=0.2)

        # Only VIDEO should be included (50% error rate)
        assert len(error_prone) == 1
        assert error_prone[0]["modality"] == "video"
        assert error_prone[0]["error_rate"] == 0.5
        assert "error1" in error_prone[0]["error_breakdown"]

    def test_reset_modality_stats(self, tracker):
        """Test resetting specific modality stats"""
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)
        tracker.record_modality_execution(QueryModality.DOCUMENT, 50, True)

        tracker.reset_modality_stats(QueryModality.VIDEO)

        # VIDEO should be reset
        video_stats = tracker.get_modality_stats(QueryModality.VIDEO)
        assert video_stats["total_requests"] == 0

        # DOCUMENT should still have data
        doc_stats = tracker.get_modality_stats(QueryModality.DOCUMENT)
        assert doc_stats["total_requests"] == 1

    def test_reset_all_stats(self, tracker):
        """Test resetting all statistics"""
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)
        tracker.record_modality_execution(QueryModality.DOCUMENT, 50, True)

        tracker.reset_all_stats()

        # All should be reset
        all_stats = tracker.get_all_stats()
        assert len(all_stats) == 0

    def test_timestamps_tracked(self, tracker):
        """Test that timestamps are tracked"""
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)

        assert tracker.first_request_time[QueryModality.VIDEO] is not None
        assert tracker.last_request_time[QueryModality.VIDEO] is not None

        time.sleep(0.05)
        tracker.record_modality_execution(QueryModality.VIDEO, 100, True)

        # Last request time should be updated
        assert (
            tracker.last_request_time[QueryModality.VIDEO]
            > tracker.first_request_time[QueryModality.VIDEO]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
