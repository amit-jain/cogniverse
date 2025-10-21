"""
Per-Modality Performance Metrics

Track latency, success rates, and error patterns per modality.
Part of Phase 12: Production Readiness.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from cogniverse_agents.search.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class ModalityMetricsTracker:
    """
    Track performance metrics per modality

    Metrics tracked:
    - Latency (P50, P95, P99)
    - Success rate
    - Error count by type
    - Request volume
    - Throughput

    Example:
        tracker = ModalityMetricsTracker()

        # Record execution
        tracker.record_modality_execution(
            modality=QueryModality.VIDEO,
            latency_ms=234.5,
            success=True
        )

        # Get stats
        stats = tracker.get_modality_stats(QueryModality.VIDEO)
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker

        Args:
            window_size: Number of recent executions to track per modality
        """
        self.window_size = window_size

        # Track latencies (rolling window)
        self.modality_latencies: Dict[QueryModality, List[float]] = {
            modality: [] for modality in QueryModality
        }

        # Track errors by type
        self.modality_errors: Dict[QueryModality, Dict[str, int]] = {
            modality: defaultdict(int) for modality in QueryModality
        }

        # Track successes
        self.modality_success = defaultdict(int)

        # Track total requests
        self.modality_requests = defaultdict(int)

        # Track first/last request time
        self.first_request_time: Dict[QueryModality, Optional[datetime]] = {
            modality: None for modality in QueryModality
        }
        self.last_request_time: Dict[QueryModality, Optional[datetime]] = {
            modality: None for modality in QueryModality
        }

        logger.info(
            f"📊 Initialized ModalityMetricsTracker (window_size: {window_size})"
        )

    def record_modality_execution(
        self,
        modality: QueryModality,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """
        Record execution metrics for a modality

        Args:
            modality: Query modality
            latency_ms: Execution latency in milliseconds
            success: Whether execution succeeded
            error: Error message if failed (optional)
        """
        # Record latency
        self.modality_latencies[modality].append(latency_ms)

        # Maintain rolling window
        if len(self.modality_latencies[modality]) > self.window_size:
            self.modality_latencies[modality] = self.modality_latencies[modality][
                -self.window_size :
            ]

        # Record success/error
        self.modality_requests[modality] += 1

        if success:
            self.modality_success[modality] += 1
        else:
            error_type = error if error else "unknown_error"
            self.modality_errors[modality][error_type] += 1

        # Update timestamps
        now = datetime.now()
        if self.first_request_time[modality] is None:
            self.first_request_time[modality] = now
        self.last_request_time[modality] = now

        logger.debug(
            f"📝 Recorded {modality.value}: {latency_ms:.0f}ms, "
            f"success={success}"
        )

    def get_modality_stats(self, modality: QueryModality) -> Dict[str, Any]:
        """
        Get aggregated stats for modality

        Args:
            modality: Query modality

        Returns:
            {
                "modality": str,
                "total_requests": int,
                "success_count": int,
                "error_count": int,
                "success_rate": float,
                "p50_latency": float,
                "p95_latency": float,
                "p99_latency": float,
                "avg_latency": float,
                "min_latency": float,
                "max_latency": float,
                "error_breakdown": {error_type: count},
                "throughput_qps": float,
            }
        """
        latencies = self.modality_latencies[modality]

        if not latencies:
            return {
                "modality": modality.value,
                "total_requests": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "avg_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "error_breakdown": {},
                "throughput_qps": 0.0,
            }

        # Calculate latency percentiles
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        avg = float(np.mean(latencies))
        min_lat = float(np.min(latencies))
        max_lat = float(np.max(latencies))

        # Calculate success rate
        total_requests = self.modality_requests[modality]
        success_count = self.modality_success[modality]
        error_count = total_requests - success_count
        success_rate = success_count / total_requests if total_requests > 0 else 0.0

        # Calculate throughput
        throughput_qps = self._calculate_throughput(modality)

        return {
            "modality": modality.value,
            "total_requests": total_requests,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "avg_latency": avg,
            "min_latency": min_lat,
            "max_latency": max_lat,
            "error_breakdown": dict(self.modality_errors[modality]),
            "throughput_qps": throughput_qps,
        }

    def _calculate_throughput(self, modality: QueryModality) -> float:
        """
        Calculate throughput (QPS) for modality

        Args:
            modality: Query modality

        Returns:
            Queries per second
        """
        first = self.first_request_time[modality]
        last = self.last_request_time[modality]

        if not first or not last:
            return 0.0

        duration_seconds = (last - first).total_seconds()
        if duration_seconds == 0:
            # Only one request or all in same second
            return float(self.modality_requests[modality])

        return self.modality_requests[modality] / duration_seconds

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get stats for all modalities

        Returns:
            {modality_name: stats}
        """
        all_stats = {}
        for modality in QueryModality:
            stats = self.get_modality_stats(modality)
            if stats["total_requests"] > 0:
                # Only include modalities with activity
                all_stats[modality.value] = stats

        return all_stats

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all modalities

        Returns:
            Aggregate statistics
        """
        all_stats = self.get_all_stats()

        if not all_stats:
            return {
                "total_requests": 0,
                "overall_success_rate": 0.0,
                "active_modalities": 0,
                "avg_latency_p95": 0.0,
            }

        total_requests = sum(s["total_requests"] for s in all_stats.values())
        total_success = sum(s["success_count"] for s in all_stats.values())
        overall_success_rate = total_success / total_requests if total_requests > 0 else 0.0

        # Average P95 latency across modalities
        p95_latencies = [s["p95_latency"] for s in all_stats.values() if s["total_requests"] > 0]
        avg_p95 = float(np.mean(p95_latencies)) if p95_latencies else 0.0

        return {
            "total_requests": total_requests,
            "overall_success_rate": overall_success_rate,
            "active_modalities": len(all_stats),
            "avg_latency_p95": avg_p95,
            "modality_breakdown": {
                mod: stats["total_requests"] for mod, stats in all_stats.items()
            },
        }

    def get_slowest_modalities(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get slowest modalities by P95 latency

        Args:
            top_k: Number of slowest modalities to return

        Returns:
            List of {modality, p95_latency, total_requests}
        """
        all_stats = self.get_all_stats()

        # Sort by P95 latency
        sorted_modalities = sorted(
            all_stats.items(),
            key=lambda x: x[1]["p95_latency"],
            reverse=True,
        )

        return [
            {
                "modality": mod,
                "p95_latency": stats["p95_latency"],
                "total_requests": stats["total_requests"],
            }
            for mod, stats in sorted_modalities[:top_k]
        ]

    def get_error_prone_modalities(self, min_error_rate: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get modalities with high error rates

        Args:
            min_error_rate: Minimum error rate to include (0-1)

        Returns:
            List of {modality, error_rate, error_count}
        """
        all_stats = self.get_all_stats()

        error_prone = []
        for modality, stats in all_stats.items():
            error_rate = 1.0 - stats["success_rate"]
            if error_rate >= min_error_rate:
                error_prone.append({
                    "modality": modality,
                    "error_rate": error_rate,
                    "error_count": stats["error_count"],
                    "error_breakdown": stats["error_breakdown"],
                })

        # Sort by error rate
        error_prone.sort(key=lambda x: x["error_rate"], reverse=True)

        return error_prone

    def reset_modality_stats(self, modality: QueryModality):
        """
        Reset statistics for a specific modality

        Args:
            modality: Modality to reset
        """
        self.modality_latencies[modality] = []
        self.modality_errors[modality] = defaultdict(int)
        self.modality_success[modality] = 0
        self.modality_requests[modality] = 0
        self.first_request_time[modality] = None
        self.last_request_time[modality] = None

        logger.info(f"🗑️ Reset stats for {modality.value}")

    def reset_all_stats(self):
        """Reset all statistics"""
        for modality in QueryModality:
            self.reset_modality_stats(modality)

        logger.info("🗑️ Reset all modality stats")
