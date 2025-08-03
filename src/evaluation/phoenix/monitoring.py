"""
Real-time monitoring and metrics collection with Arize Phoenix
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading

import phoenix as px
from phoenix.trace import TraceDataset

logger = logging.getLogger(__name__)


@dataclass
class AlertThresholds:
    """Alert thresholds for monitoring"""
    latency_p95_ms: float = 1000.0
    error_rate: float = 0.05
    mrr_drop: float = 0.1
    throughput_drop: float = 0.3


@dataclass
class MetricWindow:
    """Sliding window for metric aggregation"""
    window_size: int = 100
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, value: float):
        """Add value to window"""
        self.values.append(value)
    
    def get_mean(self) -> float:
        """Get mean of window values"""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def get_p95(self) -> float:
        """Get 95th percentile of window values"""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def get_error_rate(self) -> float:
        """Get error rate (assuming values are 0 for success, 1 for error)"""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class RetrievalMonitor:
    """Real-time monitoring of retrieval performance"""
    
    def __init__(self, alert_thresholds: Optional[AlertThresholds] = None):
        self.alert_thresholds = alert_thresholds or AlertThresholds()
        self.phoenix_session = None
        
        # Metric windows for different profiles
        self.latency_windows: Dict[str, MetricWindow] = {}
        self.error_windows: Dict[str, MetricWindow] = {}
        self.mrr_windows: Dict[str, MetricWindow] = {}
        
        # Alert state
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_lock = threading.Lock()
        
        # Metrics buffer for batch logging
        self.metrics_buffer = []
        self.buffer_lock = threading.Lock()
        
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
    
    def start(self):
        """Start monitoring and Phoenix session"""
        # Launch Phoenix if not already running
        if self.phoenix_session is None:
            self.phoenix_session = px.launch_app()
            logger.info(f"Phoenix launched at: {self.phoenix_session.url if hasattr(self.phoenix_session, 'url') else 'localhost'}")
        
        # Start monitoring thread
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Retrieval monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Retrieval monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                # Process buffered metrics
                self._process_metrics_buffer()
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep for monitoring interval
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def log_retrieval_event(self, event: Dict[str, Any]):
        """Log individual retrieval event"""
        profile = event.get("profile", "unknown")
        strategy = event.get("strategy", "unknown")
        
        # Initialize windows if needed
        profile_key = f"{profile}_{strategy}"
        if profile_key not in self.latency_windows:
            self.latency_windows[profile_key] = MetricWindow()
            self.error_windows[profile_key] = MetricWindow()
            self.mrr_windows[profile_key] = MetricWindow()
        
        # Update metric windows
        if "latency_ms" in event:
            self.latency_windows[profile_key].add(event["latency_ms"])
        
        if "error" in event:
            self.error_windows[profile_key].add(1.0 if event["error"] else 0.0)
        
        if "mrr" in event:
            self.mrr_windows[profile_key].add(event["mrr"])
        
        # Buffer event for Phoenix logging
        with self.buffer_lock:
            self.metrics_buffer.append({
                "timestamp": datetime.utcnow().isoformat(),
                "profile": profile,
                "strategy": strategy,
                "query": event.get("query", ""),
                "latency_ms": event.get("latency_ms", 0),
                "num_results": event.get("num_results", 0),
                "mrr": event.get("mrr", 0),
                "error": event.get("error", False)
            })
        
        # Log to Phoenix trace
        try:
            px.log_trace(
                name="retrieval",
                inputs={"query": event.get("query", "")},
                outputs={"results": event.get("results", [])},
                metadata={
                    "profile": profile,
                    "strategy": strategy,
                    "latency_ms": event.get("latency_ms", 0),
                    "mrr": event.get("mrr", 0)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log to Phoenix: {e}")
    
    def _process_metrics_buffer(self):
        """Process buffered metrics"""
        if not self.metrics_buffer:
            return
        
        with self.buffer_lock:
            events_to_process = self.metrics_buffer.copy()
            self.metrics_buffer.clear()
        
        # Aggregate metrics by profile
        profile_metrics = {}
        for event in events_to_process:
            profile_key = f"{event['profile']}_{event['strategy']}"
            if profile_key not in profile_metrics:
                profile_metrics[profile_key] = {
                    "count": 0,
                    "total_latency": 0,
                    "total_mrr": 0,
                    "errors": 0
                }
            
            metrics = profile_metrics[profile_key]
            metrics["count"] += 1
            metrics["total_latency"] += event["latency_ms"]
            metrics["total_mrr"] += event["mrr"]
            if event.get("error"):
                metrics["errors"] += 1
        
        # Log aggregated metrics
        for profile_key, metrics in profile_metrics.items():
            if metrics["count"] > 0:
                avg_latency = metrics["total_latency"] / metrics["count"]
                avg_mrr = metrics["total_mrr"] / metrics["count"]
                error_rate = metrics["errors"] / metrics["count"]
                
                logger.info(
                    f"Profile {profile_key}: "
                    f"Count={metrics['count']}, "
                    f"AvgLatency={avg_latency:.2f}ms, "
                    f"AvgMRR={avg_mrr:.3f}, "
                    f"ErrorRate={error_rate:.3f}"
                )
    
    def _check_alerts(self):
        """Check for alert conditions"""
        alerts_to_trigger = []
        
        for profile_key, latency_window in self.latency_windows.items():
            # Check latency
            p95_latency = latency_window.get_p95()
            if p95_latency > self.alert_thresholds.latency_p95_ms:
                alerts_to_trigger.append({
                    "type": "latency",
                    "profile": profile_key,
                    "value": p95_latency,
                    "threshold": self.alert_thresholds.latency_p95_ms
                })
            
            # Check error rate
            if profile_key in self.error_windows:
                error_rate = self.error_windows[profile_key].get_error_rate()
                if error_rate > self.alert_thresholds.error_rate:
                    alerts_to_trigger.append({
                        "type": "error_rate",
                        "profile": profile_key,
                        "value": error_rate,
                        "threshold": self.alert_thresholds.error_rate
                    })
            
            # Check MRR degradation
            if profile_key in self.mrr_windows:
                current_mrr = self.mrr_windows[profile_key].get_mean()
                # Compare with historical baseline (simplified - would need historical data)
                # For now, just check if MRR is too low
                if current_mrr < 0.5:  # Threshold for acceptable MRR
                    alerts_to_trigger.append({
                        "type": "mrr_degradation",
                        "profile": profile_key,
                        "value": current_mrr,
                        "threshold": 0.5
                    })
        
        # Trigger alerts
        for alert in alerts_to_trigger:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert"""
        alert_key = f"{alert['type']}_{alert['profile']}"
        
        with self.alert_lock:
            # Check if alert is already active
            if alert_key in self.active_alerts:
                # Update existing alert
                self.active_alerts[alert_key]["last_triggered"] = datetime.utcnow()
                self.active_alerts[alert_key]["count"] += 1
            else:
                # New alert
                self.active_alerts[alert_key] = {
                    "alert": alert,
                    "first_triggered": datetime.utcnow(),
                    "last_triggered": datetime.utcnow(),
                    "count": 1
                }
                
                # Log alert
                logger.warning(
                    f"ALERT: {alert['type']} for {alert['profile']} - "
                    f"Value: {alert['value']:.3f}, Threshold: {alert['threshold']:.3f}"
                )
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        with self.alert_lock:
            return list(self.active_alerts.values())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        summary = {}
        
        for profile_key in self.latency_windows.keys():
            summary[profile_key] = {
                "latency_mean": self.latency_windows[profile_key].get_mean(),
                "latency_p95": self.latency_windows[profile_key].get_p95(),
                "error_rate": self.error_windows[profile_key].get_error_rate() if profile_key in self.error_windows else 0,
                "mrr_mean": self.mrr_windows[profile_key].get_mean() if profile_key in self.mrr_windows else 0
            }
        
        return summary
    
    def export_traces(self, output_path: str):
        """Export collected traces to file"""
        try:
            traces = px.Client().get_trace_dataset()
            traces.save(output_path)
            logger.info(f"Exported traces to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export traces: {e}")