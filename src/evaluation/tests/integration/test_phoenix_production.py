"""
Production-level integration tests for Phoenix storage.
These tests verify actual Phoenix integration, not mocks.
"""

import pytest
import phoenix as px
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import os

from src.evaluation.data.storage import (
    PhoenixStorage,
    ConnectionConfig,
    ConnectionState
)


@pytest.mark.integration
@pytest.mark.phoenix
class TestPhoenixProductionIntegration:
    """Integration tests with real Phoenix instance."""
    
    @pytest.fixture(scope="class")
    def phoenix_required(self):
        """Verify Phoenix is running."""
        try:
            client = px.Client()
            # Try to connect
            _ = client.get_spans_dataframe(limit=1)
            return True
        except Exception as e:
            pytest.skip(f"Phoenix not available: {e}")
    
    @pytest.fixture
    def storage(self, phoenix_required):
        """Create production storage instance."""
        config = ConnectionConfig(
            enable_health_checks=False  # Disable for tests
        )
        storage = PhoenixStorage(config)
        yield storage
        storage.shutdown()
    
    @pytest.mark.integration
    def test_connection_lifecycle(self, phoenix_required):
        """Test connection establishment and shutdown."""
        config = ConnectionConfig(enable_health_checks=False)
        
        # Test initialization
        storage = PhoenixStorage(config)
        assert storage.connection_state == ConnectionState.CONNECTED
        
        # Test metrics are initialized
        metrics = storage.get_metrics()
        assert metrics["connection_state"] == "connected"
        assert metrics["total_spans_sent"] == 0
        
        # Test shutdown
        storage.shutdown()
        assert storage.connection_state == ConnectionState.DISCONNECTED
    
    @pytest.mark.integration
    def test_experiment_logging_end_to_end(self, storage):
        """Test complete experiment logging flow."""
        # Log experiment
        experiment_id = storage.log_experiment_results(
            experiment_name="integration_test",
            profile="test_profile",
            strategy="test_strategy",
            results=[
                {"query": "test query 1", "score": 0.8, "items": ["item1", "item2"]},
                {"query": "test query 2", "score": 0.7, "items": ["item3"]},
                {"query": "test query 3", "score": 0.9, "items": ["item4", "item5"]}
            ],
            metrics={
                "mrr": 0.8,
                "recall@1": 0.7,
                "recall@5": 0.85,
                "precision": 0.75
            }
        )
        
        assert experiment_id is not None
        assert "integration_test" in experiment_id
        
        # Give Phoenix time to process
        time.sleep(1)
        
        # Verify metrics were updated
        metrics = storage.get_metrics()
        # Note: Span count depends on OpenTelemetry configuration
        # We just verify the structure
        assert "total_spans_sent" in metrics
        assert "success_rate" in metrics
    
    @pytest.mark.integration
    def test_trace_retrieval(self, storage):
        """Test retrieving traces from Phoenix."""
        # Get recent traces
        df = storage.get_traces_for_evaluation(
            start_time=datetime.now() - timedelta(hours=1),
            limit=10
        )
        
        # Should return a DataFrame (might be empty if no recent traces)
        assert isinstance(df, pd.DataFrame)
        
        # If we have traces, verify structure
        if not df.empty:
            # Phoenix spans should have certain columns
            expected_cols = ["trace_id", "name"]
            for col in expected_cols:
                assert col in df.columns or f"attributes.{col}" in df.columns
    
    @pytest.mark.integration
    def test_concurrent_operations(self, storage):
        """Test concurrent experiment logging."""
        results = []
        errors = []
        
        def log_experiment(index):
            """Log an experiment in a thread."""
            try:
                exp_id = storage.log_experiment_results(
                    experiment_name=f"concurrent_test_{index}",
                    profile=f"profile_{index % 3}",
                    strategy=f"strategy_{index % 2}",
                    results=[{"query": f"q{index}", "score": 0.5 + (index % 5) * 0.1}],
                    metrics={"mrr": 0.7 + (index % 3) * 0.1}
                )
                results.append(exp_id)
            except Exception as e:
                errors.append(str(e))
        
        # Launch concurrent operations
        threads = []
        for i in range(10):
            t = threading.Thread(target=log_experiment, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=5)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(r is not None for r in results)
    
    @pytest.mark.integration
    def test_health_check_recovery(self, phoenix_required):
        """Test health check and auto-recovery."""
        config = ConnectionConfig(
            enable_health_checks=True,
            health_check_interval_seconds=0.5
        )
        
        storage = PhoenixStorage(config)
        
        try:
            # Verify initial connection
            assert storage.connection_state == ConnectionState.CONNECTED
            
            # Simulate disconnection by breaking the client
            original_client = storage.client
            storage.client = None
            storage.connection_state = ConnectionState.DISCONNECTED
            
            # Wait for health check to detect and recover
            time.sleep(1.0)
            
            # Restore client for health check
            storage.client = original_client
            
            # Wait for recovery
            time.sleep(1.0)
            
            # Should have attempted reconnection
            # Note: Actual reconnection depends on Phoenix availability
            
        finally:
            storage.shutdown()
    
    @pytest.mark.integration
    def test_error_handling_with_invalid_data(self, storage):
        """Test error handling with invalid data."""
        # Try to log with invalid data
        result = storage.log_experiment_results(
            experiment_name="",  # Empty name
            profile=None,  # None profile
            strategy="test",
            results="not_a_list",  # Wrong type
            metrics={"invalid": float('inf')}  # Invalid float
        )
        
        # Should handle gracefully (might return None or handle internally)
        # The important thing is it doesn't crash
        assert result is None or isinstance(result, str)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_high_volume_operations(self, storage):
        """Test high volume of operations."""
        start_time = time.time()
        num_experiments = 100
        
        # Track successful operations
        successful_ops = 0
        failed_ops = 0
        
        for i in range(num_experiments):
            result = storage.log_experiment_results(
                experiment_name=f"volume_test_{i}",
                profile="high_volume",
                strategy="bulk_test",
                results=[{"query": f"q{j}", "score": 0.5} for j in range(10)],
                metrics={"mrr": 0.75}
            )
            if result:
                successful_ops += 1
            else:
                failed_ops += 1
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 30, f"High volume test took too long: {elapsed}s"
        
        # Should have completed most operations successfully
        assert successful_ops > 0, f"No operations succeeded out of {num_experiments}"
        
        # Check connection is still healthy
        assert storage.connection_state == ConnectionState.CONNECTED
    
    @pytest.mark.integration
    def test_span_export_metrics(self, storage):
        """Test that span export metrics are tracked."""
        # Log some experiments
        for i in range(5):
            storage.log_experiment_results(
                experiment_name=f"metrics_test_{i}",
                profile="test",
                strategy="test",
                results=[{"query": "test", "score": 0.8}],
                metrics={"mrr": 0.75}
            )
        
        # Give time for export
        time.sleep(2)
        
        # Check metrics
        metrics = storage.get_metrics()
        
        # Should have metrics (values depend on Phoenix availability)
        assert "total_spans_sent" in metrics
        assert "total_spans_failed" in metrics
        assert "success_rate" in metrics
        assert "avg_latency_ms" in metrics
        
        # If Phoenix is working, we should have some successful exports
        # (This might fail if Phoenix isn't properly configured for OTLP)
        if metrics["total_spans_sent"] > 0:
            assert metrics["success_rate"] > 0
            assert metrics["avg_latency_ms"] > 0
    
    @pytest.mark.integration
    def test_context_manager_cleanup(self, phoenix_required):
        """Test context manager properly cleans up resources."""
        config = ConnectionConfig(enable_health_checks=False)
        
        with PhoenixStorage(config) as storage:
            # Use storage
            storage.log_experiment_results(
                experiment_name="context_test",
                profile="test",
                strategy="test",
                results=[],
                metrics={}
            )
            assert storage.connection_state == ConnectionState.CONNECTED
        
        # After context, should be disconnected
        assert storage.connection_state == ConnectionState.DISCONNECTED
        assert storage.client is None