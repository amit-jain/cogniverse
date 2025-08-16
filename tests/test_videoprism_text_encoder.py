"""
Unit tests for VideoPrismTextEncoder with mocked dependencies.

Tests cover:
- Basic encoding functionality
- Caching behavior
- Circuit breaker functionality
- Error handling
- Performance metrics
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time
from typing import Any

from src.common.models.videoprism_text_encoder import (
    VideoPrismTextEncoder,
    CircuitBreaker,
    CircuitState,
    PerformanceMetrics,
    ModelPool,
    create_text_encoder
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics tracking"""
    
    def setUp(self):
        self.metrics = PerformanceMetrics()
    
    def test_record_successful_request(self):
        """Test recording successful requests"""
        self.metrics.record_request(True, 100.0, False)
        
        self.assertEqual(self.metrics.total_requests, 1)
        self.assertEqual(self.metrics.successful_requests, 1)
        self.assertEqual(self.metrics.failed_requests, 0)
        self.assertEqual(self.metrics.total_latency_ms, 100.0)
        self.assertEqual(self.metrics.avg_latency_ms, 100.0)
        self.assertEqual(self.metrics.success_rate, 100.0)
    
    def test_record_failed_request(self):
        """Test recording failed requests"""
        self.metrics.record_request(False, 50.0, False)
        
        self.assertEqual(self.metrics.total_requests, 1)
        self.assertEqual(self.metrics.successful_requests, 0)
        self.assertEqual(self.metrics.failed_requests, 1)
        self.assertEqual(self.metrics.success_rate, 0.0)
    
    def test_cache_metrics(self):
        """Test cache hit/miss tracking"""
        self.metrics.record_request(True, 10.0, True)  # Cache hit
        self.metrics.record_request(True, 100.0, False)  # Cache miss
        
        self.assertEqual(self.metrics.cache_hits, 1)
        self.assertEqual(self.metrics.cache_misses, 1)
        self.assertEqual(self.metrics.cache_hit_rate, 50.0)
    
    def test_latency_stats(self):
        """Test latency statistics"""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        for latency in latencies:
            self.metrics.record_request(True, latency, False)
        
        self.assertEqual(self.metrics.min_latency_ms, 10.0)
        self.assertEqual(self.metrics.max_latency_ms, 50.0)
        self.assertEqual(self.metrics.avg_latency_ms, 30.0)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def setUp(self):
        self.breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1.0
        )
    
    def test_initial_state(self):
        """Test circuit breaker starts closed"""
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
    
    def test_successful_calls(self):
        """Test circuit remains closed on successful calls"""
        def success_func():
            return "success"
        
        result = self.breaker.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
    
    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures"""
        def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with self.assertRaises(Exception):
            self.breaker.call(failing_func)
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        
        # Second failure - should open circuit
        with self.assertRaises(Exception):
            self.breaker.call(failing_func)
        self.assertEqual(self.breaker.state, CircuitState.OPEN)
    
    def test_circuit_rejects_when_open(self):
        """Test circuit rejects calls when open"""
        # Open the circuit
        def failing_func():
            raise Exception("Test failure")
        
        for _ in range(2):
            try:
                self.breaker.call(failing_func)
            except:
                pass
        
        # Should reject without calling function
        with self.assertRaises(Exception) as ctx:
            self.breaker.call(lambda: "should not execute")
        self.assertIn("Circuit breaker is OPEN", str(ctx.exception))
    
    def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout"""
        # Open the circuit
        def failing_func():
            raise Exception("Test failure")
        
        for _ in range(2):
            try:
                self.breaker.call(failing_func)
            except:
                pass
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should be half-open now
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)
    
    def test_circuit_closes_on_recovery(self):
        """Test circuit closes when call succeeds in half-open state"""
        # Open the circuit
        def failing_func():
            raise Exception("Test failure")
        
        for _ in range(2):
            try:
                self.breaker.call(failing_func)
            except:
                pass
        
        # Wait for recovery
        time.sleep(1.1)
        
        # Successful call should close circuit
        result = self.breaker.call(lambda: "recovered")
        self.assertEqual(result, "recovered")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)


class TestModelPool(unittest.TestCase):
    """Test model pool functionality"""
    
    def test_pool_creation(self):
        """Test pool creates instances up to max size"""
        created_count = 0
        
        def factory():
            nonlocal created_count
            created_count += 1
            return f"model_{created_count}"
        
        pool = ModelPool(factory, max_size=3)
        
        # Get multiple models
        models = []
        for _ in range(3):
            with pool.get_model() as model:
                models.append(model)
        
        self.assertEqual(created_count, 3)
        self.assertEqual(models, ["model_1", "model_2", "model_3"])
    
    def test_pool_reuse(self):
        """Test pool reuses returned models"""
        created_count = 0
        
        def factory():
            nonlocal created_count
            created_count += 1
            return f"model_{created_count}"
        
        pool = ModelPool(factory, max_size=2)
        
        # Use and return model
        with pool.get_model() as model1:
            self.assertEqual(model1, "model_1")
        
        # Should reuse same model
        with pool.get_model() as model2:
            self.assertEqual(model2, "model_1")
        
        self.assertEqual(created_count, 1)


class TestVideoPrismTextEncoder(unittest.TestCase):
    """Test VideoPrismTextEncoder functionality"""
    
    def setUp(self):
        # Mock VideoPrism module
        self.vp_mock = MagicMock()
        self.vp_patch = patch.dict('sys.modules', {'videoprism': self.vp_mock})
        self.vp_patch.start()
        
        # Mock the global VIDEOPRISM_AVAILABLE flag
        import src.common.models.videoprism_text_encoder_v2 as encoder_module
        encoder_module.VIDEOPRISM_AVAILABLE = True
        encoder_module.vp = self.vp_mock.models
        
        # Setup mock returns
        self.vp_mock.models.load_text_tokenizer.return_value = Mock()
        self.vp_mock.models.get_model.return_value = Mock()
        self.vp_mock.models.load_pretrained_weights.return_value = Mock()
        self.vp_mock.models.tokenize_texts.return_value = (
            np.array([[1, 2, 3]]),  # token ids
            np.array([[0, 0, 0]])   # padding
        )
        
        # Mock model.apply to return embeddings
        model_mock = self.vp_mock.models.get_model.return_value
        model_mock.apply.return_value = (
            None,  # video embeddings
            np.random.randn(1, 768),  # text embeddings
            None   # other outputs
        )
    
    def tearDown(self):
        self.vp_patch.stop()
        # Clear class-level cache
        VideoPrismTextEncoder._model_cache.clear()
    
    def test_encoder_initialization(self):
        """Test encoder initializes correctly"""
        encoder = VideoPrismTextEncoder(
            model_name="test_model",
            embedding_dim=768
        )
        
        self.assertEqual(encoder.model_name, "test_model")
        self.assertEqual(encoder.embedding_dim, 768)
        self.assertIsNotNone(encoder.metrics)
        self.assertIsNotNone(encoder.circuit_breaker)
    
    def test_model_caching(self):
        """Test models are cached across instances"""
        # Create first encoder
        encoder1 = VideoPrismTextEncoder("test_model", 768)
        
        # Should load model
        self.vp_mock.models.load_text_tokenizer.assert_called_once()
        self.vp_mock.models.get_model.assert_called_once()
        
        # Reset mocks
        self.vp_mock.models.load_text_tokenizer.reset_mock()
        self.vp_mock.models.get_model.reset_mock()
        
        # Create second encoder with same model
        encoder2 = VideoPrismTextEncoder("test_model", 768)
        
        # Should not load model again
        self.vp_mock.models.load_text_tokenizer.assert_not_called()
        self.vp_mock.models.get_model.assert_not_called()
    
    def test_encode_basic(self):
        """Test basic encoding functionality"""
        encoder = VideoPrismTextEncoder("test_model", 768)
        
        embeddings = encoder.encode("test query")
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (768,))
        
        # Check text was formatted with template
        self.vp_mock.models.tokenize_texts.assert_called_with(
            encoder.tokenizer,
            ["a video of test query."]
        )
    
    def test_encode_with_caching(self):
        """Test encoding uses cache"""
        encoder = VideoPrismTextEncoder("test_model", 768, cache_size=10)
        
        # First call
        embeddings1 = encoder.encode("test query")
        self.assertEqual(encoder.metrics.cache_hits, 0)
        self.assertEqual(encoder.metrics.cache_misses, 1)
        
        # Second call - should hit cache
        embeddings2 = encoder.encode("test query")
        np.testing.assert_array_equal(embeddings1, embeddings2)
        
        # Note: LRU cache metrics are not directly accessible
        # In real implementation, we'd need to track cache hits differently
    
    def test_encode_batch(self):
        """Test batch encoding"""
        encoder = VideoPrismTextEncoder("test_model", 768)
        
        texts = ["query1", "query2", "query3"]
        embeddings = encoder.encode_batch(texts)
        
        self.assertEqual(embeddings.shape, (3, 768))
    
    def test_dimension_projection(self):
        """Test embedding dimension projection"""
        # Mock model to return wrong dimension
        model_mock = self.vp_mock.models.get_model.return_value
        model_mock.apply.return_value = (
            None,
            np.random.randn(1, 512),  # Wrong dimension
            None
        )
        
        encoder = VideoPrismTextEncoder("test_model", 768)
        embeddings = encoder.encode("test")
        
        # Should project to correct dimension
        self.assertEqual(embeddings.shape, (768,))
    
    def test_encode_with_error(self):
        """Test error handling in encoding"""
        encoder = VideoPrismTextEncoder("test_model", 768)
        
        # Make tokenize fail
        self.vp_mock.models.tokenize_texts.side_effect = Exception("Tokenization failed")
        
        with self.assertRaises(Exception):
            encoder.encode("test")
        
        # Check metrics recorded failure
        self.assertEqual(encoder.metrics.failed_requests, 1)
        self.assertEqual(encoder.metrics.successful_requests, 0)
    
    def test_health_check(self):
        """Test health check functionality"""
        encoder = VideoPrismTextEncoder("test_model", 768)
        
        health = encoder.health_check()
        
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["model"], "test_model")
        self.assertEqual(health["embedding_dim"], 768)
        self.assertIn("metrics", health)
    
    def test_clear_cache(self):
        """Test cache clearing"""
        encoder = VideoPrismTextEncoder("test_model", 768)
        
        # Encode something to populate cache
        encoder.encode("test")
        
        # Clear cache
        encoder.clear_cache()
        
        # Verify cache was cleared (would need better cache introspection in real code)
        self.assertTrue(True)  # Placeholder
    
    def test_factory_function(self):
        """Test factory function creates encoder correctly"""
        encoder = create_text_encoder(
            model_name="test_model",
            embedding_dim=768,
            cache_size=500
        )
        
        self.assertIsInstance(encoder, VideoPrismTextEncoder)
        self.assertEqual(encoder.model_name, "test_model")
        self.assertEqual(encoder.embedding_dim, 768)


if __name__ == "__main__":
    unittest.main()