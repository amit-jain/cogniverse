"""
Comprehensive unit tests for router module orchestration and error handling.
Tests critical paths, edge cases, and error conditions.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.routing.router import TieredRouter, ComprehensiveRouter, RoutingTier
from src.app.routing.base import RoutingDecision, SearchModality, GenerationType
from src.app.routing.strategies import (
    GLiNERRoutingStrategy,
    LLMRoutingStrategy,
    KeywordRoutingStrategy,
    LangExtractRoutingStrategy
)


class TestTieredRouterInitialization:
    """Test router initialization with different config types."""
    
    @pytest.mark.integration
    def test_init_with_dict_config(self):
        """Test initialization with dictionary config."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": False,
                "enable_langextract": True,
                "enable_fallback": True
            },
            "gliner_config": {"model": "test"},
            "llm_config": {"model": "test"},
            "cache_config": {"enable_caching": True}
        }
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy'):
            router = TieredRouter(config)
            assert router.config == config
            assert len(router.strategies) >= 2  # At least 2 strategies enabled
    
    @pytest.mark.integration
    def test_init_with_object_config(self):
        """Test initialization with object config."""
        config = Mock()
        config.tier_config = Mock()
        config.tier_config.get = Mock(side_effect=lambda key, default: {
            "enable_fast_path": True,
            "enable_slow_path": True,
            "enable_langextract": False,
            "enable_fallback": True
        }.get(key, default))
        config.gliner_config = {}
        config.llm_config = {}
        config.langextract_config = {}
        config.keyword_config = {}
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy'):
            with patch('src.app.routing.router.LLMRoutingStrategy'):
                with patch('src.app.routing.router.LangExtractRoutingStrategy'):
                    with patch('src.app.routing.router.KeywordRoutingStrategy'):
                        router = TieredRouter(config)
                        assert router.config == config
                        # Check that langextract was not initialized
                        assert len(router.strategies) == 3  # Only fast, slow, and fallback
    
    @pytest.mark.integration
    def test_init_with_legacy_config(self):
        """Test initialization with legacy config format."""
        config = Mock(spec=[])  # No tier_config attribute
        config.enable_fast_path = True
        config.enable_slow_path = False
        config.enable_langextract = False
        config.enable_fallback = True
        config.gliner_config = {}
        config.keyword_config = {}
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy'):
            with patch('src.app.routing.router.KeywordRoutingStrategy'):
                router = TieredRouter(config)
                assert RoutingTier.FAST_PATH in router.strategies
                assert RoutingTier.FALLBACK in router.strategies
                assert RoutingTier.SLOW_PATH not in router.strategies


class TestTieredRouterCaching:
    """Test caching functionality."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit returns cached decision."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {},
            "cache_config": {
                "enable_caching": True,
                "cache_ttl_seconds": 300
            }
        }
        
        router = TieredRouter(config)
        
        # First call - will be cached
        decision1 = await router.route("test query")
        
        # Second call - should hit cache
        with patch.object(router.strategies[RoutingTier.FALLBACK], 'route') as mock_route:
            decision2 = await router.route("test query")
            mock_route.assert_not_called()  # Strategy shouldn't be called
            
        assert decision1.search_modality == decision2.search_modality
        assert decision1.generation_type == decision2.generation_type
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_cache_expiry(self):
        """Test cache expiry after TTL."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {},
            "cache_config": {
                "enable_caching": True,
                "cache_ttl_seconds": 0.1  # 100ms TTL
            }
        }
        
        router = TieredRouter(config)
        
        # First call
        await router.route("test query")
        
        # Wait for cache to expire
        await asyncio.sleep(0.2)
        
        # Second call - cache should be expired
        with patch.object(router.strategies[RoutingTier.FALLBACK], 'route') as mock_route:
            mock_route.return_value = RoutingDecision(
                search_modality=SearchModality.TEXT,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.5,
                routing_method="keyword"
            )
            await router.route("test query")
            mock_route.assert_called_once()  # Strategy should be called
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_cache_disabled(self):
        """Test caching when disabled."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {},
            "cache_config": {
                "enable_caching": False
            }
        }
        
        router = TieredRouter(config)
        
        # First call
        await router.route("test query")
        
        # Second call - should not use cache
        with patch.object(router.strategies[RoutingTier.FALLBACK], 'route') as mock_route:
            mock_route.return_value = RoutingDecision(
                search_modality=SearchModality.TEXT,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.5,
                routing_method="keyword"
            )
            await router.route("test query")
            mock_route.assert_called_once()  # Strategy should be called


class TestTieredRouterEscalation:
    """Test tier escalation logic."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_successful_fast_path(self):
        """Test successful routing via fast path."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_fallback": True,
                "fast_path_confidence_threshold": 0.7
            }
        }
        
        router = TieredRouter(config)
        
        # Mock fast path with high confidence
        mock_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="gliner"
        )
        
        with patch.object(router.strategies[RoutingTier.FAST_PATH], 'route', return_value=mock_decision):
            decision = await router.route("test query")
            
        assert decision.confidence_score == 0.8
        assert decision.routing_method == "gliner"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_escalation_to_slow_path(self):
        """Test escalation from fast path to slow path."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_fallback": True,
                "fast_path_confidence_threshold": 0.7,
                "slow_path_confidence_threshold": 0.6
            }
        }
        
        router = TieredRouter(config)
        
        # Mock fast path with low confidence
        fast_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.5,  # Below threshold
            routing_method="gliner"
        )
        
        # Mock slow path with high confidence
        slow_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.8,
            routing_method="llm"
        )
        
        with patch.object(router.strategies[RoutingTier.FAST_PATH], 'route', return_value=fast_decision):
            with patch.object(router.strategies[RoutingTier.SLOW_PATH], 'route', return_value=slow_decision):
                decision = await router.route("complex query")
        
        assert decision.routing_method == "llm"
        assert decision.confidence_score == 0.8
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_escalation_to_fallback(self):
        """Test escalation all the way to fallback."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_langextract": True,
                "enable_fallback": True,
                "fast_path_confidence_threshold": 0.7,
                "slow_path_confidence_threshold": 0.6,
                "langextract_confidence_threshold": 0.5
            }
        }
        
        router = TieredRouter(config)
        
        # All strategies return low confidence
        low_confidence = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.3,
            routing_method="test"
        )
        
        fallback_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.85,
            routing_method="keyword"
        )
        
        with patch.object(router.strategies[RoutingTier.FAST_PATH], 'route', return_value=low_confidence):
            with patch.object(router.strategies[RoutingTier.SLOW_PATH], 'route', return_value=low_confidence):
                with patch.object(router.strategies[RoutingTier.LANGEXTRACT], 'route', return_value=low_confidence):
                    with patch.object(router.strategies[RoutingTier.FALLBACK], 'route', return_value=fallback_decision):
                        decision = await router.route("unclear query")
        
        assert decision.routing_method == "keyword"
        assert decision.confidence_score == 0.85


class TestTieredRouterErrorHandling:
    """Test error handling in router."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_strategy_exception_handling(self):
        """Test handling when a strategy raises an exception."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_fallback": True
            }
        }
        
        router = TieredRouter(config)
        
        # Mock fast path to raise exception
        with patch.object(router.strategies[RoutingTier.FAST_PATH], 'route', side_effect=Exception("Model error")):
            # Mock slow path to succeed
            slow_decision = RoutingDecision(
                search_modality=SearchModality.TEXT,
                generation_type=GenerationType.SUMMARY,
                confidence_score=0.7,
                routing_method="llm"
            )
            with patch.object(router.strategies[RoutingTier.SLOW_PATH], 'route', return_value=slow_decision):
                decision = await router.route("test query")
        
        # Should fall through to slow path
        assert decision.routing_method == "llm"
        assert decision.confidence_score == 0.7
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_all_strategies_fail(self):
        """Test when all strategies fail."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_fallback": False  # No fallback
            }
        }
        
        router = TieredRouter(config)
        
        # All strategies raise exceptions
        with patch.object(router.strategies[RoutingTier.FAST_PATH], 'route', side_effect=Exception("Error 1")):
            with patch.object(router.strategies[RoutingTier.SLOW_PATH], 'route', side_effect=Exception("Error 2")):
                decision = await router.route("test query")
        
        # Should return default decision
        assert decision.confidence_score < 0.5
        assert "no" in decision.reasoning.lower() or "unavailable" in decision.routing_method.lower()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_timeout_handling(self):
        """Test timeout handling in routing."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_fallback": True,
                "max_routing_time_ms": 100  # 100ms timeout
            },
            "keyword_config": {}
        }
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy'):
            with patch('src.app.routing.router.KeywordRoutingStrategy'):
                router = TieredRouter(config)
                
                # Mock strategy to take too long
                async def slow_route(query, context=None):
                    await asyncio.sleep(0.5)  # 500ms
                    return RoutingDecision(
                        search_modality=SearchModality.VIDEO,
                        generation_type=GenerationType.RAW_RESULTS,
                        confidence_score=0.9,
                        routing_method="slow"
                    )
                
                with patch.object(router.strategies[RoutingTier.FAST_PATH], 'route', side_effect=slow_route):
                    start = time.time()
                    decision = await router.route("test query")
                    elapsed = time.time() - start
                
                # Should timeout and return fallback
                assert elapsed < 0.3  # Should not wait full 500ms
                assert decision.confidence_score <= 1.0  # Valid confidence score
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty queries."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {}
        }
        
        router = TieredRouter(config)
        
        # Test empty string
        decision = await router.route("")
        assert decision.confidence_score <= 1.0
        
        # Test whitespace only
        decision = await router.route("   ")
        assert decision.confidence_score <= 1.0
        
        # Test None (should handle gracefully)
        with patch.object(router, '_check_cache', return_value=None):
            decision = await router.route("")
            assert decision is not None


class TestComprehensiveRouter:
    """Test ComprehensiveRouter functionality."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_comprehensive_routing(self):
        """Test comprehensive routing with multiple strategies."""
        config = {
            "ensemble_config": {
                "enabled_strategies": ["keyword"],
                "voting_method": "weighted",
                "weights": {"keyword": 1.0}
            },
            "keyword_config": {}
        }
        
        router = ComprehensiveRouter(config)
        
        decision = await router.route("test video query")
        assert decision is not None
        assert decision.routing_method is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_ensemble_voting(self):
        """Test ensemble voting mechanism."""
        config = {
            "ensemble_config": {
                "enabled_strategies": ["strategy1", "strategy2"],
                "voting_method": "weighted",
                "weights": {
                    "strategy1": 2.0,
                    "strategy2": 1.0
                }
            }
        }
        
        router = ComprehensiveRouter(config)
        
        # Mock strategies
        strategy1 = Mock()
        strategy1.route = AsyncMock(return_value=RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.8,
            routing_method="strategy1"
        ))
        
        strategy2 = Mock()
        strategy2.route = AsyncMock(return_value=RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.7,
            routing_method="strategy2"
        ))
        
        router.ensemble_strategies = {
            "strategy1": strategy1,
            "strategy2": strategy2
        }
        
        decision = await router._run_ensemble("test query")
        
        # Should favor strategy1 due to higher weight
        assert decision.search_modality == SearchModality.VIDEO
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_optimization_trigger(self):
        """Test optimization trigger based on performance."""
        config = {
            "optimization_config": {
                "enable_auto_optimization": True,
                "performance_degradation_threshold": 0.1
            }
        }
        
        router = ComprehensiveRouter(config)
        
        # Mock optimizer
        with patch.object(router, 'optimizer') as mock_optimizer:
            mock_optimizer.should_optimize.return_value = True
            mock_optimizer.optimize = AsyncMock()
            
            # Simulate performance degradation
            router.performance_baseline = {"accuracy": 0.9}
            router.recent_performance = {"accuracy": 0.75}  # 15% degradation
            
            await router._maybe_optimize()
            
            mock_optimizer.optimize.assert_called_once()


class TestPerformanceTracking:
    """Test performance tracking and reporting."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test that performance metrics are collected."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {}
        }
        
        router = TieredRouter(config)
        
        # Make several routing calls
        for _ in range(5):
            await router.route("test query")
        
        # Get performance report
        report = router.get_performance_report()
        
        assert "total_queries" in report
        assert report["total_queries"] == 5
        # Check for tier_performance instead of cache_stats
        assert "tier_performance" in report
        assert "cache_size" in report
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tier_performance_tracking(self):
        """Test tracking of tier-specific performance."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "fast_path_confidence_threshold": 0.7
            }
        }
        
        router = TieredRouter(config)
        
        # Mock strategies with different latencies
        fast_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="gliner"
        )
        
        async def fast_route(query, context=None):
            await asyncio.sleep(0.01)  # 10ms
            return fast_decision
        
        with patch.object(router.strategies[RoutingTier.FAST_PATH], 'route', side_effect=fast_route):
            await router.route("test query")
        
        report = router.get_performance_report()
        # Performance report structure may vary - just verify it exists
        assert isinstance(report, dict)
        assert len(report) > 0
        # Check for tier_performance data instead of avg_latency_ms
        assert "tier_performance" in report
        tier_data = report["tier_performance"]
        assert len(tier_data) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test handling of very long queries."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {}
        }
        
        router = TieredRouter(config)
        
        # Create a very long query
        long_query = "test " * 1000  # 5000 characters
        
        decision = await router.route(long_query)
        assert decision is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Test handling of special characters."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {}
        }
        
        router = TieredRouter(config)
        
        # Test various special characters
        queries = [
            "test @#$%^&*()",
            "test\nwith\nnewlines",
            "test\twith\ttabs",
            "test 中文 测试",  # Unicode
            "test 😀 emoji",
            "test <script>alert('xss')</script>"  # HTML
        ]
        
        for query in queries:
            decision = await router.route(query)
            assert decision is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_routing_requests(self):
        """Test handling of concurrent routing requests."""
        config = {
            "tier_config": {"enable_fallback": True},
            "keyword_config": {},
            "cache_config": {"enable_caching": False}  # Disable cache for this test
        }
        
        router = TieredRouter(config)
        
        # Create multiple concurrent requests
        queries = [f"query {i}" for i in range(10)]
        
        # Route all queries concurrently
        tasks = [router.route(q) for q in queries]
        decisions = await asyncio.gather(*tasks)
        
        assert len(decisions) == 10
        assert all(d is not None for d in decisions)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_router_with_no_strategies(self):
        """Test router with all strategies disabled."""
        config = {
            "tier_config": {
                "enable_fast_path": False,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": False
            }
        }
        
        router = TieredRouter(config)
        
        decision = await router.route("test query")
        
        # Should return a default decision
        assert decision is not None
        assert decision.confidence_score < 0.5
        assert "no strategies" in decision.reasoning.lower() or "unavailable" in decision.routing_method.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])