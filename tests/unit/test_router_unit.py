"""
True unit tests for router module with proper mocking.
These tests should run fast and not require external dependencies.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.routing.router import TieredRouter, ComprehensiveRouter, RoutingTier
from src.app.routing.base import RoutingDecision, SearchModality, GenerationType


class TestTieredRouterUnit:
    """Unit tests for TieredRouter with mocked dependencies."""
    
    @pytest.mark.unit
    def test_router_initialization_with_dict_config(self):
        """Test router initialization with dictionary config."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": True
            },
            "gliner_config": {"model": "test"},
            "keyword_config": {}
        }
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy') as mock_gliner:
            with patch('src.app.routing.router.KeywordRoutingStrategy') as mock_keyword:
                router = TieredRouter(config)
                
                assert router.config == config
                assert RoutingTier.FAST_PATH in router.strategies
                assert RoutingTier.FALLBACK in router.strategies
                assert RoutingTier.SLOW_PATH not in router.strategies
                assert RoutingTier.LANGEXTRACT not in router.strategies
                mock_gliner.assert_called_once()
                mock_keyword.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_routing_with_mocked_strategies(self):
        """Test routing behavior with mocked strategies."""
        config = {
            "tier_config": {
                "enable_fast_path": False,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": True
            },
            "keyword_config": {}
        }
        
        expected_decision = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="keyword"
        )
        
        with patch('src.app.routing.router.KeywordRoutingStrategy') as mock_strategy:
            mock_instance = Mock()
            mock_instance.route = AsyncMock(return_value=expected_decision)
            mock_strategy.return_value = mock_instance
            
            router = TieredRouter(config)
            decision = await router.route("test query")
            
            assert decision.routing_method == "keyword"
            assert decision.confidence_score == 0.8
            mock_instance.route.assert_called_once_with("test query", None)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tier_escalation_logic(self):
        """Test tier escalation with mocked strategies."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_langextract": False,
                "enable_fallback": False,
                "fast_path_confidence_threshold": 0.7
            }
        }
        
        # Low confidence decision from fast path
        low_confidence = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.5,
            routing_method="gliner"
        )
        
        # High confidence decision from slow path
        high_confidence = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.8,
            routing_method="llm"
        )
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy') as mock_gliner:
            with patch('src.app.routing.router.LLMRoutingStrategy') as mock_llm:
                # Setup mocked strategies
                mock_gliner_instance = Mock()
                mock_gliner_instance.route = AsyncMock(return_value=low_confidence)
                mock_gliner.return_value = mock_gliner_instance
                
                mock_llm_instance = Mock()
                mock_llm_instance.route = AsyncMock(return_value=high_confidence)
                mock_llm.return_value = mock_llm_instance
                
                router = TieredRouter(config)
                decision = await router.route("complex query")
                
                # Should escalate to slow path due to low fast path confidence
                assert decision.routing_method == "llm"
                assert decision.confidence_score == 0.8
                
                # Verify both strategies were called
                mock_gliner_instance.route.assert_called_once()
                mock_llm_instance.route.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_caching_behavior_mocked(self):
        """Test caching behavior with mocked time and strategies."""
        config = {
            "tier_config": {
                "enable_fast_path": False,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": True
            },
            "keyword_config": {},
            "cache_config": {
                "enable_caching": True,
                "cache_ttl_seconds": 300
            }
        }
        
        decision = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.7,
            routing_method="keyword"
        )
        
        with patch('src.app.routing.router.KeywordRoutingStrategy') as mock_strategy:
            with patch('time.time', return_value=1000.0):  # Mock time
                mock_instance = Mock()
                mock_instance.route = AsyncMock(return_value=decision)
                mock_strategy.return_value = mock_instance
                
                router = TieredRouter(config)
                
                # First call - should hit strategy
                decision1 = await router.route("test query")
                assert mock_instance.route.call_count == 1
                
                # Second call - should hit cache
                decision2 = await router.route("test query")
                assert mock_instance.route.call_count == 1  # No additional calls
                
                assert decision1.routing_method == decision2.routing_method
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_with_mocked_strategies(self):
        """Test error handling when strategies fail."""
        config = {
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": True
            }
        }
        
        fallback_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.6,
            routing_method="keyword"
        )
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy') as mock_gliner:
            with patch('src.app.routing.router.KeywordRoutingStrategy') as mock_keyword:
                # Fast path raises exception
                mock_gliner_instance = Mock()
                mock_gliner_instance.route = AsyncMock(side_effect=Exception("Model error"))
                mock_gliner.return_value = mock_gliner_instance
                
                # Fallback succeeds
                mock_keyword_instance = Mock()
                mock_keyword_instance.route = AsyncMock(return_value=fallback_decision)
                mock_keyword.return_value = mock_keyword_instance
                
                router = TieredRouter(config)
                decision = await router.route("test query")
                
                # Should fall back to keyword strategy
                assert decision.routing_method == "keyword"
                assert decision.confidence_score == 0.6


class TestComprehensiveRouterUnit:
    """Unit tests for ComprehensiveRouter with mocked dependencies."""
    
    @pytest.mark.unit
    def test_comprehensive_router_initialization(self):
        """Test ComprehensiveRouter initialization with mocked strategies."""
        config = {
            "ensemble_config": {
                "enabled_strategies": ["keyword"],
                "voting_method": "weighted"
            },
            "keyword_config": {}
        }
        
        with patch('src.app.routing.router.KeywordRoutingStrategy'):
            router = ComprehensiveRouter(config)
            assert router.config == config
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_comprehensive_routing_mocked(self):
        """Test comprehensive routing with mocked strategies."""
        config = {
            "ensemble_config": {
                "enabled_strategies": ["keyword"],
                "voting_method": "weighted"
            },
            "keyword_config": {}
        }
        
        # For now, just test that it can route with basic config
        # TODO: Implement actual ensemble functionality
        with patch('src.app.routing.router.KeywordRoutingStrategy'):
            router = ComprehensiveRouter(config)
            
            # ComprehensiveRouter should inherit basic routing from TieredRouter
            decision = await router.route("test video query")
            
            assert decision is not None
            assert isinstance(decision, RoutingDecision)


class TestRouterConfigHandling:
    """Unit tests for router configuration handling."""
    
    @pytest.mark.unit
    def test_config_validation_and_defaults(self):
        """Test config validation and default values."""
        # Test with minimal config - disable all other strategies
        minimal_config = {
            "tier_config": {
                "enable_fast_path": False,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": True
            }
        }
        
        with patch('src.app.routing.router.KeywordRoutingStrategy'):
            router = TieredRouter(minimal_config)
            assert router.config == minimal_config
            
            # Should only have fallback strategy enabled
            assert RoutingTier.FALLBACK in router.strategies
            assert len(router.strategies) == 1
    
    @pytest.mark.unit
    def test_legacy_config_support(self):
        """Test support for legacy config format."""
        legacy_config = Mock(spec=[])  # No tier_config attribute
        legacy_config.enable_fast_path = True
        legacy_config.enable_slow_path = False  # Disable to avoid LLM initialization
        legacy_config.enable_langextract = False  # Disable to avoid LangExtract initialization
        legacy_config.enable_fallback = True
        legacy_config.gliner_config = {}
        legacy_config.llm_config = {}
        legacy_config.langextract_config = {}
        legacy_config.keyword_config = {}
        
        with patch('src.app.routing.router.GLiNERRoutingStrategy'):
            with patch('src.app.routing.router.KeywordRoutingStrategy'):
                router = TieredRouter(legacy_config)
                assert router.config == legacy_config
                assert len(router.strategies) == 2