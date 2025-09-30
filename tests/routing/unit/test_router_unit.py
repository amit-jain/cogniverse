"""
True unit tests for router module with proper mocking.
These tests should run fast and not require external dependencies.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.routing.base import GenerationType, RoutingDecision, SearchModality
from src.app.routing.config_validator import RouterConfigValidator
from src.app.routing.router import ComprehensiveRouter, RoutingTier, TieredRouter


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
                "enable_fallback": True,
            },
            "gliner_config": {"model": "test"},
            "keyword_config": {},
        }

        with patch("src.app.routing.router.GLiNERRoutingStrategy") as mock_gliner:
            with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_keyword:
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
                "enable_fallback": True,
            },
            "keyword_config": {},
        }

        expected_decision = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="keyword",
        )

        with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_strategy:
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
                "fast_path_confidence_threshold": 0.7,
            }
        }

        # Low confidence decision from fast path
        low_confidence = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.5,
            routing_method="gliner",
        )

        # High confidence decision from slow path
        high_confidence = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.8,
            routing_method="llm",
        )

        with patch("src.app.routing.router.GLiNERRoutingStrategy") as mock_gliner:
            with patch("src.app.routing.router.LLMRoutingStrategy") as mock_llm:
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
                "enable_fallback": True,
            },
            "keyword_config": {},
            "cache_config": {"enable_caching": True, "cache_ttl_seconds": 300},
        }

        decision = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.7,
            routing_method="keyword",
        )

        with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_strategy:
            with patch("time.time", return_value=1000.0):  # Mock time
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
                "enable_fallback": True,
            }
        }

        fallback_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.6,
            routing_method="keyword",
        )

        with patch("src.app.routing.router.GLiNERRoutingStrategy") as mock_gliner:
            with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_keyword:
                # Fast path raises exception
                mock_gliner_instance = Mock()
                mock_gliner_instance.route = AsyncMock(
                    side_effect=Exception("Model error")
                )
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
                "voting_method": "weighted",
            },
            "keyword_config": {},
        }

        with patch("src.app.routing.router.KeywordRoutingStrategy"):
            router = ComprehensiveRouter(config)
            assert router.config == config

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_comprehensive_routing_tiered_mode(self):
        """Test comprehensive routing with tiered (non-ensemble) mode."""
        config = {
            "tier_config": {
                "enable_fast_path": False,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": True,
            },
            "keyword_config": {},
        }

        expected_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.7,
            routing_method="keyword",
        )

        with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_strategy:
            mock_instance = Mock()
            mock_instance.route = AsyncMock(return_value=expected_decision)
            mock_strategy.return_value = mock_instance

            router = ComprehensiveRouter(config)
            decision = await router.route("test video query")

            assert decision.routing_method == "keyword"
            assert decision.confidence_score == 0.7
            mock_instance.route.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ensemble_routing_single_strategy(self):
        """Test ensemble routing with single strategy."""
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["keyword"],
                "voting_method": "weighted",
                "timeout_seconds": 5.0,
            },
            "tier_config": {
                "enable_fallback": True,
            },
            "keyword_config": {},
        }

        expected_decision = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="keyword",
            reasoning="Keyword-based routing",
        )

        with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_strategy:
            mock_instance = Mock()
            mock_instance.route = AsyncMock(return_value=expected_decision)
            mock_strategy.return_value = mock_instance

            with patch("src.app.routing.router.GLiNERRoutingStrategy"):
                with patch("src.app.routing.router.LLMRoutingStrategy"):
                    with patch("src.app.routing.router.LangExtractRoutingStrategy"):
                        router = ComprehensiveRouter(config)
                        decision = await router.route("test query")

                        # Should use ensemble routing
                        assert decision.routing_method == "ensemble"
                        assert decision.search_modality == SearchModality.TEXT
                        assert decision.metadata["voting_method"] == "weighted"
                        assert decision.metadata["strategies_used"] == ["keyword"]
                        assert decision.metadata["num_strategies"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ensemble_routing_multiple_strategies(self):
        """Test ensemble routing with multiple strategies voting."""
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["gliner", "keyword"],
                "voting_method": "majority",
                "min_agreement": 0.6,
            },
            "tier_config": {
                "enable_fast_path": True,
                "enable_fallback": True,
            },
            "gliner_config": {},
            "keyword_config": {},
        }

        # GLiNER decision
        gliner_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.9,
            routing_method="gliner",
            reasoning="GLiNER detected video intent",
        )

        # Keyword decision (agrees on modality)
        keyword_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.7,
            routing_method="keyword",
            reasoning="Keywords suggest video search",
        )

        with patch("src.app.routing.router.GLiNERRoutingStrategy") as mock_gliner:
            with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_keyword:
                mock_gliner_instance = Mock()
                mock_gliner_instance.route = AsyncMock(return_value=gliner_decision)
                mock_gliner.return_value = mock_gliner_instance

                mock_keyword_instance = Mock()
                mock_keyword_instance.route = AsyncMock(return_value=keyword_decision)
                mock_keyword.return_value = mock_keyword_instance

                router = ComprehensiveRouter(config)
                decision = await router.route("show me basketball videos")

                # Should use ensemble routing with majority voting
                assert decision.routing_method == "ensemble"
                assert decision.search_modality == SearchModality.VIDEO  # Both agreed
                assert decision.metadata["voting_method"] == "majority"
                assert set(decision.metadata["strategies_used"]) == {
                    "gliner",
                    "keyword",
                }
                assert decision.metadata["num_strategies"] == 2
                assert decision.metadata["agreement_score"] >= 0.6

                # Both strategies should have been called
                mock_gliner_instance.route.assert_called_once()
                mock_keyword_instance.route.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ensemble_routing_with_strategy_failure(self):
        """Test ensemble routing handles strategy failures gracefully."""
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["gliner", "keyword"],
                "voting_method": "weighted",
            },
            "tier_config": {
                "enable_fast_path": True,
                "enable_fallback": True,
            },
            "gliner_config": {},
            "keyword_config": {},
        }

        # Keyword decision (only successful strategy)
        keyword_decision = RoutingDecision(
            search_modality=SearchModality.TEXT,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.6,
            routing_method="keyword",
            reasoning="Fallback to keywords",
        )

        with patch("src.app.routing.router.GLiNERRoutingStrategy") as mock_gliner:
            with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_keyword:
                # GLiNER fails
                mock_gliner_instance = Mock()
                mock_gliner_instance.route = AsyncMock(
                    side_effect=Exception("GLiNER model error")
                )
                mock_gliner.return_value = mock_gliner_instance

                # Keyword succeeds
                mock_keyword_instance = Mock()
                mock_keyword_instance.route = AsyncMock(return_value=keyword_decision)
                mock_keyword.return_value = mock_keyword_instance

                router = ComprehensiveRouter(config)
                decision = await router.route("test query")

                # Should still work with remaining strategy
                assert decision.routing_method == "ensemble"
                assert decision.search_modality == SearchModality.TEXT
                assert decision.metadata["strategies_used"] == ["keyword"]
                assert decision.metadata["num_strategies"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ensemble_routing_timeout_fallback(self):
        """Test ensemble routing falls back to tiered when timeout occurs."""
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["keyword"],
                "voting_method": "weighted",
                "timeout_seconds": 0.1,  # Very short timeout
            },
            "tier_config": {
                "enable_fallback": True,
            },
            "keyword_config": {},
        }

        fallback_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.5,
            routing_method="keyword",
        )

        with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_strategy:
            mock_instance = Mock()

            # Simulate slow response that will timeout
            async def slow_route(query, context):
                await asyncio.sleep(1.0)  # Longer than timeout
                return fallback_decision

            mock_instance.route = AsyncMock(side_effect=slow_route)
            mock_strategy.return_value = mock_instance

            router = ComprehensiveRouter(config)
            decision = await router.route("test query")

            # Should fall back to tiered routing
            assert decision is not None
            assert isinstance(decision, RoutingDecision)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ensemble_routing_weighted_voting(self):
        """Test ensemble routing with weighted voting method."""
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["gliner", "keyword"],
                "voting_method": "weighted",
                "strategy_weights": {
                    "gliner": 2.0,  # Higher weight for GLiNER
                    "keyword": 1.0,
                },
                "min_agreement": 0.4,
            },
            "tier_config": {
                "enable_fast_path": True,
                "enable_fallback": True,
            },
            "gliner_config": {},
            "keyword_config": {},
        }

        gliner_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.8,
            routing_method="gliner",
        )

        keyword_decision = RoutingDecision(
            search_modality=SearchModality.TEXT,  # Disagrees
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.6,
            routing_method="keyword",
        )

        with patch("src.app.routing.router.GLiNERRoutingStrategy") as mock_gliner:
            with patch("src.app.routing.router.KeywordRoutingStrategy") as mock_keyword:
                mock_gliner_instance = Mock()
                mock_gliner_instance.route = AsyncMock(return_value=gliner_decision)
                mock_gliner.return_value = mock_gliner_instance

                mock_keyword_instance = Mock()
                mock_keyword_instance.route = AsyncMock(return_value=keyword_decision)
                mock_keyword.return_value = mock_keyword_instance

                router = ComprehensiveRouter(config)
                decision = await router.route("test query")

                # GLiNER should win due to higher weight (2.0 vs 1.0)
                assert decision.routing_method == "ensemble"
                assert decision.search_modality == SearchModality.VIDEO
                assert decision.metadata["voting_method"] == "weighted"


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
                "enable_fallback": True,
            }
        }

        with patch("src.app.routing.router.KeywordRoutingStrategy"):
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
        legacy_config.enable_langextract = (
            False  # Disable to avoid LangExtract initialization
        )
        legacy_config.enable_fallback = True
        legacy_config.gliner_config = {}
        legacy_config.llm_config = {}
        legacy_config.langextract_config = {}
        legacy_config.keyword_config = {}

        with patch("src.app.routing.router.GLiNERRoutingStrategy"):
            with patch("src.app.routing.router.KeywordRoutingStrategy"):
                router = TieredRouter(legacy_config)
                assert router.config == legacy_config
                assert len(router.strategies) == 2


class TestEnsembleConfigValidation:
    """Unit tests for ensemble configuration validation."""

    @pytest.mark.unit
    def test_valid_ensemble_config(self):
        """Test that valid ensemble configurations are accepted."""
        valid_configs = [
            # Minimal valid config
            {
                "enabled_strategies": ["keyword"],
            },
            # Full valid config
            {
                "enabled": True,
                "enabled_strategies": ["gliner", "keyword"],
                "voting_method": "weighted",
                "min_agreement": 0.6,
                "strategy_weights": {"gliner": 2.0, "keyword": 1.0},
                "timeout_seconds": 10.0,
            },
            # All strategies enabled
            {
                "enabled_strategies": ["gliner", "llm", "langextract", "keyword"],
                "voting_method": "confidence_weighted",
            },
        ]

        for config in valid_configs:
            # Should not raise any exceptions
            RouterConfigValidator.validate_ensemble_config(config)

    @pytest.mark.unit
    def test_missing_enabled_strategies(self):
        """Test that missing enabled_strategies raises ValueError."""
        config = {"enabled": True}

        with pytest.raises(
            ValueError, match="ensemble_config must include 'enabled_strategies' list"
        ):
            RouterConfigValidator.validate_ensemble_config(config)

    @pytest.mark.unit
    def test_empty_enabled_strategies(self):
        """Test that empty enabled_strategies raises ValueError."""
        config = {"enabled_strategies": []}

        with pytest.raises(
            ValueError, match="ensemble_config.enabled_strategies cannot be empty"
        ):
            RouterConfigValidator.validate_ensemble_config(config)

    @pytest.mark.unit
    def test_invalid_enabled_strategies_type(self):
        """Test that non-list enabled_strategies raises ValueError."""
        config = {"enabled_strategies": "keyword"}

        with pytest.raises(
            ValueError, match="ensemble_config.enabled_strategies must be a list"
        ):
            RouterConfigValidator.validate_ensemble_config(config)

    @pytest.mark.unit
    def test_invalid_strategy_names(self):
        """Test that invalid strategy names raise ValueError."""
        invalid_configs = [
            {"enabled_strategies": ["invalid_strategy"]},
            {"enabled_strategies": ["keyword", "bad_strategy"]},
            {"enabled_strategies": [123]},  # Non-string
        ]

        for config in invalid_configs:
            with pytest.raises(
                ValueError, match="Invalid strategy|Strategy name must be a string"
            ):
                RouterConfigValidator.validate_ensemble_config(config)

    @pytest.mark.unit
    def test_invalid_voting_method(self):
        """Test that invalid voting methods raise ValueError."""
        config = {"enabled_strategies": ["keyword"], "voting_method": "invalid_method"}

        with pytest.raises(ValueError, match="Invalid voting_method 'invalid_method'"):
            RouterConfigValidator.validate_ensemble_config(config)

    @pytest.mark.unit
    def test_invalid_voting_method_type(self):
        """Test that non-string voting method raises ValueError."""
        config = {"enabled_strategies": ["keyword"], "voting_method": 123}

        with pytest.raises(
            ValueError, match="ensemble_config.voting_method must be a string"
        ):
            RouterConfigValidator.validate_ensemble_config(config)

    @pytest.mark.unit
    def test_invalid_min_agreement(self):
        """Test that invalid min_agreement values raise ValueError."""
        invalid_configs = [
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "min_agreement": -0.1,  # Below 0.0
                }
            },
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "min_agreement": 1.1,  # Above 1.0
                }
            },
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "min_agreement": "0.5",  # String instead of number
                }
            },
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError, match="ensemble_config.min_agreement"):
                RouterConfigValidator.validate_ensemble_config(
                    config["ensemble_config"]
                )

    @pytest.mark.unit
    def test_invalid_strategy_weights(self):
        """Test that invalid strategy weights raise ValueError."""
        invalid_configs = [
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "strategy_weights": "not_dict",  # Not a dictionary
                }
            },
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "strategy_weights": {
                        "invalid_strategy": 1.0
                    },  # Invalid strategy name
                }
            },
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "strategy_weights": {"keyword": -1.0},  # Negative weight
                }
            },
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "strategy_weights": {"keyword": "1.0"},  # String weight
                }
            },
        ]

        for config in invalid_configs:
            with pytest.raises(
                ValueError,
                match="Strategy weight|ensemble_config.strategy_weights|Invalid strategy",
            ):
                RouterConfigValidator.validate_ensemble_config(
                    config["ensemble_config"]
                )

    @pytest.mark.unit
    def test_invalid_timeout_seconds(self):
        """Test that invalid timeout_seconds values raise ValueError."""
        invalid_configs = [
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "timeout_seconds": -1.0,  # Negative timeout
                }
            },
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "timeout_seconds": 0,  # Zero timeout
                }
            },
            {
                "ensemble_config": {
                    "enabled_strategies": ["keyword"],
                    "timeout_seconds": "10",  # String timeout
                }
            },
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError, match="ensemble_config.timeout_seconds"):
                RouterConfigValidator.validate_ensemble_config(
                    config["ensemble_config"]
                )

    @pytest.mark.unit
    def test_invalid_enabled_flag(self):
        """Test that invalid enabled flag raises ValueError."""
        config = {
            "ensemble_config": {
                "enabled": "yes",  # String instead of boolean
                "enabled_strategies": ["keyword"],
            }
        }

        with pytest.raises(
            ValueError, match="ensemble_config.enabled must be a boolean"
        ):
            RouterConfigValidator.validate_ensemble_config(config["ensemble_config"])

    @pytest.mark.unit
    def test_no_ensemble_config_passes_validation(self):
        """Test that None/empty ensemble_config passes validation."""
        # Should not raise any exceptions
        RouterConfigValidator.validate_ensemble_config(None)
        RouterConfigValidator.validate_ensemble_config({})

    @pytest.mark.unit
    def test_router_uses_validator_on_initialization(self):
        """Test that router calls validator during initialization."""
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["keyword"],
            }
        }

        with patch("src.app.routing.router.KeywordRoutingStrategy"):
            # Should not raise any exceptions - validation should pass
            router = ComprehensiveRouter(config)
            assert router is not None

    @pytest.mark.unit
    def test_router_without_ensemble_config_passes_validation(self):
        """Test that router works correctly without ensemble_config."""
        config = {
            "tier_config": {
                "enable_fast_path": False,
                "enable_slow_path": False,
                "enable_langextract": False,
                "enable_fallback": True,
            }
        }

        with patch("src.app.routing.router.KeywordRoutingStrategy"):
            # Should not raise any exceptions - router initializes without ensemble config
            router = ComprehensiveRouter(config)
            assert router is not None
            assert router._get_ensemble_config() is None
