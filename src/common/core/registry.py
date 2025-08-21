"""
Central Strategy Registry - Single point of access for all strategy information.

This registry provides:
1. Centralized strategy management
2. Validation of strategy configurations
3. Easy access to strategy details without hardcoding
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

from src.app.ingestion.strategy import StrategyConfig, Strategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Singleton registry for all strategy configurations.
    This is THE central place to get strategy information.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.strategy_config = StrategyConfig()
            self._initialized = True
            logger.info("StrategyRegistry initialized")
    
    def get_strategy(self, profile_name: str) -> Strategy:
        """
        Get complete strategy for a profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Complete Strategy object with all configuration
            
        Raises:
            ValueError: If profile not found
        """
        return self.strategy_config.get_strategy(profile_name)
    
    def get_ranking_strategy_config(self, profile_name: str, ranking_name: str) -> Dict[str, Any]:
        """
        Get specific ranking strategy configuration.
        
        Args:
            profile_name: Name of the profile
            ranking_name: Name of the ranking strategy
            
        Returns:
            Complete ranking strategy configuration
            
        Raises:
            ValueError: If profile or ranking strategy not found
        """
        strategy = self.get_strategy(profile_name)
        
        if ranking_name not in strategy.ranking_strategies:
            available = list(strategy.ranking_strategies.keys())
            raise ValueError(
                f"Ranking strategy '{ranking_name}' not found for profile '{profile_name}'. "
                f"Available strategies: {available}"
            )
        
        return strategy.ranking_strategies[ranking_name]
    
    def validate_ranking_strategy(self, profile_name: str, ranking_name: str) -> bool:
        """
        Validate if a ranking strategy is valid for a profile.
        
        Args:
            profile_name: Name of the profile
            ranking_name: Name of the ranking strategy
            
        Returns:
            True if valid, raises exception otherwise
        """
        try:
            self.get_ranking_strategy_config(profile_name, ranking_name)
            return True
        except ValueError as e:
            logger.error(f"Invalid ranking strategy: {e}")
            raise
    
    def get_query_requirements(self, profile_name: str, ranking_name: str) -> Dict[str, Any]:
        """
        Get query requirements for a specific ranking strategy.
        
        Returns dict with:
        - needs_embeddings: bool
        - needs_text: bool
        - query_tensors: dict of tensor names and types
        - nearestNeighbor: dict with field and tensor if applicable
        """
        rank_config = self.get_ranking_strategy_config(profile_name, ranking_name)
        
        return {
            "needs_embeddings": (
                rank_config.get("needs_float_embeddings", False) or 
                rank_config.get("needs_binary_embeddings", False)
            ),
            "needs_float": rank_config.get("needs_float_embeddings", False),
            "needs_binary": rank_config.get("needs_binary_embeddings", False),
            "needs_text": rank_config.get("needs_text_query", False),
            "query_tensors": rank_config.get("inputs", {}),
            "use_nearestneighbor": rank_config.get("use_nearestneighbor", False),
            "nearestneighbor_config": {
                "field": rank_config.get("nearestneighbor_field"),
                "tensor": rank_config.get("nearestneighbor_tensor")
            } if rank_config.get("use_nearestneighbor") else None
        }
    
    def get_default_ranking_strategy(self, profile_name: str) -> str:
        """Get the default ranking strategy for a profile."""
        strategy = self.get_strategy(profile_name)
        return strategy.default_ranking
    
    def list_profiles(self) -> list:
        """List all available profiles."""
        return list(self.strategy_config.config.get("video_processing_profiles", {}).keys())
    
    def list_ranking_strategies(self, profile_name: str) -> list:
        """List all ranking strategies for a profile."""
        strategy = self.get_strategy(profile_name)
        return list(strategy.ranking_strategies.keys())
    
    def reload(self):
        """Reload all configurations from disk."""
        self.strategy_config = StrategyConfig()
        logger.info("StrategyRegistry reloaded")


# Global instance for easy access - lazily initialized
_registry = None

def get_registry() -> StrategyRegistry:
    """Get the global StrategyRegistry instance."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry