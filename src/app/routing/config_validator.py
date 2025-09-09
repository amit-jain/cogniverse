"""Configuration validation for routing components."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RouterConfigValidator:
    """Validates router configuration to ensure proper structure and values."""

    @staticmethod
    def validate_ensemble_config(ensemble_config: dict[str, Any] | None) -> None:
        """
        Validate ensemble configuration if present.
        
        Args:
            ensemble_config: The ensemble configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not ensemble_config:
            return
            
        # Validate enabled flag
        if "enabled" in ensemble_config:
            if not isinstance(ensemble_config["enabled"], bool):
                raise ValueError("ensemble_config.enabled must be a boolean")
        
        # Validate enabled_strategies
        if "enabled_strategies" not in ensemble_config:
            raise ValueError("ensemble_config must include 'enabled_strategies' list")
            
        enabled_strategies = ensemble_config["enabled_strategies"]
        if not isinstance(enabled_strategies, list):
            raise ValueError("ensemble_config.enabled_strategies must be a list")
            
        if not enabled_strategies:
            raise ValueError("ensemble_config.enabled_strategies cannot be empty")
            
        # Validate strategy names
        valid_strategies = {"gliner", "llm", "langextract", "keyword"}
        for strategy in enabled_strategies:
            if not isinstance(strategy, str):
                raise ValueError(f"Strategy name must be a string, got: {type(strategy)}")
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid strategy '{strategy}'. Valid strategies: {valid_strategies}")
        
        # Validate voting_method
        voting_method = ensemble_config.get("voting_method", "weighted")
        if not isinstance(voting_method, str):
            raise ValueError("ensemble_config.voting_method must be a string")
            
        valid_voting_methods = {"majority", "weighted", "confidence_weighted"}
        if voting_method not in valid_voting_methods:
            raise ValueError(f"Invalid voting_method '{voting_method}'. Valid methods: {valid_voting_methods}")
        
        # Validate min_agreement
        if "min_agreement" in ensemble_config:
            min_agreement = ensemble_config["min_agreement"]
            if not isinstance(min_agreement, (int, float)):
                raise ValueError("ensemble_config.min_agreement must be a number")
            if not (0.0 <= min_agreement <= 1.0):
                raise ValueError("ensemble_config.min_agreement must be between 0.0 and 1.0")
        
        # Validate strategy_weights if present
        if "strategy_weights" in ensemble_config:
            strategy_weights = ensemble_config["strategy_weights"]
            if not isinstance(strategy_weights, dict):
                raise ValueError("ensemble_config.strategy_weights must be a dictionary")
            
            for strategy, weight in strategy_weights.items():
                if not isinstance(strategy, str):
                    raise ValueError(f"Strategy weight key must be a string, got: {type(strategy)}")
                if strategy not in valid_strategies:
                    raise ValueError(f"Invalid strategy in weights '{strategy}'. Valid strategies: {valid_strategies}")
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Strategy weight must be a number, got: {type(weight)}")
                if weight <= 0:
                    raise ValueError(f"Strategy weight must be positive, got: {weight}")
        
        # Validate timeout_seconds
        if "timeout_seconds" in ensemble_config:
            timeout = ensemble_config["timeout_seconds"]
            if not isinstance(timeout, (int, float)):
                raise ValueError("ensemble_config.timeout_seconds must be a number")
            if timeout <= 0:
                raise ValueError("ensemble_config.timeout_seconds must be positive")
        
        logger.debug("Ensemble configuration validation passed")
