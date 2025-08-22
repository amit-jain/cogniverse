#!/usr/bin/env python3
"""
StrategyFactory - Creates strategy sets from explicit configuration.

Uses actual class names in config - no string mappings or if/elif logic.
"""

import importlib
from typing import Any

from .processing_strategy_set import ProcessingStrategySet
from .processor_base import BaseStrategy


class StrategyFactory:
    """Factory for creating strategy sets from explicit configuration."""

    @classmethod
    def create_from_profile_config(
        cls, profile_config: dict[str, Any]
    ) -> ProcessingStrategySet:
        """
        Create strategy set from profile configuration.

        Expects config format:
        {
          "strategies": {
            "segmentation": {
              "class": "FrameSegmentationStrategy",
              "params": {"fps": 1.0}
            },
            "transcription": {
              "class": "AudioTranscriptionStrategy",
              "params": {"model": "whisper-large-v3"}
            }
          }
        }

        Args:
            profile_config: Profile configuration dict

        Returns:
            ProcessingStrategySet with configured strategies
        """
        strategies_config = profile_config.get("strategies", {})
        strategies = {}

        for strategy_type, strategy_config in strategies_config.items():
            class_name = strategy_config.get("class")
            params = strategy_config.get("params", {})

            if class_name:
                strategy_instance = cls._create_strategy_instance(class_name, params)
                if strategy_instance:
                    strategies[strategy_type] = strategy_instance

        return ProcessingStrategySet(**strategies)

    @classmethod
    def _create_strategy_instance(
        cls, class_name: str, params: dict[str, Any]
    ) -> BaseStrategy:
        """
        Create strategy instance from class name and parameters.

        Args:
            class_name: Name of strategy class (e.g., "FrameSegmentationStrategy")
            params: Parameters to pass to strategy constructor

        Returns:
            Strategy instance or None if creation failed
        """
        try:
            # Import the strategies module
            strategies_module = importlib.import_module("src.app.ingestion.strategies")

            # Get the class
            strategy_class = getattr(strategies_module, class_name)

            # Create instance with parameters
            return strategy_class(**params)

        except (ImportError, AttributeError, TypeError) as e:
            print(f"Failed to create strategy {class_name}: {e}")
            return None
