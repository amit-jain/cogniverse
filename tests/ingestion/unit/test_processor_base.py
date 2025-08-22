"""
Unit tests for processor base classes.

Tests the core processor infrastructure including BaseProcessor
and BaseStrategy classes.
"""

import logging
from typing import Any

import pytest

from src.app.ingestion.processor_base import BaseProcessor, BaseStrategy


class MockTestProcessor(BaseProcessor):
    """Test processor implementation."""

    PROCESSOR_NAME = "test"

    def __init__(
        self, logger: logging.Logger, param1: str = "default", param2: int = 10
    ):
        super().__init__(logger, param1=param1, param2=param2)
        self.param1 = param1
        self.param2 = param2

    def process(self, data):
        """Mock process method."""
        return f"processed_{data}"


class MockTestStrategy(BaseStrategy):
    """Test strategy implementation."""

    def __init__(self):
        self.strategy_name = "test_strategy"

    def get_required_processors(self) -> dict[str, dict[str, Any]]:
        return {"test": {"param1": "test_value", "param2": 20}}


class TestBaseProcessor:
    """Test cases for BaseProcessor base class."""

    def test_processor_name_required(self, mock_logger):
        """Test that PROCESSOR_NAME is required."""

        class InvalidProcessor(BaseProcessor):
            pass  # Missing PROCESSOR_NAME

        with pytest.raises(ValueError, match="must define PROCESSOR_NAME"):
            InvalidProcessor(mock_logger)

    def test_processor_initialization(self, mock_logger):
        """Test processor initialization with parameters."""
        processor = MockTestProcessor(mock_logger, param1="test", param2=42)

        assert processor.PROCESSOR_NAME == "test"
        assert processor.logger == mock_logger
        assert processor.param1 == "test"
        assert processor.param2 == 42
        assert processor._config == {"param1": "test", "param2": 42}

    def test_from_config_factory_method(self, mock_logger):
        """Test creating processor from configuration."""
        config = {"param1": "from_config", "param2": 99}

        processor = MockTestProcessor.from_config(config, mock_logger)

        assert processor.param1 == "from_config"
        assert processor.param2 == 99
        assert processor.logger == mock_logger

    def test_from_config_with_defaults(self, mock_logger):
        """Test from_config uses defaults for missing parameters."""
        config = {"param1": "partial_config"}  # param2 missing

        processor = MockTestProcessor.from_config(config, mock_logger)

        assert processor.param1 == "partial_config"
        assert processor.param2 == 10  # default value

    def test_from_config_with_extra_params(self, mock_logger):
        """Test from_config ignores extra parameters."""
        config = {"param1": "valid", "param2": 55, "extra_param": "ignored"}

        processor = MockTestProcessor.from_config(config, mock_logger)

        assert processor.param1 == "valid"
        assert processor.param2 == 55
        # extra_param should be ignored, not cause error

    def test_get_config(self, mock_logger):
        """Test retrieving processor configuration."""
        processor = MockTestProcessor(mock_logger, param1="test", param2=42)

        config = processor.get_config()

        assert config == {"param1": "test", "param2": 42}

    def test_repr(self, mock_logger):
        """Test string representation of processor."""
        processor = MockTestProcessor(mock_logger, param1="test", param2=42)

        repr_str = repr(processor)

        assert "MockTestProcessor" in repr_str
        assert "test" in repr_str


class TestBaseStrategy:
    """Test cases for BaseStrategy base class."""

    def test_strategy_abstract_implementation(self):
        """Test that strategy must implement abstract methods."""

        class IncompleteStrategy(BaseStrategy):
            pass  # Missing get_required_processors

        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_get_required_processors(self):
        """Test getting required processors from strategy."""
        strategy = MockTestStrategy()

        required = strategy.get_required_processors()

        assert "test" in required
        assert required["test"]["param1"] == "test_value"
        assert required["test"]["param2"] == 20

    def test_repr(self):
        """Test string representation of strategy."""
        strategy = MockTestStrategy()

        repr_str = repr(strategy)

        assert "MockTestStrategy" in repr_str
        assert "MockTestStrategy" in repr_str


class TestProcessorFactoryMethods:
    """Test processor factory methods and introspection."""

    def test_from_config_parameter_mapping(self, mock_logger):
        """Test that from_config correctly maps parameters using introspection."""

        class ComplexProcessor(BaseProcessor):
            PROCESSOR_NAME = "complex"

            def __init__(
                self,
                logger: logging.Logger,
                required_param: str,
                optional_param: int = 100,
                **kwargs,
            ):
                super().__init__(logger, **kwargs)
                self.required_param = required_param
                self.optional_param = optional_param

        config = {
            "required_param": "must_have",
            "optional_param": 200,
            "extra_config": "ignored",
        }

        processor = ComplexProcessor.from_config(config, mock_logger)

        assert processor.required_param == "must_have"
        assert processor.optional_param == 200

    def test_from_config_missing_required_param(self, mock_logger):
        """Test from_config handles missing required parameters gracefully."""

        class RequiredParamProcessor(BaseProcessor):
            PROCESSOR_NAME = "required"

            def __init__(self, logger: logging.Logger, required_param: str):
                super().__init__(logger)
                self.required_param = required_param

        config = {}  # Missing required_param

        with pytest.raises(TypeError):
            RequiredParamProcessor.from_config(config, mock_logger)

    def test_processor_config_persistence(self, mock_logger):
        """Test that processor configuration is properly stored."""
        processor = MockTestProcessor(mock_logger, param1="stored", param2=123)

        # Config should persist through object lifecycle
        config = processor.get_config()
        assert config["param1"] == "stored"
        assert config["param2"] == 123

        # Config should be independent of instance variables
        processor.param1 = "changed"
        config = processor.get_config()
        assert config["param1"] == "stored"  # Original config unchanged
