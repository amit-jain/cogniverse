"""
Basic unit tests for BaseProcessor to improve coverage.

Tests the base processor functionality and configuration system.
"""

import pytest
from unittest.mock import Mock
import inspect

from src.app.ingestion.processor_base import BaseProcessor, BaseStrategy


class TestProcessor(BaseProcessor):
    """Test processor implementation for testing BaseProcessor."""
    PROCESSOR_NAME = "test"
    
    def __init__(self, logger, param1="default1", param2="default2", **kwargs):
        super().__init__(logger, **kwargs)
        self.param1 = param1
        self.param2 = param2


class TestProcessorNoName(BaseProcessor):
    """Test processor without PROCESSOR_NAME for testing validation."""
    pass


class TestBaseProcessor:
    """Tests for BaseProcessor."""
    
    @pytest.fixture
    def mock_logger(self):
        return Mock()
    
    def test_processor_initialization_success(self, mock_logger):
        """Test successful processor initialization."""
        processor = TestProcessor(mock_logger, param1="test1", param2="test2")
        
        assert processor.PROCESSOR_NAME == "test"
        assert processor.logger == mock_logger
        assert processor.param1 == "test1"
        assert processor.param2 == "test2"
    
    def test_processor_initialization_missing_name(self, mock_logger):
        """Test processor initialization fails without PROCESSOR_NAME."""
        with pytest.raises(ValueError, match="must define PROCESSOR_NAME"):
            TestProcessorNoName(mock_logger)
    
    def test_from_config_with_all_params(self, mock_logger):
        """Test from_config with all parameters provided."""
        config = {
            "param1": "config1",
            "param2": "config2"
        }
        
        processor = TestProcessor.from_config(config, mock_logger)
        
        assert processor.param1 == "config1"
        assert processor.param2 == "config2"
        assert processor.logger == mock_logger
    
    def test_from_config_with_partial_params(self, mock_logger):
        """Test from_config with partial parameters uses defaults."""
        config = {"param1": "config1"}
        
        processor = TestProcessor.from_config(config, mock_logger)
        
        assert processor.param1 == "config1"
        assert processor.param2 == "default2"  # default value
    
    def test_from_config_with_extra_params(self, mock_logger):
        """Test from_config ignores extra config parameters."""
        config = {
            "param1": "config1",
            "param2": "config2",
            "extra_param": "ignored"
        }
        
        processor = TestProcessor.from_config(config, mock_logger)
        
        assert processor.param1 == "config1"
        assert processor.param2 == "config2"
        # extra_param should be ignored
        assert not hasattr(processor, 'extra_param')
    
    def test_get_processor_name(self, mock_logger):
        """Test get_processor_name method."""
        processor = TestProcessor(mock_logger)
        assert processor.get_processor_name() == "test"
    
    def test_get_config(self, mock_logger):
        """Test get_config method returns kwargs."""
        extra_kwargs = {"extra1": "value1", "extra2": "value2"}
        processor = TestProcessor(mock_logger, **extra_kwargs)
        
        config = processor.get_config()
        
        assert config["extra1"] == "value1"
        assert config["extra2"] == "value2"
        # Should not include logger
        assert "logger" not in config


class TestBaseStrategy:
    """Tests for BaseStrategy."""
    
    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy()
    
    def test_get_required_processors_is_abstract(self):
        """Test that get_required_processors is abstract."""
        assert hasattr(BaseStrategy, 'get_required_processors')
        # Check that it's marked as abstract
        assert getattr(BaseStrategy.get_required_processors, '__isabstractmethod__', False)