"""
Unit tests for ProcessorManager.

Tests the processor discovery, registration, and management functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import logging
from typing import Dict, Any

from src.app.ingestion.processor_manager import ProcessorManager
from src.app.ingestion.processor_base import BaseProcessor
from src.app.ingestion.processing_strategy_set import ProcessingStrategySet


class MockProcessorA(BaseProcessor):
    """Mock processor A for testing."""
    
    PROCESSOR_NAME = "processor_a"
    
    def __init__(self, logger: logging.Logger, param_a: str = "default_a"):
        super().__init__(logger, param_a=param_a)
        self.param_a = param_a


class MockProcessorB(BaseProcessor):
    """Mock processor B for testing."""
    
    PROCESSOR_NAME = "processor_b"
    
    def __init__(self, logger: logging.Logger, param_b: int = 10):
        super().__init__(logger, param_b=param_b)
        self.param_b = param_b


class TestProcessorManager:
    """Test cases for ProcessorManager."""
    
    @pytest.fixture
    def manager(self, mock_logger):
        """Create a processor manager for testing."""
        with patch('src.app.ingestion.processor_manager.pkgutil.iter_modules') as mock_iter:
            # Mock empty processor discovery to avoid loading real processors
            mock_iter.return_value = []
            manager = ProcessorManager(mock_logger)
            # Manually add test processors
            manager._processor_classes["processor_a"] = MockProcessorA
            manager._processor_classes["processor_b"] = MockProcessorB
            return manager
    
    def test_processor_manager_initialization(self, mock_logger):
        """Test processor manager initialization."""
        with patch('src.app.ingestion.processor_manager.pkgutil.iter_modules') as mock_iter:
            mock_iter.return_value = []
            
            manager = ProcessorManager(mock_logger)
            
            assert manager.logger == mock_logger
            assert isinstance(manager._processors, dict)
            assert isinstance(manager._processor_classes, dict)
    
    def test_processor_discovery_logging(self, mock_logger):
        """Test that processor discovery is logged."""
        with patch('src.app.ingestion.processor_manager.pkgutil.iter_modules') as mock_iter:
            mock_iter.return_value = []
            
            manager = ProcessorManager(mock_logger)
            
            # Should log discovery results
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "Discovered" in call_args
            assert "processor types" in call_args
    
    def test_get_processor_class_existing(self, manager):
        """Test retrieving existing processor class."""
        processor_class = manager.get_processor_class("processor_a")
        
        assert processor_class == MockProcessorA
    
    def test_get_processor_class_nonexistent(self, manager):
        """Test retrieving non-existent processor class returns None."""
        processor_class = manager.get_processor_class("nonexistent")
        
        assert processor_class is None
    
    def test_create_processor_success(self, manager):
        """Test successful processor creation."""
        config = {"param_a": "test_value"}
        
        processor = manager.create_processor("processor_a", config)
        
        assert isinstance(processor, MockProcessorA)
        assert processor.param_a == "test_value"
        assert processor.logger == manager.logger
    
    def test_create_processor_nonexistent(self, manager):
        """Test creating non-existent processor returns None."""
        processor = manager.create_processor("nonexistent", {})
        
        assert processor is None
    
    def test_create_processor_with_defaults(self, manager):
        """Test creating processor uses default configuration."""
        processor = manager.create_processor("processor_a", {})
        
        assert isinstance(processor, MockProcessorA)
        assert processor.param_a == "default_a"
    
    def test_get_processor_creates_and_caches(self, manager):
        """Test that get_processor creates and caches processors."""
        config = {"param_b": 42}
        
        # First call should create processor
        processor1 = manager.get_processor("processor_b", config)
        assert isinstance(processor1, MockProcessorB)
        assert processor1.param_b == 42
        
        # Second call should return cached processor
        processor2 = manager.get_processor("processor_b", config)
        assert processor2 is processor1  # Same instance
    
    def test_get_processor_different_configs(self, manager):
        """Test that different configs create different processors."""
        config1 = {"param_b": 10}
        config2 = {"param_b": 20}
        
        processor1 = manager.get_processor("processor_b", config1)
        processor2 = manager.get_processor("processor_b", config2)
        
        assert processor1 is not processor2
        assert processor1.param_b == 10
        assert processor2.param_b == 20
    
    def test_get_processor_nonexistent(self, manager):
        """Test get_processor returns None for non-existent processors."""
        processor = manager.get_processor("nonexistent", {})
        
        assert processor is None
    
    def test_initialize_from_strategies(self, manager):
        """Test initializing processors from strategy set."""
        from src.app.ingestion.strategies import ChunkSegmentationStrategy
        
        # Mock strategy that requires our test processors
        mock_strategy = Mock()
        mock_strategy.get_required_processors.return_value = {
            "processor_a": {"param_a": "from_strategy"},
            "processor_b": {"param_b": 99}
        }
        
        strategy_set = Mock()
        strategy_set.get_all_required_processors.return_value = {
            "processor_a": {"param_a": "from_strategy"},
            "processor_b": {"param_b": 99}
        }
        
        manager.initialize_from_strategies(strategy_set)
        
        # Should have created processors based on strategy requirements
        processor_a = manager.get_processor("processor_a")
        processor_b = manager.get_processor("processor_b")
        
        assert processor_a is not None
        assert processor_b is not None
        assert processor_a.param_a == "from_strategy"
        assert processor_b.param_b == 99
    
    def test_list_available_processors(self, manager):
        """Test listing available processor types."""
        available = manager.list_available_processors()
        
        assert "processor_a" in available
        assert "processor_b" in available
        assert len(available) >= 2
    
    def test_clear_cache(self, manager):
        """Test clearing processor cache."""
        # Create some cached processors
        manager.get_processor("processor_a", {"param_a": "cached"})
        manager.get_processor("processor_b", {"param_b": 123})
        
        assert len(manager._processors) > 0
        
        manager.clear_cache()
        
        assert len(manager._processors) == 0
    
    def test_processor_cache_key_generation(self, manager):
        """Test that cache keys are generated correctly."""
        # This tests the internal _get_cache_key method indirectly
        config1 = {"param_a": "test"}
        config2 = {"param_a": "test"}
        config3 = {"param_a": "different"}
        
        processor1 = manager.get_processor("processor_a", config1)
        processor2 = manager.get_processor("processor_a", config2)
        processor3 = manager.get_processor("processor_a", config3)
        
        # Same config should return same instance
        assert processor1 is processor2
        # Different config should return different instance
        assert processor1 is not processor3


class TestProcessorDiscovery:
    """Test processor auto-discovery functionality."""
    
    def test_discover_processors_invalid_directory(self, mock_logger):
        """Test discovery with non-existent plugin directory."""
        fake_path = Path("/fake/path/that/does/not/exist")
        
        with patch('src.app.ingestion.processor_manager.pkgutil.iter_modules') as mock_iter:
            mock_iter.return_value = []
            
            manager = ProcessorManager(mock_logger, plugin_dir=fake_path)
            
            # Should log warning about missing directory
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Plugin directory not found" in call_args
    
    @patch('src.app.ingestion.processor_manager.importlib.import_module')
    @patch('src.app.ingestion.processor_manager.pkgutil.iter_modules')
    def test_discover_processors_module_loading(self, mock_iter, mock_import, mock_logger):
        """Test processor discovery loads modules correctly."""
        # Mock module discovery
        mock_importer = Mock()
        mock_iter.return_value = [
            (mock_importer, "test_processor", False),
            (mock_importer, "another_processor", False),
        ]
        
        # Mock module with processor class
        mock_module = Mock()
        mock_processor_class = Mock()
        mock_processor_class.__bases__ = (BaseProcessor,)
        mock_processor_class.PROCESSOR_NAME = "test_processor"
        mock_module.__dict__ = {"TestProcessor": mock_processor_class}
        mock_import.return_value = mock_module
        
        manager = ProcessorManager(mock_logger)
        
        # Should attempt to import modules
        assert mock_import.call_count >= 1
    
    @patch('src.app.ingestion.processor_manager.importlib.import_module')
    @patch('src.app.ingestion.processor_manager.pkgutil.iter_modules')
    def test_discover_processors_handles_import_errors(self, mock_iter, mock_import, mock_logger):
        """Test that import errors during discovery are handled gracefully."""
        mock_importer = Mock()
        mock_iter.return_value = [(mock_importer, "broken_processor", False)]
        
        # Simulate import error
        mock_import.side_effect = ImportError("Module not found")
        
        # Should not raise exception
        manager = ProcessorManager(mock_logger)
        
        # Should log the error
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[0][0]
        assert "Error importing module" in call_args