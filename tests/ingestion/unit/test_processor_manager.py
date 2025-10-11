#!/usr/bin/env python3
"""
Unit tests for ProcessorManager.

Tests the processor discovery, registration, and management functionality.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from cogniverse_runtime.ingestion.processor_base import BaseProcessor, BaseStrategy
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager


class MockProcessorA(BaseProcessor):
    """Mock processor A for testing."""

    PROCESSOR_NAME = "processor_a"

    def __init__(self, logger: logging.Logger, param_a: str = "default_a"):
        super().__init__(logger)
        self.param_a = param_a

    @classmethod
    def from_config(cls, config: dict, logger: logging.Logger):
        # Remove processor_name from config if present (added by ProcessorManager)
        config = config.copy()
        config.pop("processor_name", None)
        return cls(logger, **config)

    def process(self, video_path: Path, output_dir: Path = None, **kwargs):
        return {"result": "processor_a"}


class MockProcessorB(BaseProcessor):
    """Mock processor B for testing."""

    PROCESSOR_NAME = "processor_b"

    def __init__(self, logger: logging.Logger, param_b: int = 10):
        super().__init__(logger)
        self.param_b = param_b

    @classmethod
    def from_config(cls, config: dict, logger: logging.Logger):
        # Remove processor_name from config if present (added by ProcessorManager)
        config = config.copy()
        config.pop("processor_name", None)
        return cls(logger, **config)

    def process(self, video_path: Path, output_dir: Path = None, **kwargs):
        return {"result": "processor_b"}


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MagicMock()


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, required_processors):
        self.required_processors = required_processors

    def get_required_processors(self):
        return self.required_processors


class MockStrategySet:
    """Mock strategy set for testing."""

    def __init__(self, strategies):
        self.strategies = strategies

    def get_all_strategies(self):
        return self.strategies


class TestProcessorManager:
    """Test cases for ProcessorManager."""

    @pytest.fixture
    def manager(self, mock_logger):
        """Create a processor manager for testing."""
        with patch(
            "cogniverse_core.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            # Mock empty processor discovery to avoid loading real processors
            mock_iter.return_value = []
            manager = ProcessorManager(mock_logger)
            # Manually add test processors
            manager._processor_classes["processor_a"] = MockProcessorA
            manager._processor_classes["processor_b"] = MockProcessorB
            return manager

    def test_processor_manager_initialization(self, mock_logger):
        """Test processor manager initialization."""
        with patch(
            "cogniverse_core.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)

            assert manager.logger == mock_logger
            assert isinstance(manager._processors, dict)
            assert isinstance(manager._processor_classes, dict)

    def test_processor_discovery_logging(self, mock_logger):
        """Test that processor discovery is logged."""
        with patch(
            "cogniverse_core.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            ProcessorManager(mock_logger)

            # Should log discovery results
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "Discovered" in call_args
            assert "processor types" in call_args

    def test_list_available_processor_types(self, manager):
        """Test listing available processor types."""
        available_types = manager.list_available_processor_types()

        assert "processor_a" in available_types
        assert "processor_b" in available_types
        assert len(available_types) == 2

    def test_has_processor_not_initialized(self, manager):
        """Test has_processor for non-initialized processor."""
        assert manager.has_processor("processor_a") is False

    def test_get_processor_not_initialized(self, manager):
        """Test get_processor for non-initialized processor."""
        processor = manager.get_processor("processor_a")
        assert processor is None

    def test_initialize_from_strategies(self, manager):
        """Test initializing processors from strategies."""
        strategies = [
            MockStrategy({"processor_a": {"param_a": "test"}}),
            MockStrategy({"processor_b": {"param_b": 20}}),
        ]
        strategy_set = MockStrategySet(strategies)

        manager.initialize_from_strategies(strategy_set)

        # Check processors were created
        assert manager.has_processor("processor_a")
        assert manager.has_processor("processor_b")

        # Get and verify processors
        proc_a = manager.get_processor("processor_a")
        proc_b = manager.get_processor("processor_b")

        assert isinstance(proc_a, MockProcessorA)
        assert isinstance(proc_b, MockProcessorB)
        assert proc_a.param_a == "test"
        assert proc_b.param_b == 20

    def test_initialize_from_strategies_missing_type(self, manager):
        """Test initialize with invalid processor type."""
        strategies = [
            MockStrategy({"nonexistent": {"config": "doesn't matter"}}),
        ]
        strategy_set = MockStrategySet(strategies)

        # Should raise ValueError for unknown processor type
        with pytest.raises(ValueError, match="Unknown processor type: nonexistent"):
            manager.initialize_from_strategies(strategy_set)

    def test_list_processors(self, manager):
        """Test listing initialized processors."""
        # Initially empty
        assert manager.list_processors() == []

        # Initialize some processors
        strategies = [
            MockStrategy({"processor_a": {}}),
            MockStrategy({"processor_b": {}}),
        ]
        strategy_set = MockStrategySet(strategies)
        manager.initialize_from_strategies(strategy_set)

        # Should list initialized processors
        processors = manager.list_processors()
        assert "processor_a" in processors
        assert "processor_b" in processors
        assert len(processors) == 2

    def test_cleanup(self, manager):
        """Test processor cleanup."""
        # Initialize processors
        strategies = [
            MockStrategy({"processor_a": {}}),
            MockStrategy({"processor_b": {}}),
        ]
        strategy_set = MockStrategySet(strategies)
        manager.initialize_from_strategies(strategy_set)

        # Add cleanup method to mock processors
        for proc in manager._processors.values():
            proc.cleanup = Mock()

        manager.cleanup()

        # All processors should have cleanup called
        for proc in manager._processors.values():
            proc.cleanup.assert_called_once()

    def test_processor_discovery_with_plugin_dir(self, mock_logger, tmp_path):
        """Test processor discovery with custom plugin directory."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        with patch(
            "cogniverse_core.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            # Return empty for plugin discovery
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger, plugin_dir=plugin_dir)

            # Should be called once for the plugin dir
            assert mock_iter.call_count == 1

    def test_processor_class_registration(self, manager):
        """Test that processor classes are registered correctly."""
        # Check initial state
        assert "processor_a" in manager._processor_classes
        assert "processor_b" in manager._processor_classes

        # Verify classes are correct
        assert manager._processor_classes["processor_a"] == MockProcessorA
        assert manager._processor_classes["processor_b"] == MockProcessorB

    def test_init_from_requirements(self, manager):
        """Test _init_from_requirements method."""
        requirements = {
            "keyframe": {"param_a": "custom_value"},
            "audio": {"param_b": 30}
        }

        # Map processor names to types for this test
        manager._processor_classes["keyframe"] = MockProcessorA
        manager._processor_classes["audio"] = MockProcessorB

        manager._init_from_requirements(requirements)

        # Check processors were created
        assert manager.has_processor("keyframe")
        assert manager.has_processor("audio")

        keyframe_proc = manager.get_processor("keyframe")
        audio_proc = manager.get_processor("audio")

        assert keyframe_proc.param_a == "custom_value"
        assert audio_proc.param_b == 30


class TestProcessorManagerIntegration:
    """Integration tests for ProcessorManager."""

    def test_full_processor_lifecycle(self, mock_logger):
        """Test complete processor lifecycle from discovery to cleanup."""
        with patch(
            "cogniverse_core.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            # Create manager
            manager = ProcessorManager(mock_logger)
            manager._processor_classes["test_proc"] = MockProcessorA

            # Initialize processor
            strategies = [
                MockStrategy({"test_proc": {"param_a": "lifecycle_test"}})
            ]
            strategy_set = MockStrategySet(strategies)
            manager.initialize_from_strategies(strategy_set)

            # Use processor
            processor = manager.get_processor("test_proc")
            assert processor is not None
            assert processor.param_a == "lifecycle_test"

            # List processors
            processors = manager.list_processors()
            assert "test_proc" in processors

            # Cleanup
            processor.cleanup = Mock()
            manager.cleanup()
            processor.cleanup.assert_called_once()

    def test_multiple_processor_instances(self, mock_logger):
        """Test managing multiple instances of different processor types."""
        with patch(
            "cogniverse_core.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["type_a"] = MockProcessorA
            manager._processor_classes["type_b"] = MockProcessorB

            # Initialize multiple processors
            strategies = [
                MockStrategy({"type_a": {"param_a": "first"}}),
                MockStrategy({"type_b": {"param_b": 100}}),
            ]
            strategy_set = MockStrategySet(strategies)
            manager.initialize_from_strategies(strategy_set)

            # Verify all processors exist
            assert len(manager.list_processors()) == 2
            assert manager.has_processor("type_a")
            assert manager.has_processor("type_b")

            # Verify processors
            proc_a = manager.get_processor("type_a")
            proc_b = manager.get_processor("type_b")

            assert proc_a.param_a == "first"
            assert proc_b.param_b == 100