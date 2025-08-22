"""
Integration tests for pipeline orchestration.

Tests the coordination between ProcessorManager, strategies,
and individual processors in the ingestion pipeline.
"""

from unittest.mock import Mock, patch

import pytest

from src.app.ingestion.processing_strategy_set import ProcessingStrategySet
from src.app.ingestion.processor_manager import ProcessorManager
from src.app.ingestion.strategies import (
    ChunkSegmentationStrategy,
    FrameSegmentationStrategy,
)


@pytest.mark.integration
class TestPipelineOrchestration:
    """Integration tests for pipeline orchestration."""

    @pytest.fixture
    def strategy_set(self):
        """Create a strategy set for testing."""
        frame_strategy = FrameSegmentationStrategy(max_frames=10, fps=1.0)

        strategy_set = ProcessingStrategySet(segmentation=frame_strategy)
        return strategy_set

    @pytest.fixture
    def chunk_strategy_set(self):
        """Create a chunk-based strategy set."""
        chunk_strategy = ChunkSegmentationStrategy(
            chunk_duration=15.0, chunk_overlap=2.0, cache_chunks=False
        )

        strategy_set = ProcessingStrategySet(segmentation=chunk_strategy)
        return strategy_set

    def test_processor_manager_strategy_integration(self, mock_logger, strategy_set):
        """Test that ProcessorManager correctly integrates with strategies."""
        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)

            # Manually add processors for testing (since we mocked discovery)
            from tests.ingestion.conftest import MockProcessor

            manager._processor_classes["keyframe"] = MockProcessor
            manager._processor_classes["audio"] = MockProcessor
            manager._processor_classes["chunk"] = MockProcessor

            # Initialize from strategies
            manager.initialize_from_strategies(strategy_set)

            # Should have created processors based on strategy requirements
            required_processors = strategy_set.get_all_required_processors()
            for processor_name in required_processors:
                processor = manager.get_processor(processor_name)
                assert processor is not None
                assert processor.name == processor_name

    def test_frame_strategy_processor_requirements(self, mock_logger):
        """Test that frame strategy requires correct processors."""
        frame_strategy = FrameSegmentationStrategy(max_frames=50)
        strategy_set = ProcessingStrategySet(segmentation=frame_strategy)

        requirements = strategy_set.get_all_required_processors()

        # Frame strategy should require keyframe processor
        assert "keyframe" in requirements
        keyframe_config = requirements["keyframe"]
        assert keyframe_config["max_frames"] == 50
        assert (
            keyframe_config.get("fps") is not None
            or keyframe_config.get("max_frames") == 50
        )

    def test_chunk_strategy_processor_requirements(self, mock_logger):
        """Test that chunk strategy requires correct processors."""
        chunk_strategy = ChunkSegmentationStrategy(
            chunk_duration=30.0, chunk_overlap=5.0, cache_chunks=True
        )
        strategy_set = ProcessingStrategySet(segmentation=chunk_strategy)

        requirements = strategy_set.get_all_required_processors()

        # Chunk strategy should require chunk processor
        assert "chunk" in requirements
        chunk_config = requirements["chunk"]
        assert chunk_config["chunk_duration"] == 30.0
        assert chunk_config["chunk_overlap"] == 5.0
        assert chunk_config["cache_chunks"] is True

    def test_multiple_strategies_processor_requirements(self, mock_logger):
        """Test combining multiple strategies in a strategy set."""
        frame_strategy = FrameSegmentationStrategy(max_frames=25)
        ChunkSegmentationStrategy(chunk_duration=20.0)

        # Note: This tests the theoretical case where both strategies might be used
        # In practice, segmentation strategies are typically mutually exclusive
        strategy_set = ProcessingStrategySet(segmentation=frame_strategy)

        # For now, test with just frame strategy
        requirements = strategy_set.get_all_required_processors()
        assert "keyframe" in requirements

    @patch("src.app.ingestion.processors.keyframe_processor.KeyframeProcessor")
    def test_end_to_end_frame_processing_workflow(
        self,
        mock_keyframe_class,
        mock_logger,
        strategy_set,
        temp_dir,
        sample_video_path,
    ):
        """Test complete workflow from strategy to processor execution."""
        # Mock the keyframe processor
        mock_keyframe = Mock()
        mock_keyframe.PROCESSOR_NAME = "keyframe"
        mock_keyframe_result = {
            "video_id": "test_video",
            "total_keyframes": 3,
            "fps": 1.0,
            "keyframes": [
                {"frame_index": 0, "timestamp": 0.0, "filename": "frame_0.jpg"},
                {"frame_index": 30, "timestamp": 1.0, "filename": "frame_1.jpg"},
                {"frame_index": 60, "timestamp": 2.0, "filename": "frame_2.jpg"},
            ],
        }
        mock_keyframe.extract_keyframes.return_value = mock_keyframe_result
        mock_keyframe_class.return_value = mock_keyframe
        mock_keyframe_class.from_config = Mock(return_value=mock_keyframe)

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            # Create manager and initialize from strategies
            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = mock_keyframe_class
            manager.initialize_from_strategies(strategy_set)

            # Get the configured processor
            keyframe_processor = manager.get_processor("keyframe")
            assert keyframe_processor is not None

            # Execute processing
            result = keyframe_processor.extract_keyframes(sample_video_path, temp_dir)

            # Verify result
            assert result == mock_keyframe_result
            assert result["total_keyframes"] == 3
            mock_keyframe.extract_keyframes.assert_called_once_with(
                sample_video_path, temp_dir
            )

    @patch("src.app.ingestion.processors.chunk_processor.ChunkProcessor")
    def test_end_to_end_chunk_processing_workflow(
        self,
        mock_chunk_class,
        mock_logger,
        chunk_strategy_set,
        temp_dir,
        sample_video_path,
    ):
        """Test complete workflow for chunk-based processing."""
        # Mock the chunk processor
        mock_chunk = Mock()
        mock_chunk.PROCESSOR_NAME = "chunk"
        mock_chunk_result = {
            "video_id": "test_video",
            "total_chunks": 2,
            "chunks": [
                {"start_time": 0.0, "end_time": 15.0, "filename": "chunk_0.mp4"},
                {"start_time": 13.0, "end_time": 28.0, "filename": "chunk_1.mp4"},
            ],
        }
        mock_chunk.extract_chunks.return_value = mock_chunk_result
        mock_chunk_class.return_value = mock_chunk
        mock_chunk_class.from_config = Mock(return_value=mock_chunk)

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            # Create manager and initialize from strategies
            manager = ProcessorManager(mock_logger)
            manager._processor_classes["chunk"] = mock_chunk_class
            manager.initialize_from_strategies(chunk_strategy_set)

            # Get the configured processor
            chunk_processor = manager.get_processor("chunk")
            assert chunk_processor is not None

            # Execute processing
            result = chunk_processor.extract_chunks(
                sample_video_path, video_duration=30.0
            )

            # Verify result
            assert result == mock_chunk_result
            assert result["total_chunks"] == 2
            mock_chunk.extract_chunks.assert_called_once()

    def test_strategy_configuration_propagation(self, mock_logger, strategy_set):
        """Test that strategy configurations are properly propagated to processors."""
        frame_strategy = FrameSegmentationStrategy(
            max_frames=100, fps=0.5, threshold=0.85
        )
        strategy_set = ProcessingStrategySet(segmentation=frame_strategy)

        # Get processor requirements
        requirements = strategy_set.get_all_required_processors()
        keyframe_config = requirements["keyframe"]

        # Configuration should be propagated from strategy
        assert keyframe_config["max_frames"] == 100
        assert keyframe_config["fps"] == 0.5
        assert keyframe_config["threshold"] == 0.85

    def test_processor_error_handling_in_pipeline(
        self, mock_logger, strategy_set, temp_dir, sample_video_path
    ):
        """Test error handling when processors fail in the pipeline."""
        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            # Create a processor that fails
            class FailingProcessor:
                PROCESSOR_NAME = "keyframe"

                def __init__(self, logger, **kwargs):
                    self.logger = logger

                @classmethod
                def from_config(cls, config, logger):
                    return cls(logger, **config)

                def extract_keyframes(self, *args, **kwargs):
                    raise Exception("Processing failed")

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = FailingProcessor
            manager.initialize_from_strategies(strategy_set)

            # Get processor and test error propagation
            keyframe_processor = manager.get_processor("keyframe")

            with pytest.raises(Exception, match="Processing failed"):
                keyframe_processor.extract_keyframes(sample_video_path, temp_dir)

    def test_processor_cache_consistency_across_pipeline(
        self, mock_logger, strategy_set
    ):
        """Test that processor caching works consistently across the pipeline."""
        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)

            # Mock processor class
            from tests.ingestion.conftest import MockProcessor

            manager._processor_classes["keyframe"] = MockProcessor

            manager.initialize_from_strategies(strategy_set)

            # Get same processor multiple times
            processor1 = manager.get_processor("keyframe")
            processor2 = manager.get_processor("keyframe")

            # Should return same cached instance
            assert processor1 is processor2

    def test_concurrent_processor_access(self, mock_logger, strategy_set):
        """Test that processor manager handles concurrent access safely."""
        import threading

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            from tests.ingestion.conftest import MockProcessor

            manager._processor_classes["keyframe"] = MockProcessor

            # Results from concurrent access
            results = []

            def get_processor():
                processor = manager.get_processor("keyframe")
                results.append(processor)

            # Create multiple threads accessing processor simultaneously
            threads = [threading.Thread(target=get_processor) for _ in range(5)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # All threads should get the same processor instance
            assert len(results) == 5
            first_processor = results[0]
            for processor in results[1:]:
                assert processor is first_processor


@pytest.mark.integration
class TestStrategySetIntegration:
    """Integration tests for ProcessingStrategySet."""

    def test_strategy_set_processor_aggregation(self):
        """Test that strategy set properly aggregates processor requirements."""
        frame_strategy = FrameSegmentationStrategy(max_frames=20)

        # In a real scenario, you might have additional strategies
        strategy_set = ProcessingStrategySet(segmentation=frame_strategy)

        all_requirements = strategy_set.get_all_required_processors()

        # Should include all processor requirements from all strategies
        assert len(all_requirements) > 0
        assert "keyframe" in all_requirements

    def test_strategy_set_configuration_merging(self):
        """Test configuration merging when multiple strategies require same processor."""
        # This would test the case where multiple strategies need the same processor
        # For now, we test with a single strategy

        frame_strategy = FrameSegmentationStrategy(max_frames=30, threshold=0.9)

        strategy_set = ProcessingStrategySet(segmentation=frame_strategy)
        requirements = strategy_set.get_all_required_processors()

        keyframe_config = requirements["keyframe"]
        assert keyframe_config["max_frames"] == 30
        assert keyframe_config["threshold"] == 0.9
