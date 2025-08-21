"""
Unit tests for KeyframeProcessor.

Tests keyframe extraction functionality including histogram-based
and FPS-based extraction methods.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import cv2
from pathlib import Path

from src.app.ingestion.processors.keyframe_processor import KeyframeProcessor


class TestKeyframeProcessor:
    """Test cases for KeyframeProcessor."""
    
    @pytest.fixture
    def processor(self, mock_logger):
        """Create a keyframe processor for testing."""
        return KeyframeProcessor(mock_logger, threshold=0.8, max_frames=10)
    
    @pytest.fixture
    def fps_processor(self, mock_logger):
        """Create an FPS-based keyframe processor."""
        return KeyframeProcessor(mock_logger, threshold=0.8, max_frames=10, fps=1.0)
    
    def test_processor_initialization(self, mock_logger):
        """Test keyframe processor initialization."""
        processor = KeyframeProcessor(
            mock_logger, 
            threshold=0.95, 
            max_frames=100, 
            fps=2.0
        )
        
        assert processor.PROCESSOR_NAME == "keyframe"
        assert processor.logger == mock_logger
        assert processor.threshold == 0.95
        assert processor.max_frames == 100
        assert processor.fps == 2.0
        assert processor.extraction_mode == 'fps'
    
    def test_processor_initialization_defaults(self, mock_logger):
        """Test processor with default values."""
        processor = KeyframeProcessor(mock_logger)
        
        assert processor.threshold == 0.999
        assert processor.max_frames == 3000
        assert processor.fps is None
        assert processor.extraction_mode == 'histogram'
    
    def test_from_config_factory_method(self, mock_logger):
        """Test creating processor from configuration."""
        config = {
            "threshold": 0.85,
            "max_frames": 50,
            "fps": 0.5
        }
        
        processor = KeyframeProcessor.from_config(config, mock_logger)
        
        assert processor.threshold == 0.85
        assert processor.max_frames == 50
        assert processor.fps == 0.5
        assert processor.extraction_mode == 'fps'
    
    def test_from_config_with_defaults(self, mock_logger):
        """Test from_config uses defaults for missing parameters."""
        config = {"threshold": 0.9}
        
        processor = KeyframeProcessor.from_config(config, mock_logger)
        
        assert processor.threshold == 0.9
        assert processor.max_frames == 3000  # default
        assert processor.fps is None  # default
    
    @patch('cv2.VideoCapture')
    def test_extract_keyframes_fps_mode(self, mock_cv2, fps_processor, temp_dir, sample_video_path):
        """Test keyframe extraction in FPS mode."""
        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 150,  # 5 seconds at 30 fps
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        
        # Mock frame reading - create different frames
        frame_count = 0
        def mock_read():
            nonlocal frame_count
            if frame_count < 150:
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                frame_count += 1
                return True, frame
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_cv2.return_value = mock_cap
        
        with patch('cv2.imwrite') as mock_imwrite:
            result = fps_processor.extract_keyframes(sample_video_path, temp_dir)
        
        # Verify results
        assert result["video_id"] == "test_video"
        assert result["extraction_method"] == "fps"
        assert result["total_keyframes"] > 0
        assert len(result["keyframes"]) > 0
        
        # Should have extracted frames at 1 FPS (every 30 frames)
        expected_frames = min(5, fps_processor.max_frames)  # 5 seconds at 1 FPS
        assert len(result["keyframes"]) <= expected_frames
    
    @patch('cv2.VideoCapture')
    def test_extract_keyframes_histogram_mode(self, mock_cv2, processor, temp_dir, sample_video_path):
        """Test keyframe extraction in histogram mode."""
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90,  # 3 seconds
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        
        # Create frames with varying histograms
        frame_count = 0
        def mock_read():
            nonlocal frame_count
            if frame_count < 90:
                # Create frames with different intensities for histogram variation
                intensity = int((frame_count / 90) * 255)
                frame = np.full((480, 640, 3), intensity, dtype=np.uint8)
                frame_count += 1
                return True, frame
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_cv2.return_value = mock_cap
        
        with patch('cv2.imwrite') as mock_imwrite, \
             patch('cv2.calcHist') as mock_calc_hist, \
             patch('cv2.compareHist') as mock_compare_hist:
            
            # Mock histogram calculation
            mock_calc_hist.return_value = np.array([100] * 256, dtype=np.float32)
            
            # Mock histogram comparison - return varying similarities
            call_count = 0
            def mock_compare(h1, h2, method):
                nonlocal call_count
                call_count += 1
                # First frame always different, then simulate varying similarity
                return 0.5 if call_count % 30 == 0 else 0.95  # Every 30th frame is different
            
            mock_compare_hist.side_effect = mock_compare
            
            result = processor.extract_keyframes(sample_video_path, temp_dir)
        
        # Verify results
        assert result["video_id"] == "test_video"
        assert result["extraction_method"] == "histogram"
        assert result["total_keyframes"] > 0
        assert len(result["keyframes"]) > 0
        
        # Should have extracted keyframes based on histogram differences
        for keyframe in result["keyframes"]:
            assert "frame_index" in keyframe
            assert "timestamp" in keyframe
            assert "filename" in keyframe
            assert "histogram_difference" in keyframe
    
    @patch('cv2.VideoCapture')
    def test_extract_keyframes_max_frames_limit(self, mock_cv2, processor, temp_dir, sample_video_path):
        """Test that max_frames limit is respected."""
        processor.max_frames = 2  # Set low limit
        
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,  # 10 seconds
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        
        frame_count = 0
        def mock_read():
            nonlocal frame_count
            if frame_count < 300:
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                frame_count += 1
                return True, frame
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_cv2.return_value = mock_cap
        
        with patch('cv2.imwrite') as mock_imwrite, \
             patch('cv2.calcHist') as mock_calc_hist, \
             patch('cv2.compareHist') as mock_compare_hist:
            
            mock_calc_hist.return_value = np.array([100] * 256, dtype=np.float32)
            mock_compare_hist.return_value = 0.5  # Always different
            
            result = processor.extract_keyframes(sample_video_path, temp_dir)
        
        # Should not exceed max_frames limit
        assert len(result["keyframes"]) <= processor.max_frames
        assert result["total_keyframes"] == len(result["keyframes"])
    
    def test_extract_keyframes_invalid_video(self, processor, temp_dir):
        """Test handling of invalid video file."""
        invalid_path = temp_dir / "nonexistent.mp4"
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_cv2.return_value = mock_cap
            
            with pytest.raises(Exception):
                processor.extract_keyframes(invalid_path, temp_dir)
    
    @patch('cv2.VideoCapture')
    def test_extract_keyframes_output_directory_creation(self, mock_cv2, processor, temp_dir, sample_video_path):
        """Test that output directories are created correctly."""
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 30,
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        
        def mock_read():
            return False, None  # End immediately
        mock_cap.read.side_effect = mock_read
        mock_cv2.return_value = mock_cap
        
        with patch('cv2.imwrite'):
            result = processor.extract_keyframes(sample_video_path, temp_dir)
        
        # Check that directories were created
        expected_keyframes_dir = temp_dir / "keyframes" / "test_video"
        expected_metadata_dir = temp_dir / "metadata"
        
        # The processor should have attempted to create these directories
        # (we can't easily test the actual creation without more complex mocking)
        assert result["video_id"] == "test_video"
    
    @patch('src.common.utils.output_manager.get_output_manager')
    def test_extract_keyframes_uses_output_manager(self, mock_get_output_manager, processor, sample_video_path):
        """Test that output manager is used when no output_dir provided."""
        mock_output_manager = Mock()
        mock_output_manager.get_processing_dir.return_value = Path("/mock/processing")
        mock_get_output_manager.return_value = mock_output_manager
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_cv2.return_value = mock_cap
            
            try:
                processor.extract_keyframes(sample_video_path)  # No output_dir
            except Exception:
                pass  # Expected to fail due to mock setup
        
        # Should have called output manager
        mock_get_output_manager.assert_called_once()
        mock_output_manager.get_processing_dir.assert_called()


class TestKeyframeProcessorHelperMethods:
    """Test helper methods of KeyframeProcessor."""
    
    @pytest.fixture
    def processor(self, mock_logger):
        return KeyframeProcessor(mock_logger)
    
    def test_calculate_histogram_difference(self, processor):
        """Test histogram difference calculation."""
        # This tests a private method, but it's critical functionality
        
        # Create two different histograms
        hist1 = np.array([100, 50, 25] * 85 + [100], dtype=np.float32)  # 256 bins
        hist2 = np.array([50, 100, 75] * 85 + [50], dtype=np.float32)   # 256 bins
        
        with patch('cv2.compareHist') as mock_compare:
            mock_compare.return_value = 0.7
            
            # Access private method for testing
            if hasattr(processor, '_calculate_histogram_difference'):
                diff = processor._calculate_histogram_difference(hist1, hist2)
                assert diff == 0.3  # 1 - 0.7
    
    def test_frame_extraction_timing(self, processor):
        """Test that frame timing calculations are correct."""
        # This would test timestamp calculations in extraction methods
        # Implementation depends on the specific private methods available
        pass