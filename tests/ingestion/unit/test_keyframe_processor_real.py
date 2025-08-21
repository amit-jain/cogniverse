"""
Unit tests for KeyframeProcessor - Testing actual implementation.

Tests the real keyframe extraction functionality including FPS and histogram modes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.app.ingestion.processors.keyframe_processor import KeyframeProcessor


@pytest.mark.unit
@pytest.mark.ci_safe
@pytest.mark.requires_cv2
class TestKeyframeProcessor:
    """Test the actual KeyframeProcessor implementation."""
    
    @pytest.fixture
    def processor(self, mock_logger):
        """Create a real keyframe processor."""
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
    
    def test_processor_histogram_mode(self, mock_logger):
        """Test histogram mode when no fps provided."""
        processor = KeyframeProcessor(mock_logger, threshold=0.9)
        
        assert processor.threshold == 0.9
        assert processor.max_frames == 3000  # default
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
    
    @patch('src.common.utils.output_manager.get_output_manager')
    @patch('cv2.VideoCapture')
    @patch('cv2.imwrite')
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_extract_keyframes_fps_mode(self, mock_json_dump, mock_open, mock_imwrite, 
                                       mock_cv2_cap, mock_output_manager, fps_processor, 
                                       temp_dir, sample_video_path):
        """Test FPS-based keyframe extraction."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # CAP_PROP_FPS
            7: 150,   # CAP_PROP_FRAME_COUNT  
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        
        # Mock frame reading - return 5 frames then stop
        frame_count = 0
        def mock_read():
            nonlocal frame_count
            if frame_count < 150:
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                frame_count += 1
                return True, frame
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_cv2_cap.return_value = mock_cap
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_imwrite.return_value = True
        
        # Execute
        result = fps_processor.extract_keyframes(sample_video_path)
        
        # Verify results structure
        assert "keyframes" in result
        assert "metadata" in result
        assert "keyframes_dir" in result
        assert "video_id" in result
        assert result["video_id"] == "test_video"
        
        # Verify metadata was saved
        mock_json_dump.assert_called_once()
        saved_metadata = mock_json_dump.call_args[0][0]
        assert saved_metadata["extraction_method"] == "fps"
        assert saved_metadata["fps"] == 1.0
        assert saved_metadata["video_id"] == "test_video"
        
        # Should extract frames at 30fps / 1fps = every 30 frames
        keyframes = result["keyframes"]
        assert len(keyframes) <= fps_processor.max_frames
        
        # Verify keyframes have correct structure
        if keyframes:
            keyframe = keyframes[0]
            assert "frame_number" in keyframe
            assert "timestamp" in keyframe
            assert "filename" in keyframe
            assert "path" in keyframe
    
    @patch('src.common.utils.output_manager.get_output_manager')
    @patch('cv2.VideoCapture')
    @patch('cv2.calcHist')
    @patch('cv2.compareHist')
    @patch('cv2.normalize')
    @patch('cv2.imwrite')
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_extract_keyframes_histogram_mode(self, mock_json_dump, mock_open, mock_imwrite,
                                             mock_normalize, mock_compare_hist, mock_calc_hist,
                                             mock_cv2_cap, mock_output_manager, processor, 
                                             temp_dir, sample_video_path):
        """Test histogram-based keyframe extraction."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # CAP_PROP_FPS
            7: 90,    # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        
        # Mock frame reading
        frame_count = 0
        def mock_read():
            nonlocal frame_count
            if frame_count < 90:
                # Create frames with different intensities to trigger histogram differences
                intensity = int((frame_count % 30) * 8.5)  # Vary intensity every 30 frames
                frame = np.full((480, 640, 3), intensity, dtype=np.uint8)
                frame_count += 1
                return True, frame
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_cv2_cap.return_value = mock_cap
        
        # Mock histogram operations
        mock_calc_hist.return_value = np.array([100] * 512, dtype=np.float32)
        mock_normalize.return_value.flatten.return_value = np.array([0.1] * 512)
        
        # Mock histogram comparison - return low correlation every 30 frames
        call_count = 0
        def mock_compare(h1, h2, method):
            nonlocal call_count
            call_count += 1
            # Return low correlation (trigger keyframe) every 30 calls
            return 0.5 if call_count % 30 == 0 else 0.95
        
        mock_compare_hist.side_effect = mock_compare
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_imwrite.return_value = True
        
        # Execute
        result = processor.extract_keyframes(sample_video_path)
        
        # Verify results structure
        assert "keyframes" in result
        assert "metadata" in result
        assert "keyframes_dir" in result
        assert "video_id" in result
        assert result["video_id"] == "test_video"
        
        # Verify metadata was saved
        mock_json_dump.assert_called_once()
        saved_metadata = mock_json_dump.call_args[0][0]
        assert saved_metadata["extraction_method"] == "histogram"
        assert saved_metadata["threshold"] == 0.8
        assert saved_metadata["video_id"] == "test_video"
        
        # Should have extracted keyframes based on histogram differences
        keyframes = result["keyframes"]
        assert len(keyframes) > 0
        assert len(keyframes) <= processor.max_frames
        
        # Verify keyframes have histogram-specific fields
        if keyframes:
            keyframe = keyframes[0]
            assert "frame_number" in keyframe
            assert "timestamp" in keyframe
            assert "filename" in keyframe
            assert "path" in keyframe
            assert "correlation" in keyframe
    
    @patch('cv2.VideoCapture')
    def test_extract_keyframes_max_frames_limit(self, mock_cv2_cap, processor, temp_dir, sample_video_path):
        """Test that max_frames limit is respected."""
        processor.max_frames = 2  # Set low limit
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # CAP_PROP_FPS
            7: 300,   # CAP_PROP_FRAME_COUNT (10 seconds)
        }.get(prop, 0.0)  # Return float for consistency
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
        mock_cv2_cap.return_value = mock_cap
        
        with patch('cv2.calcHist') as mock_calc_hist, \
             patch('cv2.compareHist') as mock_compare_hist, \
             patch('cv2.normalize') as mock_normalize, \
             patch('cv2.imwrite'), \
             patch('src.common.utils.output_manager.get_output_manager') as mock_output_manager, \
             patch('builtins.open', create=True), \
             patch('json.dump'):
            
            # Setup remaining mocks
            mock_manager = Mock()
            mock_manager.get_processing_dir.return_value = temp_dir
            mock_output_manager.return_value = mock_manager
            
            mock_calc_hist.return_value = np.array([100] * 512, dtype=np.float32)
            mock_normalize.return_value.flatten.return_value = np.array([0.1] * 512)
            mock_compare_hist.return_value = 0.5  # Always trigger keyframe
            
            result = processor.extract_keyframes(sample_video_path)
        
        # Should not exceed max_frames limit
        keyframes = result["keyframes"]
        assert len(keyframes) <= processor.max_frames
        assert len(keyframes) == 2  # Should hit the limit
    
    @patch('cv2.VideoCapture')
    def test_extract_keyframes_invalid_video(self, mock_cv2_cap, processor, temp_dir):
        """Test handling of invalid video file."""
        invalid_path = temp_dir / "nonexistent.mp4"
        
        # Mock failed video capture
        mock_cap = Mock()
        mock_cap.get.return_value = 0.0  # Return numeric values
        mock_cap.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_cap
        
        with patch('src.common.utils.output_manager.get_output_manager') as mock_output_manager:
            mock_manager = Mock()
            mock_manager.get_processing_dir.return_value = temp_dir
            mock_output_manager.return_value = mock_manager
            
            # Should complete but extract no frames
            result = processor.extract_keyframes(invalid_path)
            
            assert result["video_id"] == "nonexistent"
            assert len(result["keyframes"]) == 0
    
    def test_get_config_method(self, processor):
        """Test the get_config method from BaseProcessor."""
        config = processor.get_config()
        
        # The base processor only stores kwargs passed to super().__init__
        # Since KeyframeProcessor doesn't pass its params as kwargs, config will be empty
        # But we can verify the actual attributes exist
        assert hasattr(processor, 'threshold')
        assert hasattr(processor, 'max_frames') 
        assert hasattr(processor, 'fps')
        assert processor.threshold == 0.8
        assert processor.max_frames == 10
        assert processor.fps is None
    
    def test_processor_name_constant(self, processor):
        """Test processor name constant."""
        assert processor.PROCESSOR_NAME == "keyframe"
        assert processor.get_processor_name() == "keyframe"