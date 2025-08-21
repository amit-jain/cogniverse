"""
Unit tests for AudioProcessor - Testing actual implementation.

Tests real audio transcription functionality using Whisper.
"""

import pytest
from unittest.mock import Mock, patch

from src.app.ingestion.processors.audio_processor import AudioProcessor


@pytest.mark.unit
@pytest.mark.ci_safe  
@pytest.mark.requires_whisper
class TestAudioProcessor:
    """Test the actual AudioProcessor implementation."""
    
    @pytest.fixture
    def processor(self, mock_logger):
        """Create a real audio processor."""
        return AudioProcessor(mock_logger, model="whisper-base", language="en")
    
    @pytest.fixture
    def auto_processor(self, mock_logger):
        """Create processor with auto language detection."""
        return AudioProcessor(mock_logger, model="whisper-tiny", language="auto")
    
    def test_processor_initialization(self, mock_logger):
        """Test audio processor initialization."""
        processor = AudioProcessor(
            mock_logger,
            model="whisper-large-v3",
            language="es"
        )
        
        assert processor.PROCESSOR_NAME == "audio"
        assert processor.logger == mock_logger
        assert processor.model == "whisper-large-v3"
        assert processor.language == "es"
        assert processor._whisper is None  # Lazy loading
    
    def test_processor_defaults(self, mock_logger):
        """Test processor with default values."""
        processor = AudioProcessor(mock_logger)
        
        assert processor.model == "whisper-large-v3"
        assert processor.language == "auto"
        assert processor._whisper is None
    
    def test_from_config_factory_method(self, mock_logger):
        """Test creating processor from configuration."""
        config = {
            "model": "whisper-medium",
            "language": "fr"
        }
        
        processor = AudioProcessor.from_config(config, mock_logger)
        
        assert processor.model == "whisper-medium"
        assert processor.language == "fr"
    
    def test_from_config_with_defaults(self, mock_logger):
        """Test from_config uses defaults for missing parameters."""
        config = {"model": "whisper-small"}
        
        processor = AudioProcessor.from_config(config, mock_logger)
        
        assert processor.model == "whisper-small"
        assert processor.language == "auto"  # default
    
    @patch('src.app.ingestion.processors.audio_processor.whisper')
    def test_lazy_whisper_loading(self, mock_whisper, processor):
        """Test that Whisper model is loaded lazily."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        # Initially no model loaded
        assert processor._whisper is None
        
        # Load model
        processor._load_whisper()
        
        # Model should be loaded and cached
        assert processor._whisper == mock_model
        mock_whisper.load_model.assert_called_once_with("base")  # whisper-base maps to "base"
        
        # Second call should use cached model
        processor._load_whisper()
        mock_whisper.load_model.assert_called_once()  # Still only called once
    
    @patch('src.app.ingestion.processors.audio_processor.whisper')
    def test_model_name_mapping(self, mock_whisper, mock_logger):
        """Test that model names are correctly mapped."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        test_cases = [
            ("whisper-large-v3", "large-v3"),
            ("whisper-large-v2", "large-v2"),
            ("whisper-medium", "medium"),
            ("whisper-small", "small"),
            ("whisper-base", "base"),
            ("whisper-tiny", "tiny"),
            ("unknown-model", "large-v3"),  # fallback
        ]
        
        for our_model, expected_whisper_model in test_cases:
            processor = AudioProcessor(mock_logger, model=our_model)
            processor._load_whisper()
            
            # Check that the correct model was requested
            calls = mock_whisper.load_model.call_args_list
            last_call = calls[-1]
            assert last_call[0][0] == expected_whisper_model
    
    @patch('src.app.ingestion.processors.audio_processor.whisper')
    def test_whisper_loading_error_handling(self, mock_whisper, processor):
        """Test handling of Whisper model loading errors."""
        mock_whisper.load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            processor._load_whisper()
        
        # Should log error
        processor.logger.error.assert_called()
    
    @patch('src.app.ingestion.processors.audio_processor.whisper')
    @patch('src.common.utils.output_manager.get_output_manager')
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_transcribe_audio_success(self, mock_json_dump, mock_open, mock_output_manager,
                                     mock_whisper, processor, temp_dir, sample_video_path):
        """Test successful audio transcription."""
        # Mock Whisper model
        mock_model = Mock()
        mock_transcription = {
            "text": "This is a test transcription.",
            "language": "en",
            "duration": 5.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": " This is a test"
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": " transcription."
                }
            ]
        }
        mock_model.transcribe.return_value = mock_transcription
        mock_load_model.return_value = mock_model
        
        # Mock output manager
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = processor.transcribe_audio(sample_video_path)
        
        # Verify transcription result structure
        assert result["video_id"] == "test_video"
        assert result["full_text"] == "This is a test transcription."
        assert result["language"] == "en"
        assert result["duration"] == 5.0
        assert result["model"] == "whisper-base"
        assert len(result["segments"]) == 2
        
        # Verify segments are properly formatted
        segment1 = result["segments"][0]
        assert segment1["start"] == 0.0
        assert segment1["end"] == 2.5
        assert segment1["text"] == "This is a test"  # stripped
        
        segment2 = result["segments"][1]
        assert segment2["start"] == 2.5
        assert segment2["end"] == 5.0
        assert segment2["text"] == "transcription."  # stripped
        
        # Should have called Whisper model with correct options
        mock_model.transcribe.assert_called_once()
        transcribe_args = mock_model.transcribe.call_args
        assert str(sample_video_path) in str(transcribe_args[0])
        
        # Should have called with language="en" (not auto)
        transcribe_kwargs = transcribe_args[1]
        assert transcribe_kwargs["language"] == "en"
        assert transcribe_kwargs["task"] == "transcribe"
        
        # Should have saved transcript to file
        mock_json_dump.assert_called_once()
        saved_data = mock_json_dump.call_args[0][0]
        assert saved_data["video_id"] == "test_video"
        assert saved_data["full_text"] == "This is a test transcription."
    
    @patch('whisper.load_model')
    @patch('src.common.utils.output_manager.get_output_manager')
    def test_transcribe_audio_with_auto_language(self, mock_output_manager, mock_load_model,
                                               auto_processor, temp_dir, sample_video_path):
        """Test transcription with automatic language detection."""
        # Mock Whisper model
        mock_model = Mock()
        mock_transcription = {
            "text": "Hola mundo",
            "language": "es",  # Detected Spanish
            "duration": 2.0,
            "segments": [{"start": 0.0, "end": 2.0, "text": " Hola mundo"}]
        }
        mock_model.transcribe.return_value = mock_transcription
        mock_load_model.return_value = mock_model
        
        # Mock output manager
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager
        
        with patch('builtins.open', create=True), patch('json.dump'):
            result = auto_processor.transcribe_audio(sample_video_path)
        
        assert result["language"] == "es"
        assert result["full_text"] == "Hola mundo"
        
        # Should have called transcribe with language=None for auto-detection
        mock_model.transcribe.assert_called_once()
        transcribe_kwargs = mock_model.transcribe.call_args[1]
        assert transcribe_kwargs["language"] is None  # auto detection
    
    @patch('whisper.load_model')
    def test_transcribe_audio_with_cache_hit(self, mock_load_model, processor, sample_video_path):
        """Test transcription with cache hit."""
        # Mock cache with existing transcript
        mock_cache = Mock()
        cached_transcript = {
            "video_id": "test_video",
            "full_text": "Cached transcript",
            "language": "en",
            "segments": []
        }
        mock_cache.get_transcript.return_value = cached_transcript
        
        result = processor.transcribe_audio(sample_video_path, cache=mock_cache)
        
        # Should return cached result without loading Whisper
        assert result == cached_transcript
        mock_load_model.assert_not_called()
        mock_cache.get_transcript.assert_called_once_with(sample_video_path, "test_video")
    
    @patch('whisper.load_model')
    @patch('src.common.utils.output_manager.get_output_manager')
    def test_transcribe_audio_with_cache_miss_and_save(self, mock_output_manager, mock_load_model,
                                                      processor, temp_dir, sample_video_path):
        """Test transcription with cache miss and subsequent save."""
        # Mock cache with no existing transcript
        mock_cache = Mock()
        mock_cache.get_transcript.return_value = None
        
        # Mock Whisper model
        mock_model = Mock()
        mock_transcription = {
            "text": "New transcript",
            "language": "en",
            "duration": 3.0,
            "segments": [{"start": 0.0, "end": 3.0, "text": " New transcript"}]
        }
        mock_model.transcribe.return_value = mock_transcription
        mock_load_model.return_value = mock_model
        
        # Mock output manager
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager
        
        with patch('builtins.open', create=True), patch('json.dump'):
            result = processor.transcribe_audio(sample_video_path, cache=mock_cache)
        
        # Should have checked cache first
        mock_cache.get_transcript.assert_called_once_with(sample_video_path, "test_video")
        
        # Should have saved to cache after transcription
        mock_cache.set_transcript.assert_called_once()
        cache_save_args = mock_cache.set_transcript.call_args[0]
        assert cache_save_args[0] == sample_video_path
        assert cache_save_args[1] == "test_video"
        assert cache_save_args[2]["full_text"] == "New transcript"
    
    @patch('whisper.load_model')
    def test_transcribe_audio_whisper_error(self, mock_load_model, processor, sample_video_path):
        """Test handling of Whisper transcription errors."""
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Whisper transcription failed")
        mock_load_model.return_value = mock_model
        
        with patch('src.common.utils.output_manager.get_output_manager'):
            result = processor.transcribe_audio(sample_video_path)
        
        # Should return error result instead of raising
        assert result["video_id"] == "test_video"
        assert "error" in result
        assert "Whisper transcription failed" in result["error"]
        assert result["full_text"] == ""
        assert result["segments"] == []
        
        # Should log error
        processor.logger.error.assert_called()
    
    def test_cleanup_method(self, processor):
        """Test cleanup method."""
        # Set up a mock whisper model
        processor._whisper = Mock()
        
        processor.cleanup()
        
        # Should clear the whisper model
        assert processor._whisper is None
    
    def test_get_config_method(self, processor):
        """Test the get_config method from BaseProcessor."""
        config = processor.get_config()
        
        # The base processor only stores kwargs passed to super().__init__
        # Since AudioProcessor doesn't pass its params as kwargs, config will be empty
        # But we can verify the actual attributes exist
        assert hasattr(processor, 'model')
        assert hasattr(processor, 'language') 
        assert processor.model == "whisper-base"
        assert processor.language == "en"
    
    def test_processor_name_constant(self, processor):
        """Test processor name constant."""
        assert processor.PROCESSOR_NAME == "audio"
        assert processor.get_processor_name() == "audio"