"""
Unit tests for AudioProcessor.

Tests audio transcription functionality using Whisper models.
"""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "config.json"
VLLM_ASR_URL = "http://cogniverse-vllm-asr:8000"


def _build_audio_processor(profile_config: dict, service_urls: dict) -> AudioProcessor:
    """Run the real factory→manager wiring and return the audio processor.

    Mirrors the production path: ``StrategyFactory`` lifts
    ``inference_services.transcription`` onto the strategy, then
    ``ProcessorManager`` resolves the service name to an ``endpoint`` URL
    before constructing the ``AudioProcessor``. The processor's
    ``endpoint`` attribute is the remote-mode flag the transcribe path reads.
    """
    strategy_set = StrategyFactory.create_from_profile_config(profile_config)
    manager = ProcessorManager(logging.getLogger("test_audio_routing"))
    manager.initialize_from_strategies(strategy_set, service_urls)
    return manager.get_processor("audio")


class TestRemoteTranscriptionRouting:
    """Round-trip wiring: profile config → REMOTE vs in-process AudioProcessor."""

    def test_vllm_asr_profile_builds_remote_processor(self):
        """``inference_services.transcription: vllm_asr`` + a resolved URL
        constructs an AudioProcessor in REMOTE mode: its ``endpoint`` is the
        resolved vLLM ASR URL, so the transcribe path POSTs to the pod
        instead of loading Whisper locally."""
        profile_config = {
            "inference_services": {"transcription": "vllm_asr"},
            "strategies": {
                "transcription": {
                    "class": "AudioTranscriptionStrategy",
                    "params": {"model": "openai/whisper-large-v3-turbo"},
                }
            },
        }

        processor = _build_audio_processor(profile_config, {"vllm_asr": VLLM_ASR_URL})

        assert isinstance(processor, AudioProcessor)
        assert processor.endpoint == VLLM_ASR_URL, (
            "vllm_asr routing must set endpoint = resolved URL (REMOTE mode); "
            f"got {processor.endpoint!r}"
        )

    def test_no_transcription_service_builds_local_processor(self):
        """Control: without ``inference_services.transcription`` the processor
        has no ``endpoint`` and runs Whisper in-process (LOCAL mode)."""
        profile_config = {
            "inference_services": {"embedding": "vllm_colpali"},
            "strategies": {
                "transcription": {
                    "class": "AudioTranscriptionStrategy",
                    "params": {"model": "base"},
                }
            },
        }

        processor = _build_audio_processor(profile_config, {})

        assert isinstance(processor, AudioProcessor)
        assert processor.endpoint is None, (
            "no transcription service must leave endpoint unset (LOCAL mode); "
            f"got {processor.endpoint!r}"
        )


class TestConfigTranscriptionRouting:
    """The shipped configs/config.json routes audio transcription to vllm_asr."""

    @pytest.fixture(scope="class")
    def transcription_profiles(self) -> dict[str, dict]:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        profiles = cfg["backend"]["profiles"]
        return {
            name: p
            for name, p in profiles.items()
            if p.get("strategies", {}).get("transcription", {}).get("class")
            == "AudioTranscriptionStrategy"
        }

    def test_every_transcription_profile_routes_to_vllm_asr(
        self, transcription_profiles
    ):
        """Every profile that runs AudioTranscriptionStrategy sets
        ``inference_services.transcription: vllm_asr`` and the
        ``openai/whisper-large-v3-turbo`` model, and builds a REMOTE
        processor when vllm_asr resolves."""
        assert transcription_profiles, (
            "expected at least one AudioTranscriptionStrategy profile in config"
        )
        for name, profile in transcription_profiles.items():
            inf = profile.get("inference_services") or {}
            assert inf.get("transcription") == "vllm_asr", (
                f"profile {name!r} must route transcription to vllm_asr; "
                f"got {inf.get('transcription')!r}"
            )
            params = profile["strategies"]["transcription"].get("params", {})
            assert params.get("model") == "openai/whisper-large-v3-turbo", (
                f"profile {name!r} transcription model must be "
                f"openai/whisper-large-v3-turbo; got {params.get('model')!r}"
            )
            processor = _build_audio_processor(profile, {"vllm_asr": VLLM_ASR_URL})
            assert processor.endpoint == VLLM_ASR_URL, (
                f"profile {name!r} must build a REMOTE audio processor"
            )


@pytest.mark.requires_whisper
class TestAudioProcessor:
    """Test cases for AudioProcessor."""

    @pytest.fixture
    def processor(self, mock_logger):
        """Create an audio processor for testing."""
        return AudioProcessor(mock_logger, model="whisper-base", language="en")

    def test_processor_initialization(self, mock_logger):
        """Test audio processor initialization."""
        processor = AudioProcessor(
            mock_logger, model="whisper-large-v3", language="auto"
        )

        assert processor.PROCESSOR_NAME == "audio"
        assert processor.logger == mock_logger
        assert processor.model == "whisper-large-v3"
        assert processor.language == "auto"
        assert processor._whisper is None  # Lazy loading

    def test_processor_initialization_defaults(self, mock_logger):
        """Test processor with default values — defaults to `base` so the
        ingestion pod's memory footprint stays bounded; profiles that need
        higher accuracy must pass model=large-v3 explicitly."""
        processor = AudioProcessor(mock_logger)

        assert processor.model == "base"
        assert processor.language == "auto"

    def test_from_config_factory_method(self, mock_logger):
        """Test creating processor from configuration."""
        config = {"model": "whisper-base", "language": "es"}

        processor = AudioProcessor.from_config(config, mock_logger)

        assert processor.model == "whisper-base"
        assert processor.language == "es"

    def test_from_config_with_defaults(self, mock_logger):
        """Test from_config uses defaults for missing parameters."""
        config = {"model": "whisper-tiny"}

        processor = AudioProcessor.from_config(config, mock_logger)

        assert processor.model == "whisper-tiny"
        assert processor.language == "auto"  # default

    @patch("whisper.load_model")
    def test_lazy_whisper_loading(self, mock_load_model, processor):
        """Test that Whisper model is loaded lazily."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Initially no model loaded
        assert processor._whisper is None

        # Load model
        processor._load_whisper()

        # Model should be loaded and cached
        assert processor._whisper == mock_model
        mock_load_model.assert_called_once_with("base")

        # Second call should use cached model
        processor._load_whisper()
        mock_load_model.assert_called_once()  # Still only called once

    @patch("whisper.load_model")
    def test_whisper_loading_error_handling(
        self, mock_load_model, processor, mock_logger
    ):
        """Test handling of Whisper model loading errors."""
        mock_load_model.side_effect = Exception("Failed to load model")

        with pytest.raises(Exception, match="Failed to load model"):
            processor._load_whisper()

        # Should log error
        mock_logger.error.assert_called()

    @patch("whisper.load_model")
    def test_transcribe_audio_success(
        self, mock_load_model, processor, temp_dir, sample_video_path
    ):
        """Test successful audio transcription."""
        # Mock Whisper model
        mock_model = Mock()
        mock_transcription = {
            "text": "This is a test transcription.",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "This is a test"},
                {"start": 2.5, "end": 5.0, "text": "transcription."},
            ],
            "language": "en",
        }
        mock_model.transcribe.return_value = mock_transcription
        mock_load_model.return_value = mock_model

        with patch(
            "cogniverse_core.common.utils.output_manager.get_output_manager"
        ) as mock_get_output_manager:
            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager

            result = processor.transcribe_audio(sample_video_path)

        # Verify transcription result
        assert result["video_id"] == "test_video"
        assert result["full_text"] == "This is a test transcription."
        assert result["language"] == "en"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.5
        assert result["segments"][0]["text"] == "This is a test"

        # Should have called Whisper model
        mock_model.transcribe.assert_called_once()
        transcribe_args = mock_model.transcribe.call_args
        assert str(sample_video_path) in str(transcribe_args)

    @patch("whisper.load_model")
    def test_transcribe_audio_with_language_detection(
        self, mock_load_model, mock_logger, temp_dir, sample_video_path
    ):
        """Test transcription with automatic language detection."""
        processor = AudioProcessor(mock_logger, language="auto")

        mock_model = Mock()
        mock_transcription = {
            "text": "Hola mundo",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hola mundo"}],
            "language": "es",  # Detected Spanish
        }
        mock_model.transcribe.return_value = mock_transcription
        mock_load_model.return_value = mock_model

        with patch(
            "cogniverse_core.common.utils.output_manager.get_output_manager"
        ) as mock_get_output_manager:
            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager

            result = processor.transcribe_audio(sample_video_path)

        assert result["language"] == "es"
        assert result["full_text"] == "Hola mundo"

        # Should have called transcribe without language parameter for auto-detection
        mock_model.transcribe.assert_called_once()

    @patch("whisper.load_model")
    def test_transcribe_audio_with_specific_language(
        self, mock_load_model, mock_logger, temp_dir, sample_video_path
    ):
        """Test transcription with specific language setting."""
        processor = AudioProcessor(mock_logger, language="fr")

        mock_model = Mock()
        mock_transcription = {
            "text": "Bonjour le monde",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Bonjour le monde"}],
            "language": "fr",
        }
        mock_model.transcribe.return_value = mock_transcription
        mock_load_model.return_value = mock_model

        with patch(
            "cogniverse_core.common.utils.output_manager.get_output_manager"
        ) as mock_get_output_manager:
            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager

            processor.transcribe_audio(sample_video_path)

        # Should pass language parameter to Whisper
        mock_model.transcribe.assert_called_once()
        transcribe_kwargs = mock_model.transcribe.call_args.kwargs
        assert transcribe_kwargs.get("language") == "fr"

    @patch("whisper.load_model")
    def test_transcribe_audio_file_not_found(
        self, mock_load_model, processor, temp_dir
    ):
        """Test handling of missing audio file."""
        nonexistent_file = temp_dir / "nonexistent.mp4"

        mock_model = Mock()
        mock_model.transcribe.side_effect = FileNotFoundError("File not found")
        mock_load_model.return_value = mock_model

        with patch(
            "cogniverse_core.common.utils.output_manager.get_output_manager"
        ) as mock_output_manager:
            mock_output_manager_instance = Mock()
            mock_output_manager_instance.get_processing_dir.return_value = temp_dir
            mock_output_manager.return_value = mock_output_manager_instance

            result = processor.transcribe_audio(nonexistent_file)
            # Should return error result instead of raising
            assert "error" in result
            assert "File not found" in result["error"]

    @patch("whisper.load_model")
    def test_transcribe_audio_whisper_error(
        self, mock_load_model, processor, temp_dir, sample_video_path
    ):
        """Test handling of Whisper transcription errors."""
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Whisper error")
        mock_load_model.return_value = mock_model

        with patch(
            "cogniverse_core.common.utils.output_manager.get_output_manager"
        ) as mock_output_manager:
            mock_output_manager_instance = Mock()
            mock_output_manager_instance.get_processing_dir.return_value = temp_dir
            mock_output_manager.return_value = mock_output_manager_instance

            result = processor.transcribe_audio(sample_video_path)
            # Should return error result instead of raising
            assert "error" in result
            assert "Whisper error" in result["error"]

    def test_transcribe_audio_output_directory_structure(
        self, processor, temp_dir, sample_video_path
    ):
        """Test that output directories are structured correctly."""
        with (
            patch("whisper.load_model") as mock_load_model,
            patch(
                "cogniverse_core.common.utils.output_manager.get_output_manager"
            ) as mock_get_output_manager,
        ):
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                "text": "Test",
                "segments": [],
                "language": "en",
            }
            mock_load_model.return_value = mock_model

            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager

            processor.transcribe_audio(sample_video_path)

        # Should use output manager to get correct directory structure
        mock_get_output_manager.assert_called_once()
        mock_output_manager.get_processing_dir.assert_called_with("transcripts")

    @patch("whisper.load_model")
    def test_transcribe_audio_metadata_persistence(
        self, mock_load_model, processor, temp_dir, sample_video_path
    ):
        """Test that transcription metadata is saved correctly."""
        mock_model = Mock()
        mock_transcription = {
            "text": "Persistent test",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Persistent test"}],
            "language": "en",
        }
        mock_model.transcribe.return_value = mock_transcription
        mock_load_model.return_value = mock_model

        with (
            patch(
                "cogniverse_core.common.utils.output_manager.get_output_manager"
            ) as mock_get_output_manager,
            patch("builtins.open", create=True) as mock_open,
            patch("json.dump") as mock_json_dump,
        ):
            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager

            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            processor.transcribe_audio(sample_video_path)

        # Should have saved transcription to JSON file
        mock_json_dump.assert_called()
        saved_data = mock_json_dump.call_args[0][0]
        assert saved_data["full_text"] == "Persistent test"
        assert saved_data["language"] == "en"
        assert len(saved_data["segments"]) == 1

    def test_get_config(self, processor):
        """Test retrieving processor configuration."""
        config = processor.get_config()

        # BaseProcessor only stores kwargs passed to super().__init__
        # Since AudioProcessor doesn't pass its params as kwargs, config will be empty
        assert config == {}
        # But we can verify the actual attributes exist
        assert hasattr(processor, "model")
        assert hasattr(processor, "language")
        assert processor.model == "whisper-base"
        assert processor.language == "en"


@pytest.mark.requires_whisper
class TestAudioProcessorCaching:
    """Test caching behavior of AudioProcessor."""

    @pytest.fixture
    def processor(self, mock_logger):
        return AudioProcessor(mock_logger)

    @patch("whisper.load_model")
    def test_whisper_model_caching(self, mock_load_model, processor):
        """Test that Whisper model is cached after first load."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # First load
        processor._load_whisper()
        assert processor._whisper == mock_model

        # Second load should use cached version
        cached_model = processor._whisper
        processor._load_whisper()
        assert processor._whisper is cached_model

        # load_model should only be called once
        mock_load_model.assert_called_once()

    @patch("whisper.load_model")
    def test_multiple_processor_instances_separate_caches(
        self, mock_load_model, mock_logger
    ):
        """Test that different processor instances have separate model caches."""
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_load_model.side_effect = [mock_model1, mock_model2]

        processor1 = AudioProcessor(mock_logger, model="whisper-base")
        processor2 = AudioProcessor(mock_logger, model="whisper-large-v3")

        processor1._load_whisper()
        processor2._load_whisper()

        # Should have separate cached models
        assert processor1._whisper == mock_model1
        assert processor2._whisper == mock_model2
        assert processor1._whisper is not processor2._whisper

        # Should have called load_model twice with different models
        assert mock_load_model.call_count == 2
        mock_load_model.assert_any_call("base")
        mock_load_model.assert_any_call("large-v3")
