"""
Tests for audio ingestion pipeline components.

Validates:
1. AudioFileSegmentationStrategy produces audio file list
2. AudioEmbeddingStrategy processor requirements
3. Audio profile loads correctly from config
4. StrategyFactory creates audio strategy set
5. ProcessingStrategySet handles audio_file segmentation dispatch
"""

import json
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.strategies import (
    AudioEmbeddingStrategy,
    AudioFileSegmentationStrategy,
    AudioTranscriptionStrategy,
    NoDescriptionStrategy,
)
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory


@pytest.fixture
def audio_dir(tmp_path):
    """Create a temporary directory with dummy audio files."""
    a_dir = tmp_path / "test_audio"
    a_dir.mkdir()
    for i in range(4):
        ext = [".mp3", ".wav", ".flac", ".ogg"][i]
        (a_dir / f"clip_{i:03d}{ext}").write_bytes(b"\x00" * 64)
    return a_dir


class TestAudioFileSegmentationStrategy:
    def test_get_required_processors(self):
        strategy = AudioFileSegmentationStrategy(max_files=50)
        processors = strategy.get_required_processors()
        assert "audio_file" in processors
        assert processors["audio_file"]["max_files"] == 50

    def test_default_max_files(self):
        strategy = AudioFileSegmentationStrategy()
        processors = strategy.get_required_processors()
        assert processors["audio_file"]["max_files"] == 10000


class TestAudioEmbeddingStrategy:
    def test_get_required_processors(self):
        strategy = AudioEmbeddingStrategy()
        processors = strategy.get_required_processors()
        assert "embedding" in processors
        assert processors["embedding"]["type"] == "audio"
        assert processors["embedding"]["clap_model"] == "laion/clap-htsat-unfused"
        assert processors["embedding"]["colbert_model"] == "lightonai/GTE-ModernColBERT-v1"

    def test_custom_models(self):
        strategy = AudioEmbeddingStrategy(
            clap_model="custom/clap",
            colbert_model="custom/colbert",
        )
        processors = strategy.get_required_processors()
        assert processors["embedding"]["clap_model"] == "custom/clap"
        assert processors["embedding"]["colbert_model"] == "custom/colbert"


class TestAudioProfileConfig:
    def test_audio_profile_exists_in_config(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profiles = config["backend"]["profiles"]
        assert "audio_clap_semantic" in profiles

    def test_audio_profile_type_is_audio(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profile = config["backend"]["profiles"]["audio_clap_semantic"]
        assert profile["type"] == "audio"

    def test_audio_profile_strategies(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profile = config["backend"]["profiles"]["audio_clap_semantic"]
        strategies = profile["strategies"]
        assert strategies["segmentation"]["class"] == "AudioFileSegmentationStrategy"
        assert strategies["transcription"]["class"] == "AudioTranscriptionStrategy"
        assert strategies["description"]["class"] == "NoDescriptionStrategy"
        assert strategies["embedding"]["class"] == "AudioEmbeddingStrategy"

    def test_audio_profile_enables_transcription(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        pipeline_config = config["backend"]["profiles"]["audio_clap_semantic"][
            "pipeline_config"
        ]
        assert pipeline_config["transcribe_audio"] is True
        assert pipeline_config["generate_descriptions"] is False
        assert pipeline_config["generate_embeddings"] is True

    def test_audio_profile_schema_config(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        schema_config = config["backend"]["profiles"]["audio_clap_semantic"][
            "schema_config"
        ]
        assert schema_config["acoustic_embedding_dim"] == 512
        assert schema_config["semantic_embedding_dim"] == 128
        assert schema_config["semantic_binary_dim"] == 16


class TestAudioSchemaFile:
    def test_audio_schema_exists(self):
        schema_path = Path("configs/schemas/audio_content_schema.json")
        assert schema_path.exists()

    def test_audio_schema_fields(self):
        with open("configs/schemas/audio_content_schema.json") as f:
            schema = json.load(f)
        field_names = [field["name"] for field in schema["document"]["fields"]]
        assert "audio_id" in field_names
        assert "audio_title" in field_names
        assert "audio_transcript" in field_names
        assert "audio_path" in field_names
        assert "audio_duration" in field_names
        assert "audio_language" in field_names
        assert "acoustic_embedding" in field_names
        assert "semantic_embedding" in field_names
        assert "semantic_embedding_binary" in field_names

    def test_audio_schema_embedding_dimensions(self):
        with open("configs/schemas/audio_content_schema.json") as f:
            schema = json.load(f)
        fields = {f["name"]: f for f in schema["document"]["fields"]}
        assert fields["acoustic_embedding"]["type"] == "tensor<float>(v[512])"
        assert fields["semantic_embedding"]["type"] == "tensor<bfloat16>(token{}, v[128])"
        assert fields["semantic_embedding_binary"]["type"] == "tensor<int8>(token{}, v[16])"

    def test_audio_schema_acoustic_hnsw_index(self):
        with open("configs/schemas/audio_content_schema.json") as f:
            schema = json.load(f)
        fields = {f["name"]: f for f in schema["document"]["fields"]}
        assert "index" in fields["acoustic_embedding"]["indexing"]
        assert "hnsw" in fields["acoustic_embedding"]["index"]

    def test_audio_schema_semantic_colbert_index(self):
        with open("configs/schemas/audio_content_schema.json") as f:
            schema = json.load(f)
        fields = {f["name"]: f for f in schema["document"]["fields"]}
        assert "index" in fields["semantic_embedding_binary"]["indexing"]
        assert fields["semantic_embedding"]["indexing"] == ["attribute"]

    def test_audio_schema_rank_profiles(self):
        with open("configs/schemas/audio_content_schema.json") as f:
            schema = json.load(f)
        profile_names = [rp["name"] for rp in schema["rank_profiles"]]
        assert "default" in profile_names
        assert "transcript_search" in profile_names
        assert "acoustic_similarity" in profile_names
        assert "semantic_float" in profile_names
        assert "semantic_binary" in profile_names
        assert "phased_semantic" in profile_names
        assert "hybrid_semantic_bm25" in profile_names
        assert "hybrid_acoustic_bm25" in profile_names


class TestStrategyFactoryAudioProfile:
    def test_factory_creates_audio_strategy_set(self):
        profile_config = {
            "strategies": {
                "segmentation": {
                    "class": "AudioFileSegmentationStrategy",
                    "params": {"max_files": 100},
                },
                "transcription": {"class": "AudioTranscriptionStrategy", "params": {}},
                "description": {"class": "NoDescriptionStrategy", "params": {}},
                "embedding": {"class": "AudioEmbeddingStrategy", "params": {}},
            }
        }
        strategy_set = StrategyFactory.create_from_profile_config(profile_config)
        assert isinstance(strategy_set.segmentation, AudioFileSegmentationStrategy)
        assert isinstance(strategy_set.transcription, AudioTranscriptionStrategy)
        assert isinstance(strategy_set.description, NoDescriptionStrategy)
        assert isinstance(strategy_set.embedding, AudioEmbeddingStrategy)


class TestAudioSegmentationDispatch:
    """Test that ProcessingStrategySet correctly dispatches audio_file segmentation."""

    @pytest.mark.asyncio
    async def test_audio_segmentation_produces_file_list(self, audio_dir):
        strategy = AudioFileSegmentationStrategy(max_files=100)

        class MockContext:
            profile_output_dir = audio_dir.parent / "output"
            logger = type("L", (), {
                "info": staticmethod(lambda msg: None),
                "warning": staticmethod(lambda msg: None),
                "error": staticmethod(lambda msg: None),
            })()

        MockContext.profile_output_dir.mkdir(exist_ok=True)

        strategy_set = ProcessingStrategySet(
            segmentation=strategy,
            transcription=AudioTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=AudioEmbeddingStrategy(),
        )

        result = await strategy_set._process_segmentation(
            strategy, audio_dir, None, MockContext()
        )

        assert "audio_files" in result
        audio_files = result["audio_files"]
        assert len(audio_files) == 4

        for af in audio_files:
            assert "audio_id" in af
            assert "path" in af
            assert "filename" in af
            assert Path(af["path"]).exists()

    @pytest.mark.asyncio
    async def test_audio_segmentation_respects_max_files(self, audio_dir):
        strategy = AudioFileSegmentationStrategy(max_files=2)

        class MockContext:
            profile_output_dir = audio_dir.parent / "output2"
            logger = type("L", (), {
                "info": staticmethod(lambda msg: None),
                "warning": staticmethod(lambda msg: None),
                "error": staticmethod(lambda msg: None),
            })()

        MockContext.profile_output_dir.mkdir(exist_ok=True)

        strategy_set = ProcessingStrategySet(segmentation=strategy)

        result = await strategy_set._process_segmentation(
            strategy, audio_dir, None, MockContext()
        )

        assert len(result["audio_files"]) == 2

    @pytest.mark.asyncio
    async def test_audio_segmentation_single_file(self, audio_dir):
        """Single audio file path should work directly."""
        strategy = AudioFileSegmentationStrategy()
        single_file = audio_dir / "clip_000.mp3"

        class MockContext:
            profile_output_dir = audio_dir.parent / "output3"
            logger = type("L", (), {
                "info": staticmethod(lambda msg: None),
                "warning": staticmethod(lambda msg: None),
                "error": staticmethod(lambda msg: None),
            })()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        result = await strategy_set._process_segmentation(
            strategy, single_file, None, MockContext()
        )

        assert len(result["audio_files"]) == 1
        assert result["audio_files"][0]["filename"] == "clip_000.mp3"

    @pytest.mark.asyncio
    async def test_audio_segmentation_empty_dir_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        strategy = AudioFileSegmentationStrategy()

        class MockContext:
            profile_output_dir = tmp_path / "output4"
            logger = type("L", (), {
                "info": staticmethod(lambda msg: None),
                "warning": staticmethod(lambda msg: None),
                "error": staticmethod(lambda msg: None),
            })()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        with pytest.raises(ValueError, match="No audio files found"):
            await strategy_set._process_segmentation(
                strategy, empty_dir, None, MockContext()
            )

    @pytest.mark.asyncio
    async def test_audio_segmentation_non_audio_file_raises(self, tmp_path):
        bad_file = tmp_path / "document.pdf"
        bad_file.write_bytes(b"\x00" * 64)
        strategy = AudioFileSegmentationStrategy()

        class MockContext:
            profile_output_dir = tmp_path / "output5"
            logger = type("L", (), {
                "info": staticmethod(lambda msg: None),
                "warning": staticmethod(lambda msg: None),
                "error": staticmethod(lambda msg: None),
            })()

        MockContext.profile_output_dir.mkdir(exist_ok=True)
        strategy_set = ProcessingStrategySet(segmentation=strategy)

        with pytest.raises(ValueError, match="Expected audio file or directory"):
            await strategy_set._process_segmentation(
                strategy, bad_file, None, MockContext()
            )
