"""
Tests for image ingestion pipeline components.

Validates:
1. ImageSegmentationStrategy produces keyframe-compatible output
2. Image profile loads correctly from config
3. StrategyFactory creates image strategy set
4. ProcessingStrategySet handles image segmentation dispatch
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.strategies import (
    ImageSegmentationStrategy,
    MultiVectorEmbeddingStrategy,
    NoDescriptionStrategy,
    NoTranscriptionStrategy,
)
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory


@pytest.fixture
def image_dir(tmp_path):
    """Create a temporary directory with test images."""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_dir / f"test_image_{i:03d}.jpg")
    return img_dir


class TestImageSegmentationStrategy:
    def test_get_required_processors(self):
        strategy = ImageSegmentationStrategy(max_images=100)
        processors = strategy.get_required_processors()
        assert "image" in processors
        assert processors["image"]["max_images"] == 100

    def test_default_max_images(self):
        strategy = ImageSegmentationStrategy()
        processors = strategy.get_required_processors()
        assert processors["image"]["max_images"] == 10000


class TestImageProfileConfig:
    def test_image_profile_exists_in_config(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profiles = config["backend"]["profiles"]
        assert "image_colpali_mv" in profiles

    def test_image_profile_type_is_image(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profile = config["backend"]["profiles"]["image_colpali_mv"]
        assert profile["type"] == "image"

    def test_image_profile_strategies(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        profile = config["backend"]["profiles"]["image_colpali_mv"]
        strategies = profile["strategies"]
        assert strategies["segmentation"]["class"] == "ImageSegmentationStrategy"
        assert strategies["transcription"]["class"] == "NoTranscriptionStrategy"
        assert strategies["description"]["class"] == "NoDescriptionStrategy"
        assert strategies["embedding"]["class"] == "MultiVectorEmbeddingStrategy"

    def test_image_profile_disables_audio(self):
        with open("configs/config.json") as f:
            config = json.load(f)
        pipeline_config = config["backend"]["profiles"]["image_colpali_mv"][
            "pipeline_config"
        ]
        assert pipeline_config["transcribe_audio"] is False
        assert pipeline_config["generate_descriptions"] is False
        assert pipeline_config["generate_embeddings"] is True


class TestImageSchemaFile:
    def test_image_schema_exists(self):
        schema_path = Path("configs/schemas/image_colpali_mv_schema.json")
        assert schema_path.exists()

    def test_image_schema_fields(self):
        with open("configs/schemas/image_colpali_mv_schema.json") as f:
            schema = json.load(f)
        field_names = [f["name"] for f in schema["document"]["fields"]]
        assert "image_id" in field_names
        assert "image_title" in field_names
        assert "image_description" in field_names
        assert "embedding" in field_names
        assert "embedding_binary" in field_names

    def test_image_schema_embedding_dimensions(self):
        with open("configs/schemas/image_colpali_mv_schema.json") as f:
            schema = json.load(f)
        fields = {f["name"]: f for f in schema["document"]["fields"]}
        assert "tensor<bfloat16>(patch{}, v[128])" == fields["embedding"]["type"]
        assert "tensor<int8>(patch{}, v[16])" == fields["embedding_binary"]["type"]


class TestStrategyFactoryImageProfile:
    def test_factory_creates_image_strategy_set(self):
        profile_config = {
            "strategies": {
                "segmentation": {
                    "class": "ImageSegmentationStrategy",
                    "params": {"max_images": 50},
                },
                "transcription": {"class": "NoTranscriptionStrategy", "params": {}},
                "description": {"class": "NoDescriptionStrategy", "params": {}},
                "embedding": {"class": "MultiVectorEmbeddingStrategy", "params": {}},
            }
        }
        strategy_set = StrategyFactory.create_from_profile_config(profile_config)
        assert isinstance(strategy_set.segmentation, ImageSegmentationStrategy)
        assert isinstance(strategy_set.transcription, NoTranscriptionStrategy)
        assert isinstance(strategy_set.description, NoDescriptionStrategy)
        assert isinstance(strategy_set.embedding, MultiVectorEmbeddingStrategy)


class TestImageSegmentationDispatch:
    """Test that ProcessingStrategySet correctly dispatches image segmentation."""

    @pytest.mark.asyncio
    async def test_image_segmentation_produces_keyframe_format(self, image_dir):
        """Image segmentation should produce output compatible with keyframe format."""
        strategy = ImageSegmentationStrategy(max_images=10)

        # Create a mock pipeline context
        class MockContext:
            profile_output_dir = image_dir.parent / "output"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                    "warning": staticmethod(lambda msg: None),
                    "error": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)

        strategy_set = ProcessingStrategySet(
            segmentation=strategy,
            transcription=NoTranscriptionStrategy(),
            description=NoDescriptionStrategy(),
            embedding=MultiVectorEmbeddingStrategy(),
        )

        result = await strategy_set._process_segmentation(
            strategy, image_dir, None, MockContext()
        )

        assert "keyframes" in result
        keyframes_data = result["keyframes"]
        assert "keyframes" in keyframes_data
        assert "stats" in keyframes_data
        assert len(keyframes_data["keyframes"]) == 5

        # Verify each keyframe has the expected fields
        for kf in keyframes_data["keyframes"]:
            assert "frame_id" in kf
            assert "path" in kf
            assert "filename" in kf
            assert Path(kf["path"]).exists()

        assert keyframes_data["stats"]["extraction_method"] == "image_load"
        assert keyframes_data["stats"]["total_keyframes"] == 5

    @pytest.mark.asyncio
    async def test_image_segmentation_respects_max_images(self, image_dir):
        """Should limit images to max_images."""
        strategy = ImageSegmentationStrategy(max_images=3)

        class MockContext:
            profile_output_dir = image_dir.parent / "output2"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                    "warning": staticmethod(lambda msg: None),
                    "error": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)

        strategy_set = ProcessingStrategySet(
            segmentation=strategy,
            transcription=NoTranscriptionStrategy(),
        )

        result = await strategy_set._process_segmentation(
            strategy, image_dir, None, MockContext()
        )

        assert len(result["keyframes"]["keyframes"]) == 3

    @pytest.mark.asyncio
    async def test_image_segmentation_empty_dir_raises(self, tmp_path):
        """Should raise ValueError for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        strategy = ImageSegmentationStrategy()

        class MockContext:
            profile_output_dir = tmp_path / "output3"
            logger = type(
                "L",
                (),
                {
                    "info": staticmethod(lambda msg: None),
                    "warning": staticmethod(lambda msg: None),
                    "error": staticmethod(lambda msg: None),
                },
            )()

        MockContext.profile_output_dir.mkdir(exist_ok=True)

        strategy_set = ProcessingStrategySet(segmentation=strategy)

        with pytest.raises(ValueError, match="No image files found"):
            await strategy_set._process_segmentation(
                strategy, empty_dir, None, MockContext()
            )
