"""
Integration tests for video ingestion with different backends and models.

Tests actual document ingestion with various video processing profiles.
"""

import time
from pathlib import Path

import pytest

from cogniverse_foundation.config.utils import create_default_config_manager
from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.markers import (
    skip_heavy_models_in_ci,
    skip_if_ci,
    skip_if_low_memory,
)


@pytest.mark.integration
@pytest.mark.requires_cv2
class TestRealProcessorExtraction:
    """Test real keyframe and chunk extraction with a real video file."""

    @pytest.fixture
    def real_video_path(self, tmp_path):
        """Create a real 5-second video with varying frames using OpenCV."""
        import cv2
        import numpy as np

        video_path = tmp_path / "real_test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        width, height = 640, 480
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for i in range(fps * 5):
            hue = int((i / (fps * 5)) * 180)
            frame = np.ones((height, width, 3), dtype=np.uint8)
            frame[:, :] = [hue, 255, 255]
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            out.write(frame)

        out.release()
        return video_path

    def test_keyframe_extraction_with_real_video(self, real_video_path, tmp_path):
        """Test real keyframe extraction from a real video file."""
        import logging

        from cogniverse_runtime.ingestion.processors.keyframe_processor import (
            KeyframeProcessor,
        )

        logger = logging.getLogger("test_keyframe")
        processor = KeyframeProcessor(logger, max_frames=5, threshold=0.8)

        output_dir = tmp_path / "keyframes"
        output_dir.mkdir()

        result = processor.extract_keyframes(real_video_path, output_dir=output_dir)

        assert result["video_id"] == "real_test_video"
        assert "keyframes" in result
        assert len(result["keyframes"]) > 0
        assert "metadata" in result

        for kf in result["keyframes"]:
            assert "timestamp" in kf
            assert "filename" in kf
            assert kf["timestamp"] >= 0.0

        keyframes_dir = Path(result["keyframes_dir"])
        keyframe_files = list(keyframes_dir.glob("*.jpg"))
        assert len(keyframe_files) == len(result["keyframes"]), (
            "Each keyframe should have a corresponding image file on disk"
        )

    @pytest.mark.requires_ffmpeg
    def test_chunk_extraction_with_real_video(self, real_video_path, tmp_path):
        """Test real chunk extraction from a real video file."""
        import logging

        from cogniverse_runtime.ingestion.processors.chunk_processor import (
            ChunkProcessor,
        )

        logger = logging.getLogger("test_chunk")
        processor = ChunkProcessor(logger, chunk_duration=2.0, chunk_overlap=0.0)

        output_dir = tmp_path / "chunks"
        output_dir.mkdir()

        result = processor.extract_chunks(real_video_path, output_dir=output_dir)

        assert result["video_id"] == "real_test_video"
        assert "chunks" in result
        assert len(result["chunks"]) > 0
        assert "metadata" in result

        for chunk in result["chunks"]:
            assert "start_time" in chunk
            assert "end_time" in chunk
            assert chunk["end_time"] > chunk["start_time"]

        chunks_dir = Path(result["chunks_dir"])
        chunk_files = list(chunks_dir.glob("*.mp4"))
        assert len(chunk_files) == len(result["chunks"]), (
            "Each chunk should have a corresponding video file on disk"
        )


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestVespaBackendIngestion:
    """Test ingestion with actual Vespa backend."""

    @pytest.fixture(scope="class")
    def vespa_backend(self):
        """
        Start Vespa Docker container (schemas deployed by pipeline, not fixture).

        Note: This fixture ONLY starts the container. Schema deployment happens
        automatically when VideoIngestionPipeline creates backends via BackendRegistry.
        """
        import os

        manager = VespaTestManager(app_name="test-ingestion", http_port=8082)

        # Start Vespa container (no schema deployment)
        if not manager.setup_application_directory():
            pytest.skip("Failed to setup application directory")

        if not manager.deploy_test_application():
            pytest.skip("Failed to deploy Vespa test application")

        # Set BACKEND_URL for the tests to use
        old_backend_url = os.environ.get("BACKEND_URL")
        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(manager.http_port)

        yield manager

        # Cleanup environment
        if old_backend_url is not None:
            os.environ["BACKEND_URL"] = old_backend_url
        elif "BACKEND_URL" in os.environ:
            del os.environ["BACKEND_URL"]
        if "BACKEND_PORT" in os.environ:
            del os.environ["BACKEND_PORT"]

        # Cleanup container
        manager.cleanup()

    @pytest.fixture
    def vespa_test_videos(self):
        """Get test videos for Vespa integration."""
        test_dir = Path("data/testset/evaluation/sample_videos")
        if test_dir.exists():
            return list(test_dir.glob("*.mp4"))[:2]  # Limit to 2 videos
        else:
            pytest.fail(
                "Test videos not available at data/testset/evaluation/sample_videos"
            )

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_lightweight_vespa_ingestion(
        self, vespa_backend, vespa_test_videos, tmp_path
    ):
        """Test lightweight ingestion to Vespa (no heavy models)."""
        # Test with basic frame extraction only
        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        config_manager = create_default_config_manager()
        config = PipelineConfig.from_config(
            tenant_id="default", config_manager=config_manager
        )
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.transcribe_audio = False
        config.generate_descriptions = False
        config.max_frames_per_video = 1

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        pipeline = VideoIngestionPipeline(
            tenant_id="test_tenant",
            config=config,
            config_manager=config_manager,
            schema_loader=schema_loader,
            schema_name="video_colpali_smol500_mv_frame",
        )

        # Process just one video
        result = await pipeline.process_video_async(vespa_test_videos[0])

        assert result is not None
        assert "video_id" in result

    @pytest.mark.local_only
    @pytest.mark.requires_colpali
    @skip_heavy_models_in_ci
    @pytest.mark.asyncio
    async def test_colpali_vespa_ingestion(
        self, vespa_backend, vespa_test_videos, tmp_path
    ):
        """Test ColPali model ingestion to Vespa (local only)."""
        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        config_manager = create_default_config_manager()
        config = PipelineConfig.from_config(
            tenant_id="default", config_manager=config_manager
        )
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.max_frames_per_video = 2

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        pipeline = VideoIngestionPipeline(
            tenant_id="test_tenant",
            config=config,
            config_manager=config_manager,
            schema_loader=schema_loader,
            schema_name="video_colpali_smol500_mv_frame",
        )
        result = await pipeline.process_video_async(vespa_test_videos[0])

        assert result is not None
        assert "embeddings" in result.get("results", {})

    @pytest.mark.local_only
    @pytest.mark.requires_videoprism
    @skip_heavy_models_in_ci
    @skip_if_low_memory
    @pytest.mark.asyncio
    async def test_videoprism_vespa_ingestion(
        self, vespa_backend, vespa_test_videos, tmp_path
    ):
        """Test VideoPrism model ingestion to Vespa (local only)."""
        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        config_manager = create_default_config_manager()
        config = PipelineConfig.from_config(
            tenant_id="default", config_manager=config_manager
        )
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.max_frames_per_video = 1

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        pipeline = VideoIngestionPipeline(
            tenant_id="test_tenant",
            config=config,
            config_manager=config_manager,
            schema_loader=schema_loader,
            schema_name="video_videoprism_base_mv_chunk_30s",
        )
        result = await pipeline.process_video_async(vespa_test_videos[0])

        assert result is not None
        assert "embeddings" in result.get("results", {})

    @pytest.mark.local_only
    @pytest.mark.requires_colqwen
    @skip_heavy_models_in_ci
    @pytest.mark.asyncio
    async def test_colqwen_vespa_ingestion(
        self, vespa_backend, vespa_test_videos, tmp_path
    ):
        """Test ColQwen model ingestion to Vespa (local only)."""
        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        config_manager = create_default_config_manager()
        config = PipelineConfig.from_config(
            tenant_id="default", config_manager=config_manager
        )
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.max_frames_per_video = 1

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        pipeline = VideoIngestionPipeline(
            tenant_id="test_tenant",
            config=config,
            config_manager=config_manager,
            schema_loader=schema_loader,
            schema_name="video_colqwen_omni_mv_chunk_30s",
        )
        result = await pipeline.process_video_async(vespa_test_videos[0])

        assert result is not None
        assert "embeddings" in result.get("results", {})


@pytest.mark.integration
@pytest.mark.local_only
@skip_if_ci
class TestComprehensiveIngestion:
    """Comprehensive ingestion tests (local development only)."""

    @pytest.fixture
    def all_test_videos(self):
        """Get all available test videos."""
        test_dir = Path("data/testset/evaluation/sample_videos")
        if test_dir.exists():
            return list(test_dir.glob("*.mp4"))
        else:
            pytest.fail(
                "Test videos not available at data/testset/evaluation/sample_videos"
            )

    @pytest.mark.slow
    @pytest.mark.requires_vespa
    @pytest.mark.asyncio
    async def test_multi_profile_ingestion(
        self, ingestion_vespa_backend, all_test_videos, tmp_path
    ):
        """Test ingestion with multiple profiles."""
        profiles_to_test = [
            "video_colpali_smol500_mv_frame",
            "video_videoprism_base_mv_chunk_30s",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        results = {}
        for profile in profiles_to_test:
            try:
                config_manager = create_default_config_manager()
                config = PipelineConfig.from_config(
                    tenant_id="default", config_manager=config_manager
                )
                config.video_dir = all_test_videos[0].parent
                config.search_backend = "vespa"
                config.max_frames_per_video = 1

                from cogniverse_core.schemas.filesystem_loader import (
                    FilesystemSchemaLoader,
                )

                schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
                pipeline = VideoIngestionPipeline(
                    tenant_id="test_tenant",
                    config=config,
                    config_manager=config_manager,
                    schema_loader=schema_loader,
                    schema_name=profile,
                )
                result = await pipeline.process_video_async(all_test_videos[0])
                results[profile] = result

            except Exception as e:
                # Log but don't fail - some models might not be available
                print(f"Profile {profile} failed: {e}")

        # At least one profile should succeed
        assert len(results) > 0

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_ingestion_performance(
        self, ingestion_vespa_backend, all_test_videos, tmp_path
    ):
        """Benchmark ingestion performance."""

        from cogniverse_runtime.ingestion.pipeline import (
            PipelineConfig,
            VideoIngestionPipeline,
        )

        config_manager = create_default_config_manager()
        config = PipelineConfig.from_config(
            tenant_id="default", config_manager=config_manager
        )
        config.video_dir = all_test_videos[0].parent
        config.search_backend = "vespa"
        config.max_frames_per_video = 5

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        pipeline = VideoIngestionPipeline(
            tenant_id="test_tenant",
            config=config,
            config_manager=config_manager,
            schema_loader=schema_loader,
            schema_name="video_colpali_smol500_mv_frame",
        )

        start_time = time.time()
        result = await pipeline.process_video_async(all_test_videos[0])
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.2f} seconds")

        assert result is not None
        assert processing_time < 7200  # ColPali + Whisper on CPU takes ~1.5h per video


@pytest.mark.integration
class TestProfileConfigPropagation:
    """Test that profile config keys survive the full config system roundtrip.

    Regression test for model_loader being lost when tenant profiles
    (stored via POST /admin/profiles) replace system profiles (from config.json)
    during ConfigUtils._ensure_backend_config() merge.
    """

    def test_model_loader_survives_config_roundtrip(
        self, ingestion_vespa_backend
    ):
        """model_loader from config.json must reach EmbeddingGeneratorImpl.

        The full path: config.json → ConfigUtils.get("backend") → profiles →
        create_embedding_generator → EmbeddingGeneratorImpl.profile_config.
        If model_loader is missing, embeddings silently fail to generate.
        """
        config_manager = create_default_config_manager()
        from cogniverse_foundation.config.utils import get_config

        config = get_config(
            tenant_id="default", config_manager=config_manager
        )
        backend = config.get("backend")
        profiles = backend.get("profiles", {})

        # Check every profile that has model_loader in config.json
        import json

        config_path = Path("configs/config.json")
        if not config_path.exists():
            pytest.skip("configs/config.json not found")

        raw_config = json.loads(config_path.read_text())
        raw_profiles = raw_config.get("backend", {}).get("profiles", {})

        for profile_name, raw_profile in raw_profiles.items():
            raw_loader = raw_profile.get("model_loader")
            if not raw_loader:
                continue

            merged_profile = profiles.get(profile_name, {})
            merged_loader = merged_profile.get("model_loader")

            assert merged_loader == raw_loader, (
                f"Profile '{profile_name}': model_loader={merged_loader!r} "
                f"after config merge, expected {raw_loader!r}. "
                f"Tenant profile override likely erased the system value."
            )

    def test_tenant_deployed_profile_preserves_system_model_loader(
        self, ingestion_vespa_backend
    ):
        """Deploying a tenant profile must not erase system model_loader.

        Simulates: POST /admin/profiles → stores tenant BackendConfig → merge
        with system config → model_loader must still be present.
        """
        config_manager = create_default_config_manager()
        from cogniverse_foundation.config.unified_config import (
            BackendConfig,
            BackendProfileConfig,
        )

        # Store a tenant profile WITHOUT model_loader (mimics admin API)
        tenant_profile = BackendProfileConfig(
            profile_name="video_colpali_smol500_mv_frame",
            type="video",
            embedding_model="vidore/colsmol-500m",
            embedding_type="multi_vector",
            schema_name="video_colpali_smol500_mv_frame",
            strategies={"float_float": {}},
        )
        assert tenant_profile.model_loader == ""  # Empty by default

        tenant_backend = BackendConfig(
            tenant_id="merge_test_tenant",
            profiles={"video_colpali_smol500_mv_frame": tenant_profile},
        )
        config_manager.set_backend_config(
            tenant_backend, tenant_id="merge_test_tenant"
        )

        # Now get config through ConfigUtils — should merge with system
        from cogniverse_foundation.config.utils import get_config

        config = get_config(
            tenant_id="merge_test_tenant", config_manager=config_manager
        )
        backend = config.get("backend")
        merged = backend.get("profiles", {}).get(
            "video_colpali_smol500_mv_frame", {}
        )

        assert merged.get("model_loader") == "colpali", (
            f"model_loader lost after tenant merge: got {merged.get('model_loader')!r}. "
            f"ConfigUtils._ensure_backend_config must merge, not replace profiles."
        )
