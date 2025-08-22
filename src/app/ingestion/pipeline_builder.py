#!/usr/bin/env python3
"""
Pipeline Builder - Clean initialization using Builder pattern.

Provides a fluent interface for configuring the VideoIngestionPipeline
without complex constructor parameters.
"""

from pathlib import Path
from typing import Any

from src.common.config import get_config

from .pipeline import PipelineConfig, VideoIngestionPipeline

# Logging imported where needed


class VideoIngestionPipelineBuilder:
    """Builder for VideoIngestionPipeline with fluent interface."""

    def __init__(self):
        """Initialize builder with default values."""
        self._config: PipelineConfig | None = None
        self._app_config: dict[str, Any] | None = None
        self._schema_name: str | None = None
        self._debug_mode: bool = False
        self._video_dir: Path | None = None
        self._output_dir: Path | None = None
        self._backend: str | None = None
        self._max_frames: int | None = None
        self._max_concurrent: int = 3

    def with_config(self, config: PipelineConfig) -> "VideoIngestionPipelineBuilder":
        """Set pipeline configuration."""
        self._config = config
        return self

    def with_app_config(
        self, app_config: dict[str, Any]
    ) -> "VideoIngestionPipelineBuilder":
        """Set application configuration."""
        self._app_config = app_config
        return self

    def with_schema(self, schema_name: str) -> "VideoIngestionPipelineBuilder":
        """Set schema/profile name."""
        self._schema_name = schema_name
        return self

    def with_debug(self, debug_mode: bool = True) -> "VideoIngestionPipelineBuilder":
        """Enable debug mode."""
        self._debug_mode = debug_mode
        return self

    def with_video_dir(self, video_dir: Path) -> "VideoIngestionPipelineBuilder":
        """Set video input directory."""
        self._video_dir = Path(video_dir)
        return self

    def with_output_dir(self, output_dir: Path) -> "VideoIngestionPipelineBuilder":
        """Set output directory."""
        self._output_dir = Path(output_dir)
        return self

    def with_backend(self, backend: str) -> "VideoIngestionPipelineBuilder":
        """Set search backend (vespa, byaldi, etc.)."""
        self._backend = backend
        return self

    def with_max_frames(self, max_frames: int) -> "VideoIngestionPipelineBuilder":
        """Set maximum frames per video."""
        self._max_frames = max_frames
        return self

    def with_concurrency(self, max_concurrent: int) -> "VideoIngestionPipelineBuilder":
        """Set maximum concurrent video processing."""
        self._max_concurrent = max_concurrent
        return self

    def build(self) -> VideoIngestionPipeline:
        """Build the configured VideoIngestionPipeline."""
        # Build config if not provided
        if not self._config:
            config_dict = {}

            if self._video_dir:
                config_dict["video_dir"] = self._video_dir
            if self._output_dir:
                config_dict["output_dir"] = self._output_dir
            if self._backend:
                config_dict["search_backend"] = self._backend
            if self._max_frames:
                config_dict["max_frames_per_video"] = self._max_frames

            self._config = (
                PipelineConfig(**config_dict)
                if config_dict
                else PipelineConfig.from_config()
            )

        # Use provided app config or load default
        app_config = self._app_config or get_config()

        # Create pipeline instance
        pipeline = VideoIngestionPipeline(
            config=self._config,
            app_config=app_config,
            schema_name=self._schema_name,
            debug_mode=self._debug_mode,
        )

        return pipeline


class PipelineConfigBuilder:
    """Builder for PipelineConfig with fluent interface."""

    def __init__(self):
        """Initialize builder with default values."""
        self._extract_keyframes = True
        self._transcribe_audio = True
        self._generate_descriptions = True
        self._generate_embeddings = True
        self._keyframe_threshold = 0.999
        self._max_frames_per_video = 3000
        self._vlm_batch_size = 500
        self._video_dir = Path("data/videos")
        self._output_dir = Path("outputs/processing")
        self._search_backend = "vespa"

    def extract_keyframes(self, enabled: bool = True) -> "PipelineConfigBuilder":
        """Enable/disable keyframe extraction."""
        self._extract_keyframes = enabled
        return self

    def transcribe_audio(self, enabled: bool = True) -> "PipelineConfigBuilder":
        """Enable/disable audio transcription."""
        self._transcribe_audio = enabled
        return self

    def generate_descriptions(self, enabled: bool = True) -> "PipelineConfigBuilder":
        """Enable/disable description generation."""
        self._generate_descriptions = enabled
        return self

    def generate_embeddings(self, enabled: bool = True) -> "PipelineConfigBuilder":
        """Enable/disable embedding generation."""
        self._generate_embeddings = enabled
        return self

    def keyframe_threshold(self, threshold: float) -> "PipelineConfigBuilder":
        """Set keyframe similarity threshold."""
        self._keyframe_threshold = threshold
        return self

    def max_frames_per_video(self, max_frames: int) -> "PipelineConfigBuilder":
        """Set maximum frames per video."""
        self._max_frames_per_video = max_frames
        return self

    def vlm_batch_size(self, batch_size: int) -> "PipelineConfigBuilder":
        """Set VLM processing batch size."""
        self._vlm_batch_size = batch_size
        return self

    def video_dir(self, directory: Path) -> "PipelineConfigBuilder":
        """Set video input directory."""
        self._video_dir = Path(directory)
        return self

    def output_dir(self, directory: Path) -> "PipelineConfigBuilder":
        """Set output directory."""
        self._output_dir = Path(directory)
        return self

    def backend(self, backend: str) -> "PipelineConfigBuilder":
        """Set search backend."""
        self._search_backend = backend
        return self

    def build(self) -> PipelineConfig:
        """Build the configured PipelineConfig."""
        return PipelineConfig(
            extract_keyframes=self._extract_keyframes,
            transcribe_audio=self._transcribe_audio,
            generate_descriptions=self._generate_descriptions,
            generate_embeddings=self._generate_embeddings,
            keyframe_threshold=self._keyframe_threshold,
            max_frames_per_video=self._max_frames_per_video,
            vlm_batch_size=self._vlm_batch_size,
            video_dir=self._video_dir,
            output_dir=self._output_dir,
            search_backend=self._search_backend,
        )


# Convenience functions for common patterns
def create_pipeline() -> VideoIngestionPipelineBuilder:
    """Create a new pipeline builder."""
    return VideoIngestionPipelineBuilder()


def create_config() -> PipelineConfigBuilder:
    """Create a new config builder."""
    return PipelineConfigBuilder()


def build_simple_pipeline(
    video_dir: Path, schema: str, backend: str = "vespa", debug: bool = False
) -> VideoIngestionPipeline:
    """Build a simple pipeline with common settings."""
    return (
        create_pipeline()
        .with_video_dir(video_dir)
        .with_schema(schema)
        .with_backend(backend)
        .with_debug(debug)
        .build()
    )


def build_test_pipeline(
    video_dir: Path, schema: str, max_frames: int = 10
) -> VideoIngestionPipeline:
    """Build a pipeline for testing with limited frames."""
    config = (
        create_config()
        .video_dir(video_dir)
        .max_frames_per_video(max_frames)
        .backend("vespa")
        .build()
    )

    return (
        create_pipeline()
        .with_config(config)
        .with_schema(schema)
        .with_debug(True)
        .build()
    )
