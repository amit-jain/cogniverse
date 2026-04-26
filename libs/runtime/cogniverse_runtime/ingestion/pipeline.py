#!/usr/bin/env python3
"""
Unified Video Processing Pipeline with Async Optimizations

Consolidates all video processing steps into a single configurable pipeline:
1. Keyframe extraction
2. Audio transcription
3. VLM description generation
4. Vector embeddings
5. Index creation

Features:
- Single event loop for all async operations
- Concurrent cache checks
- Parallel video processing support
- Configurable concurrency levels

Based on the configurable pipeline design from CLAUDE.md.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from cogniverse_core.common.cache.pipeline_cache import PipelineArtifactCache
from cogniverse_core.common.media import MediaConfig, MediaLocator
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.events import (
    EventQueue,
    TaskState,
    create_complete_event,
    create_error_event,
    create_progress_event,
    create_status_event,
)
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.ingestion.exceptions import (
    PipelineException,
    wrap_content_error,
)
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.processors.embedding_generator import (
    create_embedding_generator,
)
from cogniverse_runtime.ingestion.strategy_factory import StrategyFactory


class PipelineStep(Enum):
    """Pipeline steps that can be enabled/disabled"""

    EXTRACT_KEYFRAMES = "extract_keyframes"
    EXTRACT_CHUNKS = "extract_chunks"
    TRANSCRIBE_AUDIO = "transcribe_audio"
    GENERATE_DESCRIPTIONS = "generate_descriptions"
    GENERATE_EMBEDDINGS = "generate_embeddings"


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline"""

    extract_keyframes: bool = True
    transcribe_audio: bool = True
    generate_descriptions: bool = True
    generate_embeddings: bool = True

    # Processing parameters
    keyframe_threshold: float = 0.999
    max_frames_per_video: int = 3000
    vlm_batch_size: int = 500

    # Paths
    video_dir: Path = Path("data/videos")
    output_dir: Path = None  # Will be set from OutputManager

    # Media root URI for non-filesystem ingestion sources (s3://, pvc://, etc.).
    # When set, the pipeline enumerates videos via MediaLocator.list instead of
    # globbing video_dir. None preserves the legacy local-directory behavior.
    media_root_uri: Optional[str] = None

    # Backend selection
    search_backend: str = "byaldi"  # "byaldi" or "vespa"

    @classmethod
    def from_config(cls, tenant_id: str, config_manager) -> "PipelineConfig":
        """Load pipeline config from main config file"""
        config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        pipeline_config = config.get("pipeline_config", {})

        # Get output directory from OutputManager
        from cogniverse_core.common.utils.output_manager import get_output_manager

        output_manager = get_output_manager()

        return cls(
            extract_keyframes=pipeline_config.get("extract_keyframes", True),
            transcribe_audio=pipeline_config.get("transcribe_audio", True),
            generate_descriptions=pipeline_config.get("generate_descriptions", True),
            generate_embeddings=pipeline_config.get("generate_embeddings", True),
            keyframe_threshold=pipeline_config.get("keyframe_threshold", 0.999),
            max_frames_per_video=pipeline_config.get("max_frames_per_video", 3000),
            vlm_batch_size=pipeline_config.get("vlm_batch_size", 500),
            search_backend=config.get("search_backend", "byaldi"),
            output_dir=output_manager.get_processing_dir(),
            media_root_uri=pipeline_config.get("media_root_uri"),
        )

    @classmethod
    def from_profile(cls, profile_name: str) -> "PipelineConfig":
        """Load pipeline config for a specific profile"""
        from cogniverse_foundation.config.utils import create_default_config_manager

        config_manager = create_default_config_manager()
        config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)

        # Get profile-specific config from backend section
        backend_config = config.get("backend", {})
        profiles = backend_config.get("profiles", {})
        if profile_name not in profiles:
            raise ValueError(f"Profile '{profile_name}' not found in config")

        profile_config = profiles[profile_name]
        pipeline_config = profile_config.get("pipeline_config", {})

        # Get output directory from OutputManager
        from cogniverse_core.common.utils.output_manager import get_output_manager

        output_manager = get_output_manager()

        return cls(
            extract_keyframes=pipeline_config.get("extract_keyframes", True),
            transcribe_audio=pipeline_config.get("transcribe_audio", True),
            generate_descriptions=pipeline_config.get("generate_descriptions", True),
            generate_embeddings=pipeline_config.get("generate_embeddings", True),
            keyframe_threshold=pipeline_config.get("keyframe_threshold", 0.999),
            max_frames_per_video=pipeline_config.get("max_frames_per_video", 3000),
            vlm_batch_size=pipeline_config.get("vlm_batch_size", 500),
            search_backend=config.get("search_backend", "byaldi"),
            output_dir=output_manager.get_processing_dir(),
            media_root_uri=pipeline_config.get("media_root_uri"),
        )


class VideoIngestionPipeline:
    """
    Unified video processing pipeline with async optimizations.

    This class now incorporates async optimizations by default:
    - Single event loop for all async operations
    - Concurrent cache checks
    - Support for parallel video processing

    The pipeline can run in serial mode (max_concurrent=1) or
    parallel mode (max_concurrent>1) as needed.
    """

    def __init__(
        self,
        tenant_id: str,
        config: PipelineConfig | None = None,
        app_config: dict[str, Any] | None = None,
        config_manager=None,
        schema_loader=None,
        schema_name: str | None = None,
        debug_mode: bool = False,
        event_queue: Optional[EventQueue] = None,
    ):
        """
        Initialize the video ingestion pipeline with async support

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Pipeline configuration
            app_config: Application configuration
            config_manager: ConfigManager instance (required if app_config not provided)
            schema_loader: SchemaLoader instance (optional, for backend operations)
            schema_name: Schema/profile name
            debug_mode: Enable debug logging
            event_queue: Optional EventQueue for real-time progress notifications

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        self.tenant_id = tenant_id
        self.config_manager = config_manager
        self.schema_loader = schema_loader
        self.event_queue = event_queue
        self.job_id: Optional[str] = None  # Set when processing starts

        if config is None:
            if config_manager is None:
                raise ValueError(
                    "config_manager is required when config is not provided"
                )
            self.config = PipelineConfig.from_config(
                tenant_id=tenant_id, config_manager=config_manager
            )
        else:
            self.config = config

        if app_config is None:
            if config_manager is None:
                raise ValueError(
                    "config_manager is required when app_config is not provided"
                )
            self.app_config = get_config(
                tenant_id=tenant_id, config_manager=config_manager
            )
        else:
            self.app_config = app_config
        self.schema_name = schema_name
        self.debug_mode = debug_mode

        # Initialize logging with unique logger per profile
        logger_name = (
            f"{self.__class__.__name__}_{schema_name}"
            if schema_name
            else self.__class__.__name__
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.handlers.clear()
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        self.logger.setLevel(log_level)
        self._setup_logging()

        self.logger.info(
            f"VideoIngestionPipeline initialized - logging to: {self.log_file}"
        )
        self.logger.info(f"Backend: {self.config.search_backend}")
        if self.schema_name:
            self.logger.info(f"Schema/Profile: {self.schema_name}")
        self.logger.info(f"Output directory: {self.profile_output_dir}")
        self.logger.info(f"Pipeline config: {self.config}")

        self._init_cache()
        self._init_locator()
        self._resolve_strategy()
        self.processor_manager = ProcessorManager(self.logger)
        self.strategy_set = self._create_strategy_set_from_config()
        # Resolve inference-service names to URLs from system_config so
        # processors that need a remote sidecar (e.g. AudioProcessor →
        # whisper) receive a concrete endpoint without reading env vars.
        # Tests that pass ``config`` + ``app_config`` directly (no
        # config_manager) get an empty map; any strategy that requests an
        # inference_service in that scenario will fail loud at processor
        # init, which is the right behaviour.
        if self.config_manager is None:
            service_urls: dict[str, str] = {}
        else:
            service_urls = (
                self.config_manager.get_system_config().inference_service_urls
            )
        self.processor_manager.initialize_from_strategies(
            self.strategy_set, service_urls=service_urls
        )
        self._init_backend()
        self.logger.info("Using system event loop for async operations")

    def _setup_logging(self):
        """Setup logging for this pipeline instance"""
        from cogniverse_core.common.utils.output_manager import get_output_manager

        output_manager = get_output_manager()

        # Create profile-specific directory if schema_name is provided
        if self.schema_name:
            self.profile_output_dir = (
                output_manager.get_processing_dir() / f"profile_{self.schema_name}"
            )
        else:
            self.profile_output_dir = output_manager.get_processing_dir()

        self.profile_output_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = int(time.time())
        log_filename = (
            f"video_processing_{self.schema_name}_{timestamp}.log"
            if self.schema_name
            else f"video_processing_{timestamp}.log"
        )
        self.log_file = output_manager.get_logs_dir() / log_filename

        file_handler = logging.FileHandler(self.log_file)
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        file_handler.setLevel(log_level)
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(detailed_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        simple_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(simple_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        self.logger.setLevel(log_level)

        if self.debug_mode:
            self.logger.info("🔧 Debug mode enabled - detailed logging active")

    def _init_cache(self):
        """Initialize pipeline cache if available"""
        self.cache = None

        # Check if cache is enabled in pipeline_cache config
        cache_config_dict = self.app_config.get("pipeline_cache", {})

        if not cache_config_dict.get("enabled", False):
            self.logger.info("Pipeline cache is disabled")
            return

        from cogniverse_core.common.cache import CacheConfig, CacheManager

        serialization_format = cache_config_dict.get("serialization_format", "pickle")
        cache_config = CacheConfig(
            backends=cache_config_dict.get("backends", []),
            default_ttl=cache_config_dict.get("default_ttl", 0),
            enable_compression=cache_config_dict.get("enable_compression", True),
            serialization_format=serialization_format,
        )

        self.cache_manager = CacheManager(cache_config)
        self.cache = PipelineArtifactCache(
            self.cache_manager,
            ttl=cache_config_dict.get("default_ttl", 0),
            profile=self.schema_name,
        )
        self.logger.info(f"Initialized pipeline cache for profile: {self.schema_name}")

    def _init_locator(self):
        """Initialize the MediaLocator from app config."""
        media_section = self.app_config.get("media", {})
        media_config = (
            MediaConfig.from_dict(media_section) if media_section else MediaConfig()
        )
        self.locator = MediaLocator(tenant_id=self.tenant_id, config=media_config)
        self.logger.info(
            "Initialized MediaLocator (default_scheme=%s, uri_prefix=%r)",
            media_config.default_uri_scheme,
            media_config.uri_prefix,
        )

    def _canonical_uri(self, video_path_or_uri: Path | str) -> str:
        """Return the canonical URI for a path-or-URI input."""
        raw = str(video_path_or_uri)
        return raw if "://" in raw else f"file://{Path(raw).resolve()}"

    def _display_name(self, video_path_or_uri: Path | str) -> str:
        """Human-readable name for logging, works for both Paths and URIs."""
        raw = str(video_path_or_uri)
        if "://" in raw:
            return Path(raw.split("://", 1)[1]).name or raw
        return Path(raw).name

    def _resolve_strategy(self):
        self.strategy = None
        self.single_vector_processor = None

    def _create_strategy_set_from_config(self):
        """Create strategy set from profile configuration."""
        if not self.schema_name:
            # No profile specified, use basic defaults
            profile_config = {
                "strategies": {
                    "segmentation": {
                        "class": "FrameSegmentationStrategy",
                        "params": {"fps": 1.0},
                    },
                    "transcription": {
                        "class": "AudioTranscriptionStrategy",
                        "params": {},
                    },
                    "description": {"class": "NoDescriptionStrategy", "params": {}},
                    "embedding": {
                        "class": "MultiVectorEmbeddingStrategy",
                        "params": {},
                    },
                }
            }
            return StrategyFactory.create_from_profile_config(profile_config)

        backend_config = self.app_config.get("backend", {})
        profiles = backend_config.get("profiles", {})
        profile_config = profiles.get(self.schema_name, {})

        if "strategies" not in profile_config:
            raise ValueError(
                f"Profile {self.schema_name} missing 'strategies' configuration. "
                f"All profiles must use explicit strategy configuration."
            )

        return StrategyFactory.create_from_profile_config(profile_config)

    def _get_chunk_duration(self) -> float:
        """Get chunk duration from profile configuration"""
        if not self.schema_name:
            return 30.0  # Default

        backend_config = self.app_config.get("backend", {})
        profiles = backend_config.get("profiles", {})
        profile_config = profiles.get(self.schema_name, {})

        # Extract from processing type (e.g., "direct_video/chunks:30s")
        processing_type = profile_config.get("strategy", {}).get("processing", "")
        if ":" in processing_type:
            duration_str = processing_type.split(":")[1]
            if duration_str.endswith("s"):
                return float(duration_str[:-1])

        pipeline_config = profile_config.get("pipeline_config", {})
        return pipeline_config.get("chunk_duration", 30.0)

    def _init_backend(self):
        """Initialize embedding generation backend"""
        if not self.config.generate_embeddings:
            self.embedding_generator = None
            return

        self.logger.info(
            f"Initializing embedding generator with schema_name={self.schema_name}, tenant_id={self.tenant_id}"
        )
        self.embedding_generator = create_embedding_generator(
            config=self.app_config,
            schema_name=self.schema_name,
            tenant_id=self.tenant_id,
            logger=self.logger,
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )
        self.logger.info(
            f"Initialized embedding generator for tenant {self.tenant_id} "
            f"with backend: {self.config.search_backend}"
        )

        # Set embedding generator for single vector processor if needed
        if self.single_vector_processor:
            self.single_vector_processor.embedding_generator = self.embedding_generator

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        import cv2

        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e:
            self.logger.warning(f"Could not determine duration for {video_path}: {e}")
            return 0.0

    async def _check_cache_async(self, video_path: Path) -> dict[str, Any]:
        """
        Check all cache entries concurrently
        Returns dict with cached results or None for each step
        """
        if not self.cache:
            return {}

        cache_tasks = []
        cache_keys = []

        # Get profile configuration for cache parameters
        backend_config = self.app_config.get("backend", {})
        profiles = backend_config.get("profiles", {})
        profile_config = profiles.get(self.schema_name, {})
        pipeline_config = profile_config.get("pipeline_config", {})

        # Keyframes cache check
        if self.config.extract_keyframes:
            strategy = pipeline_config.get("keyframe_strategy", "similarity")
            cache_tasks.append(
                self.cache.get_keyframes(
                    str(video_path),
                    strategy=strategy,
                    threshold=pipeline_config.get(
                        "keyframe_threshold", self.config.keyframe_threshold
                    ),
                    fps=pipeline_config.get("keyframe_fps", 1.0),
                    max_frames=self.config.max_frames_per_video,
                    load_images=True,
                )
            )
            cache_keys.append("keyframes")

        # Transcript cache check
        if self.config.transcribe_audio and self.audio_transcriber:
            cache_tasks.append(
                self.cache.get_transcript(
                    str(video_path),
                    model_size=self.audio_transcriber.model_size,
                    language=None,
                )
            )
            cache_keys.append("transcript")

        # Descriptions cache check
        vlm_processor = self.processor_manager.get_processor("vlm")
        if self.config.generate_descriptions and vlm_processor:
            cache_tasks.append(
                self.cache.get_descriptions(
                    str(video_path),
                    model_name="Qwen/Qwen2-VL-7B-Instruct",
                    batch_size=self.config.vlm_batch_size,
                )
            )
            cache_keys.append("descriptions")

        # Execute all cache checks concurrently
        if cache_tasks:
            self.logger.info(
                f"Checking {len(cache_tasks)} cache entries concurrently for {video_path.name}"
            )
            start_time = time.time()
            cache_results = await asyncio.gather(*cache_tasks, return_exceptions=True)
            elapsed = time.time() - start_time
            self.logger.info(f"Cache checks completed in {elapsed:.2f}s")

            # Build results dict
            results = {}
            for key, result in zip(cache_keys, cache_results, strict=False):
                if isinstance(result, Exception):
                    self.logger.warning(f"Cache check failed for {key}: {result}")
                    results[key] = None
                else:
                    results[key] = result
                    if result:
                        self.logger.info(f"Cache HIT for {key}")
                    else:
                        self.logger.info(f"Cache MISS for {key}")

            return results

        return {}

    async def _save_to_cache_async(
        self, video_path: Path, step: str, data: Any, **kwargs
    ) -> None:
        """
        Save data to cache asynchronously
        """
        if not self.cache or not data:
            return

        if step == "keyframes":
            import cv2

            video_id = video_path.stem
            keyframes_dir = self.profile_output_dir / "keyframes" / video_id
            images = {}
            for kf in data.get("keyframes", []):
                if "filename" in kf:
                    frame_path = keyframes_dir / kf["filename"]
                    if frame_path.exists():
                        image = cv2.imread(str(frame_path))
                        if image is not None:
                            images[str(kf["frame_id"])] = image
            await self.cache.set_keyframes(str(video_path), data, images, **kwargs)
        elif step == "transcript":
            await self.cache.set_transcript(str(video_path), data, **kwargs)
        elif step == "descriptions":
            await self.cache.set_descriptions(
                str(video_path), data.get("descriptions", {}), **kwargs
            )
        else:
            raise ValueError(f"Unknown cache step: {step!r}")
        self.logger.info(f"Cached {step} for {video_path.name}")

    def _extract_base_video_data(self, results: dict[str, Any]) -> dict[str, Any]:
        """Extract base video metadata from results"""
        return {
            "video_id": results.get("video_id"),
            "video_path": results.get("video_path"),
            "duration": results.get("duration", 0),
            "output_dir": str(self.profile_output_dir),
        }

    def _add_strategy_metadata(self, video_data: dict[str, Any]) -> dict[str, Any]:
        """Add strategy-related metadata to video data"""
        if self.strategy:
            video_data["processing_type"] = self.strategy.processing_type
            video_data["storage_mode"] = self.strategy.storage_mode
            video_data["schema_name"] = self.strategy.schema_name
        return video_data

    def _process_chunk_data(
        self, video_data: dict[str, Any], results: dict[str, Any]
    ) -> dict[str, Any]:
        """Process chunk-based video data.

        ``ChunkSegmentationStrategy`` lands its output under
        ``results["results"]["video_chunks"]`` (a dict with a
        ``chunks`` list inside). Older code paths that wrote directly
        to ``results["results"]["chunks"]`` are still supported.
        """
        inner = results.get("results", {})
        if "chunks" in inner:
            video_data["chunks"] = inner["chunks"]
        elif "video_chunks" in inner:
            chunk_result = inner["video_chunks"]
            if isinstance(chunk_result, dict):
                video_data["chunks"] = chunk_result.get("chunks", [])
            else:
                video_data["chunks"] = chunk_result
        else:
            self.logger.warning("No chunks data found in results")
            return video_data

        if "transcript" in inner:
            video_data["transcript"] = inner["transcript"]

        self.logger.info(
            "Using video chunks data (%d chunks)", len(video_data.get("chunks", []))
        )
        return video_data

    def _process_single_vector_data(
        self, video_data: dict[str, Any], results: dict[str, Any]
    ) -> dict[str, Any]:
        """Process single vector video data"""
        processing_data = results["results"]["single_vector_processing"]
        # Use dictionary segments (VideoSegment objects already converted to dicts)
        video_data["segments"] = processing_data["segments"]
        video_data["processing_metadata"] = processing_data["metadata"]
        video_data["full_transcript"] = processing_data["full_transcript"]
        video_data["document_structure"] = processing_data["document_structure"]
        self.logger.info(
            f"Using single vector processing data with {len(video_data['segments'])} segments"
        )
        return video_data

    def _process_frame_data(
        self, video_data: dict[str, Any], results: dict[str, Any]
    ) -> dict[str, Any]:
        """Process frame-based video data"""
        keyframes_data = results["results"]["keyframes"]

        # Pass through the keyframes directly - no renaming needed
        video_data["keyframes"] = keyframes_data

        # Add transcript if available
        if "transcript" in results.get("results", {}):
            video_data["transcript"] = results["results"]["transcript"]

        # Add descriptions if available
        if "descriptions" in results.get("results", {}):
            video_data["descriptions"] = results["results"]["descriptions"]

        return video_data

    def _prepare_video_data(self, results: dict[str, Any]) -> dict[str, Any]:
        """Prepare video data for embedding generation based on processing type"""
        video_data = self._extract_base_video_data(results)
        video_data = self._add_strategy_metadata(video_data)

        if "document_files" in results:
            video_data["document_files"] = results["document_files"]
            return video_data
        elif "audio_files" in results:
            video_data["audio_files"] = results["audio_files"]
            if "transcript" in results:
                video_data["transcript"] = results["transcript"]
            return video_data

        # Handle both 'chunks' and 'video_chunks' keys
        if "chunks" in results.get("results", {}) or "video_chunks" in results.get(
            "results", {}
        ):
            video_data = self._process_chunk_data(video_data, results)
        elif "single_vector_processing" in results.get("results", {}):
            video_data = self._process_single_vector_data(video_data, results)
        elif (
            "keyframes" in results.get("results", {})
            and video_data.get("processing_type") != "video_chunks"
        ):
            video_data = self._process_frame_data(video_data, results)

        return video_data

    def _convert_embedding_result(self, result: Any) -> dict[str, Any]:
        """Convert EmbeddingResult to dict format expected by pipeline"""
        return {
            "video_id": result.video_id,
            "total_documents": result.total_documents,
            "documents_processed": result.documents_processed,
            "documents_fed": result.documents_fed,
            "processing_time": result.processing_time,
            "errors": result.errors,
            "metadata": result.metadata,
            "backend": self.config.search_backend,
        }

    async def generate_embeddings(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings for search backend using EmbeddingGenerator v2"""
        if not self.embedding_generator:
            self.logger.error("Embedding generator not initialized")
            return {"error": "Embedding generator not initialized"}

        self.logger.info("Extracting data for embedding generation...")
        self.logger.info(f"Results keys: {list(results.keys())}")

        # Prepare video data using extracted methods
        video_data = self._prepare_video_data(results)

        self.logger.info(f"Data extracted - Video ID: {video_data['video_id']}")
        self.logger.info(f"Output directory: {video_data.get('output_dir')}")

        # Run synchronous PyTorch inference in a thread pool so the event loop
        # stays responsive during concurrent batch ingestion.
        result = await asyncio.to_thread(
            self.embedding_generator.generate_embeddings,
            video_data,
            self.profile_output_dir,
        )

        # Convert result to expected format
        return self._convert_embedding_result(result)

    async def _emit_event(self, event) -> None:
        """Emit event to EventQueue if configured."""
        if self.event_queue is not None:
            await self.event_queue.enqueue(event)

    def _is_cancelled(self) -> bool:
        """Check if pipeline execution has been cancelled."""
        if self.event_queue is None:
            return False
        return self.event_queue.cancellation_token.is_cancelled

    async def process_video_async_with_strategies(
        self, video_path: Path | str
    ) -> dict[str, Any]:
        """
        Process video using the strategy pattern - strategies orchestrate everything

        Accepts either a local ``Path`` or a URI string (``file://``, ``s3://``,
        ``pvc://``, ``http(s)://``). URIs are resolved to a local path via the
        :class:`MediaLocator` once at this entry point; processors downstream
        continue to receive a ``Path``.
        """
        if isinstance(video_path, str):
            video_uri = self._canonical_uri(video_path)
            video_path = self.locator.localize(video_path)
        else:
            video_uri = self._canonical_uri(video_path)

        video_id = video_path.stem

        self.logger.info(f"Starting async video processing with strategies: {video_id}")
        self.logger.info(f"Video path: {video_path}")
        self.logger.info(f"Source URI: {video_uri}")

        print(f"\n🎬 Processing video (async): {video_path.name}")
        print("=" * 60)

        self.video_path = video_path
        self.video_uri = video_uri

        # Prepare base results
        results = self._prepare_base_results(video_path)

        # Emit video processing start event
        if self.job_id:
            await self._emit_event(
                create_status_event(
                    task_id=self.job_id,
                    tenant_id=self.tenant_id,
                    state=TaskState.WORKING,
                    phase=f"video_{video_id}",
                    message=f"Processing video: {video_path.name}",
                )
            )

        try:
            # Check for cancellation before processing
            if self._is_cancelled():
                self.logger.info(f"Cancelled before processing video: {video_id}")
                results["status"] = "cancelled"
                results["error"] = "Pipeline cancelled"
                return results

            # Let strategy set orchestrate everything
            self.logger.info("Delegating to ProcessingStrategySet.process()")
            processing_results = await self.strategy_set.process(
                video_path=video_path,
                processor_manager=self.processor_manager,
                pipeline_context=self,
            )

            # Add processing results to our results structure
            results["results"] = processing_results

            # Calculate total time
            total_time = time.time() - results["started_at"]
            results["status"] = "completed"
            results["total_processing_time"] = total_time

            self.logger.info(f"Async video processing completed in {total_time:.2f}s")
            print(f"\n✅ Video processing completed in {total_time:.1f}s")

            # Emit video completion event
            if self.job_id:
                await self._emit_event(
                    create_progress_event(
                        task_id=self.job_id,
                        tenant_id=self.tenant_id,
                        current=1,
                        total=1,
                        step=f"video_{video_id}_complete",
                        details={
                            "video_id": video_id,
                            "processing_time": total_time,
                        },
                    )
                )

            return results

        except PipelineException as e:
            self.logger.error(
                f"Video processing failed with pipeline error: {e}", exc_info=True
            )
            print(f"\n❌ Video processing failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["error_type"] = type(e).__name__
            results["error_context"] = getattr(e, "context", {})

            # Emit error event
            if self.job_id:
                await self._emit_event(
                    create_error_event(
                        task_id=self.job_id,
                        tenant_id=self.tenant_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"video_id": video_id, "video_path": str(video_path)},
                        recoverable=True,
                    )
                )

            return results
        except Exception as e:
            # Wrap unexpected exceptions as ContentProcessingError
            wrapped_error = wrap_content_error(
                video_path, "unknown", self.schema_name, e
            )
            self.logger.error(
                f"Video processing failed with unexpected error: {wrapped_error}",
                exc_info=True,
            )
            print(f"\n❌ Video processing failed: {wrapped_error}")
            results["status"] = "failed"
            results["error"] = str(wrapped_error)
            results["error_type"] = type(wrapped_error).__name__
            results["error_context"] = wrapped_error.context

            # Emit error event
            if self.job_id:
                await self._emit_event(
                    create_error_event(
                        task_id=self.job_id,
                        tenant_id=self.tenant_id,
                        error_type=type(wrapped_error).__name__,
                        error_message=str(wrapped_error),
                        context=wrapped_error.context,
                        recoverable=False,
                    )
                )

            return results

    def _prepare_base_results(self, video_path: Path) -> dict[str, Any]:
        """Prepare base results structure"""
        config_dict = self.config.__dict__.copy()
        config_dict["video_dir"] = str(config_dict["video_dir"])
        config_dict["output_dir"] = str(config_dict["output_dir"])

        return {
            "video_id": video_path.stem,
            "video_path": str(video_path),
            "source_url": getattr(self, "video_uri", self._canonical_uri(video_path)),
            "duration": self._get_video_duration(video_path),
            "pipeline_config": config_dict,
            "results": {},
            "started_at": time.time(),
            "async_optimized": True,
        }

    async def _get_cached_data(
        self, video_path: Path, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Get cached data with timing"""
        if not self.cache:
            return {}

        cache_start = time.time()
        cached_data = await self._check_cache_async(video_path)
        cache_time = time.time() - cache_start
        self.logger.info(f"Concurrent cache checks completed in {cache_time:.2f}s")
        results["cache_check_time"] = cache_time
        return cached_data

    async def _process_segmentation(
        self, video_path: Path, cached_data: dict[str, Any], results: dict[str, Any]
    ) -> dict[str, Any]:
        """Process video segmentation based on strategy"""
        # Special handling for different segmentation types
        from cogniverse_runtime.ingestion.strategies import (
            FrameSegmentationStrategy,
            SingleVectorSegmentationStrategy,
        )

        if isinstance(self.strategy_set.segmentation, SingleVectorSegmentationStrategy):
            # Single-vector needs transcript first
            transcript_data = None
            if self.config.transcribe_audio:
                transcript_data = await self.strategy_set.transcription.transcribe(
                    video_path, self, cached_data
                )
                if transcript_data:
                    results["results"]["transcript"] = transcript_data

            # Process with single-vector
            seg_result = await self.strategy_set.segmentation.segment(
                video_path, self, transcript_data
            )
            if "single_vector_processing" in seg_result:
                results["results"]["single_vector_processing"] = seg_result[
                    "single_vector_processing"
                ]
        else:
            # Frame or chunk segmentation
            if isinstance(self.strategy_set.segmentation, FrameSegmentationStrategy):
                # Check cache for frames
                if cached_data.get("keyframes"):
                    cached_keyframes = cached_data["keyframes"]
                    if isinstance(cached_keyframes, tuple):
                        keyframes_data, _ = (
                            cached_keyframes  # Data is first, images second
                        )
                    else:
                        keyframes_data = cached_keyframes
                    results["results"]["keyframes"] = keyframes_data
                    self.logger.info(
                        f"Using cached keyframes: {len(keyframes_data.get('keyframes', []))} frames"
                    )
                else:
                    seg_result = await self.strategy_set.segmentation.segment(
                        video_path, self
                    )
                    results["results"]["keyframes"] = seg_result
                    # Cache if needed
                    if self.cache and seg_result and "keyframes" in seg_result:
                        await self._save_to_cache_async(
                            video_path,
                            "keyframes",
                            seg_result,
                            profile=self.schema_name,
                        )
            else:
                # Chunk segmentation
                seg_result = await self.strategy_set.segmentation.segment(
                    video_path, self
                )
                if "chunks" in seg_result:
                    results["results"]["chunks"] = seg_result

        return results

    async def process_video_async(self, video_path: Path | str) -> dict[str, Any]:
        """
        Process video using the strategy pattern
        """
        return await self.process_video_async_with_strategies(video_path)

    async def process_videos_concurrent(
        self,
        video_files: list[Path] | list[str] | list[Path | str],
        max_concurrent: int = 3,
    ) -> dict[str, Any]:
        """
        Process multiple videos concurrently with resource control

        Args:
            video_files: List of video paths to process
            max_concurrent: Maximum number of videos to process simultaneously

        Returns:
            Dict with job_id and list of results for each video
        """
        # Generate job_id for this batch
        self.job_id = f"ingestion_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        self.logger.info(
            f"Starting ingestion job {self.job_id} with {len(video_files)} videos"
        )

        # Emit job start event
        await self._emit_event(
            create_status_event(
                task_id=self.job_id,
                tenant_id=self.tenant_id,
                state=TaskState.WORKING,
                phase="starting",
                message=f"Starting ingestion of {len(video_files)} videos",
            )
        )

        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0

        async def process_with_limit(video_path: Path | str, index: int, total: int):
            """Process a video with concurrency limit and progress tracking"""
            nonlocal completed_count
            display = self._display_name(video_path)

            # Check for cancellation
            if self._is_cancelled():
                return {
                    "video_path": str(video_path),
                    "status": "cancelled",
                    "error": "Pipeline cancelled",
                }

            async with semaphore:
                try:
                    self.logger.info(
                        f"[{index}/{total}] Starting concurrent processing: {display}"
                    )
                    print(f"\n🎯 [{index}/{total}] Processing: {display}")

                    # Emit progress event before processing
                    await self._emit_event(
                        create_progress_event(
                            task_id=self.job_id,
                            tenant_id=self.tenant_id,
                            current=completed_count,
                            total=total,
                            step=f"processing_video_{index}",
                            details={"video": display},
                        )
                    )

                    result = await self.process_video_async(video_path)

                    if result["status"] == "completed":
                        completed_count += 1
                        self.logger.info(
                            f"[{index}/{total}] Completed: {display} in {result['total_processing_time']:.1f}s"
                        )
                        print(
                            f"✅ [{index}/{total}] Completed: {display} ({result['total_processing_time']:.1f}s)"
                        )
                    else:
                        self.logger.error(
                            f"[{index}/{total}] Failed: {display} - {result.get('error')}"
                        )
                        print(f"❌ [{index}/{total}] Failed: {display}")

                    return result

                except PipelineException as e:
                    self.logger.error(
                        f"[{index}/{total}] Pipeline exception processing {display}: {e}"
                    )
                    return {
                        "video_path": str(video_path),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_context": getattr(e, "context", {}),
                        "status": "failed",
                    }
                except Exception as e:
                    wrapped_error = wrap_content_error(
                        video_path, "concurrent_processing", self.schema_name, e
                    )
                    self.logger.error(
                        f"[{index}/{total}] Unexpected exception processing {display}: {wrapped_error}"
                    )
                    return {
                        "video_path": str(video_path),
                        "error": str(wrapped_error),
                        "error_type": type(wrapped_error).__name__,
                        "error_context": wrapped_error.context,
                        "status": "failed",
                    }

        # Create tasks for all videos
        tasks = [
            process_with_limit(video, i + 1, len(video_files))
            for i, video in enumerate(video_files)
        ]

        # Process all videos concurrently
        self.logger.info(
            f"Starting concurrent processing of {len(video_files)} videos (max {max_concurrent} concurrent)"
        )
        print(
            f"\n🚀 Processing {len(video_files)} videos concurrently (max {max_concurrent} at once)"
        )

        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.time() - start_time

        # Calculate statistics
        successful = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "completed"
        )
        failed = len(results) - successful
        cancelled = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "cancelled"
        )

        self.logger.info(
            f"Concurrent processing completed: {successful}/{len(video_files)} successful in {total_time:.1f}s"
        )
        print(f"\n🏁 Concurrent processing completed in {total_time:.1f}s")
        print(f"   ✅ Successful: {successful}/{len(video_files)}")
        print(f"   ❌ Failed: {failed}/{len(video_files)}")
        print(f"   ⚡ Average time: {total_time / len(video_files):.1f}s per video")

        # Emit completion event
        if cancelled > 0:
            await self._emit_event(
                create_status_event(
                    task_id=self.job_id,
                    tenant_id=self.tenant_id,
                    state=TaskState.CANCELLED,
                    phase="completed",
                    message=f"Ingestion cancelled: {successful} completed, {cancelled} cancelled",
                )
            )
        elif failed > 0:
            await self._emit_event(
                create_complete_event(
                    task_id=self.job_id,
                    tenant_id=self.tenant_id,
                    result={
                        "successful": successful,
                        "failed": failed,
                        "total": len(video_files),
                    },
                    summary=f"Ingestion completed with errors: {successful}/{len(video_files)} successful",
                    execution_time_seconds=total_time,
                )
            )
        else:
            await self._emit_event(
                create_complete_event(
                    task_id=self.job_id,
                    tenant_id=self.tenant_id,
                    result={
                        "successful": successful,
                        "failed": 0,
                        "total": len(video_files),
                    },
                    summary=f"Ingestion completed successfully: {successful}/{len(video_files)} videos",
                    execution_time_seconds=total_time,
                )
            )

        # Return structured result with job_id
        return {
            "job_id": self.job_id,
            "status": (
                "cancelled"
                if cancelled > 0
                else ("completed" if failed == 0 else "completed_with_errors")
            ),
            "total_videos": len(video_files),
            "successful": successful,
            "failed": failed,
            "cancelled": cancelled,
            "execution_time_seconds": total_time,
            "results": results,
        }

    def get_video_files(self, video_dir: Path | None = None) -> list[Path] | list[str]:
        """Get list of videos to process.

        When ``self.config.media_root_uri`` is set, enumerates via
        :meth:`MediaLocator.list` and returns canonical URI strings.
        Otherwise globs ``video_dir`` and returns local ``Path`` objects.
        """
        if self.config.media_root_uri:
            return list(self.locator.list(self.config.media_root_uri))

        if video_dir is None:
            video_dir = self.config.video_dir

        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        video_files: list[Path] = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
        return sorted(video_files)

    def process_directory(
        self, video_dir: Path | None = None, max_concurrent: int = 3
    ) -> dict[str, Any]:
        """
        Process all videos in a directory with concurrent processing

        Args:
            video_dir: Directory containing videos. Ignored when
                ``self.config.media_root_uri`` is set.
            max_concurrent: Maximum number of videos to process simultaneously
        """
        if self.config.media_root_uri:
            source_label = self.config.media_root_uri
            video_files = self.get_video_files()
        else:
            video_dir = video_dir or self.config.video_dir
            source_label = str(video_dir)
            video_files = self.get_video_files(video_dir)

        if not video_files:
            self.logger.error(f"No video files found in {source_label}")
            print(f"❌ No video files found in {source_label}")
            return {"error": "No video files found"}

        self.logger.info(
            f"Starting concurrent batch processing: {len(video_files)} videos from {source_label}"
        )

        print(f"🎬 Found {len(video_files)} videos to process")
        print(f"📁 Output directory: {self.profile_output_dir}")
        print(f"⚙️ Pipeline config: {self.config}")
        print(f"🚀 Concurrent processing: max {max_concurrent} videos at once")

        # Convert PosixPath objects to strings for JSON serialization
        config_dict = self.config.__dict__.copy()
        config_dict["video_dir"] = str(config_dict["video_dir"])
        config_dict["output_dir"] = str(config_dict["output_dir"])

        results = {
            "profile": self.schema_name,
            "pipeline_config": config_dict,
            "output_directory": str(self.profile_output_dir),
            "total_videos": len(video_files),
            "max_concurrent": max_concurrent,
            "async_optimized": True,
            "phase2_concurrent": True,
            "started_at": time.time(),
        }

        # Process videos concurrently
        batch_result = asyncio.run(
            self.process_videos_concurrent(video_files, max_concurrent)
        )

        # Extract job_id and video results from batch result
        results["job_id"] = batch_result.get("job_id")
        video_results = batch_result.get("results", [])

        # Separate successful and failed results
        results["processed_videos"] = [
            r for r in video_results if r.get("status") == "completed"
        ]
        results["failed_videos"] = [
            r for r in video_results if r.get("status") != "completed"
        ]

        results["completed_at"] = time.time()
        results["total_processing_time"] = (
            results["completed_at"] - results["started_at"]
        )

        # Save summary (clean up any non-serializable data first)
        summary_file = self.profile_output_dir / "pipeline_summary.json"

        # Remove any raw segments from processed videos before saving
        import copy

        results_to_save = copy.deepcopy(results)
        for video_result in results_to_save.get("processed_videos", []):
            if "_raw_segments" in video_result:
                del video_result["_raw_segments"]

        with open(summary_file, "w") as f:
            json.dump(results_to_save, f, indent=2)

        # Log final summary
        self.logger.info("Concurrent batch processing completed!")
        self.logger.info(
            f"Summary - Total: {len(video_files)}, Processed: {len(results['processed_videos'])}, Failed: {len(results['failed_videos'])}"
        )
        self.logger.info(
            f"Total time: {results['total_processing_time']:.2f} seconds ({results['total_processing_time'] / 60:.1f} minutes)"
        )
        self.logger.info(
            f"Average time per video: {results['total_processing_time'] / len(video_files):.2f} seconds"
        )
        self.logger.info(f"Summary saved to: {summary_file}")

        print("\n🎉 Pipeline completed!")
        print(
            f"✅ Processed: {len(results['processed_videos'])}/{len(video_files)} videos"
        )
        print(f"❌ Failed: {len(results['failed_videos'])} videos")
        print(f"⏱️ Total time: {results['total_processing_time'] / 60:.1f} minutes")
        print(
            f"⚡ Throughput: {len(video_files) / results['total_processing_time'] * 60:.1f} videos/minute"
        )
        print(f"📄 Summary saved: {summary_file}")

        # Cleanup all processors (including VLM service shutdown)
        self.processor_manager.cleanup()
        self.logger.info("Processors cleaned up")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--backend", choices=["byaldi", "vespa"], default="vespa", help="Search backend"
    )
    parser.add_argument("--profile", type=str, help="Video processing profile")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per video")

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent videos (default: 3)",
    )

    # Pipeline step toggles
    parser.add_argument(
        "--skip-keyframes", action="store_true", help="Skip keyframe extraction"
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip audio transcription"
    )
    parser.add_argument(
        "--skip-descriptions", action="store_true", help="Skip VLM descriptions"
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true", help="Skip embedding generation"
    )

    args = parser.parse_args()

    if args.profile:
        import os

        os.environ["VIDEO_PROFILE"] = args.profile

    config = PipelineConfig.from_config()

    if args.video_dir:
        config.video_dir = args.video_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.backend:
        config.search_backend = args.backend
    if args.max_frames:
        config.max_frames_per_video = args.max_frames

    if args.skip_keyframes:
        config.extract_keyframes = False
    if args.skip_audio:
        config.transcribe_audio = False
    if args.skip_descriptions:
        config.generate_descriptions = False
    if args.skip_embeddings:
        config.generate_embeddings = False

    print("🚀 Starting Video Processing Pipeline")
    pipeline = VideoIngestionPipeline(config)
    results = pipeline.process_directory(max_concurrent=args.max_concurrent)
