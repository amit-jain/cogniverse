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

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.app.ingestion.processors import (
    KeyframeExtractor,
    AudioTranscriber,
    VLMDescriptor
)
from src.app.ingestion.processors.keyframe_extractor_fps import FPSKeyframeExtractor
from src.app.ingestion.processors.video_chunk_extractor import VideoChunkExtractor
from src.app.ingestion.processors.embedding_generator import create_embedding_generator
from src.common.cache import CacheManager, CacheConfig
from src.common.cache.pipeline_cache import PipelineArtifactCache
from src.app.ingestion.strategy import StrategyConfig


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
    
    # Backend selection
    search_backend: str = "byaldi"  # "byaldi" or "vespa"
    
    @classmethod
    def from_config(cls) -> 'PipelineConfig':
        """Load pipeline config from main config file"""
        config = get_config()
        pipeline_config = config.get("pipeline_config", {})
        
        # Get output directory from OutputManager
        from src.common.utils.output_manager import get_output_manager
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
            output_dir=output_manager.get_processing_dir()
        )
    
    @classmethod
    def from_profile(cls, profile_name: str) -> 'PipelineConfig':
        """Load pipeline config for a specific profile"""
        config = get_config()
        
        # Get profile-specific config
        profiles = config.get("video_processing_profiles", {})
        if profile_name not in profiles:
            raise ValueError(f"Profile '{profile_name}' not found in config")
        
        profile_config = profiles[profile_name]
        pipeline_config = profile_config.get("pipeline_config", {})
        
        # Get output directory from OutputManager
        from src.common.utils.output_manager import get_output_manager
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
            output_dir=output_manager.get_processing_dir()
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
    
    def __init__(self, config: Optional[PipelineConfig] = None, app_config: Optional[Dict[str, Any]] = None, schema_name: Optional[str] = None):
        """Initialize the video ingestion pipeline with async support"""
        self.config = config or PipelineConfig.from_config()
        self.app_config = app_config or get_config()
        self.schema_name = schema_name
        
        # Initialize logging with unique logger per profile
        logger_name = f"{self.__class__.__name__}_{schema_name}" if schema_name else self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        # Clear any existing handlers to avoid duplicate logging
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self._setup_logging()
        
        # Log configuration
        self.logger.info(f"VideoIngestionPipeline initialized - logging to: {self.log_file}")
        self.logger.info(f"Backend: {self.config.search_backend}")
        self.logger.info(f"Output directory: {self.profile_output_dir}")
        self.logger.info(f"Pipeline config: {self.config}")
        
        # Initialize cache
        self._init_cache()
        
        # Resolve strategy
        self._resolve_strategy()
        
        # Initialize processors
        self._init_processors()
        
        # Initialize backend
        self._init_backend()
        
        # Create a single event loop for the pipeline lifetime (async optimization)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger.info("Initialized pipeline with dedicated event loop")
    
    def _setup_logging(self):
        """Setup logging for this pipeline instance"""
        from src.common.utils.output_manager import get_output_manager
        output_manager = get_output_manager()
        
        # Create profile-specific directory if schema_name is provided
        if self.schema_name:
            self.profile_output_dir = output_manager.get_processing_dir() / f"profile_{self.schema_name}"
        else:
            self.profile_output_dir = output_manager.get_processing_dir()
        
        self.profile_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = int(time.time())
        log_filename = f"video_processing_{self.schema_name}_{timestamp}.log" if self.schema_name else f"video_processing_{timestamp}.log"
        self.log_file = output_manager.get_logs_dir() / log_filename
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler with simpler formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(simple_formatter)
        
        # Add both handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
    
    def _init_cache(self):
        """Initialize pipeline cache if available"""
        self.cache = None
        
        # Check if cache is enabled in pipeline_cache config
        cache_config_dict = self.app_config.get("pipeline_cache", {})
        
        if not cache_config_dict.get("enabled", False):
            self.logger.info("Pipeline cache is disabled")
            return
        
        try:
            # Create cache configuration
            from src.common.cache import CacheManager, CacheConfig
            
            cache_config = CacheConfig(
                backends=cache_config_dict.get("backends", []),
                default_ttl=cache_config_dict.get("default_ttl", 0),
                enable_compression=cache_config_dict.get("enable_compression", True),
                serialization_format=cache_config_dict.get("serialization_format", "pickle")
            )
            
            # Initialize cache manager
            self.cache_manager = CacheManager(cache_config)
            
            # Initialize pipeline artifact cache with profile
            self.cache = PipelineArtifactCache(
                self.cache_manager,
                ttl=cache_config_dict.get("default_ttl", 0),
                profile=self.schema_name  # Use schema_name as profile for namespacing
            )
            self.logger.info(f"Initialized pipeline cache for profile: {self.schema_name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize cache: {e}")
            self.cache = None
    
    def _resolve_strategy(self):
        """Resolve processing strategy from profile configuration"""
        self.strategy = None
        self.single_vector_processor = None
        
        if not self.schema_name:
            self.logger.info("No schema_name provided, using default frame-based processing")
            return
        
        # Get strategy from profile config
        from src.app.ingestion.strategy import Strategy, StrategyConfig
        
        # Use StrategyConfig to properly resolve strategy (as in old implementation)
        if self.schema_name:
            try:
                strategy_config = StrategyConfig()
                self.strategy = strategy_config.get_strategy(self.schema_name)
                self.logger.info(f"Resolved strategy: {self.strategy}")
            except Exception as e:
                self.logger.error(f"Failed to resolve strategy: {e}")
                self.strategy = None
        else:
            self.strategy = None
        
    
    def _init_processors(self):
        """Initialize video processing components based on configuration"""
        # Keyframe extractor
        self.keyframe_extractor = None
        self.video_chunk_extractor = None
        self.single_vector_processor = None
        
        # Initialize video chunk extractor for direct_video with chunks (multi-vector models)
        if self.strategy and self.strategy.processing_type == "direct_video" and self.strategy.segmentation in ["chunks", "windows"]:
            # This is for multi-vector models like ColQwen, VideoPrism that process video chunks
            # Get chunk duration from model config - use segment_duration or chunk_duration
            chunk_duration = self.strategy.model_config.get("segment_duration") or \
                           self.strategy.model_config.get("chunk_duration", 30.0)
            self.video_chunk_extractor = VideoChunkExtractor(
                chunk_duration=chunk_duration,
                chunk_overlap=0.0,
                cache_chunks=True
            )
            self.logger.info(f"Using VideoChunkExtractor for {self.strategy.processing_type}/{self.strategy.segmentation} processing with {chunk_duration}s chunks")
            # Don't need keyframe extractor for chunk processing
            self.keyframe_extractor = None
            self.single_vector_processor = None
        # Initialize processors based on strategy
        elif self.strategy and self.strategy.processing_type == "single_vector":
            # Initialize SingleVectorVideoProcessor
            from src.app.ingestion.processors.single_vector_processor import SingleVectorVideoProcessor
            
            model_config = self.strategy.model_config
            self.single_vector_processor = SingleVectorVideoProcessor(
                strategy=self.strategy.segmentation,
                segment_duration=model_config.get("chunk_duration", 6.0),
                segment_overlap=model_config.get("chunk_overlap", 1.0),
                sampling_fps=model_config.get("sampling_fps", 2.0),
                max_frames_per_segment=model_config.get("max_frames_per_chunk", 12),
                store_as_single_doc=(self.strategy.storage_mode == "single_doc"),
                cache=self.cache
            )
            self.logger.info(
                f"Using SingleVectorVideoProcessor - segmentation: {self.strategy.segmentation}, "
                f"storage: {self.strategy.storage_mode}"
            )
            # For single vector processing, we handle frames differently
            self.keyframe_extractor = None
            self.video_chunk_extractor = None
        elif self.config.extract_keyframes:
            # Get keyframe strategy from profile config
            keyframe_strategy = "similarity"
            if self.schema_name:
                profiles = self.app_config.get("video_processing_profiles", {})
                profile_config = profiles.get(self.schema_name, {})
                pipeline_config = profile_config.get("pipeline_config", {})
                keyframe_strategy = pipeline_config.get("keyframe_strategy", "similarity")
            
            if keyframe_strategy == "fps":
                fps = 1.0
                if self.schema_name:
                    profiles = self.app_config.get("video_processing_profiles", {})
                    profile_config = profiles.get(self.schema_name, {})
                    pipeline_config = profile_config.get("pipeline_config", {})
                    fps = pipeline_config.get("keyframe_fps", 1.0)
                self.keyframe_extractor = FPSKeyframeExtractor(
                    fps=fps,
                    max_frames=self.config.max_frames_per_video
                )
                self.logger.info(f"Using FPS-based keyframe extraction with {fps} fps")
            else:
                self.keyframe_extractor = KeyframeExtractor(
                    threshold=self.config.keyframe_threshold,
                    max_frames=self.config.max_frames_per_video
                )
                self.logger.info("Using histogram-based keyframe extraction")
        
        # Audio transcriber
        self.audio_transcriber = None
        if self.config.transcribe_audio:
            self.audio_transcriber = AudioTranscriber()
        
        # VLM descriptor
        self.vlm_descriptor = None
        if self.config.generate_descriptions:
            vlm_endpoint = self.app_config.get("vlm_endpoint_url")
            if vlm_endpoint:
                auto_start_vlm = self.app_config.get("auto_start_vlm_service", True)
                self.vlm_descriptor = VLMDescriptor(
                    vlm_endpoint=vlm_endpoint,
                    batch_size=self.config.vlm_batch_size,
                    timeout=10800,  # 3 hours
                    auto_start=auto_start_vlm
                )
    
    def _get_chunk_duration(self) -> float:
        """Get chunk duration from profile configuration"""
        if not self.schema_name:
            return 30.0  # Default
        
        profiles = self.app_config.get("video_processing_profiles", {})
        profile_config = profiles.get(self.schema_name, {})
        
        # Extract from processing type (e.g., "direct_video/chunks:30s")
        processing_type = profile_config.get("strategy", {}).get("processing", "")
        if ":" in processing_type:
            duration_str = processing_type.split(":")[1]
            if duration_str.endswith("s"):
                return float(duration_str[:-1])
        
        # Fallback to pipeline config
        pipeline_config = profile_config.get("pipeline_config", {})
        return pipeline_config.get("chunk_duration", 30.0)
    
    def _init_backend(self):
        """Initialize embedding generation backend"""
        if not self.config.generate_embeddings:
            self.embedding_generator = None
            return
        
        # Initialize embedding generator v2 directly
        try:
            self.embedding_generator = create_embedding_generator(
                config=self.app_config,
                schema_name=self.schema_name,
                logger=self.logger
            )
            self.logger.info(f"Initialized embedding generator v2 with backend: {self.config.search_backend}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding generator v2: {e}")
            self.embedding_generator = None
        
        # Set embedding generator for single vector processor if needed
        if self.single_vector_processor:
            self.single_vector_processor.embedding_generator = self.embedding_generator
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e:
            self.logger.warning(f"Failed to get video duration: {e}")
            return 0
    
    async def _check_cache_async(self, video_path: Path) -> Dict[str, Any]:
        """
        Check all cache entries concurrently
        Returns dict with cached results or None for each step
        """
        if not self.cache:
            return {}
        
        # Prepare cache check tasks
        cache_tasks = []
        cache_keys = []
        
        # Get profile configuration for cache parameters
        profiles = self.app_config.get("video_processing_profiles", {})
        profile_config = profiles.get(self.schema_name, {})
        pipeline_config = profile_config.get("pipeline_config", {})
        
        # Keyframes cache check
        if self.config.extract_keyframes:
            strategy = pipeline_config.get("keyframe_strategy", "similarity")
            cache_tasks.append(
                self.cache.get_keyframes(
                    str(video_path),
                    strategy=strategy,
                    threshold=pipeline_config.get("keyframe_threshold", self.config.keyframe_threshold),
                    fps=pipeline_config.get("keyframe_fps", 1.0),
                    max_frames=self.config.max_frames_per_video,
                    load_images=True
                )
            )
            cache_keys.append("keyframes")
        
        # Transcript cache check
        if self.config.transcribe_audio and self.audio_transcriber:
            cache_tasks.append(
                self.cache.get_transcript(
                    str(video_path),
                    model_size=getattr(self.audio_transcriber, 'model_size', "base"),
                    language=None
                )
            )
            cache_keys.append("transcript")
        
        # Descriptions cache check
        if self.config.generate_descriptions and self.vlm_descriptor:
            cache_tasks.append(
                self.cache.get_descriptions(
                    str(video_path),
                    model_name=getattr(self.vlm_descriptor, 'model_name', "Qwen/Qwen2-VL-2B-Instruct"),
                    batch_size=self.config.vlm_batch_size
                )
            )
            cache_keys.append("descriptions")
        
        # Execute all cache checks concurrently
        if cache_tasks:
            self.logger.info(f"Checking {len(cache_tasks)} cache entries concurrently for {video_path.name}")
            start_time = time.time()
            cache_results = await asyncio.gather(*cache_tasks, return_exceptions=True)
            elapsed = time.time() - start_time
            self.logger.info(f"Cache checks completed in {elapsed:.2f}s")
            
            # Build results dict
            results = {}
            for key, result in zip(cache_keys, cache_results):
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
    
    async def _save_to_cache_async(self, video_path: Path, step: str, data: Any, **kwargs) -> None:
        """
        Save data to cache asynchronously
        """
        if not self.cache or not data:
            return
        
        try:
            if step == "keyframes":
                # Load images for caching
                video_id = video_path.stem
                keyframes_dir = self.profile_output_dir / "keyframes" / video_id
                images = {}
                
                import cv2
                for kf in data.get("keyframes", []):
                    if "filename" in kf:
                        frame_path = keyframes_dir / kf["filename"]
                        if frame_path.exists():
                            image = cv2.imread(str(frame_path))
                            if image is not None:
                                images[str(kf["frame_id"])] = image
                
                await self.cache.set_keyframes(
                    str(video_path),
                    data,
                    images,
                    **kwargs
                )
            elif step == "transcript":
                await self.cache.set_transcript(
                    str(video_path),
                    data,
                    **kwargs
                )
            elif step == "descriptions":
                await self.cache.set_descriptions(
                    str(video_path),
                    data.get("descriptions", {}),
                    **kwargs
                )
            
            self.logger.info(f"Cached {step} for {video_path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cache {step}: {e}")
    
    # DEPRECATED: Sync methods not used by run_ingestion.py
    # Keeping for backward compatibility only
    def extract_keyframes(self, video_path: Path) -> Dict[str, Any]:
        """
        Extract keyframes with async cache operations using shared event loop
        """
        # Check cache using the shared event loop
        cached_result = None
        if self.cache:
            profiles = self.app_config.get("video_processing_profiles", {})
            profile_config = profiles.get(self.schema_name, {})
            pipeline_config = profile_config.get("pipeline_config", {})
            strategy = pipeline_config.get("keyframe_strategy", "similarity")
            
            # Use the shared event loop instead of asyncio.run()
            cached_result = self.loop.run_until_complete(
                self.cache.get_keyframes(
                    str(video_path),
                    strategy=strategy,
                    threshold=pipeline_config.get("keyframe_threshold", self.config.keyframe_threshold),
                    fps=pipeline_config.get("keyframe_fps", 1.0),
                    max_frames=self.config.max_frames_per_video,
                    load_images=True
                )
            )
        
        if cached_result:
            self.logger.info(f"Using cached keyframes for {video_path.name}")
            # Convert cache format to expected format
            if isinstance(cached_result, tuple):
                metadata, images = cached_result
            else:
                metadata = cached_result
                images = None
            
            # Save images to output directory for compatibility
            if images:
                video_id = video_path.stem
                keyframes_dir = self.profile_output_dir / "keyframes" / video_id
                keyframes_dir.mkdir(parents=True, exist_ok=True)
                
                import cv2
                for frame_id, image in images.items():
                    frame_path = keyframes_dir / f"frame_{int(frame_id):04d}.jpg"
                    cv2.imwrite(str(frame_path), image)
            
            return metadata
        
        # Extract keyframes
        result = self.keyframe_extractor.extract_keyframes(video_path, self.profile_output_dir)
        
        # Cache the result using shared event loop
        if self.cache and result and "keyframes" in result:
            profiles = self.app_config.get("video_processing_profiles", {})
            profile_config = profiles.get(self.schema_name, {})
            pipeline_config = profile_config.get("pipeline_config", {})
            strategy = pipeline_config.get("keyframe_strategy", "similarity")
            
            self.loop.run_until_complete(
                self._save_to_cache_async(
                    video_path,
                    "keyframes",
                    result,
                    strategy=strategy,
                    threshold=pipeline_config.get("keyframe_threshold", self.config.keyframe_threshold),
                    fps=pipeline_config.get("keyframe_fps", 1.0),
                    max_frames=self.config.max_frames_per_video
                )
            )
        
        return result
    
    def extract_chunks(self, video_path: Path) -> Dict[str, Any]:
        """Extract video chunks using VideoChunkExtractor"""
        if not self.video_chunk_extractor:
            return {"error": "VideoChunkExtractor not initialized"}
        
        # Process video chunks
        result = self.video_chunk_extractor.process_video(video_path, self.profile_output_dir)
        return result
    
    def transcribe_audio(self, video_path: Path) -> Dict[str, Any]:
        """
        Transcribe audio with async cache operations using shared event loop
        """
        # Check cache using the shared event loop
        cached_result = None
        if self.cache:
            cached_result = self.loop.run_until_complete(
                self.cache.get_transcript(
                    str(video_path),
                    model_size=getattr(self.audio_transcriber, 'model_size', "base"),
                    language=None
                )
            )
        
        if cached_result:
            self.logger.info(f"Using cached transcript for {video_path.name}")
            return cached_result
        
        # Transcribe audio
        result = self.audio_transcriber.transcribe_audio(video_path, self.profile_output_dir)
        
        # Cache the result using shared event loop
        if self.cache and result:
            self.loop.run_until_complete(
                self._save_to_cache_async(
                    video_path,
                    "transcript",
                    result,
                    model_size=getattr(self.audio_transcriber, 'model_size', "base"),
                    language=result.get("language")
                )
            )
        
        return result
    
    def generate_descriptions(self, keyframes_metadata: Dict[str, Any], video_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate VLM descriptions with async cache operations using shared event loop
        """
        if not self.vlm_descriptor:
            self.logger.warning("No VLM endpoint configured, skipping description generation")
            return {}
        
        # Check cache using the shared event loop
        cached_result = None
        if self.cache and video_path:
            cached_result = self.loop.run_until_complete(
                self.cache.get_descriptions(
                    str(video_path),
                    model_name=getattr(self.vlm_descriptor, 'model_name', "Qwen/Qwen2-VL-2B-Instruct"),
                    batch_size=self.config.vlm_batch_size
                )
            )
        
        if cached_result:
            self.logger.info(f"Using cached descriptions for {video_path.name}")
            return {"descriptions": cached_result}
        
        # Generate descriptions
        result = self.vlm_descriptor.generate_descriptions(keyframes_metadata, self.profile_output_dir)
        
        # Cache the result using shared event loop
        if self.cache and video_path and result and "descriptions" in result:
            self.loop.run_until_complete(
                self._save_to_cache_async(
                    video_path,
                    "descriptions",
                    result,
                    model_name=getattr(self.vlm_descriptor, 'model_name', "Qwen/Qwen2-VL-2B-Instruct"),
                    batch_size=self.config.vlm_batch_size
                )
            )
        
        return result
    
    def generate_embeddings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for search backend using EmbeddingGenerator v2"""
        if not self.embedding_generator:
            self.logger.error("Embedding generator not initialized")
            return {"error": "Embedding generator not initialized"}
            
        self.logger.info("Extracting data for embedding generation...")
        
        # Prepare video data for v2 generator
        video_data = {
            "video_id": results.get("video_id"),
            "video_path": results.get("video_path"),
            "duration": results.get("duration", 0),
            "output_dir": str(self.profile_output_dir)
        }
        self.logger.info(f"Results keys: {list(results.keys())}")
        
        # Add processing type from strategy
        if self.strategy:
            video_data["processing_type"] = self.strategy.processing_type
            video_data["storage_mode"] = self.strategy.storage_mode
            video_data["schema_name"] = self.strategy.schema_name
        
        # Check if we have chunks from VideoChunkExtractor
        if "chunks" in results.get("results", {}):
            # Add chunks data for direct_video/chunks processing
            chunks_data = results["results"]["chunks"]
            video_data["chunks"] = chunks_data["chunks"]
            video_data["chunk_duration"] = chunks_data.get("chunk_duration", 30.0)
            self.logger.info(f"Using video chunks data with {len(video_data['chunks'])} chunks")
            
            # Add transcript if available
            if "transcript" in results.get("results", {}):
                video_data["transcript"] = results["results"]["transcript"]
        
        # Check if we have single vector processing results
        elif "single_vector_processing" in results.get("results", {}):
            # Add single vector processing data
            processing_data = results["results"]["single_vector_processing"]
            # Use raw segments if available (for embedding generation), otherwise use serialized
            video_data["segments"] = results.get("_raw_segments", processing_data["segments"])
            video_data["processing_metadata"] = processing_data["metadata"]
            video_data["full_transcript"] = processing_data["full_transcript"]
            video_data["document_structure"] = processing_data["document_structure"]
            self.logger.info(f"Using single vector processing data with {len(video_data['segments'])} segments")
        
        # For frame-based processing, add frame information if available
        elif "keyframes" in results.get("results", {}) and video_data.get("processing_type") != "video_chunks":
            keyframes_data = results["results"]["keyframes"]
            if "keyframes" in keyframes_data:
                # Convert keyframe data to frames format expected by v2
                frames = []
                video_id = results.get("video_id")
                for kf in keyframes_data["keyframes"]:
                    # Construct frame path from filename
                    frame_path = None
                    if kf.get("path"):
                        frame_path = kf.get("path")
                    elif kf.get("filename") and video_id:
                        # Construct path from profile output dir
                        frame_path = str(self.profile_output_dir / "keyframes" / video_id / kf.get("filename"))
                    
                    frames.append({
                        "frame_id": kf.get("frame_id"),
                        "frame_path": frame_path,
                        "timestamp": kf.get("timestamp", 0.0)
                    })
                video_data["frames"] = frames
                
                # Add transcript if available
                if "transcript" in results.get("results", {}):
                    video_data["transcript"] = results["results"]["transcript"]
                
                # Add descriptions if available
                if "descriptions" in results.get("results", {}):
                    desc_data = results["results"]["descriptions"]
                    if "descriptions" in desc_data:
                        video_data["descriptions"] = desc_data["descriptions"]
        
        self.logger.info(f"Data extracted - Video ID: {video_data['video_id']}")
        self.logger.info(f"Output directory: {video_data['output_dir']}")
        
        # Generate embeddings using v2 generator
        # Call v2 generator which returns EmbeddingResult
        result = self.embedding_generator.generate_embeddings(video_data, self.profile_output_dir)
        
        # Convert EmbeddingResult to dict format expected by pipeline
        return {
            "video_id": result.video_id,
            "total_documents": result.total_documents,
            "documents_processed": result.documents_processed,
            "documents_fed": result.documents_fed,
            "processing_time": result.processing_time,
            "errors": result.errors,
            "metadata": result.metadata,
            "backend": self.config.search_backend
        }
    
    async def process_video_async(self, video_path: Path) -> Dict[str, Any]:
        """
        Async version of process_video with concurrent cache checks
        """
        video_id = video_path.stem
        
        self.logger.info(f"Starting async video processing: {video_id}")
        self.logger.info(f"Video path: {video_path}")
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        
        print(f"\n🎬 Processing video (async): {video_path.name}")
        print("=" * 60)
        
        # Convert PosixPath objects to strings for JSON serialization
        config_dict = self.config.__dict__.copy()
        config_dict["video_dir"] = str(config_dict["video_dir"])
        config_dict["output_dir"] = str(config_dict["output_dir"])
        
        results = {
            "video_id": video_id,
            "video_path": str(video_path),
            "duration": duration,
            "pipeline_config": config_dict,
            "results": {},
            "started_at": time.time(),
            "async_optimized": True
        }
        
        try:
            # Phase 1 optimization: Check all caches concurrently
            if self.cache:
                cache_start = time.time()
                cached_data = await self._check_cache_async(video_path)
                cache_time = time.time() - cache_start
                self.logger.info(f"Concurrent cache checks completed in {cache_time:.2f}s")
                results["cache_check_time"] = cache_time
            else:
                cached_data = {}
            
            # Process based on what's cached and what's needed
            
            # Check if we're using video chunk extraction
            if hasattr(self, 'video_chunk_extractor') and self.video_chunk_extractor:
                # Process video with chunk extraction
                step_start = time.time()
                self.logger.info("Using VideoChunkExtractor for direct_video/chunks processing")
                
                # Extract chunks (no cache for chunks yet)
                chunks_data = self.extract_chunks(video_path)
                results["results"]["chunks"] = chunks_data
                step_time = time.time() - step_start
                
                if "error" in chunks_data:
                    raise Exception(f"Chunk extraction failed: {chunks_data['error']}")
                else:
                    chunk_count = len(chunks_data.get("chunks", []))
                    self.logger.info(f"Extracted {chunk_count} chunks in {step_time:.2f}s")
                
                # Transcribe audio if enabled
                if self.config.transcribe_audio:
                    if cached_data.get("transcript"):
                        results["results"]["transcript"] = cached_data["transcript"]
                        self.logger.info("Using cached transcript")
                    else:
                        # Directly call the transcriber
                        transcript_data = self.audio_transcriber.transcribe_audio(video_path, self.profile_output_dir)
                        results["results"]["transcript"] = transcript_data
                        # Cache the result asynchronously
                        if self.cache and transcript_data:
                            await self._save_to_cache_async(
                                video_path,
                                "transcript",
                                transcript_data,
                                model_size=getattr(self.audio_transcriber, 'model_size', "base"),
                                language=transcript_data.get("language")
                            )
            
            # Check if we're using single vector processing
            elif self.single_vector_processor:
                # Process with single vector processor
                transcript_data = None
                if self.config.transcribe_audio:
                    if cached_data.get("transcript"):
                        transcript_data = cached_data["transcript"]
                        results["results"]["transcript"] = transcript_data
                        self.logger.info("Using cached transcript")
                    else:
                        # Directly call the transcriber
                        transcript_data = self.audio_transcriber.transcribe_audio(video_path, self.profile_output_dir)
                        results["results"]["transcript"] = transcript_data
                        # Cache the result asynchronously
                        if self.cache and transcript_data:
                            await self._save_to_cache_async(
                                video_path,
                                "transcript",
                                transcript_data,
                                model_size=getattr(self.audio_transcriber, 'model_size', "base"),
                                language=transcript_data.get("language")
                            )
                
                # Process video
                processed_data = self.single_vector_processor.process_video(
                    video_path=video_path,
                    transcript_data=transcript_data
                )
                
                # Convert for serialization
                processed_data_serializable = processed_data.copy()
                processed_data_serializable["segments"] = [
                    seg.to_dict() for seg in processed_data["segments"]
                ]
                results["results"]["single_vector_processing"] = processed_data_serializable
                results["_raw_segments"] = processed_data["segments"]
                
            else:
                # Original frame-based processing
                # Extract keyframes
                if self.config.extract_keyframes:
                    if cached_data.get("keyframes"):
                        cached_keyframes = cached_data["keyframes"]
                        # Handle tuple format from cache (metadata, images)
                        if isinstance(cached_keyframes, tuple):
                            keyframes_data, images = cached_keyframes
                        else:
                            keyframes_data = cached_keyframes
                        results["results"]["keyframes"] = keyframes_data
                        self.logger.info(f"Using cached keyframes: {len(keyframes_data.get('keyframes', []))} frames")
                    else:
                        # Directly call the appropriate extractor
                        if self.keyframe_extractor:
                            keyframes_data = self.keyframe_extractor.extract_keyframes(video_path, self.profile_output_dir)
                        else:
                            keyframes_data = {}
                        results["results"]["keyframes"] = keyframes_data
                        self.logger.info(f"Extracted {len(keyframes_data.get('keyframes', []))} keyframes")
                        # Cache the result asynchronously
                        if self.cache and keyframes_data:
                            strategy = self.config.keyframe_strategy if hasattr(self.config, 'keyframe_strategy') else "similarity"
                            await self._save_to_cache_async(
                                video_path,
                                "keyframes",
                                keyframes_data,
                                strategy=strategy,
                                threshold=self.config.keyframe_threshold if strategy == "similarity" else None,
                                fps=self.config.keyframe_fps if strategy == "fps" else None,
                                max_frames=self.config.max_frames_per_video
                            )
                
                # Transcribe audio
                if self.config.transcribe_audio:
                    if cached_data.get("transcript"):
                        transcript_data = cached_data["transcript"]
                        results["results"]["transcript"] = transcript_data
                        self.logger.info(f"Using cached transcript: {len(transcript_data.get('segments', []))} segments")
                    else:
                        # Directly call the transcriber without cache operations
                        transcript_data = self.audio_transcriber.transcribe_audio(video_path, self.profile_output_dir)
                        results["results"]["transcript"] = transcript_data
                        self.logger.info(f"Transcribed {len(transcript_data.get('segments', []))} segments")
                        # Cache the result asynchronously
                        if self.cache and transcript_data:
                            await self._save_to_cache_async(
                                video_path,
                                "transcript",
                                transcript_data,
                                model_size=getattr(self.audio_transcriber, 'model_size', "base"),
                                language=transcript_data.get("language")
                            )
                
                # Generate descriptions
                if self.config.generate_descriptions and self.config.extract_keyframes:
                    if cached_data.get("descriptions"):
                        descriptions_data = {"descriptions": cached_data["descriptions"]}
                        results["results"]["descriptions"] = descriptions_data
                        self.logger.info(f"Using cached descriptions: {len(descriptions_data.get('descriptions', {}))} items")
                    else:
                        keyframes_data = results["results"].get("keyframes", {})
                        # Directly call the descriptor without cache operations
                        if self.vlm_descriptor:
                            descriptions_data = self.vlm_descriptor.generate_descriptions(
                                keyframes_data, 
                                self.profile_output_dir
                            )
                        else:
                            descriptions_data = {}
                        results["results"]["descriptions"] = descriptions_data
                        self.logger.info(f"Generated {len(descriptions_data.get('descriptions', {}))} descriptions")
                        # Cache the result asynchronously
                        if self.cache and video_path and descriptions_data:
                            await self._save_to_cache_async(
                                video_path,
                                "descriptions",
                                descriptions_data.get("descriptions", {}),
                                model_name=getattr(self.vlm_descriptor, 'model_name', "Qwen/Qwen2-VL-2B-Instruct"),
                                batch_size=self.config.vlm_batch_size
                            )
            
            # Generate embeddings (same as before)
            if self.config.generate_embeddings:
                step_start = time.time()
                self.logger.info(f"Starting {self.config.search_backend} embedding generation")
                embeddings_data = self.generate_embeddings(results)
                results["results"]["embeddings"] = embeddings_data
                step_time = time.time() - step_start
                
                if "error" in embeddings_data:
                    raise Exception(f"Embedding generation failed: {embeddings_data['error']}")
                else:
                    embed_count = embeddings_data.get("total_documents", 0)
                    self.logger.info(f"Generated {embed_count} embeddings in {step_time:.2f}s")
            
            results["completed_at"] = time.time()
            results["total_processing_time"] = results["completed_at"] - results["started_at"]
            results["status"] = "completed"
            
            self.logger.info(f"Async video processing completed in {results['total_processing_time']:.2f}s")
            print(f"\n✅ Video processing completed in {results['total_processing_time']:.1f}s")
            
        except Exception as e:
            results["error"] = str(e)
            results["status"] = "failed"
            results["completed_at"] = time.time()
            results["total_processing_time"] = results["completed_at"] - results["started_at"]
            
            self.logger.error(f"Video processing failed: {e}", exc_info=True)
            print(f"\n❌ Video processing failed: {e}")
        
        return results
    
    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """
        DEPRECATED: Sync wrapper for backward compatibility
        Use process_video_async directly in new code
        """
        # Run the async version using the shared event loop
        return self.loop.run_until_complete(self.process_video_async(video_path))
    
    async def process_videos_concurrent(self, video_files: List[Path], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Process multiple videos concurrently with resource control
        
        Args:
            video_files: List of video paths to process
            max_concurrent: Maximum number of videos to process simultaneously
            
        Returns:
            List of results for each video
        """
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(video_path: Path, index: int, total: int):
            """Process a video with concurrency limit and progress tracking"""
            async with semaphore:
                try:
                    self.logger.info(f"[{index}/{total}] Starting concurrent processing: {video_path.name}")
                    print(f"\n🎯 [{index}/{total}] Processing: {video_path.name}")
                    
                    result = await self.process_video_async(video_path)
                    
                    if result["status"] == "completed":
                        self.logger.info(f"[{index}/{total}] Completed: {video_path.name} in {result['total_processing_time']:.1f}s")
                        print(f"✅ [{index}/{total}] Completed: {video_path.name} ({result['total_processing_time']:.1f}s)")
                    else:
                        self.logger.error(f"[{index}/{total}] Failed: {video_path.name} - {result.get('error')}")
                        print(f"❌ [{index}/{total}] Failed: {video_path.name}")
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"[{index}/{total}] Exception processing {video_path.name}: {e}")
                    return {
                        "video_path": str(video_path),
                        "error": str(e),
                        "status": "failed"
                    }
        
        # Create tasks for all videos
        tasks = [
            process_with_limit(video, i+1, len(video_files))
            for i, video in enumerate(video_files)
        ]
        
        # Process all videos concurrently
        self.logger.info(f"Starting concurrent processing of {len(video_files)} videos (max {max_concurrent} concurrent)")
        print(f"\n🚀 Processing {len(video_files)} videos concurrently (max {max_concurrent} at once)")
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "completed")
        failed = len(results) - successful
        
        self.logger.info(f"Concurrent processing completed: {successful}/{len(video_files)} successful in {total_time:.1f}s")
        print(f"\n🏁 Concurrent processing completed in {total_time:.1f}s")
        print(f"   ✅ Successful: {successful}/{len(video_files)}")
        print(f"   ❌ Failed: {failed}/{len(video_files)}")
        print(f"   ⚡ Average time: {total_time/len(video_files):.1f}s per video")
        
        return results
    
    def get_video_files(self, video_dir: Path) -> List[Path]:
        """Get list of video files from directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
        return sorted(video_files)
    
    def process_directory(self, video_dir: Optional[Path] = None, max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Process all videos in a directory with concurrent processing
        
        Args:
            video_dir: Directory containing videos
            max_concurrent: Maximum number of videos to process simultaneously
        """
        video_dir = video_dir or self.config.video_dir
        video_files = self.get_video_files(video_dir)
        
        if not video_files:
            self.logger.error(f"No video files found in {video_dir}")
            print(f"❌ No video files found in {video_dir}")
            return {"error": "No video files found"}
        
        self.logger.info(f"Starting concurrent batch processing: {len(video_files)} videos from {video_dir}")
        
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
            "started_at": time.time()
        }
        
        # Process videos concurrently
        processed_results = self.loop.run_until_complete(
            self.process_videos_concurrent(video_files, max_concurrent)
        )
        
        # Separate successful and failed results
        results["processed_videos"] = [r for r in processed_results if r.get("status") == "completed"]
        results["failed_videos"] = [r for r in processed_results if r.get("status") != "completed"]
        
        results["completed_at"] = time.time()
        results["total_processing_time"] = results["completed_at"] - results["started_at"]
        
        # Save summary (clean up any non-serializable data first)
        summary_file = self.profile_output_dir / "pipeline_summary.json"
        
        # Remove any raw segments from processed videos before saving
        import copy
        results_to_save = copy.deepcopy(results)
        for video_result in results_to_save.get("processed_videos", []):
            if "_raw_segments" in video_result:
                del video_result["_raw_segments"]
        
        with open(summary_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Log final summary
        self.logger.info("Concurrent batch processing completed!")
        self.logger.info(f"Summary - Total: {len(video_files)}, Processed: {len(results['processed_videos'])}, Failed: {len(results['failed_videos'])}")
        self.logger.info(f"Total time: {results['total_processing_time']:.2f} seconds ({results['total_processing_time']/60:.1f} minutes)")
        self.logger.info(f"Average time per video: {results['total_processing_time'] / len(video_files):.2f} seconds")
        self.logger.info(f"Summary saved to: {summary_file}")
        
        print(f"\n🎉 Pipeline completed!")
        print(f"✅ Processed: {len(results['processed_videos'])}/{len(video_files)} videos")
        print(f"❌ Failed: {len(results['failed_videos'])} videos")
        print(f"⏱️ Total time: {results['total_processing_time']/60:.1f} minutes")
        print(f"⚡ Throughput: {len(video_files)/results['total_processing_time']*60:.1f} videos/minute")
        print(f"📄 Summary saved: {summary_file}")
        
        # Stop VLM service if it was auto-started
        if self.vlm_descriptor and hasattr(self.vlm_descriptor, 'stop_service'):
            self.vlm_descriptor.stop_service()
            self.logger.info("VLM service stopped")
        
        return results
    
    def __del__(self):
        """Cleanup the event loop"""
        if hasattr(self, 'loop') and self.loop:
            try:
                self.loop.close()
            except:
                pass


# Keep AsyncVideoIngestionPipeline as an alias for backward compatibility
AsyncVideoIngestionPipeline = VideoIngestionPipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, help="Output directory")
    parser.add_argument("--backend", choices=["byaldi", "vespa"], default="vespa", help="Search backend")
    parser.add_argument("--profile", type=str, help="Video processing profile")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per video")
    
    # Concurrent processing
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent videos (default: 3)")
    
    # Pipeline step toggles
    parser.add_argument("--skip-keyframes", action="store_true", help="Skip keyframe extraction")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio transcription")
    parser.add_argument("--skip-descriptions", action="store_true", help="Skip VLM descriptions")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    
    args = parser.parse_args()
    
    # Set profile if provided
    if args.profile:
        import os
        os.environ["VIDEO_PROFILE"] = args.profile
    
    # Create pipeline config
    config = PipelineConfig.from_config()
    
    # Override with command line arguments
    if args.video_dir:
        config.video_dir = args.video_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.backend:
        config.search_backend = args.backend
    if args.max_frames:
        config.max_frames_per_video = args.max_frames
    
    # Apply skip flags
    if args.skip_keyframes:
        config.extract_keyframes = False
    if args.skip_audio:
        config.transcribe_audio = False
    if args.skip_descriptions:
        config.generate_descriptions = False
    if args.skip_embeddings:
        config.generate_embeddings = False
    
    # Run pipeline with concurrent processing
    print("🚀 Starting Video Processing Pipeline")
    pipeline = VideoIngestionPipeline(config)
    
    # Use max_concurrent from args
    max_concurrent = args.max_concurrent
    results = pipeline.process_directory(max_concurrent=max_concurrent)