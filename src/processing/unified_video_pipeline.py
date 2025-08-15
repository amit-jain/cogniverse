#!/usr/bin/env python3
"""
Unified Video Processing Pipeline

Consolidates all video processing steps into a single configurable pipeline:
1. Keyframe extraction
2. Audio transcription  
3. VLM description generation
4. Vector embeddings
5. Index creation

Based on the configurable pipeline design from CLAUDE.md.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.common.config import get_config
from src.processing.pipeline_steps import (
    KeyframeExtractor,
    AudioTranscriber,
    VLMDescriptor
)
from src.processing.pipeline_steps.keyframe_extractor_fps import FPSKeyframeExtractor
from src.processing.pipeline_steps.embedding_generator import create_embedding_generator
from src.cache import CacheManager, CacheConfig
from src.cache.pipeline_cache import PipelineArtifactCache
from src.processing.strategy import StrategyConfig


class PipelineStep(Enum):
    """Pipeline steps that can be enabled/disabled"""
    EXTRACT_KEYFRAMES = "extract_keyframes"
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
        from src.utils.output_manager import get_output_manager
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
    """Unified video processing pipeline - single entry point for all video processing"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.from_config()
        self.app_config = get_config()
        # Check environment variable first, then config
        import os
        self.active_profile = os.environ.get("VIDEO_PROFILE") or self.app_config.get("active_video_profile", "frame_based_colpali")
        self.strategy_config = StrategyConfig()
        self.setup_directories()
        self.logger = self._setup_logging()
        self._initialize_cache()
        self._initialize_pipeline_steps()
        
    def setup_directories(self):
        """Create necessary directories with profile-specific organization"""
        # Use the active profile we already set
        if self.active_profile:
            # Use profile-specific directory
            self.profile_output_dir = self.config.output_dir / f"profile_{self.active_profile}"
        else:
            # Fallback to default
            self.profile_output_dir = self.config.output_dir / "default"
        
        directories = [
            self.profile_output_dir / "keyframes",
            self.profile_output_dir / "metadata", 
            self.profile_output_dir / "descriptions",
            self.profile_output_dir / "transcripts",
            self.profile_output_dir / "embeddings",
            Path("logs")  # Add logs directory
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for the pipeline"""
        logger = logging.getLogger("VideoIngestionPipeline")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create timestamp for log file with profile name
        from src.utils.output_manager import get_output_manager
        output_manager = get_output_manager()
        timestamp = int(time.time())
        active_profile = self.app_config.get("active_video_profile", "default")
        log_file = output_manager.get_logs_dir() / f"video_processing_{active_profile}_{timestamp}.log"
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler with simpler formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log initialization
        logger.info(f"VideoIngestionPipeline initialized - logging to: {log_file}")
        logger.info(f"Backend: {self.config.search_backend}")
        logger.info(f"Output directory: {self.profile_output_dir}")
        logger.info(f"Pipeline config: {self.config}")
        
        return logger
    
    def _initialize_cache(self):
        """Initialize cache from configuration"""
        cache_config_dict = self.app_config.get("pipeline_cache", {})
        
        if not cache_config_dict.get("enabled", False):
            self.logger.info("Pipeline cache is disabled")
            self.cache = None
            return
        
        # Create cache configuration
        cache_config = CacheConfig(
            backends=cache_config_dict.get("backends", []),
            default_ttl=cache_config_dict.get("default_ttl", 0),
            enable_compression=cache_config_dict.get("enable_compression", True),
            serialization_format=cache_config_dict.get("serialization_format", "pickle")
        )
        
        # Initialize cache
        self.cache_manager = CacheManager(cache_config)
        self.cache = PipelineArtifactCache(
            self.cache_manager,
            ttl=cache_config_dict.get("default_ttl", 0),
            profile=self.active_profile  # Include profile for namespacing
        )
        self.logger.info(f"Initialized pipeline cache for profile: {self.active_profile}")
    
    def _initialize_pipeline_steps(self):
        """Initialize all pipeline step handlers"""
        # Use StrategyConfig to get complete strategy
        if self.active_profile:
            try:
                self.strategy = self.strategy_config.get_strategy(self.active_profile)
                self.logger.info(f"Resolved strategy: {self.strategy}")
            except Exception as e:
                self.logger.error(f"Failed to resolve strategy: {e}")
                self.strategy = None
        else:
            self.strategy = None
        
        # Initialize processors based on strategy
        if self.strategy and self.strategy.processing_type == "single_vector":
            # Initialize SingleVectorVideoProcessor
            from src.processing.pipeline_steps.single_vector_processor import SingleVectorVideoProcessor
            
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
        else:
            self.single_vector_processor = None
            # Initialize keyframe extractor only if needed
            if self.config.extract_keyframes:
                # Check if FPS-based extraction is configured
                extraction_method = self.app_config.get("pipeline_config.keyframe_extraction_method", "histogram")
                if extraction_method == "fps":
                    # FPS-based extraction
                    keyframe_fps = self.app_config.get("pipeline_config.keyframe_fps", 1.0)
                    self.keyframe_extractor = FPSKeyframeExtractor(
                        fps=keyframe_fps,
                        max_frames=self.config.max_frames_per_video
                    )
                    self.logger.info(f"Using FPS-based keyframe extraction at {keyframe_fps} FPS")
                else:
                    self.keyframe_extractor = KeyframeExtractor(
                        threshold=self.config.keyframe_threshold,
                        max_frames=self.config.max_frames_per_video
                    )
                    self.logger.info("Using histogram-based keyframe extraction")
            else:
                self.keyframe_extractor = None
        
        # Initialize audio transcriber only if needed
        if self.config.transcribe_audio:
            self.audio_transcriber = AudioTranscriber(
                model_size="base",
                device=None  # Auto-detect
            )
        else:
            self.audio_transcriber = None
        
        # Initialize VLM descriptor only if needed
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
            else:
                self.vlm_descriptor = None
        else:
            self.vlm_descriptor = None
            
        # Initialize embedding generator v2
        if self.config.generate_embeddings:
            try:
                # Get the config data (app_config is already a dict)
                generator_config = self.app_config.copy()
                generator_config["embedding_backend"] = self.config.search_backend
                generator_config["active_profile"] = self.active_profile
                
                # Create embedding generator v2
                self.embedding_generator = create_embedding_generator(
                    config=generator_config,
                    logger=self.logger
                )
                self.logger.info(f"Initialized embedding generator v2 with backend: {self.config.search_backend}")
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding generator v2: {e}")
                self.embedding_generator = None
        else:
            self.embedding_generator = None
            
    def get_video_files(self, video_dir: Optional[Path] = None) -> List[Path]:
        """Get all video files from directory"""
        video_dir = video_dir or self.config.video_dir
        extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
        
        video_files = []
        for ext in extensions:
            video_files.extend(video_dir.glob(ext))
            
        return sorted(video_files)
    
    def extract_keyframes(self, video_path: Path) -> Dict[str, Any]:
        """Extract keyframes from video using KeyframeExtractor"""
        # Track if we used cache
        used_cache = False
        
        # Check cache first if enabled
        if self.cache:
            # Get profile configuration to determine strategy
            profiles = self.app_config.get("video_processing_profiles", {})
            profile_config = profiles.get(self.active_profile, {})
            pipeline_config = profile_config.get("pipeline_config", {})
            
            strategy = pipeline_config.get("keyframe_strategy", "similarity")
            
            # Try to get from cache
            cached_result = asyncio.run(self.cache.get_keyframes(
                str(video_path),
                strategy=strategy,
                threshold=pipeline_config.get("keyframe_threshold", self.config.keyframe_threshold),
                fps=pipeline_config.get("keyframe_fps", 1.0),
                max_frames=self.config.max_frames_per_video,
                load_images=True  # Load images for processing
            ))
            
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
        
        # No cache or cache miss - extract keyframes
        result = self.keyframe_extractor.extract_keyframes(video_path, self.profile_output_dir)
        
        # Cache the result if cache is enabled
        if self.cache and result and "keyframes" in result:
            # Load images for caching
            video_id = video_path.stem
            keyframes_dir = self.profile_output_dir / "keyframes" / video_id
            images = {}
            
            import cv2
            for kf in result.get("keyframes", []):
                if "filename" in kf:
                    frame_path = keyframes_dir / kf["filename"]
                    if frame_path.exists():
                        image = cv2.imread(str(frame_path))
                        if image is not None:
                            images[str(kf["frame_id"])] = image
            
            # Get strategy parameters
            profiles = self.app_config.get("video_processing_profiles", {})
            profile_config = profiles.get(self.active_profile, {})
            pipeline_config = profile_config.get("pipeline_config", {})
            strategy = pipeline_config.get("keyframe_strategy", "similarity")
            
            asyncio.run(self.cache.set_keyframes(
                str(video_path),
                result,
                images,
                strategy=strategy,
                threshold=pipeline_config.get("keyframe_threshold", self.config.keyframe_threshold),
                fps=pipeline_config.get("keyframe_fps", 1.0),
                max_frames=self.config.max_frames_per_video
            ))
            self.logger.info(f"Cached keyframes for {video_path.name}")
        
        return result
    
    def transcribe_audio(self, video_path: Path) -> Dict[str, Any]:
        """Extract and transcribe audio from video using AudioTranscriber"""
        # Check cache first if enabled
        if self.cache:
            cached_result = asyncio.run(self.cache.get_transcript(
                str(video_path),
                model_size=self.audio_transcriber.model_size if hasattr(self.audio_transcriber, 'model_size') else "base",
                language=None  # Auto-detect
            ))
            
            if cached_result:
                self.logger.info(f"Using cached transcript for {video_path.name}")
                return cached_result
        
        # No cache or cache miss - transcribe audio
        result = self.audio_transcriber.transcribe_audio(video_path, self.profile_output_dir)
        
        # Cache the result if cache is enabled
        if self.cache and result:
            asyncio.run(self.cache.set_transcript(
                str(video_path),
                result,
                model_size=self.audio_transcriber.model_size if hasattr(self.audio_transcriber, 'model_size') else "base",
                language=result.get("language")
            ))
            self.logger.info(f"Cached transcript for {video_path.name}")
        
        return result
    
    def generate_descriptions(self, keyframes_metadata: Dict[str, Any], video_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate VLM descriptions for keyframes using VLMDescriptor"""
        if not self.vlm_descriptor:
            print("  ⚠️ No VLM endpoint configured, skipping description generation")
            return {}
        
        # Check cache first if enabled and video_path provided
        if self.cache and video_path:
            cached_result = asyncio.run(self.cache.get_descriptions(
                str(video_path),
                model_name=self.vlm_descriptor.model_name if hasattr(self.vlm_descriptor, 'model_name') else "Qwen/Qwen2-VL-2B-Instruct",
                batch_size=self.config.vlm_batch_size
            ))
            
            if cached_result:
                self.logger.info(f"Using cached descriptions for {video_path.name}")
                return {"descriptions": cached_result}
        
        # No cache or cache miss - generate descriptions
        result = self.vlm_descriptor.generate_descriptions(keyframes_metadata, self.profile_output_dir)
        
        # Cache the result if cache is enabled
        if self.cache and video_path and result and "descriptions" in result:
            asyncio.run(self.cache.set_descriptions(
                str(video_path),
                result["descriptions"],
                model_name=self.vlm_descriptor.model_name if hasattr(self.vlm_descriptor, 'model_name') else "Qwen/Qwen2-VL-2B-Instruct",
                batch_size=self.config.vlm_batch_size
            ))
            self.logger.info(f"Cached descriptions for {video_path.name}")
        
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
        
        # Add processing type from strategy
        if self.strategy:
            video_data["processing_type"] = self.strategy.processing_type
            video_data["storage_mode"] = self.strategy.storage_mode
            video_data["schema_name"] = self.strategy.schema_name
        
        # Check if we have single vector processing results
        if "single_vector_processing" in results.get("results", {}):
            # Add single vector processing data
            processing_data = results["results"]["single_vector_processing"]
            # Use raw segments if available (for embedding generation), otherwise use serialized
            video_data["segments"] = results.get("_raw_segments", processing_data["segments"])
            video_data["processing_metadata"] = processing_data["metadata"]
            video_data["full_transcript"] = processing_data["full_transcript"]
            video_data["document_structure"] = processing_data["document_structure"]
            self.logger.info(f"Using single vector processing data with {len(video_data['segments'])} segments")
        
        # For frame-based processing, add frame information if available
        elif "keyframes" in results.get("results", {}):
            keyframes_data = results["results"]["keyframes"]
            if "keyframes" in keyframes_data:
                # Convert keyframe data to frames format expected by v2
                frames = []
                for kf in keyframes_data["keyframes"]:
                    frames.append({
                        "frame_id": kf.get("frame_id"),
                        "frame_path": kf.get("path"),
                        "timestamp": kf.get("timestamp", 0.0)
                    })
                video_data["frames"] = frames
        
        self.logger.info(f"Data extracted - Video ID: {video_data['video_id']}")
        self.logger.info(f"Output directory: {video_data['output_dir']}")
        
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
    
    
    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """Process a single video through the complete pipeline"""
        video_id = video_path.stem
        
        self.logger.info(f"Starting video processing: {video_id}")
        self.logger.info(f"Video path: {video_path}")
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        
        print(f"\n🎬 Processing video: {video_path.name}")
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
            "started_at": time.time()
        }
        
        try:
            # Check if we're using single vector processing
            if self.single_vector_processor:
                # Process video with single vector processor
                step_start = time.time()
                self.logger.info("Using SingleVectorVideoProcessor for video processing")
                
                # Transcribe audio first if enabled
                transcript_data = None
                if self.config.transcribe_audio:
                    self.logger.info("Step 1: Starting audio transcription")
                    transcript_data = self.transcribe_audio(video_path)
                    results["results"]["transcript"] = transcript_data
                
                # Process video with single vector processor
                self.logger.info("Step 2: Processing video with SingleVectorVideoProcessor")
                processed_data = self.single_vector_processor.process_video(
                    video_path=video_path,
                    transcript_data=transcript_data
                )
                
                # Convert VideoSegment objects to dicts for JSON serialization
                processed_data_serializable = processed_data.copy()
                processed_data_serializable["segments"] = [
                    seg.to_dict() for seg in processed_data["segments"]
                ]
                results["results"]["single_vector_processing"] = processed_data_serializable
                
                # Store segments info for embedding generation (keep raw objects for processing)
                # Note: These will be passed to embedding generator but not saved to JSON
                results["_raw_segments"] = processed_data["segments"]
                
                step_time = time.time() - step_start
                self.logger.info(f"Single vector processing completed in {step_time:.2f}s - {len(processed_data['segments'])} segments")
                
            else:
                # Original frame-based processing
                # Step 1: Extract keyframes
                if self.config.extract_keyframes:
                    step_start = time.time()
                    self.logger.info("Step 1: Starting keyframe extraction")
                    keyframes_data = self.extract_keyframes(video_path)
                    results["results"]["keyframes"] = keyframes_data
                    step_time = time.time() - step_start
                    
                    if "error" in keyframes_data:
                        self.logger.error(f"Keyframe extraction failed: {keyframes_data['error']}")
                        raise Exception(f"Keyframe extraction failed: {keyframes_data['error']}")
                    else:
                        keyframe_count = len(keyframes_data.get("keyframes", []))
                        self.logger.info(f"Step 1 completed in {step_time:.2f}s - extracted {keyframe_count} keyframes")
                else:
                    self.logger.info("Step 1: Skipping keyframe extraction (disabled)")
                    print("⏭️ Skipping keyframe extraction (disabled)")
                
                # Step 2: Transcribe audio (for non-single vector processing)
                if self.config.transcribe_audio:
                    step_start = time.time()
                    self.logger.info("Step 2: Starting audio transcription")
                    transcript_data = self.transcribe_audio(video_path)
                    results["results"]["transcript"] = transcript_data
                    step_time = time.time() - step_start
                    
                    if "error" in transcript_data:
                        self.logger.error(f"Audio transcription failed: {transcript_data['error']}")
                        raise Exception(f"Audio transcription failed: {transcript_data['error']}")
                    else:
                        segment_count = len(transcript_data.get("segments", []))
                        self.logger.info(f"Step 2 completed in {step_time:.2f}s - transcribed {segment_count} segments")
                else:
                    self.logger.info("Step 2: Skipping audio transcription (disabled)")
                    print("⏭️ Skipping audio transcription (disabled)")
                
            # Step 3: Generate descriptions
            if self.config.generate_descriptions and self.config.extract_keyframes:
                step_start = time.time()
                self.logger.info("Step 3: Starting description generation")
                keyframes_data = results["results"].get("keyframes", {})
                descriptions_data = self.generate_descriptions(keyframes_data, video_path)
                results["results"]["descriptions"] = descriptions_data
                step_time = time.time() - step_start
                
                if "error" in descriptions_data:
                    self.logger.error(f"Description generation failed: {descriptions_data['error']}")
                    raise Exception(f"Description generation failed: {descriptions_data['error']}")
                else:
                    desc_count = len(descriptions_data.get("descriptions", {}))
                    self.logger.info(f"Step 3 completed in {step_time:.2f}s - generated {desc_count} descriptions")
            elif not self.config.generate_descriptions:
                self.logger.info("Step 3: Skipping description generation (disabled)")
                print("⏭️ Skipping description generation (disabled)")
            else:
                self.logger.info("Step 3: Skipping description generation (no keyframes available)")
                print("⏭️ Skipping description generation (no keyframes available)")
                
            # Step 4: Generate embeddings
            if self.config.generate_embeddings:
                step_start = time.time()
                self.logger.info(f"Step 4: Starting {self.config.search_backend} embedding generation")
                embeddings_data = self.generate_embeddings(results)
                results["results"]["embeddings"] = embeddings_data
                step_time = time.time() - step_start
                
                if "error" in embeddings_data:
                    self.logger.error(f"Embedding generation failed: {embeddings_data['error']}")
                    raise Exception(f"Embedding generation failed: {embeddings_data['error']}")
                else:
                    embed_count = embeddings_data.get("total_documents", 0)
                    self.logger.info(f"Step 4 completed in {step_time:.2f}s - generated {embed_count} embeddings")
            else:
                self.logger.info("Step 4: Skipping embedding generation (disabled)")
                print("⏭️ Skipping embedding generation (disabled)")
                
            results["completed_at"] = time.time()
            results["total_processing_time"] = results["completed_at"] - results["started_at"]
            results["status"] = "completed"
            
            self.logger.info(f"Video processing completed successfully in {results['total_processing_time']:.2f}s")
            
            # Log summary
            summary_parts = []
            if "keyframes" in results["results"]:
                keyframe_count = len(results["results"]["keyframes"].get("keyframes", []))
                summary_parts.append(f"keyframes: {keyframe_count}")
            if "transcript" in results["results"]:
                segment_count = len(results["results"]["transcript"].get("segments", []))
                summary_parts.append(f"segments: {segment_count}")
            if "descriptions" in results["results"]:
                desc_count = len(results["results"]["descriptions"].get("descriptions", {}))
                summary_parts.append(f"descriptions: {desc_count}")
            if "embeddings" in results["results"]:
                embed_count = results["results"]["embeddings"].get("total_documents", 0)
                summary_parts.append(f"embeddings: {embed_count}")
            
            self.logger.info(f"Processing summary: {', '.join(summary_parts)}")
            
            print(f"\n✅ Video processing completed in {results['total_processing_time']:.1f}s")
            
        except Exception as e:
            results["error"] = str(e)
            results["status"] = "failed"
            results["completed_at"] = time.time()
            results["total_processing_time"] = results["completed_at"] - results["started_at"]
            
            self.logger.error(f"Video processing failed: {e}", exc_info=True)
            print(f"\n❌ Video processing failed: {e}")
            
        return results
    
    def process_directory(self, video_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Process all videos in a directory"""
        video_dir = video_dir or self.config.video_dir
        video_files = self.get_video_files(video_dir)
        
        if not video_files:
            self.logger.error(f"No video files found in {video_dir}")
            print(f"❌ No video files found in {video_dir}")
            return {"error": "No video files found"}
        
        # Process all videos found
        
        self.logger.info(f"Starting batch processing: {len(video_files)} videos from {video_dir}")
        
        print(f"🎬 Found {len(video_files)} videos to process")
        print(f"📁 Output directory: {self.profile_output_dir}")
        print(f"⚙️ Pipeline config: {self.config}")
        
        # Convert PosixPath objects to strings for JSON serialization
        config_dict = self.config.__dict__.copy()
        config_dict["video_dir"] = str(config_dict["video_dir"])
        config_dict["output_dir"] = str(config_dict["output_dir"])
        
        active_profile = self.app_config.get("active_video_profile", "default")
        results = {
            "profile": active_profile,
            "pipeline_config": config_dict,
            "output_directory": str(self.profile_output_dir),
            "total_videos": len(video_files),
            "processed_videos": [],
            "failed_videos": [], 
            "started_at": time.time()
        }
        
        for i, video_path in enumerate(video_files, 1):
            self.logger.info(f"Processing video {i}/{len(video_files)}: {video_path.name}")
            print(f"\n🎯 Processing video {i}/{len(video_files)}")
            
            try:
                video_result = self.process_video(video_path)
                if video_result["status"] == "completed":
                    results["processed_videos"].append(video_result)
                    self.logger.info(f"Video {video_path.name} completed successfully")
                else:
                    results["failed_videos"].append(video_result)
                    self.logger.error(f"Video {video_path.name} failed: {video_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {video_path.name}: {e}", exc_info=True)
                print(f"❌ Failed to process {video_path.name}: {e}")
                results["failed_videos"].append({
                    "video_path": str(video_path),
                    "error": str(e)
                })
        
        results["completed_at"] = time.time()
        results["total_processing_time"] = results["completed_at"] - results["started_at"]
        
        # Save summary (clean up any non-serializable data first)
        summary_file = self.profile_output_dir / "pipeline_summary.json"
        
        # Remove any raw segments from processed videos before saving
        results_to_save = results.copy()
        for video_result in results_to_save.get("processed_videos", []):
            if "_raw_segments" in video_result:
                del video_result["_raw_segments"]
        
        with open(summary_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Log final summary
        self.logger.info("Batch processing completed!")
        self.logger.info(f"Summary - Total: {len(video_files)}, Processed: {len(results['processed_videos'])}, Failed: {len(results['failed_videos'])}")
        self.logger.info(f"Total time: {results['total_processing_time']:.2f} seconds ({results['total_processing_time']/60:.1f} minutes)")
        self.logger.info(f"Average time per video: {results['total_processing_time'] / len(video_files):.2f} seconds")
        self.logger.info(f"Summary saved to: {summary_file}")
        
        print(f"\n🎉 Pipeline completed!")
        print(f"✅ Processed: {len(results['processed_videos'])}/{len(video_files)} videos")
        print(f"❌ Failed: {len(results['failed_videos'])} videos")
        print(f"⏱️ Total time: {results['total_processing_time']/60:.1f} minutes")
        print(f"📄 Summary saved: {summary_file}")
        
        # Stop VLM service if it was auto-started
        if self.vlm_descriptor and hasattr(self.vlm_descriptor, 'stop_service'):
            self.vlm_descriptor.stop_service()
            self.logger.info("VLM service stopped")
        
        return results
    
    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using ffprobe"""
        try:
            import subprocess
            import json
            
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get('format', {}).get('duration', 0))
            return 0
        except Exception as e:
            self.logger.error(f"Failed to get video duration: {e}")
            return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Video Processing Pipeline")
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, help="Output directory")
    parser.add_argument("--config", type=Path, help="Custom pipeline config file")
    parser.add_argument("--backend", choices=["byaldi", "vespa"], help="Search backend")
    
    # Pipeline step toggles
    parser.add_argument("--skip-keyframes", action="store_true", help="Skip keyframe extraction")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio transcription")
    parser.add_argument("--skip-descriptions", action="store_true", help="Skip VLM descriptions")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    
    args = parser.parse_args()
    
    # Create pipeline config
    config = PipelineConfig.from_config()
    
    # Override with command line arguments
    if args.video_dir:
        config.video_dir = args.video_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.backend:
        config.search_backend = args.backend
        
    # Apply skip flags
    if args.skip_keyframes:
        config.extract_keyframes = False
    if args.skip_audio:
        config.transcribe_audio = False  
    if args.skip_descriptions:
        config.generate_descriptions = False
    if args.skip_embeddings:
        config.generate_embeddings = False
    
    # Run pipeline
    pipeline = VideoIngestionPipeline(config)
    results = pipeline.process_directory()