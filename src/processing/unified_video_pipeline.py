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

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.config import get_config
from src.processing.pipeline_steps import (
    KeyframeExtractor,
    AudioTranscriber,
    VLMDescriptor
)
from src.processing.pipeline_steps.keyframe_extractor_fps import FPSKeyframeExtractor
from src.processing.pipeline_steps.embedding_generator import create_embedding_generator


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
        self.setup_directories()
        self.logger = self._setup_logging()
        self._initialize_pipeline_steps()
        
    def setup_directories(self):
        """Create necessary directories with profile-specific organization"""
        # Get active profile name
        active_profile = self.app_config.get_active_profile()
        if active_profile:
            # Use profile-specific directory
            self.profile_output_dir = self.config.output_dir / f"profile_{active_profile}"
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
        active_profile = self.app_config.get_active_profile() or "default"
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
    
    def _initialize_pipeline_steps(self):
        """Initialize all pipeline step handlers"""
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
                # Get the config data from the Config object
                generator_config = self.app_config.config_data.copy()
                generator_config["embedding_backend"] = self.config.search_backend
                generator_config["active_profile"] = self.app_config.get_active_profile()
                
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
        return self.keyframe_extractor.extract_keyframes(video_path, self.profile_output_dir)
    
    def transcribe_audio(self, video_path: Path) -> Dict[str, Any]:
        """Extract and transcribe audio from video using AudioTranscriber"""
        return self.audio_transcriber.transcribe_audio(video_path, self.profile_output_dir)
    
    def generate_descriptions(self, keyframes_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate VLM descriptions for keyframes using VLMDescriptor"""
        if not self.vlm_descriptor:
            print("  ⚠️ No VLM endpoint configured, skipping description generation")
            return {}
        
        return self.vlm_descriptor.generate_descriptions(keyframes_metadata, self.profile_output_dir)
    
    
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
        
        # For frame-based processing, add frame information if available
        if "keyframes" in results.get("results", {}):
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
                
            # Step 2: Transcribe audio
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
                descriptions_data = self.generate_descriptions(keyframes_data)
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
        
        active_profile = self.app_config.get_active_profile() or "default"
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
        
        # Save summary
        summary_file = self.profile_output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
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