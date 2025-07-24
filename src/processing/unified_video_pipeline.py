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
    VLMDescriptor,
    EmbeddingGenerator
)


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
    output_dir: Path = Path("data/videos/processed")
    
    # Backend selection
    search_backend: str = "byaldi"  # "byaldi" or "vespa"
    
    @classmethod
    def from_config(cls) -> 'PipelineConfig':
        """Load pipeline config from main config file"""
        config = get_config()
        pipeline_config = config.get("pipeline_config", {})
        
        return cls(
            extract_keyframes=pipeline_config.get("extract_keyframes", True),
            transcribe_audio=pipeline_config.get("transcribe_audio", True), 
            generate_descriptions=pipeline_config.get("generate_descriptions", True),
            generate_embeddings=pipeline_config.get("generate_embeddings", True),
            keyframe_threshold=pipeline_config.get("keyframe_threshold", 0.999),
            max_frames_per_video=pipeline_config.get("max_frames_per_video", 3000),
            vlm_batch_size=pipeline_config.get("vlm_batch_size", 500),
            search_backend=config.get("search_backend", "byaldi")
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
        """Create necessary directories"""
        directories = [
            self.config.output_dir / "keyframes",
            self.config.output_dir / "metadata", 
            self.config.output_dir / "descriptions",
            self.config.output_dir / "transcripts",
            self.config.output_dir / "embeddings",
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
        
        # Create timestamp for log file
        from src.utils.output_manager import get_output_manager
        output_manager = get_output_manager()
        timestamp = int(time.time())
        log_file = output_manager.get_logs_dir() / f"video_processing_{timestamp}.log"
        
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
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Pipeline config: {self.config}")
        
        return logger
    
    def _initialize_pipeline_steps(self):
        """Initialize all pipeline step handlers"""
        # Initialize keyframe extractor
        self.keyframe_extractor = KeyframeExtractor(
            threshold=self.config.keyframe_threshold,
            max_frames=self.config.max_frames_per_video
        )
        
        # Initialize audio transcriber
        self.audio_transcriber = AudioTranscriber(
            model_size="base",
            device=None  # Auto-detect
        )
        
        # Initialize VLM descriptor
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
            
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            backend=self.config.search_backend
        )
            
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
        return self.keyframe_extractor.extract_keyframes(video_path, self.config.output_dir)
    
    def transcribe_audio(self, video_path: Path) -> Dict[str, Any]:
        """Extract and transcribe audio from video using AudioTranscriber"""
        return self.audio_transcriber.transcribe_audio(video_path, self.config.output_dir)
    
    def generate_descriptions(self, keyframes_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate VLM descriptions for keyframes using VLMDescriptor"""
        if not self.vlm_descriptor:
            print("  ⚠️ No VLM endpoint configured, skipping description generation")
            return {}
        
        return self.vlm_descriptor.generate_descriptions(keyframes_metadata, self.config.output_dir)
    
    
    def generate_embeddings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for search backend using EmbeddingGenerator"""
        self.logger.info("Extracting data for embedding generation...")
        
        # Pass minimal data to avoid large object serialization issues
        video_data = {
            "video_id": results.get("video_id"),
            "output_dir": str(self.config.output_dir)  # Pass path instead of large data structures
        }
        
        self.logger.info(f"Data extracted - Video ID: {video_data['video_id']}")
        self.logger.info(f"Output directory: {video_data['output_dir']}")
        
        result = self.embedding_generator.generate_embeddings(video_data, self.config.output_dir)
        return result
    
    
    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """Process a single video through the complete pipeline"""
        video_id = video_path.stem
        
        self.logger.info(f"Starting video processing: {video_id}")
        self.logger.info(f"Video path: {video_path}")
        
        print(f"\n🎬 Processing video: {video_path.name}")
        print("=" * 60)
        
        # Convert PosixPath objects to strings for JSON serialization
        config_dict = self.config.__dict__.copy()
        config_dict["video_dir"] = str(config_dict["video_dir"])
        config_dict["output_dir"] = str(config_dict["output_dir"])
        
        results = {
            "video_id": video_id,
            "video_path": str(video_path),
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
                    # Check if keyframes were already available (very fast completion)
                    if step_time < 0.1:
                        self.logger.info(f"Step 1 completed in {step_time:.2f}s - found {keyframe_count} existing keyframes")
                    else:
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
                    # Check if transcript was already available (very fast completion)
                    if step_time < 0.1:
                        self.logger.info(f"Step 2 completed in {step_time:.2f}s - found {segment_count} existing segments")
                    else:
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
                    # Check if descriptions were already available (very fast completion)
                    if step_time < 0.1:
                        self.logger.info(f"Step 3 completed in {step_time:.2f}s - found {desc_count} existing descriptions")
                    else:
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
                    # Check if embeddings were already available (very fast completion)
                    if step_time < 1.0:  # Embeddings take longer, so use 1 second threshold
                        self.logger.info(f"Step 4 completed in {step_time:.2f}s - found {embed_count} existing embeddings")
                    else:
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
        
        self.logger.info(f"Starting batch processing: {len(video_files)} videos from {video_dir}")
        
        print(f"🎬 Found {len(video_files)} videos to process")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"⚙️ Pipeline config: {self.config}")
        
        # Convert PosixPath objects to strings for JSON serialization
        config_dict = self.config.__dict__.copy()
        config_dict["video_dir"] = str(config_dict["video_dir"])
        config_dict["output_dir"] = str(config_dict["output_dir"])
        
        results = {
            "pipeline_config": config_dict,
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
        summary_file = self.config.output_dir / "pipeline_summary.json"
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