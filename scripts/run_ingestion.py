#!/usr/bin/env python3
"""
Unified Video Ingestion Pipeline Entry Point

Uses the VideoIngestionPipeline class as the single entry point for all video processing.
"""

import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.processing.unified_video_pipeline import VideoIngestionPipeline, PipelineConfig

def main():
    parser = argparse.ArgumentParser(description="Unified Video Processing Pipeline")
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, help="Output directory for processed data")
    parser.add_argument("--backend", choices=["byaldi", "vespa"], help="Search backend")
    
    # Pipeline step toggles
    parser.add_argument("--skip-keyframes", action="store_true", help="Skip keyframe extraction")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio transcription")
    parser.add_argument("--skip-descriptions", action="store_true", help="Skip VLM descriptions")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    
    # Processing parameters
    parser.add_argument("--keyframe-threshold", type=float, default=0.98, help="Keyframe extraction threshold")
    parser.add_argument("--max-frames", type=int, default=3000, help="Maximum frames per video")
    parser.add_argument("--vlm-batch-size", type=int, default=1000, help="VLM batch processing size")
    
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
        
    # Processing parameters
    if args.keyframe_threshold != 0.98:  # Only override if user specified
        config.keyframe_threshold = args.keyframe_threshold
    if args.max_frames != 3000:
        config.max_frames_per_video = args.max_frames
    if args.vlm_batch_size != 1000:
        config.vlm_batch_size = args.vlm_batch_size
        
    # Apply skip flags
    if args.skip_keyframes:
        config.extract_keyframes = False
    if args.skip_audio:
        config.transcribe_audio = False  
    if args.skip_descriptions:
        config.generate_descriptions = False
    if args.skip_embeddings:
        config.generate_embeddings = False
    
    print(f"🎬 Starting Unified Video Processing Pipeline")
    print(f"📁 Video directory: {config.video_dir}")
    print(f"📂 Output directory: {config.output_dir}")
    print(f"🔧 Backend: {config.search_backend}")
    print(f"⚙️ Pipeline steps enabled:")
    print(f"  - Keyframes: {config.extract_keyframes}")
    print(f"  - Audio: {config.transcribe_audio}")
    print(f"  - Descriptions: {config.generate_descriptions}")
    print(f"  - Embeddings: {config.generate_embeddings}")
    
    # Run pipeline
    pipeline = VideoIngestionPipeline(config)
    results = pipeline.process_directory()
    
    # Display final results
    if results.get("error"):
        print(f"\n❌ Pipeline failed: {results['error']}")
        return 1
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"✅ Processed: {len(results['processed_videos'])} videos")
    print(f"❌ Failed: {len(results['failed_videos'])} videos")
    print(f"⏱️ Total time: {results['total_processing_time']/60:.1f} minutes")
    
    if results['failed_videos']:
        print(f"\n⚠️ Failed videos:")
        for failed in results['failed_videos']:
            video_path = failed.get('video_path', 'unknown')
            error = failed.get('error', 'unknown error')
            print(f"  - {Path(video_path).name}: {error}")
    
    return 0

if __name__ == "__main__":
    exit(main())