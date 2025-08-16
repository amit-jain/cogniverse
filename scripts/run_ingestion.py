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

from src.app.ingestion.pipeline import VideoIngestionPipeline, PipelineConfig

def main():
    parser = argparse.ArgumentParser(description="Unified Video Processing Pipeline")
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, help="Output directory for processed data")
    parser.add_argument("--backend", choices=["byaldi", "vespa"], help="Search backend")
    parser.add_argument("--profile", nargs="+", help="Video processing profiles (space-separated, e.g., colqwen_chunks direct_video_frame)")
    
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
    
    # Get profiles to process
    from src.common.config import get_config
    app_config = get_config()
    
    if args.profile:
        profiles_to_process = args.profile
    else:
        # If no profiles specified, use the active one
        active = app_config.get("active_video_profile", "frame_based_colpali")
        profiles_to_process = [active]
    
    # Process each profile
    all_results = {}
    for profile in profiles_to_process:
        print(f"\n{'='*60}")
        print(f"🎯 Processing with profile: {profile}")
        print(f"{'='*60}")
        
        # Set the profile
        import os
        os.environ["VIDEO_PROFILE"] = profile
        # Reload config is not needed for dict
        
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
        
        print(f"🎬 Starting Video Processing Pipeline")
        print(f"📁 Video directory: {config.video_dir}")
        print(f"📂 Output directory: {config.output_dir}")
        print(f"🔧 Backend: {config.search_backend}")
        print(f"🎯 Profile: {profile}")
        print(f"⚙️ Pipeline steps enabled:")
        print(f"  - Keyframes: {config.extract_keyframes}")
        print(f"  - Audio: {config.transcribe_audio}")
        print(f"  - Descriptions: {config.generate_descriptions}")
        print(f"  - Embeddings: {config.generate_embeddings}")
        
        # Run pipeline
        pipeline = VideoIngestionPipeline(config)
        results = pipeline.process_directory()
        
        # Store results
        all_results[profile] = results
        
        # Display results for this profile
        if results.get("error"):
            print(f"\n❌ Pipeline failed for {profile}: {results['error']}")
        else:
            print(f"\n✅ Profile {profile} completed!")
            print(f"   Processed: {len(results['processed_videos'])} videos")
            print(f"   Failed: {len(results['failed_videos'])} videos")
            print(f"   Time: {results['total_processing_time']/60:.1f} minutes")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 Overall Summary")
    print(f"{'='*60}")
    print(f"Processed {len(profiles_to_process)} profiles")
    
    for profile, results in all_results.items():
        status = "✅" if not results.get("error") else "❌"
        print(f"{status} {profile}: {len(results.get('processed_videos', []))} videos processed")
    
    return 0

if __name__ == "__main__":
    exit(main())