#!/usr/bin/env python3
"""
Async Video Ingestion Pipeline - Faster concurrent processing
"""

import argparse
import asyncio
import time
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.app.ingestion.pipeline import VideoIngestionPipeline, PipelineConfig

async def main_async():
    parser = argparse.ArgumentParser(description="Async Video Processing Pipeline with Performance Optimizations")
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, help="Output directory for processed data")
    parser.add_argument("--backend", choices=["byaldi", "vespa"], help="Search backend")
    parser.add_argument("--profile", nargs="+", help="Video processing profiles (space-separated, e.g., colqwen_chunks direct_video_frame)")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent videos to process (default: 3)")
    parser.add_argument("--enable-async-vespa", action="store_true", help="Enable async Vespa feeding (Phase 3 optimization)")
    
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
        
        # Enable async Vespa if requested
        if args.enable_async_vespa and config.search_backend == "vespa":
            # Get the actual schema name from the profile config
            profile_config = config.config.get('video_processing_profiles', {}).get(profile, {})
            schema_name = profile_config.get('schema_name', profile)
            
            # CRITICAL: Log the schema mapping to ensure correct assignment
            print(f"\n🔍 SCHEMA MAPPING CHECK:")
            print(f"   Profile name: {profile}")
            print(f"   Expected schema: {schema_name}")
            print(f"   Profile config keys: {list(profile_config.keys())[:5]}...")
            
            config.backend_config = {
                'vespa_url': 'http://localhost',
                'vespa_port': 8080,
                'schema_name': schema_name,
                'use_async_ingestion': True
            }
            print(f"⚡ Async Vespa feeding enabled for schema: {schema_name}")
            print(f"   Backend config: {config.backend_config}")
        
        # Set schema_name in config for the pipeline
        config.schema_name = profile
        
        # Initialize pipeline with config, app_config, and schema_name
        # The VideoIngestionPipeline now includes all async optimizations
        pipeline = VideoIngestionPipeline(config, app_config, profile)
        
        # Get video files
        video_files = list(config.video_dir.glob('*.mp4'))
        if not video_files:
            print(f"❌ No MP4 files found in {config.video_dir}")
            continue
        
        print(f"\n📹 Found {len(video_files)} videos to process")
        
        # Process videos concurrently with timing
        start_time = time.time()
        results = await pipeline.process_videos_concurrent(video_files, max_concurrent=args.max_concurrent)
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get('status') == 'completed')
        total_docs_fed = sum(
            r.get('results', {}).get('embeddings', {}).get('documents_fed', 0)
            for r in results if r.get('status') == 'completed'
        )
        
        # Display results for this profile
        print(f"\n✅ Profile {profile} completed!")
        print(f"   Time: {total_time:.2f} seconds")
        print(f"   Videos: {successful}/{len(video_files)} successful")
        print(f"   Documents fed: {total_docs_fed}")
        print(f"   Throughput: {total_docs_fed/total_time:.1f} docs/sec" if total_time > 0 else "")
        print(f"   Avg per video: {total_time/len(video_files):.2f} seconds")
        
        # Store results
        all_results[profile] = {
            'results': results,
            'time': total_time,
            'docs_fed': total_docs_fed,
            'successful': successful
        }
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 Overall Summary")
    print(f"{'='*60}")
    print(f"Processed {len(profiles_to_process)} profiles")
    
    for profile, result_data in all_results.items():
        print(f"✅ {profile}: {result_data['successful']} videos, {result_data['docs_fed']} docs in {result_data['time']:.1f}s")
    
    return 0

def main():
    """Wrapper to run async main"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    exit(main())