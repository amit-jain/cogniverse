#!/usr/bin/env python3
"""
Async Video Ingestion Pipeline - Using Builder Pattern for clean initialization
"""

import argparse
import asyncio

# Add project root to path
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.app.ingestion.pipeline_builder import (
    build_simple_pipeline,
    build_test_pipeline,
    create_config,
    create_pipeline,
)


async def main_async():
    parser = argparse.ArgumentParser(description="Video Processing Pipeline with Builder Pattern")
    parser.add_argument("--tenant-id", type=str, default="default_tenant", help="Tenant identifier (default: default_tenant)")
    parser.add_argument("--video_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output_dir", type=Path, help="Output directory for processed data")
    parser.add_argument("--backend", choices=["byaldi", "vespa"], default="vespa", help="Search backend")
    parser.add_argument("--profile", nargs="+", help="Video processing profiles (space-separated)")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent videos to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Processing parameters
    parser.add_argument("--max-frames", type=int, help="Maximum frames per video")
    parser.add_argument("--test-mode", action="store_true", help="Use test mode with limited frames")

    args = parser.parse_args()

    # Get profiles to process
    from src.common.config_utils import get_config
    from cogniverse_core.config.manager import ConfigManager
    config_manager = ConfigManager()
    app_config = get_config(tenant_id="default", config_manager=config_manager)
    
    if args.profile:
        profiles_to_process = args.profile
    else:
        # If no profiles specified, use the active one
        active = app_config.get("active_video_profile", "video_colpali_smol500_mv_frame")
        profiles_to_process = [active]
    
    # Process each profile using Builder pattern
    all_results = {}
    for profile in profiles_to_process:
        print(f"\n{'='*60}")
        print(f"🎯 Processing with profile: {profile}")
        print(f"{'='*60}")
        
        # METHOD 1: Test mode - use test pipeline builder
        if args.test_mode:
            print("🧪 Using test pipeline builder...")
            pipeline = build_test_pipeline(
                tenant_id=args.tenant_id,
                video_dir=args.video_dir or Path("data/testset/evaluation/sample_videos"),
                schema=profile,
                max_frames=args.max_frames or 10
            )

        # METHOD 2: Simple usage - use simple pipeline builder
        elif not args.output_dir and not args.max_frames:
            print("🚀 Using simple pipeline builder...")
            pipeline = build_simple_pipeline(
                tenant_id=args.tenant_id,
                video_dir=args.video_dir or Path("data/testset/evaluation/sample_videos"),
                schema=profile,
                backend=args.backend,
                debug=args.debug
            )
            
        # METHOD 3: Advanced usage - use fluent builder with custom config
        else:
            print("🔧 Using advanced pipeline builder...")

            # Build custom config if needed
            if args.output_dir or args.max_frames:
                config_builder = (create_config()
                                 .video_dir(args.video_dir or Path("data/testset/evaluation/sample_videos"))
                                 .backend(args.backend))

                if args.output_dir:
                    config_builder = config_builder.output_dir(args.output_dir)
                if args.max_frames:
                    config_builder = config_builder.max_frames_per_video(args.max_frames)

                config = config_builder.build()

                # Create pipeline with custom config
                pipeline = (create_pipeline()
                           .with_tenant_id(args.tenant_id)
                           .with_config(config)
                           .with_schema(profile)
                           .with_debug(args.debug)
                           .with_concurrency(args.max_concurrent)
                           .build())
            else:
                # Use fluent builder directly
                pipeline = (create_pipeline()
                           .with_tenant_id(args.tenant_id)
                           .with_video_dir(args.video_dir or Path("data/testset/evaluation/sample_videos"))
                           .with_schema(profile)
                           .with_backend(args.backend)
                           .with_debug(args.debug)
                           .with_concurrency(args.max_concurrent)
                           .build())
        
        print("🎬 Starting Video Processing Pipeline")
        print(f"📁 Video directory: {pipeline.config.video_dir}")
        print(f"📂 Output directory: {pipeline.config.output_dir}")
        print(f"🔧 Backend: {pipeline.config.search_backend}")
        print(f"🎯 Profile: {profile}")
        if args.max_frames:
            print(f"🖼️ Max frames: {args.max_frames}")
        if args.debug:
            print("🐛 Debug mode: Enabled")
        
        # Get video files from the pipeline's video directory
        video_files = list(pipeline.config.video_dir.glob('*.mp4'))
        if not video_files:
            print(f"❌ No MP4 files found in {pipeline.config.video_dir}")
            continue
            
        print(f"📹 Found {len(video_files)} videos to process")
        
        # Process videos using concurrent async method
        start_time = time.time()
        results = await pipeline.process_videos_concurrent(video_files, max_concurrent=args.max_concurrent)
        total_time = time.time() - start_time
        
        # Calculate statistics from results list
        if results:
            num_videos = len(results)
            successful = sum(1 for result in results if result.get('status') == 'completed')
            total_docs_fed = sum(
                result.get('results', {}).get('embeddings', {}).get('documents_fed', 0)
                for result in results if result.get('status') == 'completed'
            )
            failed = num_videos - successful
        else:
            num_videos = successful = total_docs_fed = failed = 0
        
        # Determine status icon based on success rate
        if successful == 0:
            status_icon = "❌"
            status_text = "failed"
        elif successful < num_videos:
            status_icon = "⚠️"
            status_text = "partially completed"
        else:
            status_icon = "✅"
            status_text = "completed"
            
        print(f"\n{status_icon} Profile {profile} {status_text}!")
        print(f"   Time: {total_time:.2f} seconds")
        print(f"   Videos: {successful}/{num_videos} successful")
        if failed > 0:
            print(f"   Failed: {failed} videos")
        print(f"   Documents fed: {total_docs_fed}")
        if total_time > 0 and total_docs_fed > 0:
            print(f"   Throughput: {total_docs_fed/total_time:.1f} docs/sec")
        if num_videos > 0:
            print(f"   Avg per video: {total_time/num_videos:.2f} seconds")
        
        # Store results
        all_results[profile] = {
            'results': results,
            'time': total_time,
            'docs_fed': total_docs_fed,
            'successful': successful,
            'failed': failed,
            'total': num_videos,
            'status': status_text
        }
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 Overall Summary")
    print(f"{'='*60}")
    print(f"Processed {len(profiles_to_process)} profiles")
    
    for profile, result_data in all_results.items():
        # Use appropriate icon based on status
        if result_data['status'] == 'failed':
            icon = "❌"
        elif result_data['status'] == 'partially completed':
            icon = "⚠️"
        else:
            icon = "✅"
            
        status_msg = f"{icon} {profile}: {result_data['successful']}/{result_data['total']} videos succeeded"
        
        if result_data['successful'] > 0:
            status_msg += f", {result_data['docs_fed']} docs in {result_data['time']:.1f}s"
        
        print(status_msg)
    
    return 0

def main():
    """Wrapper to run async main"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    exit(main())
