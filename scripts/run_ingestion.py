#!/usr/bin/env python3
"""
Content Ingestion Pipeline - Supports video, image, audio, and document content.

Uses Builder Pattern for clean initialization.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cogniverse_runtime.ingestion.pipeline_builder import (
    build_simple_pipeline,
    build_test_pipeline,
    create_config,
    create_pipeline,
)

IMAGE_EXTENSIONS = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"}
VIDEO_EXTENSIONS = {"*.mp4", "*.avi", "*.mov", "*.mkv"}


def discover_content_files(content_dir: Path, content_type: str) -> list[Path]:
    """Discover content files based on content type.

    For video: returns individual video files (each processed separately).
    For image: returns the directory itself (all images processed as one batch).
    """
    if content_type == "image":
        # For images, the directory itself is the content item.
        # ImageSegmentationStrategy will discover individual files inside.
        return [content_dir]

    # Default: discover video files
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(content_dir.glob(ext))
    return sorted(files)


async def main_async():
    parser = argparse.ArgumentParser(
        description="Content Ingestion Pipeline (video, image, audio, document)"
    )
    parser.add_argument(
        "--tenant-id", type=str, default="default_tenant",
        help="Tenant identifier (default: default_tenant)",
    )
    parser.add_argument("--video_dir", type=Path, help="Directory containing content files")
    parser.add_argument("--content-dir", type=Path, help="Directory containing content files (alias for --video_dir)")
    parser.add_argument("--output_dir", type=Path, help="Output directory for processed data")
    parser.add_argument(
        "--backend", choices=["byaldi", "vespa"], default="vespa",
        help="Search backend",
    )
    parser.add_argument("--profile", nargs="+", help="Processing profiles (space-separated)")
    parser.add_argument(
        "--content-type", choices=["video", "image", "audio", "document"],
        default="video", help="Content type to ingest (default: video)",
    )
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent items to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per video / images per batch")
    parser.add_argument("--test-mode", action="store_true", help="Use test mode with limited frames")

    args = parser.parse_args()

    content_dir = args.content_dir or args.video_dir

    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )

    config_manager = create_default_config_manager()
    app_config = get_config(tenant_id="default", config_manager=config_manager)

    if args.profile:
        profiles_to_process = args.profile
    else:
        active = app_config.get("active_video_profile", "video_colpali_smol500_mv_frame")
        profiles_to_process = [active]

    all_results = {}
    for profile in profiles_to_process:
        print(f"\n{'='*60}")
        print(f"Processing with profile: {profile}")
        print(f"{'='*60}")

        default_dir = Path("data/testset/evaluation/sample_videos")
        input_dir = content_dir or default_dir

        if args.test_mode:
            pipeline = build_test_pipeline(
                tenant_id=args.tenant_id,
                video_dir=input_dir,
                schema=profile,
                max_frames=args.max_frames or 10,
            )
        elif not args.output_dir and not args.max_frames:
            pipeline = build_simple_pipeline(
                tenant_id=args.tenant_id,
                video_dir=input_dir,
                schema=profile,
                backend=args.backend,
                debug=args.debug,
            )
        else:
            config_builder = create_config().video_dir(input_dir).backend(args.backend)
            if args.output_dir:
                config_builder = config_builder.output_dir(args.output_dir)
            if args.max_frames:
                config_builder = config_builder.max_frames_per_video(args.max_frames)
            config = config_builder.build()

            pipeline = (
                create_pipeline()
                .with_tenant_id(args.tenant_id)
                .with_config(config)
                .with_schema(profile)
                .with_debug(args.debug)
                .with_concurrency(args.max_concurrent)
                .build()
            )

        print(f"Content type: {args.content_type}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {pipeline.config.output_dir}")
        print(f"Backend: {pipeline.config.search_backend}")
        print(f"Profile: {profile}")

        content_files = discover_content_files(input_dir, args.content_type)
        if not content_files:
            print(f"No {args.content_type} files found in {input_dir}")
            continue

        print(f"Found {len(content_files)} {args.content_type} item(s) to process")

        start_time = time.time()
        job_result = await pipeline.process_videos_concurrent(
            content_files, max_concurrent=args.max_concurrent
        )
        total_time = time.time() - start_time

        if job_result:
            num_items = job_result.get("total_videos", 0)
            successful = job_result.get("successful", 0)
            failed = job_result.get("failed", 0)
            per_item_results = job_result.get("results", [])
            total_docs_fed = sum(
                r.get("results", {}).get("embeddings", {}).get("documents_fed", 0)
                for r in per_item_results
                if isinstance(r, dict) and r.get("status") == "completed"
            )
        else:
            num_items = successful = total_docs_fed = failed = 0

        if successful == 0:
            status_text = "failed"
        elif successful < num_items:
            status_text = "partially completed"
        else:
            status_text = "completed"

        print(f"\nProfile {profile} {status_text}!")
        print(f"   Time: {total_time:.2f} seconds")
        print(f"   Items: {successful}/{num_items} successful")
        if failed > 0:
            print(f"   Failed: {failed}")
        print(f"   Documents fed: {total_docs_fed}")
        if total_time > 0 and total_docs_fed > 0:
            print(f"   Throughput: {total_docs_fed/total_time:.1f} docs/sec")

        all_results[profile] = {
            "results": job_result,
            "time": total_time,
            "docs_fed": total_docs_fed,
            "successful": successful,
            "failed": failed,
            "total": num_items,
            "status": status_text,
        }

    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")
    print(f"Processed {len(profiles_to_process)} profiles")

    for profile, result_data in all_results.items():
        status_msg = (
            f"  {profile}: {result_data['successful']}/{result_data['total']} succeeded"
        )
        if result_data["successful"] > 0:
            status_msg += f", {result_data['docs_fed']} docs in {result_data['time']:.1f}s"
        print(status_msg)

    return 0


def main():
    """Wrapper to run async main"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())
