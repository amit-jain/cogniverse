#!/usr/bin/env python3
"""
Generate evaluation dataset from video metadata and content.

This script generates queries based on:
1. Video filenames and metadata
2. Transcripts (if available)
3. Frame descriptions (if available)
4. LLM-generated queries
"""

import click
import json
from pathlib import Path
import random
from typing import List, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.data.datasets import DatasetManager


@click.command()
@click.option('--video-dir', type=click.Path(exists=True), required=True, help='Directory containing videos')
@click.option('--metadata-file', type=click.Path(exists=True), help='JSON file with video metadata')
@click.option('--num-queries', default=50, help='Number of queries to generate')
@click.option('--output-json', help='Output JSON file')
@click.option('--dataset-name', help='Create Phoenix dataset directly')
@click.option('--use-llm', is_flag=True, help='Use LLM to generate queries')
def generate_dataset(video_dir, metadata_file, num_queries, output_json, dataset_name, use_llm):
    """
    Generate evaluation dataset from video collection.
    
    Examples:
        # Generate from video directory
        python generate_dataset_from_videos.py --video-dir data/videos --num-queries 100
        
        # Use metadata file
        python generate_dataset_from_videos.py --video-dir data/videos --metadata-file metadata.json
        
        # Use LLM for query generation
        python generate_dataset_from_videos.py --video-dir data/videos --use-llm --dataset-name generated_v1
    """
    video_path = Path(video_dir)
    videos = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))
    
    click.echo(f"Found {len(videos)} videos")
    
    # Load metadata if provided
    metadata = {}
    if metadata_file:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Generate queries
    dataset_queries = []
    
    # Strategy 1: Filename-based queries
    dataset_queries.extend(generate_filename_queries(videos, metadata))
    
    # Strategy 2: Metadata-based queries
    if metadata:
        dataset_queries.extend(generate_metadata_queries(metadata))
    
    # Strategy 3: Combination queries (multiple videos)
    dataset_queries.extend(generate_combination_queries(videos, metadata))
    
    # Strategy 4: LLM-generated queries
    if use_llm:
        dataset_queries.extend(generate_llm_queries(videos, metadata))
    
    # Limit to requested number
    if len(dataset_queries) > num_queries:
        dataset_queries = random.sample(dataset_queries, num_queries)
    
    click.echo(f"Generated {len(dataset_queries)} queries")
    
    # Output results
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(dataset_queries, f, indent=2)
        click.echo(f"Saved to: {output_json}")
    
    if dataset_name:
        dataset_manager = DatasetManager()
        dataset_id = dataset_manager.create_from_queries(
            queries=dataset_queries,
            dataset_name=dataset_name,
            description=f"Generated from {len(videos)} videos"
        )
        click.echo(f"Created dataset '{dataset_name}' with ID: {dataset_id}")
    
    # Show samples
    click.echo("\nSample queries:")
    for q in dataset_queries[:5]:
        click.echo(f"  - {q['query']}")
        click.echo(f"    Expected: {q['expected_videos'][:2]}...")


def generate_filename_queries(videos: List[Path], metadata: Dict) -> List[Dict]:
    """Generate queries based on video filenames."""
    queries = []
    
    for video in videos[:20]:  # Limit for efficiency
        name = video.stem.replace('_', ' ').replace('-', ' ')
        
        # Extract potential keywords
        words = name.lower().split()
        
        # Generate different query types
        if len(words) >= 2:
            # Partial name query
            queries.append({
                'query': ' '.join(words[:2]),
                'expected_videos': [video.name],
                'category': 'filename_based'
            })
            
            # Keyword query
            if len(words) > 2:
                keywords = random.sample(words, min(2, len(words)))
                queries.append({
                    'query': ' '.join(keywords),
                    'expected_videos': [video.name],
                    'category': 'keyword'
                })
    
    return queries


def generate_metadata_queries(metadata: Dict) -> List[Dict]:
    """Generate queries from video metadata."""
    queries = []
    
    for video_id, video_meta in metadata.items():
        # Use description if available
        if 'description' in video_meta:
            desc = video_meta['description']
            # Take first sentence or first N words
            query = desc.split('.')[0][:100]
            queries.append({
                'query': query,
                'expected_videos': [video_id],
                'category': 'description'
            })
        
        # Use tags if available
        if 'tags' in video_meta:
            tags = video_meta['tags']
            if tags:
                queries.append({
                    'query': ' '.join(random.sample(tags, min(3, len(tags)))),
                    'expected_videos': [video_id],
                    'category': 'tags'
                })
        
        # Use transcript snippets if available
        if 'transcript' in video_meta:
            transcript = video_meta['transcript']
            if transcript:
                # Take a random sentence
                sentences = transcript.split('.')
                if sentences:
                    query = random.choice(sentences).strip()[:100]
                    if query:
                        queries.append({
                            'query': query,
                            'expected_videos': [video_id],
                            'category': 'transcript'
                        })
    
    return queries


def generate_combination_queries(videos: List[Path], metadata: Dict) -> List[Dict]:
    """Generate queries that should return multiple videos."""
    queries = []
    
    # Group videos by common patterns
    patterns = {}
    for video in videos:
        # Extract potential pattern (e.g., "meeting_001" -> "meeting")
        name_parts = video.stem.split('_')
        if name_parts:
            pattern = name_parts[0]
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(video.name)
    
    # Create queries for patterns with multiple videos
    for pattern, video_list in patterns.items():
        if len(video_list) > 1 and len(video_list) <= 5:
            queries.append({
                'query': f"{pattern} videos",
                'expected_videos': video_list[:3],  # Limit to 3 for evaluation
                'category': 'multi_result'
            })
            
            # More specific multi-result query
            queries.append({
                'query': f"all {pattern} footage",
                'expected_videos': video_list[:3],
                'category': 'multi_result'
            })
    
    return queries


def generate_llm_queries(videos: List[Path], metadata: Dict) -> List[Dict]:
    """Generate queries using LLM (requires API key)."""
    queries = []
    
    # This is a placeholder - implement based on your LLM preference
    click.echo("LLM query generation not implemented yet")
    
    # Example implementation outline:
    # for video in random.sample(videos, min(10, len(videos))):
    #     prompt = f"Generate 3 natural search queries for a video named '{video.stem}'"
    #     # Call LLM API
    #     # Parse responses
    #     # Add to queries list
    
    return queries


if __name__ == '__main__':
    generate_dataset()