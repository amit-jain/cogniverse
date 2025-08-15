#!/usr/bin/env python3
"""
Import standard video retrieval benchmarks and adapt them for evaluation.

Supports common formats like:
- MSR-VTT
- ActivityNet Captions
- DiDeMo
- Custom JSON formats
"""

import click
import json
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.data.datasets import DatasetManager


@click.command()
@click.option('--format', type=click.Choice(['msr-vtt', 'activitynet', 'didemo', 'custom']), required=True)
@click.option('--input-file', type=click.Path(exists=True), required=True)
@click.option('--video-mapping', type=click.Path(exists=True), help='JSON mapping benchmark IDs to your video IDs')
@click.option('--dataset-name', required=True, help='Name for Phoenix dataset')
@click.option('--max-queries', type=int, help='Limit number of queries to import')
def import_benchmark(format, input_file, video_mapping, dataset_name, max_queries):
    """
    Import benchmark datasets for evaluation.
    
    Examples:
        # Import MSR-VTT style dataset
        python import_benchmark_dataset.py --format msr-vtt --input-file msr_vtt.json --dataset-name msr_vtt_eval
        
        # Import with video ID mapping
        python import_benchmark_dataset.py --format custom --input-file benchmark.json \\
            --video-mapping id_mapping.json --dataset-name benchmark_eval
    """
    
    # Load video ID mapping if provided
    id_mapping = {}
    if video_mapping:
        with open(video_mapping, 'r') as f:
            id_mapping = json.load(f)
    
    # Parse based on format
    if format == 'msr-vtt':
        queries = parse_msr_vtt(input_file, id_mapping)
    elif format == 'activitynet':
        queries = parse_activitynet(input_file, id_mapping)
    elif format == 'didemo':
        queries = parse_didemo(input_file, id_mapping)
    elif format == 'custom':
        queries = parse_custom(input_file, id_mapping)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Limit queries if requested
    if max_queries and len(queries) > max_queries:
        queries = queries[:max_queries]
    
    click.echo(f"Parsed {len(queries)} queries from {format} format")
    
    # Create dataset
    dataset_manager = DatasetManager()
    dataset_id = dataset_manager.create_from_queries(
        queries=queries,
        dataset_name=dataset_name,
        description=f"Imported from {format} benchmark"
    )
    
    click.echo(f"Created dataset '{dataset_name}' with ID: {dataset_id}")
    
    # Show statistics
    categories = {}
    for q in queries:
        cat = q.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    click.echo("\nDataset statistics:")
    click.echo(f"Total queries: {len(queries)}")
    for cat, count in categories.items():
        click.echo(f"  {cat}: {count}")


def parse_msr_vtt(input_file: str, id_mapping: dict) -> list:
    """Parse MSR-VTT format dataset."""
    queries = []
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for item in data.get('sentences', []):
        video_id = item.get('video_id', '')
        caption = item.get('caption', '')
        
        if not caption:
            continue
        
        # Map video ID if mapping provided
        if id_mapping:
            video_id = id_mapping.get(video_id, video_id)
        
        queries.append({
            'query': caption,
            'expected_videos': [video_id],
            'category': 'caption'
        })
    
    return queries


def parse_activitynet(input_file: str, id_mapping: dict) -> list:
    """Parse ActivityNet Captions format."""
    queries = []
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for video_id, video_data in data.items():
        # Map video ID if needed
        mapped_id = id_mapping.get(video_id, video_id) if id_mapping else video_id
        
        for sentence in video_data.get('sentences', []):
            queries.append({
                'query': sentence,
                'expected_videos': [mapped_id],
                'category': 'activity'
            })
    
    return queries


def parse_didemo(input_file: str, id_mapping: dict) -> list:
    """Parse DiDeMo format dataset."""
    queries = []
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for item in data:
        video_id = item.get('video', '')
        description = item.get('description', '')
        
        if not description:
            continue
        
        # Map video ID if needed
        if id_mapping:
            video_id = id_mapping.get(video_id, video_id)
        
        queries.append({
            'query': description,
            'expected_videos': [video_id],
            'category': 'temporal'  # DiDeMo focuses on temporal localization
        })
    
    return queries


def parse_custom(input_file: str, id_mapping: dict) -> list:
    """
    Parse custom JSON format.
    
    Expected format:
    [
        {
            "query": "text query",
            "video_id": "video1",  // or "videos": ["video1", "video2"]
            "category": "optional_category"
        }
    ]
    """
    queries = []
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for item in data:
        query = item.get('query', item.get('text', item.get('caption', '')))
        
        if not query:
            continue
        
        # Handle different video ID fields
        if 'videos' in item:
            video_ids = item['videos']
        elif 'video_id' in item:
            video_ids = [item['video_id']]
        elif 'video' in item:
            video_ids = [item['video']]
        else:
            continue
        
        # Map video IDs if needed
        if id_mapping:
            video_ids = [id_mapping.get(v, v) for v in video_ids]
        
        queries.append({
            'query': query,
            'expected_videos': video_ids,
            'category': item.get('category', 'general')
        })
    
    return queries


if __name__ == '__main__':
    import_benchmark()