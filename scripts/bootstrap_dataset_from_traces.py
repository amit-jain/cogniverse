#!/usr/bin/env python3
"""
Bootstrap evaluation dataset from existing Phoenix traces.

This script helps create datasets by:
1. Extracting successful searches from Phoenix
2. Using high-confidence results as ground truth
3. Optionally filtering by metrics or patterns
"""

import json
import sys
from pathlib import Path

import click
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.data.datasets import DatasetManager
from src.evaluation.data.traces import TraceManager


@click.command()
@click.option('--hours', default=24, help='Hours to look back for traces')
@click.option('--min-score', default=0.7, type=float, help='Minimum score for ground truth')
@click.option('--min-results', default=1, type=int, help='Minimum number of results')
@click.option('--max-results', default=5, type=int, help='Maximum number of results for ground truth')
@click.option('--output-csv', help='Output CSV file path')
@click.option('--output-json', help='Output JSON file path')
@click.option('--dataset-name', help='Create Phoenix dataset directly')
@click.option('--filter-profile', help='Only include specific profile')
@click.option('--filter-strategy', help='Only include specific strategy')
@click.option('--dedupe', is_flag=True, help='Remove duplicate queries')
def bootstrap_dataset(
    hours, min_score, min_results, max_results,
    output_csv, output_json, dataset_name,
    filter_profile, filter_strategy, dedupe
):
    """
    Bootstrap evaluation dataset from existing traces.
    
    Examples:
        # Create dataset from last 24 hours of high-confidence results
        python bootstrap_dataset_from_traces.py --hours 24 --min-score 0.8 --dataset-name bootstrap_v1
        
        # Export to CSV for manual review
        python bootstrap_dataset_from_traces.py --output-csv review.csv --dedupe
        
        # Filter by specific configuration
        python bootstrap_dataset_from_traces.py --filter-profile frame_based_colpali --output-json queries.json
    """
    trace_manager = TraceManager()
    
    click.echo(f"Fetching traces from last {hours} hours...")
    
    # Get recent traces
    df = trace_manager.get_recent_traces(hours_back=hours, limit=1000)
    
    if df.empty:
        click.echo("No traces found")
        return
    
    click.echo(f"Found {len(df)} traces")
    
    # Extract trace data
    traces = trace_manager.extract_trace_data(df)
    
    # Filter and process traces
    dataset_queries = []
    seen_queries = set()
    
    for trace in traces:
        # Apply filters
        if filter_profile and trace.get('profile') != filter_profile:
            continue
        if filter_strategy and trace.get('strategy') != filter_strategy:
            continue
            
        query = trace.get('query', '').strip()
        if not query:
            continue
            
        # Skip duplicates if requested
        if dedupe and query.lower() in seen_queries:
            continue
            
        results = trace.get('results', [])
        if not results:
            continue
            
        # Filter by result count
        if len(results) < min_results:
            continue
            
        # Get high-confidence results as ground truth
        expected_videos = []
        for result in results[:max_results]:
            score = result.get('score', 0)
            if score >= min_score:
                video_id = result.get('video_id') or result.get('source_id')
                if video_id:
                    expected_videos.append(video_id)
        
        if not expected_videos:
            continue
            
        # Categorize query (simple heuristic)
        category = categorize_query(query)
        
        # Add to dataset
        dataset_queries.append({
            'query': query,
            'expected_videos': expected_videos,
            'category': category,
            'source_trace_id': trace.get('trace_id'),
            'source_profile': trace.get('profile'),
            'source_strategy': trace.get('strategy'),
            'avg_score': sum(r.get('score', 0) for r in results[:len(expected_videos)]) / len(expected_videos)
        })
        
        if dedupe:
            seen_queries.add(query.lower())
    
    click.echo(f"Processed {len(dataset_queries)} unique queries")
    
    # Output results
    if output_csv:
        df_output = pd.DataFrame(dataset_queries)
        # Format expected_videos as comma-separated string for CSV
        df_output['expected_videos'] = df_output['expected_videos'].apply(lambda x: ','.join(x))
        df_output.to_csv(output_csv, index=False)
        click.echo(f"Saved to CSV: {output_csv}")
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(dataset_queries, f, indent=2)
        click.echo(f"Saved to JSON: {output_json}")
    
    if dataset_name:
        # Create Phoenix dataset
        dataset_manager = DatasetManager()
        
        # Format for Phoenix (only keep essential fields)
        phoenix_queries = [
            {
                'query': q['query'],
                'expected_videos': q['expected_videos'],
                'category': q['category']
            }
            for q in dataset_queries
        ]
        
        dataset_id = dataset_manager.create_from_queries(
            queries=phoenix_queries,
            dataset_name=dataset_name,
            description=f"Bootstrapped from {hours}h of traces (min_score={min_score})"
        )
        
        click.echo(f"Created Phoenix dataset '{dataset_name}' with ID: {dataset_id}")
    
    # Print statistics
    click.echo("\n" + "="*50)
    click.echo("Dataset Statistics:")
    click.echo(f"Total queries: {len(dataset_queries)}")
    
    if dataset_queries:
        categories = {}
        for q in dataset_queries:
            cat = q['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        click.echo("\nCategories:")
        for cat, count in sorted(categories.items()):
            click.echo(f"  {cat}: {count}")
        
        avg_videos = sum(len(q['expected_videos']) for q in dataset_queries) / len(dataset_queries)
        click.echo(f"\nAverage expected videos per query: {avg_videos:.2f}")
        
        # Show sample queries
        click.echo("\nSample queries:")
        for q in dataset_queries[:5]:
            click.echo(f"  - {q['query'][:60]}... ({len(q['expected_videos'])} videos)")


def categorize_query(query: str) -> str:
    """
    Simple heuristic to categorize queries.
    
    You can make this more sophisticated using:
    - Keyword matching
    - NLP classification
    - LLM-based categorization
    """
    query_lower = query.lower()
    
    # Temporal indicators
    temporal_keywords = ['after', 'before', 'then', 'when', 'during', 'while', 'later', 'earlier']
    if any(keyword in query_lower for keyword in temporal_keywords):
        return 'temporal'
    
    # Audio/speech indicators
    audio_keywords = ['said', 'mentioned', 'discussed', 'talked', 'speaking', 'audio', 'sound']
    if any(keyword in query_lower for keyword in audio_keywords):
        return 'audio'
    
    # Activity indicators
    activity_keywords = ['walking', 'running', 'playing', 'doing', 'performing', 'action']
    if any(keyword in query_lower.split() for keyword in activity_keywords):
        return 'activity'
    
    # Default to visual
    return 'visual'


if __name__ == '__main__':
    bootstrap_dataset()
