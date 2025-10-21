#!/usr/bin/env python3
"""
Interactive dataset builder for creating evaluation datasets.

This tool allows you to:
1. Search for videos interactively
2. Mark correct results
3. Build a dataset incrementally
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import click

sys.path.append(str(Path(__file__).parent.parent))

from src.app.search.service import SearchService
from src.evaluation.data.datasets import DatasetManager


class InteractiveDatasetBuilder:
    """Interactive tool for building evaluation datasets."""
    
    def __init__(self, profile: str = "frame_based_colpali", strategy: str = "binary_binary"):
        self.search_service = SearchService(profile=profile, strategy=strategy)
        self.dataset_queries = []
        self.profile = profile
        self.strategy = strategy
    
    def search_and_annotate(self, query: str) -> Dict[str, Any]:
        """Search and let user annotate results."""
        click.echo(f"\nSearching for: '{query}'")
        click.echo("-" * 50)
        
        # Perform search
        results = self.search_service.search(query, top_k=10)
        
        if not results:
            click.echo("No results found")
            return None
        
        # Display results
        click.echo("\nSearch Results:")
        for i, result in enumerate(results, 1):
            video_id = result.get('video_id', result.get('source_id', 'unknown'))
            score = result.get('score', 0)
            content = result.get('content', '')[:100]
            
            click.echo(f"{i}. [{score:.3f}] {video_id}")
            if content:
                click.echo(f"   {content}...")
        
        # Get user annotation
        click.echo("\nMark relevant results (comma-separated numbers, or 'none' for no relevant results):")
        click.echo("Examples: 1,2,3 or 1,3,5 or none")
        
        while True:
            selection = click.prompt("Relevant results", type=str)
            
            if selection.lower() == 'none':
                return {
                    'query': query,
                    'expected_videos': [],
                    'category': self.categorize_query(query),
                    'annotator_notes': 'No relevant results'
                }
            
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                if all(0 <= i < len(results) for i in indices):
                    break
                else:
                    click.echo("Invalid selection. Please enter valid numbers.")
            except:
                click.echo("Invalid format. Please enter comma-separated numbers or 'none'.")
        
        # Extract selected videos
        expected_videos = []
        for idx in indices:
            result = results[idx]
            video_id = result.get('video_id', result.get('source_id'))
            if video_id:
                expected_videos.append(video_id)
        
        # Get category
        category = click.prompt(
            "Category",
            type=click.Choice(['visual', 'audio', 'temporal', 'activity', 'other']),
            default='visual'
        )
        
        # Optional notes
        notes = click.prompt("Notes (optional)", default="", show_default=False)
        
        return {
            'query': query,
            'expected_videos': expected_videos,
            'category': category,
            'annotator_notes': notes if notes else None,
            'search_profile': self.profile,
            'search_strategy': self.strategy
        }
    
    def categorize_query(self, query: str) -> str:
        """Auto-categorize query with option to override."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['after', 'before', 'when', 'during']):
            return 'temporal'
        elif any(word in query_lower for word in ['said', 'mentioned', 'discussed']):
            return 'audio'
        elif any(word in query_lower for word in ['walking', 'running', 'playing']):
            return 'activity'
        else:
            return 'visual'
    
    def run_interactive_session(self):
        """Run interactive dataset building session."""
        click.echo("="*60)
        click.echo("Interactive Dataset Builder")
        click.echo("="*60)
        click.echo(f"Profile: {self.profile}")
        click.echo(f"Strategy: {self.strategy}")
        click.echo("\nEnter queries to search and annotate.")
        click.echo("Type 'done' to finish, 'stats' to see progress, 'save' to save progress")
        click.echo("-"*60)
        
        while True:
            query = click.prompt("\nEnter query", type=str)
            
            if query.lower() == 'done':
                break
            elif query.lower() == 'stats':
                self.show_statistics()
                continue
            elif query.lower() == 'save':
                self.save_progress()
                continue
            
            # Search and annotate
            annotation = self.search_and_annotate(query)
            
            if annotation:
                self.dataset_queries.append(annotation)
                click.echo(f"\n✓ Added query to dataset (Total: {len(self.dataset_queries)})")
            
            # Ask if user wants to continue
            if not click.confirm("\nAdd another query?", default=True):
                break
        
        # Final summary and save
        self.show_statistics()
        
        if self.dataset_queries and click.confirm("\nSave dataset?", default=True):
            self.save_final_dataset()
    
    def show_statistics(self):
        """Show current dataset statistics."""
        if not self.dataset_queries:
            click.echo("\nNo queries in dataset yet.")
            return
        
        click.echo("\n" + "="*60)
        click.echo("Dataset Statistics:")
        click.echo(f"Total queries: {len(self.dataset_queries)}")
        
        # Category breakdown
        categories = {}
        for q in self.dataset_queries:
            cat = q['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        click.echo("\nCategories:")
        for cat, count in sorted(categories.items()):
            click.echo(f"  {cat}: {count}")
        
        # Coverage stats
        total_videos = set()
        for q in self.dataset_queries:
            total_videos.update(q['expected_videos'])
        
        click.echo(f"\nUnique videos covered: {len(total_videos)}")
        
        avg_videos = sum(len(q['expected_videos']) for q in self.dataset_queries) / len(self.dataset_queries)
        click.echo(f"Average videos per query: {avg_videos:.2f}")
    
    def save_progress(self):
        """Save current progress to file."""
        if not self.dataset_queries:
            click.echo("No queries to save.")
            return
        
        filename = click.prompt("Save as", default="dataset_progress.json")
        
        with open(filename, 'w') as f:
            json.dump(self.dataset_queries, f, indent=2)
        
        click.echo(f"✓ Saved {len(self.dataset_queries)} queries to {filename}")
    
    def save_final_dataset(self):
        """Save final dataset."""
        # Save to JSON
        json_file = click.prompt("JSON filename", default="dataset.json")
        with open(json_file, 'w') as f:
            json.dump(self.dataset_queries, f, indent=2)
        click.echo(f"✓ Saved to {json_file}")
        
        # Optionally create Phoenix dataset
        if click.confirm("Create Phoenix dataset?", default=True):
            dataset_name = click.prompt("Dataset name")
            
            dataset_manager = DatasetManager()
            dataset_id = dataset_manager.create_from_queries(
                queries=self.dataset_queries,
                dataset_name=dataset_name,
                description=f"Interactive dataset with {len(self.dataset_queries)} queries"
            )
            
            click.echo(f"✓ Created Phoenix dataset '{dataset_name}' with ID: {dataset_id}")


@click.command()
@click.option('--profile', default='frame_based_colpali', help='Search profile to use')
@click.option('--strategy', default='binary_binary', help='Search strategy to use')
@click.option('--load', type=click.Path(exists=True), help='Load previous progress')
def main(profile, strategy, load):
    """
    Interactive dataset builder for evaluation.
    
    Examples:
        # Start new session
        python interactive_dataset_builder.py
        
        # Use specific configuration
        python interactive_dataset_builder.py --profile direct_video_global --strategy dense_dense
        
        # Continue from saved progress
        python interactive_dataset_builder.py --load dataset_progress.json
    """
    builder = InteractiveDatasetBuilder(profile=profile, strategy=strategy)
    
    if load:
        with open(load, 'r') as f:
            builder.dataset_queries = json.load(f)
        click.echo(f"Loaded {len(builder.dataset_queries)} queries from {load}")
    
    builder.run_interactive_session()


if __name__ == '__main__':
    main()
