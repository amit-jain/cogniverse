#!/usr/bin/env python3
"""
Extract VideoPrism results for elephant dream queries
"""

import json
from pathlib import Path
from tabulate import tabulate

def extract_videoprism_results():
    # Load the latest test results
    results_file = Path("outputs/test_results/comprehensive_video_query_20250730_102143.json")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Filter for elephant dream queries and VideoPrism strategies
    elephant_queries = [
        "clip_dark textured wall w",
        "clip_cracked stone wall s", 
        "clip_worn brown tiles on ",
        "clip_aged tile pattern wi"
    ]
    
    videoprism_strategies = ['binary_binary', 'float_float', 'phased']
    
    # Extract results
    videoprism_results = {}
    
    for result in results:
        query = result['query']
        strategy = result['strategy']
        
        # Check if it's an elephant dream query
        query_text = query.split('_', 2)[-1] if '_' in query else query
        
        if any(eq in query_text for eq in elephant_queries) and strategy in videoprism_strategies:
            if query_text not in videoprism_results:
                videoprism_results[query_text] = {}
            
            videoprism_results[query_text][strategy] = {
                'top_video': result['top_video'],
                'score': result['top_score'],
                'rank': result.get('expected_rank', 'Not in top 10')
            }
    
    # Print results
    print("\n=== VideoPrism Results for Elephant Dream Queries ===\n")
    
    for query, strategies in videoprism_results.items():
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        rows = []
        for strategy, data in sorted(strategies.items()):
            rows.append([
                strategy,
                data['top_video'],
                f"{data['score']:.4f}",
                data['rank'] if data['rank'] else "Not in top 10"
            ])
        
        print(tabulate(rows, headers=['Strategy', 'Top Video', 'Score', 'Elephant Dream Rank'], tablefmt='grid'))
    
    # Now let's check what videos VideoPrism returned and see their descriptions
    returned_videos = set()
    for query_results in videoprism_results.values():
        for result in query_results.values():
            if result['top_video'] != 'elephant_dream_clip':
                returned_videos.add(result['top_video'])
    
    print("\n\n=== Videos that VideoPrism returned instead of elephant_dream ===")
    for video in sorted(returned_videos):
        print(f"\n{video}:")
        
        # Try to find descriptions for this video
        desc_file = Path(f"outputs/processing/profile_frame_based_colpali/descriptions/{video}.json")
        if desc_file.exists():
            with open(desc_file, 'r') as f:
                descriptions = json.load(f)
                # Show first few frame descriptions
                for i, (frame, desc) in enumerate(list(descriptions.items())[:3]):
                    print(f"  Frame {frame}: {desc[:150]}...")
        else:
            print("  No descriptions found")

if __name__ == "__main__":
    extract_videoprism_results()