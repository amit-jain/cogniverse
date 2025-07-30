#!/usr/bin/env python3
"""
Analyze comprehensive test results and create detailed tables
"""

import json
import sys
from pathlib import Path
from tabulate import tabulate
import pandas as pd


def load_all_results():
    """Load results from all profile test runs"""
    results_dir = Path("outputs/test_results")
    
    # Find all comprehensive test result files
    result_files = sorted(results_dir.glob("comprehensive_video_query_*.json"))
    
    if not result_files:
        print("No result files found!")
        return None
        
    # Load the most recent file
    latest_file = result_files[-1]
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
        
    return data


def extract_query_info(query_key):
    """Extract clean query text and video from query key"""
    parts = query_key.split('_', 2)
    if len(parts) >= 3:
        video_id = '_'.join(parts[:2]) if parts[0] not in ['elephant', 'for'] else parts[0] + '_' + parts[1]
        query_text = parts[2]
    else:
        video_id = parts[0]
        query_text = parts[1] if len(parts) > 1 else "unknown"
    return video_id, query_text


def create_summary_table(results):
    """Create a summary table of all queries and their results across strategies"""
    # Group results by query
    queries = {}
    
    for result in results:
        query_key = result['query']
        strategy = result['strategy']
        
        if query_key not in queries:
            queries[query_key] = {
                'expected_video': result['expected_video'],
                'strategies': {}
            }
            
        queries[query_key]['strategies'][strategy] = {
            'top_video': result['top_video'],
            'score': result['top_score'],
            'found': result['expected_found'],
            'rank': result['expected_rank']
        }
    
    # Create table rows
    rows = []
    for query_key, query_data in sorted(queries.items()):
        _, query_text = extract_query_info(query_key)
        expected = query_data['expected_video']
        
        row = {
            'Query': query_text[:30] + '...' if len(query_text) > 30 else query_text,
            'Expected': expected
        }
        
        # Add results for each strategy
        for strategy in ['bm25_only', 'float_float', 'binary_binary', 'phased', 
                        'hybrid_float_bm25', 'hybrid_binary_bm25']:
            if strategy in query_data['strategies']:
                strat_result = query_data['strategies'][strategy]
                if strat_result['found']:
                    status = f"✅ (rank {strat_result['rank']})"
                else:
                    status = f"❌ {strat_result['top_video'][:10]}..."
                row[strategy] = status
            else:
                row[strategy] = "N/A"
                
        rows.append(row)
        
    return rows


def create_profile_comparison_table(all_data):
    """Create a table comparing performance across different profiles"""
    # This would require loading multiple result files for different profiles
    # For now, we'll analyze the single file which contains results for one profile
    
    results = all_data['results']
    
    # Count successes by strategy
    strategy_stats = {}
    
    for result in results:
        strategy = result['strategy']
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                'total': 0,
                'found': 0,
                'scores': []
            }
            
        strategy_stats[strategy]['total'] += 1
        if result['expected_found']:
            strategy_stats[strategy]['found'] += 1
        strategy_stats[strategy]['scores'].append(result['top_score'])
    
    # Create summary rows
    rows = []
    for strategy, stats in sorted(strategy_stats.items()):
        accuracy = stats['found'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
        
        rows.append({
            'Strategy': strategy,
            'Accuracy': f"{accuracy:.1f}%",
            'Found': f"{stats['found']}/{stats['total']}",
            'Avg Score': f"{avg_score:.2f}"
        })
        
    return rows


def create_detailed_results_csv(all_data):
    """Create a detailed CSV with all results"""
    results = all_data['results']
    
    # Prepare data for DataFrame
    rows = []
    for result in results:
        _, query_text = extract_query_info(result['query'])
        
        row = {
            'Query': query_text,
            'Expected_Video': result['expected_video'],
            'Strategy': result['strategy'],
            'Top_Result': result['top_video'],
            'Score': result['top_score'],
            'Found': 'Yes' if result['expected_found'] else 'No',
            'Rank': result['expected_rank'] if result['expected_rank'] else 'Not in top 10'
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = Path("outputs/test_results/comprehensive_analysis.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return df


def main():
    # Load results
    all_data = load_all_results()
    if not all_data:
        return
        
    results = all_data['results']
    
    # Create summary table
    print("\n" + "="*120)
    print("QUERY RESULTS ACROSS ALL STRATEGIES")
    print("="*120)
    
    summary_rows = create_summary_table(results)
    print(tabulate(summary_rows, headers="keys", tablefmt="grid"))
    
    # Create strategy comparison
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    
    comparison_rows = create_profile_comparison_table(all_data)
    print(tabulate(comparison_rows, headers="keys", tablefmt="grid"))
    
    # Create detailed CSV
    df = create_detailed_results_csv(all_data)
    
    # Print some interesting statistics
    print("\n" + "="*80)
    print("ADDITIONAL STATISTICS")
    print("="*80)
    
    # Videos that are hardest to find
    video_success_rates = {}
    for result in results:
        video = result['expected_video']
        if video not in video_success_rates:
            video_success_rates[video] = {'found': 0, 'total': 0}
        video_success_rates[video]['total'] += 1
        if result['expected_found']:
            video_success_rates[video]['found'] += 1
    
    print("\nVideo Success Rates:")
    for video, stats in sorted(video_success_rates.items()):
        rate = stats['found'] / stats['total'] * 100
        print(f"  {video}: {rate:.1f}% ({stats['found']}/{stats['total']})")
    
    # Queries that fail most often
    query_failure_counts = {}
    for result in results:
        if not result['expected_found']:
            query = result['query']
            query_failure_counts[query] = query_failure_counts.get(query, 0) + 1
    
    print("\nMost Difficult Queries (failure count):")
    for query, count in sorted(query_failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        _, query_text = extract_query_info(query)
        print(f"  '{query_text}': {count} failures")


if __name__ == "__main__":
    main()
