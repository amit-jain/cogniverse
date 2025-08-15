#!/usr/bin/env python3
"""
Comprehensive Video Query Test v2 - with better visual queries and metrics
"""

import json
import numpy as np
import sys
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tabulate import tabulate
import pandas as pd
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))
from src.tools.config import get_config
from src.search.search_service import SearchService


# Better visual queries that don't rely on text descriptions
VISUAL_TEST_QUERIES = [
    # Snow/Winter scenes (v_-IMXSEIabMM)
    {
        "query": "person wearing winter clothes outdoors in daylight",
        "expected_videos": ["v_-IMXSEIabMM"],
        "category": "action"
    },
    {
        "query": "white snow on ground with blue sky",
        "expected_videos": ["v_-IMXSEIabMM"],
        "category": "scene"
    },
    {
        "query": "person holding long tool or stick outdoors",
        "expected_videos": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo"],
        "category": "object"
    },
    
    # Elephant Dream (dark industrial/mechanical scenes)
    {
        "query": "industrial machinery with metallic surfaces",
        "expected_videos": ["elephant_dream_clip"],
        "category": "scene"
    },
    {
        "query": "dark indoor scene with artificial lighting",
        "expected_videos": ["elephant_dream_clip"],
        "category": "lighting"
    },
    {
        "query": "mechanical or robotic elements",
        "expected_videos": ["elephant_dream_clip"],
        "category": "object"
    },
    
    # For Bigger Blazes (device on table)
    {
        "query": "electronic device screen showing colorful content",
        "expected_videos": ["for_bigger_blazes"],
        "category": "object"
    },
    {
        "query": "wooden table or desk surface",
        "expected_videos": ["for_bigger_blazes"],
        "category": "scene"
    },
    {
        "query": "bright yellow or orange colors prominently displayed",
        "expected_videos": ["for_bigger_blazes"],
        "category": "color"
    },
    
    # Big Buck Bunny (animated nature scenes)
    {
        "query": "animated character or cartoon style visuals",
        "expected_videos": ["big_buck_bunny_clip"],
        "category": "style"
    },
    {
        "query": "outdoor nature scene with vegetation",
        "expected_videos": ["big_buck_bunny_clip"],
        "category": "scene"
    },
    {
        "query": "daytime outdoor scene with natural lighting",
        "expected_videos": ["big_buck_bunny_clip", "v_-IMXSEIabMM"],
        "category": "lighting"
    }
]


def calculate_metrics(results: List[Dict], expected_videos: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """Calculate MRR, NDCG, and Recall@k metrics"""
    metrics = {}
    
    # Get video IDs from results
    result_videos = [r.get('video_id', '') for r in results]
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, video in enumerate(result_videos):
        if video in expected_videos:
            mrr = 1.0 / (i + 1)
            break
    metrics['mrr'] = mrr
    
    # Recall@k
    for k in k_values:
        if k <= len(result_videos):
            recall_k = len(set(result_videos[:k]) & set(expected_videos)) / len(expected_videos)
            metrics[f'recall@{k}'] = recall_k
    
    # NDCG@k (Normalized Discounted Cumulative Gain)
    for k in k_values:
        if k <= len(result_videos):
            # Binary relevance: 1 if video is expected, 0 otherwise
            relevances = [1 if vid in expected_videos else 0 for vid in result_videos[:k]]
            
            # DCG
            dcg = relevances[0] if relevances else 0
            for i in range(1, len(relevances)):
                dcg += relevances[i] / np.log2(i + 2)
            
            # Ideal DCG (all relevant items at top)
            ideal_relevances = [1] * min(len(expected_videos), k) + [0] * max(0, k - len(expected_videos))
            idcg = ideal_relevances[0] if ideal_relevances else 0
            for i in range(1, len(ideal_relevances)):
                idcg += ideal_relevances[i] / np.log2(i + 2)
            
            # NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg@{k}'] = ndcg
    
    return metrics


def test_profile_with_queries(profile: str, queries: List[Dict], test_multiple_strategies: bool = False) -> Dict:
    """Test a profile with all queries and return detailed results"""
    
    # Get config and create search service
    config = get_config()
    
    try:
        search_service = SearchService(config, profile)
    except Exception as e:
        print(f"❌ Failed to create search service for {profile}: {e}")
        return {
            'profile': profile,
            'error': str(e)
        }
    
    # Determine strategies to test
    if test_multiple_strategies:
        if profile == 'video_colpali_smol500_mv_frame':
            strategies_to_test = [
                ('binary_binary', 'Visual Only'),
                ('hybrid_binary_bm25_no_description', 'Hybrid No Desc'),
                ('hybrid_binary_bm25', 'Hybrid + Desc'),
                ('bm25_only', 'Text Only')
            ]
        elif 'global' in profile:
            strategies_to_test = [
                ('binary_binary', 'Binary Visual'),
                ('float_binary', 'Float-Binary Hybrid'),
                ('phased', 'Phased (Binary→Float)'),
                ('float_float', 'Float Visual')
            ]
        else:
            strategies_to_test = [
                ('float_float', 'Float Visual'),
                ('float_binary', 'Float-Binary Hybrid'),
                ('binary_binary', 'Binary Visual'),
                ('phased', 'Phased (Binary→Float)')
            ]
    else:
        # Use default strategy for profile
        strategies_to_test = [(None, 'Default')]
    
    # Get model name from config
    config = get_config()
    profiles = config.get("video_processing_profiles", {})
    model_name = profiles.get(profile, {}).get("embedding_model", "Unknown")
    
    # Results storage
    profile_results = {
        'profile': profile,
        'model': model_name,
        'strategies': {} if test_multiple_strategies else None,
        'queries': [],
        'aggregate_metrics': {}
    }
    
    for strategy, strategy_name in strategies_to_test:
        strategy_results = {
            'name': strategy_name,
            'queries': [],
            'aggregate_metrics': {}
        }
        
        all_metrics = []
        
        for query_data in queries:
            query = query_data['query']
            expected_videos = query_data['expected_videos']
            
            try:
                # Execute search using SearchService with optional ranking strategy
                search_results = search_service.search(query, top_k=10, ranking_strategy=strategy)
                
                # Convert SearchResult objects to dicts and extract video IDs
                result_dicts = [r.to_dict() for r in search_results]
                results = []
                for r in result_dicts:
                    video_id = r.get('source_id', r['document_id'].split('_')[0])
                    results.append({
                        'video_id': video_id,
                        'score': r['score'],
                        'document_id': r['document_id']
                    })
                
                # Calculate metrics
                metrics = calculate_metrics(results, expected_videos)
                all_metrics.append(metrics)
                
                # Store query results
                query_result = {
                    'query': query,
                    'expected': expected_videos,
                    'results': [r['video_id'] for r in results[:5]],  # Top 5
                    'metrics': metrics,
                    'top_result_correct': results[0]['video_id'] in expected_videos if results else False
                }
                strategy_results['queries'].append(query_result)
                
            except Exception as e:
                error_msg = str(e)
                # Truncate long embedding arrays in error messages
                if 'query(qtb)' in error_msg and len(error_msg) > 500:
                    # Find the embedding array and truncate it
                    start_idx = error_msg.find('[')
                    end_idx = error_msg.find(']', start_idx)
                    if start_idx != -1 and end_idx != -1 and end_idx - start_idx > 100:
                        embedding_preview = error_msg[start_idx:start_idx+50] + "... (truncated) ..." + error_msg[end_idx-20:end_idx+1]
                        error_msg = error_msg[:start_idx] + embedding_preview + error_msg[end_idx+1:]
                
                print(f"❌ Search failed for query '{query}' with strategy '{strategy}': {error_msg}")
                query_result = {
                    'query': query,
                    'expected': expected_videos,
                    'results': [],
                    'metrics': {'recall@1': 0, 'recall@5': 0, 'recall@10': 0, 'mrr': 0},
                    'top_result_correct': False,
                    'error': error_msg
                }
                strategy_results['queries'].append(query_result)
        
        # Calculate aggregate metrics for this strategy
        if all_metrics:
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                strategy_results['aggregate_metrics'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        if test_multiple_strategies:
            profile_results['strategies'][strategy or 'default'] = strategy_results
        else:
            # For single strategy, put results at top level
            profile_results['queries'] = strategy_results['queries']
            profile_results['aggregate_metrics'] = strategy_results['aggregate_metrics']
    
    # For backward compatibility when testing multiple strategies
    if test_multiple_strategies and profile_results['strategies']:
        # Use the first strategy as default for top-level results
        first_strategy = list(profile_results['strategies'].values())[0]
        profile_results['queries'] = first_strategy['queries']
        profile_results['aggregate_metrics'] = first_strategy['aggregate_metrics']
    
    return profile_results


def get_best_strategy_for_profile(profile: str) -> str:
    """Get the best ranking strategy for a profile based on previous results"""
    best_strategies = {
        'video_colpali_smol500_mv_frame': 'binary_binary',  # Changed to visual-only for fair comparison
        'video_colqwen_omni_sv_chunk': 'hybrid_binary_bm25',
        'video_videoprism_base_sv_global': 'binary_binary',
        'video_videoprism_large_sv_global': 'binary_binary',
        'single__video_videoprism_large_6s': 'default'  # Use default for video_chunks
    }
    return best_strategies.get(profile, 'float_float')


def needs_text_for_strategy(strategy: str) -> bool:
    """Check if strategy needs text query"""
    return strategy in ['bm25_only', 'hybrid_float_bm25', 'hybrid_binary_bm25', 
                       'hybrid_bm25_float', 'hybrid_bm25_binary',
                       'hybrid_float_bm25_no_description', 'hybrid_binary_bm25_no_description',
                       'hybrid_bm25_float_no_description', 'hybrid_bm25_binary_no_description']


def needs_embeddings_for_strategy(strategy: str) -> bool:
    """Check if strategy needs embeddings"""
    return strategy not in ['bm25_only', 'bm25_no_description']


def create_comprehensive_results_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create the comprehensive table with all queries and results as requested"""
    
    rows = []
    
    for profile_results in all_results:
        profile = profile_results['profile']
        
        # Skip profiles that failed to initialize
        if 'error' in profile_results:
            continue
            
        for query_result in profile_results['queries']:
            query = query_result['query']
            expected = ', '.join(query_result['expected'])
            results = query_result['results']
            
            # Format results with checkmarks
            formatted_results = []
            for i, video in enumerate(results):
                if video in query_result['expected']:
                    formatted_results.append(f"✅ {video}")
                else:
                    formatted_results.append(f"❌ {video}")
            
            # Create result string
            result_str = ' > '.join(formatted_results[:3])  # Show top 3
            if len(results) > 3:
                result_str += ' ...'
            
            # Add metrics
            metrics = query_result['metrics']
            mrr = metrics.get('mrr', 0)
            recall_5 = metrics.get('recall@5', 0)
            
            rows.append({
                'Profile': profile,
                'Query': query[:50] + '...' if len(query) > 50 else query,
                'Expected': expected,
                'Results (Top 3)': result_str,
                'MRR': f"{mrr:.3f}",
                'R@5': f"{recall_5:.3f}"
            })
    
    return pd.DataFrame(rows)


def create_metrics_summary_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create summary table of metrics across profiles"""
    
    rows = []
    
    for profile_results in all_results:
        profile = profile_results['profile']
        
        # Skip profiles that failed to initialize
        if 'error' in profile_results:
            rows.append({
                'Profile': profile,
                'Model': 'FAILED',
                'MRR': 'ERROR',
                'RECALL@1': 'ERROR',
                'RECALL@5': 'ERROR',
                'RECALL@10': 'ERROR',
                'NDCG@5': 'ERROR',
                'NDCG@10': 'ERROR'
            })
            continue
            
        metrics = profile_results.get('aggregate_metrics', {})
        
        row = {
            'Profile': profile,
            'Model': profile_results.get('model', 'N/A')[:30] + '...' if profile_results.get('model', '') and len(profile_results.get('model', '')) > 30 else profile_results.get('model', 'N/A')
        }
        
        # Add mean metrics
        for metric_name in ['mrr', 'recall@1', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']:
            if metric_name in metrics:
                row[metric_name.upper()] = f"{metrics[metric_name]['mean']:.3f}"
            else:
                row[metric_name.upper()] = "N/A"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive video query test v2")
    parser.add_argument("--profiles", nargs="+", 
                       default=["video_colpali_smol500_mv_frame", "video_colqwen_omni_sv_chunk", 
                               "video_videoprism_base_sv_global", "video_videoprism_large_sv_global",
                               "single__video_videoprism_large_6s"],
                       help="Profiles to test")
    parser.add_argument("--output-format", choices=["table", "html", "csv"], default="table",
                       help="Output format")
    parser.add_argument("--test-multiple-strategies", action="store_true",
                       help="Test multiple ranking strategies for each profile")
    
    args = parser.parse_args()
    
    if args.test_multiple_strategies:
        print("📊 Testing multiple ranking strategies for each profile...")
        print("    This will help identify the optimal strategy for each model\n")
    
    # Get config
    config = get_config()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "outputs" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== COMPREHENSIVE VIDEO QUERY TEST v2 ===")
    print(f"Testing {len(args.profiles)} profiles with {len(VISUAL_TEST_QUERIES)} visual queries")
    print(f"Metrics: MRR, Recall@k, NDCG@k\n")
    
    all_results = []
    
    # Test each profile
    for profile in args.profiles:
        print(f"\n🔍 Testing profile: {profile}")
        
        try:
            results = test_profile_with_queries(profile, VISUAL_TEST_QUERIES, 
                                               test_multiple_strategies=args.test_multiple_strategies)
            all_results.append(results)
            print(f"✅ Completed {profile}")
        except Exception as e:
            print(f"❌ Error testing {profile}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comprehensive results table
    print("\n" + "="*120)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*120)
    
    results_df = create_comprehensive_results_table(all_results)
    
    if args.output_format == "table":
        print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
    elif args.output_format == "csv":
        csv_path = output_dir / f"comprehensive_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # Create metrics summary
    print("\n" + "="*120)
    print("METRICS SUMMARY")
    print("="*120)
    
    metrics_df = create_metrics_summary_table(all_results)
    print(tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False))
    
    # If testing multiple strategies, show comparison
    if args.test_multiple_strategies:
        print("\n" + "="*120)
        print("STRATEGY COMPARISON BY PROFILE")
        print("="*120)
        
        for result in all_results:
            if 'strategies' in result and len(result['strategies']) > 1:
                print(f"\n### {result['profile']} ({result['model']})")
                strategy_rows = []
                for strategy_key, strategy_data in result['strategies'].items():
                    metrics = strategy_data['aggregate_metrics']
                    strategy_rows.append({
                        'Strategy': strategy_data['name'],
                        'Ranking Profile': strategy_key,
                        'MRR': f"{metrics['mrr']['mean']:.3f}",
                        'Recall@1': f"{metrics.get('recall@1', {}).get('mean', 0):.3f}",
                        'Recall@5': f"{metrics.get('recall@5', {}).get('mean', 0):.3f}",
                        'NDCG@5': f"{metrics.get('ndcg@5', {}).get('mean', 0):.3f}"
                    })
                
                strategy_df = pd.DataFrame(strategy_rows)
                print(tabulate(strategy_df, headers='keys', tablefmt='grid', showindex=False))
                print()
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f"comprehensive_v2_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'queries': VISUAL_TEST_QUERIES,
            'results': all_results
        }, f, indent=2)
    print(f"\n📊 Detailed results saved to: {json_path}")


if __name__ == "__main__":
    main()
