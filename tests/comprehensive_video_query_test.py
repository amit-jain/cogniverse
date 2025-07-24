#!/usr/bin/env python3
"""
Comprehensive test using actual video descriptions and transcripts to query specific videos
Tests float vs binary search with content-specific queries
"""

import json
import numpy as np
import torch
import sys
import argparse
from pathlib import Path
from colpali_engine.models import ColIdefics3, ColIdefics3Processor

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))
from src.tools.config import get_config
from src.processing.vespa.vespa_search_client import VespaVideoSearchClient
from tests.test_utils import TestResultsFormatter

def execute_search_with_client(search_client, strategy, query, embeddings_np=None, silent=False):
    """Execute a search using the VespaVideoSearchClient"""
    try:
        # Prepare search params
        search_params = {
            "query": query if query else "",  # Empty string for pure visual search
            "ranking": strategy,
            "top_k": 10
        }
        
        # Execute search
        results = search_client.search(search_params, embeddings_np)
        
        # Convert to expected format
        formatted_results = []
        for i, result in enumerate(results[:10]):
            formatted_result = {
                'video_id': result.get('video_id'),
                'frame_id': result.get('frame_id'),
                'score': result.get('relevance', 0)
            }
            formatted_results.append(formatted_result)
            if not silent and i < 3:
                print(f"   {i+1}. Score: {result.get('relevance', 0):.4f}, Video: {result.get('video_id')}, Frame: {result.get('frame_id')}")
        
        return formatted_results
            
    except Exception as e:
        if not silent:
            print(f"   ❌ Exception: {e}")
        return None

def display_strategy_performance_table(strategy_performance, formatter):
    """Display strategy performance using the formatter's ranking strategy table method"""
    print(formatter.format_ranking_strategy_table(strategy_performance))


def analyze_strategy_failures(strategy_performance, strategies_to_analyze=None):
    """Analyze failures for specific strategies with detailed insights"""
    
    if strategies_to_analyze is None:
        # Default to analyzing high-performing visual strategies that had failures
        strategies_to_analyze = ['binary_binary', 'phased']
    
    print("\n" + "=" * 120)
    print("DETAILED FAILURE ANALYSIS")
    print("=" * 120)
    
    for strategy in strategies_to_analyze:
        if strategy not in strategy_performance:
            continue
            
        results = strategy_performance[strategy]
        failures = [r for r in results if not r['expected_found']]
        
        if not failures:
            continue
            
        total = len(results)
        accuracy = (len(results) - len(failures)) / total * 100
        
        print(f"\n{strategy.upper()} FAILURES ({accuracy:.1f}% accuracy - {len(failures)} failures out of {total}):")
        print("-" * 80)
        
        # Group failures by video
        video_failures = {}
        for failure in failures:
            video = failure['expected_video']
            if video not in video_failures:
                video_failures[video] = []
            video_failures[video].append(failure)
        
        # Analyze each failed query
        for video, video_fails in video_failures.items():
            print(f"\nExpected video: {video}")
            
            # Get successful queries for comparison
            successes = [r for r in results if r['expected_video'] == video and r['expected_found']]
            
            print(f"  Success rate: {len(successes)}/{len(successes) + len(video_fails)} queries")
            
            print("\n  Failed queries:")
            for fail in video_fails:
                query_text = fail['query_key'].split('_', 2)[-1]
                print(f"    - '{query_text}'")
                print(f"      Got: {fail['top_video']} (score: {fail['top_score']:.4f})")
                if fail['expected_rank']:
                    print(f"      Expected video rank: {fail['expected_rank']}")
                
            if successes:
                print("\n  Successful queries for comparison:")
                for success in successes:
                    query_text = success['query_key'].split('_', 2)[-1]
                    print(f"    - '{query_text}' (score: {success['top_score']:.4f})")
        
        # Pattern analysis
        print(f"\n  Failure patterns for {strategy}:")
        failed_queries = [f['query_key'].split('_', 2)[-1] for f in failures]
        successful_queries = [r['query_key'].split('_', 2)[-1] for r in results if r['expected_found']]
        
        # Check for common patterns
        action_words = ['ing', 'clear', 'show', 'display']
        abstract_patterns = ['pattern', 'aged', 'worn']
        
        failed_actions = sum(1 for q in failed_queries if any(w in q for w in action_words))
        failed_abstract = sum(1 for q in failed_queries if any(w in q for w in abstract_patterns))
        
        if failed_actions > 0:
            print(f"    - {failed_actions}/{len(failures)} failures contain action words")
        if failed_abstract > 0:
            print(f"    - {failed_abstract}/{len(failures)} failures contain abstract descriptors")
        
        # Score analysis
        if failures:
            avg_fail_score = sum(f['top_score'] for f in failures) / len(failures)
            if successes:
                avg_success_score = sum(s['top_score'] for s in successes) / len(successes)
                print(f"    - Average failure score: {avg_fail_score:.4f}")
                print(f"    - Average success score: {avg_success_score:.4f}")
                print(f"    - Score difference: {avg_fail_score - avg_success_score:.4f}")


def display_cross_strategy_comparison(strategy_performance, query_filter=None):
    """Show how different strategies performed on the same queries"""
    
    print("\n" + "=" * 120)
    print("CROSS-STRATEGY COMPARISON FOR DIFFICULT QUERIES")
    print("=" * 120)
    
    # Collect all unique queries
    all_queries = set()
    for strategy, results in strategy_performance.items():
        for result in results:
            all_queries.add((result['query_key'], result['expected_video']))
    
    # Find queries that failed in at least one strategy
    difficult_queries = []
    for query_key, expected_video in all_queries:
        failures = 0
        for strategy, results in strategy_performance.items():
            for result in results:
                if result['query_key'] == query_key and not result['expected_found']:
                    failures += 1
                    break
        if failures > 0:
            difficult_queries.append((query_key, expected_video))
    
    # Sort by number of strategies that failed
    difficult_queries.sort(key=lambda x: sum(
        1 for s, r in strategy_performance.items() 
        for res in r if res['query_key'] == x[0] and not res['expected_found']
    ), reverse=True)
    
    # Show top difficult queries
    for query_key, expected_video in difficult_queries[:5]:  # Top 5 most difficult
        query_text = query_key.split('_', 2)[-1]
        print(f"\nQuery: '{query_text}' (expecting {expected_video})")
        print("-" * 80)
        
        # Show each strategy's performance
        results_by_strategy = []
        for strategy, results in strategy_performance.items():
            for result in results:
                if result['query_key'] == query_key:
                    results_by_strategy.append((strategy, result))
                    break
        
        # Sort by success then score
        results_by_strategy.sort(key=lambda x: (-x[1]['expected_found'], -x[1]['top_score']))
        
        print("Strategy Performance:")
        for strategy, result in results_by_strategy:
            status = "✅" if result['expected_found'] else "❌"
            rank_info = f"Rank: {result['expected_rank']}" if result['expected_rank'] else "Not in top 10"
            print(f"  {strategy:35s}: {status} Top: {result['top_video']:20s} Score: {result['top_score']:8.4f} {rank_info}")

def test_all_ranking_strategies(search_client, query, video_id, embeddings_np, vespa_schema="video_frame"):
    """Test all ranking strategies and return performance data"""
    
    # Define all strategies
    text_only = ["bm25_only", "bm25_no_description"]
    visual_only = ["float_float", "binary_binary", "float_binary", "phased"]
    hybrid = ["hybrid_float_bm25", "hybrid_binary_bm25", "hybrid_bm25_binary", "hybrid_bm25_float",
              "hybrid_float_bm25_no_description", "hybrid_binary_bm25_no_description",
              "hybrid_bm25_binary_no_description", "hybrid_bm25_float_no_description"]
    
    all_strategies = text_only + visual_only + hybrid
    strategy_results = {}
    
    for strategy in all_strategies:
        # Determine if we need text query and/or embeddings
        needs_text = strategy in text_only + hybrid
        needs_embeddings = strategy in visual_only + hybrid
        
        # Execute search
        results = execute_search_with_client(
            search_client,
            strategy,
            query if needs_text else "",
            embeddings_np if needs_embeddings else None,
            silent=True
        )
        
        if results:
            # Check if expected video is in top results
            expected_in_top = any(r['video_id'] == video_id for r in results[:3])
            expected_rank = next((i+1 for i, r in enumerate(results) if r['video_id'] == video_id), None)
            
            strategy_results[strategy] = {
                'top_score': results[0]['score'] if results else 0,
                'top_video': results[0]['video_id'] if results else "N/A",
                'results_count': len(results),
                'expected_found': expected_in_top,
                'expected_rank': expected_rank,
                'top_3_videos': [r['video_id'] for r in results[:3]]
            }
        else:
            strategy_results[strategy] = {
                'top_score': 0,
                'top_video': "ERROR",
                'results_count': 0,
                'expected_found': False,
                'expected_rank': None,
                'top_3_videos': []
            }
    
    return strategy_results

def analyze_results(expected_video, query, binary_results, float_results, output_format="text"):
    """Analyze and compare binary vs float results"""
    
    analysis = {}
    
    if not binary_results or not float_results:
        print(f"   ⚠️ Missing results - cannot compare")
        return
    
    # Get top video IDs
    binary_videos = [r['video_id'] for r in binary_results]
    float_videos = [r['video_id'] for r in float_results]
    
    binary_top = binary_videos[0] if binary_videos else None
    float_top = float_videos[0] if float_videos else None
    
    print(f"\n📊 ANALYSIS:")
    print(f"   Expected video: {expected_video}")
    print(f"   Binary top result: {binary_top}")
    print(f"   Float top result: {float_top}")
    
    # Check if expected video appears in results
    binary_has_expected = expected_video in binary_videos
    float_has_expected = expected_video in float_videos
    
    print(f"   Binary finds expected: {'✅' if binary_has_expected else '❌'}")
    print(f"   Float finds expected: {'✅' if float_has_expected else '❌'}")
    
    # Overall assessment
    if binary_top == float_top:
        print(f"   ✅ CONSISTENT: Both return same top video")
        analysis['consistent'] = True
    else:
        print(f"   ❌ INCONSISTENT: Different top videos")
        analysis['consistent'] = False
        
    if binary_has_expected and float_has_expected:
        analysis['match_expected'] = "Both ✅"
    elif binary_has_expected and not float_has_expected:
        print(f"   🚨 FLOAT ISSUE: Binary finds expected video, float doesn't")
        analysis['match_expected'] = "Binary only"
    elif float_has_expected and not binary_has_expected:
        print(f"   🚨 BINARY ISSUE: Float finds expected video, binary doesn't")
        analysis['match_expected'] = "Float only"
    else:
        print(f"   ⚠️ BOTH MISS: Neither finds the expected video")
        analysis['match_expected'] = "Neither ❌"
    
    return analysis


def comprehensive_video_test(custom_query=None, output_format="table", test_all_strategies=False):
    """Test using actual video content to create targeted queries"""
    
    print("=== COMPREHENSIVE VIDEO CONTENT TEST ===")
    
    # Initialize results formatter
    formatter = TestResultsFormatter("comprehensive_video_query")
    all_results = []
    strategy_performance = {} if test_all_strategies else None
    
    # Load config
    config = get_config()
    vespa_schema = config.get("vespa_schema", "video_frame")
    model_name = config.get("colpali_model", "vidore/colsmol-500m")
    vespa_url = config.get("vespa_url", "http://localhost")
    vespa_port = config.get("vespa_port", 8080)
    
    # Use same device detection as video agent
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Loading ColPali model: {model_name}")
    print(f"Device: {device}, dtype: {dtype}")
    
    col_model = ColIdefics3.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    ).eval()
    
    col_processor = ColIdefics3Processor.from_pretrained(model_name)
    
    # Initialize search client
    search_client = VespaVideoSearchClient(vespa_url=vespa_url, vespa_port=vespa_port)
    print(f"✅ Initialized search client for {vespa_url}:{vespa_port}")
    
    # Define test cases based on actual video content (excluding Big Buck Bunny to avoid bias)
    test_cases = [
        {
            "video_id": "v_-IMXSEIabMM",
            "description": "Snow removal/winter scene",
            "queries": [
                "person shoveling snow in winter",
                "snow covered driveway", 
                "winter storm snow removal",
                "shovel clearing snowy ground"
            ]
        },
        {
            "video_id": "elephant_dream_clip",
            "description": "Dark tiles/wall scene",
            "queries": [
                "dark textured wall with tiles",
                "cracked stone wall surface", 
                "worn brown tiles on wall",
                "aged tile pattern with cracks"
            ]
        },
        {
            "video_id": "for_bigger_blazes",
            "description": "Tablet/phone showing flower on wooden table",
            "queries": [
                "smartphone on wooden table",
                "tablet displaying yellow flower",
                "phone screen showing sunflower",
                "green cup on wooden surface"
            ]
        }
    ]
    
    # Test each video with its specific content
    for test_case in test_cases:
        video_id = test_case["video_id"]
        description = test_case["description"]
        queries = test_case["queries"]
        
        print(f"\n{'='*80}")
        print(f"TESTING VIDEO: {video_id}")
        print(f"DESCRIPTION: {description}")
        print(f"{'='*80}")
        
        # Test each query for this video
        for query_idx, query in enumerate(queries):
            print(f"\n--- Query {query_idx + 1}: '{query}' ---")
            
            # Generate embeddings
            batch_queries = col_processor.process_queries([query]).to(device)
            with torch.no_grad():
                query_embeddings = col_model(**batch_queries)
            
            # Convert to numpy array (keep as numpy for search client)
            embeddings_np = query_embeddings.cpu().numpy().squeeze(0)
            print(f"Query embedding shape: {embeddings_np.shape}")
            
            if test_all_strategies:
                # Test all ranking strategies
                print(f"🔍 Testing all ranking strategies...")
                strategy_results = test_all_ranking_strategies(search_client, query, video_id, embeddings_np, vespa_schema)
                
                # Store strategy performance
                for strategy, results in strategy_results.items():
                    key = f"{video_id}_{query[:20]}"
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = []
                    strategy_performance[strategy].append({
                        'query_key': key,
                        'expected_video': video_id,
                        **results
                    })
            else:
                # Original binary/float comparison
                # Test binary search
                print(f"🔢 BINARY SEARCH:")
                binary_results = execute_search_with_client(
                    search_client,
                    "binary_binary",
                    "",  # No text query for pure visual search
                    embeddings_np
                )
                
                # Test float search
                print(f"🔍 FLOAT SEARCH:")
                float_results = execute_search_with_client(
                    search_client,
                    "float_float",
                    "",  # No text query for pure visual search
                    embeddings_np
                )
                
                # Analyze results
                analysis = analyze_results(video_id, query, binary_results, float_results, output_format)
                
                # Collect results for final summary
                if binary_results and float_results:
                    all_results.append({
                        "Video": video_id,
                        "Query": query[:40] + "...",
                        "Binary Top": binary_results[0]['video_id'] if binary_results else "N/A",
                        "Binary Score": f"{binary_results[0]['score']:.4f}" if binary_results else "N/A",
                        "Float Top": float_results[0]['video_id'] if float_results else "N/A",
                        "Float Score": f"{float_results[0]['score']:.4f}" if float_results else "N/A",
                        "Match Expected": analysis.get('match_expected', 'N/A')
                    })
    
    # Display results based on mode
    if test_all_strategies and strategy_performance:
        display_strategy_performance_table(strategy_performance, formatter)
        
        # Add failure analysis for high-performing strategies
        analyze_strategy_failures(strategy_performance, ['binary_binary', 'phased', 'hybrid_bm25_binary', 'hybrid_bm25_float'])
        
        # Add cross-strategy comparison for difficult queries
        display_cross_strategy_comparison(strategy_performance)
        
        # Convert strategy_performance to all_results for saving
        for strategy, results_list in strategy_performance.items():
            for result in results_list:
                all_results.append({
                    "strategy": strategy,
                    "query": result.get('query_key', ''),
                    "expected_video": result.get('expected_video', ''),
                    "top_video": result.get('top_video', ''),
                    "top_score": result.get('top_score', 0),
                    "expected_found": result.get('expected_found', False),
                    "expected_rank": result.get('expected_rank', None)
                })
    elif all_results and output_format == "table":
        # Original summary table for binary/float comparison
        print("\n" + "=" * 100)
        print("SUMMARY TABLE - All Query Results")
        print("=" * 100)
        print(formatter.format_table(all_results, tablefmt="grid"))
        
        # Calculate statistics
        total_queries = len(all_results)
        consistent_count = sum(1 for r in all_results if "Both ✅" in r.get("Match Expected", ""))
        binary_only = sum(1 for r in all_results if "Binary only" in r.get("Match Expected", ""))
        float_only = sum(1 for r in all_results if "Float only" in r.get("Match Expected", ""))
        neither = sum(1 for r in all_results if "Neither ❌" in r.get("Match Expected", ""))
        
        stats = {
            "Total Queries": total_queries,
            "Both Found Expected": f"{consistent_count} ({consistent_count/total_queries*100:.1f}%)",
            "Binary Only": f"{binary_only} ({binary_only/total_queries*100:.1f}%)",
            "Float Only": f"{float_only} ({float_only/total_queries*100:.1f}%)",
            "Neither Found": f"{neither} ({neither/total_queries*100:.1f}%)"
        }
        
        formatter.print_summary(all_results, stats)
        
    # Always save results automatically
    if all_results:
        csv_path = formatter.save_csv(all_results)
        json_path = formatter.save_json(all_results)
        print(f"\n📊 Results saved to:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive video query test")
    parser.add_argument("query", nargs="?", default=None,
                       help="Optional custom query (uses pre-defined queries if not provided)")
    parser.add_argument("--format", choices=["table", "text"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--all-strategies", action="store_true",
                       help="Test all ranking strategies instead of just binary/float")
    
    args = parser.parse_args()
    
    config = get_config()
    output_format = config.get("test_output_format", args.format)
    
    comprehensive_video_test(custom_query=args.query,
                           output_format=output_format, 
                           test_all_strategies=args.all_strategies)