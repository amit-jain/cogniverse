#!/usr/bin/env python3
"""
Test script for updated Vespa search client with different ranking profiles
Tests both with and without embeddings to validate input requirements
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np
import json
import csv
from datetime import datetime
from collections import defaultdict

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from src.app.search.service import SearchService
from src.backends.vespa.vespa_search_client import VespaVideoSearchClient
from src.common.config_utils import get_config

# Setup logging with more verbose output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def create_search_services(config):
    """Create search services for different profiles"""
    services = {}
    
    # Define profiles to test
    profiles = [
        "frame_based_colpali",
        "colqwen_chunks",
        "direct_video_frame",
        "direct_video_global"
    ]
    
    for profile in profiles:
        try:
            services[profile] = SearchService(config, profile)
            logger.info(f"✅ Created search service for profile: {profile}")
        except Exception as e:
            logger.error(f"❌ Failed to create service for {profile}: {e}")
    
    return services

def analyze_ranking_results(results_df, strategy_performance):
    """Analyze ranking results to find patterns and insights"""
    
    print("\n" + "=" * 80)
    print("RANKING STRATEGY ANALYSIS")
    print("=" * 80)
    
    # 1. Score distribution analysis
    print("\n1. SCORE DISTRIBUTION BY STRATEGY:")
    print("-" * 50)
    
    # Group results by strategy and show score stats
    for strategy, perf in strategy_performance.items():
        if 'error' not in perf and perf['results_count'] > 0:
            strategy_results = [r for r in results_df if r['strategy'] == strategy]
            if strategy_results:
                scores = [r['score'] for r in strategy_results[:10]]  # Top 10
                print(f"\n{strategy}:")
                print(f"  Top score: {perf['top_score']:.4f}")
                print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
                print(f"  Top videos: {', '.join(perf['top_videos'][:3])}")
    
    # 2. Video dominance analysis
    print("\n\n2. VIDEO DOMINANCE ANALYSIS:")
    print("-" * 50)
    
    video_counts = defaultdict(int)
    video_strategies = defaultdict(set)
    
    for result in results_df:
        if result['rank'] <= 3:  # Top 3 only
            video_counts[result['video_id']] += 1
            video_strategies[result['video_id']].add(result['strategy'])
    
    # Sort by frequency
    sorted_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost frequently appearing videos in top 3:")
    for video, count in sorted_videos[:5]:
        strategies = video_strategies[video]
        print(f"  {video}: {count} times across {len(strategies)} strategies")
        print(f"    Strategies: {', '.join(sorted(strategies))}")
    
    # 3. Strategy grouping by results
    print("\n\n3. STRATEGY SIMILARITY ANALYSIS:")
    print("-" * 50)
    
    # Find strategies with identical top videos
    strategy_signatures = {}
    for strategy, perf in strategy_performance.items():
        if 'error' not in perf and perf['results_count'] > 0:
            signature = tuple(perf['top_videos'][:3])
            if signature not in strategy_signatures:
                strategy_signatures[signature] = []
            strategy_signatures[signature].append(strategy)
    
    print("\nStrategies with identical top 3 results:")
    for signature, strategies in strategy_signatures.items():
        if len(strategies) > 1:
            print(f"  {', '.join(strategies)}")
            print(f"    Top videos: {', '.join(signature)}")
    
    # 4. Performance tiers
    print("\n\n4. PERFORMANCE TIERS (by top score):")
    print("-" * 50)
    
    # Sort strategies by top score
    sorted_strategies = sorted(
        [(s, p) for s, p in strategy_performance.items() if 'error' not in p],
        key=lambda x: x[1]['top_score'],
        reverse=True
    )
    
    # Group into tiers
    if sorted_strategies:
        max_score = sorted_strategies[0][1]['top_score']
        
        print("\nHigh performers (>50% of max score):")
        for strategy, perf in sorted_strategies:
            if perf['top_score'] > max_score * 0.5:
                print(f"  {strategy}: {perf['top_score']:.4f}")
        
        print("\nMedium performers (10-50% of max score):")
        for strategy, perf in sorted_strategies:
            if max_score * 0.1 < perf['top_score'] <= max_score * 0.5:
                print(f"  {strategy}: {perf['top_score']:.4f}")
        
        print("\nLow performers (<10% of max score):")
        for strategy, perf in sorted_strategies:
            if perf['top_score'] <= max_score * 0.1:
                print(f"  {strategy}: {perf['top_score']:.4f}")


def test_ranking_strategies(query=None, table_output=False, show_analysis=True):
    """Test all ranking strategies with appropriate inputs"""
    
    print("\n--- STARTING test_ranking_strategies ---")
    
    # Always save results to test_results directory
    from src.common.utils.output_manager import get_output_manager
    from datetime import datetime
    output_manager = get_output_manager()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    auto_csv_path = output_manager.get_test_results_dir() / f"search_client_test_{timestamp}.csv"
    
    # Initialize client
    print("Initializing VespaVideoSearchClient...")
    client = VespaVideoSearchClient()
    
    # Check health
    print("Checking Vespa health...")
    if not client.health_check():
        logger.error("❌ Vespa health check failed")
        print("ERROR: Vespa health check failed")
        return
    
    logger.info("✅ Vespa connection healthy")
    print("✅ Vespa connection healthy")
    
    # Test query - use provided query or default
    test_query = query if query else "doctor explaining medical procedures"
    logger.info(f"Using test query: '{test_query}'")
    print(f"Using test query: '{test_query}'")
    
    # Store all results for table output
    all_results = []
    strategy_performance = {}
    
    # Test strategies that only need text
    text_only_strategies = [
        RankingStrategy.BM25_ONLY.value,
        RankingStrategy.BM25_NO_DESCRIPTION.value
    ]
    
    print(f"\n=== Testing Text-Only Strategies ===")
    logger.info(f"\n=== Testing Text-Only Strategies ===")
    for strategy in text_only_strategies:
        if not table_output:
            print(f"\n🔍 Testing {strategy}")
            logger.info(f"\n🔍 Testing {strategy}")
        try:
            results = client.search({
                "query": test_query,
                "ranking": strategy,
                "top_k": 10
            })
            
            # Store results for table output
            strategy_performance[strategy] = {
                'top_score': results[0]['relevance'] if results else 0,
                'results_count': len(results),
                'top_videos': [r['video_id'] for r in results[:3]]
            }
            
            # Store detailed results
            for result in results:
                all_results.append({
                    'strategy': strategy,
                    'rank': len([r for r in all_results if r['strategy'] == strategy]) + 1,
                    'video_id': result['video_id'],
                    'frame_id': result.get('frame_id', 'N/A'),
                    'score': result['relevance'],
                    'timestamp': result.get('start_time', 'N/A')
                })
            
            if not table_output:
                print(f"✅ {strategy}: Got {len(results)} results")
                logger.info(f"✅ {strategy}: Got {len(results)} results")
                for i, result in enumerate(results[:2]):
                    score_line = f"  {i+1}. Score: {result['relevance']:.4f}, Video: {result['video_id']}"
                    print(score_line)
                    logger.info(score_line)
        except Exception as e:
            strategy_performance[strategy] = {
                'top_score': 0,
                'results_count': 0,
                'top_videos': [],
                'error': str(e)
            }
            if not table_output:
                error_line = f"❌ {strategy}: {e}"
                print(error_line)  
                logger.error(error_line)
    
    # Load ColPali model for visual strategies
    print(f"\n=== Loading ColPali Model for Visual Strategies ===")
    logger.info(f"\n=== Loading ColPali Model for Visual Strategies ===")
    col_model, col_processor, device = load_colpali_model()
    print("ColPali model loaded, generating embeddings...")
    embeddings = generate_test_embeddings(col_model, col_processor, device, test_query)
    print(f"Embeddings generated with shape: {embeddings.shape}")
    
    # Test pure visual strategies
    visual_strategies = [
        RankingStrategy.FLOAT_FLOAT.value,
        RankingStrategy.BINARY_BINARY.value,
        RankingStrategy.FLOAT_BINARY.value,
        RankingStrategy.PHASED.value
    ]
    
    print(f"\n=== Testing Pure Visual Strategies ===")
    logger.info(f"\n=== Testing Pure Visual Strategies ===")
    for strategy in visual_strategies:
        if not table_output:
            print(f"\n🔍 Testing {strategy}")
            logger.info(f"\n🔍 Testing {strategy}")
        try:
            results = client.search({
                "query": "",  # Empty query for pure visual
                "ranking": strategy,
                "top_k": 10
            }, embeddings=embeddings)
            
            # Store results for table output
            strategy_performance[strategy] = {
                'top_score': results[0]['relevance'] if results else 0,
                'results_count': len(results),
                'top_videos': [r['video_id'] for r in results[:3]]
            }
            
            # Store detailed results
            for result in results:
                all_results.append({
                    'strategy': strategy,
                    'rank': len([r for r in all_results if r['strategy'] == strategy]) + 1,
                    'video_id': result['video_id'],
                    'frame_id': result.get('frame_id', 'N/A'),
                    'score': result['relevance'],
                    'timestamp': result.get('start_time', 'N/A')
                })
            
            if not table_output:
                print(f"✅ {strategy}: Got {len(results)} results")
                logger.info(f"✅ {strategy}: Got {len(results)} results")
                for i, result in enumerate(results[:2]):
                    score_line = f"  {i+1}. Score: {result['relevance']:.4f}, Video: {result['video_id']}"
                    print(score_line)
                    logger.info(score_line)
        except Exception as e:
            strategy_performance[strategy] = {
                'top_score': 0,
                'results_count': 0,
                'top_videos': [],
                'error': str(e)
            }
            if not table_output:
                error_line = f"❌ {strategy}: {e}"
                print(error_line)
                logger.error(error_line)
    
    # Test hybrid strategies
    hybrid_strategies = [
        RankingStrategy.HYBRID_FLOAT_BM25.value,
        RankingStrategy.HYBRID_BINARY_BM25.value,
        RankingStrategy.HYBRID_BM25_BINARY.value,
        RankingStrategy.HYBRID_BM25_FLOAT.value,
        RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC.value,
        RankingStrategy.HYBRID_BINARY_BM25_NO_DESC.value,
        RankingStrategy.HYBRID_BM25_BINARY_NO_DESC.value,
        RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC.value
    ]
    
    print(f"\n=== Testing Hybrid Strategies ===")
    logger.info(f"\n=== Testing Hybrid Strategies ===")
    for strategy in hybrid_strategies:
        if not table_output:
            print(f"\n🔍 Testing {strategy}")
            logger.info(f"\n🔍 Testing {strategy}")
        try:
            results = client.search({
                "query": test_query,
                "ranking": strategy,
                "top_k": 10
            }, embeddings=embeddings)
            
            # Store results for table output
            strategy_performance[strategy] = {
                'top_score': results[0]['relevance'] if results else 0,
                'results_count': len(results),
                'top_videos': [r['video_id'] for r in results[:3]]
            }
            
            # Store detailed results
            for result in results:
                all_results.append({
                    'strategy': strategy,
                    'rank': len([r for r in all_results if r['strategy'] == strategy]) + 1,
                    'video_id': result['video_id'],
                    'frame_id': result.get('frame_id', 'N/A'),
                    'score': result['relevance'],
                    'timestamp': result.get('start_time', 'N/A')
                })
            
            if not table_output:
                print(f"✅ {strategy}: Got {len(results)} results")
                logger.info(f"✅ {strategy}: Got {len(results)} results")
                for i, result in enumerate(results[:2]):
                    score_line = f"  {i+1}. Score: {result['relevance']:.4f}, Video: {result['video_id']}"
                    print(score_line)
                    logger.info(score_line)
        except Exception as e:
            strategy_performance[strategy] = {
                'top_score': 0,
                'results_count': 0,
                'top_videos': [],
                'error': str(e)
            }
            if not table_output:
                error_line = f"❌ {strategy}: {e}"
                print(error_line)
                logger.error(error_line)
    
    # Display results in table format if requested
    if table_output:
        display_results_table(strategy_performance, all_results)
        
    # Show analysis if requested and not in table mode
    if show_analysis and not table_output:
        analyze_ranking_results(all_results, strategy_performance)
        
    # Always save results automatically
    save_results_to_csv(all_results, str(auto_csv_path))
    
    # Also save summary as JSON
    summary_path = auto_csv_path.with_suffix('.json')
    summary_data = {
        "test_date": datetime.now().isoformat(),
        "query": test_query,
        "total_strategies": len(strategy_performance),
        "strategy_performance": strategy_performance,
        "top_strategies": [
            {"strategy": k, "score": v["top_score"], "results_count": v["results_count"]}
            for k, v in sorted(strategy_performance.items(), key=lambda x: x[1]["top_score"], reverse=True)[:5]
        ]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"📊 Summary saved to {summary_path}")
    
    # Test validation errors (skip in table mode)
    if not table_output:
        logger.info(f"\n=== Testing Input Validation ===")
        
        # Test missing embeddings
        try:
            client.search({
                "query": test_query,
                "ranking": RankingStrategy.HYBRID_FLOAT_BM25.value,
                "top_k": 3
            })
            logger.error("❌ Should have failed without embeddings")
        except ValueError as e:
            logger.info(f"✅ Correctly caught missing embeddings: {e}")
    
    # Test missing text query
    try:
        client.search({
            "query": "",  # Empty query
            "ranking": RankingStrategy.BM25_ONLY.value,
            "top_k": 3
        })
        logger.error("❌ Should have failed without text query")
    except ValueError as e:
        logger.info(f"✅ Correctly caught missing text query: {e}")
    
    # Test strategy recommendation
    logger.info(f"\n=== Testing Strategy Recommendation ===")
    
    # Text-only query
    rec1 = client.recommend_strategy("find documents about machine learning", has_embeddings=False)
    logger.info(f"Text-only query → {rec1}")
    
    # Visual query with embeddings
    rec2 = client.recommend_strategy("person walking", has_embeddings=True)
    logger.info(f"Visual query with embeddings → {rec2}")
    
    # Visual query with embeddings (speed priority)
    rec3 = client.recommend_strategy("person walking", has_embeddings=True, speed_priority=True)
    logger.info(f"Visual query with speed priority → {rec3}")

def display_results_table(strategy_performance, all_results):
    """Display results in a comprehensive table format"""
    from tests.test_utils import TestResultsFormatter
    
    # Create formatter
    formatter = TestResultsFormatter("vespa_ranking_test")
    
    # Convert strategy_performance to the format expected by formatter
    formatted_performance = {}
    for strategy, perf in strategy_performance.items():
        if 'error' not in perf:
            # Store results for each strategy
            strategy_results = [r for r in all_results if r['strategy'] == strategy]
            formatted_performance[strategy] = {
                'avg_score': perf['top_score'],  # Using top score as avg for single query
                'accuracy': 100.0 if perf.get('results_count', 0) > 0 else 0,
                'found_count': perf.get('results_count', 0),
                'total_queries': 1,
                'results': [{
                    'query_key': 'test_query',
                    'top_video': result['video_id'],
                    'top_score': result['score'],
                    'expected_found': True,
                    'expected_rank': result['rank']
                } for result in strategy_results[:5]]
            }
        else:
            formatted_performance[strategy] = {
                'avg_score': 0,
                'accuracy': 0,
                'found_count': 0,
                'total_queries': 1,
                'results': []
            }
    
    # Use formatter to display the table
    print(formatter.format_ranking_strategy_table(formatted_performance))

def save_results_to_csv(all_results, filename):
    """Save all results to CSV file"""
    # If filename doesn't include a directory, use OutputManager
    if '/' not in filename:
        from src.common.utils.output_manager import get_output_manager
        output_manager = get_output_manager()
        filepath = output_manager.get_test_results_dir() / filename
    else:
        filepath = Path(filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['strategy', 'rank', 'video_id', 'frame_id', 'score', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n📊 Results saved to {filepath}")

def test_default_ranking(query=None):
    """Test default ranking profile (should be 'default' from schema)"""
    client = VespaVideoSearchClient()
    
    logger.info(f"\n=== Testing Default Ranking Profile ===")
    
    # Use provided query or default
    test_query = query if query else "doctor explaining medical procedures"
    
    try:
        # Test with no ranking specified - should use default
        results = client.search({
            "query": test_query,
            "top_k": 3
        })
        logger.info(f"✅ Default ranking: Got {len(results)} results")
        for i, result in enumerate(results[:2]):
            logger.info(f"  {i+1}. Score: {result['relevance']:.4f}, Video: {result['video_id']}")
    except Exception as e:
        logger.error(f"❌ Default ranking failed: {e}")

if __name__ == "__main__":
    import argparse
    
    print("=" * 60)
    print("STARTING VESPA RANKING STRATEGY TEST")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test different Vespa ranking strategies")
    parser.add_argument("--query", "-q", type=str, 
                       help="Custom query to test (default: 'doctor explaining medical procedures')")
    parser.add_argument("--table", "-t", action="store_true",
                       help="Display results in table format sorted by performance")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Skip the detailed analysis output")
    args = parser.parse_args()
    
    print(f"Arguments parsed: query={args.query}, table={args.table}")
    
    logger.info("🚀 Starting comprehensive search client test...")
    print("Logger initialized, starting tests...")
    
    try:
        test_ranking_strategies(args.query, table_output=args.table, show_analysis=not args.no_analysis)
        print("Ranking strategies test completed")
        
        if not args.table:  # Skip default ranking test in table mode
            test_default_ranking(args.query)
            print("Default ranking test completed")
        
    except Exception as e:
        print(f"ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n🎉 Test completed!")
    print("ALL TESTS FINISHED")