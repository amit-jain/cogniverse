#!/usr/bin/env python3
"""Comprehensive query utilities for all ranking strategies"""

import numpy as np
import torch
from binascii import hexlify
from typing import Dict, List, Any

def binarize_token_vectors_hex(vectors: torch.Tensor) -> Dict[str, str]:
    """Convert float token vectors to binary hex format for Vespa queries."""
    binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=1).astype(np.int8)
    vespa_token_feed = dict()
    for index in range(0, len(binarized_token_vectors)):
        vespa_token_feed[index] = str(hexlify(binarized_token_vectors[index].tobytes()), "utf-8")
    return vespa_token_feed

def float_query_token_vectors(vectors: torch.Tensor) -> Dict[str, List[float]]:
    """Convert float token vectors to Vespa float query format."""
    vespa_token_feed = dict()
    for index in range(0, len(vectors)):
        vespa_token_feed[index] = vectors[index].tolist()
    return vespa_token_feed

def build_query_params(
    ranking_profile: str,
    float_tensors: Dict[str, List[float]] = None,
    binary_tensors: Dict[str, str] = None,
    text_query: str = "",
    hits: int = 10
) -> Dict[str, Any]:
    """
    Build query parameters for any ranking strategy.
    
    Args:
        ranking_profile: One of the 9 ranking strategies
        float_tensors: Float query tensors (if needed)
        binary_tensors: Binary query tensors (if needed) 
        text_query: Text query for BM25 components
        hits: Number of hits to return
    
    Returns:
        Query parameters for Vespa search
    """
    
    # Base query parameters
    query_params = {
        'ranking': ranking_profile,
        'hits': hits
    }
    
    # Build YQL based on ranking profile
    if ranking_profile == "bm25_only":
        # Pure text search
        if text_query:
            text_fields = ["video_title", "frame_description", "audio_transcript"]
            text_conditions = [f'{field} contains "{text_query}"' for field in text_fields]
            query_params['yql'] = f'select * from sources * where ({" OR ".join(text_conditions)})'
        else:
            query_params['yql'] = 'select * from sources * where true'
    
    else:
        # Visual search (with optional text filtering)
        where_clauses = []
        if text_query:
            text_fields = ["video_title", "frame_description", "audio_transcript"]
            text_conditions = [f'{field} contains "{text_query}"' for field in text_fields]
            where_clauses.append(f"({' OR '.join(text_conditions)})")
        
        if not where_clauses:
            where_clauses.append("true")
        
        query_params['yql'] = f'select * from sources * where {" AND ".join(where_clauses)}'
    
    # Add tensor inputs based on ranking profile
    if ranking_profile in ["float_float", "hybrid_float_bm25", "bm25_float_rerank"]:
        # Float-only profiles
        if float_tensors:
            for token_idx, vector in float_tensors.items():
                query_params[f'input.query(qt).querytoken{token_idx}'] = str(vector)
    
    elif ranking_profile == "binary_binary":
        # Binary-only profile
        if binary_tensors:
            for token_idx, hex_value in binary_tensors.items():
                query_params[f'input.query(qtb).querytoken{token_idx}'] = hex_value
    
    elif ranking_profile in ["float_binary", "phased", "binary_bm25", "bm25_binary_rerank"]:
        # Profiles that need both float and binary
        if float_tensors:
            for token_idx, vector in float_tensors.items():
                query_params[f'input.query(qt).querytoken{token_idx}'] = str(vector)
        if binary_tensors:
            for token_idx, hex_value in binary_tensors.items():
                query_params[f'input.query(qtb).querytoken{token_idx}'] = hex_value
    
    return query_params

def benchmark_all_strategies(
    query_text: str = "",
    num_tokens: int = 2,
    hits: int = 5,
    vespa_url: str = "http://localhost:8080"
) -> Dict[str, Any]:
    """
    Benchmark all 9 ranking strategies with a single query.
    
    Args:
        query_text: Text query for BM25 components
        num_tokens: Number of query tokens to generate
        hits: Number of hits per strategy
        vespa_url: Vespa endpoint URL
    
    Returns:
        Results from all ranking strategies
    """
    import requests
    import time
    
    # Generate dummy query tensors for testing
    dummy_tensors = torch.randn(num_tokens, 128)
    float_tensors = float_query_token_vectors(dummy_tensors)
    binary_tensors = binarize_token_vectors_hex(dummy_tensors)
    
    strategies = [
        "bm25_only",
        "float_float", 
        "binary_binary",
        "float_binary",
        "phased",
        "hybrid_float_bm25",
        "binary_bm25",
        "bm25_binary_rerank",
        "bm25_float_rerank"
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"🔍 Testing {strategy}...")
        
        try:
            start_time = time.time()
            
            # Build query parameters for this strategy
            query_params = build_query_params(
                ranking_profile=strategy,
                float_tensors=float_tensors,
                binary_tensors=binary_tensors,
                text_query=query_text,
                hits=hits
            )
            
            # Execute query
            response = requests.get(f"{vespa_url}/search/", params=query_params)
            query_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                hits_returned = result.get('root', {}).get('children', [])
                
                results[strategy] = {
                    "status": "success",
                    "hits_count": len(hits_returned),
                    "query_time_ms": round(query_time * 1000, 2),
                    "sample_scores": [hit.get('relevance', 0) for hit in hits_returned[:3]]
                }
                
                print(f"   ✅ {len(hits_returned)} hits in {query_time*1000:.1f}ms")
                
            else:
                results[strategy] = {
                    "status": "error",
                    "error": response.text[:200],
                    "query_time_ms": round(query_time * 1000, 2)
                }
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            results[strategy] = {
                "status": "exception",
                "error": str(e),
                "query_time_ms": 0
            }
            print(f"   ❌ Exception: {e}")
    
    return results

def print_benchmark_results(results: Dict[str, Any]):
    """Print formatted benchmark results."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE RANKING STRATEGY BENCHMARK")
    print("="*60)
    
    successful = [k for k, v in results.items() if v.get('status') == 'success']
    failed = [k for k, v in results.items() if v.get('status') != 'success']
    
    print(f"✅ Successful strategies: {len(successful)}/{len(results)}")
    print(f"❌ Failed strategies: {len(failed)}")
    
    if successful:
        print(f"\n🏆 PERFORMANCE RANKING (by speed):")
        sorted_by_speed = sorted(successful, key=lambda k: results[k]['query_time_ms'])
        
        for i, strategy in enumerate(sorted_by_speed, 1):
            data = results[strategy]
            hits = data['hits_count']
            time_ms = data['query_time_ms']
            scores = data['sample_scores']
            avg_score = sum(scores) / len(scores) if scores else 0
            
            print(f"   {i}. {strategy}")
            print(f"      Time: {time_ms}ms | Hits: {hits} | Avg Score: {avg_score:.3f}")
    
    if failed:
        print(f"\n❌ FAILED STRATEGIES:")
        for strategy in failed:
            error = results[strategy].get('error', 'Unknown error')
            print(f"   {strategy}: {error}")
    
    print("\n💡 RECOMMENDATIONS:")
    if successful:
        fastest = min(successful, key=lambda k: results[k]['query_time_ms'])
        print(f"   Fastest: {fastest} ({results[fastest]['query_time_ms']}ms)")
        
        best_hits = max(successful, key=lambda k: results[k]['hits_count'])
        print(f"   Most hits: {best_hits} ({results[best_hits]['hits_count']} hits)")

def create_dummy_query_tensors(num_tokens: int = 2, embedding_dim: int = 128) -> torch.Tensor:
    """Create dummy query tensors for testing."""
    return torch.randn(num_tokens, embedding_dim)

if __name__ == "__main__":
    print("🎬 Testing Comprehensive Video Search Ranking Strategies\n")
    
    # Test with a sample query
    results = benchmark_all_strategies(
        query_text="fire",
        num_tokens=2,
        hits=5
    )
    
    # Print results
    print_benchmark_results(results)