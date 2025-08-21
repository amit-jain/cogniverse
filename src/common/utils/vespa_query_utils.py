#!/usr/bin/env python3
"""Utility functions for Vespa ColPali queries"""

import numpy as np
import torch
from binascii import hexlify
from typing import Dict, List, Any

def binarize_token_vectors_hex(vectors: torch.Tensor) -> Dict[str, str]:
    """
    Convert float token vectors to binary hex format for Vespa queries.
    
    Args:
        vectors: Token vectors as torch.Tensor with shape (num_tokens, 128)
    
    Returns:
        Dictionary mapping token indices to hex-encoded binary strings
    """
    binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=1).astype(
        np.int8
    )
    vespa_token_feed = dict()
    for index in range(0, len(binarized_token_vectors)):
        vespa_token_feed[index] = str(
            hexlify(binarized_token_vectors[index].tobytes()), "utf-8"
        )
    return vespa_token_feed

def float_query_token_vectors(vectors: torch.Tensor) -> Dict[str, List[float]]:
    """
    Convert float token vectors to Vespa float query format.
    
    Args:
        vectors: Token vectors as torch.Tensor with shape (num_tokens, 128)
    
    Returns:
        Dictionary mapping token indices to float vector lists
    """
    vespa_token_feed = dict()
    for index in range(0, len(vectors)):
        vespa_token_feed[index] = vectors[index].tolist()
    return vespa_token_feed

def build_binary_query_params(binary_query_tensors: Dict[str, str], float_query_tensors: Dict[str, List[float]], target_hits: int = 100) -> Dict[str, Any]:
    """
    Build query parameters for 2-phase ColPali search: binary candidates + float reranking.
    
    Args:
        binary_query_tensors: Binary query tensors as hex strings (dict with int keys)
        float_query_tensors: Float query tensors for reranking
        target_hits: Number of target hits for nearestNeighbor
    
    Returns:
        Dictionary of query parameters for Vespa search
    """
    binary_query_vectors = dict()
    nn_operators = list()
    
    for index in range(len(binary_query_tensors)):
        name = f"input.query(binary_vector_{index})"
        nn_argument = f"binary_vector_{index}"
        value = binary_query_tensors[index]
        binary_query_vectors[name] = value
        nn_operators.append(f"({{targetHits:{target_hits}}}nearestNeighbor(colpali_binary, {nn_argument}))")
    
    # Combine with OR operator
    yql_where = " OR ".join(nn_operators) if nn_operators else "true"
    
    query_params = {
        'yql': f'select * from sources * where {yql_where}',
        'ranking': 'colpali_binary_float',
        'hits': target_hits
    }
    
    # Add binary vectors
    query_params.update(binary_query_vectors)
    
    # Add float vectors for reranking
    for token_idx, vector in float_query_tensors.items():
        query_params[f'input.query(qt).querytoken{token_idx}'] = str(vector)
    
    return query_params

def build_binary_query_params_ranking(binary_query_tensors: Dict[str, str], hits: int = 10) -> Dict[str, Any]:
    """
    Alternative: Build query parameters using ranking profile approach.
    """
    query_params = {
        'yql': 'select * from sources * where true',
        'ranking': 'colpali_binary_float',
        'hits': hits
    }
    
    # Add binary query tensors using schema format: query(qtb).querytoken{i}
    for token_idx, hex_value in binary_query_tensors.items():
        query_params[f'input.query(qtb).querytoken{token_idx}'] = hex_value
    
    return query_params

def build_float_query_params(float_query_tensors: Dict[str, List[float]], hits: int = 10) -> Dict[str, Any]:
    """
    Build query parameters for float ColPali search.
    
    Args:
        float_query_tensors: Float query tensors 
        hits: Number of hits to return
    
    Returns:
        Dictionary of query parameters for Vespa search
    """
    query_params = {
        'yql': 'select * from sources * where true',
        'ranking': 'colpali',
        'hits': hits
    }
    
    # Add float query tensors
    for token_idx, vector in float_query_tensors.items():
        query_params[f'input.query(qt).querytoken{token_idx}'] = str(vector)
    
    return query_params

def build_hybrid_query_params(
    float_query_tensors: Dict[str, List[float]], 
    text_query: str = "", 
    hits: int = 10
) -> Dict[str, Any]:
    """
    Build query parameters for hybrid (text + visual) search.
    
    Args:
        float_query_tensors: Float query tensors for visual search
        text_query: Text query string for BM25 search
        hits: Number of hits to return
    
    Returns:
        Dictionary of query parameters for Vespa search
    """
    # Build WHERE clause
    where_clauses = []
    if text_query:
        # Add text search conditions
        text_fields = ["video_title", "frame_description", "audio_transcript"]
        text_conditions = [f'{field} contains "{text_query}"' for field in text_fields]
        where_clauses.append(f"({' OR '.join(text_conditions)})")
    
    if not where_clauses:
        where_clauses.append("true")
    
    yql_where = " AND ".join(where_clauses)
    
    query_params = {
        'yql': f'select * from sources * where {yql_where}',
        'ranking': 'hybrid_search',
        'hits': hits
    }
    
    # Add float query tensors for visual component
    for token_idx, vector in float_query_tensors.items():
        query_params[f'input.query(qt).querytoken{token_idx}'] = str(vector)
    
    return query_params

def create_dummy_query_tensors(num_tokens: int = 2, embedding_dim: int = 128) -> torch.Tensor:
    """
    Create dummy query tensors for testing.
    
    Args:
        num_tokens: Number of query tokens
        embedding_dim: Embedding dimension (usually 128 for ColPali)
    
    Returns:
        Random query tensor for testing
    """
    return torch.randn(num_tokens, embedding_dim)

def test_vespa_colpali_query(vespa_url: str = "http://localhost:8080") -> bool:
    """
    Test ColPali querying functionality with dummy data.
    
    Args:
        vespa_url: Vespa endpoint URL
    
    Returns:
        True if all query types work, False otherwise
    """
    import requests
    
    # Create dummy query tensors
    dummy_tensors = create_dummy_query_tensors(num_tokens=2)
    
    print("🧪 Testing ColPali query formats...")
    
    # Test 1: Float query
    try:
        float_tensors = float_query_token_vectors(dummy_tensors)
        query_params = build_float_query_params(float_tensors)
        
        response = requests.get(f"{vespa_url}/search/", params=query_params)
        print(f"✅ Float query: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Float query failed: {e}")
        return False
    
    # Test 2: Binary 2-phase query  
    try:
        binary_tensors = binarize_token_vectors_hex(dummy_tensors)
        float_tensors = float_query_token_vectors(dummy_tensors)
        query_params = build_binary_query_params(binary_tensors, float_tensors)
        
        response = requests.get(f"{vespa_url}/search/", params=query_params)
        print(f"✅ Binary 2-phase query: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Binary 2-phase query failed: {e}")
        return False
    
    # Test 3: Hybrid query
    try:
        float_tensors = float_query_token_vectors(dummy_tensors)
        query_params = build_hybrid_query_params(float_tensors, "fire")
        
        response = requests.get(f"{vespa_url}/search/", params=query_params)
        print(f"✅ Hybrid query: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Hybrid query failed: {e}")
        return False
    
    print("🎉 All ColPali query formats working!")
    return True

if __name__ == "__main__":
    test_vespa_colpali_query()