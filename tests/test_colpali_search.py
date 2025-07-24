#!/usr/bin/env python3
"""
Test ColPali text-to-video search (semantic search from text queries)
This tests the core functionality used by the multi-agent system
"""

import requests
import torch
import numpy as np
from pathlib import Path
import sys
import json
import argparse

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from vespa.application import Vespa
from src.tools.config import get_config
from tests.test_utils import TestResultsFormatter


def test_colpali_search(output_format="table", save_results=False):
    """Test ColPali search with proper tensor format"""
    
    # Initialize results formatter
    formatter = TestResultsFormatter("colpali_binary_search")
    
    # Load ColPali model for query encoding
    model_name = "vidore/colsmol-500m"
    device = "cpu"
    
    print(f"Loading ColPali model: {model_name}")
    col_model = ColIdefics3.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    ).eval()
    
    col_processor = ColIdefics3Processor.from_pretrained(model_name)
    print("✅ ColPali model loaded")
    
    # Create a test query
    test_query = "doctor explaining medical procedures"
    print(f"\nTest query: '{test_query}'")
    
    # Encode query
    batch_queries = col_processor.process_queries([test_query]).to(device)
    with torch.no_grad():
        query_embeddings = col_model(**batch_queries)
    
    # Convert to proper format - squeeze to get 2D tensor [patches, dims]
    embedding = query_embeddings.cpu().squeeze(0)  # Remove batch dimension
    print(f"Query embedding shape: {embedding.shape}")
    
    # Format embeddings exactly as shown in the example
    # The query tensor api does not support hex formats yet
    float_embedding = {index: vector.tolist() for index, vector in enumerate(embedding)}
    binary_embedding = {
        index: np.packbits(np.where(vector > 0, 1, 0), axis=0).astype(np.int8).tolist()
        for index, vector in enumerate(embedding)
    }
    
    print(f"Created query tensors with {len(float_embedding)} patches")
    
    # Load config to get schema name
    config = get_config()
    vespa_schema = config.get("vespa_schema", "video_frame")
    
    # Prepare search request
    search_url = "http://localhost:8080/search/"
    
    search_body = {
        "yql": f"select * from {vespa_schema} where true",  # brute force search, rank all frames
        "ranking": "default",  # Test default binary visual search
        "hits": 5,
        "timeout": 10,
        "query": test_query,  # Add text query for BM25 component
        "input.query(qtb)": binary_embedding,  # Binary embedding for visual search
    }
    
    print("\n🔍 Performing ColPali similarity search...")
    
    # Use Vespa Python client like the search_client does
    vespa_app = Vespa(url="http://localhost:8080")
    
    try:
        response = vespa_app.query(body=search_body)
        
        print(f"✅ Search completed! Found {len(response.hits)} results")
        
        # Collect results for formatting
        results = []
        for i, hit in enumerate(response.hits[:5]):
            fields = hit["fields"]
            result = {
                "Rank": i + 1,
                "Video ID": fields.get('video_id'),
                "Frame ID": fields.get('frame_id'),
                "Score": f"{hit.get('relevance'):.4f}",
                "Description": fields.get('frame_description', '')[:60] + "..."
            }
            results.append(result)
        
        # Display results based on format
        if output_format == "table":
            print("\n" + formatter.format_table(results))
        else:
            for i, hit in enumerate(response.hits[:5]):
                fields = hit["fields"]
                print(f"\nResult {i+1}:")
                print(f"  - Frame ID: {fields.get('frame_id')}")
                print(f"  - Video ID: {fields.get('video_id')}")
                print(f"  - Relevance score: {hit.get('relevance')}")
                print(f"  - Description: {fields.get('frame_description', '')[:100]}...")
        
        # Save results if requested
        if save_results:
            csv_path = formatter.save_csv(results)
            print(f"\n📊 Results saved to: {csv_path}")
            
        return results
            
    except Exception as e:
        print(f"Search failed: {e}")


def test_float_float_search(output_format="table", save_results=False):
    """Test pure float visual search"""
    
    # Initialize results formatter
    formatter = TestResultsFormatter("colpali_float_float")
    
    # Load config
    config = get_config()
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
    
    print(f"\n\n=== Testing Float-Float Visual Search ===")
    print(f"Loading ColPali model from config: {model_name}")
    print(f"Device: {device}, dtype: {dtype}")
    
    col_model = ColIdefics3.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    ).eval()
    
    col_processor = ColIdefics3Processor.from_pretrained(model_name)
    print("✅ ColPali model loaded")
    
    # Create a test query
    test_query = "doctor explaining medical procedures"
    print(f"\nTest query: '{test_query}'")
    
    # Encode query
    batch_queries = col_processor.process_queries([test_query]).to(device)
    with torch.no_grad():
        query_embeddings = col_model(**batch_queries)
    
    # Convert to numpy
    embeddings_np = query_embeddings.cpu().numpy().squeeze(0)
    print(f"Query embedding shape: {embeddings_np.shape}")
    print(f"Embeddings dtype: {embeddings_np.dtype}")
    
    # Use the search client directly
    from src.processing.vespa.vespa_search_client import VespaVideoSearchClient
    
    # Initialize search client
    search_client = VespaVideoSearchClient(vespa_url=vespa_url, vespa_port=vespa_port)
    
    # Prepare search params - NO text query for pure visual search
    search_params = {
        "query": "",  # Empty query for pure visual search
        "ranking": "float_float",
        "top_k": 10
    }
    
    print(f"\n🔍 Performing Float-Float visual search...")
    print(f"   Vespa: {vespa_url}:{vespa_port}")
    print(f"   Ranking: float_float")
    
    try:
        # Use the search client with proper embedding format
        results = search_client.search(search_params, embeddings_np)
        
        print(f"✅ Search completed! Found {len(results)} results")
        
        # Format results for display
        formatted_results = []
        for i, result in enumerate(results[:5]):
            formatted_result = {
                "Rank": i + 1,
                "Video ID": result.get('video_id'),
                "Frame ID": result.get('frame_id'),
                "Score": f"{result.get('relevance'):.4f}",
                "Description": result.get('frame_description', '')[:60] + "..."
            }
            formatted_results.append(formatted_result)
        
        # Display results based on format
        if output_format == "table":
            print("\n" + formatter.format_table(formatted_results))
        else:
            for i, result in enumerate(results[:5]):
                print(f"\nResult {i+1}:")
                print(f"  - Video ID: {result.get('video_id')}")
                print(f"  - Frame ID: {result.get('frame_id')}")
                print(f"  - Relevance score: {result.get('relevance')}")
                print(f"  - Description: {result.get('frame_description', '')[:100]}...")
        
        # Save results if requested
        if save_results:
            csv_path = formatter.save_csv(formatted_results)
            print(f"\n📊 Results saved to: {csv_path}")
            
        return formatted_results
            
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_float_bm25(output_format="table", save_results=False):
    """Test hybrid search with float embeddings - exactly like video agent"""
    
    # Initialize results formatter
    formatter = TestResultsFormatter("colpali_hybrid_float_bm25")
    
    # Load config
    config = get_config()
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
    
    print(f"\n\n=== Testing Hybrid Float BM25 (like video agent) ===")
    print(f"Loading ColPali model from config: {model_name}")
    print(f"Device: {device}, dtype: {dtype}")
    
    col_model = ColIdefics3.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    ).eval()
    
    col_processor = ColIdefics3Processor.from_pretrained(model_name)
    print("✅ ColPali model loaded")
    
    # Create a test query
    test_query = "doctor explaining medical procedures"
    print(f"\nTest query: '{test_query}'")
    
    # Encode query - EXACTLY like video agent does
    batch_queries = col_processor.process_queries([test_query]).to(device)
    with torch.no_grad():
        query_embeddings = col_model(**batch_queries)
    
    # Convert to numpy like video agent
    embeddings_np = query_embeddings.cpu().numpy().squeeze(0)
    print(f"Query embedding shape: {embeddings_np.shape}")
    print(f"Embeddings dtype: {embeddings_np.dtype}")
    
    # Use the search client directly to ensure proper formatting
    from src.processing.vespa.vespa_search_client import VespaVideoSearchClient
    
    # Initialize search client
    search_client = VespaVideoSearchClient(vespa_url=vespa_url, vespa_port=vespa_port)
    
    # Prepare search params
    search_params = {
        "query": test_query,
        "ranking": "hybrid_float_bm25",
        "top_k": 10
    }
    
    print(f"\n🔍 Performing Hybrid Float+BM25 search...")
    print(f"   Vespa: {vespa_url}:{vespa_port}")
    print(f"   Ranking: hybrid_float_bm25")
    
    try:
        # Use the search client with proper embedding format
        results = search_client.search(search_params, embeddings_np)
        
        print(f"✅ Search completed! Found {len(results)} results")
        
        # Format results for display
        formatted_results = []
        for i, result in enumerate(results[:5]):
            formatted_result = {
                "Rank": i + 1,
                "Video ID": result.get('video_id'),
                "Frame ID": result.get('frame_id'),
                "Score": f"{result.get('relevance'):.4f}",
                "Description": result.get('frame_description', '')[:40] + "...",
                "Transcript": result.get('audio_transcript', '')[:40] + "..."
            }
            formatted_results.append(formatted_result)
        
        # Display results based on format
        if output_format == "table":
            print("\n" + formatter.format_table(formatted_results))
        else:
            for i, result in enumerate(results[:5]):
                print(f"\nResult {i+1}:")
                print(f"  - Video ID: {result.get('video_id')}")
                print(f"  - Frame ID: {result.get('frame_id')}")
                print(f"  - Relevance score: {result.get('relevance')}")
                print(f"  - Description: {result.get('frame_description', '')[:100]}...")
                print(f"  - Transcript snippet: {result.get('audio_transcript', '')[:100]}...")
        
        # Save results if requested
        if save_results:
            csv_path = formatter.save_csv(formatted_results)
            print(f"\n📊 Results saved to: {csv_path}")
            
        return formatted_results
            
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ColPali search functionality")
    parser.add_argument("--format", choices=["table", "text"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--save", action="store_true",
                       help="Save results to CSV file")
    parser.add_argument("--test", choices=["binary", "float", "hybrid", "all"], default="all",
                       help="Which test to run (default: all)")
    
    args = parser.parse_args()
    
    config = get_config()
    output_format = config.get("test_output_format", args.format)
    
    print("Running ColPali search tests...")
    print("=" * 60)
    
    if args.test in ["binary", "all"]:
        print("\n🔍 Binary Search Test:")
        test_colpali_search(output_format=output_format, save_results=args.save)
    
    if args.test in ["float", "all"]:
        print("\n🔍 Float-Float Search Test:")
        test_float_float_search(output_format=output_format, save_results=args.save)
    
    if args.test in ["hybrid", "all"]:
        print("\n🔍 Hybrid Float+BM25 Test:")
        test_hybrid_float_bm25(output_format=output_format, save_results=args.save)