#!/usr/bin/env python3
"""
Test document similarity search (find similar frames to a given frame)
This is useful for UI features like "find more frames like this one"
"""

import requests
import json
import numpy as np

def test_document_similarity():
    """Test finding similar frames to a given document (reverse image search)"""
    
    # First, get embeddings from one document to use as query
    doc_id = "big_buck_bunny_clip_frame_0"
    doc_url = f"http://localhost:8080/document/v1/video/video_frame/docid/{doc_id}"
    
    response = requests.get(doc_url)
    if response.status_code != 200:
        print(f"Failed to get document: {response.status_code}")
        return
    
    doc = response.json()
    
    # Extract embeddings from the document
    # The embeddings are stored in "blocks" format
    colpali_embedding = doc["fields"]["colpali_embedding"]
    
    print(f"Found embeddings with type: {colpali_embedding.get('type')}")
    
    # Convert the blocks format to the query format needed
    # We need to extract the tensor values and create the float/binary dictionaries
    blocks = colpali_embedding.get("blocks", [])
    
    if not blocks:
        print("No embedding blocks found")
        return
    
    # For testing, let's create a simple query using the first few patches
    # In real usage, you'd encode a text query using ColPali model
    float_embedding = {}
    binary_embedding = {}
    
    # Extract first 10 patches for testing
    patch_count = 0
    for patch_idx in sorted(blocks.keys())[:10]:
        # Get values for this patch (it's directly an array)
        values = blocks[patch_idx]
        if values:
            float_embedding[str(patch_count)] = values
            # Create binary version
            binary_vals = np.packbits(np.where(np.array(values) > 0, 1, 0)).astype(np.int8).tolist()
            binary_embedding[str(patch_count)] = binary_vals
            patch_count += 1
    
    print(f"Created query with {len(float_embedding)} patches")
    
    # Now perform the search using the correct format
    search_url = "http://localhost:8080/search/"
    
    search_body = {
        "yql": "select * from video_frame where true",  # brute force search
        "ranking.profile": "default",  # Use the default profile
        "hits": 5,
        "timeout": "10s",
        "input.query(qt)": float_embedding,
        "input.query(qtb)": binary_embedding,
        "ranking.rerankCount": 20
    }
    
    print("\nPerforming ColPali similarity search...")
    
    response = requests.post(search_url, json=search_body)
    
    print(f"Search response status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        
        if "error" in results:
            print(f"Search error: {results['error']}")
            return
            
        hits = results.get("root", {}).get("children", [])
        print(f"\n✅ Found {len(hits)} similar frames!")
        
        for i, hit in enumerate(hits[:5]):
            fields = hit["fields"]
            print(f"\nResult {i+1}:")
            print(f"  - Frame ID: {fields.get('frame_id')}")
            print(f"  - Video ID: {fields.get('video_id')}")
            print(f"  - Relevance score: {hit.get('relevance')}")
            print(f"  - Description: {fields.get('frame_description', '')[:80]}...")
    else:
        print(f"Search failed: {response.text[:500]}")

if __name__ == "__main__":
    test_document_similarity()