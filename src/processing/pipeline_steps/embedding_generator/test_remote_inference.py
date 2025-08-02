#!/usr/bin/env python3
"""
Test script for remote inference support in model loaders.

This script demonstrates how remote inference works with the updated
model loader implementation.
"""

import logging
from pathlib import Path
from model_loaders import get_or_load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_remote")


def test_remote_colpali():
    """Test remote ColPali model loading"""
    print("\n=== Testing Remote ColPali ===")
    
    # Config with remote inference
    config = {
        "remote_inference_url": "http://localhost:8080/infinity",
        "remote_inference_api_key": "test-api-key",
        "remote_inference_provider": "infinity"
    }
    
    # Load model (should create RemoteColPaliLoader)
    model, processor = get_or_load_model("vidore/colpali-v1.2", config, logger)
    
    # Verify it's a RemoteInferenceClient
    print(f"Model type: {type(model).__name__}")
    print(f"Endpoint URL: {model.endpoint_url}")
    
    # Test processing
    images = ["test_image1.jpg", "test_image2.jpg"]
    result = model.process_images(images)
    print(f"Embeddings shape: {result['embeddings'].shape}")
    print(f"Processing time: {result['processing_time']}s")


def test_remote_videoprism():
    """Test remote VideoPrism model loading"""
    print("\n=== Testing Remote VideoPrism ===")
    
    # Config with remote inference
    config = {
        "remote_inference_url": "https://modal.example.com/videoprism",
        "remote_inference_api_key": "modal-secret-key",
        "remote_inference_provider": "modal"
    }
    
    # Load model (should create RemoteVideoPrismLoader)
    model, processor = get_or_load_model("videoprism_public_v1_base_hf", config, logger)
    
    # Verify it's wrapped correctly
    print(f"Model type: {type(model).__name__}")
    print(f"Has process_video_segment: {hasattr(model, 'process_video_segment')}")
    
    # Test processing
    result = model.process_video_segment(Path("test_video.mp4"), 0.0, 30.0)
    print(f"Embeddings shape: {result['embeddings_np'].shape}")
    print(f"Processing time: {result['processing_time']}s")


def test_local_fallback():
    """Test that local loading still works"""
    print("\n=== Testing Local Fallback ===")
    
    # Config without remote inference
    config = {
        "device": "cpu"
    }
    
    # Should load local model
    model, processor = get_or_load_model("vidore/colpali-v1.2", config, logger)
    
    # This will fail if colpali_engine is not installed, which is expected
    print(f"Model type: {type(model).__name__}")
    print("Local model loading attempted (may fail if not installed)")


def test_caching():
    """Test that caching works correctly with remote endpoints"""
    print("\n=== Testing Cache Behavior ===")
    
    config1 = {
        "remote_inference_url": "http://endpoint1.com",
    }
    
    config2 = {
        "remote_inference_url": "http://endpoint2.com",
    }
    
    # Load same model with different endpoints
    model1, _ = get_or_load_model("vidore/colpali-v1.2", config1, logger)
    model2, _ = get_or_load_model("vidore/colpali-v1.2", config2, logger)
    
    # Should be different instances
    print(f"Same instance: {model1 is model2}")  # Should be False
    
    # Load again with same endpoint
    model3, _ = get_or_load_model("vidore/colpali-v1.2", config1, logger)
    print(f"Cached instance: {model1 is model3}")  # Should be True


if __name__ == "__main__":
    test_remote_colpali()
    test_remote_videoprism()
    test_caching()
    
    try:
        test_local_fallback()
    except Exception as e:
        print(f"Local fallback test failed (expected if colpali_engine not installed): {e}")