#!/usr/bin/env python3
"""
Test script for refactored embedding generator
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    try:
        from .base_embedding_generator import BaseEmbeddingGenerator, EmbeddingResult, ProcessingConfig
        logger.info("✓ base_embedding_generator imported successfully")
        
        from .model_loaders import get_or_load_model, ModelLoaderFactory
        logger.info("✓ model_loaders imported successfully")
        
        from .document_builders import DocumentBuilderFactory, DocumentMetadata
        logger.info("✓ document_builders imported successfully")
        
        from .embedding_processors import EmbeddingProcessor
        logger.info("✓ embedding_processors imported successfully")
        
        from .vespa_client import VespaClient
        logger.info("✓ vespa_client imported successfully")
        
        from .vespa_embedding_generator import VespaEmbeddingGenerator
        logger.info("✓ vespa_embedding_generator imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False

def test_document_builder():
    """Test document builder factory"""
    try:
        from .document_builders import DocumentBuilderFactory, DocumentMetadata
        
        # Test different schema types
        schemas = ["video_frame", "video_colqwen", "video_videoprism_base"]
        
        for schema in schemas:
            builder = DocumentBuilderFactory.create_builder(schema)
            logger.info(f"✓ Created builder for schema: {schema}")
            
            # Test document creation
            metadata = DocumentMetadata(
                video_id="test_video",
                video_title="Test Video",
                segment_idx=0,
                start_time=0.0,
                end_time=30.0
            )
            
            doc = builder.build_document(
                metadata,
                {"float_embeddings": {0: "test_hex"}, "binary_embeddings": {0: "test_binary"}}
            )
            
            assert "put" in doc
            assert "fields" in doc
            logger.info(f"✓ Built document for schema: {schema}")
        
        return True
    except Exception as e:
        logger.error(f"Document builder test failed: {e}")
        return False

def test_embedding_processor():
    """Test embedding processor"""
    try:
        from .embedding_processors import EmbeddingProcessor
        import numpy as np
        
        processor = EmbeddingProcessor(logger)
        logger.info("✓ Created embedding processor")
        
        # Test hex conversion
        test_embeddings = np.random.randn(10, 768).astype(np.float32)
        
        float_emb = processor.convert_to_float_embeddings(test_embeddings)
        assert len(float_emb) == 10
        assert all(isinstance(v, str) for v in float_emb.values())
        logger.info("✓ Float embedding conversion works")
        
        binary_emb = processor.convert_to_binary_embeddings(test_embeddings)
        assert len(binary_emb) == 10
        assert all(isinstance(v, str) for v in binary_emb.values())
        logger.info("✓ Binary embedding conversion works")
        
        return True
    except Exception as e:
        logger.error(f"Embedding processor test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting refactored embedding generator tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Document Builder", test_document_builder),
        ("Embedding Processor", test_embedding_processor)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n--- Test Summary ---")
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        logger.info("\n✅ All tests passed!")
    else:
        logger.error("\n❌ Some tests failed!")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())