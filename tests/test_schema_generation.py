#!/usr/bin/env python3
"""
Unit tests for schema generation from templates
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.generate_schema_from_template import generate_schema
from string import Template


class TestSchemaGeneration(unittest.TestCase):
    """Test schema generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_profiles = {
            "test_colpali": {
                "vespa_schema": "video_colpali",
                "embedding_model": "vidore/colsmol-500m",
                "embedding_type": "frame_based",
                "schema_config": {
                    "schema_name": "video_colpali",
                    "model_name": "ColPali",
                    "num_patches": 1024,
                    "embedding_dim": 128,
                    "binary_dim": 16
                }
            },
            "test_colqwen": {
                "vespa_schema": "video_colqwen",
                "embedding_model": "vidore/colqwen-omni-v0.1",
                "embedding_type": "direct_video_segment",
                "schema_config": {
                    "schema_name": "video_colqwen",
                    "model_name": "ColQwen-Omni",
                    "num_patches": 1024,
                    "embedding_dim": 128,
                    "binary_dim": 16
                }
            },
            "test_videoprism_base": {
                "vespa_schema": "video_videoprism_base",
                "embedding_model": "videoprism_public_v1_base_hf",
                "embedding_type": "direct_video_frame",
                "schema_config": {
                    "schema_name": "video_videoprism_base",
                    "model_name": "VideoPrism-Base",
                    "num_patches": 4096,
                    "embedding_dim": 768,
                    "binary_dim": 96
                }
            },
            "test_videoprism_large": {
                "vespa_schema": "video_videoprism_large",
                "embedding_model": "videoprism_public_v1_large_hf",
                "embedding_type": "direct_video_frame",
                "schema_config": {
                    "schema_name": "video_videoprism_large",
                    "model_name": "VideoPrism-Large",
                    "num_patches": 2048,
                    "embedding_dim": 1024,
                    "binary_dim": 128
                }
            }
        }
    
    def test_binary_dimension_calculation(self):
        """Test that binary dimensions are correctly calculated"""
        test_cases = [
            (128, 16),    # ColPali: 128/8 = 16
            (768, 96),    # VideoPrism Base: 768/8 = 96
            (1024, 128),  # VideoPrism Large: 1024/8 = 128
            (256, 32),    # Generic test
            (512, 64),    # Generic test
        ]
        
        for embedding_dim, expected_binary_dim in test_cases:
            actual_binary_dim = embedding_dim // 8
            self.assertEqual(actual_binary_dim, expected_binary_dim,
                           f"Binary dimension for {embedding_dim} should be {expected_binary_dim}")
    
    def test_schema_generation_colqwen(self):
        """Test schema generation for ColQwen-Omni model"""
        profile = self.test_profiles["test_colqwen"]
        schema_content = generate_schema("test_colqwen", profile)
        
        self.assertIsNotNone(schema_content)
        
        # Check key substitutions
        self.assertIn("schema video_colqwen {", schema_content)
        self.assertIn("document video_colqwen {", schema_content)
        self.assertIn("# ColQwen-Omni: 1024 patches × 128 dims", schema_content)
        self.assertIn("field embedding type tensor<bfloat16>(patch{}, v[128])", schema_content)
        self.assertIn("field embedding_binary type tensor<int8>(patch{}, v[16])", schema_content)
        
        # Check query tensor dimensions in ranking profiles
        self.assertIn("query(qt) tensor<float>(querytoken{}, v[128])", schema_content)
        self.assertIn("query(qtb) tensor<int8>(querytoken{}, v[16])", schema_content)
    
    def test_schema_generation_colpali(self):
        """Test schema generation for ColPali model"""
        profile = self.test_profiles["test_colpali"]
        schema_content = generate_schema("test_colpali", profile)
        
        self.assertIsNotNone(schema_content)
        
        # Check key substitutions
        self.assertIn("schema video_colpali {", schema_content)
        self.assertIn("document video_colpali {", schema_content)
        self.assertIn("# ColPali: 1024 patches × 128 dims", schema_content)
        self.assertIn("field embedding type tensor<bfloat16>(patch{}, v[128])", schema_content)
        self.assertIn("field embedding_binary type tensor<int8>(patch{}, v[16])", schema_content)
        
        # Check query tensor dimensions in ranking profiles
        self.assertIn("query(qt) tensor<float>(querytoken{}, v[128])", schema_content)
        self.assertIn("query(qtb) tensor<int8>(querytoken{}, v[16])", schema_content)
    
    def test_schema_generation_videoprism_base(self):
        """Test schema generation for VideoPrism Base model"""
        profile = self.test_profiles["test_videoprism_base"]
        schema_content = generate_schema("test_videoprism_base", profile)
        
        self.assertIsNotNone(schema_content)
        
        # Check key substitutions
        self.assertIn("schema video_videoprism_base {", schema_content)
        self.assertIn("# VideoPrism-Base: 4096 patches × 768 dims", schema_content)
        self.assertIn("field embedding type tensor<bfloat16>(patch{}, v[768])", schema_content)
        self.assertIn("field embedding_binary type tensor<int8>(patch{}, v[96])", schema_content)
        
        # Check query tensor dimensions
        self.assertIn("query(qt) tensor<float>(querytoken{}, v[768])", schema_content)
        self.assertIn("query(qtb) tensor<int8>(querytoken{}, v[96])", schema_content)
    
    def test_schema_generation_videoprism_large(self):
        """Test schema generation for VideoPrism Large model"""
        profile = self.test_profiles["test_videoprism_large"]
        schema_content = generate_schema("test_videoprism_large", profile)
        
        self.assertIsNotNone(schema_content)
        
        # Check key substitutions
        self.assertIn("schema video_videoprism_large {", schema_content)
        self.assertIn("# VideoPrism-Large: 2048 patches × 1024 dims", schema_content)
        self.assertIn("field embedding type tensor<bfloat16>(patch{}, v[1024])", schema_content)
        self.assertIn("field embedding_binary type tensor<int8>(patch{}, v[128])", schema_content)
        
        # Check query tensor dimensions
        self.assertIn("query(qt) tensor<float>(querytoken{}, v[1024])", schema_content)
        self.assertIn("query(qtb) tensor<int8>(querytoken{}, v[128])", schema_content)
    
    def test_missing_schema_config(self):
        """Test handling of profile without schema_config"""
        profile = {
            "vespa_schema": "video_frame",
            "embedding_model": "some-model",
            "embedding_type": "frame_based"
            # No schema_config
        }
        
        schema_content = generate_schema("test_profile", profile)
        self.assertIsNone(schema_content)
    
    def test_tensor_dimension_consistency(self):
        """Test that tensor dimensions are consistent throughout the schema"""
        for profile_name, profile in self.test_profiles.items():
            schema_content = generate_schema(profile_name, profile)
            self.assertIsNotNone(schema_content)
            
            config = profile["schema_config"]
            embedding_dim = config["embedding_dim"]
            binary_dim = config["binary_dim"]
            
            # Count occurrences of dimension specifications
            embedding_dim_count = schema_content.count(f"v[{embedding_dim}]")
            binary_dim_count = schema_content.count(f"v[{binary_dim}]")
            
            # Each dimension should appear multiple times in the schema
            # (in field definitions and ranking profiles)
            self.assertGreater(embedding_dim_count, 5,
                             f"Embedding dimension v[{embedding_dim}] should appear multiple times")
            self.assertGreater(binary_dim_count, 5,
                             f"Binary dimension v[{binary_dim}] should appear multiple times")
    
    def test_ranking_profiles_present(self):
        """Test that all expected ranking profiles are present"""
        expected_profiles = [
            "rank-profile default",
            "rank-profile bm25_only",
            "rank-profile float_float",
            "rank-profile binary_binary",
            "rank-profile float_binary",
            "rank-profile phased",
            "rank-profile hybrid_float_bm25",
            "rank-profile hybrid_binary_bm25",
            "rank-profile hybrid_bm25_binary",
            "rank-profile hybrid_bm25_float",
            "rank-profile hybrid_float_bm25_no_description",
            "rank-profile hybrid_binary_bm25_no_description",
            "rank-profile hybrid_bm25_binary_no_description",
            "rank-profile hybrid_bm25_float_no_description"
        ]
        
        profile = self.test_profiles["test_colpali"]
        schema_content = generate_schema("test_colpali", profile)
        
        for ranking_profile in expected_profiles:
            self.assertIn(ranking_profile, schema_content,
                         f"Schema should contain {ranking_profile}")
    
    def test_fieldset_configuration(self):
        """Test that fieldset is properly configured"""
        profile = self.test_profiles["test_colpali"]
        schema_content = generate_schema("test_colpali", profile)
        
        # Check fieldset definition
        self.assertIn("fieldset default {", schema_content)
        self.assertIn("fields: video_title, frame_description, audio_transcript", schema_content)
        
        # Check BM25 usage of fieldset
        self.assertIn("bm25(default)", schema_content)
    
    def test_template_variable_replacement(self):
        """Test that all template variables are replaced"""
        profile = self.test_profiles["test_colpali"]
        schema_content = generate_schema("test_colpali", profile)
        
        # Check that no template variables remain
        self.assertNotIn("${", schema_content)
        self.assertNotIn("$SCHEMA_NAME", schema_content)
        self.assertNotIn("$MODEL_NAME", schema_content)
        self.assertNotIn("$NUM_PATCHES", schema_content)
        self.assertNotIn("$EMBEDDING_DIM", schema_content)
        self.assertNotIn("$BINARY_DIM", schema_content)
    
    def test_patch_counts(self):
        """Test that patch counts are correctly documented"""
        test_cases = [
            ("test_colpali", "1024 patches"),
            ("test_colqwen", "1024 patches"),
            ("test_videoprism_base", "4096 patches"),
            ("test_videoprism_large", "2048 patches")
        ]
        
        for profile_name, expected_patch_text in test_cases:
            profile = self.test_profiles[profile_name]
            schema_content = generate_schema(profile_name, profile)
            self.assertIn(expected_patch_text, schema_content,
                         f"Schema for {profile_name} should mention {expected_patch_text}")


if __name__ == "__main__":
    unittest.main()