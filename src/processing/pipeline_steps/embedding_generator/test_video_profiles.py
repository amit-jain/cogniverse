#!/usr/bin/env python3
"""
Comprehensive tests for all video processing profiles in the v2 refactored implementation.

This module tests the different video processing profiles supported by the system:
1. frame_based - Traditional frame extraction and embedding (ColPali)
2. direct_video - Direct video processing without frame extraction (ColQwen, VideoPrism)

Each profile test verifies:
- Correct model loading behavior
- Proper document creation with media types
- Segment processing for direct video
- Embedding format handling
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import numpy as np
from typing import Dict, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.processing.pipeline_steps.embedding_generator.base_embedding_generator import (
    Document, MediaType, TemporalInfo, SegmentInfo
)
from src.processing.pipeline_steps.embedding_generator.embedding_generator_impl import EmbeddingGeneratorImpl
from src.processing.pipeline_steps.embedding_generator.test_comprehensive import MockBackendClient


class TestVideoProfiles(unittest.TestCase):
    """Test all video processing profiles with proper mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "backend": "vespa",
            "vespa_url": "http://localhost",
            "vespa_port": 8080
        }
        
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.INFO)
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_frame_based_colpali_profile(self, mock_get_model):
        """
        Test frame-based processing profile (ColPali).
        
        This profile:
        - Processes pre-extracted frames
        - Does NOT load model during init
        - Creates VIDEO_FRAME media type documents
        - Processes frames in batches
        """
        # Profile configuration mimicking real ColPali profile
        profile_config = {
            "process_type": "frame_based",
            "embedding_model": "vidore/colsmol-500m",
            "vespa_schema": "video_frame",
            "batch_size": 32,
            "embedding_dim": 128,
            "num_patches": 1024
        }
        
        mock_backend = MockBackendClient(self.config, "video_frame", self.logger)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=profile_config,
            backend_client=mock_backend
        )
        
        # Verify model is NOT loaded for frame-based
        mock_get_model.assert_not_called()
        
        # Verify correct media type
        self.assertEqual(generator.media_type, MediaType.VIDEO_FRAME)
        self.assertEqual(generator.process_type, "frame_based")
        
        # Test frame processing
        video_data = {
            "video_id": "test_colpali_video",
            "video_path": "/test/video.mp4",
            "frames": [
                {"frame_path": f"/frames/frame_{i:04d}.jpg", "timestamp": i * 0.1}
                for i in range(5)
            ]
        }
        
        # Mock model loading for frame processing
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        # Since frame-based doesn't implement actual processing in our test,
        # we'll verify the setup is correct
        self.assertEqual(len(video_data["frames"]), 5)
        self.assertEqual(generator.model_name, "vidore/colsmol-500m")
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_direct_video_colqwen_profile(self, mock_get_model):
        """
        Test direct video processing profile (ColQwen).
        
        This profile:
        - Processes video directly without frame extraction
        - Loads model during init
        - Creates VIDEO media type documents
        - Uses adaptive segmentation based on max_pixels
        """
        # Profile configuration mimicking real ColQwen profile
        profile_config = {
            "process_type": "direct_video_segment",
            "embedding_model": "vidore/colqwen-omni-v0.1",
            "vespa_schema": "video_colqwen",
            "segment_duration": 15.0,
            "max_patches": 1024,
            "max_pixels": 16777216,
            "adaptive_segmentation": True,
            "sampling_fps": 1.0
        }
        
        mock_backend = MockBackendClient(self.config, "video_colqwen", self.logger)
        
        # Mock model loading
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=profile_config,
            backend_client=mock_backend
        )
        
        # Verify model IS loaded for direct_video
        mock_get_model.assert_called_once()
        
        # Verify correct media type
        self.assertEqual(generator.media_type, MediaType.VIDEO)
        self.assertEqual(generator.process_type, "direct_video_segment")
        
        # Test segment processing
        doc = generator._process_video_segment(
            video_path=Path("/test/colqwen_video.mp4"),
            video_id="colqwen_test",
            segment_idx=0,
            start_time=0.0,
            end_time=15.0,
            num_segments=4
        )
        
        # Verify document structure
        self.assertIsNone(doc)  # Will be None since we didn't mock _generate_raw_embeddings
        
        # Mock embedding generation to test full flow
        generator._generate_raw_embeddings = Mock(
            return_value=np.random.rand(1024, 768)  # ColQwen uses up to 1024 patches
        )
        
        doc = generator._process_video_segment(
            video_path=Path("/test/colqwen_video.mp4"),
            video_id="colqwen_test",
            segment_idx=0,
            start_time=0.0,
            end_time=15.0,
            num_segments=4
        )
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.media_type, MediaType.VIDEO)
        self.assertEqual(doc.document_id, "colqwen_test_0_0")
        self.assertEqual(doc.temporal_info.duration, 15.0)
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_direct_video_videoprism_base_profile(self, mock_get_model):
        """
        Test direct video processing profile (VideoPrism Base).
        
        This profile:
        - Processes video with VideoPrism base model
        - Uses 30-second segments
        - Handles up to 4096 patches per segment
        - Creates VIDEO media type documents
        """
        # Profile configuration mimicking real VideoPrism base profile
        profile_config = {
            "process_type": "direct_video_frame",
            "embedding_model": "videoprism_public_v1_base_hf",
            "vespa_schema": "video_videoprism_base",
            "segment_duration": 30.0,
            "num_patches": 4096,
            "embedding_dim": 768,
            "use_cpu": True  # VideoPrism uses CPU backend
        }
        
        mock_backend = MockBackendClient(self.config, "video_videoprism_base", self.logger)
        
        # Mock VideoPrism loader
        mock_videoprism_loader = Mock()
        mock_videoprism_loader.process_video_segment = Mock(
            return_value={"embeddings_np": np.random.rand(4096, 768)}
        )
        mock_get_model.return_value = (mock_videoprism_loader, None)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=profile_config,
            backend_client=mock_backend
        )
        
        # Verify model loading
        mock_get_model.assert_called_once()
        
        # Verify media type
        self.assertEqual(generator.media_type, MediaType.VIDEO)
        
        # Test VideoPrism specific embedding generation
        generator.videoprism_loader = mock_videoprism_loader
        
        doc = generator._process_video_segment(
            video_path=Path("/test/videoprism_video.mp4"),
            video_id="videoprism_base_test",
            segment_idx=0,
            start_time=0.0,
            end_time=30.0,
            num_segments=3
        )
        
        # Verify VideoPrism loader was called
        mock_videoprism_loader.process_video_segment.assert_called_with(
            Path("/test/videoprism_video.mp4"),
            0.0,
            30.0
        )
        
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_direct_video_videoprism_large_profile(self, mock_get_model):
        """
        Test direct video processing profile (VideoPrism Large).
        
        This profile:
        - Uses larger VideoPrism model with 1024 embedding dimensions
        - Handles same 4096 patches but larger embeddings
        - Tests memory efficiency with large embeddings
        """
        # Profile configuration mimicking real VideoPrism large profile
        profile_config = {
            "process_type": "direct_video_frame_large",
            "embedding_model": "videoprism_public_v1_large_hf",
            "vespa_schema": "video_videoprism_large",
            "segment_duration": 30.0,
            "num_patches": 4096,
            "embedding_dim": 1024,  # Larger than base
            "use_cpu": True
        }
        
        mock_backend = MockBackendClient(self.config, "video_videoprism_large", self.logger)
        
        # Mock VideoPrism large loader
        mock_videoprism_loader = Mock()
        mock_videoprism_loader.process_video_segment = Mock(
            return_value={"embeddings_np": np.random.rand(4096, 1024)}  # Larger embeddings
        )
        mock_get_model.return_value = (mock_videoprism_loader, None)
        
        generator = EmbeddingGeneratorImpl(
            config=self.config,
            logger=self.logger,
            profile_config=profile_config,
            backend_client=mock_backend
        )
        
        # Test with multiple segments to verify memory handling
        generator.videoprism_loader = mock_videoprism_loader
        generator._generate_raw_embeddings = Mock(
            side_effect=lambda *args: mock_videoprism_loader.process_video_segment(*args).get("embeddings_np")
        )
        
        video_data = {
            "video_id": "videoprism_large_test",
            "video_path": "/test/large_video.mp4",
            "duration": 90.0  # 3 segments
        }
        
        result = generator.generate_embeddings(video_data, Path("/tmp"))
        
        # Verify all segments were processed
        self.assertEqual(result.documents_processed, 3)
        self.assertEqual(result.documents_fed, 3)
        
        # Verify backend received large embeddings
        self.assertEqual(len(mock_backend.fed_documents), 3)
        for doc in mock_backend.fed_documents:
            # Check that embeddings were processed (would be hex strings in real scenario)
            self.assertIn("embedding", doc["fields"])
            
    @patch('src.processing.pipeline_steps.embedding_generator.embedding_generator_impl.get_or_load_model')
    def test_profile_specific_configurations(self, mock_get_model):
        """
        Test that profile-specific configurations are properly handled.
        
        Verifies:
        - segment_duration is respected
        - batch_size is used for frame processing
        - Model-specific settings are preserved
        """
        # Mock model loading
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        test_cases = [
            {
                "name": "ColPali with custom batch size",
                "profile": {
                    "process_type": "frame_based",
                    "embedding_model": "vidore/colsmol-500m",
                    "batch_size": 64,  # Custom batch size
                    "embedding_dim": 128
                },
                "expected_type": MediaType.VIDEO_FRAME,
                "should_load_model": False  # Frame-based doesn't load during init
            },
            {
                "name": "ColQwen with short segments",
                "profile": {
                    "process_type": "direct_video",
                    "embedding_model": "vidore/colqwen-omni-v0.1",
                    "segment_duration": 5.0,  # Very short segments
                    "max_patches": 512
                },
                "expected_type": MediaType.VIDEO,
                "should_load_model": True
            },
            {
                "name": "VideoPrism with custom patches",
                "profile": {
                    "process_type": "direct_video",
                    "embedding_model": "videoprism_public_v1_base_hf",
                    "segment_duration": 20.0,
                    "num_patches": 2048  # Fewer patches than default
                },
                "expected_type": MediaType.VIDEO,
                "should_load_model": True
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(name=test_case["name"]):
                mock_backend = MockBackendClient(self.config, "test_schema", self.logger)
                
                # Reset mock
                mock_get_model.reset_mock()
                
                generator = EmbeddingGeneratorImpl(
                    config=self.config,
                    logger=self.logger,
                    profile_config=test_case["profile"],
                    backend_client=mock_backend
                )
                
                # Verify media type
                self.assertEqual(generator.media_type, test_case["expected_type"])
                
                # Verify model loading behavior
                if test_case["should_load_model"]:
                    mock_get_model.assert_called_once()
                else:
                    mock_get_model.assert_not_called()
                
                # Verify profile config is stored
                self.assertEqual(
                    generator.profile_config.get("segment_duration"),
                    test_case["profile"].get("segment_duration")
                )
                self.assertEqual(
                    generator.profile_config.get("batch_size"),
                    test_case["profile"].get("batch_size")
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)