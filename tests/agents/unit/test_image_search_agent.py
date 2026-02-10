"""
Unit tests for Image Search Agent

Tests ColPali-based image search with Vespa integration.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from PIL import Image

from cogniverse_agents.image_search_agent import (
    ImageResult,
    ImageSearchAgent,
    ImageSearchDeps,
)


class TestImageSearchAgent:
    """Unit tests for ImageSearchAgent"""

    def setup_method(self):
        """Set up test fixtures"""
        self.agent = ImageSearchAgent(
            deps=ImageSearchDeps(
                tenant_id="test_tenant",
                vespa_endpoint="http://localhost:8080",
            ),
            port=8005,
        )

    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent is not None
        assert self.agent._colpali_model is None  # Lazy loaded
        assert self.agent._colpali_processor is None  # Lazy loaded
        assert self.agent._query_encoder is None  # Lazy loaded
        assert self.agent._vespa_endpoint == "http://localhost:8080"

    @patch("cogniverse_agents.image_search_agent.get_or_load_model")
    def test_colpali_model_lazy_loading(self, mock_get_model):
        """Test ColPali model is lazy loaded on first access"""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_get_model.return_value = (mock_model, mock_processor)

        # Access colpali_model property
        model = self.agent.colpali_model

        # Verify model was loaded
        assert model is not None
        # Check that it was called with model_name, config, logger
        assert mock_get_model.called
        call_args = mock_get_model.call_args[0]
        assert call_args[0] == "vidore/colsmol-500m"  # model_name
        assert "colpali_model" in call_args[1]  # config

    @patch("cogniverse_core.query.encoders.get_or_load_model")
    def test_query_encoder_initialization(self, mock_get_model):
        """Test query encoder is initialized correctly"""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_get_model.return_value = (mock_model, mock_processor)

        # Access query_encoder property
        encoder = self.agent.query_encoder

        # Verify encoder was created
        assert encoder is not None
        assert hasattr(encoder, "model")
        # Verify the model is set (don't compare full object due to verbose repr)
        assert encoder.model is not None

    @pytest.mark.asyncio
    @patch.object(ImageSearchAgent, "query_encoder", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_images_semantic(self, mock_post, mock_query_encoder):
        """Test semantic image search using ColPali"""
        # Mock query encoder
        mock_encoder = MagicMock()
        mock_embedding = np.random.randn(1024, 128)
        mock_encoder.encode.return_value = mock_embedding
        mock_query_encoder.return_value = mock_encoder

        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "root": {
                "children": [
                    {
                        "relevance": 0.95,
                        "fields": {
                            "image_id": "img_001",
                            "source_url": "http://example.com/img1.jpg",
                            "image_title": "Test Image 1",
                            "image_description": "A red car",
                            "detected_objects": ["car"],
                            "detected_scenes": ["outdoor"],
                        },
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        # Execute search
        results = await self.agent.search_images(
            query="red car", search_mode="semantic", limit=20
        )

        # Verify results
        assert len(results) == 1
        assert results[0].image_id == "img_001"
        assert results[0].title == "Test Image 1"
        assert results[0].description == "A red car"
        assert results[0].relevance_score == 0.95

        # Verify Vespa was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "colpali_similarity" in str(call_args)

    @pytest.mark.asyncio
    @patch.object(ImageSearchAgent, "query_encoder", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_images_hybrid(self, mock_post, mock_query_encoder):
        """Test hybrid image search (BM25 + ColPali)"""
        # Mock query encoder
        mock_encoder = MagicMock()
        mock_embedding = np.random.randn(1024, 128)
        mock_encoder.encode.return_value = mock_embedding
        mock_query_encoder.return_value = mock_encoder

        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        # Execute search
        await self.agent.search_images(
            query="sports car", search_mode="hybrid", limit=10
        )

        # Verify Vespa was called with hybrid profile
        call_args = mock_post.call_args
        assert "hybrid_image" in str(call_args)

    @pytest.mark.asyncio
    @patch.object(ImageSearchAgent, "colpali_model", new_callable=PropertyMock)
    @patch.object(ImageSearchAgent, "colpali_processor", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_find_similar_images(self, mock_post, mock_processor, mock_model):
        """Test finding similar images using ColPali"""
        # Mock model and processor
        mock_model_obj = MagicMock()
        mock_processor_obj = MagicMock()
        mock_model.return_value = mock_model_obj
        mock_processor.return_value = mock_processor_obj

        # Mock processor.process_images
        mock_batch_inputs = MagicMock()
        mock_batch_inputs.to.return_value = mock_batch_inputs
        mock_processor_obj.process_images.return_value = mock_batch_inputs

        # Mock model device
        mock_model_obj.device = "cpu"

        # Mock model output
        mock_embeddings_tensor = MagicMock()
        mock_embeddings = np.random.randn(1, 1024, 128)
        mock_embeddings_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = mock_embeddings.squeeze(
            0
        )
        mock_model_obj.return_value = mock_embeddings_tensor

        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        # Execute similar image search
        reference_image = Image.new("RGB", (100, 100))
        await self.agent.find_similar_images(reference_image=reference_image, limit=20)

        # Verify Vespa was called
        mock_post.assert_called_once()

    @patch("cogniverse_agents.image_search_agent.get_or_load_model")
    def test_encode_image(self, mock_get_model):
        """Test image encoding with ColPali"""
        # Mock model and processor
        mock_model_obj = MagicMock()
        mock_processor_obj = MagicMock()
        mock_get_model.return_value = (mock_model_obj, mock_processor_obj)

        # Mock processor.process_images
        mock_batch_inputs = MagicMock()
        mock_batch_inputs.to.return_value = mock_batch_inputs
        mock_processor_obj.process_images.return_value = mock_batch_inputs

        # Mock model device
        mock_model_obj.device = "cpu"

        # Mock model output with correct shape [batch, patches, dim]
        mock_embeddings_tensor = MagicMock()
        mock_embeddings = np.random.randn(1, 1024, 128)  # [batch, patches, dim]
        mock_embeddings_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = mock_embeddings.squeeze(
            0
        )
        mock_model_obj.return_value = mock_embeddings_tensor

        # Encode image
        image = Image.new("RGB", (100, 100))
        embedding = self.agent._encode_image(image)

        # Verify embedding shape
        assert embedding.shape == (1024, 128)

    @pytest.mark.asyncio
    async def test_dspy_to_a2a_output(self):
        """Test DSPy output to A2A format conversion"""
        # Create sample results
        results = [
            ImageResult(
                image_id="img_001",
                image_url="http://example.com/img1.jpg",
                title="Test Image",
                description="A test image",
                relevance_score=0.95,
                detected_objects=["car", "road"],
                detected_scenes=["outdoor"],
            )
        ]

        # Convert to A2A output
        a2a_output = self.agent._dspy_to_a2a_output({"results": results, "count": 1})

        # Verify format
        assert a2a_output["status"] == "success"
        assert a2a_output["result_type"] == "image_search_results"
        assert a2a_output["count"] == 1
        assert len(a2a_output["results"]) == 1
        assert a2a_output["results"][0]["image_id"] == "img_001"

    def test_get_agent_skills(self):
        """Test agent skills definition"""
        skills = self.agent._get_agent_skills()

        # Verify skills
        assert len(skills) >= 2
        skill_names = [s["name"] for s in skills]
        assert "search_images" in skill_names
        assert "find_similar_images" in skill_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
