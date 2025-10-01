"""
Unit tests for SyntheticDataGenerator
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.app.routing.synthetic_data_generator import (
    SyntheticDataGenerator,
    ModalityExample,
)
from src.app.search.multi_modal_reranker import QueryModality


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator functionality"""

    @pytest.fixture
    def generator(self):
        """Create generator instance without Vespa client"""
        return SyntheticDataGenerator(vespa_client=None)

    @pytest.fixture
    def generator_with_vespa(self):
        """Create generator instance with mocked Vespa client"""
        mock_vespa = MagicMock()
        return SyntheticDataGenerator(vespa_client=mock_vespa)

    def test_initialization_without_vespa(self, generator):
        """Test initialization without Vespa client"""
        assert generator.vespa_client is None
        assert QueryModality.VIDEO in generator.fallback_topics
        assert len(generator.fallback_topics[QueryModality.VIDEO]) > 0

    def test_initialization_with_vespa(self, generator_with_vespa):
        """Test initialization with Vespa client"""
        assert generator_with_vespa.vespa_client is not None

    def test_modality_templates_exist(self, generator):
        """Test that templates exist for all modalities"""
        assert QueryModality.VIDEO in generator.MODALITY_TEMPLATES
        assert QueryModality.DOCUMENT in generator.MODALITY_TEMPLATES
        assert QueryModality.IMAGE in generator.MODALITY_TEMPLATES
        assert QueryModality.AUDIO in generator.MODALITY_TEMPLATES

        # Check templates are non-empty
        for modality, templates in generator.MODALITY_TEMPLATES.items():
            assert len(templates) > 0
            assert all("{topic}" in t for t in templates)

    def test_fallback_topics_exist(self, generator):
        """Test that fallback topics exist for all modalities"""
        assert QueryModality.VIDEO in generator.fallback_topics
        assert QueryModality.DOCUMENT in generator.fallback_topics
        assert QueryModality.IMAGE in generator.fallback_topics
        assert QueryModality.AUDIO in generator.fallback_topics

        # Check topics are non-empty
        for modality, topics in generator.fallback_topics.items():
            assert len(topics) > 0

    def test_infer_agent_from_modality(self, generator):
        """Test agent inference from modality"""
        assert generator._infer_agent_from_modality(QueryModality.VIDEO, "test") == "video_search_agent"
        assert generator._infer_agent_from_modality(QueryModality.DOCUMENT, "test") == "document_agent"
        assert generator._infer_agent_from_modality(QueryModality.IMAGE, "test") == "image_search_agent"
        assert generator._infer_agent_from_modality(QueryModality.AUDIO, "test") == "audio_analysis_agent"

    def test_extract_topics_from_content(self, generator):
        """Test topic extraction from content samples"""
        content_samples = [
            {
                "title": "Machine Learning Tutorial",
                "description": "Learn about neural networks and deep learning basics"
            },
            {
                "title": "Python Programming Guide",
                "description": "Introduction to python programming for data science"
            }
        ]

        topics = generator._extract_topics(content_samples)

        # Should extract some meaningful bigrams/trigrams
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_extract_topics_empty_content(self, generator):
        """Test topic extraction with empty content"""
        content_samples = []
        topics = generator._extract_topics(content_samples)
        assert topics == []

    def test_extract_entities_from_content(self, generator):
        """Test entity extraction from content samples"""
        content_samples = [
            {
                "title": "TensorFlow Tutorial",
                "description": "Using PyTorch and TensorFlow for Deep Learning"
            },
            {
                "title": "Python Programming",
                "description": "Learn Python with NumPy and Pandas"
            }
        ]

        entities = generator._extract_entities(content_samples)

        # Should extract capitalized entities
        assert isinstance(entities, list)
        # TensorFlow, PyTorch, Deep Learning, Python, NumPy, Pandas
        assert len(entities) > 0

    def test_extract_temporal_patterns(self, generator):
        """Test temporal pattern extraction"""
        content_samples = [
            {
                "title": "AI in 2024",
                "description": "Latest developments in 2023 and 2024",
                "timestamp": datetime(2024, 9, 1)
            },
            {
                "title": "Recent Advances",
                "description": "New techniques from 2023",
                "timestamp": datetime(2024, 10, 1)
            }
        ]

        temporal = generator._extract_temporal_patterns(content_samples)

        assert isinstance(temporal, list)
        assert "2024" in temporal or "2023" in temporal or "recent" in temporal

    def test_extract_content_types(self, generator):
        """Test content type extraction"""
        content_samples = [
            {
                "title": "Beginner's Tutorial",
                "description": "An introduction to machine learning"
            },
            {
                "title": "Advanced Guide",
                "description": "Research and analysis of neural networks"
            }
        ]

        content_types = generator._extract_content_types(content_samples)

        assert isinstance(content_types, list)
        # Should find: tutorial, beginner, introduction, advanced, guide, research, analysis
        assert len(content_types) > 0

    @pytest.mark.asyncio
    async def test_extract_patterns_without_vespa(self, generator):
        """Test pattern extraction falls back to defaults without Vespa"""
        patterns = await generator._extract_content_patterns(QueryModality.VIDEO)

        assert "topics" in patterns
        assert "entities" in patterns
        assert "temporal" in patterns
        assert "content_types" in patterns

        # Should use fallback topics
        assert len(patterns["topics"]) > 0
        assert "machine learning" in patterns["topics"] or "neural networks" in patterns["topics"]

    def test_generate_example_from_patterns_basic(self, generator):
        """Test generating example from patterns"""
        patterns = {
            "topics": ["machine learning", "neural networks"],
            "entities": ["TensorFlow", "PyTorch"],
            "temporal": ["2024", "recent"],
            "content_types": ["tutorial"]
        }

        example = generator._generate_example_from_patterns(QueryModality.VIDEO, patterns)

        assert isinstance(example, ModalityExample)
        assert example.modality == QueryModality.VIDEO
        assert example.is_synthetic is True
        assert example.synthetic_source == "ingested_content"
        assert example.success is True
        assert len(example.query) > 0
        assert example.correct_agent == "video_search_agent"

    def test_generate_example_query_structure(self, generator):
        """Test that generated queries have proper structure"""
        patterns = {
            "topics": ["deep learning"],
            "entities": [],
            "temporal": [],
            "content_types": []
        }

        # Generate multiple examples to check variety
        examples = [
            generator._generate_example_from_patterns(QueryModality.VIDEO, patterns)
            for _ in range(10)
        ]

        # All should contain the topic
        for example in examples:
            assert "deep learning" in example.query.lower()

        # Should have some variety in templates
        unique_queries = set(ex.query for ex in examples)
        assert len(unique_queries) > 1  # Not all the same

    def test_generate_example_different_modalities(self, generator):
        """Test generating examples for different modalities"""
        patterns = {
            "topics": ["machine learning"],
            "entities": [],
            "temporal": [],
            "content_types": []
        }

        video_ex = generator._generate_example_from_patterns(QueryModality.VIDEO, patterns)
        doc_ex = generator._generate_example_from_patterns(QueryModality.DOCUMENT, patterns)
        image_ex = generator._generate_example_from_patterns(QueryModality.IMAGE, patterns)
        audio_ex = generator._generate_example_from_patterns(QueryModality.AUDIO, patterns)

        # Different modalities should get different agents
        assert video_ex.correct_agent == "video_search_agent"
        assert doc_ex.correct_agent == "document_agent"
        assert image_ex.correct_agent == "image_search_agent"
        assert audio_ex.correct_agent == "audio_analysis_agent"

        # Queries should reflect modality (at least contain the topic)
        # Note: Not all templates have modality-specific words, but agents are correct
        assert "machine learning" in video_ex.query.lower()
        assert "machine learning" in doc_ex.query.lower()
        assert "machine learning" in image_ex.query.lower()
        assert "machine learning" in audio_ex.query.lower()

    @pytest.mark.asyncio
    async def test_generate_from_ingested_data_basic(self, generator):
        """Test generating synthetic data without Vespa"""
        examples = await generator.generate_from_ingested_data(
            QueryModality.VIDEO,
            target_count=10
        )

        assert len(examples) == 10
        assert all(isinstance(ex, ModalityExample) for ex in examples)
        assert all(ex.modality == QueryModality.VIDEO for ex in examples)
        assert all(ex.is_synthetic for ex in examples)
        assert all(ex.correct_agent == "video_search_agent" for ex in examples)

    @pytest.mark.asyncio
    async def test_generate_from_ingested_data_variety(self, generator):
        """Test that generated data has variety"""
        examples = await generator.generate_from_ingested_data(
            QueryModality.DOCUMENT,
            target_count=20
        )

        # Should have variety in queries
        unique_queries = set(ex.query for ex in examples)
        assert len(unique_queries) > 5  # At least some variety

        # All should be valid
        assert all(len(ex.query) > 0 for ex in examples)

    def test_generate_validation_set(self, generator):
        """Test generating validation set"""
        validation_set = generator.generate_validation_set(QueryModality.VIDEO, size=10)

        assert len(validation_set) == 10
        assert all(isinstance(ex, ModalityExample) for ex in validation_set)
        assert all(ex.is_synthetic for ex in validation_set)
        assert all(ex.modality == QueryModality.VIDEO for ex in validation_set)

    @pytest.mark.asyncio
    async def test_generate_with_vespa_client(self, generator_with_vespa):
        """Test generation with mocked Vespa client"""
        # Mock Vespa response
        mock_response = {
            "root": {
                "children": [
                    {
                        "fields": {
                            "title": "Neural Networks Tutorial 2024",
                            "description": "Learn about deep learning with TensorFlow",
                            "metadata": {},
                            "timestamp": datetime(2024, 9, 1).isoformat()
                        }
                    },
                    {
                        "fields": {
                            "title": "Machine Learning Basics",
                            "description": "Introduction to ML using PyTorch",
                            "metadata": {},
                            "timestamp": datetime(2024, 8, 1).isoformat()
                        }
                    }
                ]
            }
        }

        generator_with_vespa.vespa_client.query = AsyncMock(return_value=mock_response)

        examples = await generator_with_vespa.generate_from_ingested_data(
            QueryModality.VIDEO,
            target_count=5
        )

        assert len(examples) == 5
        # Should have used patterns from Vespa content
        # (topics like "neural networks", "machine learning", entities like "TensorFlow", "PyTorch")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
