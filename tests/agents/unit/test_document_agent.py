"""
Unit tests for Document Agent

Tests dual-strategy document search with ColPali and text embeddings.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from cogniverse_agents.document_agent import (
    DocumentAgent,
    DocumentAgentDeps,
    DocumentResult,
)


class TestDocumentAgent:
    """Unit tests for DocumentAgent"""

    def setup_method(self):
        """Set up test fixtures"""
        self.agent = DocumentAgent(
            deps=DocumentAgentDeps(
                tenant_id="test_tenant",
                vespa_endpoint="http://localhost:8080",
            ),
            port=8007,
        )

    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent is not None
        assert self.agent._colpali_model is None  # Lazy loaded
        assert self.agent._text_embedding_model is None  # Lazy loaded
        assert self.agent._vespa_endpoint == "http://localhost:8080"

    @patch("cogniverse_agents.document_agent.get_or_load_model")
    def test_colpali_model_lazy_loading(self, mock_get_model):
        """Test ColPali model is lazy loaded on first access"""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_get_model.return_value = (mock_model, mock_processor)

        # Access colpali_model property
        model = self.agent.colpali_model

        # Verify model was loaded
        assert model is not None
        assert mock_get_model.called

    @patch("sentence_transformers.SentenceTransformer")
    def test_text_embedding_model_lazy_loading(self, mock_sentence_transformer):
        """Test text embedding model is lazy loaded"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Access text_embedding_model property
        model = self.agent.text_embedding_model

        # Verify model was loaded
        assert model is not None
        mock_sentence_transformer.assert_called_once_with(
            "sentence-transformers/all-mpnet-base-v2"
        )

    def test_select_strategy_visual(self):
        """Test strategy selection for visual queries"""
        queries = [
            "show me diagrams about neural networks",
            "find charts comparing performance",
            "what does the table show",
        ]
        for query in queries:
            strategy = self.agent._select_strategy(query)
            assert (
                strategy == "visual"
            ), f"Expected visual for '{query}', got {strategy}"

    def test_select_strategy_text(self):
        """Test strategy selection for text queries"""
        queries = [
            "explain the concept of machine learning",
            "list all definitions in the document",
            "who is the author of this paper",
        ]
        for query in queries:
            strategy = self.agent._select_strategy(query)
            assert strategy == "text", f"Expected text for '{query}', got {strategy}"

    def test_select_strategy_hybrid(self):
        """Test strategy selection defaults to hybrid for uncertain queries"""
        query = "machine learning applications"
        strategy = self.agent._select_strategy(query)
        assert strategy == "hybrid"

    @pytest.mark.asyncio
    @patch.object(DocumentAgent, "query_encoder", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_visual(self, mock_post, mock_query_encoder):
        """Test visual search strategy"""
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
                            "document_id": "doc_001_p1",
                            "source_url": "http://example.com/doc1.pdf",
                            "document_title": "Machine Learning Basics",
                            "page_number": 1,
                            "page_count": 10,
                            "document_type": "pdf",
                        },
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        # Execute visual search
        results = await self.agent._search_visual("neural networks diagram", limit=20)

        # Verify results
        assert len(results) == 1
        assert results[0].document_id == "doc_001_p1"
        assert results[0].title == "Machine Learning Basics"
        assert results[0].page_number == 1
        assert results[0].strategy_used == "visual"
        assert results[0].relevance_score == 0.95

        # Verify Vespa was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "colpali" in str(call_args)

    @pytest.mark.asyncio
    @patch.object(DocumentAgent, "text_embedding_model", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_text(self, mock_post, mock_text_model):
        """Test text search strategy"""
        # Mock text embedding model
        mock_model = MagicMock()
        mock_embedding = np.random.randn(768)
        mock_model.encode.return_value = mock_embedding
        mock_text_model.return_value = mock_model

        # Mock Vespa response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "root": {
                "children": [
                    {
                        "relevance": 0.88,
                        "fields": {
                            "document_id": "doc_002",
                            "source_url": "http://example.com/doc2.pdf",
                            "document_title": "Deep Learning Theory",
                            "page_count": 25,
                            "document_type": "pdf",
                            "full_text": "Deep learning is a subset of machine learning...",
                        },
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        # Execute text search
        results = await self.agent._search_text("definition of deep learning", limit=20)

        # Verify results
        assert len(results) == 1
        assert results[0].document_id == "doc_002"
        assert results[0].title == "Deep Learning Theory"
        assert results[0].strategy_used == "text"
        assert results[0].relevance_score == 0.88
        assert "Deep learning" in results[0].content_preview

        # Verify Vespa was called with hybrid_bm25_semantic profile
        call_args = mock_post.call_args
        assert "hybrid_bm25_semantic" in str(call_args)

    @pytest.mark.asyncio
    async def test_search_hybrid(self):
        """Test hybrid search combines both strategies"""
        # Mock both search methods
        visual_result = DocumentResult(
            document_id="doc_001",
            document_url="http://example.com/doc1.pdf",
            title="Doc 1",
            relevance_score=0.9,
            strategy_used="visual",
        )
        text_result = DocumentResult(
            document_id="doc_002",
            document_url="http://example.com/doc2.pdf",
            title="Doc 2",
            relevance_score=0.8,
            strategy_used="text",
        )

        with patch.object(self.agent, "_search_visual", return_value=[visual_result]):
            with patch.object(self.agent, "_search_text", return_value=[text_result]):
                results = await self.agent._search_hybrid("test query", limit=20)

                # Verify both strategies were called
                assert len(results) == 2
                # Results should be marked as hybrid
                for result in results:
                    assert result.strategy_used == "hybrid"

    def test_fuse_results(self):
        """Test Reciprocal Rank Fusion algorithm"""
        visual_results = [
            DocumentResult(
                document_id="doc_001",
                document_url="url1",
                title="Doc 1",
                relevance_score=0.9,
            ),
            DocumentResult(
                document_id="doc_002",
                document_url="url2",
                title="Doc 2",
                relevance_score=0.8,
            ),
        ]
        text_results = [
            DocumentResult(
                document_id="doc_002",
                document_url="url2",
                title="Doc 2",
                relevance_score=0.85,
            ),
            DocumentResult(
                document_id="doc_003",
                document_url="url3",
                title="Doc 3",
                relevance_score=0.7,
            ),
        ]

        fused = self.agent._fuse_results(visual_results, text_results, limit=10, k=60)

        # doc_002 appears in both lists, should be ranked highest
        assert len(fused) == 3
        assert fused[0].document_id == "doc_002"  # Appears in both, highest RRF score

    @pytest.mark.asyncio
    async def test_search_documents_auto_strategy(self):
        """Test auto-strategy selection in search_documents"""
        with patch.object(self.agent, "_select_strategy", return_value="visual"):
            with patch.object(
                self.agent, "_search_visual", return_value=[]
            ) as mock_visual:
                await self.agent.search_documents(
                    "diagram of neural network", strategy="auto"
                )
                mock_visual.assert_called_once()

    @pytest.mark.asyncio
    async def test_dspy_to_a2a_output(self):
        """Test DSPy output to A2A format conversion"""
        results = [
            DocumentResult(
                document_id="doc_001",
                document_url="http://example.com/doc1.pdf",
                title="Test Document",
                page_number=1,
                page_count=10,
                document_type="pdf",
                content_preview="Preview text",
                relevance_score=0.95,
                strategy_used="hybrid",
            )
        ]

        a2a_output = self.agent._dspy_to_a2a_output({"results": results, "count": 1})

        # Verify format
        assert a2a_output["status"] == "success"
        assert a2a_output["result_type"] == "document_search_results"
        assert a2a_output["count"] == 1
        assert len(a2a_output["results"]) == 1
        assert a2a_output["results"][0]["document_id"] == "doc_001"
        assert a2a_output["results"][0]["strategy_used"] == "hybrid"

    def test_get_agent_skills(self):
        """Test agent skills definition"""
        skills = self.agent._get_agent_skills()

        # Verify skills
        assert len(skills) >= 1
        skill_names = [s["name"] for s in skills]
        assert "search_documents" in skill_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
