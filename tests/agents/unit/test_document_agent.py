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

    def test_select_strategy_visual(self):
        """Test strategy selection for visual queries"""
        queries = [
            "show me diagrams about neural networks",
            "find charts comparing performance",
            "what does the table show",
        ]
        for query in queries:
            strategy = self.agent._select_strategy(query)
            assert strategy == "visual", (
                f"Expected visual for '{query}', got {strategy}"
            )

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
        """Test visual search strategy.

        End-to-end retrieval against real Vespa is covered by
        test_document_agent_visual_search_vespa.py; this unit guards the query
        contract the document_visual schema declares (float_float profile,
        query(qt) mapped tensor, tenant-scoped schema) and the document_path
        parse.
        """
        # ColPali query encoder → 2 per-token (patch) 128-d vectors.
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.zeros((2, 128), dtype=np.float32)
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
                            "document_path": "s3://example/doc1.pdf",
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
        assert results[0].document_url == "s3://example/doc1.pdf"
        assert results[0].title == "Machine Learning Basics"
        assert results[0].page_number == 1
        assert results[0].strategy_used == "visual"
        assert results[0].relevance_score == 0.95

        # The query must match the document_visual schema's declared contract:
        # the phased profile (binary recall + float rerank) with both the float
        # query(qt) and binarized query(qtb) mapped tensors, and the
        # tenant-scoped schema name.
        mock_post.assert_called_once()
        sent = mock_post.call_args.kwargs["json"]
        assert sent["ranking.profile"] == "phased"
        assert isinstance(sent["input.query(qt)"], dict)
        assert isinstance(sent["input.query(qtb)"], dict)
        # qtb is the packed binary form: 128 dims -> 16 int8 bytes per token.
        assert all(len(v) == 16 for v in sent["input.query(qtb)"].values())
        assert "document_visual_" in sent["yql"]
        assert "str(" not in str(sent["input.query(qt)"])

    @pytest.mark.asyncio
    @patch.object(DocumentAgent, "query_encoder", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_visual_flat_qt_for_single_vector(
        self, mock_post, mock_query_encoder
    ):
        """A (dim,) query embedding must serialize as a flat list, not a
        dict of scalars keyed by element index."""
        emb = np.full(128, -1.0, dtype=np.float32)
        emb[0] = 1.0
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = emb
        mock_query_encoder.return_value = mock_encoder

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        await self.agent._search_visual("q", limit=5)

        sent = mock_post.call_args.kwargs["json"]
        assert sent["input.query(qt)"] == [1.0] + [-1.0] * 127
        # Packed binary of [1, 0, ..., 0]: first byte 0b10000000 as int8.
        assert sent["input.query(qtb)"] == [-128] + [0] * 15

    @pytest.mark.asyncio
    @patch.object(DocumentAgent, "query_encoder", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_visual_dict_qt_for_multivector(
        self, mock_post, mock_query_encoder
    ):
        """A (N, dim) query embedding serializes as {token_index: vector} with
        one packed-binary row per token."""
        emb = np.zeros((2, 128), dtype=np.float32)
        emb[0, 0] = 1.0
        emb[1, 8] = 1.0
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = emb
        mock_query_encoder.return_value = mock_encoder

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        await self.agent._search_visual("q", limit=5)

        sent = mock_post.call_args.kwargs["json"]
        row0 = [1.0] + [0.0] * 127
        row1 = [0.0] * 8 + [1.0] + [0.0] * 119
        assert sent["input.query(qt)"] == {"0": row0, "1": row1}
        assert sent["input.query(qtb)"] == {
            "0": [-128] + [0] * 15,
            "1": [0, -128] + [0] * 14,
        }

    @pytest.mark.asyncio
    @patch.object(DocumentAgent, "text_query_encoder", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_text_flat_qt_for_single_vector(self, mock_post, mock_encoder):
        """A (dim,) text query embedding must serialize as a flat list."""
        emb = np.full(128, 0.5, dtype=np.float32)
        emb[3] = -0.5
        enc = MagicMock()
        enc.encode.return_value = emb
        mock_encoder.return_value = enc

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"root": {"children": []}}
        mock_post.return_value = mock_response

        await self.agent._search_text("q", limit=5)

        sent = mock_post.call_args.kwargs["json"]
        expected = [0.5] * 128
        expected[3] = -0.5
        assert sent["input.query(qt)"] == expected

    @pytest.mark.asyncio
    @patch.object(DocumentAgent, "text_query_encoder", new_callable=PropertyMock)
    @patch("requests.post")
    async def test_search_text(self, mock_post, mock_encoder):
        """Test text search strategy.

        End-to-end retrieval against real Vespa is covered by
        test_document_agent_text_search_vespa.py; this unit guards the query
        contract the document_text schema declares (profile, input name,
        tenant-scoped schema) and the document_path parse.
        """
        # ColBERT query encoder → 2 per-token 128-d vectors.
        enc = MagicMock()
        enc.encode.return_value = np.zeros((2, 128), dtype=np.float32)
        mock_encoder.return_value = enc

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
                            "document_path": "s3://example/doc2.pdf",
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
        assert results[0].document_url == "s3://example/doc2.pdf"
        assert results[0].title == "Deep Learning Theory"
        assert results[0].strategy_used == "text"
        assert results[0].relevance_score == 0.88
        assert "Deep learning" in results[0].content_preview

        # The query must match the document_text schema's declared contract:
        # the hybrid_float_bm25 profile, the query(qt) mapped tensor input, and
        # the tenant-scoped schema name.
        sent = mock_post.call_args.kwargs["json"]
        assert sent["ranking.profile"] == "hybrid_float_bm25"
        assert "input.query(qt)" in sent
        assert isinstance(sent["input.query(qt)"], dict)
        assert "document_text_" in sent["yql"]
        assert "hybrid_bm25_semantic" not in str(mock_post.call_args)

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


class TestSearchDocumentsEventLoop:
    async def test_slow_memory_recall_does_not_block_event_loop(self):
        """Mem0 recall is a synchronous LLM/vector round trip —
        search_documents must run it off the loop (the orchestrator already
        does); a blocked loop stalls every concurrent request on the worker."""
        import asyncio
        import time as _time

        agent = DocumentAgent(
            deps=DocumentAgentDeps(
                tenant_id="test_tenant",
                vespa_endpoint="http://localhost:8080",
            ),
            port=8007,
        )
        agent.is_memory_enabled = lambda: True

        def slow_recall(query, top_k=3):
            _time.sleep(0.2)
            return ""

        agent.get_relevant_context = slow_recall

        async def fake_text_search(query, limit):
            return []

        agent._search_text = fake_text_search

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        results = await agent.search_documents("cats", strategy="text", limit=1)
        task.cancel()

        assert results == []
        assert ticks >= 5, f"event loop starved during recall: {ticks} ticks"
