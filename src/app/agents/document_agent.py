"""
Document Analysis Agent with Dual Strategy

Implements two parallel approaches for document search:
1. Visual Strategy (ColPali): Treats document pages as images
2. Text Strategy: Traditional text extraction + semantic embeddings

Enables comparison and auto-strategy selection based on query type.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dspy

from src.app.agents.dspy_a2a_agent_base import DSPyA2AAgentBase
from src.app.agents.memory_aware_mixin import MemoryAwareMixin
from src.app.agents.query_encoders import ColPaliQueryEncoder
from src.common.models.model_loaders import get_or_load_model

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """Result from document search"""

    document_id: str
    document_url: str
    title: str
    page_number: Optional[int] = None
    page_count: Optional[int] = None
    document_type: str = "pdf"
    content_preview: str = ""
    relevance_score: float = 0.0
    strategy_used: str = "unknown"  # visual, text, or hybrid
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentAgent(MemoryAwareMixin, DSPyA2AAgentBase):
    """
    Document analysis and search with dual strategy support
    Enhanced with memory capabilities for learning from search patterns.

    Implements both:
    - ColPali visual document understanding (page-as-image)
    - Traditional text extraction + semantic search

    Enables comparison and auto-strategy selection based on query type.
    """

    def __init__(
        self,
        tenant_id: str,
        vespa_endpoint: str = "http://localhost:8080",
        colpali_model: str = "vidore/colsmol-500m",
        port: int = 8007,
    ):
        """
        Initialize Document Agent with dual strategy support

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            vespa_endpoint: Vespa endpoint URL
            colpali_model: ColPali model for visual strategy
            port: A2A server port

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        # Initialize memory mixin first
        MemoryAwareMixin.__init__(self)
        self.tenant_id = tenant_id

        # Create DSPy module
        class DocumentSearchSignature(dspy.Signature):
            query: str = dspy.InputField(desc="Document search query")
            strategy: str = dspy.InputField(
                desc="Search strategy: visual, text, hybrid, auto"
            )
            result: str = dspy.OutputField(desc="Search results")

        class DocumentSearchModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, query: str, strategy: str = "auto"):
                return dspy.Prediction(
                    result=f"Searching documents: {query} (strategy: {strategy})"
                )

        # Initialize A2A base
        DSPyA2AAgentBase.__init__(
            self,
            agent_name="DocumentAgent",
            agent_description="Document search with visual and text strategies",
            dspy_module=DocumentSearchModule(),
            capabilities=[
                "document_search",
                "visual_search",
                "text_search",
                "hybrid_search",
            ],
            port=port,
            version="1.0.0",
        )

        # Initialize memory for document agent
        memory_initialized = self.initialize_memory(
            agent_name="document_agent",
            tenant_id=tenant_id,
        )
        if memory_initialized:
            logger.info(
                f"✅ Memory initialized for document_agent (tenant: {tenant_id})"
            )
        else:
            logger.info("ℹ️  Memory disabled or not configured for document_agent")

        self._vespa_endpoint = vespa_endpoint
        self._colpali_model_name = colpali_model

        # Lazy load models
        self._colpali_model = None
        self._colpali_processor = None
        self._query_encoder = None
        self._text_embedding_model = None

        logger.info("📄 Initialized DocumentAgent with dual strategy support")

    @property
    def colpali_model(self):
        """Lazy load ColPali model for visual strategy"""
        if self._colpali_model is None:
            logger.info(f"Loading ColPali model: {self._colpali_model_name}")
            config = {"colpali_model": self._colpali_model_name}
            self._colpali_model, self._colpali_processor = get_or_load_model(
                self._colpali_model_name, config, logger
            )
            logger.info("✅ ColPali model loaded")
        return self._colpali_model

    @property
    def colpali_processor(self):
        """Get ColPali processor"""
        if self._colpali_processor is None:
            _ = self.colpali_model  # Trigger load
        return self._colpali_processor

    @property
    def query_encoder(self):
        """Get ColPali query encoder for visual strategy"""
        if self._query_encoder is None:
            self._query_encoder = ColPaliQueryEncoder(
                model_name=self._colpali_model_name
            )
        return self._query_encoder

    @property
    def text_embedding_model(self):
        """Lazy load text embedding model for text strategy"""
        if self._text_embedding_model is None:
            logger.info("Loading text embedding model...")
            from sentence_transformers import SentenceTransformer

            self._text_embedding_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            )
            logger.info("✅ Text embedding model loaded")
        return self._text_embedding_model

    async def search_documents(
        self,
        query: str,
        strategy: str = "auto",
        limit: int = 20,
    ) -> List[DocumentResult]:
        """
        Search documents using specified or auto-selected strategy

        Args:
            query: User query
            strategy: "visual", "text", "hybrid", or "auto"
            limit: Number of results

        Returns:
            List of DocumentResult with relevance scores
        """
        logger.info(f"🔍 Searching documents: query='{query}', strategy={strategy}")

        # Retrieve relevant search patterns from memory
        if self.is_memory_enabled():
            memory_context = self.get_relevant_context(query, top_k=3)
            if memory_context:
                logger.info(f"📚 Retrieved memory context for query: {query[:50]}...")

        # Auto-select strategy if needed
        if strategy == "auto":
            strategy = self._select_strategy(query)
            logger.info(f"📊 Auto-selected strategy: {strategy}")

        try:
            if strategy == "visual":
                results = await self._search_visual(query, limit)
            elif strategy == "text":
                results = await self._search_text(query, limit)
            elif strategy == "hybrid":
                results = await self._search_hybrid(query, limit)
            else:
                # Default to hybrid
                results = await self._search_hybrid(query, limit)

            logger.info(f"✅ Found {len(results)} document results")

            # Store successful search in memory
            if self.is_memory_enabled() and results:
                self.remember_success(
                    query=query,
                    result={
                        "result_count": len(results),
                        "strategy": strategy,
                        "top_result": results[0].title if results else None,
                    },
                    metadata={
                        "search_strategy": strategy,
                        "limit": limit,
                    },
                )
                logger.debug("💾 Stored successful document search in memory")

            return results

        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")

            # Store failure in memory
            if self.is_memory_enabled():
                self.remember_failure(
                    query=query,
                    error=str(e),
                    metadata={"search_strategy": strategy, "limit": limit},
                )
                logger.debug("💾 Stored document search failure in memory")

            return []

    def _select_strategy(self, query: str) -> str:
        """
        Auto-select strategy based on query characteristics

        Visual strategy preferred for:
        - Queries about charts, diagrams, tables, layouts
        - Visual questions
        - Queries mentioning visual elements

        Text strategy preferred for:
        - Keyword searches
        - Named entity queries
        - Exact phrase matching
        - Long-form semantic queries

        Hybrid for complex or uncertain queries
        """
        visual_keywords = [
            "chart",
            "diagram",
            "table",
            "figure",
            "graph",
            "layout",
            "screenshot",
            "visual",
            "image",
            "picture",
            "illustration",
            "drawing",
            "map",
            "plot",
        ]
        text_keywords = [
            "definition",
            "explain",
            "list",
            "summary",
            "mention",
            "quote",
            "section",
            "paragraph",
            "author",
            "reference",
            "citation",
            "abstract",
        ]

        query_lower = query.lower()

        visual_score = sum(1 for kw in visual_keywords if kw in query_lower)
        text_score = sum(1 for kw in text_keywords if kw in query_lower)

        if visual_score > text_score:
            return "visual"
        elif text_score > visual_score:
            return "text"
        else:
            return "hybrid"

    async def _search_visual(self, query: str, limit: int) -> List[DocumentResult]:
        """
        ColPali visual search - treats pages as images

        Advantages:
        - Preserves layout, tables, charts, diagrams
        - No text extraction needed
        - Handles complex layouts better
        - Visual question answering capability
        """
        import requests

        logger.info("🖼️  Using visual strategy (ColPali page-as-image)")

        # Encode query with ColPali
        query_embedding = self.query_encoder.encode(query)

        # Build Vespa query for document_visual schema
        yql = "select * from document_visual where true"
        query_embedding_flat = query_embedding.flatten().tolist()

        params = {
            "yql": yql,
            "hits": limit,
            "ranking.profile": "colpali",
            "input.query(qt)": str(query_embedding_flat),
        }

        # Execute search
        response = requests.post(
            f"{self._vespa_endpoint}/search/", json=params, timeout=10
        )

        if response.status_code != 200:
            logger.error(
                f"Vespa search failed: {response.status_code} - {response.text}"
            )
            return []

        # Parse results
        results = []
        data = response.json()

        for hit in data.get("root", {}).get("children", []):
            fields = hit.get("fields", {})
            results.append(
                DocumentResult(
                    document_id=fields.get("document_id", ""),
                    document_url=fields.get("source_url", ""),
                    title=fields.get("document_title", ""),
                    page_number=fields.get("page_number"),
                    page_count=fields.get("page_count"),
                    document_type=fields.get("document_type", "pdf"),
                    relevance_score=hit.get("relevance", 0.0),
                    strategy_used="visual",
                )
            )

        return results

    async def _search_text(self, query: str, limit: int) -> List[DocumentResult]:
        """
        Traditional text-based search

        Advantages:
        - Better for keyword/phrase matching
        - Named entity queries
        - Exact text searches
        - Lower latency
        """
        import requests

        logger.info("📝 Using text strategy (extraction + semantic)")

        # Generate text embedding for query
        query_embedding = self.text_embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Build Vespa query for document_text schema
        yql = "select * from document_text where userQuery()"

        params = {
            "yql": yql,
            "query": query,
            "hits": limit,
            "ranking.profile": "hybrid_bm25_semantic",
            "input.query(q)": query_embedding.tolist(),
        }

        # Execute search
        response = requests.post(
            f"{self._vespa_endpoint}/search/", json=params, timeout=10
        )

        if response.status_code != 200:
            logger.error(
                f"Vespa search failed: {response.status_code} - {response.text}"
            )
            return []

        # Parse results
        results = []
        data = response.json()

        for hit in data.get("root", {}).get("children", []):
            fields = hit.get("fields", {})
            results.append(
                DocumentResult(
                    document_id=fields.get("document_id", ""),
                    document_url=fields.get("source_url", ""),
                    title=fields.get("document_title", ""),
                    page_count=fields.get("page_count"),
                    document_type=fields.get("document_type", "pdf"),
                    content_preview=fields.get("full_text", "")[:200],
                    relevance_score=hit.get("relevance", 0.0),
                    strategy_used="text",
                )
            )

        return results

    async def _search_hybrid(self, query: str, limit: int) -> List[DocumentResult]:
        """
        Combine visual and text search results using Reciprocal Rank Fusion

        RRF score = sum(1 / (k + rank_i)) across all ranking lists
        """
        logger.info("🔀 Using hybrid strategy (visual + text fusion)")

        # Execute both searches in parallel
        visual_results = await self._search_visual(query, limit)
        text_results = await self._search_text(query, limit)

        # Apply Reciprocal Rank Fusion
        fused_results = self._fuse_results(visual_results, text_results, limit)

        # Mark as hybrid
        for result in fused_results:
            result.strategy_used = "hybrid"

        return fused_results

    def _fuse_results(
        self,
        visual_results: List[DocumentResult],
        text_results: List[DocumentResult],
        limit: int,
        k: int = 60,
    ) -> List[DocumentResult]:
        """
        Reciprocal Rank Fusion algorithm

        RRF score = sum(1 / (k + rank_i)) across all ranking lists

        Args:
            visual_results: Results from visual strategy
            text_results: Results from text strategy
            limit: Maximum results to return
            k: RRF constant (default 60)

        Returns:
            Fused and sorted results
        """
        scores = {}
        doc_objects = {}

        # Add visual results
        for rank, result in enumerate(visual_results, 1):
            doc_id = result.document_id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in doc_objects:
                doc_objects[doc_id] = result

        # Add text results
        for rank, result in enumerate(text_results, 1):
            doc_id = result.document_id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in doc_objects:
                doc_objects[doc_id] = result

        # Sort by RRF score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top results with updated scores
        results = []
        for doc_id, rrf_score in fused[:limit]:
            doc = doc_objects[doc_id]
            doc.relevance_score = rrf_score
            results.append(doc)

        return results

    # DSPyA2AAgentBase abstract method implementations
    async def _process_with_dspy(self, dspy_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process document search request"""
        query = dspy_input.get("query", "")
        strategy = dspy_input.get("strategy", "auto")
        limit = dspy_input.get("limit", 20)

        results = await self.search_documents(
            query=query, strategy=strategy, limit=limit
        )

        return {"results": results, "count": len(results)}

    def _dspy_to_a2a_output(self, dspy_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DSPy output to A2A format"""
        results = dspy_output.get("results", [])

        return {
            "status": "success",
            "result_type": "document_search_results",
            "count": len(results),
            "results": [
                {
                    "document_id": r.document_id,
                    "document_url": r.document_url,
                    "title": r.title,
                    "page_number": r.page_number,
                    "page_count": r.page_count,
                    "document_type": r.document_type,
                    "content_preview": r.content_preview,
                    "relevance_score": r.relevance_score,
                    "strategy_used": r.strategy_used,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define agent skills for A2A protocol"""
        return [
            {
                "name": "search_documents",
                "description": "Search documents using visual, text, or hybrid strategies",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "strategy": {
                        "type": "string",
                        "required": False,
                        "default": "auto",
                    },
                    "limit": {"type": "integer", "required": False, "default": 20},
                },
            },
        ]


if __name__ == "__main__":
    import asyncio

    async def test_agent():
        agent = DocumentAgent()
        results = await agent.search_documents(
            "machine learning algorithms", strategy="auto"
        )
        print(f"Found {len(results)} results")
        for r in results[:3]:
            print(
                f"  - {r.title} (strategy: {r.strategy_used}, score: {r.relevance_score:.3f})"
            )

    asyncio.run(test_agent())
