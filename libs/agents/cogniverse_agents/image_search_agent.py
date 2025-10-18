"""
Image Search Agent using ColPali

Uses ColPali multi-vector embeddings for image similarity search,
same approach as video frames. Connects to Vespa for real search.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dspy
import numpy as np
import torch
from cogniverse_core.agents.dspy_a2a_base import DSPyA2AAgentBase
from cogniverse_core.common.models.model_loaders import get_or_load_model
from PIL import Image

from cogniverse_agents.query.encoders import ColPaliQueryEncoder

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """Result from image search"""

    image_id: str
    image_url: str
    title: str
    description: str
    relevance_score: float
    detected_objects: List[str] = field(default_factory=list)
    detected_scenes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageSearchAgent(DSPyA2AAgentBase):
    """
    Image search using ColPali multi-vector embeddings

    Capabilities:
    - Image similarity search using ColPali (same as video frames)
    - Hybrid search (BM25 text + ColPali semantic)
    - Image-to-image similarity search
    - Real Vespa backend integration
    """

    def __init__(
        self,
        tenant_id: str,
        vespa_endpoint: str = "http://localhost:8080",
        colpali_model: str = "vidore/colsmol-500m",
        port: int = 8005,
    ):
        """
        Initialize Image Search Agent

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            vespa_endpoint: Vespa endpoint URL
            colpali_model: ColPali model name
            port: A2A server port

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        self.tenant_id = tenant_id

        # Create DSPy module
        class ImageSearchSignature(dspy.Signature):
            query: str = dspy.InputField(desc="Image search query")
            mode: str = dspy.InputField(desc="Search mode: semantic, hybrid")
            result: str = dspy.OutputField(desc="Search results")

        class ImageSearchModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, query: str, mode: str = "semantic"):
                return dspy.Prediction(
                    result=f"Searching images: {query} (mode: {mode})"
                )

        # Initialize A2A base
        super().__init__(
            agent_name="ImageSearchAgent",
            agent_description="Image search using ColPali multi-vector embeddings",
            dspy_module=ImageSearchModule(),
            capabilities=["image_search", "image_similarity", "hybrid_search"],
            port=port,
            version="1.0.0",
        )

        self._vespa_endpoint = vespa_endpoint
        self._colpali_model_name = colpali_model

        # Lazy load models
        self._colpali_model = None
        self._colpali_processor = None
        self._query_encoder = None

        logger.info("🖼️  Initialized ImageSearchAgent with ColPali")

    @property
    def colpali_model(self):
        """Lazy load ColPali model"""
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
            # Trigger model loading
            _ = self.colpali_model
        return self._colpali_processor

    @property
    def query_encoder(self):
        """Get query encoder"""
        if self._query_encoder is None:
            self._query_encoder = ColPaliQueryEncoder(
                model_name=self._colpali_model_name
            )
        return self._query_encoder

    async def search_images(
        self,
        query: str,
        search_mode: str = "semantic",
        limit: int = 20,
        visual_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ImageResult]:
        """
        Search images using ColPali embeddings

        Args:
            query: Text query
            search_mode: "semantic" (ColPali only) or "hybrid" (BM25 + ColPali)
            limit: Number of results
            visual_filters: Optional filters (objects, scenes)

        Returns:
            List of ImageResult with relevance scores
        """
        logger.info(f"🔍 Searching images: query='{query}', mode={search_mode}")

        try:
            # Encode query with ColPali
            query_embedding = self.query_encoder.encode(query)

            # Search Vespa
            results = await self._search_vespa(
                query_embedding=query_embedding,
                query_text=query,
                search_mode=search_mode,
                limit=limit,
                filters=visual_filters,
            )

            logger.info(f"✅ Found {len(results)} image results")
            return results

        except Exception as e:
            logger.error(f"❌ Image search failed: {e}")
            return []

    async def find_similar_images(
        self,
        reference_image: Image.Image,
        limit: int = 20,
    ) -> List[ImageResult]:
        """
        Find visually similar images using ColPali

        Args:
            reference_image: PIL Image object
            limit: Number of results

        Returns:
            List of similar images
        """
        logger.info("🔍 Finding similar images")

        try:
            # Encode image with ColPali
            image_embedding = self._encode_image(reference_image)

            # Search Vespa with image embedding
            results = await self._search_vespa(
                query_embedding=image_embedding,
                query_text="",
                search_mode="semantic",
                limit=limit,
            )

            logger.info(f"✅ Found {len(results)} similar images")
            return results

        except Exception as e:
            logger.error(f"❌ Similar image search failed: {e}")
            return []

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode image using ColPali

        Args:
            image: PIL Image

        Returns:
            ColPali multi-vector embedding [1024, 128]
        """
        # Process image with ColPali (reusing existing pattern from embedding_generator)
        batch_inputs = self.colpali_processor.process_images([image]).to(
            self.colpali_model.device
        )

        # Get embeddings
        with torch.no_grad():
            embeddings = self.colpali_model(**batch_inputs)  # Returns tensor directly

        # Reshape to [1024, 128] format (remove batch dimension)
        embeddings_np = embeddings.squeeze(0).cpu().numpy()

        # Pad or truncate to exactly 1024 patches
        if embeddings_np.shape[0] < 1024:
            padding = np.zeros((1024 - embeddings_np.shape[0], embeddings_np.shape[1]))
            embeddings_np = np.vstack([embeddings_np, padding])
        elif embeddings_np.shape[0] > 1024:
            embeddings_np = embeddings_np[:1024]

        return embeddings_np

    async def _search_vespa(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        search_mode: str,
        limit: int,
        filters: Optional[Dict] = None,
    ) -> List[ImageResult]:
        """
        Search Vespa with ColPali embeddings

        Args:
            query_embedding: ColPali query embedding [1024, 128]
            query_text: Original text query
            search_mode: "semantic" or "hybrid"
            limit: Number of results
            filters: Optional filters

        Returns:
            List of ImageResult
        """
        import requests

        # Build YQL query
        where_clause = "true"
        if filters:
            filter_parts = []
            if "detected_objects" in filters:
                filter_parts.append(
                    f"contains(detected_objects, '{filters['detected_objects']}')"
                )
            if "detected_scenes" in filters:
                filter_parts.append(
                    f"contains(detected_scenes, '{filters['detected_scenes']}')"
                )
            if filter_parts:
                where_clause = " AND ".join(filter_parts)

        yql = f"select * from image_content where {where_clause}"

        # Choose rank profile
        rank_profile = (
            "colpali_similarity" if search_mode == "semantic" else "hybrid_image"
        )

        # Flatten embedding for Vespa query
        query_embedding_flat = query_embedding.flatten().tolist()

        # Build Vespa request
        params = {
            "yql": yql,
            "hits": limit,
            "ranking.profile": rank_profile,
            "input.query(q)": str(query_embedding_flat),
        }

        if search_mode == "hybrid" and query_text:
            params["query"] = query_text

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
                ImageResult(
                    image_id=fields.get("image_id", ""),
                    image_url=fields.get("source_url", ""),
                    title=fields.get("image_title", ""),
                    description=fields.get("image_description", ""),
                    relevance_score=hit.get("relevance", 0.0),
                    detected_objects=fields.get("detected_objects", []),
                    detected_scenes=fields.get("detected_scenes", []),
                )
            )

        return results

    # DSPyA2AAgentBase abstract method implementations
    async def _process_with_dspy(self, dspy_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process image search request"""
        query = dspy_input.get("query", "")
        search_mode = dspy_input.get("search_mode", "semantic")
        limit = dspy_input.get("limit", 20)
        visual_filters = dspy_input.get("visual_filters")

        results = await self.search_images(
            query=query,
            search_mode=search_mode,
            limit=limit,
            visual_filters=visual_filters,
        )

        return {"results": results, "count": len(results)}

    def _dspy_to_a2a_output(self, dspy_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert search results to A2A response format"""
        results = dspy_output.get("results", [])

        return {
            "status": "success",
            "result_type": "image_search_results",
            "count": dspy_output.get("count", 0),
            "results": [
                {
                    "image_id": r.image_id,
                    "image_url": r.image_url,
                    "title": r.title,
                    "description": r.description,
                    "relevance_score": r.relevance_score,
                    "detected_objects": r.detected_objects,
                    "detected_scenes": r.detected_scenes,
                }
                for r in results
            ],
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define image search agent skills for A2A protocol"""
        return [
            {
                "name": "search_images",
                "description": "Search images using ColPali multi-vector embeddings",
                "input": {"query": "str", "search_mode": "str", "limit": "int"},
                "output": {"results": "List[ImageResult]", "count": "int"},
            },
            {
                "name": "find_similar_images",
                "description": "Find visually similar images",
                "input": {"reference_image": "Image", "limit": "int"},
                "output": {"results": "List[ImageResult]"},
            },
        ]
