"""Search endpoints - unified interface for search operations."""

import logging
from typing import Any, Dict, Optional

from cogniverse_core.config.manager import ConfigManager
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cogniverse_runtime.search.service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    profile: Optional[str] = None
    strategy: Optional[str] = "hybrid"
    top_k: int = 10
    filters: Dict[str, Any] = {}
    tenant_id: Optional[str] = None
    org_id: Optional[str] = None


@router.post("/")
async def search(request: SearchRequest) -> Dict[str, Any]:
    """Execute a search query."""
    try:
        config_manager = ConfigManager.get_instance()
        config = config_manager.get_config()

        # Create search service
        search_service = SearchService(
            config=config,
            profile=request.profile or config.get("default_profile", "default"),
        )

        # Execute search
        results = search_service.search(
            query=request.query,
            top_k=request.top_k,
            strategy=request.strategy,
            filters=request.filters,
            tenant_id=request.tenant_id,
            org_id=request.org_id,
        )

        return {
            "query": request.query,
            "profile": request.profile,
            "strategy": request.strategy,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_strategies() -> Dict[str, Any]:
    """List available search strategies."""
    return {
        "strategies": [
            {"name": "semantic", "description": "Pure semantic similarity search"},
            {"name": "bm25", "description": "BM25 keyword-based search"},
            {"name": "hybrid", "description": "Combines semantic and BM25"},
            {"name": "learned", "description": "Learned reranking with ML model"},
            {
                "name": "multi_modal",
                "description": "Multi-modal reranking (text, video, audio)",
            },
        ]
    }


@router.get("/profiles")
async def list_profiles() -> Dict[str, Any]:
    """List available search profiles."""
    config_manager = ConfigManager.get_instance()
    config = config_manager.get_config()

    backend_config = config.get("backend", {})
    profiles = backend_config.get("profiles", {})

    return {
        "count": len(profiles),
        "profiles": [
            {
                "name": name,
                "model": profile.get("embedding_model"),
                "type": profile.get("type"),
            }
            for name, profile in profiles.items()
        ],
    }


@router.post("/rerank")
async def rerank_results(request: Dict[str, Any]) -> Dict[str, Any]:
    """Rerank search results using specified strategy."""
    try:
        query = request.get("query")
        results = request.get("results", [])
        strategy = request.get("strategy", "learned")

        if not query or not results:
            raise HTTPException(
                status_code=400, detail="Query and results are required"
            )

        # Import reranker based on strategy
        if strategy == "learned":
            from cogniverse_agents.search.learned_reranker import (
                LearnedReranker,
            )

            reranker = LearnedReranker()
        elif strategy == "hybrid":
            from cogniverse_agents.search.hybrid_reranker import (
                HybridReranker,
            )

            reranker = HybridReranker()
        elif strategy == "multi_modal":
            from cogniverse_agents.search.multi_modal_reranker import (
                MultiModalReranker,
            )

            reranker = MultiModalReranker()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

        # Rerank
        reranked = reranker.rerank(query=query, results=results)

        return {
            "query": query,
            "strategy": strategy,
            "original_count": len(results),
            "reranked_count": len(reranked),
            "results": reranked,
        }

    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
