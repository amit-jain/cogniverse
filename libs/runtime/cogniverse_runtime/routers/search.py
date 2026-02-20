"""Search endpoints - unified interface for search operations."""

import json
import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_runtime.search.service import SearchService
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()


# FastAPI dependencies - will be overridden in main.py via app.dependency_overrides
def get_config_manager_dependency() -> ConfigManager:
    """
    FastAPI dependency for ConfigManager.

    This function should be overridden in main.py using app.dependency_overrides.
    If not overridden, it raises an error to fail fast.
    """
    raise RuntimeError(
        "ConfigManager dependency not configured. "
        "Override this dependency in main.py using app.dependency_overrides."
    )


def get_schema_loader_dependency() -> SchemaLoader:
    """
    FastAPI dependency for SchemaLoader.

    This function should be overridden in main.py using app.dependency_overrides.
    If not overridden, it raises an error to fail fast.
    """
    raise RuntimeError(
        "SchemaLoader dependency not configured. "
        "Override this dependency in main.py using app.dependency_overrides."
    )


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    profile: Optional[str] = None
    strategy: Optional[str] = "hybrid"
    top_k: int = 10
    filters: Dict[str, Any] = {}
    tenant_id: Optional[str] = None
    org_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    profile: Optional[str]
    strategy: Optional[str]
    results_count: int
    results: list
    session_id: Optional[str] = None


@router.post("/", response_model=None)
async def search(
    request: SearchRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Union[StreamingResponse, SearchResponse]:
    """Execute a search query. Returns SSE stream if stream=True, else JSON."""
    tenant_id = request.tenant_id or "default"

    telemetry_manager = get_telemetry_manager()

    # Use session_span if session_id provided, otherwise regular span
    if request.session_id:
        context_manager = telemetry_manager.session_span(
            "api.search.request",
            tenant_id=tenant_id,
            session_id=request.session_id,
            attributes={
                "query": request.query,
                "profile": request.profile,
                "strategy": request.strategy,
                "top_k": request.top_k,
                "stream": request.stream,
            },
        )
    else:
        context_manager = telemetry_manager.span(
            "api.search.request",
            tenant_id=tenant_id,
            attributes={
                "query": request.query,
                "profile": request.profile,
                "strategy": request.strategy,
                "top_k": request.top_k,
                "stream": request.stream,
            },
        )

    with context_manager as span:
        try:
            config = get_config(tenant_id=tenant_id, config_manager=config_manager)

            # Create search service with required dependencies
            search_service = SearchService(
                config=config,
                profile=request.profile or config.get("default_profile", "default"),
                tenant_id=tenant_id,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )

            if request.stream:
                # Streaming response
                async def generate():
                    try:
                        # Emit status event
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching...', 'query': request.query})}\n\n"

                        # Execute search
                        results = search_service.search(
                            query=request.query,
                            top_k=request.top_k,
                            ranking_strategy=request.strategy,
                            filters=request.filters,
                            tenant_id=request.tenant_id,
                            org_id=request.org_id,
                        )

                        span.set_attribute("results_count", len(results))

                        # Emit final event with results
                        final_data = {
                            "type": "final",
                            "data": {
                                "query": request.query,
                                "profile": request.profile,
                                "strategy": request.strategy,
                                "results_count": len(results),
                                "results": [r.to_dict() for r in results],
                                "session_id": request.session_id,
                            },
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"

                    except Exception as e:
                        error_event = {
                            "type": "error",
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                        yield f"data: {json.dumps(error_event)}\n\n"

                return StreamingResponse(generate(), media_type="text/event-stream")

            else:
                # Non-streaming response
                results = search_service.search(
                    query=request.query,
                    top_k=request.top_k,
                    ranking_strategy=request.strategy,
                    filters=request.filters,
                    tenant_id=request.tenant_id,
                    org_id=request.org_id,
                )

                span.set_attribute("results_count", len(results))

                return SearchResponse(
                    query=request.query,
                    profile=request.profile,
                    strategy=request.strategy,
                    results_count=len(results),
                    results=[r.to_dict() for r in results],
                    session_id=request.session_id,
                )

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
async def list_profiles(
    tenant_id: str = Query(default="default", description="Tenant identifier"),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Dict[str, Any]:
    """List available search profiles for a tenant.

    Returns profile name, type, and model — safe for user-facing display.
    Detailed config (pipeline, strategies, schema internals) is admin-only
    via GET /admin/profiles.
    """
    config = get_config(tenant_id=tenant_id, config_manager=config_manager)

    backend_config = config.get("backend", {})
    profiles = backend_config.get("profiles", {})

    return {
        "tenant_id": tenant_id,
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
