"""Search endpoints - unified interface for search operations."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from cogniverse_agents.search.service import SearchService
from cogniverse_agents.search.vespa_query import VespaSearchDegraded
from cogniverse_core.common.tenant_utils import (
    assert_tenant_exists,
    require_tenant_id,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()


# FastAPI dependencies - will be overridden in main.py via app.dependency_overrides
def get_config_manager_dependency() -> ConfigManager:
    """FastAPI dependency for ConfigManager.

    Overridden in main.py via ``app.dependency_overrides``. If the override
    is missing the runtime is mid-startup or partially wired; surface a 503
    so clients retry rather than a 500 (uncaught ``RuntimeError`` would
    bubble to FastAPI's default 500 handler).
    """
    raise HTTPException(
        status_code=503,
        detail="ConfigManager dependency not configured; service initialising",
    )


def get_schema_loader_dependency() -> SchemaLoader:
    """FastAPI dependency for SchemaLoader.

    Same partial-startup semantics as ``get_config_manager_dependency``.
    """
    raise HTTPException(
        status_code=503,
        detail="SchemaLoader dependency not configured; service initialising",
    )


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1)
    profile: Optional[str] = None
    strategy: Optional[str] = "default"
    # Bounded like the graph search route: a negative value reached Vespa as
    # hits=<0 (backend 400 -> customer 500) and an unbounded one is a
    # heap-sized allocation request.
    top_k: int = Field(10, ge=1, le=1000)
    filters: Dict[str, Any] = {}
    tenant_id: Optional[str] = None
    org_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    profile: Optional[str]
    strategy: Optional[str]
    results_count: int
    results: list
    session_id: Optional[str] = None


def _resolve_service_and_profile(
    tenant_id: str,
    requested_profile: Optional[str],
    config_manager: ConfigManager,
    schema_loader: SchemaLoader,
) -> tuple[SearchService, str]:
    """Build the config-backed SearchService and resolve the search profile.

    Runs the ConfigUtils ensure-chain (system + routing + telemetry + backend
    config reads — synchronous Vespa/file I/O) and the profile lookup, so
    callers offload it via ``asyncio.to_thread`` to keep the loop free.

    Profile resolution: request wins, else ``active_video_profile``, else the
    first registered backend profile. No silent "default" string fallback —
    the string "default" isn't a valid profile.
    """
    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    search_service = SearchService(
        config=config,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    profile = requested_profile or config.get("active_video_profile")
    if not profile:
        profiles_dict = config.get("backend", {}).get("profiles", {}) or {}
        if profiles_dict:
            profile = next(iter(profiles_dict))
    if not profile:
        raise HTTPException(
            status_code=400,
            detail=(
                "No profile specified on the request and no "
                "active_video_profile configured on the runtime."
            ),
        )
    return search_service, profile


@router.post("/", response_model=None)
async def search(
    request: SearchRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Union[StreamingResponse, SearchResponse]:
    """Execute a search query. Returns SSE stream if stream=True, else JSON."""
    try:
        tenant_id = require_tenant_id(request.tenant_id, source="SearchRequest")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await assert_tenant_exists(tenant_id)

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
            component="search_service",
        )

    with context_manager as span:
        try:
            # Config ensure-chain (sync Vespa reads) + service construction run
            # off the loop; the search itself is offloaded below.
            search_service, profile = await asyncio.to_thread(
                _resolve_service_and_profile,
                tenant_id,
                request.profile,
                config_manager,
                schema_loader,
            )

            if request.stream:
                # Streaming response
                async def generate():
                    try:
                        # Emit status event
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching...', 'query': request.query})}\n\n"

                        # Execute search off the event loop — encoder
                        # inference + Vespa HTTP are synchronous.
                        results = await asyncio.to_thread(
                            search_service.search,
                            query=request.query,
                            profile=profile,
                            tenant_id=tenant_id,
                            top_k=request.top_k,
                            ranking_strategy=request.strategy,
                            filters=request.filters,
                        )

                        span.set_attribute("results_count", len(results))

                        # Emit final event with results
                        final_data = {
                            "type": "final",
                            "data": {
                                "query": request.query,
                                "profile": profile,
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
                # Non-streaming response; search runs sync (encoder + Vespa
                # HTTP), so keep it off the event loop.
                results = await asyncio.to_thread(
                    search_service.search,
                    query=request.query,
                    profile=profile,
                    tenant_id=tenant_id,
                    top_k=request.top_k,
                    ranking_strategy=request.strategy,
                    filters=request.filters,
                )

                span.set_attribute("results_count", len(results))

                return SearchResponse(
                    query=request.query,
                    profile=profile,
                    strategy=request.strategy,
                    results_count=len(results),
                    results=[r.to_dict() for r in results],
                    session_id=request.session_id,
                )

        except HTTPException:
            # Client errors raised above (e.g. 400 "no profile") must keep their
            # status — the broad handler below would otherwise mask them as 500.
            raise
        except VespaSearchDegraded as e:
            # Vespa soft-timeout / partial coverage — a transient, retryable
            # backend fault, not a server bug. 503 tells the caller to retry,
            # matching /agents/{name}/process; the broad handler below would
            # otherwise mask it as an opaque 500.
            logger.warning(f"Search degraded: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            # Bad request input (unknown profile/strategy, missing schema) — a
            # client error, not a server fault. 400, not 500.
            logger.info(f"Search rejected invalid input: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_strategies(
    tenant_id: str = Query(..., description="Tenant identifier (required)"),
    profile: Optional[str] = Query(
        None, description="Profile name; defaults to the tenant's active profile"
    ),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Dict[str, Any]:
    """List the ranking strategies ``POST /search`` accepts for a profile.

    Strategies are per-profile (derived from its schema), so this requires a
    tenant and resolves the profile the same way ``POST /search`` does. The
    returned names can be passed straight to the ``strategy`` field.
    """
    # Config ensure-chain + service construction run off the loop.
    search_service, resolved = await asyncio.to_thread(
        _resolve_service_and_profile,
        tenant_id,
        profile,
        config_manager,
        schema_loader,
    )

    try:
        # The first extraction globs+parses the schema JSONs (memoized after);
        # keep even that off the event loop.
        strategies = await asyncio.to_thread(
            search_service.get_available_strategies, resolved, tenant_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "tenant_id": tenant_id,
        "profile": resolved,
        "count": len(strategies),
        "strategies": strategies,
    }


@router.get("/profiles")
async def list_profiles(
    tenant_id: str = Query(..., description="Tenant identifier (required)"),
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
async def rerank_results(
    request: Dict[str, Any],
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
) -> Dict[str, Any]:
    """Rerank search results using specified strategy."""
    try:
        from cogniverse_core.common.tenant_utils import require_tenant_id

        query = request.get("query")
        results = request.get("results", [])
        strategy = request.get("strategy", "learned")
        tenant_id = require_tenant_id(
            request.get("tenant_id"), source="/search/rerank body"
        )

        if not query or not results:
            raise HTTPException(
                status_code=400, detail="Query and results are required"
            )

        # Select + run the reranker via the shared service (same path the
        # evaluation harness uses). Unknown strategy raises ValueError →
        # surfaced as 400 by the handler below.
        from cogniverse_agents.search.rerank_service import rerank_result_dicts

        reranked = await rerank_result_dicts(
            query=query,
            results=results,
            strategy=strategy,
            tenant_id=tenant_id,
            config_manager=config_manager,
        )

        return {
            "query": query,
            "strategy": strategy,
            "original_count": len(results),
            "reranked_count": len(reranked),
            "results": reranked,
        }

    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        # Client-input validators (require_tenant_id, unknown strategy) raise
        # ValueError; a non-scalar score raises TypeError from float() coercion.
        # Both are bad input — surface as 400, not 500.
        logger.warning(f"Rerank bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
