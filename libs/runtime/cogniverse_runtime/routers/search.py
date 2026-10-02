"""Search endpoints - unified interface for search operations."""

import asyncio
import json
import logging
from typing import Any, Dict, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from cogniverse_agents.search.service import SearchService
from cogniverse_agents.search.vespa_query import VespaSearchDegraded
from cogniverse_core.common.tenant_utils import (
    assert_tenant_exists,
    require_tenant_id,
)
from cogniverse_core.query.encoders import (
    EncoderNotConfiguredError,
    EncoderUnavailableError,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config, resolve_default_profile
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_runtime.http_errors import (
    failure_body,
    failure_response,
    query_encoder_failure,
    record_failure,
)
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()


SEARCH_DEGRADED = "search_degraded"
INVALID_SEARCH_REQUEST = "invalid_search_request"
SEARCH_FAILED = "search_failed"


def _search_failure(
    exc: Exception, *, profile: Optional[str], strategy: Optional[str]
) -> tuple[int, Dict[str, Any], Optional[Dict[str, str]]]:
    """Status, body and headers for a failed search, from typed fields only.

    The cause is logged and recorded on the active span; the body never
    carries its text, which can name backend hosts, URLs or file paths.
    """
    if isinstance(exc, (EncoderNotConfiguredError, EncoderUnavailableError)):
        record_failure(exc, "query_encoder")
        return query_encoder_failure(exc, profile=profile, strategy=strategy)
    if isinstance(exc, VespaSearchDegraded):
        # A Vespa soft-timeout / partial coverage is transient: 503, retry.
        record_failure(exc, SEARCH_DEGRADED)
        return (
            503,
            failure_body(
                SEARCH_DEGRADED,
                "The search backend answered with degraded coverage; retry the search.",
                exc,
                profile=profile,
                strategy=strategy,
            ),
            None,
        )
    if isinstance(exc, ValueError):
        # Request input the profile or schema cannot serve (unknown profile or
        # strategy, bad granularity): the caller's error, not a server fault.
        record_failure(exc, INVALID_SEARCH_REQUEST, level=logging.WARNING)
        named = [
            f"{kind} '{value}'"
            for kind, value in (("profile", profile), ("strategy", strategy))
            if value
        ]
        subject = "Search with " + " and ".join(named) if named else "Search"
        return (
            400,
            failure_body(
                INVALID_SEARCH_REQUEST,
                f"{subject} was rejected; GET /search/profiles and GET "
                "/search/strategies list what this tenant accepts.",
                exc,
                profile=profile,
                strategy=strategy,
            ),
            None,
        )
    record_failure(exc, SEARCH_FAILED)
    subject = f"Search with profile '{profile}'" if profile else "Search"
    return (
        500,
        failure_body(
            SEARCH_FAILED,
            f"{subject} failed; the runtime log names the cause.",
            exc,
            profile=profile,
            strategy=strategy,
        ),
        None,
    )


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
    result_granularity: Optional[Literal["source", "segment"]] = None
    # Bounded like the graph search route: a negative value reached Vespa as
    # hits=<0 (backend 400 -> customer 500) and an unbounded one is a
    # heap-sized allocation request.
    top_k: int = Field(10, ge=1, le=1000)
    filters: Dict[str, Any] = Field(default_factory=dict)
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
    source_search_incomplete: bool
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

    Profile resolution: request wins, else the tenant's default profile via
    the shared ``resolve_default_profile`` (``backend.default_profiles.video.
    profile`` then ``active_video_profile``). With neither, the request is
    refused rather than run on a catalog profile nothing selected.
    """
    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    search_service = SearchService(
        config=config,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    # The one resolver upload and the dispatcher also call: a tenant that set
    # backend.default_profiles.video.profile while active_video_profile still
    # named another profile ingested into one corpus and queried another.
    profile = requested_profile or resolve_default_profile(config)
    if not profile:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No profile specified on the request and tenant {tenant_id!r} "
                "has no configured default video profile."
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
    # Combine a separately-supplied org_id with a simple tenant_id into the
    # canonical org:tenant form (matching the admin tenant route); org_id was
    # otherwise silently dropped and the search hit the wrong namespace.
    combined_tenant = request.tenant_id
    if request.org_id and request.tenant_id and ":" not in request.tenant_id:
        combined_tenant = f"{request.org_id}:{request.tenant_id}"
    try:
        tenant_id = require_tenant_id(combined_tenant, source="SearchRequest")
    except ValueError as exc:
        raise failure_response(
            400,
            "invalid_tenant_id",
            "tenant_id is required on a search request, as '<org>:<tenant>' or "
            "'<tenant>'.",
            exc,
            tenant_id=request.tenant_id,
        )

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
                "result_granularity": request.result_granularity,
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
                "result_granularity": request.result_granularity,
                "top_k": request.top_k,
                "stream": request.stream,
            },
            component="search_service",
        )

    profile = request.profile
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
                            result_granularity=request.result_granularity,
                            filters=request.filters,
                        )

                        span.set_attribute("results_count", len(results))
                        span.set_attribute(
                            "source_search_incomplete",
                            results.source_search_incomplete,
                        )

                        # Emit final event with results
                        final_data = {
                            "type": "final",
                            "data": {
                                "query": request.query,
                                "profile": profile,
                                "strategy": request.strategy,
                                "results_count": len(results),
                                "results": [r.to_dict() for r in results],
                                "source_search_incomplete": (
                                    results.source_search_incomplete
                                ),
                                "session_id": request.session_id,
                            },
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"

                    except Exception as e:
                        _, body, _ = _search_failure(
                            e, profile=profile, strategy=request.strategy
                        )
                        error_event = {
                            "type": "error",
                            "error": body["message"],
                            "error_type": type(e).__name__,
                            "detail": body,
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
                    result_granularity=request.result_granularity,
                    filters=request.filters,
                )

                span.set_attribute("results_count", len(results))
                span.set_attribute(
                    "source_search_incomplete", results.source_search_incomplete
                )

                return SearchResponse(
                    query=request.query,
                    profile=profile,
                    strategy=request.strategy,
                    results_count=len(results),
                    results=[r.to_dict() for r in results],
                    source_search_incomplete=results.source_search_incomplete,
                    session_id=request.session_id,
                )

        except HTTPException:
            # Client errors raised above (e.g. 400 "no profile") must keep their
            # status — the broad handler below would otherwise mask them as 500.
            raise
        except Exception as e:
            status, body, headers = _search_failure(
                e, profile=profile, strategy=request.strategy
            )
            raise HTTPException(status_code=status, detail=body, headers=headers)


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
    """List the search profiles a tenant can actually use.

    A profile is advertised only when its embedding inference service is
    configured AND this tenant's schema for it is deployed — a profile with no
    deployed schema answers every search from an application that does not
    carry its documents. Returns profile name, type, and model — safe for
    user-facing display. Detailed config (pipeline, strategies, schema
    internals) is admin-only via GET /admin/profiles.
    """
    from cogniverse_agents.profile_selection_agent import servable_tenant_profiles

    # Config store + schema registry reads; keep both off the event loop.
    servable = await asyncio.to_thread(
        servable_tenant_profiles, config_manager, tenant_id
    )

    return {
        "tenant_id": tenant_id,
        "count": len(servable),
        "profiles": [
            {
                "name": name,
                "model": profile.embedding_model,
                "type": profile.type,
            }
            for name, profile in servable
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
        raise failure_response(
            500,
            "rerank_failed",
            f"Rerank with strategy '{request.get('strategy', 'learned')}' "
            "failed; the runtime log names the cause.",
            e,
        )
