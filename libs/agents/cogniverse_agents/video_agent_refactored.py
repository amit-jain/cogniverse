"""Refactored Video Search Agent using unified search service.

Profile-agnostic: accepts profile and tenant_id per-request.
Single SearchService instance handles all profiles via encoder caching.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_foundation.config.manager import ConfigManager

from fastapi import FastAPI, HTTPException

from cogniverse_agents.search.base import SearchResult
from cogniverse_agents.search.service import SearchService
from cogniverse_agents.tools.a2a_utils import DataPart, Task
from cogniverse_foundation.config.utils import get_config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Search Agent (Refactored)",
    description="Video search agent using unified search architecture",
    version="3.0.0",
)


class VideoSearchAgent:
    """Video search agent using unified, profile-agnostic search service."""

    def __init__(
        self,
        config_manager: "ConfigManager" = None,
        schema_loader=None,
    ):
        """
        Initialize video search agent (profile-agnostic).

        Args:
            config_manager: ConfigManager instance (required for dependency injection)
            schema_loader: SchemaLoader instance (required for dependency injection)

        Raises:
            ValueError: If config_manager or schema_loader is not provided
        """
        if config_manager is None:
            raise ValueError(
                "config_manager is required for VideoSearchAgent. "
                "Pass create_default_config_manager() explicitly."
            )

        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for VideoSearchAgent. "
                "Pass FilesystemSchemaLoader or SchemaLoader instance explicitly."
            )

        self.config_manager = config_manager
        self.config = get_config(tenant_id="default", config_manager=config_manager)
        self.schema_loader = schema_loader

        # Default profile from config (used when caller doesn't specify)
        self.default_profile = (
            self.config.get("active_video_profile") or "video_colpali_smol500_mv_frame"
        )

        # Single profile-agnostic search service
        self.search_service = SearchService(
            self.config,
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )

        logger.info(
            f"VideoSearchAgent initialized (default_profile={self.default_profile})"
        )

    def search(
        self,
        query: str,
        tenant_id: str,
        profile: Optional[str] = None,
        top_k: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for videos.

        Args:
            query: Search query
            profile: Profile to use (defaults to config active_video_profile)
            tenant_id: Tenant identifier
            top_k: Number of results
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of search results
        """
        effective_profile = profile or self.default_profile
        logger.info(
            f"Searching: '{query}' profile={effective_profile} tenant={tenant_id}"
        )

        filters: Optional[Dict] = None
        if start_date or end_date:
            filters = {}
            if start_date:
                filters["start_date"] = start_date
            if end_date:
                filters["end_date"] = end_date

        return self.search_service.search(
            query=query,
            profile=effective_profile,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )


# Global agent instance
video_agent: Optional[VideoSearchAgent] = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    global video_agent

    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    try:
        video_agent = VideoSearchAgent(
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        logger.info("Video search agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "video_search",
        "default_profile": video_agent.default_profile if video_agent else None,
    }


@app.post("/search")
async def search_endpoint(request: dict):
    """Direct search endpoint for dashboard integration."""
    if not video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    strategies = request.get("strategies", ["default"])
    tenant_id = request.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    profile = request.get("profile")
    top_k = request.get("top_k", 10)

    try:
        all_results = []
        for strategy in strategies:
            results = video_agent.search_service.search(
                query=query,
                profile=profile or video_agent.default_profile,
                tenant_id=tenant_id,
                top_k=top_k,
                ranking_strategy=strategy if strategy != "default" else None,
                filters={
                    k: v
                    for k, v in {
                        "start_date": request.get("start_date"),
                        "end_date": request.get("end_date"),
                    }.items()
                    if v is not None
                }
                or None,
            )
            for r in results:
                result_dict = r.to_dict() if hasattr(r, "to_dict") else r
                result_dict["ranking_strategy"] = strategy
                all_results.append(result_dict)

        return {
            "status": "completed",
            "query": query,
            "results_count": len(all_results),
            "results": all_results,
            "profile": profile or video_agent.default_profile,
            "strategies": strategies,
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_task(task: Task):
    """Process search task (A2A protocol)."""
    if not video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not task.messages:
        raise HTTPException(status_code=400, detail="No messages in task")

    last_message = task.messages[-1]
    data_part = next(
        (part for part in last_message.parts if isinstance(part, DataPart)), None
    )

    if not data_part:
        raise HTTPException(status_code=400, detail="No data in message")

    query_data = data_part.data
    query = query_data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    try:
        results = video_agent.search(
            query=query,
            profile=query_data.get("profile"),
            tenant_id=query_data["tenant_id"],
            top_k=query_data.get("top_k", 10),
            start_date=query_data.get("start_date"),
            end_date=query_data.get("end_date"),
        )

        return {"task_id": task.id, "status": "completed", "results": results}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
