"""Refactored Video Search Agent using unified search service."""

import logging
import os
from typing import TYPE_CHECKING, List, Optional

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
    version="2.0.0",
)


class VideoSearchAgent:
    """Video search agent using unified search service."""

    def __init__(
        self,
        profile: Optional[str] = None,
        tenant_id: str = "default",
        config_manager: "ConfigManager" = None,
        schema_loader=None,
    ):
        """
        Initialize video search agent.

        Args:
            profile: Profile name to use (optional)
            tenant_id: Tenant identifier for config scoping
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

        self.tenant_id = tenant_id
        self.config_manager = config_manager
        self.config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        self.schema_loader = schema_loader

        # Determine profile
        if profile:
            self.profile = profile
        else:
            self.profile = self.config.get_active_profile() or "frame_based_colpali"

        logger.info(f"Initializing VideoSearchAgent with profile: {self.profile}")

        # Initialize search service
        try:
            self.search_service = SearchService(
                self.config,
                self.profile,
                tenant_id=self.tenant_id,
                config_manager=self.config_manager,
                schema_loader=self.schema_loader,
            )
            logger.info("Search service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for videos.

        Args:
            query: Search query
            top_k: Number of results
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of search results
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k})")

        # Build filters
        filters = {}
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date

        # Search
        try:
            results = self.search_service.search(
                query=query, top_k=top_k, filters=filters if filters else None
            )

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise


# Global agent instance
video_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    global video_agent

    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager

    # Get profile from environment or config
    profile = os.environ.get("VIDEO_PROFILE")
    tenant_id = os.environ.get("TENANT_ID", "default")
    config_manager = create_default_config_manager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    try:
        video_agent = VideoSearchAgent(
            profile=profile,
            tenant_id=tenant_id,
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
        "profile": video_agent.profile if video_agent else None,
    }


@app.post("/search")
async def search_endpoint(request: dict):
    """Direct search endpoint for dashboard integration."""
    if not video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    try:
        results = video_agent.search(
            query=query,
            top_k=request.get("top_k", 10),
            start_date=request.get("start_date"),
            end_date=request.get("end_date"),
        )

        return {
            "status": "completed",
            "query": query,
            "results_count": len(results),
            "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in results],
            "profile": video_agent.profile,
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

    # Extract query from last message
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

    # Search
    try:
        results = video_agent.search(
            query=query,
            top_k=query_data.get("top_k", 10),
            start_date=query_data.get("start_date"),
            end_date=query_data.get("end_date"),
        )

        return {"task_id": task.id, "status": "completed", "results": results}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
