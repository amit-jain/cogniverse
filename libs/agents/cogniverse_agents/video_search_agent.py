"""
Enhanced Video Search Agent with support for both text-to-video and video-to-video search.
Integrates with existing VideoSearchAgent and adds video upload/encoding capabilities.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import uvicorn
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareAgentMixin
from cogniverse_core.registries.backend_registry import get_backend_registry
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from cogniverse_agents.query.encoders import QueryEncoderFactory

# Enhanced query support from DSPy routing system
from cogniverse_agents.routing_agent import RoutingDecision
from cogniverse_agents.tools.a2a_utils import DataPart, TextPart

logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """Context for relationship-aware video search"""

    original_query: str
    enhanced_query: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    routing_metadata: Dict[str, Any]
    confidence: float = 0.0


class RelationshipAwareSearchParams(BaseModel):
    """Enhanced search parameters with relationship context"""

    query: str = Field(..., description="Search query (may be enhanced)")
    original_query: Optional[str] = Field(None, description="Original user query")
    enhanced_query: Optional[str] = Field(
        None, description="Relationship-enhanced query"
    )
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted relationships"
    )
    top_k: int = Field(10, description="Number of results to return")
    ranking_strategy: Optional[str] = Field(None, description="Vespa ranking strategy")
    start_date: Optional[str] = Field(None, description="Start date filter")
    end_date: Optional[str] = Field(None, description="End date filter")
    confidence_threshold: float = Field(0.0, description="Minimum confidence score")
    use_relationship_boost: bool = Field(
        True, description="Boost results based on relationship context"
    )


# --- Enhanced Data Models ---
class VideoPart(BaseModel):
    """Video content part for A2A messages"""

    type: Literal["video"] = "video"
    video_data: bytes = Field(..., description="Raw video file bytes")
    filename: Optional[str] = Field(None, description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")


class ImagePart(BaseModel):
    """Image content part for A2A messages"""

    type: Literal["image"] = "image"
    image_data: bytes = Field(..., description="Raw image file bytes")
    filename: Optional[str] = Field(None, description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")


# --- Video Processing Components ---
class VideoProcessor:
    """Handles video upload, processing, and encoding to embeddings"""

    def __init__(self, query_encoder):
        self.query_encoder = query_encoder
        self.temp_dir = Path(tempfile.gettempdir()) / "video_search_agent"
        self.temp_dir.mkdir(exist_ok=True)

    def process_video_file(self, video_data: bytes, filename: str) -> np.ndarray:
        """
        Process uploaded video file and extract embeddings.

        Args:
            video_data: Raw video file bytes
            filename: Original filename

        Returns:
            Video embeddings as numpy array
        """
        # Save video to temporary file
        temp_video_path = self.temp_dir / f"temp_{filename}"

        try:
            with open(temp_video_path, "wb") as f:
                f.write(video_data)

            logger.info(f"Processing video file: {filename}")

            # Extract embeddings using query encoder's video processing capability
            embeddings = self._extract_video_embeddings(temp_video_path)

            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings

        finally:
            # Clean up temporary file
            if temp_video_path.exists():
                temp_video_path.unlink()

    def process_image_file(self, image_data: bytes, filename: str) -> np.ndarray:
        """
        Process uploaded image file and extract embeddings.

        Args:
            image_data: Raw image file bytes
            filename: Original filename

        Returns:
            Image embeddings as numpy array
        """
        # Save image to temporary file
        temp_image_path = self.temp_dir / f"temp_{filename}"

        try:
            with open(temp_image_path, "wb") as f:
                f.write(image_data)

            logger.info(f"Processing image file: {filename}")

            # Extract embeddings using query encoder's image processing capability
            embeddings = self._extract_image_embeddings(temp_image_path)

            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings

        finally:
            # Clean up temporary file
            if temp_image_path.exists():
                temp_image_path.unlink()

    def _extract_video_embeddings(self, video_path: Path) -> np.ndarray:
        """Extract embeddings from video file using the query encoder"""
        if hasattr(self.query_encoder, "encode_video"):
            return self.query_encoder.encode_video(str(video_path))
        elif hasattr(self.query_encoder, "encode_frames"):
            # For frame-based encoders, extract frames and encode
            return self._extract_frames_and_encode(video_path)
        else:
            raise NotImplementedError("Query encoder does not support video encoding")

    def _extract_image_embeddings(self, image_path: Path) -> np.ndarray:
        """Extract embeddings from image file using the query encoder"""
        if hasattr(self.query_encoder, "encode_image"):
            return self.query_encoder.encode_image(str(image_path))
        elif hasattr(self.query_encoder, "encode"):
            # For text encoders, this won't work - need to implement image support
            raise NotImplementedError("Query encoder does not support image encoding")
        else:
            raise NotImplementedError("Query encoder does not support image encoding")

    def _extract_frames_and_encode(self, video_path: Path) -> np.ndarray:
        """Extract frames from video and encode them"""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            frames = []

            # Extract a few key frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Extract frames at 1 FPS or every 30 frames, whichever is less frequent
            step = max(fps, 30)

            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    if len(frames) >= 10:  # Limit to 10 frames
                        break

            cap.release()

            if not frames:
                raise ValueError("No frames extracted from video")

            # Encode frames using the query encoder
            if hasattr(self.query_encoder, "encode_frames"):
                return self.query_encoder.encode_frames(frames)
            else:
                # Save frames as temporary images and encode
                frame_embeddings = []
                for i, frame in enumerate(frames):
                    temp_frame_path = self.temp_dir / f"temp_frame_{i}.jpg"
                    cv2.imwrite(str(temp_frame_path), frame)

                    try:
                        if hasattr(self.query_encoder, "encode_image"):
                            frame_emb = self.query_encoder.encode_image(
                                str(temp_frame_path)
                            )
                            frame_embeddings.append(frame_emb)
                    finally:
                        temp_frame_path.unlink()

                if not frame_embeddings:
                    raise ValueError("No frame embeddings extracted")

                # Average the frame embeddings
                return np.mean(frame_embeddings, axis=0)

        except ImportError:
            raise ImportError(
                "OpenCV is required for video frame extraction. Install with: pip install opencv-python"
            )


# --- Enhanced Video Search Agent ---
class VideoSearchAgent(MemoryAwareMixin, TenantAwareAgentMixin):
    """
    Enhanced video search agent supporting both text-to-video and video-to-video search.
    Enhanced with memory capabilities for learning from search patterns.
    """

    def __init__(self, tenant_id: str, schema_loader=None, config_manager=None, **kwargs):
        """
        Initialize enhanced video search agent

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            schema_loader: SchemaLoader instance (REQUIRED for dependency injection)
            config_manager: ConfigManager instance (REQUIRED for dependency injection)
            **kwargs: Additional configuration options

        Raises:
            ValueError: If tenant_id is empty or None or schema_loader is None
        """
        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for VideoSearchAgent. "
                "Dependency injection is mandatory - pass SchemaLoader instance explicitly."
            )
        if config_manager is None:
            # Create default if not provided (for backward compatibility)
            from cogniverse_core.config.utils import create_default_config_manager
            config_manager = create_default_config_manager()

        # Store dependencies for use in initialization
        self.schema_loader = schema_loader
        self.config_manager = config_manager

        # Initialize tenant support via TenantAwareAgentMixin
        # This validates tenant_id and stores it (eliminates duplication)
        TenantAwareAgentMixin.__init__(self, tenant_id=tenant_id)

        super().__init__()  # Initialize MemoryAwareMixin
        logger.info(f"Initializing VideoSearchAgent for tenant: {tenant_id}...")

        # self.config already set by TenantAwareAgentMixin

        # Check environment variables first, then kwargs, then defaults
        vespa_url = kwargs.get("vespa_url") or os.getenv(
            "VESPA_URL", "http://localhost"
        )

        # Handle port from env var or kwargs (extract port from URL if needed)
        env_port = os.getenv("VESPA_PORT")
        kwargs_port = kwargs.get("vespa_port")

        if kwargs_port:
            vespa_port = int(kwargs_port)
        elif env_port:
            vespa_port = int(env_port)
            # If we got port from env var, make sure vespa_url doesn't already include it
            if ":" in vespa_url and vespa_url.count(":") >= 2:
                # Remove existing port from URL
                vespa_url = ":".join(vespa_url.split(":")[:-1])
        elif ":" in vespa_url and vespa_url.count(":") >= 2:
            # Extract port from URL like "http://localhost:8081"
            vespa_port = int(vespa_url.split(":")[-1])
            # Remove port from URL since VespaVideoSearchClient will add it
            vespa_url = ":".join(vespa_url.split(":")[:-1])
        else:
            vespa_port = 8080

        # Extract vespa_host for memory initialization (strip protocol if present)
        vespa_host = vespa_url.replace("http://", "").replace("https://", "")

        # Get config port for schema deployment (separate from data port)
        # System tests use config_port = data_port + 10991 (e.g., 8082 data, 19073 config)
        # Production uses default config_port = 19071
        vespa_config_port = kwargs.get("vespa_config_port")
        if not vespa_config_port:
            # Calculate config port: if data port is 8080 (default), use 19071 (default)
            # Otherwise, assume pattern: config_port = data_port + 10991
            if vespa_port == 8080:
                vespa_config_port = 19071
            else:
                vespa_config_port = vespa_port + 10991

        # Initialize memory for video agent with correct Vespa ports
        # NOTE: Memory schema creation is controlled separately from video schema
        # Default is True so memory always works, but tests can disable if managing schemas themselves
        memory_initialized = self.initialize_memory(
            agent_name="video_agent",
            tenant_id=tenant_id,
            backend_host=vespa_host,
            backend_port=vespa_port,
            backend_config_port=vespa_config_port,
            auto_create_schema=kwargs.get("auto_create_memory_schema", True),
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )
        if memory_initialized:
            logger.info(f"✅ Memory initialized for video_agent (tenant: {tenant_id})")
        else:
            logger.info("ℹ️  Memory disabled or not configured for video_agent")

        # Get model from active profile - check environment variable first
        active_profile = (
            os.getenv("VESPA_SCHEMA")
            or kwargs.get("profile")
            or self.config.get("active_video_profile")
            or "video_colpali_smol500_mv_frame"
        )

        # Store active_profile for use in search methods
        self.active_profile = active_profile

        backend_config_data = self.config.get("backend", {})
        profiles = backend_config_data.get("profiles", {})

        if active_profile and active_profile in profiles:
            model_name = profiles[active_profile].get(
                "embedding_model", "vidore/colsmol-500m"
            )
            self.embedding_type = profiles[active_profile].get(
                "embedding_type", "frame_based"
            )
        else:
            model_name = kwargs.get("model_name", "vidore/colsmol-500m")
            self.embedding_type = "frame_based"

        # Initialize search backend via backend registry
        try:
            logger.info("Enhanced Video Search Agent configuration:")
            logger.info(f"  - Tenant ID: {tenant_id}")
            logger.info(f"  - Vespa URL: {vespa_url}")
            logger.info(f"  - Vespa Port: {vespa_port}")
            logger.info(f"  - Active Profile: {active_profile}")
            logger.info(f"  - Model Name: {model_name}")
            logger.info(f"  - Environment VESPA_URL: {os.getenv('VESPA_URL')}")
            logger.info(f"  - Environment VESPA_PORT: {os.getenv('VESPA_PORT')}")
            logger.info(f"  - Environment VESPA_SCHEMA: {os.getenv('VESPA_SCHEMA')}")

            # Build backend config
            # Pass base schema name - backend will add tenant suffix
            backend_config = {
                "vespa_url": vespa_url,
                "vespa_port": vespa_port,
                "vespa_config_port": vespa_config_port,
                "schema_name": active_profile,  # Base schema name, backend adds tenant suffix
                "tenant_id": tenant_id,
                "profile": active_profile,  # VespaBackend checks for "profile", not "active_video_profile"
                "backend": backend_config_data,  # Pass entire backend section with profiles, default_profiles, etc.
            }

            # Get search backend from registry
            registry = get_backend_registry()
            # Use self.config_manager and self.schema_loader from dependency injection

            self.search_backend = registry.get_search_backend(
                "vespa", tenant_id, backend_config,
                config_manager=self.config_manager,
                schema_loader=self.schema_loader
            )

            logger.info(
                f"Search backend initialized at {vespa_url}:{vespa_port} "
                f"for tenant {tenant_id} with profile {active_profile}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize search backend: {e}")
            raise

        # Initialize query encoder
        try:
            self.query_encoder = QueryEncoderFactory.create_encoder(
                active_profile, model_name, config=self.config
            )
            logger.info(f"Query encoder initialized for profile: {active_profile}")
        except Exception as e:
            logger.error(f"Failed to initialize query encoder: {e}")
            raise

        # Initialize video processor
        self.video_processor = VideoProcessor(self.query_encoder)

        logger.info("VideoSearchAgent initialization complete")

    def search_by_text(
        self, query: str, top_k: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search videos using text query.

        Args:
            query: Text search query
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        logger.info(f"Text-to-video search: '{query}' (top_k={top_k})")

        # Retrieve relevant search patterns from memory
        if self.is_memory_enabled():
            memory_context = self.get_relevant_context(query, top_k=3)
            if memory_context:
                logger.info(f"📚 Retrieved memory context for query: {query[:50]}...")

        try:
            # Encode text query
            query_embeddings = self.query_encoder.encode(query)

            # Execute search with backend using query_dict format
            # Note: Date filters not yet supported in new backend
            if kwargs.get("start_date") or kwargs.get("end_date"):
                logger.warning("Date filters (start_date/end_date) not yet supported by VespaSearchBackend")

            query_dict = {
                "query": query,
                "type": "video",
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,  # TODO: Add date filter support
                "strategy": kwargs.get("ranking", "binary_binary"),
                "profile": self.active_profile,  # Pass active profile for backend
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format for compatibility
            results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata
                }
                results.append(result_dict)

            logger.info(f"Text search completed: {len(results)} results")

            # Store successful search in memory
            if self.is_memory_enabled() and results:
                self.remember_success(
                    query=query,
                    result={
                        "result_count": len(results),
                        "top_result": results[0] if results else None,
                    },
                    metadata={
                        "search_type": "text",
                        "top_k": top_k,
                        "ranking": kwargs.get("ranking", "hybrid_float_bm25"),
                    },
                )
                logger.debug("💾 Stored successful text search in memory")

            return results

        except Exception as e:
            logger.error(f"Text search failed: {e}")

            # Store failure in memory
            if self.is_memory_enabled():
                self.remember_failure(
                    query=query,
                    error=str(e),
                    metadata={"search_type": "text", "top_k": top_k},
                )
                logger.debug("💾 Stored search failure in memory")

            raise

    def search_by_video(
        self, video_data: bytes, filename: str, top_k: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search videos using video query.

        Args:
            video_data: Raw video file bytes
            filename: Original filename
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        logger.info(f"Video-to-video search with file: '{filename}' (top_k={top_k})")

        try:
            # Extract embeddings from uploaded video
            query_embeddings = self.video_processor.process_video_file(
                video_data, filename
            )

            # Execute search with backend using query_dict format
            if kwargs.get("start_date") or kwargs.get("end_date"):
                logger.warning("Date filters (start_date/end_date) not yet supported by VespaSearchBackend")

            query_dict = {
                "query": f"Video similarity search for {filename}",
                "type": "video",
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,
                "strategy": kwargs.get("ranking", "binary_binary"),
                "profile": self.active_profile,  # Pass active profile for backend
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format for compatibility
            results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata
                }
                results.append(result_dict)

            logger.info(f"Video search completed: {len(results)} results")

            # Store successful video search in memory
            if self.is_memory_enabled() and results:
                self.remember_success(
                    query=f"Video: {filename}",
                    result={
                        "result_count": len(results),
                        "top_result": results[0] if results else None,
                    },
                    metadata={
                        "search_type": "video",
                        "filename": filename,
                        "top_k": top_k,
                        "ranking": kwargs.get("ranking", "hybrid_float_bm25"),
                    },
                )
                logger.debug("💾 Stored successful video search in memory")

            return results

        except Exception as e:
            logger.error(f"Video search failed: {e}")

            # Store failure in memory
            if self.is_memory_enabled():
                self.remember_failure(
                    query=f"Video: {filename}",
                    error=str(e),
                    metadata={
                        "search_type": "video",
                        "filename": filename,
                        "top_k": top_k,
                    },
                )
                logger.debug("💾 Stored video search failure in memory")

            raise

    def search_by_image(
        self, image_data: bytes, filename: str, top_k: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search videos using image query.

        Args:
            image_data: Raw image file bytes
            filename: Original filename
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        logger.info(f"Image-to-video search with file: '{filename}' (top_k={top_k})")

        try:
            # Extract embeddings from uploaded image
            query_embeddings = self.video_processor.process_image_file(
                image_data, filename
            )

            # Execute search with backend using query_dict format
            if kwargs.get("start_date") or kwargs.get("end_date"):
                logger.warning("Date filters (start_date/end_date) not yet supported by VespaSearchBackend")

            query_dict = {
                "query": f"Image similarity search for {filename}",
                "type": "video",
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,
                "strategy": kwargs.get("ranking", "binary_binary"),
                "profile": self.active_profile,  # Pass active profile for backend
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format for compatibility
            results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata
                }
                results.append(result_dict)

            logger.info(f"Image search completed: {len(results)} results")

            # Store successful image search in memory
            if self.is_memory_enabled() and results:
                self.remember_success(
                    query=f"Image: {filename}",
                    result={
                        "result_count": len(results),
                        "top_result": results[0] if results else None,
                    },
                    metadata={
                        "search_type": "image",
                        "filename": filename,
                        "top_k": top_k,
                        "ranking": kwargs.get("ranking", "hybrid_float_bm25"),
                    },
                )
                logger.debug("💾 Stored successful image search in memory")

            return results

        except Exception as e:
            logger.error(f"Image search failed: {e}")

            # Store failure in memory
            if self.is_memory_enabled():
                self.remember_failure(
                    query=f"Image: {filename}",
                    error=str(e),
                    metadata={
                        "search_type": "image",
                        "filename": filename,
                        "top_k": top_k,
                    },
                )
                logger.debug("💾 Stored image search failure in memory")

            raise

    def process_enhanced_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced A2A task with multimedia support.

        Args:
            task: Enhanced A2A task

        Returns:
            Search results
        """
        if not task.messages:
            raise ValueError("Task contains no messages")

        last_message = task.messages[-1]
        results = []
        search_type = "unknown"

        for part in last_message.parts:
            if isinstance(part, DataPart):
                # Text search
                query_data = part.data
                query = query_data.get("query")

                if query:
                    search_type = "text"
                    text_results = self.search_by_text(
                        query=query,
                        top_k=query_data.get("top_k", 10),
                        start_date=query_data.get("start_date"),
                        end_date=query_data.get("end_date"),
                        ranking=query_data.get("ranking"),
                    )
                    results.extend(text_results)

            elif isinstance(part, VideoPart):
                # Video search
                search_type = "video"
                video_results = self.search_by_video(
                    video_data=part.video_data,
                    filename=part.filename or "uploaded_video.mp4",
                    top_k=10,  # Could be configurable
                )
                results.extend(video_results)

            elif isinstance(part, ImagePart):
                # Image search
                search_type = "image"
                image_results = self.search_by_image(
                    image_data=part.image_data,
                    filename=part.filename or "uploaded_image.jpg",
                    top_k=10,  # Could be configurable
                )
                results.extend(image_results)

            elif isinstance(part, TextPart):
                # Simple text search
                search_type = "text"
                text_results = self.search_by_text(query=part.text, top_k=10)
                results.extend(text_results)

        if not results:
            logger.warning("No valid search parts found in task")

        return {
            "task_id": task.id,
            "status": "completed",
            "search_type": search_type,
            "results": results,
            "total_results": len(results),
        }

    def search_with_routing_decision(
        self, routing_decision: RoutingDecision, top_k: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """
        Search with enhanced query and relationship context from DSPy routing.

        Args:
            routing_decision: Decision from RoutingAgent with relationship context
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            Enhanced search results with relationship context
        """
        logger.info(
            f"Relationship-aware search with confidence: {routing_decision.confidence:.3f}"
        )

        # Create enhanced search context
        search_context = SearchContext(
            original_query=routing_decision.routing_metadata.get("original_query", ""),
            enhanced_query=routing_decision.enhanced_query,
            entities=routing_decision.extracted_entities,
            relationships=routing_decision.extracted_relationships,
            routing_metadata=routing_decision.routing_metadata,
            confidence=routing_decision.confidence,
        )

        # Perform relationship-aware search
        return self.search_with_relationship_context(
            search_context, top_k=top_k, **kwargs
        )

    def search_with_relationship_context(
        self, context: SearchContext, top_k: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform relationship-aware video search with enhanced context.

        Args:
            context: Enhanced search context with relationships
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            Enhanced search results
        """
        try:
            # Determine which query to use (enhanced vs original)
            search_query = (
                context.enhanced_query
                if context.enhanced_query
                else context.original_query
            )

            logger.info(
                f"Enhanced search - Original: '{context.original_query}', Enhanced: '{context.enhanced_query}'"
            )

            # Build relationship-aware search parameters
            search_params = self._build_enhanced_search_params(context, top_k, **kwargs)

            # Execute search using enhanced query with backend
            query_embeddings = self.query_encoder.encode(search_query)

            if search_params.start_date or search_params.end_date:
                logger.warning("Date filters (start_date/end_date) not yet supported by VespaSearchBackend")

            query_dict = {
                "query": search_query,
                "type": "video",
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,
                "strategy": search_params.ranking_strategy
                or kwargs.get("ranking", "binary_binary"),
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format for compatibility
            raw_results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata
                }
                raw_results.append(result_dict)

            # Enhance results with relationship context
            enhanced_results = self._enhance_results_with_relationships(
                raw_results, context, search_params
            )

            return {
                "status": "completed",
                "search_type": "relationship_aware",
                "search_context": {
                    "original_query": context.original_query,
                    "enhanced_query": context.enhanced_query,
                    "entities_found": len(context.entities),
                    "relationships_found": len(context.relationships),
                    "routing_confidence": context.confidence,
                },
                "results": enhanced_results,
                "total_results": len(enhanced_results),
                "relationship_metadata": {
                    "entities": context.entities,
                    "relationships": context.relationships,
                    "enhancement_applied": bool(context.enhanced_query),
                    "confidence_threshold": search_params.confidence_threshold,
                },
            }

        except Exception as e:
            logger.error(f"Relationship-aware search failed: {e}")
            # Fallback to basic search
            return self._fallback_search(context.original_query, top_k, **kwargs)

    def _build_enhanced_search_params(
        self, context: SearchContext, top_k: int, **kwargs
    ) -> RelationshipAwareSearchParams:
        """Build enhanced search parameters from relationship context"""

        return RelationshipAwareSearchParams(
            query=context.enhanced_query or context.original_query,
            original_query=context.original_query,
            enhanced_query=context.enhanced_query,
            entities=context.entities,
            relationships=context.relationships,
            top_k=top_k,
            ranking_strategy=kwargs.get("ranking_strategy"),
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
            confidence_threshold=kwargs.get("confidence_threshold", 0.0),
            use_relationship_boost=kwargs.get("use_relationship_boost", True),
        )

    def _enhance_results_with_relationships(
        self,
        raw_results: List[Dict[str, Any]],
        context: SearchContext,
        search_params: RelationshipAwareSearchParams,
    ) -> List[Dict[str, Any]]:
        """Enhance search results with relationship context and scoring"""

        enhanced_results = []

        for result in raw_results:
            enhanced_result = result.copy()

            # Add relationship relevance scoring
            relationship_score = self._calculate_relationship_relevance(
                result, context.entities, context.relationships
            )

            # Add enhanced metadata
            enhanced_result["relationship_metadata"] = {
                "relationship_relevance_score": relationship_score,
                "matched_entities": self._find_matching_entities(
                    result, context.entities
                ),
                "matched_relationships": self._find_matching_relationships(
                    result, context.relationships
                ),
                "enhancement_confidence": context.confidence,
            }

            # Apply relationship boost to score if enabled
            if search_params.use_relationship_boost and relationship_score > 0:
                original_score = result.get("score", 0.0)
                boost_factor = 1.0 + (relationship_score * 0.2)  # Up to 20% boost
                enhanced_result["boosted_score"] = original_score * boost_factor
                enhanced_result["boost_applied"] = boost_factor
            else:
                enhanced_result["boosted_score"] = result.get("score", 0.0)
                enhanced_result["boost_applied"] = 1.0

            # Filter by confidence threshold
            if enhanced_result["boosted_score"] >= search_params.confidence_threshold:
                enhanced_results.append(enhanced_result)

        # Sort by boosted score
        enhanced_results.sort(key=lambda x: x["boosted_score"], reverse=True)

        return enhanced_results[: search_params.top_k]

    def _calculate_relationship_relevance(
        self,
        result: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> float:
        """Calculate relevance score based on entity and relationship matches"""

        score = 0.0

        # Check result content for entity matches
        result_text = " ".join(
            [
                result.get("title", ""),
                result.get("description", ""),
                result.get("content", ""),
                str(result.get("metadata", {})),
            ]
        ).lower()

        # Score entity matches
        for entity in entities:
            entity_text = entity.get("text", "").lower()
            if entity_text and entity_text in result_text:
                score += 0.3  # Entity match bonus

        # Score relationship matches (higher weight)
        for relationship in relationships:
            subject = relationship.get("subject", "").lower()
            relation = relationship.get("relation", "").lower()
            object_text = relationship.get("object", "").lower()

            matches = 0
            if subject and subject in result_text:
                matches += 1
            if relation and relation in result_text:
                matches += 1
            if object_text and object_text in result_text:
                matches += 1

            if matches >= 2:  # At least 2 parts of relationship match
                score += 0.5
            elif matches == 1:
                score += 0.2

        return min(score, 1.0)  # Cap at 1.0

    def _find_matching_entities(
        self, result: Dict[str, Any], entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find entities that match in the result"""

        result_text = " ".join(
            [
                result.get("title", ""),
                result.get("description", ""),
                result.get("content", ""),
            ]
        ).lower()

        matches = []
        for entity in entities:
            entity_text = entity.get("text", "").lower()
            if entity_text and entity_text in result_text:
                matches.append(entity)

        return matches

    def _find_matching_relationships(
        self, result: Dict[str, Any], relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find relationships that match in the result"""

        result_text = " ".join(
            [
                result.get("title", ""),
                result.get("description", ""),
                result.get("content", ""),
            ]
        ).lower()

        matches = []
        for relationship in relationships:
            subject = relationship.get("subject", "").lower()
            relation = relationship.get("relation", "").lower()
            object_text = relationship.get("object", "").lower()

            # Check if relationship components appear in result
            match_count = 0
            if subject and subject in result_text:
                match_count += 1
            if relation and relation in result_text:
                match_count += 1
            if object_text and object_text in result_text:
                match_count += 1

            if match_count >= 2:  # Require at least 2 components
                matches.append({**relationship, "match_strength": match_count / 3.0})

        return matches

    def _fallback_search(self, query: str, top_k: int, **kwargs) -> Dict[str, Any]:
        """Fallback to basic search when enhanced search fails"""

        logger.warning("Falling back to basic text search")

        try:
            results = self.search_by_text(query, top_k, **kwargs)
            return {
                "status": "completed_with_fallback",
                "search_type": "basic_text",
                "results": results,
                "total_results": len(results),
                "fallback_reason": "Enhanced search failed",
            }
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": [],
                "total_results": 0,
            }

    def process_routing_decision_task(
        self, routing_decision: RoutingDecision, task_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a routing decision as a task for A2A compatibility.

        Args:
            routing_decision: Enhanced routing decision from DSPy system
            task_id: Optional task ID

        Returns:
            A2A-compatible task result
        """

        result = self.search_with_routing_decision(routing_decision)

        # Add A2A task compatibility
        result["task_id"] = task_id or "routing_decision_search"
        result["agent"] = "video_search_agent"
        result["routing_decision_metadata"] = {
            "recommended_agent": routing_decision.recommended_agent,
            "confidence": routing_decision.confidence,
            "reasoning": routing_decision.reasoning,
            "fallback_agents": routing_decision.fallback_agents,
        }

        return result


# --- FastAPI Server ---
app = FastAPI(
    title="Enhanced Video Search Agent",
    description="Video search agent with support for text, video, and image queries",
    version="3.0.0",
)

# Global agent instance - initialized on startup
enhanced_video_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global enhanced_video_agent

    # Tenant ID is REQUIRED
    tenant_id = os.getenv("TENANT_ID")
    if not tenant_id:
        error_msg = "TENANT_ID environment variable is required"
        logger.error(error_msg)
        # Don't raise during tests
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise ValueError(error_msg)
        else:
            logger.warning("PYTEST_CURRENT_TEST detected - using 'test_tenant' as tenant_id")
            tenant_id = "test_tenant"

    agent_config = {
        "vespa_url": os.getenv("VESPA_URL", "http://localhost"),
        "vespa_port": int(os.getenv("VESPA_PORT", 8080)),
    }

    try:
        enhanced_video_agent = VideoSearchAgent(tenant_id=tenant_id, **agent_config)
        logger.info(f"Enhanced video search agent initialized for tenant: {tenant_id}")
    except Exception as e:
        logger.error(f"Failed to initialize enhanced agent: {e}")
        # Don't raise during tests
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not enhanced_video_agent:
        return {"status": "initializing", "agent": "enhanced_video_search"}

    return {
        "status": "healthy",
        "agent": "enhanced_video_search",
        "capabilities": ["text_search", "video_search", "image_search"],
        "embedding_type": enhanced_video_agent.embedding_type,
    }


@app.get("/agent.json")
async def get_agent_card():
    """Agent card with enhanced capabilities"""
    return {
        "name": "VideoSearchAgent",
        "description": "Advanced video search with text, video, and image query support",
        "url": "/process",
        "version": "3.0.0",
        "protocol": "a2a",
        "protocol_version": "0.2.1",
        "capabilities": [
            "text_search",
            "video_search",
            "image_search",
            "multimodal_search",
        ],
        "skills": [
            {
                "name": "textVideoSearch",
                "description": "Search videos using text queries",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "videoVideoSearch",
                "description": "Search videos using video files as queries",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "video_data": {"type": "string", "format": "binary"},
                        "filename": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["video_data"],
                },
            },
            {
                "name": "imageVideoSearch",
                "description": "Search videos using image files as queries",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_data": {"type": "string", "format": "binary"},
                        "filename": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["image_data"],
                },
            },
        ],
    }


@app.post("/process")
async def process_task(task: Dict[str, Any]):
    """Process enhanced search task"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = enhanced_video_agent.process_enhanced_task(task)
        return result
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/video")
async def upload_video_search(file: UploadFile = File(...), top_k: int = 10):
    """Upload video file and search for similar videos"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        video_data = await file.read()
        results = enhanced_video_agent.search_by_video(
            video_data=video_data,
            filename=file.filename or "uploaded_video.mp4",
            top_k=top_k,
        )

        return {
            "status": "completed",
            "search_type": "video",
            "filename": file.filename,
            "results": results,
            "total_results": len(results),
        }

    except Exception as e:
        logger.error(f"Video upload search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/image")
async def upload_image_search(file: UploadFile = File(...), top_k: int = 10):
    """Upload image file and search for similar videos"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        image_data = await file.read()
        results = enhanced_video_agent.search_by_image(
            image_data=image_data,
            filename=file.filename or "uploaded_image.jpg",
            top_k=top_k,
        )

        return {
            "status": "completed",
            "search_type": "image",
            "filename": file.filename,
            "results": results,
            "total_results": len(results),
        }

    except Exception as e:
        logger.error(f"Image upload search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/enhanced")
async def enhanced_search(params: RelationshipAwareSearchParams):
    """Enhanced search with relationship context"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Create search context from parameters
        search_context = SearchContext(
            original_query=params.original_query or params.query,
            enhanced_query=params.enhanced_query or params.query,
            entities=params.entities,
            relationships=params.relationships,
            routing_metadata={},
            confidence=1.0,  # Default confidence since no routing decision
        )

        # Perform relationship-aware search
        result = enhanced_video_agent.search_with_relationship_context(
            search_context,
            top_k=params.top_k,
            ranking_strategy=params.ranking_strategy,
            start_date=params.start_date,
            end_date=params.end_date,
            confidence_threshold=params.confidence_threshold,
            use_relationship_boost=params.use_relationship_boost,
        )

        return result

    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/routing-decision")
async def search_with_routing_decision(routing_decision: dict, top_k: int = 10):
    """Search using a routing decision from RoutingAgent"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Convert dict to RoutingDecision object (simplified)
        from cogniverse_agents.routing_agent import RoutingDecision

        decision = RoutingDecision(
            recommended_agent=routing_decision.get("recommended_agent", "video_search"),
            confidence=routing_decision.get("confidence", 0.0),
            reasoning=routing_decision.get("reasoning", ""),
            fallback_agents=routing_decision.get("fallback_agents", []),
            enhanced_query=routing_decision.get("enhanced_query", ""),
            extracted_entities=routing_decision.get("extracted_entities", []),
            extracted_relationships=routing_decision.get("extracted_relationships", []),
            routing_metadata=routing_decision.get("routing_metadata", {}),
        )

        # Process with relationship-aware search
        result = enhanced_video_agent.process_routing_decision_task(decision)

        return result

    except Exception as e:
        logger.error(f"Routing decision search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- A2A Protocol Enhanced Support ---
@app.post("/tasks/send")
async def handle_enhanced_a2a_task(task: dict):
    """Enhanced A2A task handler with routing decision support"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Check if this is a routing decision task
        if "routing_decision" in task:
            routing_data = task["routing_decision"]

            # Create RoutingDecision from task data
            from cogniverse_agents.routing_agent import RoutingDecision

            routing_decision = RoutingDecision(
                recommended_agent=routing_data.get("recommended_agent", "video_search"),
                confidence=routing_data.get("confidence", 0.0),
                reasoning=routing_data.get("reasoning", ""),
                fallback_agents=routing_data.get("fallback_agents", []),
                enhanced_query=routing_data.get("enhanced_query", ""),
                extracted_entities=routing_data.get("extracted_entities", []),
                extracted_relationships=routing_data.get("extracted_relationships", []),
                routing_metadata=routing_data.get("routing_metadata", {}),
            )

            return enhanced_video_agent.process_routing_decision_task(
                routing_decision, task_id=task.get("id", "enhanced_a2a_task")
            )

        # Handle standard enhanced A2A task
        else:
            # Process task directly as dict
            return enhanced_video_agent.process_enhanced_task(task)

    except Exception as e:
        logger.error(f"Enhanced A2A task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Video Search Agent Server")
    parser.add_argument(
        "--port", type=int, default=8002, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--profile", type=str, help="Video processing profile to use")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.profile:
        os.environ["VIDEO_PROFILE"] = args.profile

    logger.info(f"Starting Enhanced Video Search Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
