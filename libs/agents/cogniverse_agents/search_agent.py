"""
Generic Multi-Modal Search Agent with full A2A support.

This agent provides search capabilities across multiple modalities:
- Text queries (text-to-content search)
- Video queries (video-to-content search)
- Image queries (image-to-content search)
- Audio queries (audio-to-content search)
- Document queries (document-to-content search)

Enhanced with:
- DSPy 3.0 + A2A protocol integration
- Memory capabilities for learning from search patterns
- Relationship-aware search
- Multi-profile ensemble search with RRF fusion
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import dspy
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

# Enhanced query support from DSPy routing system
from cogniverse_agents.routing_agent import RoutingDecision
from cogniverse_agents.tools.a2a_utils import DataPart, TextPart
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.backend_registry import get_backend_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Type-Safe Input/Output/Dependencies
# =============================================================================


class SearchInput(AgentInput):
    """Type-safe input for search operations.

    The `rlm` field enables optional RLM (Recursive Language Model) inference
    for A/B testing. When set, RLM will be used to process large result sets.

    Example A/B Testing:
        # Group A: Standard search (no RLM)
        input_a = SearchInput(query="...", rlm=None)

        # Group B: RLM-enabled search
        input_b = SearchInput(query="...", rlm=RLMOptions(enabled=True))
    """

    query: str = Field(..., description="Search query")
    modality: str = Field(
        "video", description="Content modality (video/image/text/audio/document)"
    )
    top_k: int = Field(10, description="Number of results to return")
    profiles: Optional[List[str]] = Field(
        None, description="List of profiles for ensemble search"
    )
    rrf_k: int = Field(60, description="RRF constant for fusion")
    video_data: Optional[bytes] = Field(
        None, description="Video data for video-based search"
    )
    video_filename: Optional[str] = Field(None, description="Video filename")
    image_data: Optional[bytes] = Field(
        None, description="Image data for image-based search"
    )
    image_filename: Optional[str] = Field(None, description="Image filename")
    start_date: Optional[str] = Field(None, description="Start date filter")
    end_date: Optional[str] = Field(None, description="End date filter")

    # RLM configuration for A/B testing
    rlm: Optional[RLMOptions] = Field(
        None,
        description="RLM configuration. None=disabled, set RLMOptions to enable RLM inference for A/B testing",
    )


class SearchOutput(AgentOutput):
    """Type-safe output from search operations"""

    query: str = Field(..., description="Original query")
    enhanced_query: Optional[str] = Field(None, description="DSPy-enhanced query")
    modality: str = Field("video", description="Content modality")
    search_mode: str = Field(
        "single_profile", description="Search mode: single_profile, ensemble"
    )
    profile: Optional[str] = Field(None, description="Active profile (for single mode)")
    profiles: Optional[List[str]] = Field(
        None, description="Profiles used (for ensemble mode)"
    )
    rrf_k: Optional[int] = Field(None, description="RRF constant (for ensemble mode)")
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(0, description="Total number of results")

    # RLM synthesis (when RLM is enabled via input.rlm)
    rlm_synthesis: Optional[str] = Field(
        None, description="RLM-synthesized answer from results (only when RLM enabled)"
    )
    rlm_telemetry: Optional[Dict[str, Any]] = Field(
        None, description="RLM telemetry metrics for A/B testing"
    )


class SearchAgentDeps(AgentDeps):
    """Dependencies for search agent"""

    # schema_loader and config_manager are stored as Any because they are not pydantic models
    # They are set after initialization
    backend_url: str = Field("http://localhost", description="Backend URL")
    backend_port: int = Field(8080, description="Backend port")
    backend_config_port: Optional[int] = Field(None, description="Backend config port")
    profile: Optional[str] = Field(None, description="Active profile")
    backend_type: str = Field("vespa", description="Backend type")
    model_name: Optional[str] = Field(None, description="Model name")
    auto_create_memory_schema: bool = Field(
        True, description="Auto-create memory schema"
    )


# DSPy Module for Generic Search
class GenericSearchSignature(dspy.Signature):
    """DSPy signature for generic search operations"""

    query: str = dspy.InputField(desc="Search query")
    modality: str = dspy.InputField(
        desc="Content modality to search (video/image/text/audio/document)"
    )
    top_k: int = dspy.InputField(desc="Number of results to return")

    search_strategy: str = dspy.OutputField(desc="Recommended search strategy")
    enhanced_query: str = dspy.OutputField(desc="Enhanced query for better retrieval")
    confidence: float = dspy.OutputField(desc="Confidence in search approach (0-1)")


class GenericSearchModule(dspy.Module):
    """DSPy module for intelligent generic search"""

    def __init__(self):
        super().__init__()
        # Use lightweight reasoning for search optimization
        self.search_optimizer = dspy.ChainOfThought(GenericSearchSignature)

    def forward(self, query: str, modality: str = "video", top_k: int = 10):
        """Forward pass for search optimization"""
        try:
            result = self.search_optimizer(query=query, modality=modality, top_k=top_k)
            return result
        except Exception as e:
            logger.warning(f"DSPy search optimization failed: {e}, using defaults")
            # Fallback to simple prediction
            return dspy.Prediction(
                search_strategy="hybrid", enhanced_query=query, confidence=0.5
            )


@dataclass
class SearchContext:
    """Context for relationship-aware search"""

    original_query: str
    enhanced_query: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    routing_metadata: Dict[str, Any]
    confidence: float = 0.0


class RelationshipAwareSearchParams(BaseModel):
    """Enhanced search parameters with relationship context"""

    query: str = Field(..., description="Search query (may be enhanced)")
    modality: str = Field(
        "video", description="Content modality (video/image/text/audio/document)"
    )
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
    ranking_strategy: Optional[str] = Field(
        None, description="Backend ranking strategy"
    )
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


# --- Content Processing Components ---
class ContentProcessor:
    """Handles content upload, processing, and encoding to embeddings"""

    def __init__(self, query_encoder):
        self.query_encoder = query_encoder
        self.temp_dir = Path(tempfile.gettempdir()) / "search_agent"
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
        temp_video_path = self.temp_dir / f"temp_{filename}"

        try:
            with open(temp_video_path, "wb") as f:
                f.write(video_data)

            logger.info(f"Processing video file: {filename}")
            embeddings = self._extract_video_embeddings(temp_video_path)
            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings

        finally:
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
        temp_image_path = self.temp_dir / f"temp_{filename}"

        try:
            with open(temp_image_path, "wb") as f:
                f.write(image_data)

            logger.info(f"Processing image file: {filename}")
            embeddings = self._extract_image_embeddings(temp_image_path)
            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings

        finally:
            if temp_image_path.exists():
                temp_image_path.unlink()

    def _extract_video_embeddings(self, video_path: Path) -> np.ndarray:
        """Extract embeddings from video file using the query encoder"""
        if hasattr(self.query_encoder, "encode_video"):
            return self.query_encoder.encode_video(str(video_path))
        elif hasattr(self.query_encoder, "encode_frames"):
            return self._extract_frames_and_encode(video_path)
        else:
            raise NotImplementedError("Query encoder does not support video encoding")

    def _extract_image_embeddings(self, image_path: Path) -> np.ndarray:
        """Extract embeddings from image file using the query encoder"""
        if hasattr(self.query_encoder, "encode_image"):
            return self.query_encoder.encode_image(str(image_path))
        else:
            raise NotImplementedError("Query encoder does not support image encoding")

    def _extract_frames_and_encode(self, video_path: Path) -> np.ndarray:
        """Extract frames from video and encode them"""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            frames = []

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            step = max(fps, 30)

            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    if len(frames) >= 10:
                        break

            cap.release()

            if not frames:
                raise ValueError("No frames extracted from video")

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

                return np.mean(frame_embeddings, axis=0)

        except ImportError:
            raise ImportError(
                "OpenCV is required for video frame extraction. Install with: pip install opencv-python"
            )


# --- Generic Multi-Modal Search Agent ---
class SearchAgent(
    RLMAwareMixin,
    MemoryAwareMixin,
    A2AAgent[SearchInput, SearchOutput, SearchAgentDeps],
):
    """
    Type-safe generic multi-modal search agent with full A2A protocol support.

    Supports search across multiple modalities:
    - Video content search
    - Image search
    - Text search
    - Audio search
    - Document search

    Enhanced with memory capabilities, relationship-aware search, and ensemble search.
    """

    def __init__(
        self,
        deps: SearchAgentDeps,
        schema_loader=None,
        config_manager=None,
        port: int = 8002,
    ):
        """
        Initialize generic search agent with typed dependencies.

        Args:
            deps: Typed dependencies with tenant_id and configuration
            schema_loader: SchemaLoader instance (REQUIRED for dependency injection)
            config_manager: ConfigManager instance (optional, will create default if None)
            port: A2A server port

        Raises:
            TypeError: If deps is not SearchAgentDeps
            ValidationError: If deps fails Pydantic validation
            ValueError: If schema_loader is None
        """
        if schema_loader is None:
            raise ValueError(
                "schema_loader is required for SearchAgent. "
                "Dependency injection is mandatory - pass SchemaLoader instance explicitly."
            )

        # Debug: Log config_manager state
        logger_temp = logging.getLogger(__name__)
        if config_manager is None:
            logger_temp.warning(
                "⚠️  SearchAgent received config_manager=None, creating default"
            )
            from cogniverse_foundation.config.utils import create_default_config_manager

            config_manager = create_default_config_manager()
        else:
            db_path = (
                getattr(config_manager.store, "db_path", "unknown")
                if hasattr(config_manager, "store")
                else "no store"
            )
            logger_temp.warning(
                f"✅ SearchAgent received config_manager with DB: {db_path}"
            )

        # Store dependencies for use in initialization
        self.schema_loader = schema_loader
        self._config_manager = config_manager

        # Note: MemoryAwareMixin is initialized via cooperative inheritance
        # in super().__init__() call later. Memory will be initialized after
        # calling initialize_memory() explicitly.

        logger.info(f"Initializing SearchAgent for tenant: {deps.tenant_id}...")

        # Use deps values, with environment variable fallbacks
        backend_url = deps.backend_url or os.getenv("BACKEND_URL", "http://localhost")

        # Handle port from env var or deps
        env_port = os.getenv("BACKEND_PORT")
        if deps.backend_port != 8080:  # Non-default means explicitly set
            backend_port = deps.backend_port
        elif env_port:
            backend_port = int(env_port)
            if ":" in backend_url and backend_url.count(":") >= 2:
                backend_url = ":".join(backend_url.split(":")[:-1])
        elif ":" in backend_url and backend_url.count(":") >= 2:
            backend_port = int(backend_url.split(":")[-1])
            backend_url = ":".join(backend_url.split(":")[:-1])
        else:
            backend_port = 8080

        # Extract backend_host for memory initialization
        backend_host = backend_url.replace("http://", "").replace("https://", "")

        # Get config port for schema deployment
        backend_config_port = deps.backend_config_port
        if not backend_config_port:
            if backend_port == 8080:
                backend_config_port = 19071
            else:
                backend_config_port = backend_port + 10991

        # Create DSPy search module
        self.search_module = GenericSearchModule()

        # Create A2A config
        a2a_config = A2AAgentConfig(
            agent_name="search_agent",
            agent_description="Type-safe multi-modal search with text, video, image, audio, and document support",
            capabilities=[
                "search",
                "multi_modal_search",
                "video_search",
                "image_search",
                "text_search",
                "audio_search",
                "document_search",
                "relationship_aware_search",
                "ensemble_search",
            ],
            port=port,
            version="4.0.0",
        )

        # Initialize A2A base - this also sets up config_manager
        super().__init__(deps=deps, config=a2a_config, dspy_module=self.search_module)

        # Override config_manager with the injected one
        self.config_manager = config_manager

        # Load tenant-specific configuration (separate from A2AAgentConfig in self.config)
        from cogniverse_foundation.config.utils import get_config

        self.search_config = get_config(
            tenant_id=deps.tenant_id, config_manager=config_manager
        )

        # Initialize memory for search agent
        memory_initialized = self.initialize_memory(
            agent_name="search_agent",
            tenant_id=deps.tenant_id,
            backend_host=backend_host,
            backend_port=backend_port,
            backend_config_port=backend_config_port,
            auto_create_schema=deps.auto_create_memory_schema,
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        )
        if memory_initialized:
            logger.info(
                f"✅ Memory initialized for search_agent (tenant: {deps.tenant_id})"
            )
        else:
            logger.info("ℹ️  Memory disabled or not configured for search_agent")

        # Get model from active profile
        active_profile_raw = (
            os.getenv("BACKEND_PROFILE")
            or deps.profile
            or self.search_config.get("active_video_profile")
            or "video_colpali_smol500_mv_frame"
        )

        # Ensure active_profile is a string (config may return dict)
        if isinstance(active_profile_raw, dict):
            active_profile = active_profile_raw.get(
                "name", "video_colpali_smol500_mv_frame"
            )
        else:
            active_profile = (
                str(active_profile_raw)
                if active_profile_raw
                else "video_colpali_smol500_mv_frame"
            )

        self.active_profile = active_profile

        backend_config_data = self.search_config.get("backend", {})
        profiles_raw = backend_config_data.get("profiles", {})

        # Ensure profiles is a dict (handle list format or other edge cases)
        if isinstance(profiles_raw, dict):
            profiles = profiles_raw
        elif isinstance(profiles_raw, list):
            # Convert list of profile dicts to dict keyed by name
            profiles = {
                p.get("name", f"profile_{i}"): p
                for i, p in enumerate(profiles_raw)
                if isinstance(p, dict)
            }
        else:
            profiles = {}

        if active_profile and isinstance(profiles, dict) and active_profile in profiles:
            model_name = profiles[active_profile].get(
                "embedding_model", "vidore/colsmol-500m"
            )
            self.embedding_type = profiles[active_profile].get(
                "embedding_type", "frame_based"
            )
        else:
            model_name = deps.model_name or "vidore/colsmol-500m"
            self.embedding_type = "frame_based"

        backend_type = deps.backend_type or self.search_config.get(
            "backend_type", "vespa"
        )

        # Initialize search backend via backend registry
        try:
            logger.info("Generic Search Agent configuration:")
            logger.info(f"  - Tenant ID: {deps.tenant_id}")
            logger.info(f"  - Backend Type: {backend_type}")
            logger.info(f"  - Backend URL: {backend_url}")
            logger.info(f"  - Backend Port: {backend_port}")
            logger.info(f"  - Active Profile: {active_profile}")
            logger.info(f"  - Model Name: {model_name}")

            backend_config = {
                "url": backend_url,
                "port": backend_port,
                "config_port": backend_config_port,
                "schema_name": active_profile,
                "tenant_id": deps.tenant_id,
                "profile": active_profile,
                "backend": backend_config_data,
            }

            registry = get_backend_registry()
            self.search_backend = registry.get_search_backend(
                backend_type,
                deps.tenant_id,
                backend_config,
                config_manager=self.config_manager,
                schema_loader=self.schema_loader,
            )

            logger.info(
                f"Search backend initialized at {backend_url}:{backend_port} "
                f"for tenant {deps.tenant_id} with profile {active_profile}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize search backend: {e}")
            raise

        # Initialize query encoder
        try:
            self.query_encoder = QueryEncoderFactory.create_encoder(
                active_profile, model_name, config=self.search_config
            )
            logger.info(f"Query encoder initialized for profile: {active_profile}")
        except Exception as e:
            logger.error(f"Failed to initialize query encoder: {e}")
            raise

        # Initialize content processor
        self.content_processor = ContentProcessor(self.query_encoder)

        logger.info(f"SearchAgent initialized for tenant: {deps.tenant_id}")

    def _fuse_results_rrf(
        self,
        profile_results: Dict[str, List[Dict[str, Any]]],
        k: int = 60,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple profiles using Reciprocal Rank Fusion (RRF).

        Formula: score(doc) = Σ_profiles (1 / (k + rank_in_profile))

        Args:
            profile_results: Dict mapping profile names to their result lists
            k: RRF constant (default 60, typical range: 20-100)
            top_k: Number of final results to return

        Returns:
            Fused and re-ranked results
        """
        logger.info(
            f"Fusing results from {len(profile_results)} profiles using RRF (k={k})"
        )

        # Accumulate RRF scores by document ID
        doc_scores = {}  # doc_id -> {score, result_data}

        # Calculate RRF scores
        for profile_name, results in profile_results.items():
            for rank, result in enumerate(results):
                doc_id = result["id"]
                rrf_score = 1.0 / (k + rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "score": 0.0,
                        "result": result,
                        "profile_ranks": {},
                        "profile_scores": {},
                    }

                doc_scores[doc_id]["score"] += rrf_score
                doc_scores[doc_id]["profile_ranks"][profile_name] = rank
                doc_scores[doc_id]["profile_scores"][profile_name] = result.get(
                    "score", 0.0
                )

        # Sort by RRF score
        fused_results = []
        for doc_id, doc_data in doc_scores.items():
            result = doc_data["result"].copy()
            result["rrf_score"] = doc_data["score"]
            result["profile_ranks"] = doc_data["profile_ranks"]
            result["profile_scores"] = doc_data["profile_scores"]
            result["num_profiles"] = len(doc_data["profile_ranks"])
            fused_results.append(result)

        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        logger.info(
            f"RRF fusion complete: {len(fused_results)} unique documents from {len(profile_results)} profiles"
        )

        return fused_results[:top_k]

    async def _search_ensemble(
        self,
        query: str,
        profiles: List[str],
        modality: str = "video",
        top_k: int = 10,
        rrf_k: int = 60,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Execute parallel search across multiple profiles and fuse with RRF.

        Args:
            query: Text search query
            profiles: List of profile names to query
            modality: Content modality to search
            top_k: Number of final results to return
            rrf_k: RRF constant for fusion
            **kwargs: Additional search parameters

        Returns:
            Fused results from all profiles
        """
        logger.info(f"Ensemble search with {len(profiles)} profiles: {profiles}")

        import asyncio

        # Pre-compute query embeddings for each profile in parallel
        async def encode_for_profile(profile_name: str):
            """Encode query for specific profile"""
            try:
                # Get profile config
                backend_config_data = self.search_config.get("backend", {})
                profiles_config = backend_config_data.get("profiles", {})

                if profile_name not in profiles_config:
                    logger.warning(
                        f"Profile {profile_name} not found in config, using active profile encoder"
                    )
                    return profile_name, self.query_encoder.encode(query)

                # Create encoder for this profile
                profile_config = profiles_config[profile_name]
                model_name = profile_config.get(
                    "embedding_model", "vidore/colsmol-500m"
                )

                # Reuse active encoder if same model
                if profile_name == self.active_profile:
                    embeddings = self.query_encoder.encode(query)
                else:
                    # Create temporary encoder for this profile
                    from cogniverse_core.query.encoders import QueryEncoderFactory

                    encoder = QueryEncoderFactory.create_encoder(
                        profile_name, model_name, config=self.search_config
                    )
                    embeddings = encoder.encode(query)

                logger.debug(
                    f"Encoded query for profile {profile_name}: shape {embeddings.shape}"
                )
                return profile_name, embeddings

            except Exception as e:
                logger.error(f"Failed to encode query for profile {profile_name}: {e}")
                return profile_name, None

        # Encode queries in parallel
        encoding_tasks = [encode_for_profile(p) for p in profiles]
        profile_embeddings = await asyncio.gather(*encoding_tasks)

        # Filter successful encodings
        valid_embeddings = {
            profile: emb for profile, emb in profile_embeddings if emb is not None
        }

        if not valid_embeddings:
            raise ValueError("Failed to encode query for any profile")

        logger.info(
            f"Encoded query for {len(valid_embeddings)}/{len(profiles)} profiles"
        )

        # Execute searches in parallel using shared thread pool
        import concurrent.futures

        loop = asyncio.get_event_loop()

        async def search_profile(profile_name: str, query_embeddings, executor):
            """Execute search for single profile"""
            try:
                query_dict = {
                    "query": query,
                    "type": modality,
                    "query_embeddings": query_embeddings,
                    "top_k": top_k * 2,  # Fetch 2x results for better fusion
                    "filters": None,
                    "strategy": kwargs.get("ranking", "binary_binary"),
                    "profile": profile_name,
                }

                # Execute synchronous search in shared thread pool
                search_results = await loop.run_in_executor(
                    executor, self.search_backend.search, query_dict
                )

                # Convert SearchResult objects to dict
                results = []
                for sr in search_results:
                    result_dict = {
                        "id": sr.document.id,
                        "score": sr.score,
                        **sr.document.metadata,
                    }
                    results.append(result_dict)

                logger.info(f"Profile {profile_name}: {len(results)} results")
                return profile_name, results

            except Exception as e:
                logger.error(f"Search failed for profile {profile_name}: {e}")
                return profile_name, []

        # Create shared thread pool and run searches in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(valid_embeddings)
        ) as executor:
            search_tasks = [
                search_profile(profile, embeddings, executor)
                for profile, embeddings in valid_embeddings.items()
            ]
            profile_results_list = await asyncio.gather(*search_tasks)

        # Convert to dict
        profile_results = {
            profile: results for profile, results in profile_results_list if results
        }

        if not profile_results:
            logger.warning("No results from any profile")
            return []

        # Fuse results using RRF
        fused_results = self._fuse_results_rrf(profile_results, k=rrf_k, top_k=top_k)

        # Store successful ensemble search in memory
        if self.is_memory_enabled() and fused_results:
            self.remember_success(
                query=query,
                result={
                    "result_count": len(fused_results),
                    "top_result": fused_results[0] if fused_results else None,
                    "profiles_used": list(profile_results.keys()),
                },
                metadata={
                    "search_type": "ensemble",
                    "modality": modality,
                    "profiles": profiles,
                    "rrf_k": rrf_k,
                    "top_k": top_k,
                },
            )
            logger.debug("💾 Stored successful ensemble search in memory")

        return fused_results

    def _search_by_text(
        self, query: str, modality: str = "video", top_k: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Internal: Search content using text query.

        Args:
            query: Text search query
            modality: Content modality to search (video/image/text/audio/document)
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        logger.info(f"Text-to-{modality} search: '{query}' (top_k={top_k})")

        # Retrieve relevant search patterns from memory
        if self.is_memory_enabled():
            memory_context = self.get_relevant_context(query, top_k=3)
            if memory_context:
                logger.info(f"📚 Retrieved memory context for query: {query[:50]}...")

        try:
            # Encode text query
            query_embeddings = self.query_encoder.encode(query)

            # Execute search with backend
            if kwargs.get("start_date") or kwargs.get("end_date"):
                logger.warning(
                    "Date filters (start_date/end_date) not yet supported by backend"
                )

            query_dict = {
                "query": query,
                "type": modality,
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,
                "strategy": kwargs.get("ranking", "binary_binary"),
                "profile": self.active_profile,
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format
            results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata,
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
                        "modality": modality,
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
                    metadata={
                        "search_type": "text",
                        "modality": modality,
                        "top_k": top_k,
                    },
                )
                logger.debug("💾 Stored search failure in memory")

            raise

    def search_by_text(
        self, query: str, modality: str = "video", top_k: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search content using text query.

        Args:
            query: Text search query
            modality: Content modality to search (video/image/text/audio/document)
            top_k: Number of results to return
            **kwargs: Additional search parameters (ranking, etc.)

        Returns:
            List of search results
        """
        return self._search_by_text(query, modality=modality, top_k=top_k, **kwargs)

    def _search_by_video(
        self,
        video_data: bytes,
        filename: str,
        modality: str = "video",
        top_k: int = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Internal: Search content using video query.

        Args:
            video_data: Raw video file bytes
            filename: Original filename
            modality: Content modality to search (usually "video")
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        logger.info(
            f"Video-to-{modality} search with file: '{filename}' (top_k={top_k})"
        )

        try:
            # Extract embeddings from uploaded video
            query_embeddings = self.content_processor.process_video_file(
                video_data, filename
            )

            # Execute search with backend
            if kwargs.get("start_date") or kwargs.get("end_date"):
                logger.warning(
                    "Date filters (start_date/end_date) not yet supported by backend"
                )

            query_dict = {
                "query": f"Video similarity search for {filename}",
                "type": modality,
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,
                "strategy": kwargs.get("ranking", "binary_binary"),
                "profile": self.active_profile,
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format
            results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata,
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
                        "modality": modality,
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
                        "modality": modality,
                        "filename": filename,
                        "top_k": top_k,
                    },
                )
                logger.debug("💾 Stored video search failure in memory")

            raise

    def _search_by_image(
        self,
        image_data: bytes,
        filename: str,
        modality: str = "video",
        top_k: int = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Internal: Search content using image query.

        Args:
            image_data: Raw image file bytes
            filename: Original filename
            modality: Content modality to search (video/image)
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        logger.info(
            f"Image-to-{modality} search with file: '{filename}' (top_k={top_k})"
        )

        try:
            # Extract embeddings from uploaded image
            query_embeddings = self.content_processor.process_image_file(
                image_data, filename
            )

            # Execute search with backend
            if kwargs.get("start_date") or kwargs.get("end_date"):
                logger.warning(
                    "Date filters (start_date/end_date) not yet supported by backend"
                )

            query_dict = {
                "query": f"Image similarity search for {filename}",
                "type": modality,
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,
                "strategy": kwargs.get("ranking", "binary_binary"),
                "profile": self.active_profile,
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format
            results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata,
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
                        "modality": modality,
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
                        "modality": modality,
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
        modality = "video"  # Default modality

        for part in last_message.parts:
            if isinstance(part, DataPart):
                # Text search
                query_data = part.data
                query = query_data.get("query")
                modality = query_data.get("modality", "video")

                if query:
                    search_type = "text"
                    text_results = self._search_by_text(
                        query=query,
                        modality=modality,
                        top_k=query_data.get("top_k", 10),
                        start_date=query_data.get("start_date"),
                        end_date=query_data.get("end_date"),
                        ranking=query_data.get("ranking"),
                    )
                    results.extend(text_results)

            elif isinstance(part, VideoPart):
                # Video search
                search_type = "video"
                video_results = self._search_by_video(
                    video_data=part.video_data,
                    filename=part.filename or "uploaded_video.mp4",
                    modality="video",
                    top_k=10,
                )
                results.extend(video_results)

            elif isinstance(part, ImagePart):
                # Image search
                search_type = "image"
                image_results = self._search_by_image(
                    image_data=part.image_data,
                    filename=part.filename or "uploaded_image.jpg",
                    modality=modality,
                    top_k=10,
                )
                results.extend(image_results)

            elif isinstance(part, TextPart):
                # Simple text search
                search_type = "text"
                text_results = self._search_by_text(
                    query=part.text, modality="video", top_k=10
                )
                results.extend(text_results)

        if not results:
            logger.warning("No valid search parts found in task")

        return {
            "task_id": task.id,
            "status": "completed",
            "search_type": search_type,
            "modality": modality,
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
        Perform relationship-aware content search with enhanced context.

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
                logger.warning(
                    "Date filters (start_date/end_date) not yet supported by backend"
                )

            query_dict = {
                "query": search_query,
                "type": search_params.modality,
                "query_embeddings": query_embeddings,
                "top_k": top_k,
                "filters": None,
                "strategy": search_params.ranking_strategy
                or kwargs.get("ranking", "binary_binary"),
            }

            search_results = self.search_backend.search(query_dict)

            # Convert SearchResult objects to dict format
            raw_results = []
            for sr in search_results:
                result_dict = {
                    "id": sr.document.id,
                    "score": sr.score,
                    **sr.document.metadata,
                }
                raw_results.append(result_dict)

            # Enhance results with relationship context
            enhanced_results = self._enhance_results_with_relationships(
                raw_results, context, search_params
            )

            return {
                "status": "completed",
                "search_type": "relationship_aware",
                "modality": search_params.modality,
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
            return self._fallback_search(context.original_query, top_k, **kwargs)

    def _build_enhanced_search_params(
        self, context: SearchContext, top_k: int, **kwargs
    ) -> RelationshipAwareSearchParams:
        """Build enhanced search parameters from relationship context"""

        return RelationshipAwareSearchParams(
            query=context.enhanced_query or context.original_query,
            modality=kwargs.get("modality", "video"),
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
                boost_factor = 1.0 + (relationship_score * 0.2)
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
                score += 0.3

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

            if matches >= 2:
                score += 0.5
            elif matches == 1:
                score += 0.2

        return min(score, 1.0)

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

            match_count = 0
            if subject and subject in result_text:
                match_count += 1
            if relation and relation in result_text:
                match_count += 1
            if object_text and object_text in result_text:
                match_count += 1

            if match_count >= 2:
                matches.append({**relationship, "match_strength": match_count / 3.0})

        return matches

    def _fallback_search(self, query: str, top_k: int, **kwargs) -> Dict[str, Any]:
        """Fallback to basic search when enhanced search fails"""

        logger.warning("Falling back to basic text search")

        try:
            results = self._search_by_text(query, top_k=top_k, **kwargs)
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
        result["agent"] = "search_agent"
        result["routing_decision_metadata"] = {
            "recommended_agent": routing_decision.recommended_agent,
            "confidence": routing_decision.confidence,
            "reasoning": routing_decision.reasoning,
            "fallback_agents": routing_decision.fallback_agents,
        }

        return result

    # ==========================================================================
    # Type-safe process method (required by AgentBase)
    # ==========================================================================

    async def _process_impl(
        self, input: Union[SearchInput, Dict[str, Any]]
    ) -> SearchOutput:
        """
        Process search request with typed input/output.

        Args:
            input: Typed input with query, modality, top_k, etc. (or dict)

        Returns:
            SearchOutput with results and metadata
        """
        # Handle dict input for backward compatibility with tests
        if isinstance(input, dict):
            input = self.validate_input(input)

        query = input.query
        modality = input.modality
        top_k = input.top_k

        # Track search mode and profile info for output
        search_mode = "single_profile"
        profile = self.active_profile
        profiles_used = None
        rrf_k_used = None
        enhanced_query = None

        # Check for ensemble mode (multiple profiles)
        if input.profiles and len(input.profiles) > 1:
            logger.info(f"Ensemble mode detected: {len(input.profiles)} profiles")
            search_mode = "ensemble"
            profile = None
            profiles_used = input.profiles
            rrf_k_used = input.rrf_k

            results = await self._search_ensemble(
                query=query,
                profiles=input.profiles,
                modality=modality,
                top_k=top_k,
                rrf_k=input.rrf_k,
                start_date=input.start_date,
                end_date=input.end_date,
            )
        elif input.video_data:
            # Video-based search
            results = self._search_by_video(
                video_data=input.video_data,
                filename=input.video_filename or "video.mp4",
                modality=modality,
                top_k=top_k,
            )
        elif input.image_data:
            # Image-based search
            results = self._search_by_image(
                image_data=input.image_data,
                filename=input.image_filename or "image.jpg",
                modality=modality,
                top_k=top_k,
            )
        else:
            # Text-based search with optional DSPy optimization
            search_query = query
            try:
                dspy_result = self.search_module.forward(
                    query=query, modality=modality, top_k=top_k
                )
                # Use enhanced query from DSPy if confidence is high
                if hasattr(dspy_result, "enhanced_query") and hasattr(
                    dspy_result, "confidence"
                ):
                    if dspy_result.confidence > 0.7:
                        search_query = dspy_result.enhanced_query
                        enhanced_query = search_query
                        logger.info(f"Using DSPy-enhanced query: {search_query}")
            except Exception as e:
                logger.warning(f"DSPy optimization failed: {e}, using original query")

            results = self._search_by_text(
                query=search_query,
                modality=modality,
                top_k=top_k,
                start_date=input.start_date,
                end_date=input.end_date,
            )

        # RLM Processing: Synthesize answer from results if RLM is enabled
        rlm_synthesis = None
        rlm_telemetry = None

        # Build context from results for RLM decision
        results_context = "\n".join(
            [
                f"Result {i+1}: {r.get('summary', r.get('title', str(r)))}"
                for i, r in enumerate(results[:20])
            ]
        )

        if self.should_use_rlm_for_query(input.rlm, results_context):
            logger.info(f"RLM enabled for query: {query[:50]}...")
            try:
                rlm_result = self.process_with_rlm(
                    query=query,
                    context=results_context,
                    rlm_options=input.rlm,
                )
                rlm_synthesis = rlm_result.answer
                rlm_telemetry = self.get_rlm_telemetry(rlm_result, len(results_context))
                logger.info(
                    f"RLM synthesis complete: depth={rlm_result.depth_reached}, "
                    f"calls={rlm_result.total_calls}, latency={rlm_result.latency_ms:.0f}ms"
                )
            except Exception as e:
                logger.error(f"RLM processing failed: {e}")
                rlm_telemetry = {
                    "rlm_enabled": False,
                    "rlm_attempted": True,
                    "rlm_error": str(e),
                }

        return SearchOutput(
            query=query,
            enhanced_query=enhanced_query,
            modality=modality,
            search_mode=search_mode,
            profile=profile,
            profiles=profiles_used,
            rrf_k=rrf_k_used,
            results=results,
            total_results=len(results),
            rlm_synthesis=rlm_synthesis,
            rlm_telemetry=rlm_telemetry,
        )

    # Note: _dspy_to_a2a_output and _get_agent_skills handled by A2AAgent base class


# --- FastAPI Server ---
app = FastAPI(
    title="Generic Multi-Modal Search Agent",
    description="Search agent with support for text, video, image, audio, and document queries",
    version="4.0.0",
)

# Global agent instance
search_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global search_agent

    tenant_id = os.getenv("TENANT_ID")
    if not tenant_id:
        error_msg = "TENANT_ID environment variable is required"
        logger.error(error_msg)
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise ValueError(error_msg)
        else:
            logger.warning(
                "PYTEST_CURRENT_TEST detected - using 'test_tenant' as tenant_id"
            )
            tenant_id = "test_tenant"

    try:
        # Create deps with configuration from environment
        deps = SearchAgentDeps(
            tenant_id=tenant_id,
            backend_url=os.getenv("BACKEND_URL", "http://localhost"),
            backend_port=int(os.getenv("BACKEND_PORT", 8080)),
        )

        # Note: schema_loader is required for SearchAgent
        # In production, inject via proper dependency container
        from cogniverse_core.schemas.schema_loader import SchemaLoader

        schema_loader = SchemaLoader()

        search_agent = SearchAgent(deps=deps, schema_loader=schema_loader)
        logger.info(f"Generic search agent initialized for tenant: {tenant_id}")
    except Exception as e:
        logger.error(f"Failed to initialize search agent: {e}")
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not search_agent:
        return {"status": "initializing", "agent": "search_agent"}

    return {
        "status": "healthy",
        "agent": "search_agent",
        "capabilities": [
            "text_search",
            "video_search",
            "image_search",
            "multi_modal_search",
            "relationship_aware_search",
        ],
        "embedding_type": search_agent.embedding_type,
    }


@app.get("/agent.json")
async def get_agent_card():
    """Agent card with enhanced capabilities"""
    return {
        "name": "SearchAgent",
        "description": "Type-safe multi-modal search with text, video, image, audio, and document support",
        "url": "/process",
        "version": "4.0.0",
        "protocol": "a2a",
        "protocol_version": "0.2.1",
        "capabilities": [
            "text_search",
            "video_search",
            "image_search",
            "audio_search",
            "document_search",
            "multi_modal_search",
            "relationship_aware_search",
        ],
        "skills": search_agent.get_agent_skills() if search_agent else [],
    }


@app.post("/process")
async def process_task(task: Dict[str, Any]):
    """Process search task"""
    if not search_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = search_agent.process_enhanced_task(task)
        return result
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/video")
async def upload_video_search(
    file: UploadFile = File(...), top_k: int = 10, modality: str = "video"
):
    """Upload video file and search for similar content"""
    if not search_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        video_data = await file.read()
        results = search_agent.search_by_video(
            video_data=video_data,
            filename=file.filename or "uploaded_video.mp4",
            modality=modality,
            top_k=top_k,
        )

        return {
            "status": "completed",
            "search_type": "video",
            "modality": modality,
            "filename": file.filename,
            "results": results,
            "total_results": len(results),
        }

    except Exception as e:
        logger.error(f"Video upload search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/image")
async def upload_image_search(
    file: UploadFile = File(...), top_k: int = 10, modality: str = "video"
):
    """Upload image file and search for similar content"""
    if not search_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        image_data = await file.read()
        results = search_agent.search_by_image(
            image_data=image_data,
            filename=file.filename or "uploaded_image.jpg",
            modality=modality,
            top_k=top_k,
        )

        return {
            "status": "completed",
            "search_type": "image",
            "modality": modality,
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
    if not search_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Create search context from parameters
        search_context = SearchContext(
            original_query=params.original_query or params.query,
            enhanced_query=params.enhanced_query or params.query,
            entities=params.entities,
            relationships=params.relationships,
            routing_metadata={},
            confidence=1.0,
        )

        # Perform relationship-aware search
        result = search_agent.search_with_relationship_context(
            search_context,
            top_k=params.top_k,
            modality=params.modality,
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
    if not search_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        from cogniverse_agents.routing_agent import RoutingDecision

        decision = RoutingDecision(
            recommended_agent=routing_decision.get("recommended_agent", "search_agent"),
            confidence=routing_decision.get("confidence", 0.0),
            reasoning=routing_decision.get("reasoning", ""),
            fallback_agents=routing_decision.get("fallback_agents", []),
            enhanced_query=routing_decision.get("enhanced_query", ""),
            extracted_entities=routing_decision.get("extracted_entities", []),
            extracted_relationships=routing_decision.get("extracted_relationships", []),
            routing_metadata=routing_decision.get("routing_metadata", {}),
        )

        result = search_agent.process_routing_decision_task(decision)

        return result

    except Exception as e:
        logger.error(f"Routing decision search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- A2A Protocol Enhanced Support ---
@app.post("/tasks/send")
async def handle_enhanced_a2a_task(task: dict):
    """Enhanced A2A task handler with routing decision support"""
    if not search_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Check if this is a routing decision task
        if "routing_decision" in task:
            routing_data = task["routing_decision"]

            from cogniverse_agents.routing_agent import RoutingDecision

            routing_decision = RoutingDecision(
                recommended_agent=routing_data.get("recommended_agent", "search_agent"),
                confidence=routing_data.get("confidence", 0.0),
                reasoning=routing_data.get("reasoning", ""),
                fallback_agents=routing_data.get("fallback_agents", []),
                enhanced_query=routing_data.get("enhanced_query", ""),
                extracted_entities=routing_data.get("extracted_entities", []),
                extracted_relationships=routing_data.get("extracted_relationships", []),
                routing_metadata=routing_data.get("routing_metadata", {}),
            )

            return search_agent.process_routing_decision_task(
                routing_decision, task_id=task.get("id", "enhanced_a2a_task")
            )

        # Handle standard enhanced A2A task
        else:
            return search_agent.process_enhanced_task(task)

    except Exception as e:
        logger.error(f"Enhanced A2A task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generic Multi-Modal Search Agent Server"
    )
    parser.add_argument(
        "--port", type=int, default=8002, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--profile", type=str, help="Processing profile to use")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.profile:
        os.environ["BACKEND_PROFILE"] = args.profile

    logger.info(f"Starting Generic Multi-Modal Search Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
