"""
Routing Agent with DSPy 3.0, Production Features, and Advanced Optimization

This agent combines all advanced routing capabilities:
- DSPy 3.0 with local SmolLM 3B for intelligent routing
- Relationship extraction and query enhancement
- Production features: caching, parallel execution, memory
- Advanced optimization with GRPO and SIMBA
- Multi-modal reranking and cross-modal optimization

Key Features:
- Complete routing solution with all production features
- Extracts relationships for intelligent routing decisions
- Enhances queries with relationship context
- Maintains A2A protocol compatibility
- Full telemetry and MLflow integration
- Type-safe input/output with Pydantic validation
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import ConfigDict, Field

if TYPE_CHECKING:
    pass  # Type hints are inline now

# DSPy 3.0 imports
import dspy

# Advanced optimization
from cogniverse_agents.routing.advanced_optimizer import (
    AdvancedOptimizerConfig,
    AdvancedRoutingOptimizer,
)
from cogniverse_agents.routing.contextual_analyzer import ContextualAnalyzer
from cogniverse_agents.routing.cross_modal_optimizer import CrossModalOptimizer
from cogniverse_agents.routing.dspy_relationship_router import (
    ComposableQueryAnalysisModule,
    DSPyAdvancedRoutingModule,
)
from cogniverse_agents.routing.dspy_routing_signatures import (
    BasicQueryAnalysisSignature,
)
from cogniverse_agents.routing.lazy_executor import LazyModalityExecutor

# MLflow integration
from cogniverse_agents.routing.mlflow_integration import (
    ExperimentConfig,
    MLflowIntegration,
)
from cogniverse_agents.routing.modality_cache import ModalityCacheManager
from cogniverse_agents.routing.modality_metrics import ModalityMetricsTracker
from cogniverse_agents.routing.parallel_executor import ParallelAgentExecutor
from cogniverse_agents.routing.query_enhancement_engine import QueryEnhancementPipeline
from cogniverse_agents.routing.relationship_extraction_tools import (
    GLiNERRelationshipExtractor,
    SpaCyDependencyAnalyzer,
)
from cogniverse_agents.search.multi_modal_reranker import (
    MultiModalReranker,
    QueryModality,
)

# Type-safe agent base
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput

# Production features from RoutingAgent
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareAgentMixin

# Centralized LLM config
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

# A2A protocol imports


# =============================================================================
# Type-Safe Input/Output/Dependencies Models
# =============================================================================


class RoutingInput(AgentInput):
    """
    Type-safe input for routing agent.

    All fields are validated by Pydantic at runtime.
    IDE provides autocomplete for all fields.
    """

    query: str = Field(..., description="User query to route")
    tenant_id: str = Field(..., description="Tenant identifier (per-request)")
    context: Optional[str] = Field(None, description="Optional context information")
    require_orchestration: Optional[bool] = Field(
        None, description="Force orchestration decision"
    )


class RoutingOutput(AgentOutput):
    """
    Type-safe output from routing agent.

    Pydantic model for routing output with validation.
    """

    query: str = Field(..., description="Original query")
    recommended_agent: str = Field(..., description="Agent to route to")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence")
    reasoning: str = Field(..., description="Reasoning for the decision")
    fallback_agents: List[str] = Field(
        default_factory=list, description="Fallback agents if primary fails"
    )
    enhanced_query: str = Field("", description="Enhanced query with context")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted relationships"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    query_variants: List[Dict[str, str]] = Field(
        default_factory=list, description="Query variants for parallel fusion search"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Decision timestamp"
    )

    model_config = ConfigDict(extra="forbid")

    # Convenience properties mapping internal field names to external API names
    @property
    def extracted_entities(self) -> List[Dict[str, Any]]:
        return self.entities

    @property
    def extracted_relationships(self) -> List[Dict[str, Any]]:
        return self.relationships

    @property
    def routing_metadata(self) -> Dict[str, Any]:
        return self.metadata


class RoutingDeps(AgentDeps):
    """
    Type-safe dependencies for routing agent.

    Contains configuration and services the agent needs.
    Tenant-agnostic at startup — tenant_id arrives per-request via RoutingInput.
    """

    telemetry_config: Any = Field(..., description="Telemetry configuration")

    # Centralized LLM configuration (resolved from llm_config.resolve("routing_agent"))
    llm_config: LLMEndpointConfig = Field(
        default_factory=lambda: LLMEndpointConfig(
            model="ollama/smollm3:3b",
            api_base="http://localhost:11434",
            temperature=0.1,
            max_tokens=1000,
        ),
        description="LLM endpoint configuration for DSPy routing",
    )

    # Routing thresholds
    confidence_threshold: float = Field(0.7, description="Min confidence threshold")
    relationship_weight: float = Field(0.3, description="Weight for relationships")
    enhancement_weight: float = Field(0.4, description="Weight for enhancement")

    # Agent capabilities mapping
    agent_capabilities: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "search_agent": [
                "video_content_search",
                "visual_query_analysis",
                "multimodal_retrieval",
                "temporal_video_analysis",
            ],
            "summarizer_agent": [
                "content_summarization",
                "key_point_extraction",
                "document_synthesis",
                "report_generation",
            ],
            "detailed_report_agent": [
                "comprehensive_analysis",
                "detailed_reporting",
                "data_correlation",
                "in_depth_investigation",
            ],
        },
        description="Agent capability mapping",
    )

    # Query fusion configuration (variants always generated by composable module)
    query_fusion_config: Dict[str, Any] = Field(
        default_factory=lambda: {"include_original": True, "rrf_k": 60},
        description="Query fusion config: include_original and rrf_k",
    )

    # ComposableQueryAnalysisModule thresholds
    entity_confidence_threshold: float = Field(
        0.6, description="Min avg GLiNER confidence for fast path"
    )
    min_entities_for_fast_path: int = Field(
        1, description="Min entities needed for GLiNER fast path"
    )

    # Enable/disable features
    enable_relationship_extraction: bool = Field(
        True, description="Enable relationship extraction"
    )
    enable_query_enhancement: bool = Field(True, description="Enable query enhancement")
    enable_fallback_routing: bool = Field(True, description="Enable fallback routing")
    enable_confidence_calibration: bool = Field(
        True, description="Enable confidence calibration"
    )
    enable_advanced_optimization: bool = Field(
        True, description="Enable GRPO optimization"
    )

    # Production features
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_size_per_modality: int = Field(1000, description="Cache size per modality")
    cache_ttl_seconds: int = Field(300, description="Cache TTL in seconds")
    enable_parallel_execution: bool = Field(
        True, description="Enable parallel execution"
    )
    max_concurrent_agents: int = Field(5, description="Max concurrent agents")
    enable_memory: bool = Field(False, description="Enable memory (requires Mem0)")
    memory_backend_host: Optional[str] = Field(
        None, description="Backend host for memory storage"
    )
    memory_backend_port: Optional[int] = Field(
        None, description="Backend port for memory storage"
    )
    memory_llm_model: Optional[str] = Field(
        None, description="LLM model for memory extraction"
    )
    memory_embedding_model: Optional[str] = Field(
        None, description="Embedding model for memory search"
    )
    memory_llm_base_url: Optional[str] = Field(
        None, description="LLM API base URL for memory"
    )
    memory_config_manager: Any = Field(None, description="ConfigManager for memory")
    memory_schema_loader: Any = Field(None, description="SchemaLoader for memory")
    enable_contextual_analysis: bool = Field(
        True, description="Enable contextual analysis"
    )
    enable_metrics_tracking: bool = Field(True, description="Enable metrics tracking")
    enable_multi_modal_reranking: bool = Field(
        True, description="Enable multi-modal reranking"
    )
    enable_cross_modal_optimization: bool = Field(
        True, description="Enable cross-modal optimization"
    )

    # Advanced optimization configuration
    optimizer_config: Optional[AdvancedOptimizerConfig] = Field(
        None, description="GRPO optimizer config"
    )

    # MLflow integration configuration
    enable_mlflow_tracking: bool = Field(False, description="Enable MLflow tracking")
    mlflow_experiment_name: str = Field(
        "routing_agent", description="MLflow experiment name"
    )
    mlflow_tracking_uri: str = Field(
        "http://localhost:5000", description="MLflow tracking URI"
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class RoutingAgent(
    A2AAgent[RoutingInput, RoutingOutput, RoutingDeps],
    MemoryAwareMixin,
    TenantAwareAgentMixin,
):
    """
    Type-safe Routing Agent with complete production features.

    This agent combines all routing capabilities:
    - DSPy 3.0 with local SmolLM 3B for intelligent routing
    - Relationship extraction and query enhancement
    - Full production features (caching, parallel execution, memory)
    - Advanced optimization (GRPO, SIMBA)
    - Multi-modal reranking and cross-modal optimization
    - Type-safe input/output with Pydantic validation

    Type Parameters (inherited from A2AAgent):
        RoutingInput: Query, context, orchestration flag
        RoutingOutput: Agent recommendation, confidence, entities, relationships
        RoutingDeps: Telemetry config, model config, feature flags
    """

    def __init__(
        self,
        deps: RoutingDeps,
        port: int = 8001,
    ):
        """
        Initialize Routing Agent with typed dependencies.

        Args:
            deps: Typed dependencies with telemetry_config and routing settings
            port: A2A server port

        Raises:
            TypeError: If deps is not RoutingDeps
            ValidationError: If deps fails Pydantic validation
        """
        # Store telemetry config for production components
        self.telemetry_config = deps.telemetry_config
        self.logger = logging.getLogger(__name__)

        # Initialize telemetry manager first (needed by enhancement pipeline, MLflow, etc.)
        self._initialize_telemetry_manager()

        # Initialize DSPy 3.0 with local SmolLM
        self._configure_dspy(deps)

        # Initialize query analysis and enhancement pipeline
        self._initialize_enhancement_pipeline(deps)

        # Initialize DSPy routing module
        self._initialize_routing_module()

        # Initialize advanced optimization (lazy per-tenant)
        self._initialize_advanced_optimizer(deps)

        # Initialize MLflow tracking
        self._initialize_mlflow_tracking(deps)

        # Initialize production features (telemetry_manager already done)
        self._initialize_production_components(deps)

        # Create A2A config from deps
        a2a_config = A2AAgentConfig(
            agent_name="routing_agent",
            agent_description="Intelligent routing with relationship extraction and query enhancement",
            capabilities=self._get_routing_capabilities(deps),
            port=port,
        )

        # Initialize A2A base with typed deps
        super().__init__(
            deps=deps,
            config=a2a_config,
            dspy_module=self.routing_module,
        )

        self._routing_stats = {
            "total_queries": 0,
            "successful_routes": 0,
            "enhanced_queries": 0,
            "relationship_extractions": 0,
            "confidence_scores": [],
        }

    @property
    def enhanced_system_available(self) -> bool:
        """Check if enhanced routing features are available"""
        has_analysis_module = (
            self.deps.enable_relationship_extraction
            and self.analysis_module is not None
        )

        has_query_enhancement = (
            self.deps.enable_query_enhancement and self.query_enhancer is not None
        )

        has_routing_module = self.routing_module is not None

        return has_analysis_module and has_query_enhancement and has_routing_module

    def _initialize_telemetry_manager(self) -> None:
        """Initialize telemetry manager early (needed by enhancement pipeline, MLflow, etc.)."""
        if self.telemetry_config.enabled:
            from cogniverse_foundation.telemetry.manager import TelemetryManager

            self.telemetry_manager = TelemetryManager(config=self.telemetry_config)
            self.logger.info("Telemetry manager initialized")
        else:
            self.telemetry_manager = None

    def _configure_dspy(self, deps: RoutingDeps) -> None:
        """Configure DSPy LM instance (scoped via dspy.context, not global)."""
        self._dspy_lm = create_dspy_lm(deps.llm_config)
        self.logger.info(
            f"Created DSPy LM: {deps.llm_config.model} at {deps.llm_config.api_base}"
        )

    def _initialize_enhancement_pipeline(self, deps: RoutingDeps) -> None:
        """Initialize query analysis and enhancement using ComposableQueryAnalysisModule."""
        try:
            # Create the composable analysis module (used by both pipeline and routing)
            if deps.enable_relationship_extraction or deps.enable_query_enhancement:
                gliner_extractor = GLiNERRelationshipExtractor()
                spacy_analyzer = SpaCyDependencyAnalyzer()
                self.analysis_module = ComposableQueryAnalysisModule(
                    gliner_extractor=gliner_extractor,
                    spacy_analyzer=spacy_analyzer,
                    entity_confidence_threshold=deps.entity_confidence_threshold,
                    min_entities_for_fast_path=deps.min_entities_for_fast_path,
                )
                self.logger.info(
                    f"ComposableQueryAnalysisModule initialized "
                    f"(threshold={deps.entity_confidence_threshold}, "
                    f"min_entities={deps.min_entities_for_fast_path})"
                )
            else:
                self.analysis_module = None

            # Query enhancement pipeline wraps the composable module
            if deps.enable_query_enhancement and self.analysis_module:
                telemetry_provider = self._get_telemetry_provider("default")
                self.query_enhancer = QueryEnhancementPipeline(
                    analysis_module=self.analysis_module,
                    query_fusion_config=deps.query_fusion_config,
                    telemetry_provider=telemetry_provider,
                    tenant_id="default",
                )
                self.logger.info("Query enhancement pipeline initialized")
            else:
                self.query_enhancer = None

        except Exception as e:
            self.logger.error(f"Failed to initialize enhancement pipeline: {e}")
            self.analysis_module = None
            self.query_enhancer = None

    def _initialize_routing_module(self) -> None:
        """Initialize DSPy routing module"""
        try:
            self.routing_module = DSPyAdvancedRoutingModule(
                analysis_module=getattr(self, "analysis_module", None),
            )
            self.logger.info("DSPy advanced routing module initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize routing module: {e}")
            # Create basic fallback module
            self._create_fallback_routing_module()

    def _create_fallback_routing_module(self) -> None:
        """Create fallback routing module for graceful degradation"""

        class FallbackRoutingModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.analyze = dspy.ChainOfThought(BasicQueryAnalysisSignature)

            def forward(self, query: str, context: Optional[str] = None):
                try:
                    return self.analyze(query=query)
                except Exception:
                    # Ultimate fallback
                    return dspy.Prediction(
                        primary_intent="search",
                        needs_video_search=True,
                        recommended_agent="search_agent",
                        confidence=0.5,
                    )

        self.routing_module = FallbackRoutingModule()
        self.logger.warning("Using fallback routing module")

    def _initialize_advanced_optimizer(self, deps: RoutingDeps) -> None:
        """Configure advanced optimization (lazy per-tenant init)."""
        self._optimizer_config = deps.optimizer_config or AdvancedOptimizerConfig()
        self._enable_advanced_optimization = deps.enable_advanced_optimization
        # Per-tenant optimizers created lazily in _get_optimizer(tenant_id)
        self._tenant_optimizers: Dict[str, AdvancedRoutingOptimizer] = {}
        self.grpo_optimizer = None  # Kept for attribute access; use _get_optimizer()

    def _get_optimizer(self, tenant_id: str) -> Optional["AdvancedRoutingOptimizer"]:
        """Get or create per-tenant optimizer (lazy initialization)."""
        if not self._enable_advanced_optimization:
            return None
        if tenant_id not in self._tenant_optimizers:
            try:
                telemetry_provider = self._get_telemetry_provider(tenant_id)
                self._tenant_optimizers[tenant_id] = AdvancedRoutingOptimizer(
                    tenant_id=tenant_id,
                    llm_config=self.deps.llm_config,
                    telemetry_provider=telemetry_provider,
                    config=self._optimizer_config,
                )
                self.logger.info(
                    f"Advanced routing optimizer initialized for tenant: {tenant_id}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize advanced optimizer: {e}")
                return None
        return self._tenant_optimizers[tenant_id]

    def _get_telemetry_provider(self, tenant_id: str):
        """Get telemetry provider for a tenant."""
        if self.telemetry_manager is not None:
            return self.telemetry_manager.get_provider(tenant_id=tenant_id)
        raise RuntimeError(
            "Telemetry manager not initialized — cannot create artifact storage"
        )

    def _get_cross_modal_optimizer(
        self, tenant_id: str
    ) -> Optional[CrossModalOptimizer]:
        """Get or create per-tenant cross-modal optimizer (lazy initialization)."""
        if not self._enable_cross_modal_optimization:
            return None
        if tenant_id not in self._tenant_cross_modal_optimizers:
            try:
                telemetry_provider = self._get_telemetry_provider(tenant_id)
                self._tenant_cross_modal_optimizers[tenant_id] = CrossModalOptimizer(
                    telemetry_provider=telemetry_provider,
                    tenant_id=tenant_id,
                )
                self.logger.info(
                    f"Cross-modal optimizer initialized for tenant: {tenant_id}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize cross-modal optimizer: {e}")
                return None
        return self._tenant_cross_modal_optimizers[tenant_id]

    def _ensure_memory_for_tenant(self, tenant_id: str) -> None:
        """Initialize memory for a tenant if not already done."""
        if not self.deps.enable_memory:
            return
        if tenant_id in self._memory_initialized_tenants:
            return
        try:
            self.initialize_memory(
                agent_name="routing_agent",
                tenant_id=tenant_id,
                backend_host=self._memory_config["backend_host"],
                backend_port=self._memory_config["backend_port"],
                llm_model=self._memory_config["llm_model"],
                embedding_model=self._memory_config["embedding_model"],
                llm_base_url=self._memory_config["llm_base_url"],
                config_manager=self._memory_config["config_manager"],
                schema_loader=self._memory_config["schema_loader"],
            )
            self._memory_initialized_tenants.add(tenant_id)
            self.logger.info(f"Memory initialized for tenant: {tenant_id}")
        except Exception as e:
            self.logger.error(
                f"Failed to initialize memory for tenant {tenant_id}: {e}"
            )

    def _initialize_mlflow_tracking(self, deps: RoutingDeps) -> None:
        """Initialize MLflow tracking integration."""
        try:
            if deps.enable_mlflow_tracking:
                mlflow_config = ExperimentConfig(
                    experiment_name=deps.mlflow_experiment_name,
                    tracking_uri=deps.mlflow_tracking_uri,
                    description="Enhanced routing agent with DSPy 3.0, GRPO, SIMBA, and adaptive thresholds",
                    tags={
                        "component": "routing_agent",
                        "version": "1.0.0",
                        "features": "dspy,grpo,simba,adaptive_thresholds",
                    },
                    auto_log_parameters=True,
                    auto_log_metrics=True,
                    track_dspy_modules=True,
                    track_optimization_state=True,
                    track_threshold_parameters=True,
                )

                telemetry_provider = self._get_telemetry_provider("default")
                self.mlflow_integration = MLflowIntegration(
                    config=mlflow_config,
                    telemetry_provider=telemetry_provider,
                    tenant_id="default",
                )
                self.logger.info("MLflow tracking integration initialized")
            else:
                self.mlflow_integration = None
                self.logger.info("MLflow tracking disabled")

        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow tracking: {e}")
            self.mlflow_integration = None

    def _initialize_production_components(self, deps: RoutingDeps) -> None:
        """Initialize production-ready components from RoutingAgent"""
        try:
            # telemetry_manager already initialized in _initialize_telemetry_manager()

            # Initialize caching
            if deps.enable_caching:
                self.cache_manager = ModalityCacheManager(
                    cache_size_per_modality=deps.cache_size_per_modality
                )
                self.logger.info("Cache manager initialized")
            else:
                self.cache_manager = None

            # Initialize parallel execution
            if deps.enable_parallel_execution:
                self.parallel_executor = ParallelAgentExecutor(
                    max_concurrent_agents=deps.max_concurrent_agents
                )
                self.logger.info("Parallel executor initialized")
            else:
                self.parallel_executor = None

            # Initialize memory system (if enabled)
            if deps.enable_memory:
                for field in (
                    "memory_backend_host",
                    "memory_backend_port",
                    "memory_llm_model",
                    "memory_embedding_model",
                    "memory_llm_base_url",
                    "memory_config_manager",
                    "memory_schema_loader",
                ):
                    if getattr(deps, field) is None:
                        raise ValueError(
                            f"enable_memory=True but {field} is None — "
                            "all memory_* fields are required when memory is enabled"
                        )
                # Memory is initialized lazily per-tenant on first request
                # via _ensure_memory_for_tenant(tenant_id)
                self._memory_config = {
                    "backend_host": deps.memory_backend_host,
                    "backend_port": deps.memory_backend_port,
                    "llm_model": deps.memory_llm_model,
                    "embedding_model": deps.memory_embedding_model,
                    "llm_base_url": deps.memory_llm_base_url,
                    "config_manager": deps.memory_config_manager,
                    "schema_loader": deps.memory_schema_loader,
                }
                self._memory_initialized_tenants: set = set()
                self.logger.info("Memory system configured (lazy per-tenant init)")

            # Initialize contextual analysis
            if deps.enable_contextual_analysis:
                self.contextual_analyzer = ContextualAnalyzer(
                    max_history_size=50,
                    context_window_minutes=30,
                    min_preference_count=3,
                )
                self.logger.info("Contextual analyzer initialized")
            else:
                self.contextual_analyzer = None

            # Initialize metrics tracking
            if deps.enable_metrics_tracking:
                self.metrics_tracker = ModalityMetricsTracker(window_size=1000)
                self.logger.info("Metrics tracker initialized")
            else:
                self.metrics_tracker = None

            # Initialize multi-modal reranking
            if deps.enable_multi_modal_reranking:
                self.multi_modal_reranker = MultiModalReranker()
                self.logger.info("Multi-modal reranker initialized")
            else:
                self.multi_modal_reranker = None

            # Cross-modal optimization (lazy per-tenant)
            self._enable_cross_modal_optimization = deps.enable_cross_modal_optimization
            self._tenant_cross_modal_optimizers: Dict[str, CrossModalOptimizer] = {}
            self.cross_modal_optimizer = None  # Use _get_cross_modal_optimizer()

            # Initialize lazy executor (always on for efficiency)
            self.lazy_executor = LazyModalityExecutor()
            self.logger.info("Lazy executor initialized")

            self.logger.info("Production components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize production components: {e}")
            # Set defaults for graceful degradation
            self.telemetry_manager = None
            self.cache_manager = None
            self.parallel_executor = None
            self.contextual_analyzer = None
            self.metrics_tracker = None
            self.multi_modal_reranker = None
            self.cross_modal_optimizer = None
            self.lazy_executor = None

    def _get_routing_capabilities(self, deps: RoutingDeps) -> List[str]:
        """Get routing agent capabilities"""
        capabilities = ["intelligent_routing", "query_analysis", "agent_orchestration"]

        if deps.enable_relationship_extraction:
            capabilities.extend(
                ["relationship_extraction", "entity_recognition", "semantic_analysis"]
            )

        if deps.enable_query_enhancement:
            capabilities.extend(
                ["query_enhancement", "query_rewriting", "context_enrichment"]
            )

        if deps.enable_advanced_optimization:
            capabilities.extend(
                [
                    "advanced_optimization",
                    "adaptive_learning",
                    "performance_optimization",
                ]
            )

        if deps.enable_mlflow_tracking:
            capabilities.extend(
                ["mlflow_tracking", "experiment_management", "performance_monitoring"]
            )

        # Production capabilities
        if deps.enable_caching:
            capabilities.append("result_caching")
        if deps.enable_parallel_execution:
            capabilities.append("parallel_agent_execution")
        if deps.enable_memory:
            capabilities.append("conversation_memory")
        if deps.enable_contextual_analysis:
            capabilities.append("contextual_analysis")
        if deps.enable_metrics_tracking:
            capabilities.append("performance_metrics")
        if deps.enable_multi_modal_reranking:
            capabilities.append("multi_modal_reranking")
        if deps.enable_cross_modal_optimization:
            capabilities.append("cross_modal_optimization")

        return capabilities

    def _agent_to_modality(self, agent_name: str) -> QueryModality:
        """
        Map agent name to its corresponding modality.

        Uses the routing decision to determine modality, not keywords.
        """
        agent_modality_map = {
            "search_agent": QueryModality.VIDEO,
            "video_search_agent": QueryModality.VIDEO,
            "document_agent": QueryModality.DOCUMENT,
            "document_search_agent": QueryModality.DOCUMENT,
            "image_search_agent": QueryModality.IMAGE,
            "audio_search_agent": QueryModality.AUDIO,
            "audio_analysis_agent": QueryModality.AUDIO,
            "summarizer_agent": QueryModality.TEXT,
            "detailed_report_agent": QueryModality.TEXT,
        }
        return agent_modality_map.get(agent_name, QueryModality.TEXT)

    async def route_query(
        self,
        query: str,
        context: Optional[str] = None,
        tenant_id: Optional[str] = None,
        require_orchestration: Optional[bool] = None,
    ) -> RoutingOutput:
        """
        Enhanced query routing with relationship extraction and query enhancement

        This is the main routing method that combines all routing components
        """
        self._routing_stats["total_queries"] += 1
        start_time = datetime.now()

        # Enforce mandatory tenant_id for telemetry isolation
        if not tenant_id:
            raise ValueError(
                "tenant_id is required for routing operations. "
                "Tenant isolation is mandatory - cannot default to 'unknown'."
            )

        # Create telemetry span context manager if available
        span_context = None
        if hasattr(self, "telemetry_manager") and self.telemetry_manager:
            self.logger.info(f"Creating telemetry span for tenant: {tenant_id}")
            span_context = self.telemetry_manager.span(
                "cogniverse.routing", tenant_id=tenant_id
            )
        else:
            self.logger.debug("No telemetry manager available, using nullcontext")
            # Create a dummy context manager that does nothing
            from contextlib import nullcontext

            span_context = nullcontext()

        with span_context as span:
            if span:
                self.logger.info(f"Telemetry span created successfully: {span}")
            else:
                self.logger.warning(
                    "Span context returned None - no telemetry span created"
                )
            try:
                # Check cache first by searching all modality buckets (if enabled)
                if self.cache_manager:
                    cached_decision = self.cache_manager.get_cached_result_any_modality(
                        query=query, ttl_seconds=self.deps.cache_ttl_seconds
                    )
                    if cached_decision:
                        self.logger.info(f"Cache hit for query: {query[:50]}...")
                        return cached_decision

                # Add contextual analysis (if enabled)
                if self.contextual_analyzer and tenant_id:
                    contextual_insights = self.contextual_analyzer.get_contextual_hints(
                        current_query=query
                    )
                    self.logger.info(f"Contextual insights: {contextual_insights}")

                # Extract entities/relationships and enhance query
                (
                    entities,
                    relationships,
                    enhanced_query,
                    enhancement_metadata,
                ) = await self._analyze_and_enhance_query(query)

                # DSPy-powered routing decision (baseline)
                baseline_routing_result = await self._make_routing_decision(
                    original_query=query,
                    enhanced_query=enhanced_query,
                    entities=entities,
                    relationships=relationships,
                    context=context,
                )

                # Apply GRPO optimization if available
                optimized_routing_result = await self._apply_grpo_optimization(
                    query=query,
                    entities=entities,
                    relationships=relationships,
                    enhanced_query=enhanced_query,
                    baseline_prediction=baseline_routing_result,
                    tenant_id=tenant_id,
                )

                # Use optimized result if available, otherwise baseline
                final_routing_result = (
                    optimized_routing_result or baseline_routing_result
                )

                # Determine if orchestration is needed
                needs_orchestration = self._assess_orchestration_need(
                    query,
                    entities,
                    relationships,
                    final_routing_result,
                    require_orchestration,
                )

                # Create structured routing decision
                decision = RoutingOutput(
                    query=query,
                    recommended_agent=final_routing_result.get(
                        "recommended_agent", "search_agent"
                    ),
                    confidence=final_routing_result.get("confidence", 0.5),
                    reasoning=final_routing_result.get(
                        "reasoning", "Default routing decision"
                    ),
                    fallback_agents=self._get_fallback_agents(
                        final_routing_result.get("recommended_agent")
                    ),
                    enhanced_query=enhanced_query,
                    entities=entities,
                    relationships=relationships,
                    metadata={
                        **enhancement_metadata,
                        "processing_time_ms": (
                            datetime.now() - start_time
                        ).total_seconds()
                        * 1000,
                        "baseline_routing_result": baseline_routing_result,
                        "optimized_routing_result": optimized_routing_result,
                        "grpo_applied": optimized_routing_result is not None,
                        "tenant_id": tenant_id,
                        "needs_orchestration": needs_orchestration,
                        "orchestration_signals": self._get_orchestration_signals(
                            query, entities, relationships
                        ),
                    },
                    query_variants=enhancement_metadata.get("query_variants", []),
                )

                # Update statistics
                self._update_routing_stats(decision)

                # Map recommended agent to modality for caching and metrics
                agent_modality = self._agent_to_modality(decision.recommended_agent)

                # Cache the decision (if enabled)
                if self.cache_manager:
                    self.cache_manager.cache_result(
                        query=query, modality=agent_modality, result=decision
                    )

                # Track metrics (if enabled)
                if self.metrics_tracker:
                    self.metrics_tracker.record_modality_execution(
                        modality=agent_modality,
                        latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        success=True,
                    )

                # Update contextual analyzer (if enabled)
                if self.contextual_analyzer and tenant_id:
                    self.contextual_analyzer.update_context(
                        query=query,
                        detected_modalities=[decision.recommended_agent],
                        result=decision,
                        result_count=1 if decision.confidence > 0.5 else 0,
                    )

                # Log successful routing
                self.logger.info(
                    f"Query routed to {decision.recommended_agent} "
                    f"(confidence: {decision.confidence:.3f}, "
                    f"relationships: {len(relationships)}, "
                    f"enhanced: {'yes' if enhanced_query != query else 'no'})"
                )

                # Update telemetry span with final attributes
                if span and hasattr(span, "set_attribute"):
                    self.logger.info("Setting telemetry span attributes")
                    span.set_attribute("routing.query", query)
                    span.set_attribute(
                        "routing.chosen_agent", decision.recommended_agent
                    )
                    span.set_attribute("routing.confidence", decision.confidence)
                    span.set_attribute(
                        "routing.processing_time",
                        (datetime.now() - start_time).total_seconds() * 1000,
                    )
                    span.set_attribute("routing.reasoning", decision.reasoning)
                    span.set_attribute("routing.entities_count", len(entities))
                    span.set_attribute(
                        "routing.relationships_count", len(relationships)
                    )
                    span.set_attribute("routing.enhanced", enhanced_query != query)
                    self.logger.info(
                        f"Telemetry span attributes set for query: {query[:50]}..."
                    )
                else:
                    self.logger.warning(
                        f"Cannot set span attributes - span={span}, has_set_attribute={hasattr(span, 'set_attribute') if span else False}"
                    )

                return decision

            except Exception as e:
                self.logger.error(f"Routing failed for query '{query}': {e}")
                return self._create_fallback_decision(query, str(e))

    async def _analyze_and_enhance_query(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
        """Extract entities/relationships and enhance query in one pass.

        Returns:
            Tuple of (entities, relationships, enhanced_query, enhancement_metadata)
        """
        if not self.deps.enable_query_enhancement or not self.query_enhancer:
            return [], [], query, {}

        try:
            enhancement_result = (
                await self.query_enhancer.enhance_query_with_relationships(
                    query, search_context="general"
                )
            )

            entities = enhancement_result.get("extracted_entities", [])
            relationships = enhancement_result.get("extracted_relationships", [])
            enhanced_query = enhancement_result.get("enhanced_query", query)
            query_variants = enhancement_result.get("query_variants", [])

            enhancement_metadata = {
                "quality_score": enhancement_result.get("quality_score", 0.5),
                "enhancement_strategy": enhancement_result.get(
                    "enhancement_strategy", "none"
                ),
                "semantic_expansions": enhancement_result.get(
                    "semantic_expansions", []
                ),
                "query_variants": query_variants,
                "rrf_k": self.query_enhancer.query_fusion_config.get("rrf_k", 60),
            }

            if entities:
                self._routing_stats["relationship_extractions"] += 1
            if enhanced_query != query:
                self._routing_stats["enhanced_queries"] += 1

            self.logger.debug(
                f"Analyzed query: {len(entities)} entities, "
                f"{len(relationships)} relationships, "
                f"enhanced: '{query}' -> '{enhanced_query}'"
            )

            return entities, relationships, enhanced_query, enhancement_metadata

        except Exception as e:
            self.logger.warning(f"Query analysis and enhancement failed: {e}")
            return [], [], query, {}

    async def _apply_grpo_optimization(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhanced_query: str,
        baseline_prediction: Dict[str, Any],
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply GRPO optimization to routing decision."""
        optimizer = self._get_optimizer(tenant_id)
        if not optimizer:
            return None

        try:
            optimized_result = await optimizer.optimize_routing_decision(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
                baseline_prediction=baseline_prediction,
            )

            self.logger.debug(
                f"GRPO optimization applied: baseline_agent={baseline_prediction.get('recommended_agent')}, "
                f"optimized_agent={optimized_result.get('recommended_agent')}"
            )

            return optimized_result

        except Exception as e:
            self.logger.warning(f"GRPO optimization failed: {e}")
            return None

    async def _make_routing_decision(
        self,
        original_query: str,
        enhanced_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make DSPy-powered routing decision."""
        try:
            # Use enhanced query for routing if available, fallback to original
            routing_query = (
                enhanced_query if enhanced_query != original_query else original_query
            )

            # Prepare context with relationship information
            routing_context = self._prepare_routing_context(
                context, entities, relationships
            )

            # DSPy routing decision (scoped LM via context)
            with dspy.context(lm=self._dspy_lm):
                dspy_result = self.routing_module.forward(
                    query=routing_query, context=routing_context
                )

            # Extract routing information from DSPy result
            routing_info = {
                "recommended_agent": getattr(
                    dspy_result, "recommended_agent", "search_agent"
                ),
                "confidence": getattr(dspy_result, "confidence", 0.5),
                "reasoning": getattr(dspy_result, "reasoning", "DSPy routing decision"),
                "primary_intent": getattr(dspy_result, "primary_intent", "search"),
                "complexity_score": getattr(dspy_result, "complexity_score", 0.5),
            }

            # Apply confidence calibration if enabled
            if self.deps.enable_confidence_calibration:
                routing_info["confidence"] = self._calibrate_confidence(
                    routing_info["confidence"],
                    len(entities),
                    len(relationships),
                    enhanced_query != original_query,
                )

            return routing_info

        except Exception as e:
            self.logger.error(f"DSPy routing decision failed: {e}")
            return {
                "recommended_agent": "search_agent",
                "confidence": 0.3,
                "reasoning": f"Fallback routing due to error: {e}",
                "primary_intent": "search",
                "complexity_score": 0.5,
            }

    def _prepare_routing_context(
        self,
        original_context: Optional[str],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> str:
        """Prepare enriched context for routing decision"""
        context_parts = []

        if original_context:
            context_parts.append(f"Original context: {original_context}")

        if entities:
            entity_texts = [e.get("text", "") for e in entities[:5]]  # Top 5 entities
            context_parts.append(f"Key entities: {', '.join(entity_texts)}")

        if relationships:
            relationship_summaries = []
            for rel in relationships[:3]:  # Top 3 relationships
                subj = rel.get("subject", "")
                obj = rel.get("object", "")
                relation = rel.get("relation", "")
                if subj and obj and relation:
                    relationship_summaries.append(f"{subj} {relation} {obj}")

            if relationship_summaries:
                context_parts.append(
                    f"Key relationships: {'; '.join(relationship_summaries)}"
                )

        return " | ".join(context_parts) if context_parts else ""

    def _calibrate_confidence(
        self,
        base_confidence: float,
        entity_count: int,
        relationship_count: int,
        query_enhanced: bool,
    ) -> float:
        """Calibrate confidence based on enhancement features"""
        confidence = base_confidence

        # Boost confidence if we have good entity extraction
        if entity_count > 2:
            confidence += 0.1 * min(entity_count / 5, 0.2)

        # Boost confidence if we have relationship information
        if relationship_count > 0:
            confidence += self.deps.relationship_weight * min(
                relationship_count / 3, 0.15
            )

        # Boost confidence if query was enhanced
        if query_enhanced:
            confidence += self.deps.enhancement_weight * 0.1

        # Ensure confidence stays in valid range
        return min(max(confidence, 0.0), 1.0)

    def _get_fallback_agents(self, primary_agent: str) -> List[str]:
        """Get fallback agents based on primary recommendation"""
        if not self.deps.enable_fallback_routing:
            return []

        fallback_mapping = {
            "search_agent": ["summarizer_agent", "detailed_report_agent"],
            "summarizer_agent": ["detailed_report_agent", "search_agent"],
            "detailed_report_agent": ["summarizer_agent", "search_agent"],
        }

        return fallback_mapping.get(primary_agent, ["search_agent"])

    def _assess_orchestration_need(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        routing_result: Dict[str, Any],
        require_orchestration: Optional[bool],
    ) -> bool:
        """Assess if query requires multi-agent orchestration"""
        if require_orchestration is not None:
            return require_orchestration

        # Heuristics for detecting orchestration need
        orchestration_signals = 0

        # Check for multiple action verbs (find, analyze, summarize, compare, etc.)
        action_verbs = [
            "find",
            "search",
            "analyze",
            "summarize",
            "compare",
            "generate",
            "create",
            "extract",
            "identify",
        ]
        query_lower = query.lower()
        action_count = sum(1 for verb in action_verbs if verb in query_lower)
        if action_count >= 2:
            orchestration_signals += 1

        # Check for complex conjunctions (and, then, also, plus, etc.)
        conjunctions = ["and", "then", "also", "plus", "followed by", "as well as"]
        conjunction_count = sum(1 for conj in conjunctions if conj in query_lower)
        if conjunction_count >= 1:
            orchestration_signals += 1

        # Check query complexity (length, entity count, relationship count)
        if len(query.split()) > 15:  # Long queries often need orchestration
            orchestration_signals += 1

        if len(entities) > 5:  # Many entities suggest complex processing
            orchestration_signals += 1

        if (
            len(relationships) > 3
        ):  # Complex relationships suggest multi-step processing
            orchestration_signals += 1

        # Check for sequential indicators (first, then, finally, etc.)
        sequential_indicators = [
            "first",
            "then",
            "finally",
            "after",
            "before",
            "next",
            "subsequently",
        ]
        if any(indicator in query_lower for indicator in sequential_indicators):
            orchestration_signals += 1

        # Check routing confidence - low confidence might benefit from orchestration
        if routing_result.get("confidence", 0.5) < 0.6:
            orchestration_signals += 1

        # Orchestration needed if we have >= 3 signals
        return orchestration_signals >= 3

    def _get_orchestration_signals(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get detailed orchestration signals for debugging/analysis"""
        query_lower = query.lower()

        # Action verbs detection
        action_verbs = [
            "find",
            "search",
            "analyze",
            "summarize",
            "compare",
            "generate",
            "create",
            "extract",
            "identify",
        ]
        found_actions = [verb for verb in action_verbs if verb in query_lower]

        # Conjunctions detection
        conjunctions = ["and", "then", "also", "plus", "followed by", "as well as"]
        found_conjunctions = [conj for conj in conjunctions if conj in query_lower]

        # Sequential indicators
        sequential_indicators = [
            "first",
            "then",
            "finally",
            "after",
            "before",
            "next",
            "subsequently",
        ]
        found_sequential = [
            indicator for indicator in sequential_indicators if indicator in query_lower
        ]

        return {
            "query_length": len(query.split()),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "action_verbs": found_actions,
            "conjunctions": found_conjunctions,
            "sequential_indicators": found_sequential,
            "complexity_score": (
                len(query.split()) / 20  # Normalize query length
                + len(entities) / 10  # Normalize entity count
                + len(relationships) / 5  # Normalize relationship count
            ),
        }

    def _create_fallback_decision(self, query: str, error: str) -> RoutingOutput:
        """Create fallback routing decision when processing fails"""
        return RoutingOutput(
            query=query,
            recommended_agent="search_agent",  # Default fallback
            confidence=0.2,
            reasoning=f"Fallback routing due to processing error: {error}",
            fallback_agents=["summarizer_agent", "detailed_report_agent"],
            enhanced_query=query,
            entities=[],
            relationships=[],
            metadata={"error": error, "fallback": True},
        )

    def _update_routing_stats(self, decision: RoutingOutput) -> None:
        """Update routing statistics for telemetry"""
        if decision.confidence >= self.deps.confidence_threshold:
            self._routing_stats["successful_routes"] += 1

        self._routing_stats["confidence_scores"].append(decision.confidence)

        # Keep only last 1000 confidence scores for memory management
        if len(self._routing_stats["confidence_scores"]) > 1000:
            self._routing_stats["confidence_scores"] = self._routing_stats[
                "confidence_scores"
            ][-1000:]

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        stats = self._routing_stats.copy()

        if stats["confidence_scores"]:
            confidence_scores = stats["confidence_scores"]
            stats["average_confidence"] = sum(confidence_scores) / len(
                confidence_scores
            )
            stats["confidence_std"] = (
                sum((x - stats["average_confidence"]) ** 2 for x in confidence_scores)
                / len(confidence_scores)
            ) ** 0.5
        else:
            stats["average_confidence"] = 0.0
            stats["confidence_std"] = 0.0

        stats["success_rate"] = (
            stats["successful_routes"] / stats["total_queries"]
            if stats["total_queries"] > 0
            else 0.0
        )

        stats["enhancement_rate"] = (
            stats["enhanced_queries"] / stats["total_queries"]
            if stats["total_queries"] > 0
            else 0.0
        )

        return stats

    # ==========================================================================
    # Type-safe process implementation (required by AgentBase)
    # ==========================================================================

    async def _process_impl(self, input: RoutingInput) -> RoutingOutput:
        """
        Core processing logic for routing.

        The A2AAgent base class handles conversion from A2A protocol format
        to RoutingInput and from RoutingOutput back to A2A format.

        For streaming, use agent.process(input, stream=True) which calls
        _process_stream_impl() for custom streaming behavior.

        Args:
            input: Validated RoutingInput with query, context, require_orchestration

        Returns:
            RoutingOutput with agent recommendation, confidence, entities, relationships
        """
        # Delegate to route_query which does all the work
        return await self.route_query(
            query=input.query,
            context=input.context,
            tenant_id=input.tenant_id,
            require_orchestration=input.require_orchestration,
        )

    async def record_routing_outcome(
        self,
        decision: RoutingOutput,
        search_quality: float,
        agent_success: bool,
        processing_time: float = 0.0,
        user_satisfaction: Optional[float] = None,
        *,
        tenant_id: str,
    ) -> Optional[float]:
        """
        Record routing outcome for GRPO learning.

        Args:
            decision: The routing decision that was made
            search_quality: Quality of search results (0-1)
            agent_success: Whether the chosen agent completed successfully
            processing_time: Total processing time in seconds
            user_satisfaction: Optional explicit user feedback (0-1)
            tenant_id: Tenant identifier (REQUIRED per-request)

        Returns:
            Computed reward if GRPO is enabled, None otherwise
        """
        optimizer = self._get_optimizer(tenant_id)
        if not optimizer:
            return None

        try:
            reward = await optimizer.record_routing_experience(
                query=decision.query,
                entities=decision.entities,
                relationships=decision.relationships,
                enhanced_query=decision.enhanced_query,
                chosen_agent=decision.recommended_agent,
                routing_confidence=decision.confidence,
                search_quality=search_quality,
                agent_success=agent_success,
                processing_time=processing_time,
                user_satisfaction=user_satisfaction,
            )

            # Log to MLflow if enabled
            if self.mlflow_integration:
                try:
                    performance_metrics = {
                        "search_quality": search_quality,
                        "agent_success": 1.0 if agent_success else 0.0,
                        "processing_time": processing_time,
                        "routing_confidence": decision.confidence,
                        "grpo_reward": reward if reward is not None else 0.0,
                    }

                    if user_satisfaction is not None:
                        performance_metrics["user_satisfaction"] = user_satisfaction

                    # Use MLflow context if available
                    if (
                        hasattr(self.mlflow_integration, "current_run")
                        and self.mlflow_integration.current_run
                    ):
                        await self.mlflow_integration.log_routing_performance(
                            query=decision.query,
                            routing_decision=decision.metadata,
                            performance_metrics=performance_metrics,
                            step=self._routing_stats["total_queries"],
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to log to MLflow: {e}")

            self.logger.info(
                f"Recorded routing outcome: agent={decision.recommended_agent}, "
                f"success={agent_success}, reward={reward:.3f}"
            )

            return reward

        except Exception as e:
            self.logger.error(f"Failed to record routing outcome: {e}")
            return None

    async def analyze_and_route_with_relationships(
        self,
        query: str,
        *,
        tenant_id: str,
        enable_relationship_extraction: bool = True,
        enable_query_enhancement: bool = True,
        context: Optional[str] = None,
    ) -> RoutingOutput:
        """
        Analyze query and route with relationship extraction and enhancement

        This method is used by the orchestrator and other systems that need
        explicit control over the enhancement features.

        Args:
            query: User query to analyze and route
            enable_relationship_extraction: Whether to extract relationships
            enable_query_enhancement: Whether to enhance the query
            context: Optional context information
            tenant_id: Tenant identifier for multi-tenancy isolation

        Returns:
            Routing decision with relationship context
        """
        # Temporarily override config settings if needed
        original_rel_setting = self.deps.enable_relationship_extraction
        original_enh_setting = self.deps.enable_query_enhancement

        self.deps.enable_relationship_extraction = enable_relationship_extraction
        self.deps.enable_query_enhancement = enable_query_enhancement

        try:
            decision = await self.route_query(
                query=query, context=context, tenant_id=tenant_id
            )

            return decision

        finally:
            # Restore original settings
            self.deps.enable_relationship_extraction = original_rel_setting
            self.deps.enable_query_enhancement = original_enh_setting

    def get_grpo_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get GRPO optimization status and metrics for a tenant."""
        optimizer = self._get_optimizer(tenant_id)
        if not optimizer:
            return {"grpo_enabled": False, "reason": "GRPO optimizer not initialized"}

        try:
            status = optimizer.get_optimization_status()
            status["grpo_enabled"] = True
            return status

        except Exception as e:
            return {"grpo_enabled": True, "error": str(e), "status": "error"}

    async def reset_grpo_optimization(self, tenant_id: str) -> bool:
        """Reset GRPO optimization state for a tenant."""
        optimizer = self._get_optimizer(tenant_id)
        if not optimizer:
            return False

        try:
            await optimizer.reset_optimization()
            self.logger.info(f"GRPO optimization state reset for tenant: {tenant_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reset GRPO optimization: {e}")
            return False

    def get_mlflow_status(self) -> Dict[str, Any]:
        """Get MLflow tracking status and experiment information"""
        if not self.mlflow_integration:
            return {"mlflow_enabled": False, "reason": "MLflow tracking disabled"}

        try:
            return {
                "mlflow_enabled": True,
                "experiment_summary": self.mlflow_integration.get_experiment_summary(),
                "current_run_active": self.mlflow_integration.current_run is not None,
                "total_samples_logged": self.mlflow_integration.total_samples_logged,
                "tracking_uri": self.deps.mlflow_tracking_uri,
            }

        except Exception as e:
            return {"mlflow_enabled": True, "error": str(e), "status": "error"}

    def start_mlflow_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        """Start MLflow run context manager"""
        if not self.mlflow_integration:
            # Return dummy context manager
            from contextlib import nullcontext

            return nullcontext()

        additional_tags = {
            "agent_type": "routing_agent",
            "grpo_enabled": self.deps.enable_advanced_optimization,
            "simba_enabled": self.deps.enable_query_enhancement,
            "adaptive_thresholds_enabled": True,
            **(tags or {}),
        }

        return self.mlflow_integration.start_run(
            run_name=run_name, tags=additional_tags
        )

    async def log_optimization_metrics(self, tenant_id: str) -> None:
        """Log optimization metrics from GRPO, SIMBA, and adaptive thresholds."""
        if not self.mlflow_integration or not self.mlflow_integration.current_run:
            return

        try:
            # Get GRPO metrics for this tenant
            grpo_metrics = None
            optimizer = self._get_optimizer(tenant_id)
            if optimizer:
                grpo_metrics = optimizer.get_optimization_status()

            # Get SIMBA metrics (would need to access query enhancer)
            simba_metrics = None
            if hasattr(self, "query_enhancer") and hasattr(
                self.query_enhancer, "get_simba_status"
            ):
                simba_metrics = self.query_enhancer.get_simba_status()

            # Get threshold metrics (would need adaptive threshold learner)
            threshold_metrics = None
            # This would be implemented when adaptive threshold learner is integrated

            # Log to MLflow
            await self.mlflow_integration.log_optimization_metrics(
                grpo_metrics=grpo_metrics,
                simba_metrics=simba_metrics,
                threshold_metrics=threshold_metrics,
                step=self._routing_stats["total_queries"],
            )

        except Exception as e:
            self.logger.warning(f"Failed to log optimization metrics to MLflow: {e}")

    async def save_dspy_model(
        self, model_name: str = "enhanced_routing_model", description: str = ""
    ):
        """Save current DSPy routing module to MLflow model registry"""
        if not self.mlflow_integration or not self.routing_module:
            return None

        try:
            tags = {
                "model_type": "dspy_routing_module",
                "agent_type": "routing_agent",
                "features": "relationship_extraction,query_enhancement,grpo_optimization",
            }

            model_uri = self.mlflow_integration.save_dspy_model(
                model=self.routing_module,
                model_name=model_name,
                description=description
                or "Enhanced routing agent DSPy module with relationship extraction and optimization",
                tags=tags,
            )

            if model_uri:
                self.logger.info(f"DSPy model saved to MLflow: {model_uri}")

            return model_uri

        except Exception as e:
            self.logger.error(f"Failed to save DSPy model to MLflow: {e}")
            return None

    def cleanup_mlflow(self):
        """Cleanup MLflow integration"""
        if self.mlflow_integration:
            try:
                self.mlflow_integration.cleanup()
                self.logger.info("MLflow integration cleaned up")
            except Exception as e:
                self.logger.error(f"MLflow cleanup failed: {e}")


def create_routing_agent(
    tenant_id: str,
    telemetry_config: Any,
    port: int = 8001,
    **kwargs: Any,
) -> RoutingAgent:
    """
    Factory function to create Routing Agent with typed dependencies.

    Args:
        tenant_id: Tenant identifier (required)
        telemetry_config: Telemetry configuration (required)
        port: A2A server port
        **kwargs: Additional RoutingDeps fields (model_name, enable_caching, etc.)

    Returns:
        Configured RoutingAgent instance
    """
    deps = RoutingDeps(
        tenant_id=tenant_id,
        telemetry_config=telemetry_config,
        **kwargs,
    )
    return RoutingAgent(deps=deps, port=port)


def create_default_routing_deps(tenant_id: str, telemetry_config: Any) -> RoutingDeps:
    """Create default routing dependencies"""
    return RoutingDeps(tenant_id=tenant_id, telemetry_config=telemetry_config)


# Example usage and configuration
if __name__ == "__main__":
    import asyncio
    from dataclasses import dataclass

    @dataclass
    class MockTelemetryConfig:
        enabled: bool = False

    async def main():
        # Create typed dependencies with all settings
        deps = RoutingDeps(
            tenant_id="demo-tenant",
            telemetry_config=MockTelemetryConfig(enabled=False),
            llm_config=LLMEndpointConfig(
                model="ollama/smollm3:3b",
                api_base="http://localhost:11434",
                temperature=0.1,
                max_tokens=1000,
            ),
            confidence_threshold=0.7,
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
            enable_advanced_optimization=True,
            enable_mlflow_tracking=False,  # Disabled for demo
            mlflow_experiment_name="enhanced_routing_demo",
            mlflow_tracking_uri="http://localhost:5000",
        )

        # Create agent with typed deps
        agent = RoutingAgent(deps=deps, port=8001)

        # Test queries using typed input
        test_queries = [
            "Show me videos of robots playing soccer in tournaments",
            "Summarize the key findings from the latest AI research papers",
            "Generate a detailed report on renewable energy trends",
        ]

        for query in test_queries:
            print(f"\nProcessing: {query}")

            # Use typed input
            routing_input = RoutingInput(query=query)

            # Process returns typed output
            decision: RoutingOutput = await agent.process(routing_input)

            print(
                f"Route to: {decision.recommended_agent} "
                f"(confidence: {decision.confidence:.3f})"
            )
            print(f"Enhanced query: {decision.enhanced_query}")
            print(f"Entities: {len(decision.entities)}")
            print(f"Relationships: {len(decision.relationships)}")

            # Simulate routing outcome for GRPO learning
            await agent.record_routing_outcome(
                decision=decision,
                search_quality=0.8,
                agent_success=True,
                processing_time=0.15,
                user_satisfaction=0.85,
            )

        # Print statistics
        print("\nRouting Statistics:")
        stats = agent.get_routing_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        # Print MLflow status
        print("\nMLflow Status:")
        mlflow_status = agent.get_mlflow_status()
        for key, value in mlflow_status.items():
            if isinstance(value, dict):
                print(f"  {key}: {json.dumps(value, indent=4)}")
            else:
                print(f"  {key}: {value}")

        # Cleanup
        agent.cleanup_mlflow()

    asyncio.run(main())
