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
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# DSPy 3.0 imports
import dspy
from dspy import LM

# Phase 1-3 component imports
from src.app.agents.dspy_a2a_agent_base import DSPyA2AAgentBase

# Production features from RoutingAgent
from src.app.agents.memory_aware_mixin import MemoryAwareMixin

# Phase 6: Advanced optimization
from src.app.routing.advanced_optimizer import (
    AdvancedOptimizerConfig,
    AdvancedRoutingOptimizer,
)
from src.app.routing.contextual_analyzer import ContextualAnalyzer
from src.app.routing.cross_modal_optimizer import CrossModalOptimizer
from src.app.routing.dspy_relationship_router import DSPyAdvancedRoutingModule
from src.app.routing.dspy_routing_signatures import (
    BasicQueryAnalysisSignature,
)
from src.app.routing.lazy_executor import LazyModalityExecutor

# Phase 6.4: MLflow integration
from src.app.routing.mlflow_integration import ExperimentConfig, MLflowIntegration
from src.app.routing.modality_cache import ModalityCacheManager
from src.app.routing.parallel_executor import ParallelAgentExecutor
from src.app.routing.query_enhancement_engine import QueryEnhancementPipeline
from src.app.routing.relationship_extraction_tools import RelationshipExtractorTool
from src.app.search.multi_modal_reranker import MultiModalReranker
from src.app.telemetry.modality_metrics import ModalityMetricsTracker

# A2A protocol imports


@dataclass
class RoutingDecision:
    """Structured routing decision with confidence and reasoning"""

    query: str
    recommended_agent: str
    confidence: float
    reasoning: str
    fallback_agents: List[str] = field(default_factory=list)
    enhanced_query: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Legacy compatibility properties
    @property
    def extracted_entities(self) -> List[Dict[str, Any]]:
        return self.entities

    @property
    def extracted_relationships(self) -> List[Dict[str, Any]]:
        return self.relationships

    @property
    def routing_metadata(self) -> Dict[str, Any]:
        return self.metadata


@dataclass
class RoutingConfig:
    """Configuration for Enhanced Routing Agent"""

    # DSPy LM Configuration (defaults to local SmolLM 3B)
    model_name: str = "smollm3:3b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "dummy"  # Not needed for local Ollama

    # Routing thresholds
    confidence_threshold: float = 0.7
    relationship_weight: float = 0.3
    enhancement_weight: float = 0.4

    # Agent capabilities mapping
    agent_capabilities: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "video_search_agent": [
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
        }
    )

    # Enable/disable features
    enable_relationship_extraction: bool = True
    enable_query_enhancement: bool = True
    enable_fallback_routing: bool = True
    enable_confidence_calibration: bool = True
    enable_advanced_optimization: bool = True

    # Production features
    enable_caching: bool = True
    cache_size_per_modality: int = 1000
    cache_ttl_seconds: int = 300
    enable_parallel_execution: bool = True
    max_concurrent_agents: int = 5
    enable_memory: bool = False  # Disabled by default (requires Mem0)
    enable_contextual_analysis: bool = True
    enable_metrics_tracking: bool = True
    enable_multi_modal_reranking: bool = True
    enable_cross_modal_optimization: bool = True

    # Advanced optimization configuration
    optimizer_config: Optional[AdvancedOptimizerConfig] = None
    optimization_storage_dir: str = "data/optimization"

    # MLflow integration configuration
    enable_mlflow_tracking: bool = (
        False  # Disabled by default to avoid connection issues
    )
    mlflow_experiment_name: str = "routing_agent"
    mlflow_tracking_uri: str = "http://localhost:5000"


class RoutingAgent(DSPyA2AAgentBase, MemoryAwareMixin):
    """
    Routing Agent with complete production features and advanced optimization

    This agent combines all routing capabilities:
    - DSPy 3.0 with local SmolLM 3B for intelligent routing
    - Relationship extraction and query enhancement
    - Full production features (caching, parallel execution, memory)
    - Advanced optimization (GRPO, SIMBA)
    - Multi-modal reranking and cross-modal optimization
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        port: int = 8001,
        enable_telemetry: bool = True,
    ):
        self.config = config or RoutingConfig()
        self.logger = logging.getLogger(__name__)

        # Set telemetry flag BEFORE initializing production components
        self.enable_telemetry = enable_telemetry

        # Initialize DSPy 3.0 with local SmolLM
        self._configure_dspy()

        # Initialize Phase 2 & 3 components
        self._initialize_enhancement_pipeline()

        # Initialize DSPy routing module
        self._initialize_routing_module()

        # Initialize Phase 6: Advanced optimization
        self._initialize_advanced_optimizer()

        # Initialize Phase 6.4: MLflow integration
        self._initialize_mlflow_tracking()

        # Initialize production features
        self._initialize_production_components()

        # Initialize A2A base with DSPy module
        super().__init__(
            agent_name="routing_agent",
            agent_description="Intelligent routing with relationship extraction and query enhancement",
            dspy_module=self.routing_module,
            capabilities=self._get_routing_capabilities(),
            port=port,
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
        has_relationship_extraction = (
            self.config.enable_relationship_extraction
            and self.relationship_extractor is not None
        )

        has_query_enhancement = (
            self.config.enable_query_enhancement and self.query_enhancer is not None
        )

        has_routing_module = self.routing_module is not None

        return (
            has_relationship_extraction and has_query_enhancement and has_routing_module
        )

    def _configure_dspy(self) -> None:
        """Configure DSPy 3.0 with local SmolLM via Ollama"""
        try:
            # Configure DSPy to use local SmolLM 3B via Ollama
            lm = LM(
                model=self.config.model_name,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                # Additional LM configurations
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent routing decisions
                top_p=0.9,
            )

            dspy.settings.configure(lm=lm)
            self.logger.info(
                f"DSPy configured with {self.config.model_name} at {self.config.base_url}"
            )

        except Exception as e:
            self.logger.error(f"Failed to configure DSPy with local SmolLM: {e}")
            # Fallback to mock for development/testing
            self.logger.warning("Falling back to mock LM for development")

    def _initialize_enhancement_pipeline(self) -> None:
        """Initialize Phase 2 & 3 components"""
        try:
            # Phase 2: Relationship extraction
            if self.config.enable_relationship_extraction:
                self.relationship_extractor = RelationshipExtractorTool()
                self.logger.info("Relationship extraction tool initialized")

            # Phase 3: Query enhancement
            if self.config.enable_query_enhancement:
                self.query_enhancer = QueryEnhancementPipeline()
                self.logger.info("Query enhancement pipeline initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize enhancement pipeline: {e}")
            # Set None values for graceful degradation
            self.relationship_extractor = None
            self.query_enhancer = None

    def _initialize_routing_module(self) -> None:
        """Initialize DSPy routing module"""
        try:
            self.routing_module = DSPyAdvancedRoutingModule()
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
                        recommended_agent="video_search_agent",
                        confidence=0.5,
                    )

        self.routing_module = FallbackRoutingModule()
        self.logger.warning("Using fallback routing module")

    def _initialize_advanced_optimizer(self) -> None:
        """Initialize Phase 6: Advanced optimization"""
        try:
            if self.config.enable_advanced_optimization:
                optimizer_config = (
                    self.config.optimizer_config or AdvancedOptimizerConfig()
                )
                self.grpo_optimizer = AdvancedRoutingOptimizer(
                    config=optimizer_config,
                    storage_dir=self.config.optimization_storage_dir,
                )
                self.logger.info("Advanced routing optimizer initialized")
            else:
                self.grpo_optimizer = None
                self.logger.info("Advanced optimization disabled")

        except Exception as e:
            self.logger.error(f"Failed to initialize advanced optimizer: {e}")
            self.grpo_optimizer = None

    def _initialize_mlflow_tracking(self) -> None:
        """Initialize Phase 6.4: MLflow tracking integration"""
        try:
            if self.config.enable_mlflow_tracking:
                mlflow_config = ExperimentConfig(
                    experiment_name=self.config.mlflow_experiment_name,
                    tracking_uri=self.config.mlflow_tracking_uri,
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

                self.mlflow_integration = MLflowIntegration(
                    config=mlflow_config,
                    storage_dir=f"{self.config.optimization_storage_dir}/mlflow",
                )
                self.logger.info("MLflow tracking integration initialized")
            else:
                self.mlflow_integration = None
                self.logger.info("MLflow tracking disabled")

        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow tracking: {e}")
            self.mlflow_integration = None

    def _initialize_production_components(self) -> None:
        """Initialize production-ready components from RoutingAgent"""
        try:
            # Initialize telemetry manager
            if self.enable_telemetry:
                from src.app.telemetry.config import TelemetryConfig
                from src.app.telemetry.manager import TelemetryManager
                telemetry_config = TelemetryConfig()
                self.telemetry_manager = TelemetryManager(config=telemetry_config)
                self.logger.info("📊 Telemetry manager initialized")
            else:
                self.telemetry_manager = None

            # Initialize caching
            if self.config.enable_caching:
                self.cache_manager = ModalityCacheManager(
                    cache_size_per_modality=self.config.cache_size_per_modality
                )
                self.logger.info("💾 Cache manager initialized")
            else:
                self.cache_manager = None

            # Initialize parallel execution
            if self.config.enable_parallel_execution:
                self.parallel_executor = ParallelAgentExecutor(
                    max_concurrent_agents=self.config.max_concurrent_agents
                )
                self.logger.info("⚡ Parallel executor initialized")
            else:
                self.parallel_executor = None

            # Initialize memory system (if enabled)
            if self.config.enable_memory:
                try:
                    # MemoryAwareMixin provides memory initialization
                    self.initialize_memory(
                        user_id="routing_agent",
                        session_id="default"
                    )
                    self.logger.info("🧠 Memory system initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize memory: {e}")

            # Initialize contextual analysis
            if self.config.enable_contextual_analysis:
                self.contextual_analyzer = ContextualAnalyzer(
                    max_history_size=50,
                    context_window_minutes=30,
                    min_preference_count=3
                )
                self.logger.info("📊 Contextual analyzer initialized")
            else:
                self.contextual_analyzer = None

            # Initialize metrics tracking
            if self.config.enable_metrics_tracking:
                self.metrics_tracker = ModalityMetricsTracker(window_size=1000)
                self.logger.info("📈 Metrics tracker initialized")
            else:
                self.metrics_tracker = None

            # Initialize multi-modal reranking
            if self.config.enable_multi_modal_reranking:
                self.multi_modal_reranker = MultiModalReranker()
                self.logger.info("🎯 Multi-modal reranker initialized")
            else:
                self.multi_modal_reranker = None

            # Initialize cross-modal optimization
            if self.config.enable_cross_modal_optimization:
                self.cross_modal_optimizer = CrossModalOptimizer()
                self.logger.info("🔄 Cross-modal optimizer initialized")
            else:
                self.cross_modal_optimizer = None

            # Initialize lazy executor (always on for efficiency)
            self.lazy_executor = LazyModalityExecutor()
            self.logger.info("💤 Lazy executor initialized")

            self.logger.info("✅ Production components initialized successfully")

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

    def _get_routing_capabilities(self) -> List[str]:
        """Get routing agent capabilities"""
        capabilities = ["intelligent_routing", "query_analysis", "agent_orchestration"]

        if self.config.enable_relationship_extraction:
            capabilities.extend(
                ["relationship_extraction", "entity_recognition", "semantic_analysis"]
            )

        if self.config.enable_query_enhancement:
            capabilities.extend(
                ["query_enhancement", "query_rewriting", "context_enrichment"]
            )

        if self.config.enable_advanced_optimization:
            capabilities.extend(
                [
                    "advanced_optimization",
                    "adaptive_learning",
                    "performance_optimization",
                ]
            )

        if self.config.enable_mlflow_tracking:
            capabilities.extend(
                ["mlflow_tracking", "experiment_management", "performance_monitoring"]
            )

        # Production capabilities
        if self.config.enable_caching:
            capabilities.append("result_caching")
        if self.config.enable_parallel_execution:
            capabilities.append("parallel_agent_execution")
        if self.config.enable_memory:
            capabilities.append("conversation_memory")
        if self.config.enable_contextual_analysis:
            capabilities.append("contextual_analysis")
        if self.config.enable_metrics_tracking:
            capabilities.append("performance_metrics")
        if self.config.enable_multi_modal_reranking:
            capabilities.append("multi_modal_reranking")
        if self.config.enable_cross_modal_optimization:
            capabilities.append("cross_modal_optimization")

        return capabilities

    async def route_query(
        self,
        query: str,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
        require_orchestration: Optional[bool] = None,
    ) -> RoutingDecision:
        """
        Enhanced query routing with relationship extraction and query enhancement

        This is the main routing method that combines all Phase 1-3 components
        """
        self._routing_stats["total_queries"] += 1
        start_time = datetime.now()

        # Create telemetry span context manager if available
        span_context = None
        if hasattr(self, 'telemetry_manager') and self.telemetry_manager:
            from src.app.telemetry.config import SERVICE_NAME_ORCHESTRATION
            self.logger.info(f"Creating telemetry span for user_id: {user_id or 'unknown'}")
            span_context = self.telemetry_manager.span(
                "cogniverse.routing",
                tenant_id=user_id or "unknown",
                service_name=SERVICE_NAME_ORCHESTRATION  # Use orchestration service for routing spans
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
                self.logger.warning("Span context returned None - no telemetry span created")
            try:
                # Check cache first (if enabled)
                if self.cache_manager:
                    from src.app.search.multi_modal_reranker import QueryModality
                    cached_decision = self.cache_manager.get_cached_result(
                        query=query,
                        modality=QueryModality.TEXT,  # Use TEXT as default for routing
                        ttl_seconds=self.config.cache_ttl_seconds
                    )
                    if cached_decision:
                        self.logger.info(f"Cache hit for query: {query[:50]}...")
                        return cached_decision

                # Add contextual analysis (if enabled)
                if self.contextual_analyzer and user_id:
                    contextual_insights = self.contextual_analyzer.get_contextual_hints(
                        current_query=query
                    )
                    self.logger.info(f"Contextual insights: {contextual_insights}")

                # Phase 2: Extract relationships and entities
                entities, relationships = await self._extract_relationships(query)

                # Phase 3: Enhance query with relationship context
                enhanced_query, enhancement_metadata = await self._enhance_query(
                    query, entities, relationships
                )

                # Phase 1: DSPy-powered routing decision (baseline)
                baseline_routing_result = await self._make_routing_decision(
                    original_query=query,
                    enhanced_query=enhanced_query,
                    entities=entities,
                    relationships=relationships,
                    context=context,
                )

                # Phase 6: Apply GRPO optimization if available
                optimized_routing_result = await self._apply_grpo_optimization(
                    query=query,
                    entities=entities,
                    relationships=relationships,
                    enhanced_query=enhanced_query,
                    baseline_prediction=baseline_routing_result,
                )

                # Use optimized result if available, otherwise baseline
                final_routing_result = optimized_routing_result or baseline_routing_result

                # Determine if orchestration is needed
                needs_orchestration = self._assess_orchestration_need(
                    query,
                    entities,
                    relationships,
                    final_routing_result,
                    require_orchestration,
                )

                # Create structured routing decision
                decision = RoutingDecision(
                    query=query,
                    recommended_agent=final_routing_result.get(
                        "recommended_agent", "video_search_agent"
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
                        "processing_time_ms": (datetime.now() - start_time).total_seconds()
                        * 1000,
                        "baseline_routing_result": baseline_routing_result,
                        "optimized_routing_result": optimized_routing_result,
                        "grpo_applied": optimized_routing_result is not None,
                        "user_id": user_id,
                        "needs_orchestration": needs_orchestration,
                        "orchestration_signals": self._get_orchestration_signals(
                            query, entities, relationships
                        ),
                    },
                )

                # Update statistics
                self._update_routing_stats(decision)

                # Cache the decision (if enabled)
                if self.cache_manager:
                    from src.app.search.multi_modal_reranker import QueryModality
                    self.cache_manager.cache_result(
                        query=query,
                        modality=QueryModality.TEXT,
                        result=decision
                    )

                # Track metrics (if enabled)
                if self.metrics_tracker:
                    from src.app.search.multi_modal_reranker import QueryModality
                    self.metrics_tracker.record_modality_execution(
                        modality=QueryModality.TEXT,
                        latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        success=True
                    )

                # Update contextual analyzer (if enabled)
                if self.contextual_analyzer and user_id:
                    self.contextual_analyzer.update_context(
                        query=query,
                        detected_modalities=[decision.recommended_agent],
                        result=decision,
                        result_count=1 if decision.confidence > 0.5 else 0
                    )

                # Log successful routing
                self.logger.info(
                    f"Query routed to {decision.recommended_agent} "
                    f"(confidence: {decision.confidence:.3f}, "
                    f"relationships: {len(relationships)}, "
                    f"enhanced: {'yes' if enhanced_query != query else 'no'})"
                )

                # Update telemetry span with final attributes
                if span and hasattr(span, 'set_attribute'):
                    self.logger.info("Setting telemetry span attributes")
                    span.set_attribute("routing.query", query)
                    span.set_attribute("routing.chosen_agent", decision.recommended_agent)
                    span.set_attribute("routing.confidence", decision.confidence)
                    span.set_attribute("routing.processing_time", (datetime.now() - start_time).total_seconds() * 1000)
                    span.set_attribute("routing.reasoning", decision.reasoning)
                    span.set_attribute("routing.entities_count", len(entities))
                    span.set_attribute("routing.relationships_count", len(relationships))
                    span.set_attribute("routing.enhanced", enhanced_query != query)
                    self.logger.info(f"Telemetry span attributes set for query: {query[:50]}...")
                else:
                    self.logger.warning(f"Cannot set span attributes - span={span}, has_set_attribute={hasattr(span, 'set_attribute') if span else False}")

                return decision

            except Exception as e:
                self.logger.error(f"Routing failed for query '{query}': {e}")
                return self._create_fallback_decision(query, str(e))

    async def _extract_relationships(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Phase 2: Extract relationships and entities from query"""
        if (
            not self.config.enable_relationship_extraction
            or not self.relationship_extractor
        ):
            return [], []

        try:
            extraction_result = (
                await self.relationship_extractor.extract_comprehensive_relationships(
                    query
                )
            )

            entities = extraction_result.get("entities", [])
            relationships = extraction_result.get("relationships", [])

            self._routing_stats["relationship_extractions"] += 1

            self.logger.debug(
                f"Extracted {len(entities)} entities and {len(relationships)} relationships"
            )

            return entities, relationships

        except Exception as e:
            self.logger.warning(f"Relationship extraction failed: {e}")
            return [], []

    async def _enhance_query(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Phase 3: Enhance query with relationship context"""
        if not self.config.enable_query_enhancement or not self.query_enhancer:
            return query, {}

        try:
            enhancement_result = (
                await self.query_enhancer.enhance_query_with_relationships(
                    query, entities, relationships
                )
            )

            enhanced_query = enhancement_result.get("enhanced_query", query)
            enhancement_metadata = {
                "quality_score": enhancement_result.get("quality_score", 0.5),
                "enhancement_strategy": enhancement_result.get(
                    "enhancement_strategy", "none"
                ),
                "semantic_expansions": enhancement_result.get(
                    "semantic_expansions", []
                ),
            }

            if enhanced_query != query:
                self._routing_stats["enhanced_queries"] += 1

            self.logger.debug(f"Query enhanced: '{query}' -> '{enhanced_query}'")

            return enhanced_query, enhancement_metadata

        except Exception as e:
            self.logger.warning(f"Query enhancement failed: {e}")
            return query, {}

    async def _apply_grpo_optimization(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhanced_query: str,
        baseline_prediction: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Phase 6: Apply GRPO optimization to routing decision"""
        if not self.grpo_optimizer:
            return None

        try:
            optimized_result = await self.grpo_optimizer.optimize_routing_decision(
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
        """Phase 1: Make DSPy-powered routing decision"""
        try:
            # Use enhanced query for routing if available, fallback to original
            routing_query = (
                enhanced_query if enhanced_query != original_query else original_query
            )

            # Prepare context with relationship information
            routing_context = self._prepare_routing_context(
                context, entities, relationships
            )

            # DSPy routing decision
            dspy_result = self.routing_module.forward(
                query=routing_query, context=routing_context
            )

            # Extract routing information from DSPy result
            routing_info = {
                "recommended_agent": getattr(
                    dspy_result, "recommended_agent", "video_search_agent"
                ),
                "confidence": getattr(dspy_result, "confidence", 0.5),
                "reasoning": getattr(dspy_result, "reasoning", "DSPy routing decision"),
                "primary_intent": getattr(dspy_result, "primary_intent", "search"),
                "complexity_score": getattr(dspy_result, "complexity_score", 0.5),
            }

            # Apply confidence calibration if enabled
            if self.config.enable_confidence_calibration:
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
                "recommended_agent": "video_search_agent",
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
            confidence += self.config.relationship_weight * min(
                relationship_count / 3, 0.15
            )

        # Boost confidence if query was enhanced
        if query_enhanced:
            confidence += self.config.enhancement_weight * 0.1

        # Ensure confidence stays in valid range
        return min(max(confidence, 0.0), 1.0)

    def _get_fallback_agents(self, primary_agent: str) -> List[str]:
        """Get fallback agents based on primary recommendation"""
        if not self.config.enable_fallback_routing:
            return []

        fallback_mapping = {
            "video_search_agent": ["summarizer_agent", "detailed_report_agent"],
            "summarizer_agent": ["detailed_report_agent", "video_search_agent"],
            "detailed_report_agent": ["summarizer_agent", "video_search_agent"],
        }

        return fallback_mapping.get(primary_agent, ["video_search_agent"])

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

    def _create_fallback_decision(self, query: str, error: str) -> RoutingDecision:
        """Create fallback routing decision when processing fails"""
        return RoutingDecision(
            query=query,
            recommended_agent="video_search_agent",  # Default fallback
            confidence=0.2,
            reasoning=f"Fallback routing due to processing error: {error}",
            fallback_agents=["summarizer_agent", "detailed_report_agent"],
            enhanced_query=query,
            entities=[],
            relationships=[],
            metadata={"error": error, "fallback": True},
        )

    def _update_routing_stats(self, decision: RoutingDecision) -> None:
        """Update routing statistics for telemetry"""
        if decision.confidence >= self.config.confidence_threshold:
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

    # DSPyA2AAgentBase implementation
    async def _process_with_dspy(self, dspy_input: Dict[str, Any]) -> Any:
        """Process A2A input with DSPy routing logic"""
        query = dspy_input.get("query", "")
        context = dspy_input.get("context")
        user_id = dspy_input.get("user_id")

        routing_decision = await self.route_query(query, context, user_id)

        return {
            "agent": routing_decision.recommended_agent,
            "confidence": routing_decision.confidence,
            "reasoning": routing_decision.reasoning,
            "enhanced_query": routing_decision.enhanced_query,
            "fallback_agents": routing_decision.fallback_agents,
            "entities": routing_decision.extracted_entities,
            "relationships": routing_decision.extracted_relationships,
            "metadata": routing_decision.routing_metadata,
        }

    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """Convert DSPy routing output to A2A format"""
        if isinstance(dspy_output, dict):
            return {
                "status": "success",
                "routing_decision": dspy_output,
                "agent": self.agent_name,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "status": "success",
                "routing_decision": str(dspy_output),
                "agent": self.agent_name,
                "timestamp": datetime.now().isoformat(),
            }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define enhanced routing agent skills for A2A protocol"""
        base_skills = [
            {
                "name": "route_query",
                "description": "Route queries to appropriate agents with relationship-aware intelligence",
                "input_schema": {
                    "query": "string",
                    "context": "string (optional)",
                    "user_id": "string (optional)",
                },
                "output_schema": {
                    "recommended_agent": "string",
                    "confidence": "number",
                    "reasoning": "string",
                    "enhanced_query": "string",
                    "entities": "array",
                    "relationships": "array",
                },
            }
        ]

        if self.config.enable_relationship_extraction:
            base_skills.extend(
                [
                    {
                        "name": "extract_relationships",
                        "description": "Extract entities and relationships from text",
                        "input_schema": {"query": "string"},
                        "output_schema": {
                            "entities": "array",
                            "relationships": "array",
                        },
                    }
                ]
            )

        if self.config.enable_query_enhancement:
            base_skills.append(
                {
                    "name": "enhance_query",
                    "description": "Enhance queries with relationship context",
                    "input_schema": {
                        "query": "string",
                        "entities": "array",
                        "relationships": "array",
                    },
                    "output_schema": {"enhanced_query": "string", "metadata": "object"},
                }
            )

        return base_skills

    async def record_routing_outcome(
        self,
        decision: RoutingDecision,
        search_quality: float,
        agent_success: bool,
        processing_time: float = 0.0,
        user_satisfaction: Optional[float] = None,
    ) -> Optional[float]:
        """
        Record routing outcome for GRPO learning

        Args:
            decision: The routing decision that was made
            search_quality: Quality of search results (0-1)
            agent_success: Whether the chosen agent completed successfully
            processing_time: Total processing time in seconds
            user_satisfaction: Optional explicit user feedback (0-1)

        Returns:
            Computed reward if GRPO is enabled, None otherwise
        """
        if not self.grpo_optimizer:
            return None

        try:
            reward = await self.grpo_optimizer.record_routing_experience(
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
        enable_relationship_extraction: bool = True,
        enable_query_enhancement: bool = True,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Analyze query and route with relationship extraction and enhancement

        This method is used by the orchestrator and other systems that need
        explicit control over the enhancement features.

        Args:
            query: User query to analyze and route
            enable_relationship_extraction: Whether to extract relationships
            enable_query_enhancement: Whether to enhance the query
            context: Optional context information
            user_id: Optional user identifier

        Returns:
            Routing decision with relationship context
        """
        # Temporarily override config settings if needed
        original_rel_setting = self.config.enable_relationship_extraction
        original_enh_setting = self.config.enable_query_enhancement

        self.config.enable_relationship_extraction = enable_relationship_extraction
        self.config.enable_query_enhancement = enable_query_enhancement

        try:
            decision = await self.route_query(
                query=query, context=context, user_id=user_id
            )

            return decision

        finally:
            # Restore original settings
            self.config.enable_relationship_extraction = original_rel_setting
            self.config.enable_query_enhancement = original_enh_setting

    def get_grpo_status(self) -> Dict[str, Any]:
        """Get GRPO optimization status and metrics"""
        if not self.grpo_optimizer:
            return {"grpo_enabled": False, "reason": "GRPO optimizer not initialized"}

        try:
            status = self.grpo_optimizer.get_optimization_status()
            status["grpo_enabled"] = True
            return status

        except Exception as e:
            return {"grpo_enabled": True, "error": str(e), "status": "error"}

    async def reset_grpo_optimization(self) -> bool:
        """Reset GRPO optimization state (useful for testing)"""
        if not self.grpo_optimizer:
            return False

        try:
            await self.grpo_optimizer.reset_optimization()
            self.logger.info("GRPO optimization state reset")
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
                "tracking_uri": self.config.mlflow_tracking_uri,
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
            "grpo_enabled": self.config.enable_grpo_optimization,
            "simba_enabled": self.config.enable_query_enhancement,
            "adaptive_thresholds_enabled": True,
            **(tags or {}),
        }

        return self.mlflow_integration.start_run(
            run_name=run_name, tags=additional_tags
        )

    async def log_optimization_metrics(self):
        """Log optimization metrics from GRPO, SIMBA, and adaptive thresholds"""
        if not self.mlflow_integration or not self.mlflow_integration.current_run:
            return

        try:
            # Get GRPO metrics
            grpo_metrics = None
            if self.grpo_optimizer:
                grpo_metrics = self.grpo_optimizer.get_optimization_status()

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
    config: Optional[RoutingConfig] = None, port: int = 8001
) -> RoutingAgent:
    """Factory function to create Routing Agent"""
    return RoutingAgent(config=config, port=port)


def create_default_routing_config() -> RoutingConfig:
    """Create default routing configuration"""
    return RoutingConfig()


# Example usage and configuration
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create enhanced routing agent with local SmolLM 3B and MLflow tracking
        config = RoutingConfig(
            model_name="smollm3:3b",
            base_url="http://localhost:11434/v1",
            confidence_threshold=0.7,
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
            enable_grpo_optimization=True,
            enable_mlflow_tracking=True,
            mlflow_experiment_name="enhanced_routing_demo",
            mlflow_tracking_uri="http://localhost:5000",
        )

        agent = create_routing_agent(config, port=8001)

        # Start MLflow run for this demonstration
        with agent.start_mlflow_run(run_name="routing_demo", tags={"demo": "true"}):
            # Test queries
            test_queries = [
                "Show me videos of robots playing soccer in tournaments",
                "Summarize the key findings from the latest AI research papers",
                "Generate a detailed report on renewable energy trends",
            ]

            for query in test_queries:
                print(f"\nProcessing: {query}")
                decision = await agent.route_query(query)
                print(
                    f"Route to: {decision.recommended_agent} (confidence: {decision.confidence:.3f})"
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

            # Log optimization metrics
            await agent.log_optimization_metrics()

            # Save model to registry
            await agent.save_dspy_model("demo_routing_model", "Demo model from example")

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
