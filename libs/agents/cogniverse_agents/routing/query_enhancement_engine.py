"""
Query Enhancement Engine for DSPy 3.0 Routing System

This module implements query enhancement using the ComposableQueryAnalysisModule
which combines entity extraction, relationship inference, and LLM-powered query
reformulation into a single optimizable pipeline.

Phase 6.2: Enhanced with SIMBA (Similarity-Based Memory Augmentation) for
learning-based query enhancement using patterns from successful transformations.
"""

import logging
from typing import Any, Dict, List, Optional

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

from .dspy_relationship_router import ComposableQueryAnalysisModule
from .simba_query_enhancer import SIMBAConfig, SIMBAQueryEnhancer

logger = logging.getLogger(__name__)


class QueryEnhancementPipeline:
    """
    Complete query enhancement pipeline that uses ComposableQueryAnalysisModule
    for entity extraction, relationship inference, and query reformulation.

    Phase 6.2: Enhanced with SIMBA for learning-based query enhancement.
    """

    def __init__(
        self,
        analysis_module: Optional[ComposableQueryAnalysisModule] = None,
        enable_simba: bool = True,
        simba_config: Optional[SIMBAConfig] = None,
        query_fusion_config: Optional[Dict[str, Any]] = None,
        telemetry_provider: Optional[TelemetryProvider] = None,
        tenant_id: str = "default",
    ):
        """Initialize the enhancement pipeline.

        Args:
            analysis_module: ComposableQueryAnalysisModule instance. If None,
                a default one will be created when first needed.
            enable_simba: Whether to enable SIMBA learning-based enhancement.
            simba_config: SIMBA configuration.
            query_fusion_config: Config with 'include_original' and 'rrf_k' keys.
            telemetry_provider: Telemetry provider for SIMBA artifact persistence.
            tenant_id: Tenant identifier for SIMBA artifact storage.
        """
        self.analysis_module = analysis_module
        self.query_fusion_config = query_fusion_config or {
            "include_original": True,
            "rrf_k": 60,
        }

        # Phase 6.2: Initialize SIMBA enhancer
        self.enable_simba = enable_simba
        if enable_simba:
            if telemetry_provider is None:
                raise ValueError(
                    "telemetry_provider is required when enable_simba=True"
                )
            self.simba_enhancer = SIMBAQueryEnhancer(
                config=simba_config or SIMBAConfig(),
                telemetry_provider=telemetry_provider,
                tenant_id=tenant_id,
            )
            logger.info("Query enhancement pipeline initialized with SIMBA")
        else:
            self.simba_enhancer = None
            logger.info("Query enhancement pipeline initialized without SIMBA")

    def _ensure_analysis_module(self) -> ComposableQueryAnalysisModule:
        """Lazily create the analysis module if not provided."""
        if self.analysis_module is None:
            from .dspy_relationship_router import (
                create_composable_query_analysis_module,
            )

            self.analysis_module = create_composable_query_analysis_module()
        return self.analysis_module

    async def enhance_query_with_relationships(
        self,
        query: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
        search_context: str = "general",
        entity_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end query enhancement with relationship extraction and SIMBA learning.

        Args:
            query: Original query to enhance
            entities: Pre-extracted entities (ignored — module extracts its own)
            relationships: Pre-extracted relationships (ignored — module extracts its own)
            search_context: Target search system context
            entity_labels: Optional entity labels for extraction (unused, kept for API compat)

        Returns:
            Complete enhancement results
        """
        try:
            # Step 1: Try SIMBA enhancement first (Phase 6.2)
            simba_result = None
            if self.enable_simba and self.simba_enhancer:
                # SIMBA needs entities/relationships — run analysis module first
                module = self._ensure_analysis_module()
                analysis = module.forward(query, search_context)

                simba_result = await self.simba_enhancer.enhance_query_with_patterns(
                    original_query=query,
                    entities=analysis.entities,
                    relationships=analysis.relationships,
                    context=search_context,
                )

                if simba_result and simba_result.get("enhanced", False):
                    # SIMBA succeeded — use its enhancement but keep module's
                    # entities, relationships, and variants
                    enhanced_query = simba_result["enhanced_query"]
                    enhancement_strategy = simba_result["enhancement_strategy"]
                    quality_score = simba_result["confidence"]

                    # Use analysis module's variants (SIMBA doesn't generate variants)
                    query_variants = analysis.query_variants
                    include_original = self.query_fusion_config.get(
                        "include_original", True
                    )
                    if include_original and query_variants:
                        # Ensure original query is included as first variant
                        has_original = any(
                            v.get("name") == "original" for v in query_variants
                        )
                        if not has_original:
                            query_variants = [
                                {"name": "original", "query": query}
                            ] + query_variants

                    logger.info(
                        f"SIMBA enhanced query '{query[:50]}...' -> "
                        f"'{enhanced_query[:50]}...' "
                        f"(patterns: {simba_result.get('similar_patterns_used', 0)})"
                    )

                    return self._build_result(
                        query=query,
                        entities=analysis.entities,
                        relationships=analysis.relationships,
                        enhanced_query=enhanced_query,
                        enhancement_strategy=enhancement_strategy,
                        quality_score=quality_score,
                        query_variants=query_variants,
                        search_context=search_context,
                        enhancement_method="simba",
                        simba_result=simba_result,
                        path_used=analysis.path_used,
                    )

            # Step 2: Fall back to composable module
            module = self._ensure_analysis_module()
            analysis = module.forward(query, search_context)

            query_variants = analysis.query_variants
            include_original = self.query_fusion_config.get("include_original", True)
            if include_original and query_variants:
                has_original = any(v.get("name") == "original" for v in query_variants)
                if not has_original:
                    query_variants = [
                        {"name": "original", "query": query}
                    ] + query_variants

            logger.info(
                f"Composable module enhanced query '{query[:50]}...' -> "
                f"'{analysis.enhanced_query[:50]}...' "
                f"(confidence: {analysis.confidence}, path: {analysis.path_used})"
            )

            return self._build_result(
                query=query,
                entities=analysis.entities,
                relationships=analysis.relationships,
                enhanced_query=analysis.enhanced_query,
                enhancement_strategy=f"composable_{analysis.path_used}",
                quality_score=analysis.confidence,
                query_variants=query_variants,
                search_context=search_context,
                enhancement_method="composable_module",
                simba_result=None,
                path_used=analysis.path_used,
            )

        except Exception as e:
            logger.error(f"Query enhancement pipeline failed: {e}")

            # Return minimal fallback result
            return {
                "original_query": query,
                "extracted_entities": [],
                "extracted_relationships": [],
                "relationship_types": [],
                "semantic_connections": [],
                "enhanced_query": query,
                "semantic_expansions": [],
                "relationship_phrases": [],
                "enhancement_strategy": "pipeline_error",
                "search_operators": [],
                "quality_score": 0.0,
                "query_variants": [],
                "search_context": search_context,
                "processing_metadata": {"error": str(e), "fallback_used": True},
            }

    def _build_result(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhanced_query: str,
        enhancement_strategy: str,
        quality_score: float,
        query_variants: List[Dict[str, str]],
        search_context: str,
        enhancement_method: str,
        simba_result: Optional[Dict[str, Any]],
        path_used: str,
    ) -> Dict[str, Any]:
        """Build the standard result dictionary."""
        relationship_types = list(
            set(r.get("relation", "") for r in relationships if r.get("relation"))
        )
        semantic_connections = [
            f"{r['subject']} {r['relation'].replace('_', ' ')} {r['object']}"
            for r in relationships[:5]
            if all(k in r for k in ("subject", "relation", "object"))
        ]

        return {
            # Original extraction data
            "original_query": query,
            "extracted_entities": entities,
            "extracted_relationships": relationships,
            "relationship_types": relationship_types,
            "semantic_connections": semantic_connections,
            # Enhancement results
            "enhanced_query": enhanced_query,
            "semantic_expansions": [],
            "relationship_phrases": [],
            "enhancement_strategy": enhancement_strategy,
            "search_operators": [],
            "quality_score": quality_score,
            # Query fusion variants
            "query_variants": query_variants,
            # SIMBA metadata
            "simba_applied": simba_result is not None
            and simba_result.get("enhanced", False),
            "simba_patterns_used": (
                simba_result.get("similar_patterns_used", 0) if simba_result else 0
            ),
            "pattern_avg_improvement": (
                simba_result.get("pattern_avg_improvement", 0.0)
                if simba_result
                else 0.0
            ),
            # Metadata
            "search_context": search_context,
            "processing_metadata": {
                "entities_found": len(entities),
                "relationships_found": len(relationships),
                "enhancement_quality": quality_score,
                "query_complexity": [],
                "enhancement_method": enhancement_method,
                "analysis_path": path_used,
            },
        }

    async def record_enhancement_outcome(
        self,
        original_query: str,
        enhanced_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhancement_strategy: str,
        search_quality_improvement: float,
        routing_confidence_improvement: float,
        user_satisfaction: Optional[float] = None,
    ) -> None:
        """Record outcome of query enhancement for SIMBA learning."""
        if not self.enable_simba or not self.simba_enhancer:
            return

        try:
            await self.simba_enhancer.record_enhancement_outcome(
                original_query=original_query,
                enhanced_query=enhanced_query,
                entities=entities,
                relationships=relationships,
                enhancement_strategy=enhancement_strategy,
                search_quality_improvement=search_quality_improvement,
                routing_confidence_improvement=routing_confidence_improvement,
                user_satisfaction=user_satisfaction,
            )

            logger.debug(
                f"Recorded enhancement outcome for SIMBA learning: improvement={search_quality_improvement:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to record enhancement outcome: {e}")

    def get_simba_status(self) -> Dict[str, Any]:
        """Get SIMBA enhancement status and metrics."""
        if not self.enable_simba or not self.simba_enhancer:
            return {
                "simba_enabled": False,
                "reason": "SIMBA disabled or not initialized",
            }

        try:
            return self.simba_enhancer.get_enhancement_status()

        except Exception as e:
            return {"simba_enabled": True, "error": str(e), "status": "error"}

    async def reset_simba_memory(self) -> bool:
        """Reset SIMBA memory (useful for testing)."""
        if not self.enable_simba or not self.simba_enhancer:
            return False

        try:
            await self.simba_enhancer.reset_memory()
            logger.info("SIMBA enhancement memory reset")
            return True

        except Exception as e:
            logger.error(f"Failed to reset SIMBA memory: {e}")
            return False


# Factory functions


def create_enhancement_pipeline(
    analysis_module: Optional[ComposableQueryAnalysisModule] = None,
    enable_simba: bool = True,
    simba_config: Optional[SIMBAConfig] = None,
    query_fusion_config: Optional[Dict[str, Any]] = None,
    telemetry_provider: Optional[TelemetryProvider] = None,
    tenant_id: str = "default",
) -> QueryEnhancementPipeline:
    """Create complete query enhancement pipeline."""
    return QueryEnhancementPipeline(
        analysis_module=analysis_module,
        enable_simba=enable_simba,
        simba_config=simba_config,
        query_fusion_config=query_fusion_config,
        telemetry_provider=telemetry_provider,
        tenant_id=tenant_id,
    )
