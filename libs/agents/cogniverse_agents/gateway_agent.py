"""
GatewayAgent - A2A entry point for query classification and routing.

Classifies queries as "simple" (direct to execution agent) or "complex"
(forward to OrchestratorAgent) using GLiNER entity classification.
No LLM call. Target latency: <100ms.

Extracted from ComprehensiveRouter's fast path in routing/router.py.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import require_tenant_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLiNER label mappings
# ---------------------------------------------------------------------------

# Experimentally tuned: 7 labels (1 per modality + 2 generation types) produce
# significantly higher GLiNER confidence scores than the original 21 labels.
# Average top score: 0.56 (vs 0.41 with 21 labels). Zero missed queries.
# See experiment in docs/superpowers/specs/2026-04-08-a2a-architecture-restructuring-design.md
MODALITY_LABELS: Dict[str, List[str]] = {
    "video": ["video_content"],
    "text": ["text_information"],
    "audio": ["audio_content"],
    "image": ["image_content"],
    "document": ["document_content"],
}

GENERATION_LABELS: Dict[str, List[str]] = {
    "summary": ["summary_request"],
    "detailed_report": ["detailed_report_request"],
}

# Flat list of all labels for GLiNER predict_entities
ALL_LABELS: List[str] = [
    label for labels in MODALITY_LABELS.values() for label in labels
] + [label for labels in GENERATION_LABELS.values() for label in labels]

# ---------------------------------------------------------------------------
# Simple route map: (modality, generation_type) -> agent name
# ---------------------------------------------------------------------------

SIMPLE_ROUTE_MAP: Dict[
    Tuple[str, str], str
] = {
    # raw_results
    ("video", "raw_results"): "search_agent",
    ("text", "raw_results"): "search_agent",
    ("audio", "raw_results"): "audio_analysis_agent",
    ("image", "raw_results"): "image_search_agent",
    ("document", "raw_results"): "document_agent",
    # summary
    ("video", "summary"): "summarizer_agent",
    ("text", "summary"): "summarizer_agent",
    ("audio", "summary"): "summarizer_agent",
    ("image", "summary"): "summarizer_agent",
    ("document", "summary"): "summarizer_agent",
    # detailed_report
    ("video", "detailed_report"): "detailed_report_agent",
    ("text", "detailed_report"): "detailed_report_agent",
    ("audio", "detailed_report"): "detailed_report_agent",
    ("image", "detailed_report"): "detailed_report_agent",
    ("document", "detailed_report"): "detailed_report_agent",
}

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class GatewayInput(AgentInput):
    """Input for the gateway classifier."""

    query: str = Field(..., description="User query to classify and route")
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier (per-request)"
    )


class GatewayOutput(AgentOutput):
    """Output from the gateway classifier."""

    query: str = Field(..., description="Original query")
    complexity: Literal["simple", "complex"] = Field(
        ..., description="Query complexity classification"
    )
    modality: Literal["video", "text", "audio", "image", "document", "both"] = Field(
        ..., description="Detected content modality"
    )
    generation_type: Literal["raw_results", "summary", "detailed_report"] = Field(
        ..., description="Requested generation type"
    )
    routed_to: str = Field(
        ..., description="Target agent name or 'orchestrator_agent'"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    reasoning: str = Field(..., description="Brief explanation of routing decision")


class GatewayDeps(AgentDeps):
    """Dependencies for GatewayAgent."""

    gliner_model_name: str = Field(
        "urchade/gliner_large-v2.1", description="GLiNER model identifier"
    )
    gliner_threshold: float = Field(
        0.3, description="Entity detection confidence threshold"
    )
    fast_path_confidence_threshold: float = Field(
        0.4, description="Minimum confidence for simple (fast-path) routing"
    )


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class GatewayAgent(A2AAgent[GatewayInput, GatewayOutput, GatewayDeps]):
    """Classify and route queries without an LLM call.

    Uses GLiNER zero-shot NER to detect modality and generation type,
    then routes simple queries directly to the appropriate execution agent
    and complex queries to the OrchestratorAgent.
    """

    def __init__(
        self,
        deps: GatewayDeps,
        *,
        port: int = 8014,
    ) -> None:
        config = A2AAgentConfig(
            agent_name="gateway_agent",
            agent_description="Query classifier and router using GLiNER entity detection",
            capabilities=["gateway", "classification"],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        self._gliner_model = None

    def _load_artifact(self) -> None:
        """Load optimized thresholds from artifact store (if available).

        Called by the dispatcher after telemetry_manager and _artifact_tenant_id
        are injected — not from __init__ (telemetry_manager is not yet available).
        """
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        try:
            import asyncio
            import json
            from concurrent.futures import ThreadPoolExecutor

            from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

            tenant_id = getattr(self, "_artifact_tenant_id", None)
            if not tenant_id:
                raise RuntimeError(
                    f"{type(self).__name__}._load_artifact called before the "
                    f"dispatcher injected _artifact_tenant_id"
                )
            provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
            am = ArtifactManager(provider, tenant_id)

            async def _load() -> Optional[str]:
                return await am.load_blob("config", "gateway_thresholds")

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # Called from within an async context — run in a separate thread
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, _load())
                    blob = future.result()
            else:
                blob = asyncio.run(_load())

            if blob:
                config = json.loads(blob)
                if "fast_path_confidence_threshold" in config:
                    self.deps.fast_path_confidence_threshold = config["fast_path_confidence_threshold"]
                if "gliner_threshold" in config:
                    self.deps.gliner_threshold = config["gliner_threshold"]
                logger.info(
                    "GatewayAgent loaded optimized thresholds: "
                    f"fast_path={self.deps.fast_path_confidence_threshold}, "
                    f"gliner={self.deps.gliner_threshold}"
                )
        except Exception as e:
            logger.debug("No gateway artifact to load (using defaults): %s", e)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Resolve GLiNER from the module-level cache (loads once per name).

        The dispatcher creates fresh agent instances per request; without
        caching, every request reloaded the 1.4GB GLiNER weights and PyTorch
        retained them in heap, leaking through the suite.
        """
        if self._gliner_model is not None:
            return

        from cogniverse_core.common.models import get_or_load_gliner

        self._gliner_model = get_or_load_gliner(
            self.deps.gliner_model_name, logger=logger
        )

    # ------------------------------------------------------------------
    # Entity extraction and classification
    # ------------------------------------------------------------------

    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Run GLiNER entity prediction on the query."""
        self._ensure_model_loaded()
        if self._gliner_model is None:
            return []
        try:
            entities = self._gliner_model.predict_entities(
                query, ALL_LABELS, threshold=self.deps.gliner_threshold
            )
        except Exception as e:
            logger.error("GLiNER prediction failed: %s", e)
            return []
        return [
            {"text": e["text"], "label": e["label"], "score": e["score"]}
            for e in entities
        ]

    def _classify_modality(
        self, entities: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Determine the dominant modality from extracted entities.

        Returns (modality, confidence). If entities span multiple modalities,
        returns ("both", max_score).
        """
        modality_scores: Dict[str, float] = {}
        for entity in entities:
            for modality, labels in MODALITY_LABELS.items():
                if entity["label"] in labels:
                    modality_scores[modality] = max(
                        modality_scores.get(modality, 0.0), entity["score"]
                    )

        if not modality_scores:
            return "video", 0.0  # default modality, zero confidence

        if len(modality_scores) > 1:
            return "both", max(modality_scores.values())

        best_modality = max(modality_scores, key=modality_scores.get)  # type: ignore[arg-type]
        return best_modality, modality_scores[best_modality]

    def _classify_generation_type(
        self, entities: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Determine the generation type from extracted entities.

        Returns (generation_type, confidence). Defaults to "raw_results"
        when no generation-type labels are detected.
        """
        gen_scores: Dict[str, float] = {}
        for entity in entities:
            for gen_type, labels in GENERATION_LABELS.items():
                if entity["label"] in labels:
                    gen_scores[gen_type] = max(
                        gen_scores.get(gen_type, 0.0), entity["score"]
                    )

        if not gen_scores:
            return "raw_results", 1.0  # default: raw search results

        best_type = max(gen_scores, key=gen_scores.get)  # type: ignore[arg-type]
        return best_type, gen_scores[best_type]

    # Keywords that signal multi-step or analytical queries requiring orchestration
    _COMPLEXITY_KEYWORDS = frozenset({
        "analyze", "analyse", "compare", "contrast", "summarize", "summarise",
        "explain", "evaluate", "correlate", "investigate", "review", "assess",
        "combine", "merge", "synthesize", "synthesise", "report",
    })

    _MULTI_STEP_MARKERS = frozenset({
        "then", "after that", "followed by", "and also", "additionally",
        "step by step", "first", "finally", "next",
    })

    def _is_complex(
        self,
        query: str,
        modality: str,
        generation_type: str,
        entities: List[Dict[str, Any]],
        confidence: float,
    ) -> bool:
        """Decide whether a query needs orchestration.

        Complex if ANY of:
        - No entities detected (GLiNER can't classify it)
        - Low modality confidence (below threshold)
        - Multiple modalities detected ("both")
        - Generation type is detailed_report (always needs search → analysis → report)
        - Query contains analysis/comparison/synthesis verbs
        - Query contains multi-step markers ("then", "after that", "first...next")
        - Query has multiple clauses (3+ commas or 2+ "and")
        """
        # No entities → GLiNER can't classify → complex
        if not entities:
            return True

        if confidence < self.deps.fast_path_confidence_threshold:
            return True
        if modality == "both":
            return True

        # detailed_report always needs orchestration: search → analyze → write
        if generation_type == "detailed_report":
            return True

        # Check for analysis/synthesis verbs
        query_lower = query.lower()
        query_words = set(query_lower.split())
        if query_words & self._COMPLEXITY_KEYWORDS:
            return True

        # Check for multi-step markers
        if any(marker in query_lower for marker in self._MULTI_STEP_MARKERS):
            return True

        # Multiple clauses suggest a compound query
        if query_lower.count(",") >= 3 or query_lower.count(" and ") >= 2:
            return True

        return False

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _emit_gateway_span(
        self,
        *,
        tenant_id: str,
        query: str,
        complexity: str,
        modality: str,
        generation_type: str,
        routed_to: str,
        confidence: float,
    ) -> None:
        """Emit a cogniverse.gateway telemetry span."""
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return

        try:
            with self.telemetry_manager.span(
                "cogniverse.gateway",
                tenant_id=tenant_id,
                attributes={
                    "gateway.query": query[:200],
                    "gateway.complexity": complexity,
                    "gateway.modality": modality,
                    "gateway.generation_type": generation_type,
                    "gateway.routed_to": routed_to,
                    "gateway.confidence": confidence,
                },
            ):
                pass  # span auto-closes
        except Exception as e:
            logger.debug("Failed to emit gateway span: %s", e)

    def _emit_routing_span(
        self,
        *,
        tenant_id: str,
        query: str,
        complexity: str,
        modality: str,
        generation_type: str,
        routed_to: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Emit a cogniverse.routing span mirroring the RoutingAgent shape.

        Downstream evaluators (RoutingEvaluator, AnnotationAgent,
        RoutingSpanEvaluator, ModalitySpanCollector) filter on the
        `cogniverse.routing` span name and read `routing.*` attributes.
        Emitting it here lets those consumers observe gateway routing
        decisions the same way they observed legacy RoutingAgent decisions.
        """
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return

        try:
            with self.telemetry_manager.span(
                "cogniverse.routing",
                tenant_id=tenant_id,
                attributes={
                    "routing.query": query[:200],
                    "routing.chosen_agent": routed_to,
                    "routing.recommended_agent": routed_to,
                    "routing.confidence": confidence,
                    "routing.reasoning": reasoning[:200],
                    "routing.complexity": complexity,
                    "routing.modality": modality,
                    "routing.generation_type": generation_type,
                },
            ):
                pass  # span auto-closes
        except Exception as e:
            logger.debug("Failed to emit routing span: %s", e)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _process_impl(self, input: GatewayInput) -> GatewayOutput:
        """Classify and route a query."""
        import asyncio

        # GLiNER.predict_entities is sync, CPU-heavy (~200-500ms per call),
        # and would otherwise run on the event loop thread — starving every
        # other request including /health/live (readiness failure, pod
        # marked NotReady under concurrent orchestration load).
        entities = await asyncio.to_thread(self._extract_entities, input.query)
        modality, modality_confidence = self._classify_modality(entities)
        generation_type, gen_confidence = self._classify_generation_type(entities)

        # Conservative: low confidence in either dimension pushes borderline queries
        # to the orchestrator. This is safer for a gateway that should err on caution.
        overall_confidence = min(modality_confidence, gen_confidence)
        is_complex = self._is_complex(
            input.query, modality, generation_type, entities, overall_confidence
        )

        if is_complex:
            complexity = "complex"
            routed_to = "orchestrator_agent"
            reasoning = self._build_complex_reasoning(
                input.query, modality, generation_type, entities, overall_confidence
            )
        else:
            complexity = "simple"
            route_key = (modality, generation_type)
            routed_to = SIMPLE_ROUTE_MAP.get(route_key, "orchestrator_agent")
            reasoning = (
                f"Single {modality} {generation_type} query routed to {routed_to} "
                f"(confidence={overall_confidence:.2f})"
            )

        tenant_id = require_tenant_id(input.tenant_id, source="GatewayInput")
        self._emit_gateway_span(
            tenant_id=tenant_id,
            query=input.query,
            complexity=complexity,
            modality=modality,
            generation_type=generation_type,
            routed_to=routed_to,
            confidence=overall_confidence,
        )
        self._emit_routing_span(
            tenant_id=tenant_id,
            query=input.query,
            complexity=complexity,
            modality=modality,
            generation_type=generation_type,
            routed_to=routed_to,
            confidence=overall_confidence,
            reasoning=reasoning,
        )

        return GatewayOutput(
            query=input.query,
            complexity=complexity,
            modality=modality,
            generation_type=generation_type,
            routed_to=routed_to,
            confidence=overall_confidence,
            reasoning=reasoning,
        )

    def _build_complex_reasoning(
        self,
        query: str,
        modality: str,
        generation_type: str,
        entities: List[Dict[str, Any]],
        confidence: float,
    ) -> str:
        reasons = []
        if not entities:
            reasons.append("no entities detected")
        if modality == "both":
            reasons.append("multiple modalities detected")
        threshold = self.deps.fast_path_confidence_threshold
        if confidence < threshold:
            reasons.append(f"low confidence ({confidence:.2f} < {threshold})")
        if generation_type == "detailed_report":
            reasons.append("detailed report requires multi-step pipeline")

        query_lower = query.lower()
        query_words = set(query_lower.split())
        matched_keywords = query_words & self._COMPLEXITY_KEYWORDS
        if matched_keywords:
            reasons.append(f"analysis keywords: {', '.join(sorted(matched_keywords))}")

        matched_markers = [m for m in self._MULTI_STEP_MARKERS if m in query_lower]
        if matched_markers:
            reasons.append(f"multi-step markers: {', '.join(matched_markers)}")

        if query_lower.count(",") >= 3 or query_lower.count(" and ") >= 2:
            reasons.append("compound query (multiple clauses)")

        if not reasons:
            reasons.append("classified as complex by gateway")

        return f"Orchestrator needed: {'; '.join(reasons)}"
