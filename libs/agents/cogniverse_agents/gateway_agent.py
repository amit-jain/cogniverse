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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLiNER label mappings
# ---------------------------------------------------------------------------

MODALITY_LABELS: Dict[str, List[str]] = {
    "video": ["video_content", "visual_content", "media_content"],
    "text": ["document_content", "text_information", "written_content"],
    "audio": ["audio_content", "sound_content", "music_content", "podcast_content"],
    "image": [
        "image_content",
        "photo_content",
        "picture_content",
        "diagram_content",
        "chart_content",
    ],
    "document": [
        "pdf_content",
        "spreadsheet_content",
        "presentation_content",
    ],
}

GENERATION_LABELS: Dict[str, List[str]] = {
    "summary": ["summary_request"],
    "detailed_report": ["detailed_analysis", "report_request"],
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
        0.7, description="Minimum confidence for simple (fast-path) routing"
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
        deps: Optional[GatewayDeps] = None,
        *,
        port: int = 8000,
    ) -> None:
        if deps is None:
            deps = GatewayDeps()

        config = A2AAgentConfig(
            agent_name="gateway_agent",
            agent_description="Query classifier and router using GLiNER entity detection",
            capabilities=["gateway", "classification"],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        self._gliner_model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load GLiNER model with thread lock."""
        if self._gliner_model is not None:
            return

        from gliner import GLiNER

        from cogniverse_core.common.models import model_load_lock

        with model_load_lock:
            # Double-check after acquiring the lock
            if self._gliner_model is not None:
                return
            logger.info(
                "Loading GLiNER model: %s", self.deps.gliner_model_name
            )
            self._gliner_model = GLiNER.from_pretrained(
                self.deps.gliner_model_name
            )
            logger.info("GLiNER model loaded")

    # ------------------------------------------------------------------
    # Entity extraction and classification
    # ------------------------------------------------------------------

    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Run GLiNER entity prediction on the query."""
        self._ensure_model_loaded()
        entities = self._gliner_model.predict_entities(
            query, ALL_LABELS, threshold=self.deps.gliner_threshold
        )
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

    def _is_complex(
        self,
        modality: str,
        entities: List[Dict[str, Any]],
        confidence: float,
    ) -> bool:
        """Decide whether a query is complex and needs orchestration.

        Complex if:
        - No entities detected at all
        - Low confidence (below fast_path_confidence_threshold)
        - Multiple modalities detected ("both")
        """
        if not entities:
            return True
        if confidence < self.deps.fast_path_confidence_threshold:
            return True
        if modality == "both":
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

        with self.telemetry_manager.span(
            "cogniverse.gateway",
            tenant_id=tenant_id,
            attributes={
                "gateway.query": query,
                "gateway.complexity": complexity,
                "gateway.modality": modality,
                "gateway.generation_type": generation_type,
                "gateway.routed_to": routed_to,
                "gateway.confidence": confidence,
            },
        ):
            pass  # span auto-closes

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _process_impl(self, input: GatewayInput) -> GatewayOutput:
        """Classify and route a query."""
        entities = self._extract_entities(input.query)
        modality, modality_confidence = self._classify_modality(entities)
        generation_type, gen_confidence = self._classify_generation_type(entities)

        overall_confidence = min(modality_confidence, gen_confidence)
        is_complex = self._is_complex(modality, entities, overall_confidence)

        if is_complex:
            complexity = "complex"
            routed_to = "orchestrator_agent"
            reasoning = self._build_complex_reasoning(
                modality, entities, overall_confidence
            )
        else:
            complexity = "simple"
            route_key = (modality, generation_type)
            routed_to = SIMPLE_ROUTE_MAP.get(route_key, "orchestrator_agent")
            reasoning = (
                f"Single {modality} {generation_type} query routed to {routed_to} "
                f"(confidence={overall_confidence:.2f})"
            )

        tenant_id = input.tenant_id or "default"
        self._emit_gateway_span(
            tenant_id=tenant_id,
            query=input.query,
            complexity=complexity,
            modality=modality,
            generation_type=generation_type,
            routed_to=routed_to,
            confidence=overall_confidence,
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

    @staticmethod
    def _build_complex_reasoning(
        modality: str,
        entities: List[Dict[str, Any]],
        confidence: float,
    ) -> str:
        if not entities:
            return "No entities detected; forwarding to orchestrator for deeper analysis"
        if modality == "both":
            return "Multiple modalities detected; orchestrator needed for cross-modal coordination"
        return (
            f"Low classification confidence ({confidence:.2f}); "
            f"forwarding to orchestrator for robust handling"
        )
