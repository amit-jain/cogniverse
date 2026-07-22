"""
Detailed Report Agent with full A2A support, VLM integration, and thinking phase.
Generates comprehensive detailed reports with visual and technical analysis.
"""

import asyncio
import contextvars
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

# Enhanced routing support
from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
from cogniverse_agents.multimodal import KeyframeImageResolver
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_core.common.media import MediaConfig, MediaLocator
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.common.vlm_interface import VLMInterface
from cogniverse_foundation.config.semantic_router import routed_lm_context_for

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8004

# Per-request report call state (frames attached, degraded flag/reason). The
# dispatcher builds a fresh agent per request, but the standalone A2A app
# serves a module-level singleton — instance attributes there race concurrent
# requests, letting one request's degraded flag bleed into a sibling's
# metadata. A ContextVar is task-scoped, so each request reads only its own.
_REPORT_CALL_STATE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "_detailed_report_call_state", default=None
)


class DetailedReportInput(AgentInput):
    """Type-safe input for detailed report generation"""

    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    query: str = Field(..., description="Query for report")
    search_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Results to analyze"
    )
    report_type: str = Field(
        "comprehensive", description="Type: comprehensive, technical, analytical"
    )
    include_visual_analysis: bool = Field(True, description="Include visual analysis")
    include_technical_details: bool = Field(
        True, description="Include technical details"
    )
    include_recommendations: bool = Field(True, description="Include recommendations")
    max_results_to_analyze: int = Field(20, description="Maximum results to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    rlm: Optional[RLMOptions] = Field(
        None,
        description="RLM configuration. None=disabled, set RLMOptions to enable RLM inference for A/B testing",
    )

    # Enrichment fields forwarded by the orchestrator from preprocessing
    # agents. Optional — report generation works on raw query without them.
    enhanced_query: Optional[str] = Field(
        None,
        description="Rewritten query from QueryEnhancementAgent",
    )
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Entities from EntityExtractionAgent",
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relationships from EntityExtractionAgent",
    )
    query_variants: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Query variants from QueryEnhancementAgent",
    )


class DetailedReportOutput(AgentOutput):
    """Type-safe output from detailed report generation"""

    executive_summary: str = Field(..., description="Executive summary")
    detailed_findings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed findings"
    )
    visual_analysis: List[Dict[str, Any]] = Field(
        default_factory=list, description="Visual analysis"
    )
    technical_details: List[Dict[str, Any]] = Field(
        default_factory=list, description="Technical details"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    confidence_assessment: Dict[str, float] = Field(
        default_factory=dict, description="Confidence assessment"
    )
    thinking_process: Dict[str, Any] = Field(
        default_factory=dict, description="Thinking phase details"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    # RLM synthesis (when RLM is enabled via input.rlm)
    rlm_synthesis: Optional[str] = Field(
        None, description="RLM-synthesized answer from results (only when RLM enabled)"
    )
    rlm_telemetry: Optional[Dict[str, Any]] = Field(
        None, description="RLM telemetry metrics for A/B testing"
    )


class DetailedReportDeps(AgentDeps):
    """Dependencies for detailed report agent (tenant-agnostic at startup)."""

    max_report_length: int = Field(2000, description="Maximum report length")
    thinking_enabled: bool = Field(True, description="Enable thinking phase")
    visual_analysis_enabled: bool = Field(True, description="Enable visual analysis")
    technical_analysis_enabled: bool = Field(
        True, description="Enable technical analysis"
    )
    multimodal_generation_enabled: bool = Field(
        True,
        description=(
            "Attach retrieved keyframes to the report LLM. A keyframe not yet "
            "in object storage is skipped, so this degrades to text-only when "
            "frames are unavailable."
        ),
    )
    max_keyframes_to_llm: int = Field(
        4, description="Cap on keyframes attached to the report LLM"
    )


# DSPy Signatures and Module
class ReportGenerationSignature(dspy.Signature):
    """Generate comprehensive detailed reports"""

    content = dspy.InputField(desc="Search results content to analyze")
    query = dspy.InputField(desc="Original user query")
    report_type = dspy.InputField(
        desc="Type of report: comprehensive, technical, analytical"
    )
    keyframes: list[dspy.Image] = dspy.InputField(
        desc=(
            "Key frames from the top video results. Ground the report in what "
            "these frames actually show, not only the titles and scores."
        )
    )

    executive_summary = dspy.OutputField(desc="Executive summary of findings")
    key_findings = dspy.OutputField(desc="Key findings (comma-separated)")
    recommendations = dspy.OutputField(desc="Recommendations (comma-separated)")
    confidence_score = dspy.OutputField(desc="Confidence in report quality (0.0-1.0)")


class ReportGenerationModule(dspy.Module):
    """DSPy module for intelligent report generation"""

    def __init__(self):
        super().__init__()
        self.report_generator = dspy.ChainOfThought(ReportGenerationSignature)

    def forward(
        self,
        content: str,
        query: str,
        report_type: str = "comprehensive",
        *,
        keyframes: List[dspy.Image],
    ):
        """Generate report using DSPy"""
        return self.report_generator(
            content=content,
            query=query,
            report_type=report_type,
            keyframes=keyframes,
        )


@dataclass
class ReportRequest:
    """Request for detailed report generation"""

    query: str
    search_results: List[Dict[str, Any]]
    report_type: str = "comprehensive"  # comprehensive, technical, analytical
    include_visual_analysis: bool = True
    include_technical_details: bool = True
    include_recommendations: bool = True
    max_results_to_analyze: int = 20
    context: Optional[Dict[str, Any]] = None


@dataclass
class ThinkingPhase:
    """Agent's thinking process for report generation"""

    content_analysis: Dict[str, Any]
    visual_assessment: Dict[str, Any]
    technical_findings: List[str]
    patterns_identified: List[str]
    gaps_and_limitations: List[str]
    reasoning: str


@dataclass
class ReportResult:
    """Detailed report result"""

    executive_summary: str
    detailed_findings: List[Dict[str, Any]]
    visual_analysis: List[Dict[str, Any]]
    technical_details: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_assessment: Dict[str, float]
    thinking_phase: ThinkingPhase
    metadata: Dict[str, Any]
    # Enhanced fields for relationship-aware reports
    relationship_analysis: Optional[Dict[str, Any]] = None
    entity_analysis: Optional[Dict[str, Any]] = None
    enhancement_applied: bool = False


class DetailedReportAgent(
    RLMAwareMixin,
    MemoryAwareMixin,
    A2AAgent[DetailedReportInput, DetailedReportOutput, DetailedReportDeps],
):
    """
    Type-safe agent for generating comprehensive detailed reports with full A2A support.
    Provides visual analysis, technical insights, and actionable recommendations.
    """

    def __init__(
        self, deps: DetailedReportDeps, config_manager=None, port: int = DEFAULT_PORT
    ):
        """
        Initialize detailed report agent with typed dependencies.

        Args:
            deps: Typed dependencies with configuration
            config_manager: ConfigManager instance (optional, will create default if None)
            port: A2A server port

        Raises:
            TypeError: If deps is not DetailedReportDeps
            ValidationError: If deps fails Pydantic validation
        """
        logger.info("Initializing DetailedReportAgent (tenant-agnostic)...")

        # Create DSPy report generation module
        self.report_module = ReportGenerationModule()

        # Create A2A config
        config = A2AAgentConfig(
            agent_name="detailed_report_agent",
            agent_description="Type-safe detailed reports with visual and technical analysis",
            capabilities=[
                "detailed_report",
                "visual_analysis",
                "technical_analysis",
                "comprehensive_analysis",
                "relationship_aware_reporting",
            ],
            port=port,
            version="2.0.0",
        )

        # Initialize A2A base
        super().__init__(deps=deps, config=config, dspy_module=self.report_module)

        # Config manager — required from caller, no silent fallback
        if config_manager is None:
            raise ValueError(
                "config_manager is required. "
                "Pass create_default_config_manager() from startup boundary."
            )
        self._config_manager = config_manager

        # Initialize DSPy components
        self._initialize_vlm_client()

        # Initialize VLM for visual analysis (cluster-wide VLM config)
        self.vlm = VLMInterface(
            config_manager=self._config_manager, tenant_id=SYSTEM_TENANT_ID
        )

        # Configuration from deps
        self.max_report_length = deps.max_report_length
        self.thinking_enabled = deps.thinking_enabled
        self.visual_analysis_enabled = deps.visual_analysis_enabled
        self.technical_analysis_enabled = deps.technical_analysis_enabled
        self.multimodal_generation_enabled = deps.multimodal_generation_enabled
        self.max_keyframes_to_llm = deps.max_keyframes_to_llm

        # Resolves top-K keyframes from search hits into dspy.Image inputs so
        # the report LLM sees the frames, not just their titles. The locator
        # targets the object store so s3:// keyframes are fetchable at answer
        # time (a bare MediaConfig only handles file://).
        self._keyframe_resolver = KeyframeImageResolver(
            MediaLocator(
                tenant_id=SYSTEM_TENANT_ID,
                config=MediaConfig.for_object_store(
                    getattr(self, "_minio_endpoint", None)
                ),
            )
        )

        logger.info("DetailedReportAgent initialized (tenant-agnostic)")

    def _initialize_vlm_client(self):
        """Resolve the report LLM endpoint from centralized llm_config.

        The LM itself is built per request in ``routed_lm_context_for`` so it
        can be routed through the semantic router for the request tenant.
        """
        from cogniverse_foundation.config.utils import get_config

        system_config = get_config(
            tenant_id=SYSTEM_TENANT_ID, config_manager=self._config_manager
        )
        # Object-store endpoint for answer-time keyframe resolution; credentials
        # come from AWS_* env mirrored at the runtime entrypoint.
        self._minio_endpoint = self._config_manager.get_system_config().minio_endpoint
        llm_config = system_config.get_llm_config()
        self._llm_config = llm_config.resolve("detailed_report_agent")
        logger.info(
            f"Resolved detailed-report LLM endpoint: {self._llm_config.model} "
            f"at {self._llm_config.api_base}"
        )

    async def _generate_report(self, request: ReportRequest) -> ReportResult:
        """
        Internal: Generate a detailed report with comprehensive analysis

        Args:
            request: Report generation request

        Returns:
            Detailed report result with thinking phase
        """
        logger.info(f"Generating detailed report for: '{request.query}'")

        with routed_lm_context_for(
            self._config_manager,
            getattr(self, "_memory_tenant_id", None) or SYSTEM_TENANT_ID,
            "detailed_report_agent",
            endpoint=self._llm_config,
        ):
            try:
                # Thinking pass: comprehensive analysis
                if self.thinking_enabled:
                    self.emit_progress("thinking", "Analyzing content...")
                    thinking_phase = await self._thinking_phase(request)
                else:
                    thinking_phase = self._empty_thinking_phase(request)

                # Visual analysis (if enabled)
                self.emit_progress("visual_analysis", "Performing visual analysis...")
                visual_analysis = await self._perform_visual_analysis(
                    request, thinking_phase
                )

                # Executive summary
                self.emit_progress(
                    "executive_summary", "Generating executive summary..."
                )
                (
                    executive_summary,
                    lm_recommendations,
                ) = await self._generate_executive_summary(request, thinking_phase)
                executive_summary = self._enforce_max_length(
                    executive_summary, self.max_report_length
                )

                # Detailed findings
                self.emit_progress("findings", "Compiling detailed findings...")
                detailed_findings = self._generate_detailed_findings(
                    request, thinking_phase
                )

                # Technical details
                self.emit_progress(
                    "technical_details", "Generating technical details..."
                )
                technical_details = self._generate_technical_details(
                    request, thinking_phase
                )

                # Recommendations: prefer the LM's actionable list; fall back to
                # the derived template when the LM produced none. The canned
                # method already honors request.include_recommendations, so gate
                # the LM override on it too.
                self.emit_progress("recommendations", "Generating recommendations...")
                recommendations = self._generate_recommendations(
                    request, thinking_phase
                )
                if request.include_recommendations and lm_recommendations:
                    recommendations = lm_recommendations

                # Confidence assessment
                self.emit_progress("confidence", "Calculating confidence assessment...")
                confidence_assessment = self._calculate_confidence_assessment(
                    request, thinking_phase
                )

                result = ReportResult(
                    executive_summary=executive_summary,
                    detailed_findings=detailed_findings,
                    visual_analysis=visual_analysis,
                    technical_details=technical_details,
                    recommendations=recommendations,
                    confidence_assessment=confidence_assessment,
                    thinking_phase=thinking_phase,
                    metadata={
                        "results_analyzed": len(request.search_results),
                        "report_type": request.report_type,
                        "visual_analysis_enabled": request.include_visual_analysis,
                        "keyframes_attached": (_REPORT_CALL_STATE.get() or {}).get(
                            "keyframes_attached", 0
                        ),
                        # True when the answer LM call failed and the executive
                        # summary is the templated fallback, not a grounded
                        # report — callers must be able to tell the two apart.
                        "report_degraded": (_REPORT_CALL_STATE.get() or {}).get(
                            "report_degraded", False
                        ),
                        "report_degraded_reason": (_REPORT_CALL_STATE.get() or {}).get(
                            "report_degraded_reason", ""
                        ),
                        "technical_analysis_enabled": (
                            request.include_technical_details
                            and self.technical_analysis_enabled
                        ),
                        "recommendations_enabled": request.include_recommendations,
                    },
                )

                logger.info("Detailed report generation complete")
                return result

            except Exception as e:
                logger.error(f"Report generation failed: {e}")
                raise

    async def generate_report(self, request: ReportRequest) -> ReportResult:
        """
        Public method to generate detailed report.

        This is a public alias for _generate_report that provides the expected interface.

        Args:
            request: Report generation request

        Returns:
            Detailed report result with thinking phase
        """
        return await self._generate_report(request)

    async def _thinking_phase(self, request: ReportRequest) -> ThinkingPhase:
        """Comprehensive thinking phase for report generation"""
        logger.info("Starting thinking phase for detailed report...")

        # Analyze content structure and types
        content_analysis = self._analyze_content_structure(request.search_results)

        # Assess visual content availability and quality
        visual_assessment = self._assess_visual_content(request.search_results)

        # Identify technical findings
        technical_findings = self._identify_technical_findings(request.search_results)

        # Identify patterns and trends
        patterns_identified = self._identify_patterns(request.search_results)

        # Identify gaps and limitations
        gaps_and_limitations = self._identify_gaps_and_limitations(
            request.search_results
        )

        # Generate reasoning
        reasoning = self._generate_thinking_reasoning(
            request,
            content_analysis,
            visual_assessment,
            technical_findings,
            patterns_identified,
            gaps_and_limitations,
        )

        return ThinkingPhase(
            content_analysis=content_analysis,
            visual_assessment=visual_assessment,
            technical_findings=technical_findings,
            patterns_identified=patterns_identified,
            gaps_and_limitations=gaps_and_limitations,
            reasoning=reasoning,
        )

    def _empty_thinking_phase(self, request: ReportRequest) -> ThinkingPhase:
        """Neutral thinking phase used when the thinking pass is disabled."""
        return ThinkingPhase(
            content_analysis={
                "total_results": len(request.search_results),
                "content_types": {},
                "duration_distribution": {"short": 0, "medium": 0, "long": 0},
                "quality_metrics": {"high": 0, "medium": 0, "low": 0},
                "avg_relevance": 0.0,
            },
            visual_assessment={
                "has_visual_content": False,
                "visual_elements": {"thumbnails": 0, "keyframes": 0, "images": 0},
                "visual_coverage": 0,
                "visual_analysis_feasible": False,
            },
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning="",
        )

    @staticmethod
    def _enforce_max_length(text: str, max_length: int) -> str:
        """Truncate text at the last word boundary within max_length."""
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if len(text) <= max_length:
            return text
        cut = text[:max_length]
        boundary = cut.rfind(" ")
        cut = cut[:boundary] if boundary > 0 else cut[: max_length - 1]
        return cut.rstrip() + "…"

    def _analyze_content_structure(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the structure and composition of content"""
        content_types = {}
        duration_distribution = {"short": 0, "medium": 0, "long": 0}
        quality_metrics = {"high": 0, "medium": 0, "low": 0}

        for result in results:
            # Categorize content type
            content_type = result.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1

            # Categorize duration
            duration = result.get("duration", 0)
            if duration < 60:
                duration_distribution["short"] += 1
            elif duration < 300:
                duration_distribution["medium"] += 1
            else:
                duration_distribution["long"] += 1

            # Assess quality based on score/relevance
            score = result.get("score", result.get("relevance", 0))
            if score > 0.8:
                quality_metrics["high"] += 1
            elif score > 0.5:
                quality_metrics["medium"] += 1
            else:
                quality_metrics["low"] += 1

        return {
            "total_results": len(results),
            "content_types": content_types,
            "duration_distribution": duration_distribution,
            "quality_metrics": quality_metrics,
            "avg_relevance": (
                sum(r.get("score", r.get("relevance", 0)) for r in results)
                / len(results)
                if results
                else 0
            ),
        }

    def _assess_visual_content(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess visual content availability and characteristics"""
        visual_elements = {"thumbnails": 0, "keyframes": 0, "images": 0}

        has_visual_content = False

        for result in results:
            if "thumbnail" in result:
                visual_elements["thumbnails"] += 1
                has_visual_content = True

            if "keyframes" in result:
                visual_elements["keyframes"] += len(result["keyframes"])
                has_visual_content = True

            if "image_path" in result:
                visual_elements["images"] += 1
                has_visual_content = True

        return {
            "has_visual_content": has_visual_content,
            "visual_elements": visual_elements,
            "visual_coverage": sum(visual_elements.values()),
            "visual_analysis_feasible": has_visual_content
            and self.visual_analysis_enabled,
        }

    def _identify_technical_findings(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify technical aspects and findings"""
        findings = []

        # Content format analysis
        formats = set()
        for result in results:
            if "format" in result:
                formats.add(result["format"])
            content_type = result.get("content_type", "")
            if content_type:
                formats.add(content_type)

        if formats:
            findings.append(f"Content formats: {', '.join(formats)}")

        # Metadata completeness
        metadata_completeness = sum(
            1 for r in results if r.get("metadata") and len(r["metadata"]) > 2
        )
        if metadata_completeness > 0:
            findings.append(
                f"{metadata_completeness}/{len(results)} results have comprehensive metadata"
            )

        return findings

    def _identify_patterns(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns and trends in results"""
        patterns = []

        # Temporal patterns
        if any("timestamp" in r or "date" in r for r in results):
            patterns.append("Temporal data available for trend analysis")

        # Quality patterns
        avg_score = (
            sum(r.get("score", r.get("relevance", 0)) for r in results) / len(results)
            if results
            else 0
        )
        if avg_score > 0.7:
            patterns.append("High average relevance across results")
        elif avg_score < 0.3:
            patterns.append("Low average relevance - may need query refinement")

        # Content diversity
        unique_types = len(set(r.get("content_type", "unknown") for r in results))
        if unique_types > 1:
            patterns.append(f"Diverse content types ({unique_types} different types)")

        return patterns

    def _identify_gaps_and_limitations(
        self, results: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify gaps and limitations in the data"""
        gaps = []

        # Missing visual content
        if not any("thumbnail" in r or "image_path" in r for r in results):
            gaps.append("No visual content available for analysis")

        # Limited result set
        if len(results) < 5:
            gaps.append(f"Limited result set ({len(results)} results)")

        # Missing metadata
        missing_metadata_count = sum(
            1
            for r in results
            if not r.get("metadata") or len(r.get("metadata", {})) < 2
        )
        if missing_metadata_count > len(results) / 2:
            gaps.append("Many results lack comprehensive metadata")

        return gaps if gaps else ["No significant gaps identified"]

    def _generate_thinking_reasoning(
        self,
        request: ReportRequest,
        content_analysis: Dict[str, Any],
        visual_assessment: Dict[str, Any],
        technical_findings: List[str],
        patterns: List[str],
        gaps: List[str],
    ) -> str:
        """Generate reasoning for the report approach"""
        reasoning = f"""
Report Generation Strategy:
- Query: "{request.query}"
- Results: {content_analysis["total_results"]} items analyzed
- Average relevance: {content_analysis["avg_relevance"]:.2f}
- Visual content: {"Available" if visual_assessment["has_visual_content"] else "Not available"}
- Technical findings: {len(technical_findings)} aspects identified
- Patterns detected: {len(patterns)}
- Limitations: {len(gaps)} gaps identified

Approach: Will generate {request.report_type} report with focus on data quality,
technical accuracy, and actionable insights. Visual analysis {"included" if request.include_visual_analysis else "excluded"}.
""".strip()

        return reasoning

    async def _perform_visual_analysis(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Perform visual analysis using VLM"""
        if not request.include_visual_analysis or not self.visual_analysis_enabled:
            return []

        if not thinking_phase.visual_assessment.get("has_visual_content"):
            return []

        logger.info("Performing visual analysis...")

        visual_analysis = []
        for result in request.search_results[: request.max_results_to_analyze]:
            if "thumbnail" in result or "image_path" in result:
                image_path = result.get("thumbnail") or result.get("image_path")

                try:
                    analysis = await self.vlm.analyze_visual_content(
                        [image_path], request.query
                    )

                    visual_analysis.append(
                        {
                            "result_id": result.get("id", "unknown"),
                            "insights": analysis.get("insights", []),
                            # VLMInterface.analyze_visual_content emits
                            # relevance_score, not confidence — the old key was
                            # always absent, so every visual analysis reported 0.0.
                            "confidence": analysis.get("relevance_score", 0.0),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Visual analysis failed for {image_path}: {e}")

        return visual_analysis

    @staticmethod
    def _parse_llm_list(raw: Any) -> List[str]:
        """Parse a DSPy list output (declared comma-separated, but LMs often
        emit one item per line and/or bullet/number prefixes) into a clean
        list. A non-string (e.g. an unset field) yields no items."""
        if not isinstance(raw, str) or not raw.strip():
            return []
        lines = [ln for ln in raw.replace("\r", "").split("\n") if ln.strip()]
        if len(lines) <= 1:
            lines = raw.split(",")
        items: List[str] = []
        for ln in lines:
            cleaned = ln.strip().lstrip("-*•").strip()
            head, sep, tail = cleaned.partition(" ")
            if sep and head.rstrip(".)").isdigit():
                cleaned = tail.strip()
            if cleaned:
                items.append(cleaned)
        return items

    async def _generate_executive_summary(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> tuple[str, List[str]]:
        """Generate the executive summary and the LM's recommendations via DSPy.

        The report signature produces both; returning the recommendations here
        lets the report use the LM's actionable list instead of the canned
        template (with a fallback to the template when the LM produced none)."""
        total_results = thinking_phase.content_analysis.get(
            "total_results", len(request.search_results)
        )

        content_parts = [f"Total Results: {total_results}"]
        for result in request.search_results[:10]:
            title = result.get("title", result.get("video_id", "Unknown"))
            score = result.get("score", result.get("relevance", 0))
            content_parts.append(f"- {title} (score: {score:.2f})")

        content_text = "\n".join(content_parts)

        keyframe_images = []
        if (
            self.multimodal_generation_enabled
            and self.visual_analysis_enabled
            and request.include_visual_analysis
        ):
            # collect() downloads frames from object storage — off the loop,
            # like the call_dspy below, or every concurrent request stalls.
            keyframe_images = await asyncio.to_thread(
                self._keyframe_resolver.collect,
                request.search_results[: request.max_results_to_analyze],
                max_images=self.max_keyframes_to_llm,
            )
        # Observable count of frames actually attached to the LLM call — surfaced
        # in the report metadata so callers (and e2e tests) can confirm the
        # retrieved keyframes reached the answer model.
        call_state = {
            "keyframes_attached": len(keyframe_images),
            "report_degraded": False,
            "report_degraded_reason": "",
        }
        _REPORT_CALL_STATE.set(call_state)
        try:
            dspy_result = await self.call_dspy(
                self.report_module,
                output_field="executive_summary",
                content=content_text,
                query=request.query,
                report_type=request.report_type,
                keyframes=keyframe_images,
            )
            executive_summary = dspy_result.executive_summary
            raw_recommendations = getattr(dspy_result, "recommendations", "")
        except Exception as e:
            # The answer LM call failed (e.g. a payload/context overflow from
            # the attached keyframes) or returned an unusable shape. Degrade
            # to a templated stub so the request still returns — but FLAG it
            # in the metadata below, so a fallback is never indistinguishable
            # from a real grounded report.
            logger.error(f"DSPy summary generation failed: {e}")
            call_state["report_degraded"] = True
            call_state["report_degraded_reason"] = f"{type(e).__name__}: {e}"
            return (
                f"Analysis of {len(request.search_results)} results for "
                f"'{request.query}' with average relevance of "
                f"{thinking_phase.content_analysis['avg_relevance']:.2f}.",
                [],
            )
        # Recommendation parsing runs OUTSIDE the degrade guard: a bug in our
        # own post-processing must surface, not masquerade as a degraded
        # report.
        return executive_summary, self._parse_llm_list(raw_recommendations)

    def _generate_detailed_findings(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Generate detailed findings section"""
        findings = []

        # Content analysis finding
        findings.append(
            {
                "category": "Content Analysis",
                "finding": f"Analyzed {thinking_phase.content_analysis['total_results']} results",
                "details": thinking_phase.content_analysis,
                "significance": (
                    "high"
                    if thinking_phase.content_analysis["avg_relevance"] > 0.7
                    else "medium"
                ),
            }
        )

        # Pattern findings
        if thinking_phase.patterns_identified:
            findings.append(
                {
                    "category": "Patterns Identified",
                    "finding": f"{len(thinking_phase.patterns_identified)} patterns detected",
                    "details": thinking_phase.patterns_identified,
                    "significance": "medium",
                }
            )

        # Technical findings
        if thinking_phase.technical_findings:
            findings.append(
                {
                    "category": "Technical Analysis",
                    "finding": f"{len(thinking_phase.technical_findings)} technical aspects identified",
                    "details": thinking_phase.technical_findings,
                    "significance": "medium",
                }
            )

        return findings

    def _generate_technical_details(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Generate technical details section"""
        if not request.include_technical_details or not self.technical_analysis_enabled:
            return []

        technical_details = []

        # Content distribution
        technical_details.append(
            {
                "category": "Content Distribution",
                "metrics": thinking_phase.content_analysis["content_types"],
                "analysis": "Distribution of content types across results",
            }
        )

        # Quality metrics
        technical_details.append(
            {
                "category": "Quality Metrics",
                "metrics": thinking_phase.content_analysis["quality_metrics"],
                "analysis": "Quality distribution of search results",
            }
        )

        return technical_details

    def _generate_recommendations(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[str]:
        """Generate actionable recommendations"""
        if not request.include_recommendations:
            return []

        recommendations = []

        # Based on quality metrics
        avg_relevance = thinking_phase.content_analysis["avg_relevance"]
        if avg_relevance < 0.5:
            recommendations.append(
                "Consider refining the search query to improve result relevance"
            )
        elif avg_relevance > 0.8:
            recommendations.append(
                "High relevance scores indicate good query formulation"
            )

        # Based on gaps
        if thinking_phase.gaps_and_limitations:
            for gap in thinking_phase.gaps_and_limitations[:2]:
                if "metadata" in gap.lower():
                    recommendations.append(
                        "Enhance metadata collection for better analysis"
                    )
                elif "visual" in gap.lower():
                    recommendations.append(
                        "Consider incorporating visual content for richer analysis"
                    )

        # Based on patterns
        if len(thinking_phase.patterns_identified) < 2:
            recommendations.append(
                "Expand result set to identify more meaningful patterns"
            )

        return recommendations if recommendations else ["Continue current approach"]

    def _calculate_confidence_assessment(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> Dict[str, float]:
        """Calculate multi-dimensional confidence assessment"""
        # Data quality confidence
        data_quality = thinking_phase.content_analysis["avg_relevance"]

        # Completeness confidence
        completeness = min(
            1.0, len(request.search_results) / request.max_results_to_analyze
        )

        # Visual analysis confidence
        visual_confidence = (
            0.8 if thinking_phase.visual_assessment["has_visual_content"] else 0.3
        )

        # Technical confidence
        technical_confidence = (
            0.9 if len(thinking_phase.technical_findings) > 2 else 0.6
        )

        # Overall confidence
        overall = (
            data_quality * 0.4
            + completeness * 0.2
            + visual_confidence * 0.2
            + technical_confidence * 0.2
        )

        return {
            "overall": overall,
            "data_quality": data_quality,
            "completeness": completeness,
            "visual_analysis": visual_confidence,
            "technical_analysis": technical_confidence,
        }

    async def _process_impl(self, input: DetailedReportInput) -> DetailedReportOutput:
        """
        Process report generation request with typed input/output.

        Args:
            input: Typed input with query, search_results, report_type, etc.

        Returns:
            DetailedReportOutput with executive_summary, findings, recommendations, etc.
        """
        self.emit_progress("preparation", "Preparing report request...")
        if input.tenant_id is not None:
            self.set_tenant_for_context(input.tenant_id)

        # Prefer the upstream-enhanced query when the orchestrator forwards
        # one; otherwise apply per-tenant memory context to the raw query.
        if input.enhanced_query:
            report_query = input.enhanced_query
        elif input.tenant_id is not None:
            report_query = await self.inject_context_into_prompt_async(
                input.query, input.query
            )
        else:
            report_query = input.query

        merged_context: Dict[str, Any] = dict(input.context or {})
        if input.entities:
            merged_context["entities"] = input.entities
        if input.relationships:
            merged_context["relationships"] = input.relationships

        request = ReportRequest(
            query=report_query,
            search_results=input.search_results,
            context=merged_context,
            report_type=input.report_type,
            include_visual_analysis=input.include_visual_analysis,
            include_technical_details=input.include_technical_details,
            include_recommendations=input.include_recommendations,
            max_results_to_analyze=input.max_results_to_analyze,
        )

        self.emit_progress("report_generation", "Generating detailed report...")
        # Generate report
        result = await self._generate_report(request)

        # Build results context for optional RLM synthesis
        results_context = "\n\n".join(
            f"Result {i + 1}: {r.get('title', r.get('video_id', 'Unknown'))} "
            f"(score: {r.get('score', r.get('relevance', 0)):.2f})\n"
            f"{r.get('description', r.get('text', ''))}"
            for i, r in enumerate(input.search_results[: input.max_results_to_analyze])
        )

        rlm_synthesis = None
        rlm_telemetry = None

        if self.should_use_rlm_for_query(input.rlm, results_context):
            self.emit_progress("rlm_synthesis", "Synthesizing answer with RLM...")
            logger.info(f"RLM enabled for query: {input.query[:50]}...")
            try:
                rlm_result = await asyncio.to_thread(
                    self.process_with_rlm,
                    query=input.query,
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

        return DetailedReportOutput(
            executive_summary=result.executive_summary,
            detailed_findings=result.detailed_findings,
            visual_analysis=result.visual_analysis,
            technical_details=result.technical_details,
            recommendations=result.recommendations,
            confidence_assessment=result.confidence_assessment,
            thinking_process={
                "content_analysis": result.thinking_phase.content_analysis,
                "visual_assessment": result.thinking_phase.visual_assessment,
                "technical_findings": result.thinking_phase.technical_findings,
                "patterns": result.thinking_phase.patterns_identified,
                "gaps": result.thinking_phase.gaps_and_limitations,
                "reasoning": result.thinking_phase.reasoning,
            },
            metadata=result.metadata,
            rlm_synthesis=rlm_synthesis,
            rlm_telemetry=rlm_telemetry,
        )

    # Skill descriptors come from A2AAgent.get_agent_skills() (-> _get_skills()).


# --- FastAPI Server ---

# Global agent instance
detailed_report_agent = None


@asynccontextmanager
async def lifespan(application):
    """Initialize agent on startup — tenant-agnostic, no env vars."""
    global detailed_report_agent

    try:
        from cogniverse_foundation.config.utils import create_default_config_manager

        deps = DetailedReportDeps()
        # __init__ requires a config_manager (no silent fallback); build the
        # default one at the startup boundary. Without this the standalone app
        # crashed on launch and never served a request.
        detailed_report_agent = DetailedReportAgent(
            deps=deps, config_manager=create_default_config_manager()
        )
        logger.info("Detailed report agent initialized (tenant-agnostic)")
    except Exception as e:
        logger.error(f"Failed to initialize detailed report agent: {e}")
        raise
    yield


app = FastAPI(
    title="Detailed Report Agent",
    description="Generates comprehensive detailed reports with visual and technical analysis",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not detailed_report_agent:
        return {"status": "initializing", "agent": "detailed_report_agent"}

    return {
        "status": "healthy",
        "agent": "detailed_report_agent",
        "capabilities": [
            "detailed_report",
            "visual_analysis",
            "technical_analysis",
            "comprehensive_analysis",
        ],
    }


@app.get("/agent.json")
async def get_agent_card():
    """Agent card with capabilities"""
    return {
        "name": "DetailedReportAgent",
        "description": "Type-safe detailed reports with visual and technical analysis",
        "url": f"http://localhost:{DEFAULT_PORT}",
        "version": "2.0.0",
        "protocol": "a2a",
        "protocol_version": "0.2.1",
        "capabilities": [
            "detailed_report",
            "visual_analysis",
            "technical_analysis",
            "comprehensive_analysis",
            "relationship_aware_reporting",
        ],
        "skills": (
            detailed_report_agent.get_agent_skills() if detailed_report_agent else []
        ),
    }


@app.post("/process")
async def process_task(task: dict):
    """Process report generation task via plain dict payload."""
    if not detailed_report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        data = task

        request = ReportRequest(
            query=data.get("query", ""),
            search_results=data.get("search_results", []),
            context=data.get("context", {}),
            report_type=data.get("report_type", "comprehensive"),
            include_visual_analysis=data.get("include_visual_analysis", True),
            include_technical_details=data.get("include_technical_details", True),
            include_recommendations=data.get("include_recommendations", True),
            max_results_to_analyze=data.get("max_results_to_analyze", 20),
        )

        result = await detailed_report_agent.generate_report(request)

        return {
            "status": "completed",
            "executive_summary": result.executive_summary,
            "detailed_findings": result.detailed_findings,
            "visual_analysis": result.visual_analysis,
            "technical_details": result.technical_details,
            "recommendations": result.recommendations,
            "confidence_assessment": result.confidence_assessment,
            "metadata": result.metadata,
            "thinking_process": {
                "content_analysis": result.thinking_phase.content_analysis,
                "technical_findings": result.thinking_phase.technical_findings,
                "patterns": result.thinking_phase.patterns_identified,
                "reasoning": result.thinking_phase.reasoning,
            },
        }

    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_report")
async def generate_report_direct(
    query: str,
    search_results: List[Dict[str, Any]],
    report_type: str = "comprehensive",
    include_visual_analysis: bool = True,
    include_technical_details: bool = True,
    include_recommendations: bool = True,
):
    """Direct report generation endpoint"""
    if not detailed_report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        request = ReportRequest(
            query=query,
            search_results=search_results,
            report_type=report_type,
            include_visual_analysis=include_visual_analysis,
            include_technical_details=include_technical_details,
            include_recommendations=include_recommendations,
        )

        result = await detailed_report_agent.generate_report(request)

        return {
            "executive_summary": result.executive_summary,
            "detailed_findings": result.detailed_findings,
            "visual_analysis": result.visual_analysis,
            "technical_details": result.technical_details,
            "recommendations": result.recommendations,
            "confidence_assessment": result.confidence_assessment,
            "metadata": result.metadata,
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detailed Report Agent Server")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Detailed Report Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
