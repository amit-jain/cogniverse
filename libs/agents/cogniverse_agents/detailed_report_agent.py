"""
Detailed Report Agent with full A2A support, VLM integration, and thinking phase.
Generates comprehensive detailed reports with visual and technical analysis.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy
import uvicorn
from cogniverse_core.agents.dspy_a2a_base import DSPyA2AAgentBase
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareAgentMixin
from cogniverse_core.common.vlm_interface import VLMInterface
from fastapi import FastAPI, HTTPException

# Enhanced routing support
from cogniverse_agents.routing_agent import RoutingDecision
from cogniverse_agents.tools.a2a_utils import DataPart, Task

logger = logging.getLogger(__name__)


# DSPy Signatures and Module
class ReportGenerationSignature(dspy.Signature):
    """Generate comprehensive detailed reports"""

    content = dspy.InputField(desc="Search results content to analyze")
    query = dspy.InputField(desc="Original user query")
    report_type = dspy.InputField(desc="Type of report: comprehensive, technical, analytical")

    executive_summary = dspy.OutputField(desc="Executive summary of findings")
    key_findings = dspy.OutputField(desc="Key findings (comma-separated)")
    recommendations = dspy.OutputField(desc="Recommendations (comma-separated)")
    confidence_score = dspy.OutputField(desc="Confidence in report quality (0.0-1.0)")


class ReportGenerationModule(dspy.Module):
    """DSPy module for intelligent report generation"""

    def __init__(self):
        super().__init__()
        self.report_generator = dspy.ChainOfThought(ReportGenerationSignature)

    def forward(self, content: str, query: str, report_type: str = "comprehensive"):
        """Generate report using DSPy"""
        try:
            result = self.report_generator(content=content, query=query, report_type=report_type)
            return result
        except Exception as e:
            logger.warning(f"DSPy report generation failed: {e}, using fallback")
            # Extract total results from content (first line: "Total Results: N")
            # If this fails, we have a bug - crash instead of returning misleading "0 results"
            total_results = int(content.split("\n")[0].split(":")[1].strip())

            # Fallback prediction with result count
            return dspy.Prediction(
                executive_summary=f"Analysis of {total_results} results for query: {query}",
                key_findings="Content analysis, Data patterns, Technical insights",
                recommendations="Further analysis recommended, Validate findings, Review methodology",
                confidence_score=0.5
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
class EnhancedReportRequest:
    """Enhanced report request with relationship context"""

    original_query: str
    enhanced_query: Optional[str]
    search_results: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    routing_metadata: Dict[str, Any]
    routing_confidence: float
    report_type: str = "comprehensive"
    include_visual_analysis: bool = True
    include_technical_details: bool = True
    include_recommendations: bool = True
    max_results_to_analyze: int = 20
    context: Optional[Dict[str, Any]] = None
    focus_on_relationships: bool = True


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

    @property
    def confidence_score(self) -> float:
        """Return overall confidence score for backward compatibility"""
        return self.confidence_assessment.get("overall", 0.0)


class DetailedReportAgent(DSPyA2AAgentBase, TenantAwareAgentMixin):
    """
    Agent for generating comprehensive detailed reports with full A2A support.
    Provides visual analysis, technical insights, and actionable recommendations.
    """

    def __init__(self, tenant_id: str, port: int = 8004, **kwargs):
        """
        Initialize detailed report agent

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            port: A2A server port
            **kwargs: Additional configuration options

        Raises:
            ValueError: If tenant_id is empty or None
        """
        # Initialize tenant support via TenantAwareAgentMixin
        TenantAwareAgentMixin.__init__(self, tenant_id=tenant_id)

        logger.info(f"Initializing DetailedReportAgent for tenant: {tenant_id}...")

        # Initialize DSPy components
        self._initialize_vlm_client()

        # Create DSPy report generation module
        self.report_module = ReportGenerationModule()

        # Initialize VLM for visual analysis
        self.vlm = VLMInterface(config_manager=self.config_manager, tenant_id=self.tenant_id)

        # Configuration
        self.max_report_length = kwargs.get("max_report_length", 2000)
        self.thinking_enabled = kwargs.get("thinking_enabled", True)
        self.visual_analysis_enabled = kwargs.get("visual_analysis_enabled", True)
        self.technical_analysis_enabled = kwargs.get("technical_analysis_enabled", True)

        # Initialize DSPyA2AAgentBase with report module
        DSPyA2AAgentBase.__init__(
            self,
            agent_name="detailed_report_agent",
            agent_description="Generates comprehensive detailed reports with visual and technical analysis",
            dspy_module=self.report_module,
            capabilities=[
                "detailed_report",
                "visual_analysis",
                "technical_analysis",
                "comprehensive_analysis",
                "relationship_aware_reporting",
            ],
            port=port,
        )

        logger.info("DetailedReportAgent initialization complete")

    def _initialize_vlm_client(self):
        """Initialize DSPy LM from configuration"""
        llm_config = self.config.get("llm", {})
        model_name = llm_config.get("model_name")
        base_url = llm_config.get("base_url")
        api_key = llm_config.get("api_key")

        if not all([model_name, base_url]):
            raise ValueError(
                "LLM configuration missing: model_name and base_url required"
            )

        # Ensure model name has provider prefix for litellm (Ollama models)
        if ("localhost:11434" in base_url or "11434" in base_url) and not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"

        try:
            if api_key:
                dspy.settings.configure(
                    lm=dspy.LM(model=model_name, api_base=base_url, api_key=api_key)
                )
            else:
                dspy.settings.configure(lm=dspy.LM(model=model_name, api_base=base_url))
            logger.info(f"Configured DSPy LM: {model_name} at {base_url}")
        except RuntimeError as e:
            if "can only be called from the same async task" in str(e):
                logger.warning("DSPy already configured in this async context, skipping reconfiguration")
            else:
                raise

    async def _generate_report(self, request: ReportRequest) -> ReportResult:
        """
        Internal: Generate a detailed report with comprehensive analysis

        Args:
            request: Report generation request

        Returns:
            Detailed report result with thinking phase
        """
        logger.info(f"Generating detailed report for: '{request.query}'")

        try:
            # Phase 1: Thinking phase - comprehensive analysis
            thinking_phase = await self._thinking_phase(request)

            # Phase 2: Visual analysis (if enabled)
            visual_analysis = await self._perform_visual_analysis(
                request, thinking_phase
            )

            # Phase 3: Generate executive summary
            executive_summary = await self._generate_executive_summary(
                request, thinking_phase
            )

            # Phase 4: Generate detailed findings
            detailed_findings = self._generate_detailed_findings(
                request, thinking_phase
            )

            # Phase 5: Generate technical details
            technical_details = self._generate_technical_details(
                request, thinking_phase
            )

            # Phase 6: Generate recommendations
            recommendations = self._generate_recommendations(request, thinking_phase)

            # Phase 7: Confidence assessment
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
                    "technical_analysis_enabled": request.include_technical_details,
                    "recommendations_enabled": request.include_recommendations,
                },
            )

            logger.info("Detailed report generation complete")
            return result

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise

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
        avg_score = sum(r.get("score", r.get("relevance", 0)) for r in results) / len(
            results
        ) if results else 0
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
            1 for r in results if not r.get("metadata") or len(r.get("metadata", {})) < 2
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
- Results: {content_analysis['total_results']} items analyzed
- Average relevance: {content_analysis['avg_relevance']:.2f}
- Visual content: {'Available' if visual_assessment['has_visual_content'] else 'Not available'}
- Technical findings: {len(technical_findings)} aspects identified
- Patterns detected: {len(patterns)}
- Limitations: {len(gaps)} gaps identified

Approach: Will generate {request.report_type} report with focus on data quality,
technical accuracy, and actionable insights. Visual analysis {'included' if request.include_visual_analysis else 'excluded'}.
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

                    visual_analysis.append({
                        "result_id": result.get("id", "unknown"),
                        "insights": analysis.get("insights", []),
                        "confidence": analysis.get("confidence", 0.0),
                    })
                except Exception as e:
                    logger.warning(f"Visual analysis failed for {image_path}: {e}")

        return visual_analysis

    async def _generate_executive_summary(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> str:
        """Generate executive summary using DSPy"""
        total_results = thinking_phase.content_analysis.get("total_results", len(request.search_results))

        content_parts = [f"Total Results: {total_results}"]
        for result in request.search_results[:10]:
            title = result.get("title", result.get("video_id", "Unknown"))
            score = result.get("score", result.get("relevance", 0))
            content_parts.append(f"- {title} (score: {score:.2f})")

        content_text = "\n".join(content_parts)

        try:
            dspy_result = self.report_module.forward(
                content=content_text,
                query=request.query,
                report_type=request.report_type
            )

            return dspy_result.executive_summary
        except Exception as e:
            logger.error(f"DSPy summary generation failed: {e}")
            # Fallback summary
            return f"Analysis of {len(request.search_results)} results for '{request.query}' with average relevance of {thinking_phase.content_analysis['avg_relevance']:.2f}."

    def _generate_detailed_findings(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Generate detailed findings section"""
        findings = []

        # Content analysis finding
        findings.append({
            "category": "Content Analysis",
            "finding": f"Analyzed {thinking_phase.content_analysis['total_results']} results",
            "details": thinking_phase.content_analysis,
            "significance": "high" if thinking_phase.content_analysis['avg_relevance'] > 0.7 else "medium"
        })

        # Pattern findings
        if thinking_phase.patterns_identified:
            findings.append({
                "category": "Patterns Identified",
                "finding": f"{len(thinking_phase.patterns_identified)} patterns detected",
                "details": thinking_phase.patterns_identified,
                "significance": "medium"
            })

        # Technical findings
        if thinking_phase.technical_findings:
            findings.append({
                "category": "Technical Analysis",
                "finding": f"{len(thinking_phase.technical_findings)} technical aspects identified",
                "details": thinking_phase.technical_findings,
                "significance": "medium"
            })

        return findings

    def _generate_technical_details(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Generate technical details section"""
        if not request.include_technical_details:
            return []

        technical_details = []

        # Content distribution
        technical_details.append({
            "category": "Content Distribution",
            "metrics": thinking_phase.content_analysis["content_types"],
            "analysis": "Distribution of content types across results"
        })

        # Quality metrics
        technical_details.append({
            "category": "Quality Metrics",
            "metrics": thinking_phase.content_analysis["quality_metrics"],
            "analysis": "Quality distribution of search results"
        })

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
            0.8
            if thinking_phase.visual_assessment["has_visual_content"]
            else 0.3
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

    async def generate_report_with_routing_decision(
        self,
        routing_decision: RoutingDecision,
        search_results: List[Dict[str, Any]],
        **kwargs,
    ) -> ReportResult:
        """Generate report with enhanced relationship context from DSPy routing"""
        logger.info(
            f"Relationship-aware report generation with confidence: {routing_decision.confidence:.3f}"
        )

        # Use enhanced query for basic report generation
        basic_request = ReportRequest(
            query=routing_decision.enhanced_query or routing_decision.routing_metadata.get("original_query", ""),
            search_results=search_results,
            report_type=kwargs.get("report_type", "comprehensive"),
            include_visual_analysis=kwargs.get("include_visual_analysis", True),
            include_technical_details=kwargs.get("include_technical_details", True),
            include_recommendations=kwargs.get("include_recommendations", True),
            max_results_to_analyze=kwargs.get("max_results_to_analyze", 20),
        )

        # Generate basic report
        result = await self._generate_report(basic_request)

        # Add relationship metadata
        result.enhancement_applied = True
        result.metadata.update({
            "original_query": routing_decision.routing_metadata.get("original_query", ""),
            "enhanced_query": routing_decision.enhanced_query,
            "entities_found": len(routing_decision.extracted_entities),
            "relationships_found": len(routing_decision.extracted_relationships),
            "routing_confidence": routing_decision.confidence,
        })

        return result

    # DSPyA2AAgentBase implementation
    async def _process(self, dspy_input: Dict[str, Any]) -> Any:
        """Process A2A input - performs report generation"""
        query = dspy_input.get("query", "")
        search_results = dspy_input.get("search_results", [])
        report_type = dspy_input.get("report_type", "comprehensive")
        include_visual = dspy_input.get("include_visual_analysis", True)
        include_technical = dspy_input.get("include_technical_details", True)
        include_recommendations = dspy_input.get("include_recommendations", True)
        max_results = dspy_input.get("max_results_to_analyze", 20)

        # Create report request
        request = ReportRequest(
            query=query,
            search_results=search_results,
            context=dspy_input.get("context", {}),
            report_type=report_type,
            include_visual_analysis=include_visual,
            include_technical_details=include_technical,
            include_recommendations=include_recommendations,
            max_results_to_analyze=max_results,
        )

        # Generate report
        result = await self._generate_report(request)

        return {
            "query": query,
            "executive_summary": result.executive_summary,
            "detailed_findings": result.detailed_findings,
            "visual_analysis": result.visual_analysis,
            "technical_details": result.technical_details,
            "recommendations": result.recommendations,
            "confidence_assessment": result.confidence_assessment,
            "thinking_process": {
                "content_analysis": result.thinking_phase.content_analysis,
                "visual_assessment": result.thinking_phase.visual_assessment,
                "technical_findings": result.thinking_phase.technical_findings,
                "patterns": result.thinking_phase.patterns_identified,
                "gaps": result.thinking_phase.gaps_and_limitations,
                "reasoning": result.thinking_phase.reasoning,
            },
            "metadata": result.metadata,
        }

    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """Convert DSPy report output to A2A format"""
        if isinstance(dspy_output, dict):
            return {
                "status": "success",
                "agent": self.agent_name,
                **dspy_output,
            }
        else:
            return {
                "status": "success",
                "agent": self.agent_name,
                "output": str(dspy_output),
            }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define detailed report agent skills for A2A protocol"""
        return [
            {
                "name": "generate_detailed_report",
                "description": "Generate comprehensive detailed reports with visual and technical analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "search_results": {"type": "array"},
                        "report_type": {"type": "string", "enum": ["comprehensive", "technical", "analytical"]},
                        "include_visual_analysis": {"type": "boolean", "default": True},
                        "include_technical_details": {"type": "boolean", "default": True},
                        "include_recommendations": {"type": "boolean", "default": True},
                        "max_results_to_analyze": {"type": "integer", "default": 20},
                    },
                    "required": ["query", "search_results"],
                },
            },
            {
                "name": "relationship_aware_report",
                "description": "Generate reports with relationship and entity context",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "routing_decision": {"type": "object"},
                        "search_results": {"type": "array"},
                        "report_type": {"type": "string"},
                    },
                    "required": ["routing_decision", "search_results"],
                },
            },
        ]


# --- FastAPI Server ---
app = FastAPI(
    title="Detailed Report Agent",
    description="Generates comprehensive detailed reports with visual and technical analysis",
    version="2.0.0",
)

# Global agent instance
detailed_report_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global detailed_report_agent

    # Tenant ID is REQUIRED
    tenant_id = os.getenv("TENANT_ID")
    if not tenant_id:
        error_msg = "TENANT_ID environment variable is required"
        logger.error(error_msg)
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise ValueError(error_msg)
        else:
            logger.warning("PYTEST_CURRENT_TEST detected - using 'test_tenant' as tenant_id")
            tenant_id = "test_tenant"

    try:
        detailed_report_agent = DetailedReportAgent(tenant_id=tenant_id)
        logger.info(f"Detailed report agent initialized for tenant: {tenant_id}")
    except Exception as e:
        logger.error(f"Failed to initialize detailed report agent: {e}")
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise


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
        "description": "Generates comprehensive detailed reports with visual and technical analysis",
        "url": "/process",
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
        "skills": detailed_report_agent._get_agent_skills() if detailed_report_agent else [],
    }


@app.post("/process")
async def process_task(task: Task):
    """Process report generation task"""
    if not detailed_report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Extract data from task
        if not task.messages:
            raise ValueError("Task contains no messages")

        last_message = task.messages[-1]
        data_part = next(
            (part for part in last_message.parts if isinstance(part, DataPart)), None
        )

        if not data_part:
            raise ValueError("No data part found in message")

        data = data_part.data

        # Create report request
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

        # Generate report
        result = await detailed_report_agent.generate_report(request)

        return {
            "task_id": task.id,
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


@app.post("/tasks/send")
async def handle_a2a_task(task: dict):
    """A2A protocol task handler"""
    if not detailed_report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Convert dict to Task object if needed
        from cogniverse_agents.tools.a2a_utils import Task

        if not isinstance(task, Task):
            task_obj = Task(**task)
        else:
            task_obj = task

        return await process_task(task_obj)

    except Exception as e:
        logger.error(f"A2A task processing failed: {e}")
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
        "--port", type=int, default=8004, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Detailed Report Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
