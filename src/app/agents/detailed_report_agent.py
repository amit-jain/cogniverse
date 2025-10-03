"""Detailed Report Agent with VLM integration and thinking phase."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy
from fastapi import FastAPI, HTTPException

from src.app.agents.dspy_integration_mixin import DSPyDetailedReportMixin

# Enhanced routing support
from src.app.agents.enhanced_routing_agent import RoutingDecision
from src.common.a2a_mixin import A2AEndpointsMixin
from src.common.config_compat import get_config  # DEPRECATED: Migrate to ConfigManager
from src.common.health_mixin import HealthCheckMixin
from src.common.vlm_interface import VLMInterface
from src.tools.a2a_utils import DataPart, Task

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Detailed Report Agent",
    description="Generates comprehensive detailed reports with visual analysis",
    version="1.0.0",
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


class DetailedReportAgent(DSPyDetailedReportMixin, A2AEndpointsMixin, HealthCheckMixin):
    """Agent for generating comprehensive detailed reports with VLM integration"""

    def __init__(self, **kwargs):
        """Initialize detailed report agent"""
        logger.info("Initializing DetailedReportAgent...")
        super().__init__()

        self.config = get_config()
        self._initialize_vlm_client()
        self.vlm = VLMInterface()

        # A2A agent metadata
        self.agent_name = "detailed_report_agent"
        self.agent_description = (
            "Generates comprehensive detailed reports with visual analysis"
        )
        self.agent_version = "1.0.0"
        self.agent_url = (
            f"http://localhost:{self.config.get('detailed_report_agent_port', 8003)}"
        )
        self.agent_capabilities = [
            "detailed_report",
            "visual_analysis",
            "technical_analysis",
            "comprehensive_analysis",
        ]
        self.agent_skills = [
            {
                "name": "generate_detailed_report",
                "description": "Generate comprehensive detailed reports with visual and technical analysis",
                "input_types": ["search_results", "query"],
                "output_types": ["detailed_report"],
            }
        ]

        # Configuration
        self.max_report_length = kwargs.get("max_report_length", 2000)
        self.thinking_enabled = kwargs.get("thinking_enabled", True)
        self.visual_analysis_enabled = kwargs.get("visual_analysis_enabled", True)
        self.technical_analysis_enabled = kwargs.get("technical_analysis_enabled", True)

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

        if api_key:
            dspy.settings.configure(
                lm=dspy.LM(model=model_name, api_base=base_url, api_key=api_key)
            )
        else:
            dspy.settings.configure(lm=dspy.LM(model=model_name, api_base=base_url))

        logger.info(f"Configured DSPy LM: {model_name} at {base_url}")

    async def generate_report(self, request: ReportRequest) -> ReportResult:
        """
        Generate a detailed report with comprehensive analysis

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
        visual_elements = {"thumbnails": 0, "keyframes": 0, "images": 0, "charts": 0}

        has_visual_content = False
        visual_quality_indicators = []

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

            # Check for visual quality indicators
            if result.get("video_quality"):
                visual_quality_indicators.append(result["video_quality"])

        return {
            "has_visual_content": has_visual_content,
            "visual_elements": visual_elements,
            "visual_coverage": sum(visual_elements.values()),
            "quality_indicators": visual_quality_indicators,
            "visual_analysis_feasible": has_visual_content
            and self.visual_analysis_enabled,
        }

    def _identify_technical_findings(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify technical aspects and findings"""
        findings = []

        # Video format analysis
        formats = set()
        resolutions = set()
        frame_rates = set()

        for result in results:
            if "format" in result:
                formats.add(result["format"])
            if "resolution" in result:
                resolutions.add(result["resolution"])
            if "frame_rate" in result:
                frame_rates.add(result["frame_rate"])

        if formats:
            findings.append(f"Video formats detected: {', '.join(formats)}")
        if resolutions:
            findings.append(f"Resolutions found: {', '.join(map(str, resolutions))}")
        if frame_rates:
            findings.append(f"Frame rates: {', '.join(map(str, frame_rates))} fps")

        # Embedding analysis
        embedding_dimensions = set()
        for result in results:
            if "embedding" in result and isinstance(result["embedding"], list):
                embedding_dimensions.add(len(result["embedding"]))

        if embedding_dimensions:
            findings.append(
                f"Embedding dimensions: {', '.join(map(str, embedding_dimensions))}"
            )

        return findings

    def _identify_patterns(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns and trends in the results"""
        patterns = []

        # Temporal patterns
        timestamps = [r.get("timestamp") for r in results if r.get("timestamp")]
        if timestamps:
            patterns.append(
                f"Content spans {len(set(timestamps))} distinct time periods"
            )

        # Content clustering
        titles = [r.get("title", "") for r in results]
        common_words = self._extract_common_terms(titles)
        if common_words:
            patterns.append(f"Common themes: {', '.join(common_words[:3])}")

        # Quality distribution
        scores = [r.get("score", r.get("relevance", 0)) for r in results]
        if scores:
            high_quality = sum(1 for s in scores if s > 0.8)
            if high_quality > len(scores) * 0.5:
                patterns.append("High concentration of high-quality results")
            elif high_quality < len(scores) * 0.2:
                patterns.append("Limited high-quality results available")

        return patterns

    def _identify_gaps_and_limitations(
        self, results: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify gaps and limitations in the data"""
        gaps = []

        # Missing metadata
        missing_fields = []
        common_fields = ["title", "description", "duration", "timestamp", "thumbnail"]

        for field in common_fields:
            missing_count = sum(1 for r in results if not r.get(field))
            if missing_count > len(results) * 0.3:
                missing_fields.append(field)

        if missing_fields:
            gaps.append(f"Incomplete metadata: missing {', '.join(missing_fields)}")

        # Content diversity
        unique_sources = len(set(r.get("source", "unknown") for r in results))
        if unique_sources < 3:
            gaps.append("Limited source diversity")

        # Temporal coverage
        if not any(r.get("timestamp") for r in results):
            gaps.append("No temporal information available")

        return gaps

    def _extract_common_terms(self, texts: List[str]) -> List[str]:
        """Extract common terms from text list"""
        import re
        from collections import Counter

        all_words = []
        for text in texts:
            if text:
                words = re.findall(r"\b\w+\b", text.lower())
                all_words.extend([w for w in words if len(w) > 3])

        if not all_words:
            return []

        common = Counter(all_words).most_common(5)
        return [word for word, count in common if count > 1]

    def _generate_thinking_reasoning(
        self,
        request: ReportRequest,
        content_analysis: Dict[str, Any],
        visual_assessment: Dict[str, Any],
        technical_findings: List[str],
        patterns: List[str],
        gaps: List[str],
    ) -> str:
        """Generate reasoning for the thinking phase"""
        total_results = content_analysis["total_results"]
        avg_relevance = content_analysis["avg_relevance"]
        has_visual = visual_assessment["has_visual_content"]

        reasoning = f"""
Detailed Report Analysis for: "{request.query}"

Content Overview:
- Total results: {total_results}
- Average relevance: {avg_relevance:.2f}
- Content types: {', '.join(content_analysis['content_types'].keys())}
- Visual content available: {'Yes' if has_visual else 'No'}

Technical Findings: {len(technical_findings)} identified
Patterns Identified: {len(patterns)} distinct patterns
Gaps/Limitations: {len(gaps)} areas of concern

Report Strategy:
- Will prioritize {request.report_type} analysis approach
- {'Include' if request.include_visual_analysis else 'Exclude'} visual analysis
- {'Include' if request.include_technical_details else 'Exclude'} technical details
- {'Include' if request.include_recommendations else 'Exclude'} recommendations

Analysis will focus on comprehensive coverage with attention to quality assessment
and actionable insights based on identified patterns and technical characteristics.
""".strip()

        return reasoning

    async def _perform_visual_analysis(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Perform detailed visual analysis using VLM"""
        if not self.visual_analysis_enabled or not request.include_visual_analysis:
            return []

        if not thinking_phase.visual_assessment["has_visual_content"]:
            return []

        logger.info("Performing detailed visual analysis...")

        # Extract image paths from results
        image_paths = []
        for result in request.search_results[: request.max_results_to_analyze]:
            if "thumbnail" in result:
                image_paths.append(result["thumbnail"])
            elif "image_path" in result:
                image_paths.append(result["image_path"])

        if not image_paths:
            return [{"analysis": "No visual content available for detailed analysis"}]

        try:
            visual_analysis = await self.vlm.analyze_visual_content_detailed(
                image_paths, request.query, str(thinking_phase.reasoning)
            )

            # Structure the visual analysis results
            structured_analysis = []

            for i, description in enumerate(
                visual_analysis.get("detailed_descriptions", [])[:5]
            ):
                structured_analysis.append(
                    {
                        "item_index": i,
                        "detailed_description": description,
                        "technical_assessment": (
                            visual_analysis.get("technical_analysis", [])[i]
                            if i < len(visual_analysis.get("technical_analysis", []))
                            else ""
                        ),
                        "quality_score": visual_analysis.get(
                            "quality_assessment", {}
                        ).get("overall", 0.0),
                    }
                )

            return structured_analysis

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return [{"analysis": "Visual analysis unavailable due to processing error"}]

    async def _generate_executive_summary(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> str:
        """Generate executive summary"""
        logger.info("Generating executive summary...")

        content_analysis = thinking_phase.content_analysis
        total_results = content_analysis["total_results"]
        avg_relevance = content_analysis["avg_relevance"]

        if total_results == 0:
            return f"No relevant results found for query '{request.query}'. Analysis could not be completed."

        key_patterns = thinking_phase.patterns_identified[:3]
        major_findings = thinking_phase.technical_findings[:2]

        summary = f"""
Executive Summary for "{request.query}"

Analyzed {total_results} results with an average relevance score of {avg_relevance:.2f}. 
The content analysis reveals {', '.join(content_analysis['content_types'].keys())} across 
{sum(content_analysis['duration_distribution'].values())} items.

Key findings include {len(thinking_phase.technical_findings)} technical observations 
and {len(thinking_phase.patterns_identified)} distinct patterns. 
{'Visual analysis was conducted on available content.' if thinking_phase.visual_assessment['has_visual_content'] else 'Limited visual content available for analysis.'}

{f'Primary patterns: {", ".join(key_patterns)}' if key_patterns else 'No significant patterns identified.'}
{f'Technical highlights: {", ".join(major_findings)}' if major_findings else ''}
""".strip()

        return summary

    def _generate_detailed_findings(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Generate detailed findings section"""
        findings = []

        # Content composition finding
        content_analysis = thinking_phase.content_analysis
        findings.append(
            {
                "category": "Content Composition",
                "finding": f"Analysis of {content_analysis['total_results']} results reveals diverse content types",
                "details": content_analysis["content_types"],
                "significance": (
                    "high" if len(content_analysis["content_types"]) > 2 else "medium"
                ),
            }
        )

        # Quality assessment finding
        quality_metrics = content_analysis["quality_metrics"]
        total_results = content_analysis["total_results"]
        if total_results > 0:
            high_quality_ratio = quality_metrics["high"] / total_results
            findings.append(
                {
                    "category": "Quality Assessment",
                    "finding": f"Content quality distribution shows {high_quality_ratio:.1%} high-quality results",
                    "details": quality_metrics,
                    "significance": "high" if high_quality_ratio > 0.6 else "medium",
                }
            )
        else:
            findings.append(
                {
                    "category": "Quality Assessment",
                    "finding": "No results available for quality assessment",
                    "details": quality_metrics,
                    "significance": "low",
                }
            )

        # Visual content finding
        if thinking_phase.visual_assessment["has_visual_content"]:
            visual_coverage = thinking_phase.visual_assessment["visual_coverage"]
            findings.append(
                {
                    "category": "Visual Content",
                    "finding": f"Comprehensive visual content available with {visual_coverage} visual elements",
                    "details": thinking_phase.visual_assessment["visual_elements"],
                    "significance": "high" if visual_coverage > 10 else "medium",
                }
            )

        # Pattern analysis finding
        if thinking_phase.patterns_identified:
            findings.append(
                {
                    "category": "Pattern Analysis",
                    "finding": f"Identified {len(thinking_phase.patterns_identified)} significant patterns",
                    "details": thinking_phase.patterns_identified,
                    "significance": "high",
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

        # System findings
        if thinking_phase.technical_findings:
            technical_details.append(
                {
                    "section": "System Analysis",
                    "details": thinking_phase.technical_findings,
                    "analysis": "Technical characteristics extracted from metadata analysis",
                }
            )

        # Content analysis
        content_analysis = thinking_phase.content_analysis
        content_details = [
            f"Total analyzed: {content_analysis['total_results']}",
            f"Quality distribution: {content_analysis['quality_metrics']}",
            f"Average relevance: {round(content_analysis['avg_relevance'], 3)}",
        ]
        technical_details.append(
            {
                "section": "Content Metrics",
                "details": content_details,
                "analysis": "Quantitative assessment of content quality and relevance",
            }
        )

        # Visual analysis
        if thinking_phase.visual_assessment["has_visual_content"]:
            technical_details.append(
                {
                    "section": "Visual Content Analysis",
                    "details": thinking_phase.visual_assessment["visual_elements"],
                    "analysis": f"Visual elements coverage: {thinking_phase.visual_assessment['visual_coverage']} items",
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

        # Quality-based recommendations
        content_analysis = thinking_phase.content_analysis
        total_results = content_analysis["total_results"]

        if total_results > 0:
            high_quality_ratio = (
                content_analysis["quality_metrics"]["high"] / total_results
            )
            if high_quality_ratio < 0.3:
                recommendations.append(
                    "Consider refining search criteria to improve result quality"
                )
        else:
            recommendations.append("Expand search scope to find relevant results")

        # Content diversity recommendations
        if len(content_analysis["content_types"]) == 1:
            recommendations.append(
                "Expand search to include diverse content types for comprehensive analysis"
            )

        # Visual content recommendations
        if not thinking_phase.visual_assessment["has_visual_content"]:
            recommendations.append(
                "Incorporate visual content sources to enhance analysis depth"
            )
        elif thinking_phase.visual_assessment["visual_coverage"] < 5:
            recommendations.append(
                "Increase visual content coverage for more comprehensive visual analysis"
            )

        # Gap-based recommendations
        for gap in thinking_phase.gaps_and_limitations:
            if "metadata" in gap.lower():
                recommendations.append(
                    "Improve metadata completeness for enhanced analysis capabilities"
                )
            elif "diversity" in gap.lower():
                recommendations.append(
                    "Diversify content sources to reduce bias and improve coverage"
                )

        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "Current analysis provides comprehensive coverage - maintain search approach"
            )

        return recommendations

    def _calculate_confidence_assessment(
        self, request: ReportRequest, thinking_phase: ThinkingPhase
    ) -> Dict[str, float]:
        """Calculate confidence assessment for different aspects"""
        content_analysis = thinking_phase.content_analysis

        # Content confidence based on quantity and quality
        content_confidence = min(
            0.9,
            (content_analysis["total_results"] / 20) * 0.6
            + content_analysis["avg_relevance"] * 0.4,
        )

        # Visual confidence based on visual content availability
        visual_confidence = 0.0
        if thinking_phase.visual_assessment["has_visual_content"]:
            visual_coverage = thinking_phase.visual_assessment["visual_coverage"]
            visual_confidence = min(0.9, visual_coverage / 10)

        # Technical confidence based on technical findings
        technical_confidence = min(0.9, len(thinking_phase.technical_findings) / 5)

        # Overall confidence as weighted average
        overall_confidence = (
            content_confidence * 0.4
            + visual_confidence * 0.3
            + technical_confidence * 0.3
        )

        return {
            "overall": round(overall_confidence, 2),
            "content_analysis": round(content_confidence, 2),
            "visual_analysis": round(visual_confidence, 2),
            "technical_analysis": round(technical_confidence, 2),
        }

    async def generate_report_with_routing_decision(
        self,
        routing_decision: RoutingDecision,
        search_results: List[Dict[str, Any]],
        **kwargs,
    ) -> ReportResult:
        """
        Generate detailed report with routing decision context

        Args:
            routing_decision: Routing decision with entities and relationships
            search_results: Search results to analyze
            **kwargs: Additional parameters for report generation

        Returns:
            Enhanced report result with relationship analysis
        """
        try:
            # Create enhanced request from routing decision
            enhanced_request = EnhancedReportRequest(
                original_query=routing_decision.query,
                enhanced_query=routing_decision.enhanced_query,
                search_results=search_results,
                entities=routing_decision.entities,
                relationships=routing_decision.relationships,
                routing_metadata=routing_decision.metadata,
                routing_confidence=routing_decision.confidence,
                report_type=kwargs.get("report_type", "comprehensive"),
                include_visual_analysis=kwargs.get("include_visual_analysis", True),
                include_technical_details=kwargs.get("include_technical_details", True),
                include_recommendations=kwargs.get("include_recommendations", True),
                max_results_to_analyze=kwargs.get("max_results_to_analyze", 20),
                context=kwargs.get("context"),
                focus_on_relationships=kwargs.get("focus_on_relationships", True),
            )

            # Generate enhanced report
            return await self.generate_enhanced_report(enhanced_request)

        except Exception as e:
            logger.error(f"Enhanced report generation failed: {e}")
            # Fallback to standard report generation
            fallback_request = ReportRequest(
                query=routing_decision.query, search_results=search_results, **kwargs
            )
            result = await self.generate_report(fallback_request)
            result.enhancement_applied = False
            return result

    async def generate_enhanced_report(
        self, request: EnhancedReportRequest
    ) -> ReportResult:
        """
        Generate enhanced detailed report with relationship context

        Args:
            request: Enhanced report generation request with relationship context

        Returns:
            Detailed report result with relationship analysis
        """
        logger.info(
            f"Generating enhanced detailed report for: '{request.original_query}'"
        )

        try:
            # Phase 1: Enhanced thinking phase with relationship analysis
            thinking_phase = await self._enhanced_thinking_phase(request)

            # Phase 2: Visual analysis with relationship context
            visual_analysis = await self._enhanced_visual_analysis(
                request, thinking_phase
            )

            # Phase 3: Enhanced executive summary
            executive_summary = await self._generate_enhanced_executive_summary(
                request, thinking_phase
            )

            # Phase 4: Enhanced detailed findings with relationship insights
            detailed_findings = self._generate_enhanced_detailed_findings(
                request, thinking_phase
            )

            # Phase 5: Enhanced technical details
            technical_details = self._generate_enhanced_technical_details(
                request, thinking_phase
            )

            # Phase 6: Relationship-aware recommendations
            recommendations = self._generate_relationship_aware_recommendations(
                request, thinking_phase
            )

            # Phase 7: Enhanced confidence assessment
            confidence_assessment = self._calculate_enhanced_confidence_assessment(
                request, thinking_phase
            )

            # Phase 8: Generate relationship and entity analysis
            relationship_analysis = self._analyze_relationships_in_report(request)
            entity_analysis = self._analyze_entities_in_report(request)

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
                    "enhanced_with_relationships": True,
                    "entities_identified": len(request.entities),
                    "relationships_identified": len(request.relationships),
                    "routing_confidence": request.routing_confidence,
                },
                relationship_analysis=relationship_analysis,
                entity_analysis=entity_analysis,
                enhancement_applied=True,
            )

            logger.info("Enhanced detailed report generation complete")
            return result

        except Exception as e:
            logger.error(f"Enhanced report generation failed: {e}")
            # Fallback to standard report generation
            fallback_request = ReportRequest(
                query=request.original_query,
                search_results=request.search_results,
                report_type=request.report_type,
                include_visual_analysis=request.include_visual_analysis,
                include_technical_details=request.include_technical_details,
                include_recommendations=request.include_recommendations,
                max_results_to_analyze=request.max_results_to_analyze,
                context=request.context,
            )
            result = await self.generate_report(fallback_request)
            result.enhancement_applied = False
            return result

    async def _enhanced_thinking_phase(
        self, request: EnhancedReportRequest
    ) -> ThinkingPhase:
        """Enhanced thinking phase with relationship analysis"""
        logger.info("Starting enhanced thinking phase with relationship analysis...")

        # Standard analysis
        content_analysis = self._analyze_content_structure(request.search_results)
        visual_assessment = self._assess_visual_content(request.search_results)
        technical_findings = self._identify_technical_findings(request.search_results)
        patterns_identified = self._identify_patterns(request.search_results)
        gaps_and_limitations = self._identify_gaps_and_limitations(
            request.search_results
        )

        # Enhanced analysis with relationships
        if request.focus_on_relationships and request.relationships:
            # Add relationship-specific patterns
            relationship_patterns = self._identify_relationship_patterns(request)
            patterns_identified.extend(relationship_patterns)

            # Add entity-specific technical findings
            entity_findings = self._identify_entity_specific_findings(request)
            technical_findings.extend(entity_findings)

        # Enhanced reasoning with relationship context
        reasoning = self._generate_enhanced_thinking_reasoning(
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

    def _identify_relationship_patterns(
        self, request: EnhancedReportRequest
    ) -> List[str]:
        """Identify patterns based on relationships in the query"""
        patterns = []

        # Group relationships by type
        relationship_types = {}
        for rel in request.relationships:
            rel_type = rel.get("relation", "unknown")
            if rel_type not in relationship_types:
                relationship_types[rel_type] = []
            relationship_types[rel_type].append(rel)

        if relationship_types:
            patterns.append(
                f"Relationship analysis reveals {len(relationship_types)} distinct relationship types"
            )

            # Identify dominant relationship types
            dominant_relations = sorted(
                relationship_types.items(), key=lambda x: len(x[1]), reverse=True
            )[:3]
            for rel_type, rels in dominant_relations:
                patterns.append(
                    f"Dominant relationship pattern: {rel_type} ({len(rels)} instances)"
                )

        # Identify entity connection patterns
        entities_in_relations = set()
        for rel in request.relationships:
            entities_in_relations.add(rel.get("subject", ""))
            entities_in_relations.add(rel.get("object", ""))

        if entities_in_relations and request.entities:
            connected_entities = len(
                [
                    e
                    for e in request.entities
                    if e.get("text", "") in entities_in_relations
                ]
            )
            patterns.append(
                f"Entity connectivity: {connected_entities}/{len(request.entities)} entities participate in relationships"
            )

        return patterns

    def _identify_entity_specific_findings(
        self, request: EnhancedReportRequest
    ) -> List[str]:
        """Identify technical findings specific to extracted entities"""
        findings = []

        # Analyze entity types
        entity_types = {}
        for entity in request.entities:
            entity_type = entity.get("label", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        if entity_types:
            findings.append(
                f"Entity type distribution: {', '.join([f'{k}: {v}' for k, v in entity_types.items()])}"
            )

        # Check for high-confidence entities
        high_conf_entities = [
            e for e in request.entities if e.get("confidence", 0) > 0.8
        ]
        if high_conf_entities:
            findings.append(
                f"High-confidence entity extraction: {len(high_conf_entities)}/{len(request.entities)} entities"
            )

        return findings

    def _generate_enhanced_thinking_reasoning(
        self,
        request: EnhancedReportRequest,
        content_analysis: Dict[str, Any],
        visual_assessment: Dict[str, Any],
        technical_findings: List[str],
        patterns: List[str],
        gaps: List[str],
    ) -> str:
        """Generate enhanced reasoning with relationship context"""
        total_results = content_analysis["total_results"]
        avg_relevance = content_analysis["avg_relevance"]
        has_visual = visual_assessment["has_visual_content"]

        # Enhanced query analysis
        query_enhancement_info = ""
        if request.enhanced_query and request.enhanced_query != request.original_query:
            query_enhancement_info = f"""
Query Enhancement Applied:
- Original: "{request.original_query}"
- Enhanced: "{request.enhanced_query}"
- Routing confidence: {request.routing_confidence:.2f}
"""

        reasoning = f"""
Enhanced Detailed Report Analysis for: "{request.original_query}"
{query_enhancement_info}
Content Overview:
- Total results: {total_results}
- Average relevance: {avg_relevance:.2f}
- Content types: {', '.join(content_analysis['content_types'].keys())}
- Visual content available: {'Yes' if has_visual else 'No'}

Relationship Context:
- Entities identified: {len(request.entities)}
- Relationships extracted: {len(request.relationships)}
- Relationship focus: {'Enabled' if request.focus_on_relationships else 'Disabled'}

Technical Findings: {len(technical_findings)} identified
Patterns Identified: {len(patterns)} distinct patterns (including relationship patterns)
Gaps/Limitations: {len(gaps)} areas of concern

Report Strategy:
- Will prioritize {request.report_type} analysis with relationship context
- {'Include' if request.include_visual_analysis else 'Exclude'} visual analysis
- {'Include' if request.include_technical_details else 'Exclude'} technical details
- {'Include' if request.include_recommendations else 'Exclude'} relationship-aware recommendations

Analysis will focus on comprehensive coverage with special attention to entity relationships
and contextual connections identified through the routing decision process.
""".strip()

        return reasoning

    async def _enhanced_visual_analysis(
        self, request: EnhancedReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Enhanced visual analysis with relationship context"""
        if not self.visual_analysis_enabled or not request.include_visual_analysis:
            return []

        if not thinking_phase.visual_assessment["has_visual_content"]:
            return []

        # Perform standard visual analysis
        visual_analysis = await self._perform_visual_analysis(
            ReportRequest(
                query=request.original_query,
                search_results=request.search_results,
                include_visual_analysis=request.include_visual_analysis,
                max_results_to_analyze=request.max_results_to_analyze,
            ),
            thinking_phase,
        )

        # Enhance with relationship context
        if request.focus_on_relationships and request.entities and visual_analysis:
            for analysis_item in visual_analysis:
                # Add entity relevance to visual analysis
                if "detailed_description" in analysis_item:
                    entity_matches = self._find_entity_matches_in_text(
                        analysis_item["detailed_description"], request.entities
                    )
                    if entity_matches:
                        analysis_item["entity_relevance"] = {
                            "matched_entities": entity_matches,
                            "relevance_boost": min(0.2, len(entity_matches) * 0.05),
                        }

        return visual_analysis

    async def _generate_enhanced_executive_summary(
        self, request: EnhancedReportRequest, thinking_phase: ThinkingPhase
    ) -> str:
        """Generate enhanced executive summary with relationship context"""
        # Generate base summary
        base_summary = await self._generate_executive_summary(
            ReportRequest(
                query=request.original_query,
                search_results=request.search_results,
                report_type=request.report_type,
                include_visual_analysis=request.include_visual_analysis,
                include_technical_details=request.include_technical_details,
                include_recommendations=request.include_recommendations,
                max_results_to_analyze=request.max_results_to_analyze,
                context=request.context,
            ),
            thinking_phase,
        )

        # Add relationship context
        if request.focus_on_relationships and (
            request.entities or request.relationships
        ):
            relationship_summary = f"""

Relationship Analysis Context:
This analysis incorporates {len(request.entities)} identified entities and {len(request.relationships)} relationships from the query. """

            if (
                request.enhanced_query
                and request.enhanced_query != request.original_query
            ):
                relationship_summary += f"Query enhancement was applied with {request.routing_confidence:.1%} confidence to improve retrieval relevance. "

            # Add dominant relationship insights
            if request.relationships:
                rel_types = {}
                for rel in request.relationships:
                    rel_type = rel.get("relation", "unknown")
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

                if rel_types:
                    dominant_rel = max(rel_types.items(), key=lambda x: x[1])
                    relationship_summary += f"The dominant relationship pattern is '{dominant_rel[0]}' with contextual connections enhancing result interpretation."

            base_summary += relationship_summary

        return base_summary

    def _generate_enhanced_detailed_findings(
        self, request: EnhancedReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Generate enhanced detailed findings with relationship insights"""
        # Get base findings
        base_findings = self._generate_detailed_findings(
            ReportRequest(
                query=request.original_query,
                search_results=request.search_results,
                report_type=request.report_type,
                max_results_to_analyze=request.max_results_to_analyze,
            ),
            thinking_phase,
        )

        # Add relationship-specific findings
        if request.focus_on_relationships:
            # Entity analysis finding
            if request.entities:
                entity_types = {}
                high_conf_entities = 0
                for entity in request.entities:
                    entity_type = entity.get("label", "unknown")
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                    if entity.get("confidence", 0) > 0.8:
                        high_conf_entities += 1

                base_findings.append(
                    {
                        "category": "Entity Analysis",
                        "finding": f"Identified {len(request.entities)} entities with {high_conf_entities} high-confidence extractions",
                        "details": {
                            "entity_types": entity_types,
                            "high_confidence_ratio": (
                                high_conf_entities / len(request.entities)
                                if request.entities
                                else 0
                            ),
                        },
                        "significance": (
                            "high"
                            if high_conf_entities / len(request.entities) > 0.7
                            else "medium"
                        ),
                    }
                )

            # Relationship analysis finding
            if request.relationships:
                rel_types = {}
                for rel in request.relationships:
                    rel_type = rel.get("relation", "unknown")
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

                base_findings.append(
                    {
                        "category": "Relationship Analysis",
                        "finding": f"Extracted {len(request.relationships)} relationships across {len(rel_types)} relationship types",
                        "details": {
                            "relationship_types": rel_types,
                            "complexity_score": (
                                len(rel_types) / len(request.relationships)
                                if request.relationships
                                else 0
                            ),
                        },
                        "significance": "high",
                    }
                )

            # Query enhancement finding
            if (
                request.enhanced_query
                and request.enhanced_query != request.original_query
            ):
                base_findings.append(
                    {
                        "category": "Query Enhancement",
                        "finding": f"Applied query enhancement with {request.routing_confidence:.1%} confidence",
                        "details": {
                            "original_query": request.original_query,
                            "enhanced_query": request.enhanced_query,
                            "confidence": request.routing_confidence,
                            "routing_metadata": request.routing_metadata,
                        },
                        "significance": (
                            "high" if request.routing_confidence > 0.8 else "medium"
                        ),
                    }
                )

        return base_findings

    def _generate_enhanced_technical_details(
        self, request: EnhancedReportRequest, thinking_phase: ThinkingPhase
    ) -> List[Dict[str, Any]]:
        """Generate enhanced technical details with relationship context"""
        # Get base technical details
        base_details = self._generate_technical_details(
            ReportRequest(
                query=request.original_query,
                search_results=request.search_results,
                include_technical_details=request.include_technical_details,
            ),
            thinking_phase,
        )

        # Add relationship-specific technical details
        if request.focus_on_relationships and request.include_technical_details:
            # Routing analysis
            routing_details = [
                f"Routing confidence: {request.routing_confidence:.3f}",
                f"Entities extracted: {len(request.entities)}",
                f"Relationships identified: {len(request.relationships)}",
                f"Query enhancement: {'Applied' if request.enhanced_query != request.original_query else 'Not applied'}",
            ]

            base_details.append(
                {
                    "section": "Routing Analysis",
                    "details": routing_details,
                    "analysis": "Enhanced routing system applied with relationship extraction and query enhancement",
                }
            )

            # Entity distribution analysis
            if request.entities:
                entity_analysis = {}
                for entity in request.entities:
                    entity_type = entity.get("label", "unknown")
                    if entity_type not in entity_analysis:
                        entity_analysis[entity_type] = {
                            "count": 0,
                            "avg_confidence": 0,
                            "confidences": [],
                        }
                    entity_analysis[entity_type]["count"] += 1
                    entity_analysis[entity_type]["confidences"].append(
                        entity.get("confidence", 0)
                    )

                # Calculate averages
                for entity_type in entity_analysis:
                    confidences = entity_analysis[entity_type]["confidences"]
                    entity_analysis[entity_type]["avg_confidence"] = sum(
                        confidences
                    ) / len(confidences)

                entity_details = [
                    f"{etype}: {data['count']} instances (avg conf: {data['avg_confidence']:.2f})"
                    for etype, data in entity_analysis.items()
                ]

                base_details.append(
                    {
                        "section": "Entity Distribution",
                        "details": entity_details,
                        "analysis": f"Comprehensive entity analysis across {len(entity_analysis)} entity types",
                    }
                )

        return base_details

    def _generate_relationship_aware_recommendations(
        self, request: EnhancedReportRequest, thinking_phase: ThinkingPhase
    ) -> List[str]:
        """Generate recommendations with relationship context"""
        # Get base recommendations
        base_recommendations = self._generate_recommendations(
            ReportRequest(
                query=request.original_query,
                search_results=request.search_results,
                include_recommendations=request.include_recommendations,
            ),
            thinking_phase,
        )

        # Add relationship-specific recommendations
        if request.focus_on_relationships and request.include_recommendations:
            # Entity-based recommendations
            if request.entities:
                low_conf_entities = [
                    e for e in request.entities if e.get("confidence", 0) < 0.6
                ]
                if len(low_conf_entities) > len(request.entities) * 0.3:
                    base_recommendations.append(
                        "Consider refining query terms to improve entity extraction confidence"
                    )

            # Relationship-based recommendations
            if request.relationships:
                if len(request.relationships) < 2:
                    base_recommendations.append(
                        "Expand query complexity to capture more relationship patterns"
                    )
                else:
                    # Check relationship diversity
                    rel_types = set(
                        rel.get("relation", "unknown") for rel in request.relationships
                    )
                    if len(rel_types) == 1:
                        base_recommendations.append(
                            "Diversify relationship types in queries for more comprehensive analysis"
                        )

            # Query enhancement recommendations
            if request.routing_confidence < 0.7:
                base_recommendations.append(
                    "Consider more specific query terms to improve routing confidence and result relevance"
                )
            elif request.enhanced_query == request.original_query:
                base_recommendations.append(
                    "Current query is well-structured for relationship extraction - maintain approach"
                )

        return base_recommendations

    def _calculate_enhanced_confidence_assessment(
        self, request: EnhancedReportRequest, thinking_phase: ThinkingPhase
    ) -> Dict[str, float]:
        """Calculate enhanced confidence assessment with relationship factors"""
        # Get base confidence
        base_confidence = self._calculate_confidence_assessment(
            ReportRequest(
                query=request.original_query, search_results=request.search_results
            ),
            thinking_phase,
        )

        # Add relationship confidence factors
        if request.focus_on_relationships:
            # Entity extraction confidence
            entity_confidence = 0.0
            if request.entities:
                entity_confidences = [e.get("confidence", 0) for e in request.entities]
                entity_confidence = sum(entity_confidences) / len(entity_confidences)

            # Relationship confidence (based on routing confidence)
            relationship_confidence = request.routing_confidence

            # Query enhancement confidence
            enhancement_confidence = (
                0.8 if request.enhanced_query != request.original_query else 0.5
            )

            # Calculate enhanced overall confidence
            enhanced_overall = (
                base_confidence["overall"] * 0.5
                + entity_confidence * 0.2
                + relationship_confidence * 0.2
                + enhancement_confidence * 0.1
            )

            base_confidence.update(
                {
                    "overall": round(min(enhanced_overall, 0.95), 2),
                    "entity_extraction": round(entity_confidence, 2),
                    "relationship_analysis": round(relationship_confidence, 2),
                    "query_enhancement": round(enhancement_confidence, 2),
                }
            )

        return base_confidence

    def _analyze_relationships_in_report(
        self, request: EnhancedReportRequest
    ) -> Dict[str, Any]:
        """Analyze relationships for report context"""
        if not request.relationships:
            return {"relationships_found": 0, "analysis": "No relationships extracted"}

        # Group relationships by type
        rel_types = {}
        for rel in request.relationships:
            rel_type = rel.get("relation", "unknown")
            if rel_type not in rel_types:
                rel_types[rel_type] = []
            rel_types[rel_type].append(rel)

        # Find most connected entities
        entity_connections = {}
        for rel in request.relationships:
            subject = rel.get("subject", "")
            object_entity = rel.get("object", "")

            entity_connections[subject] = entity_connections.get(subject, 0) + 1
            entity_connections[object_entity] = (
                entity_connections.get(object_entity, 0) + 1
            )

        most_connected = sorted(
            entity_connections.items(), key=lambda x: x[1], reverse=True
        )[:3]

        return {
            "relationships_found": len(request.relationships),
            "relationship_types": list(rel_types.keys()),
            "type_distribution": {k: len(v) for k, v in rel_types.items()},
            "most_connected_entities": dict(most_connected),
            "complexity_score": (
                len(rel_types) / len(request.relationships)
                if request.relationships
                else 0
            ),
        }

    def _analyze_entities_in_report(
        self, request: EnhancedReportRequest
    ) -> Dict[str, Any]:
        """Analyze entities for report context"""
        if not request.entities:
            return {"entities_found": 0, "analysis": "No entities extracted"}

        # Analyze entity types and confidence
        entity_types = {}
        confidence_sum = 0
        high_conf_count = 0

        for entity in request.entities:
            entity_type = entity.get("label", "unknown")
            confidence = entity.get("confidence", 0)

            if entity_type not in entity_types:
                entity_types[entity_type] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "confidences": [],
                }

            entity_types[entity_type]["count"] += 1
            entity_types[entity_type]["confidences"].append(confidence)
            confidence_sum += confidence

            if confidence > 0.8:
                high_conf_count += 1

        # Calculate averages
        for entity_type in entity_types:
            confidences = entity_types[entity_type]["confidences"]
            entity_types[entity_type]["avg_confidence"] = sum(confidences) / len(
                confidences
            )

        return {
            "entities_found": len(request.entities),
            "entity_types": list(entity_types.keys()),
            "type_distribution": {k: v["count"] for k, v in entity_types.items()},
            "average_confidence": confidence_sum / len(request.entities),
            "high_confidence_entities": high_conf_count,
            "confidence_ratio": high_conf_count / len(request.entities),
        }

    def _find_entity_matches_in_text(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find entity matches in given text"""
        matches = []
        text_lower = text.lower()

        for entity in entities:
            entity_text = entity.get("text", "").lower()
            if entity_text and entity_text in text_lower:
                matches.append(
                    {
                        "entity": entity_text,
                        "type": entity.get("label", "unknown"),
                        "confidence": entity.get("confidence", 0),
                    }
                )

        return matches

    async def process_a2a_task(self, task: Task) -> Dict[str, Any]:
        """Process A2A task for detailed report generation"""
        if not task.messages:
            raise HTTPException(status_code=400, detail="No messages in task")

        # Extract report request from last message
        last_message = task.messages[-1]
        data_part = next(
            (part for part in last_message.parts if isinstance(part, DataPart)), None
        )

        if not data_part:
            raise HTTPException(status_code=400, detail="No data in message")

        request_data = data_part.data

        # Check if this is an enhanced request with routing decision
        if "routing_decision" in request_data:
            routing_decision = RoutingDecision(**request_data["routing_decision"])
            search_results = request_data.get("search_results", [])

            # Generate enhanced report
            result = await self.generate_report_with_routing_decision(
                routing_decision, search_results, **request_data
            )
        else:
            # Create standard report request
            report_request = ReportRequest(
                query=request_data.get("query", ""),
                search_results=request_data.get("search_results", []),
                report_type=request_data.get("report_type", "comprehensive"),
                include_visual_analysis=request_data.get(
                    "include_visual_analysis", True
                ),
                include_technical_details=request_data.get(
                    "include_technical_details", True
                ),
                include_recommendations=request_data.get(
                    "include_recommendations", True
                ),
                max_results_to_analyze=request_data.get("max_results_to_analyze", 20),
                context=request_data.get("context"),
            )

            # Generate standard report
            result = await self.generate_report(report_request)

        # Convert result to dict for A2A response
        response_data = {
            "executive_summary": result.executive_summary,
            "detailed_findings": result.detailed_findings,
            "visual_analysis": result.visual_analysis,
            "technical_details": result.technical_details,
            "recommendations": result.recommendations,
            "confidence_assessment": result.confidence_assessment,
            "metadata": result.metadata,
        }

        # Add enhanced fields if present
        if result.relationship_analysis:
            response_data["relationship_analysis"] = result.relationship_analysis
        if result.entity_analysis:
            response_data["entity_analysis"] = result.entity_analysis
        if hasattr(result, "enhancement_applied"):
            response_data["enhancement_applied"] = result.enhancement_applied

        return {
            "task_id": task.id,
            "status": "completed",
            "result": response_data,
        }


# Global agent instance
report_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global report_agent

    try:
        report_agent = DetailedReportAgent()

        # Setup A2A standard endpoints
        report_agent.setup_a2a_endpoints(app)

        # Setup health endpoint (mixin provides implementation)
        report_agent.setup_health_endpoint(app)

        logger.info("Detailed report agent initialized with A2A endpoints")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


@app.post("/generate")
async def generate_report_endpoint(request: ReportRequest):
    """Generate standard detailed report"""
    if not report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = await report_agent.generate_report(request)
        return result

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/enhanced")
async def generate_enhanced_report_endpoint(request: EnhancedReportRequest):
    """Generate enhanced detailed report with relationship context"""
    if not report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = await report_agent.generate_enhanced_report(request)
        return result

    except Exception as e:
        logger.error(f"Enhanced report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/routing-decision")
async def generate_report_with_routing_decision_endpoint(
    routing_decision: dict, search_results: List[Dict[str, Any]], **kwargs
):
    """Generate detailed report from routing decision"""
    if not report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        from src.app.agents.enhanced_routing_agent import RoutingDecision

        routing_obj = RoutingDecision(**routing_decision)

        result = await report_agent.generate_report_with_routing_decision(
            routing_obj, search_results, **kwargs
        )
        return result

    except Exception as e:
        logger.error(f"Routing-based report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_task(task: Task):
    """Process detailed report task"""
    if not report_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = await report_agent.process_a2a_task(task)
        return result

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
