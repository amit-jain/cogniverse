"""Detailed Report Agent with VLM integration and thinking phase."""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.common.config import get_config
from src.tools.a2a_utils import A2AMessage, DataPart, Task

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


class VLMInterface:
    """Interface for Vision Language Model interactions"""
    
    def __init__(self, model_name: str = "gpt-4-vision-preview"):
        self.model_name = model_name
        self.config = get_config()
        
        # Initialize VLM client (could be OpenAI, Anthropic, etc.)
        self._initialize_vlm_client()
    
    def _initialize_vlm_client(self):
        """Initialize the VLM client"""
        try:
            # Try to use OpenAI client if available
            import openai
            api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self.client_type = "openai"
                logger.info("VLM client initialized with OpenAI")
                return
        except ImportError:
            logger.debug("OpenAI client not available")
        
        try:
            # Try Anthropic client
            import anthropic
            api_key = self.config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.client_type = "anthropic"
                logger.info("VLM client initialized with Anthropic")
                return
        except ImportError:
            logger.debug("Anthropic client not available")
        
        # Fallback to mock client for testing
        self.client = None
        self.client_type = "mock"
        logger.warning("Using mock VLM client - no API keys found")
    
    async def analyze_visual_content_detailed(self, image_paths: List[str], 
                                            query: str, context: str = "") -> Dict[str, Any]:
        """Perform detailed visual analysis"""
        if self.client_type == "mock" or not self.client:
            return {
                "detailed_descriptions": [f"Mock detailed description for {path}" for path in image_paths[:3]],
                "technical_analysis": ["Mock technical finding 1", "Mock technical finding 2"],
                "visual_patterns": ["Pattern A", "Pattern B"],
                "quality_assessment": {"overall": 0.85, "clarity": 0.9, "relevance": 0.8},
                "annotations": [{"element": "mock_element", "confidence": 0.9}]
            }
        
        # Real implementation would use the actual VLM client
        logger.info(f"Analyzing {len(image_paths)} images with detailed VLM analysis")
        return {
            "detailed_descriptions": [],
            "technical_analysis": [],
            "visual_patterns": [],
            "quality_assessment": {},
            "annotations": []
        }


class DetailedReportAgent:
    """Agent for generating comprehensive detailed reports with VLM integration"""
    
    def __init__(self, **kwargs):
        """Initialize detailed report agent"""
        logger.info("Initializing DetailedReportAgent...")
        
        self.config = get_config()
        self.vlm = VLMInterface(kwargs.get("vlm_model", "gpt-4-vision-preview"))
        
        # Configuration
        self.max_report_length = kwargs.get("max_report_length", 2000)
        self.thinking_enabled = kwargs.get("thinking_enabled", True)
        self.visual_analysis_enabled = kwargs.get("visual_analysis_enabled", True)
        self.technical_analysis_enabled = kwargs.get("technical_analysis_enabled", True)
        
        logger.info("DetailedReportAgent initialization complete")
    
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
            visual_analysis = await self._perform_visual_analysis(request, thinking_phase)
            
            # Phase 3: Generate executive summary
            executive_summary = await self._generate_executive_summary(request, thinking_phase)
            
            # Phase 4: Generate detailed findings
            detailed_findings = self._generate_detailed_findings(request, thinking_phase)
            
            # Phase 5: Generate technical details
            technical_details = self._generate_technical_details(request, thinking_phase)
            
            # Phase 6: Generate recommendations
            recommendations = self._generate_recommendations(request, thinking_phase)
            
            # Phase 7: Confidence assessment
            confidence_assessment = self._calculate_confidence_assessment(request, thinking_phase)
            
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
                    "recommendations_enabled": request.include_recommendations
                }
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
        gaps_and_limitations = self._identify_gaps_and_limitations(request.search_results)
        
        # Generate reasoning
        reasoning = self._generate_thinking_reasoning(
            request, content_analysis, visual_assessment, 
            technical_findings, patterns_identified, gaps_and_limitations
        )
        
        return ThinkingPhase(
            content_analysis=content_analysis,
            visual_assessment=visual_assessment,
            technical_findings=technical_findings,
            patterns_identified=patterns_identified,
            gaps_and_limitations=gaps_and_limitations,
            reasoning=reasoning
        )
    
    def _analyze_content_structure(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            "avg_relevance": sum(r.get("score", r.get("relevance", 0)) for r in results) / len(results) if results else 0
        }
    
    def _assess_visual_content(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess visual content availability and characteristics"""
        visual_elements = {
            "thumbnails": 0,
            "keyframes": 0,
            "images": 0,
            "charts": 0
        }
        
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
            "visual_analysis_feasible": has_visual_content and self.visual_analysis_enabled
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
            findings.append(f"Embedding dimensions: {', '.join(map(str, embedding_dimensions))}")
        
        return findings
    
    def _identify_patterns(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns and trends in the results"""
        patterns = []
        
        # Temporal patterns
        timestamps = [r.get("timestamp") for r in results if r.get("timestamp")]
        if timestamps:
            patterns.append(f"Content spans {len(set(timestamps))} distinct time periods")
        
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
    
    def _identify_gaps_and_limitations(self, results: List[Dict[str, Any]]) -> List[str]:
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
        from collections import Counter
        import re
        
        all_words = []
        for text in texts:
            if text:
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend([w for w in words if len(w) > 3])
        
        if not all_words:
            return []
        
        common = Counter(all_words).most_common(5)
        return [word for word, count in common if count > 1]
    
    def _generate_thinking_reasoning(self, request: ReportRequest, 
                                   content_analysis: Dict[str, Any],
                                   visual_assessment: Dict[str, Any],
                                   technical_findings: List[str],
                                   patterns: List[str],
                                   gaps: List[str]) -> str:
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
    
    async def _perform_visual_analysis(self, request: ReportRequest, 
                                     thinking_phase: ThinkingPhase) -> List[Dict[str, Any]]:
        """Perform detailed visual analysis using VLM"""
        if not self.visual_analysis_enabled or not request.include_visual_analysis:
            return []
        
        if not thinking_phase.visual_assessment["has_visual_content"]:
            return []
        
        logger.info("Performing detailed visual analysis...")
        
        # Extract image paths from results
        image_paths = []
        for result in request.search_results[:request.max_results_to_analyze]:
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
            
            for i, description in enumerate(visual_analysis.get("detailed_descriptions", [])[:5]):
                structured_analysis.append({
                    "item_index": i,
                    "detailed_description": description,
                    "technical_assessment": visual_analysis.get("technical_analysis", [])[i] if i < len(visual_analysis.get("technical_analysis", [])) else "",
                    "quality_score": visual_analysis.get("quality_assessment", {}).get("overall", 0.0)
                })
            
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return [{"analysis": "Visual analysis unavailable due to processing error"}]
    
    async def _generate_executive_summary(self, request: ReportRequest, 
                                        thinking_phase: ThinkingPhase) -> str:
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
    
    def _generate_detailed_findings(self, request: ReportRequest, 
                                  thinking_phase: ThinkingPhase) -> List[Dict[str, Any]]:
        """Generate detailed findings section"""
        findings = []
        
        # Content composition finding
        content_analysis = thinking_phase.content_analysis
        findings.append({
            "category": "Content Composition",
            "finding": f"Analysis of {content_analysis['total_results']} results reveals diverse content types",
            "details": content_analysis['content_types'],
            "significance": "high" if len(content_analysis['content_types']) > 2 else "medium"
        })
        
        # Quality assessment finding
        quality_metrics = content_analysis['quality_metrics']
        total_results = content_analysis['total_results']
        if total_results > 0:
            high_quality_ratio = quality_metrics['high'] / total_results
            findings.append({
                "category": "Quality Assessment",
                "finding": f"Content quality distribution shows {high_quality_ratio:.1%} high-quality results",
                "details": quality_metrics,
                "significance": "high" if high_quality_ratio > 0.6 else "medium"
            })
        else:
            findings.append({
                "category": "Quality Assessment",
                "finding": "No results available for quality assessment",
                "details": quality_metrics,
                "significance": "low"
            })
        
        # Visual content finding
        if thinking_phase.visual_assessment['has_visual_content']:
            visual_coverage = thinking_phase.visual_assessment['visual_coverage']
            findings.append({
                "category": "Visual Content",
                "finding": f"Comprehensive visual content available with {visual_coverage} visual elements",
                "details": thinking_phase.visual_assessment['visual_elements'],
                "significance": "high" if visual_coverage > 10 else "medium"
            })
        
        # Pattern analysis finding
        if thinking_phase.patterns_identified:
            findings.append({
                "category": "Pattern Analysis",
                "finding": f"Identified {len(thinking_phase.patterns_identified)} significant patterns",
                "details": thinking_phase.patterns_identified,
                "significance": "high"
            })
        
        return findings
    
    def _generate_technical_details(self, request: ReportRequest, 
                                  thinking_phase: ThinkingPhase) -> List[Dict[str, Any]]:
        """Generate technical details section"""
        if not request.include_technical_details or not self.technical_analysis_enabled:
            return []
        
        technical_details = []
        
        # System findings
        if thinking_phase.technical_findings:
            technical_details.append({
                "section": "System Analysis",
                "details": thinking_phase.technical_findings,
                "analysis": "Technical characteristics extracted from metadata analysis"
            })
        
        # Content analysis
        content_analysis = thinking_phase.content_analysis
        technical_details.append({
            "section": "Content Metrics",
            "details": {
                "total_analyzed": content_analysis['total_results'],
                "quality_distribution": content_analysis['quality_metrics'],
                "average_relevance": round(content_analysis['avg_relevance'], 3)
            },
            "analysis": "Quantitative assessment of content quality and relevance"
        })
        
        # Visual analysis
        if thinking_phase.visual_assessment['has_visual_content']:
            technical_details.append({
                "section": "Visual Content Analysis",
                "details": thinking_phase.visual_assessment['visual_elements'],
                "analysis": f"Visual elements coverage: {thinking_phase.visual_assessment['visual_coverage']} items"
            })
        
        return technical_details
    
    def _generate_recommendations(self, request: ReportRequest, 
                                thinking_phase: ThinkingPhase) -> List[str]:
        """Generate actionable recommendations"""
        if not request.include_recommendations:
            return []
        
        recommendations = []
        
        # Quality-based recommendations
        content_analysis = thinking_phase.content_analysis
        total_results = content_analysis['total_results']
        
        if total_results > 0:
            high_quality_ratio = content_analysis['quality_metrics']['high'] / total_results
            if high_quality_ratio < 0.3:
                recommendations.append("Consider refining search criteria to improve result quality")
        else:
            recommendations.append("Expand search scope to find relevant results")
        
        # Content diversity recommendations
        if len(content_analysis['content_types']) == 1:
            recommendations.append("Expand search to include diverse content types for comprehensive analysis")
        
        # Visual content recommendations
        if not thinking_phase.visual_assessment['has_visual_content']:
            recommendations.append("Incorporate visual content sources to enhance analysis depth")
        elif thinking_phase.visual_assessment['visual_coverage'] < 5:
            recommendations.append("Increase visual content coverage for more comprehensive visual analysis")
        
        # Gap-based recommendations
        for gap in thinking_phase.gaps_and_limitations:
            if "metadata" in gap.lower():
                recommendations.append("Improve metadata completeness for enhanced analysis capabilities")
            elif "diversity" in gap.lower():
                recommendations.append("Diversify content sources to reduce bias and improve coverage")
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append("Current analysis provides comprehensive coverage - maintain search approach")
        
        return recommendations
    
    def _calculate_confidence_assessment(self, request: ReportRequest, 
                                       thinking_phase: ThinkingPhase) -> Dict[str, float]:
        """Calculate confidence assessment for different aspects"""
        content_analysis = thinking_phase.content_analysis
        
        # Content confidence based on quantity and quality
        content_confidence = min(0.9, (content_analysis['total_results'] / 20) * 0.6 + 
                               content_analysis['avg_relevance'] * 0.4)
        
        # Visual confidence based on visual content availability
        visual_confidence = 0.0
        if thinking_phase.visual_assessment['has_visual_content']:
            visual_coverage = thinking_phase.visual_assessment['visual_coverage']
            visual_confidence = min(0.9, visual_coverage / 10)
        
        # Technical confidence based on technical findings
        technical_confidence = min(0.9, len(thinking_phase.technical_findings) / 5)
        
        # Overall confidence as weighted average
        overall_confidence = (content_confidence * 0.4 + 
                            visual_confidence * 0.3 + 
                            technical_confidence * 0.3)
        
        return {
            "overall": round(overall_confidence, 2),
            "content_analysis": round(content_confidence, 2),
            "visual_analysis": round(visual_confidence, 2),
            "technical_analysis": round(technical_confidence, 2)
        }
    
    async def process_a2a_task(self, task: Task) -> Dict[str, Any]:
        """Process A2A task for detailed report generation"""
        if not task.messages:
            raise HTTPException(status_code=400, detail="No messages in task")
        
        # Extract report request from last message
        last_message = task.messages[-1]
        data_part = next((part for part in last_message.parts if isinstance(part, DataPart)), None)
        
        if not data_part:
            raise HTTPException(status_code=400, detail="No data in message")
        
        request_data = data_part.data
        
        # Create report request
        report_request = ReportRequest(
            query=request_data.get("query", ""),
            search_results=request_data.get("search_results", []),
            report_type=request_data.get("report_type", "comprehensive"),
            include_visual_analysis=request_data.get("include_visual_analysis", True),
            include_technical_details=request_data.get("include_technical_details", True),
            include_recommendations=request_data.get("include_recommendations", True),
            max_results_to_analyze=request_data.get("max_results_to_analyze", 20),
            context=request_data.get("context")
        )
        
        # Generate report
        result = await self.generate_report(report_request)
        
        # Convert result to dict for A2A response
        return {
            "task_id": task.id,
            "status": "completed",
            "result": {
                "executive_summary": result.executive_summary,
                "detailed_findings": result.detailed_findings,
                "visual_analysis": result.visual_analysis,
                "technical_details": result.technical_details,
                "recommendations": result.recommendations,
                "confidence_assessment": result.confidence_assessment,
                "metadata": result.metadata
            }
        }


# Global agent instance
report_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global report_agent
    
    try:
        report_agent = DetailedReportAgent()
        logger.info("Detailed report agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "detailed_report",
        "capabilities": ["comprehensive_analysis", "visual_analysis", "technical_details", "recommendations"]
    }


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