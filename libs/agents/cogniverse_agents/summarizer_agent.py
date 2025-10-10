"""
Summarizer Agent with VLM integration and "think phase" for complex queries.
Provides intelligent summarization of search results with visual content analysis.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy
import uvicorn
from fastapi import FastAPI, HTTPException

from cogniverse_agents.dspy_integration_mixin import DSPySummaryMixin

# Enhanced routing support
from cogniverse_agents.routing_agent import RoutingDecision
from cogniverse_core.common.a2a_mixin import A2AEndpointsMixin
from cogniverse_core.config.utils import get_config
from cogniverse_core.common.health_mixin import HealthCheckMixin
from cogniverse_core.common.vlm_interface import VLMInterface
from cogniverse_core.common.a2a_utils import (
    DataPart,
    Task,
)

logger = logging.getLogger(__name__)


class SummaryGenerationSignature(dspy.Signature):
    """Generate structured summaries with key insights."""

    content = dspy.InputField(desc="Search results content to summarize")
    query = dspy.InputField(desc="Original user query")
    summary_type = dspy.InputField(
        desc="Type of summary: brief, comprehensive, detailed"
    )

    summary = dspy.OutputField(desc="Generated summary text")
    key_points = dspy.OutputField(desc="List of key points (comma-separated)")
    confidence_score = dspy.OutputField(desc="Confidence in summary quality (0.0-1.0)")


@dataclass
class SummaryRequest:
    """Request for summarization task"""

    query: str
    search_results: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None
    summary_type: str = "comprehensive"  # comprehensive, brief, bullet_points
    include_visual_analysis: bool = True
    max_results_to_analyze: int = 10


@dataclass
class EnhancedSummaryRequest:
    """Enhanced summarization request with relationship context"""

    original_query: str
    enhanced_query: Optional[str]
    search_results: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    routing_metadata: Dict[str, Any]
    routing_confidence: float
    context: Optional[Dict[str, Any]] = None
    summary_type: str = "comprehensive"
    include_visual_analysis: bool = True
    max_results_to_analyze: int = 10
    focus_on_relationships: bool = True


@dataclass
class ThinkingPhase:
    """Results from the thinking phase"""

    key_themes: List[str]
    content_categories: List[str]
    relevance_scores: Dict[str, float]
    visual_elements: List[str]
    reasoning: str
    # Enhanced fields for relationship context
    entity_insights: Optional[List[Dict[str, Any]]] = None
    relationship_patterns: Optional[List[Dict[str, Any]]] = None
    contextual_connections: Optional[List[str]] = None


@dataclass
class SummaryResult:
    """Complete summarization result"""

    summary: str
    key_points: List[str]
    visual_insights: List[str]
    confidence_score: float
    thinking_phase: ThinkingPhase
    metadata: Dict[str, Any]
    # Enhanced fields for relationship-aware summaries
    relationship_summary: Optional[str] = None
    entity_analysis: Optional[Dict[str, Any]] = None
    enhancement_applied: bool = False


class SummarizerAgent(DSPySummaryMixin, A2AEndpointsMixin, HealthCheckMixin):
    """
    Intelligent summarizer agent with VLM integration and thinking phase.
    Provides comprehensive analysis and summarization of search results.
    """

    def __init__(self, tenant_id: str, **kwargs):
        """
        Initialize summarizer agent

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            **kwargs: Additional configuration options

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        logger.info(f"Initializing SummarizerAgent for tenant: {tenant_id}...")
        super().__init__()  # Initialize DSPy mixin

        self.tenant_id = tenant_id
        self.config = get_config()

        # Initialize DSPy components
        self._initialize_vlm_client()
        self.dspy_summarizer = dspy.Predict(SummaryGenerationSignature)
        self.vlm = VLMInterface()

        # A2A agent metadata
        self.agent_name = "summarizer_agent"
        self.agent_description = (
            "Intelligent summarization with visual analysis and thinking phase"
        )
        self.agent_version = "1.0.0"
        self.agent_url = (
            f"http://localhost:{self.config.get('summarizer_agent_port', 8004)}"
        )
        self.agent_capabilities = [
            "summarization",
            "visual_analysis",
            "key_insights",
            "content_analysis",
        ]
        self.agent_skills = [
            {
                "name": "generate_summary",
                "description": "Generate intelligent summaries with visual analysis and key insights",
                "input_types": ["search_results", "query"],
                "output_types": ["summary", "key_points"],
            }
        ]

        # Configuration
        self.max_summary_length = kwargs.get("max_summary_length", 500)
        self.thinking_enabled = kwargs.get("thinking_enabled", True)
        self.visual_analysis_enabled = kwargs.get("visual_analysis_enabled", True)

        logger.info("SummarizerAgent initialization complete")

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

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """
        Generate comprehensive summary with thinking phase

        Args:
            request: Summarization request

        Returns:
            Complete summary result
        """
        logger.info(f"Starting summarization for query: '{request.query}'")
        logger.info(f"Analyzing {len(request.search_results)} search results")

        try:
            # Phase 1: Thinking phase - analyze and categorize results
            thinking_phase = await self._thinking_phase(request)

            # Phase 2: Extract visual content if enabled
            visual_insights = []
            if request.include_visual_analysis and self.visual_analysis_enabled:
                visual_insights = await self._analyze_visual_content(
                    request, thinking_phase
                )

            # Phase 3: Generate summary based on analysis
            summary = await self._generate_summary(
                request, thinking_phase, visual_insights
            )

            # Phase 4: Extract key points
            key_points = self._extract_key_points(request, thinking_phase, summary)

            # Phase 5: Calculate confidence score
            confidence_score = self._calculate_confidence(request, thinking_phase)

            result = SummaryResult(
                summary=summary,
                key_points=key_points,
                visual_insights=visual_insights,
                confidence_score=confidence_score,
                thinking_phase=thinking_phase,
                metadata={
                    "results_analyzed": len(request.search_results),
                    "summary_type": request.summary_type,
                    "visual_analysis_enabled": request.include_visual_analysis,
                    "processing_time": asyncio.get_event_loop().time(),
                },
            )

            logger.info(f"Summarization complete. Confidence: {confidence_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise

    async def _thinking_phase(self, request: SummaryRequest) -> ThinkingPhase:
        """
        Thinking phase: analyze search results and plan summarization

        Args:
            request: Summarization request

        Returns:
            Thinking phase results
        """
        logger.info("Starting thinking phase...")

        # Analyze content themes
        key_themes = self._extract_themes(request.search_results)

        # Categorize content types
        content_categories = self._categorize_content(request.search_results)

        # Calculate relevance scores
        relevance_scores = self._calculate_relevance_scores(request)

        # Identify visual elements
        visual_elements = self._identify_visual_elements(request.search_results)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            request, key_themes, content_categories, relevance_scores
        )

        thinking_phase = ThinkingPhase(
            key_themes=key_themes,
            content_categories=content_categories,
            relevance_scores=relevance_scores,
            visual_elements=visual_elements,
            reasoning=reasoning,
        )

        logger.info(
            f"Thinking phase complete. Found {len(key_themes)} themes, {len(content_categories)} categories"
        )
        return thinking_phase

    def _extract_themes(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from search results"""
        themes = set()

        for result in search_results:
            # Extract from video metadata
            if "video_id" in result:
                themes.add("video_content")

            # Extract from descriptions
            if "description" in result:
                desc = result["description"].lower()
                if "education" in desc or "tutorial" in desc:
                    themes.add("educational_content")
                if "news" in desc or "report" in desc:
                    themes.add("news_content")
                if "entertainment" in desc or "comedy" in desc:
                    themes.add("entertainment")

            # Extract from content type
            content_type = result.get("content_type", "").lower()
            if content_type:
                themes.add(f"{content_type}_content")

        return list(themes)[:10]  # Limit to top 10 themes

    def _categorize_content(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """Categorize content by type and format"""
        categories = set()

        for result in search_results:
            # Video content categories
            if "video_id" in result:
                categories.add("video")

                # Duration-based categories
                duration = result.get("duration", 0)
                if duration < 60:
                    categories.add("short_form")
                elif duration < 600:
                    categories.add("medium_form")
                else:
                    categories.add("long_form")

            # Text content categories
            if "text_content" in result:
                categories.add("text")

            # Frame-based categories
            if "frame_id" in result:
                categories.add("frame_based")

        return list(categories)

    def _calculate_relevance_scores(self, request: SummaryRequest) -> Dict[str, float]:
        """Calculate relevance scores for each result"""
        relevance_scores = {}

        for i, result in enumerate(request.search_results):
            result_id = result.get("id", f"result_{i}")

            # Base relevance from search score
            base_score = result.get("relevance", result.get("score", 0.5))

            # Adjust based on content type and query
            relevance_scores[result_id] = min(1.0, max(0.0, base_score))

        return relevance_scores

    def _identify_visual_elements(
        self, search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify visual elements in search results"""
        visual_elements = []

        for result in search_results:
            if "frame_id" in result:
                visual_elements.append("video_frames")
            if "thumbnail" in result:
                visual_elements.append("thumbnails")
            if "image_path" in result:
                visual_elements.append("images")

        return list(set(visual_elements))

    def _generate_reasoning(
        self,
        request: SummaryRequest,
        themes: List[str],
        categories: List[str],
        relevance_scores: Dict[str, float],
    ) -> str:
        """Generate reasoning for the summarization approach"""
        avg_relevance = (
            sum(relevance_scores.values()) / len(relevance_scores)
            if relevance_scores
            else 0
        )

        reasoning = f"""
Summarization Strategy:
- Query: "{request.query}"
- Results: {len(request.search_results)} items analyzed
- Average relevance: {avg_relevance:.2f}
- Key themes: {', '.join(themes[:3])}
- Content categories: {', '.join(categories)}
- Summary type: {request.summary_type}

Approach: Will focus on highest relevance results, incorporate visual analysis if available,
and structure summary based on identified themes and content categories.
""".strip()

        return reasoning

    async def _analyze_visual_content(
        self, request: SummaryRequest, thinking_phase: ThinkingPhase
    ) -> List[str]:
        """Analyze visual content using VLM"""
        # Check if visual analysis is enabled and requested
        if not self.visual_analysis_enabled or not request.include_visual_analysis:
            return []

        if not thinking_phase.visual_elements:
            return []

        logger.info("Analyzing visual content with VLM...")

        # Extract image paths from results
        image_paths = []
        for result in request.search_results[: request.max_results_to_analyze]:
            if "thumbnail" in result:
                image_paths.append(result["thumbnail"])
            elif "image_path" in result:
                image_paths.append(result["image_path"])

        if not image_paths:
            return ["No visual content available for analysis"]

        try:
            visual_analysis = await self.vlm.analyze_visual_content(
                image_paths, request.query
            )

            insights = visual_analysis.get("insights", [])
            descriptions = visual_analysis.get("descriptions", [])

            visual_insights = insights + [
                f"Visual: {desc}" for desc in descriptions[:3]
            ]

            return visual_insights[:5]

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            raise

    async def _generate_summary(
        self,
        request: SummaryRequest,
        thinking_phase: ThinkingPhase,
        visual_insights: List[str],
    ) -> str:
        """Generate the main summary text"""
        logger.info("Generating summary...")

        # Sort results by relevance
        sorted_results = sorted(
            request.search_results,
            key=lambda x: thinking_phase.relevance_scores.get(
                x.get("id", ""), x.get("relevance", x.get("score", 0))
            ),
            reverse=True,
        )

        # Take top results for summary
        top_results = sorted_results[: request.max_results_to_analyze]

        if request.summary_type == "brief":
            return self._generate_brief_summary(request, top_results, thinking_phase)
        elif request.summary_type == "bullet_points":
            return self._generate_bullet_summary(request, top_results, thinking_phase)
        else:  # comprehensive
            return self._generate_comprehensive_summary(
                request, top_results, thinking_phase, visual_insights
            )

    def _generate_brief_summary(
        self,
        request: SummaryRequest,
        results: List[Dict[str, Any]],
        thinking_phase: ThinkingPhase,
    ) -> str:
        """Generate brief summary using DSPy"""
        result_count = len(results)

        if result_count == 0:
            return f"No relevant results found for '{request.query}'."

        # Prepare content for DSPy
        content_parts = []
        for result in results[:5]:  # Top 5 results for brief summary
            title = result.get("title", result.get("video_id", "Unknown"))
            content_type = result.get("content_type", "video")
            content_parts.append(f"- {title}: {content_type}")

        content_text = "\n".join(content_parts)

        # Use DSPy for brief summary generation
        try:
            dspy_result = self.dspy_summarizer(
                content=content_text, query=request.query, summary_type="brief"
            )

            return dspy_result.summary

        except Exception as e:
            logger.error(f"DSPy brief summarization failed: {e}")
            raise

    def _generate_bullet_summary(
        self,
        request: SummaryRequest,
        results: List[Dict[str, Any]],
        thinking_phase: ThinkingPhase,
    ) -> str:
        """Generate bullet point summary"""
        points = [
            f"• Query: '{request.query}'",
            f"• Results found: {len(results)}",
            f"• Key themes: {', '.join(thinking_phase.key_themes[:3])}",
            f"• Content types: {', '.join(thinking_phase.content_categories)}",
        ]

        # Add top results
        for i, result in enumerate(results[:3]):
            title = result.get("title", result.get("video_id", f"Result {i+1}"))
            score = result.get("relevance", result.get("score", 0))
            points.append(f"• {title} (relevance: {score:.2f})")

        return "\n".join(points)

    def _generate_comprehensive_summary(
        self,
        request: SummaryRequest,
        results: List[Dict[str, Any]],
        thinking_phase: ThinkingPhase,
        visual_insights: List[str],
    ) -> str:
        """Generate comprehensive summary"""
        summary_parts = []

        # Introduction
        summary_parts.append(
            f"Search results for '{request.query}' yielded {len(results)} relevant items "
            f"spanning {', '.join(thinking_phase.key_themes[:3])}."
        )

        # Content analysis
        if thinking_phase.content_categories:
            summary_parts.append(
                f"The content includes {', '.join(thinking_phase.content_categories)} "
                f"with varying levels of relevance to the search query."
            )

        # Top results
        if results:
            top_result = results[0]
            title = top_result.get("title", top_result.get("video_id", "Top result"))
            score = top_result.get("relevance", top_result.get("score", 0))
            summary_parts.append(
                f"The most relevant result is '{title}' with a relevance score of {score:.2f}."
            )

        # Visual insights
        if visual_insights:
            summary_parts.append(
                f"Visual analysis reveals: {'. '.join(visual_insights[:2])}."
            )

        # Conclusion
        avg_relevance = (
            sum(thinking_phase.relevance_scores.values())
            / len(thinking_phase.relevance_scores)
            if thinking_phase.relevance_scores
            else 0
        )

        summary_parts.append(
            f"Overall, the search results show an average relevance of {avg_relevance:.2f}, "
            f"indicating {'strong' if avg_relevance > 0.7 else 'moderate' if avg_relevance > 0.4 else 'weak'} "
            f"alignment with the search query."
        )

        return " ".join(summary_parts)

    def _extract_key_points(
        self, request: SummaryRequest, thinking_phase: ThinkingPhase, summary: str
    ) -> List[str]:
        """Extract key points from the analysis"""
        key_points = []

        # Theme-based points
        if thinking_phase.key_themes:
            key_points.append(
                f"Main themes: {', '.join(thinking_phase.key_themes[:3])}"
            )

        # Content type points
        if thinking_phase.content_categories:
            key_points.append(
                f"Content types: {', '.join(thinking_phase.content_categories)}"
            )

        # Relevance points
        high_relevance_count = sum(
            1 for score in thinking_phase.relevance_scores.values() if score > 0.7
        )
        if high_relevance_count > 0:
            key_points.append(f"{high_relevance_count} high-relevance results found")

        # Visual content points
        if thinking_phase.visual_elements:
            key_points.append(
                f"Visual content: {', '.join(thinking_phase.visual_elements)}"
            )

        return key_points

    def _calculate_confidence(
        self, request: SummaryRequest, thinking_phase: ThinkingPhase
    ) -> float:
        """Calculate confidence score for the summary"""
        factors = []

        # Result count factor
        result_count = len(request.search_results)
        if result_count >= 5:
            factors.append(0.9)
        elif result_count >= 2:
            factors.append(0.7)
        else:
            factors.append(0.5)

        # Relevance factor
        if thinking_phase.relevance_scores:
            avg_relevance = sum(thinking_phase.relevance_scores.values()) / len(
                thinking_phase.relevance_scores
            )
            factors.append(avg_relevance)
        else:
            factors.append(0.5)

        # Theme diversity factor
        theme_count = len(thinking_phase.key_themes)
        if theme_count >= 3:
            factors.append(0.8)
        elif theme_count >= 1:
            factors.append(0.6)
        else:
            factors.append(0.4)

        return sum(factors) / len(factors)

    async def process_a2a_task(self, task: Task) -> Dict[str, Any]:
        """
        Process A2A task for summarization

        Args:
            task: A2A task with search results

        Returns:
            Summarization response
        """
        if not task.messages:
            raise ValueError("Task contains no messages")

        last_message = task.messages[-1]
        data_part = next(
            (part for part in last_message.parts if isinstance(part, DataPart)), None
        )

        if not data_part:
            raise ValueError("No data part found in message")

        data = data_part.data

        # Extract summarization request
        request = SummaryRequest(
            query=data.get("query", ""),
            search_results=data.get("search_results", []),
            context=data.get("context", {}),
            summary_type=data.get("summary_type", "comprehensive"),
            include_visual_analysis=data.get("include_visual_analysis", True),
            max_results_to_analyze=data.get("max_results_to_analyze", 10),
        )

        # Generate summary
        result = await self.summarize(request)

        return {
            "task_id": task.id,
            "status": "completed",
            "summary": result.summary,
            "key_points": result.key_points,
            "visual_insights": result.visual_insights,
            "confidence_score": result.confidence_score,
            "metadata": result.metadata,
            "thinking_process": {
                "themes": result.thinking_phase.key_themes,
                "categories": result.thinking_phase.content_categories,
                "reasoning": result.thinking_phase.reasoning,
            },
        }

    async def summarize_with_routing_decision(
        self,
        routing_decision: RoutingDecision,
        search_results: List[Dict[str, Any]],
        **kwargs,
    ) -> SummaryResult:
        """
        Generate summary with enhanced relationship context from DSPy routing.

        Args:
            routing_decision: Enhanced routing decision with relationship context
            search_results: Search results to summarize
            **kwargs: Additional summarization parameters

        Returns:
            Enhanced summary with relationship context
        """
        logger.info(
            f"Relationship-aware summarization with confidence: {routing_decision.confidence:.3f}"
        )

        # Create enhanced summary request
        enhanced_request = EnhancedSummaryRequest(
            original_query=routing_decision.routing_metadata.get("original_query", ""),
            enhanced_query=routing_decision.enhanced_query,
            search_results=search_results,
            entities=routing_decision.extracted_entities,
            relationships=routing_decision.extracted_relationships,
            routing_metadata=routing_decision.routing_metadata,
            routing_confidence=routing_decision.confidence,
            summary_type=kwargs.get("summary_type", "comprehensive"),
            include_visual_analysis=kwargs.get("include_visual_analysis", True),
            max_results_to_analyze=kwargs.get("max_results_to_analyze", 10),
            focus_on_relationships=kwargs.get("focus_on_relationships", True),
        )

        # Perform relationship-aware summarization
        return await self.summarize_with_relationships(enhanced_request)

    async def summarize_with_relationships(
        self, request: EnhancedSummaryRequest
    ) -> SummaryResult:
        """
        Generate relationship-aware summary using enhanced context.

        Args:
            request: Enhanced summary request with relationship context

        Returns:
            Enhanced summary result with relationship analysis
        """
        try:
            logger.info(
                f"Enhanced summarization - Original: '{request.original_query}', Enhanced: '{request.enhanced_query}'"
            )
            logger.info(
                f"Found {len(request.entities)} entities and {len(request.relationships)} relationships"
            )

            # Enhanced thinking phase with relationship analysis
            thinking_phase = await self._enhanced_thinking_phase(request)

            # Extract visual content if enabled
            visual_insights = []
            if request.include_visual_analysis and self.visual_analysis_enabled:
                visual_insights = await self._analyze_visual_content_with_relationships(
                    request, thinking_phase
                )

            # Generate relationship-aware summary
            summary = await self._generate_relationship_aware_summary(
                request, thinking_phase, visual_insights
            )

            # Generate relationship-specific summary
            relationship_summary = self._generate_relationship_summary(
                request, thinking_phase
            )

            # Enhanced key points with relationship context
            key_points = self._extract_enhanced_key_points(
                request, thinking_phase, summary
            )

            # Entity analysis
            entity_analysis = self._analyze_entities_in_results(request)

            # Calculate enhanced confidence
            confidence_score = self._calculate_enhanced_confidence(
                request, thinking_phase
            )

            result = SummaryResult(
                summary=summary,
                key_points=key_points,
                visual_insights=visual_insights,
                confidence_score=confidence_score,
                thinking_phase=thinking_phase,
                relationship_summary=relationship_summary,
                entity_analysis=entity_analysis,
                enhancement_applied=True,
                metadata={
                    "results_analyzed": len(request.search_results),
                    "summary_type": request.summary_type,
                    "visual_analysis_enabled": request.include_visual_analysis,
                    "original_query": request.original_query,
                    "enhanced_query": request.enhanced_query,
                    "entities_found": len(request.entities),
                    "relationships_found": len(request.relationships),
                    "routing_confidence": request.routing_confidence,
                    "processing_time": asyncio.get_event_loop().time(),
                },
            )

            logger.info(
                f"Enhanced summarization complete. Confidence: {confidence_score:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Enhanced summarization failed: {e}")
            # Re-raise the exception instead of fallback
            raise

    async def _enhanced_thinking_phase(
        self, request: EnhancedSummaryRequest
    ) -> ThinkingPhase:
        """Enhanced thinking phase with relationship analysis"""

        # Standard thinking phase analysis
        key_themes = self._extract_themes_with_relationships(request)
        content_categories = self._categorize_content(request.search_results)
        relevance_scores = self._calculate_enhanced_relevance_scores(request)
        visual_elements = self._identify_visual_elements(request.search_results)

        # Enhanced relationship analysis
        entity_insights = self._analyze_entity_insights(request)
        relationship_patterns = self._identify_relationship_patterns(request)
        contextual_connections = self._find_contextual_connections(request)

        # Enhanced reasoning with relationship context
        reasoning = self._generate_enhanced_reasoning(
            request,
            key_themes,
            content_categories,
            relevance_scores,
            entity_insights,
            relationship_patterns,
        )

        return ThinkingPhase(
            key_themes=key_themes,
            content_categories=content_categories,
            relevance_scores=relevance_scores,
            visual_elements=visual_elements,
            reasoning=reasoning,
            entity_insights=entity_insights,
            relationship_patterns=relationship_patterns,
            contextual_connections=contextual_connections,
        )

    def _extract_themes_with_relationships(
        self, request: EnhancedSummaryRequest
    ) -> List[str]:
        """Extract themes enhanced by relationship context"""
        themes = set()

        # Standard theme extraction
        standard_themes = self._extract_themes(request.search_results)
        themes.update(standard_themes)

        # Add relationship-derived themes
        for relationship in request.relationships:
            relation = relationship.get("relation", "").lower()
            if "perform" in relation or "play" in relation:
                themes.add("performance_content")
            elif "compete" in relation or "versus" in relation:
                themes.add("competitive_content")
            elif "teach" in relation or "instruct" in relation:
                themes.add("educational_content")
            elif "create" in relation or "make" in relation:
                themes.add("creative_content")

        # Add entity-derived themes
        for entity in request.entities:
            entity_type = entity.get("label", "").lower()
            if entity_type in ["person", "athlete", "performer"]:
                themes.add("people_focused")
            elif entity_type in ["event", "competition", "tournament"]:
                themes.add("event_content")
            elif entity_type in ["location", "venue", "stadium"]:
                themes.add("location_based")

        return list(themes)

    def _calculate_enhanced_relevance_scores(
        self, request: EnhancedSummaryRequest
    ) -> Dict[str, float]:
        """Calculate relevance scores enhanced by relationship context"""
        scores = {}

        for i, result in enumerate(request.search_results):
            base_score = result.get("score", result.get("relevance", 0.5))

            # Check for relationship metadata if available
            if "relationship_metadata" in result:
                rel_metadata = result["relationship_metadata"]
                relationship_boost = rel_metadata.get(
                    "relationship_relevance_score", 0.0
                )
                enhanced_score = base_score + (
                    relationship_boost * 0.2
                )  # Up to 20% boost
                scores[f"result_{i}"] = min(enhanced_score, 1.0)
            else:
                scores[f"result_{i}"] = base_score

        return scores

    def _analyze_entity_insights(
        self, request: EnhancedSummaryRequest
    ) -> List[Dict[str, Any]]:
        """Analyze entity insights from search results"""
        insights = []

        for entity in request.entities:
            entity_text = entity.get("text", "").lower()
            entity_type = entity.get("label", "unknown")

            # Count mentions in results
            mentions = 0
            result_ids = []

            for i, result in enumerate(request.search_results):
                result_content = " ".join(
                    [
                        result.get("title", ""),
                        result.get("description", ""),
                        str(result.get("metadata", {})),
                    ]
                ).lower()

                if entity_text in result_content:
                    mentions += 1
                    result_ids.append(i)

            if mentions > 0:
                insights.append(
                    {
                        "entity": entity_text,
                        "type": entity_type,
                        "mentions": mentions,
                        "result_ids": result_ids,
                        "prominence": mentions / len(request.search_results),
                    }
                )

        return insights

    def _identify_relationship_patterns(
        self, request: EnhancedSummaryRequest
    ) -> List[Dict[str, Any]]:
        """Identify relationship patterns in search results"""
        patterns = []

        # Group relationships by relation type
        relation_groups = {}
        for relationship in request.relationships:
            relation = relationship.get("relation", "").lower()
            if relation not in relation_groups:
                relation_groups[relation] = []
            relation_groups[relation].append(relationship)

        # Analyze each relation group
        for relation, relationships in relation_groups.items():
            if len(relationships) > 1:  # Only include patterns with multiple instances
                subjects = [r.get("subject", "") for r in relationships]
                objects = [r.get("object", "") for r in relationships]

                patterns.append(
                    {
                        "relation_type": relation,
                        "frequency": len(relationships),
                        "subjects": subjects,
                        "objects": objects,
                        "pattern_strength": len(relationships)
                        / len(request.relationships),
                    }
                )

        return patterns

    def _find_contextual_connections(
        self, request: EnhancedSummaryRequest
    ) -> List[str]:
        """Find contextual connections between entities and relationships"""
        connections = []

        # Find entity-relationship connections
        for entity in request.entities:
            entity_text = entity.get("text", "").lower()

            related_relations = []
            for relationship in request.relationships:
                subject = relationship.get("subject", "").lower()
                object_text = relationship.get("object", "").lower()

                if entity_text in subject or entity_text in object_text:
                    related_relations.append(relationship.get("relation", ""))

            if related_relations:
                connections.append(
                    f"{entity_text} is connected to {', '.join(set(related_relations))}"
                )

        return connections

    def _generate_enhanced_reasoning(
        self,
        request: EnhancedSummaryRequest,
        themes: List[str],
        categories: List[str],
        scores: Dict[str, float],
        entity_insights: List[Dict[str, Any]],
        relationship_patterns: List[Dict[str, Any]],
    ) -> str:
        """Generate enhanced reasoning with relationship context"""

        reasoning_parts = []

        # Basic analysis
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        reasoning_parts.append(
            f"Analysis of {len(request.search_results)} results for enhanced query '{request.enhanced_query or request.original_query}' "
            f"shows average relevance of {avg_score:.2f}."
        )

        # Entity analysis
        if entity_insights:
            prominent_entities = [e for e in entity_insights if e["prominence"] > 0.3]
            if prominent_entities:
                entity_names = [e["entity"] for e in prominent_entities[:3]]
                reasoning_parts.append(
                    f"Key entities identified: {', '.join(entity_names)} with high prominence in results."
                )

        # Relationship pattern analysis
        if relationship_patterns:
            strong_patterns = [
                p for p in relationship_patterns if p["pattern_strength"] > 0.3
            ]
            if strong_patterns:
                pattern_relations = [p["relation_type"] for p in strong_patterns[:2]]
                reasoning_parts.append(
                    f"Strong relationship patterns found: {', '.join(pattern_relations)} occurring frequently across results."
                )

        # Query enhancement impact
        if request.enhanced_query and request.enhanced_query != request.original_query:
            reasoning_parts.append(
                f"Query enhancement from '{request.original_query}' to '{request.enhanced_query}' "
                f"with routing confidence {request.routing_confidence:.2f} improved result relevance."
            )

        return " ".join(reasoning_parts)

    async def _analyze_visual_content_with_relationships(
        self, request: EnhancedSummaryRequest, thinking_phase: ThinkingPhase
    ) -> List[str]:
        """Analyze visual content with relationship context"""

        # Standard visual analysis
        standard_visual = await self._analyze_visual_content(
            SummaryRequest(
                query=request.enhanced_query or request.original_query,
                search_results=request.search_results,
                context=request.context,
                summary_type=request.summary_type,
                include_visual_analysis=request.include_visual_analysis,
                max_results_to_analyze=request.max_results_to_analyze,
            ),
            thinking_phase,
        )

        # Add relationship-aware visual insights
        visual_insights = standard_visual.copy()

        if request.focus_on_relationships and request.relationships:
            # Add insights based on relationship patterns
            for pattern in thinking_phase.relationship_patterns or []:
                if pattern["pattern_strength"] > 0.5:
                    visual_insights.append(
                        f"Visual content likely shows {pattern['relation_type']} relationships "
                        f"based on {pattern['frequency']} pattern occurrences"
                    )

        return visual_insights

    async def _generate_relationship_aware_summary(
        self,
        request: EnhancedSummaryRequest,
        thinking_phase: ThinkingPhase,
        visual_insights: List[str],
    ) -> str:
        """Generate summary with relationship awareness"""

        summary_parts = []

        # Enhanced introduction
        query_text = request.enhanced_query or request.original_query
        summary_parts.append(
            f"Enhanced search for '{query_text}' analyzed {len(request.search_results)} results "
            f"with {len(request.entities)} identified entities and {len(request.relationships)} relationship patterns."
        )

        # Entity prominence analysis
        if thinking_phase.entity_insights:
            prominent_entities = [
                e for e in thinking_phase.entity_insights if e["prominence"] > 0.2
            ]
            if prominent_entities:
                entity_list = [
                    f"{e['entity']} ({e['mentions']} mentions)"
                    for e in prominent_entities[:3]
                ]
                summary_parts.append(f"Key entities: {', '.join(entity_list)}.")

        # Relationship pattern analysis
        if thinking_phase.relationship_patterns:
            strong_patterns = [
                p
                for p in thinking_phase.relationship_patterns
                if p["pattern_strength"] > 0.3
            ]
            if strong_patterns:
                pattern_desc = [
                    f"{p['relation_type']} ({p['frequency']}x)"
                    for p in strong_patterns[:2]
                ]
                summary_parts.append(
                    f"Primary relationship patterns: {', '.join(pattern_desc)}."
                )

        # Content themes with relationship context
        if thinking_phase.key_themes:
            summary_parts.append(
                f"Content spans {', '.join(thinking_phase.key_themes[:3])} themes, "
                f"with relationship analysis revealing {', '.join(thinking_phase.contextual_connections[:2] if thinking_phase.contextual_connections else [])}."
            )

        # Top results with relationship scoring
        if request.search_results:
            top_result = request.search_results[0]
            title = top_result.get("title", top_result.get("video_id", "Top result"))
            base_score = top_result.get("score", 0)

            # Check if relationship boost was applied
            if "relationship_metadata" in top_result:
                rel_score = top_result["relationship_metadata"].get(
                    "relationship_relevance_score", 0
                )
                summary_parts.append(
                    f"Top result '{title}' shows base relevance {base_score:.2f} "
                    f"with relationship relevance boost of {rel_score:.2f}."
                )
            else:
                summary_parts.append(
                    f"Top result '{title}' has relevance score {base_score:.2f}."
                )

        # Visual insights with relationship context
        if visual_insights:
            relationship_visual = [
                v for v in visual_insights if "relationship" in v.lower()
            ]
            if relationship_visual:
                summary_parts.append(
                    f"Visual analysis: {'. '.join(relationship_visual[:1])}."
                )
            elif visual_insights:
                summary_parts.append(
                    f"Visual analysis: {'. '.join(visual_insights[:1])}."
                )

        # Enhancement conclusion
        if request.routing_confidence > 0.7:
            summary_parts.append(
                f"High routing confidence ({request.routing_confidence:.2f}) indicates strong "
                f"alignment between query enhancement and result relevance."
            )

        return " ".join(summary_parts)

    def _generate_relationship_summary(
        self, request: EnhancedSummaryRequest, thinking_phase: ThinkingPhase
    ) -> str:
        """Generate focused summary of relationship patterns"""

        if not request.relationships:
            return "No specific relationship patterns identified in the search context."

        summary_parts = []

        # Relationship overview
        summary_parts.append(
            f"Identified {len(request.relationships)} relationships across search results."
        )

        # Pattern analysis
        if thinking_phase.relationship_patterns:
            pattern_count = len(thinking_phase.relationship_patterns)
            strongest_pattern = max(
                thinking_phase.relationship_patterns,
                key=lambda p: p["pattern_strength"],
            )

            summary_parts.append(
                f"Found {pattern_count} distinct relationship patterns, "
                f"with '{strongest_pattern['relation_type']}' being most prominent ({strongest_pattern['frequency']} occurrences)."
            )

        # Entity relationships
        if thinking_phase.entity_insights and thinking_phase.contextual_connections:
            summary_parts.append(
                f"Key relationship connections: {'. '.join(thinking_phase.contextual_connections[:2])}."
            )

        return " ".join(summary_parts)

    def _extract_enhanced_key_points(
        self,
        request: EnhancedSummaryRequest,
        thinking_phase: ThinkingPhase,
        summary: str,
    ) -> List[str]:
        """Extract key points enhanced with relationship context"""

        key_points = []

        # Query enhancement impact
        if request.enhanced_query and request.enhanced_query != request.original_query:
            key_points.append(
                f"Query enhanced from '{request.original_query}' to '{request.enhanced_query}'"
            )

        # Entity prominence
        if thinking_phase.entity_insights:
            top_entities = sorted(
                thinking_phase.entity_insights,
                key=lambda e: e["prominence"],
                reverse=True,
            )[:2]
            for entity in top_entities:
                key_points.append(
                    f"{entity['entity'].title()} appears prominently ({entity['mentions']} mentions)"
                )

        # Relationship patterns
        if thinking_phase.relationship_patterns:
            top_patterns = sorted(
                thinking_phase.relationship_patterns,
                key=lambda p: p["pattern_strength"],
                reverse=True,
            )[:2]
            for pattern in top_patterns:
                key_points.append(
                    f"{pattern['relation_type'].title()} relationships occur {pattern['frequency']} times"
                )

        # Standard key points for additional context
        standard_points = self._extract_key_points(
            SummaryRequest(
                query=request.enhanced_query or request.original_query,
                search_results=request.search_results[:3],  # Top 3 for key points
                summary_type=request.summary_type,
            ),
            thinking_phase,
            summary,
        )

        key_points.extend(standard_points[:3])  # Add top 3 standard points

        return key_points[:5]  # Return top 5 key points

    def _analyze_entities_in_results(
        self, request: EnhancedSummaryRequest
    ) -> Dict[str, Any]:
        """Analyze how entities appear across search results"""

        analysis = {
            "entity_distribution": {},
            "entity_co_occurrence": [],
            "entity_result_mapping": {},
        }

        # Analyze entity distribution
        for entity in request.entities:
            entity_text = entity.get("text", "").lower()
            entity_type = entity.get("label", "unknown")

            result_appearances = []
            for i, result in enumerate(request.search_results):
                result_content = " ".join(
                    [
                        result.get("title", ""),
                        result.get("description", ""),
                        str(result.get("metadata", {})),
                    ]
                ).lower()

                if entity_text in result_content:
                    result_appearances.append(i)

            if result_appearances:
                analysis["entity_distribution"][entity_text] = {
                    "type": entity_type,
                    "appearances": len(result_appearances),
                    "result_ids": result_appearances,
                    "coverage": len(result_appearances) / len(request.search_results),
                }

        # Find entity co-occurrences
        entities_list = list(analysis["entity_distribution"].keys())
        for i, entity1 in enumerate(entities_list):
            for entity2 in entities_list[i + 1 :]:
                entity1_results = set(
                    analysis["entity_distribution"][entity1]["result_ids"]
                )
                entity2_results = set(
                    analysis["entity_distribution"][entity2]["result_ids"]
                )

                overlap = entity1_results.intersection(entity2_results)
                if overlap:
                    analysis["entity_co_occurrence"].append(
                        {
                            "entities": [entity1, entity2],
                            "shared_results": len(overlap),
                            "result_ids": list(overlap),
                        }
                    )

        return analysis

    def _calculate_enhanced_confidence(
        self, request: EnhancedSummaryRequest, thinking_phase: ThinkingPhase
    ) -> float:
        """Calculate confidence score enhanced by relationship context"""

        # Base confidence from routing
        base_confidence = request.routing_confidence

        # Entity coverage boost
        entity_boost = 0.0
        if thinking_phase.entity_insights:
            avg_prominence = sum(
                e["prominence"] for e in thinking_phase.entity_insights
            ) / len(thinking_phase.entity_insights)
            entity_boost = avg_prominence * 0.2  # Up to 20% boost

        # Relationship pattern boost
        pattern_boost = 0.0
        if thinking_phase.relationship_patterns:
            avg_pattern_strength = sum(
                p["pattern_strength"] for p in thinking_phase.relationship_patterns
            ) / len(thinking_phase.relationship_patterns)
            pattern_boost = avg_pattern_strength * 0.1  # Up to 10% boost

        # Query enhancement boost
        enhancement_boost = 0.0
        if request.enhanced_query and request.enhanced_query != request.original_query:
            enhancement_boost = 0.1  # 10% boost for successful enhancement

        # Calculate final confidence
        enhanced_confidence = min(
            base_confidence + entity_boost + pattern_boost + enhancement_boost, 1.0
        )

        return enhanced_confidence

    def process_routing_decision_task(
        self,
        routing_decision: RoutingDecision,
        search_results: List[Dict[str, Any]],
        task_id: str = None,
    ) -> Dict[str, Any]:
        """
        Process a routing decision as a task for A2A compatibility.

        Args:
            routing_decision: Enhanced routing decision from DSPy system
            search_results: Search results to summarize
            task_id: Optional task ID

        Returns:
            A2A-compatible task result
        """

        # Perform relationship-aware summarization
        result = asyncio.run(
            self.summarize_with_routing_decision(routing_decision, search_results)
        )

        # Convert to A2A format
        return {
            "task_id": task_id or "routing_decision_summary",
            "status": "completed",
            "agent": "summarizer_agent",
            "summary": result.summary,
            "key_points": result.key_points,
            "visual_insights": result.visual_insights,
            "confidence_score": result.confidence_score,
            "relationship_summary": result.relationship_summary,
            "entity_analysis": result.entity_analysis,
            "enhancement_applied": result.enhancement_applied,
            "metadata": result.metadata,
            "thinking_process": {
                "themes": result.thinking_phase.key_themes,
                "categories": result.thinking_phase.content_categories,
                "reasoning": result.thinking_phase.reasoning,
                "entity_insights": result.thinking_phase.entity_insights,
                "relationship_patterns": result.thinking_phase.relationship_patterns,
                "contextual_connections": result.thinking_phase.contextual_connections,
            },
            "routing_decision_metadata": {
                "recommended_agent": routing_decision.recommended_agent,
                "confidence": routing_decision.confidence,
                "reasoning": routing_decision.reasoning,
                "fallback_agents": routing_decision.fallback_agents,
            },
        }


# --- FastAPI Server ---
app = FastAPI(
    title="Summarizer Agent",
    description="Intelligent summarization agent with VLM integration",
    version="1.0.0",
)

# Global agent instance - initialized on startup
summarizer_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global summarizer_agent

    try:
        summarizer_agent = SummarizerAgent()

        # Setup A2A standard endpoints
        summarizer_agent.setup_a2a_endpoints(app)

        # Setup health endpoint (mixin provides implementation)
        summarizer_agent.setup_health_endpoint(app)

        logger.info("Summarizer agent initialized with A2A endpoints")
    except Exception as e:
        logger.error(f"Failed to initialize summarizer agent: {e}")
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise


@app.post("/process")
async def process_task(task: Task):
    """Process summarization task"""
    if not summarizer_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = await summarizer_agent.process_a2a_task(task)
        return result
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_direct(
    query: str,
    search_results: List[Dict[str, Any]],
    summary_type: str = "comprehensive",
    include_visual_analysis: bool = True,
):
    """Direct summarization endpoint"""
    if not summarizer_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        request = SummaryRequest(
            query=query,
            search_results=search_results,
            summary_type=summary_type,
            include_visual_analysis=include_visual_analysis,
        )

        result = await summarizer_agent.summarize(request)

        return {
            "summary": result.summary,
            "key_points": result.key_points,
            "visual_insights": result.visual_insights,
            "confidence_score": result.confidence_score,
            "metadata": result.metadata,
        }

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarizer Agent Server")
    parser.add_argument(
        "--port", type=int, default=8003, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Summarizer Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
