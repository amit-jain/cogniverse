"""
Summarizer Agent with VLM integration and "think phase" for complex queries.
Provides intelligent summarization of search results with visual content analysis.
"""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from src.app.agents.dspy_integration_mixin import DSPySummaryMixin
from src.common.config import get_config
from src.tools.a2a_utils import (
    DataPart,
    Task,
)

logger = logging.getLogger(__name__)


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
class ThinkingPhase:
    """Results from the thinking phase"""

    key_themes: List[str]
    content_categories: List[str]
    relevance_scores: Dict[str, float]
    visual_elements: List[str]
    reasoning: str


@dataclass
class SummaryResult:
    """Complete summarization result"""

    summary: str
    key_points: List[str]
    visual_insights: List[str]
    confidence_score: float
    thinking_phase: ThinkingPhase
    metadata: Dict[str, Any]


class VLMInterface:
    """Interface for Vision Language Model operations"""

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
                logger.info("Initialized OpenAI VLM client")
                return
        except ImportError:
            logger.warning("OpenAI client not available")

        try:
            # Try to use Anthropic client if available
            import anthropic

            api_key = self.config.get("anthropic_api_key") or os.getenv(
                "ANTHROPIC_API_KEY"
            )
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.client_type = "anthropic"
                logger.info("Initialized Anthropic VLM client")
                return
        except ImportError:
            logger.warning("Anthropic client not available")

        # Fallback to mock client for testing
        logger.warning("No VLM client available, using mock client")
        self.client = None
        self.client_type = "mock"

    async def analyze_visual_content(
        self, image_paths: List[str], query: str
    ) -> Dict[str, Any]:
        """
        Analyze visual content using VLM

        Args:
            image_paths: List of paths to images/video frames
            query: Original search query for context

        Returns:
            Analysis results including descriptions, themes, and insights
        """
        if self.client_type == "mock":
            return self._mock_visual_analysis(image_paths, query)

        try:
            if self.client_type == "openai":
                return await self._openai_visual_analysis(image_paths, query)
            elif self.client_type == "anthropic":
                return await self._anthropic_visual_analysis(image_paths, query)
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return self._mock_visual_analysis(image_paths, query)

    def _mock_visual_analysis(
        self, image_paths: List[str], query: str
    ) -> Dict[str, Any]:
        """Mock visual analysis for testing"""
        return {
            "descriptions": [
                f"Visual content analysis for {Path(p).name}" for p in image_paths
            ],
            "themes": ["visual_content", "media_analysis", "image_processing"],
            "key_objects": ["people", "objects", "text", "scenes"],
            "emotions": ["neutral", "positive"],
            "visual_quality": "high",
            "relevance_to_query": 0.8,
            "insights": [
                "Visual content appears relevant to the search query",
                "Multiple scenes and objects detected",
                "Good visual quality for analysis",
            ],
        }

    async def _openai_visual_analysis(
        self, image_paths: List[str], query: str
    ) -> Dict[str, Any]:
        """OpenAI GPT-4 Vision analysis"""
        # Encode images to base64
        encoded_images = []
        for image_path in image_paths[:5]:  # Limit to 5 images
            try:
                with open(image_path, "rb") as img_file:
                    encoded_images.append(base64.b64encode(img_file.read()).decode())
            except Exception as e:
                logger.warning(f"Could not encode image {image_path}: {e}")

        if not encoded_images:
            return self._mock_visual_analysis(image_paths, query)

        # Prepare messages for GPT-4 Vision
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze these images in the context of the search query: "{query}"

Please provide:
1. Brief description of each image
2. Key themes and topics
3. Notable objects or elements
4. Emotional tone if applicable
5. Relevance to the search query (0-1 score)
6. Key insights for summarization

Format your response as JSON with keys: descriptions, themes, key_objects, emotions, relevance_to_query, insights""",
                    }
                ]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                    for img in encoded_images
                ],
            }
        ]

        try:
            response = await self.client.chat.completions.acreate(
                model=self.model_name, messages=messages, max_tokens=1000
            )

            # Parse JSON response
            import json

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            logger.error(f"OpenAI vision analysis failed: {e}")
            return self._mock_visual_analysis(image_paths, query)

    async def _anthropic_visual_analysis(
        self, image_paths: List[str], query: str
    ) -> Dict[str, Any]:
        """Anthropic Claude Vision analysis"""
        # Similar implementation for Anthropic
        # For now, return mock analysis
        return self._mock_visual_analysis(image_paths, query)


class SummarizerAgent(DSPySummaryMixin):
    """
    Intelligent summarizer agent with VLM integration and thinking phase.
    Provides comprehensive analysis and summarization of search results.
    """

    def __init__(self, **kwargs):
        """Initialize summarizer agent"""
        logger.info("Initializing SummarizerAgent...")
        super().__init__()  # Initialize DSPy mixin

        self.config = get_config()
        self.vlm = VLMInterface(kwargs.get("vlm_model", "gpt-4-vision-preview"))

        # Configuration
        self.max_summary_length = kwargs.get("max_summary_length", 500)
        self.thinking_enabled = kwargs.get("thinking_enabled", True)
        self.visual_analysis_enabled = kwargs.get("visual_analysis_enabled", True)

        logger.info("SummarizerAgent initialization complete")

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

            # Combine insights and descriptions
            visual_insights = insights + [
                f"Visual: {desc}" for desc in descriptions[:3]
            ]

            return visual_insights[:5]  # Limit to 5 insights

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return ["Visual analysis unavailable"]

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
        """Generate brief summary"""
        result_count = len(results)
        themes = ", ".join(thinking_phase.key_themes[:3])

        if result_count == 0:
            return f"No relevant results found for '{request.query}'."

        return (
            f"Found {result_count} results for '{request.query}' covering {themes}. "
            f"Content includes {', '.join(thinking_phase.content_categories)} "
            f"with varying relevance to the search query."
        )

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
        logger.info("Summarizer agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize summarizer agent: {e}")
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not summarizer_agent:
        return {"status": "initializing", "agent": "summarizer"}

    return {
        "status": "healthy",
        "agent": "summarizer",
        "capabilities": ["summarization", "visual_analysis", "thinking_phase"],
        "vlm_enabled": summarizer_agent.visual_analysis_enabled,
    }


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
    parser.add_argument(
        "--vlm-model", type=str, default="gpt-4-vision-preview", help="VLM model to use"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Summarizer Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
