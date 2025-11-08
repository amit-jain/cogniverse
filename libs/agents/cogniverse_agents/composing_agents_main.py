# src/agents/composing_agents_main.py
import datetime
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool

# Configure logging
logger = logging.getLogger(__name__)

# Import our custom utilities
from cogniverse_core.config.utils import create_default_config_manager, get_config

from cogniverse_agents.tools.a2a_utils import A2AClient, format_search_results
from cogniverse_agents.tools.video_player_tool import VideoPlayerTool

# Initialize configuration
config = get_config(tenant_id="default", config_manager=create_default_config_manager())


# --- Enhanced A2A Tool for Specialist Agents ---
class EnhancedA2AClientTool(BaseTool):
    """Enhanced A2A tool with better error handling and intelligent routing."""

    def __init__(
        self, name: str, description: str, agent_url: str, result_type: str = "generic"
    ):
        super().__init__(name=name, description=description)
        self.agent_url = agent_url
        self.result_type = result_type
        self.client = A2AClient(timeout=config.get("timeout", 60.0))

    async def execute(
        self,
        query: str,
        top_k: int = 10,
        start_date: str = None,
        end_date: str = None,
        preferred_agent: str = None,
    ) -> Dict[str, Any]:
        """Execute search with enhanced error handling and result formatting."""
        start_time = time.time()
        logger.info(f"⏱️ [{self.name}] Starting execution for query: '{query}'")

        # Handle manual routing if preferred_agent is specified
        agent_url = preferred_agent if preferred_agent else self.agent_url

        try:
            # Prepare search parameters
            search_params = {"query": query, "top_k": top_k}

            # Add temporal filtering for video searches
            if start_date:
                search_params["start_date"] = start_date
            if end_date:
                search_params["end_date"] = end_date

            # Execute the search
            logger.info(f"⏱️ [{self.name}] Sending request to agent at {agent_url}")
            agent_start = time.time()
            response = await self.client.send_task(agent_url, **search_params)
            agent_end = time.time()
            logger.info(
                f"⏱️ [{self.name}] Agent response time: {agent_end - agent_start:.3f}s"
            )

            if "error" in response:
                return {
                    "success": False,
                    "error": response["error"],
                    "formatted_results": f"Error from {self.name}: {response['error']}",
                }

            # Format results for better readability
            format_start = time.time()
            results = response.get("results", [])
            formatted_results = format_search_results(results, self.result_type)
            format_end = time.time()
            logger.info(
                f"⏱️ [{self.name}] Result formatting time: {format_end - format_start:.3f}s"
            )

            total_time = time.time() - start_time
            logger.info(f"⏱️ [{self.name}] TOTAL EXECUTION TIME: {total_time:.3f}s")

            return {
                "success": True,
                "results": results,
                "formatted_results": formatted_results,
                "result_count": len(results),
                "agent_used": self.name,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "formatted_results": f"Error from {self.name}: {str(e)}",
            }


# --- Query Analysis Tool ---
class QueryAnalysisTool(BaseTool):
    """Tool for analyzing user queries to extract intent and temporal information."""

    def __init__(self):
        super().__init__(
            name="QueryAnalyzer",
            description="Analyzes user queries to extract search intent and temporal information",
        )
        # Check configuration for inference mode
        self.inference_mode = config.get("query_inference_engine", {}).get(
            "mode", "keyword"
        )

        if self.inference_mode == "gliner_only":
            # Load GLiNER model directly
            try:
                from gliner import GLiNER

                gliner_model = config.get("query_inference_engine", {}).get(
                    "current_gliner_model"
                )
                if gliner_model:
                    logger.info("🔄 Loading GLiNER model: {gliner_model}")
                    self.gliner_model = GLiNER.from_pretrained(gliner_model)
                    logger.info("✅ GLiNER model loaded successfully")
                else:
                    self.gliner_model = None
            except ImportError:
                logger.warning("⚠️ GLiNER not available, will use keyword fallback")
                self.gliner_model = None
        elif self.inference_mode == "llm":
            # Use PromptManager for LLM-based routing
            from cogniverse_core.common.utils.prompt_manager import PromptManager

            self.prompt_manager = PromptManager(
                config_manager=create_default_config_manager(),
                tenant_id="default"
            )

    async def execute(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent and temporal information."""
        import time

        start_time = time.time()
        logger.info(f"\n⏱️ [QueryAnalyzer] Starting query analysis for: '{query}'")

        analysis = {
            "original_query": query,
            "needs_video_search": False,
            "needs_text_search": False,
            "temporal_info": {},
            "cleaned_query": query.lower().strip(),
        }

        # Extract temporal information first (always done)
        temporal_start = time.time()
        temporal_info = self._extract_temporal_info(query)
        if temporal_info:
            analysis["temporal_info"] = temporal_info
        temporal_end = time.time()
        logger.info(
            f"⏱️ [QueryAnalyzer] Temporal extraction time: {temporal_end - temporal_start:.3f}s"
        )

        # Route based on configured mode
        time.time()
        if (
            self.inference_mode == "gliner_only"
            and hasattr(self, "gliner_model")
            and self.gliner_model
        ):
            # Use GLiNER for entity extraction
            labels = config.get("query_inference_engine", {}).get("gliner_labels", [])
            threshold = config.get("query_inference_engine", {}).get(
                "gliner_threshold", 0.3
            )

            try:
                # Call GLiNER directly
                gliner_start = time.time()
                entities = self.gliner_model.predict_entities(
                    query, labels, threshold=threshold
                )
                gliner_end = time.time()
                logger.info(
                    f"⏱️ [QueryAnalyzer] GLiNER inference time: {gliner_end - gliner_start:.3f}s"
                )

                if entities:
                    # Map entities to search needs
                    for entity in entities:
                        if entity["label"] in ["video_content", "visual_content"]:
                            analysis["needs_video_search"] = True
                        elif entity["label"] in [
                            "document_content",
                            "text_information",
                        ]:
                            analysis["needs_text_search"] = True
                    analysis["routing_method"] = "gliner"
                    analysis["gliner_entities"] = entities  # Include for debugging
                else:
                    logger.info(
                        "GLiNER extraction returned no entities, falling back to keyword matching"
                    )
                    return await self._keyword_based_analysis(
                        query, analysis, start_time
                    )
            except Exception as e:
                logger.info(
                    f"GLiNER extraction failed: {e}, falling back to keyword matching"
                )
                return await self._keyword_based_analysis(query, analysis, start_time)

        elif self.inference_mode == "llm" and hasattr(self, "prompt_manager"):
            # Use LLM for routing
            try:
                import json
                import re

                import requests

                # Get routing prompt
                prompt = self.prompt_manager.get_routing_prompt(query)

                # Call inference endpoint based on provider
                inference_config = config.get("inference", {})
                provider = inference_config.get("provider", "local")

                if provider == "local":
                    endpoint = inference_config.get(
                        "local_endpoint", "http://localhost:11434"
                    )
                    model = inference_config.get("model", "gemma2:2b")
                    response = requests.post(
                        f"{endpoint}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "options": {"temperature": 0.1, "num_predict": 100},
                            "stream": False,
                        },
                        timeout=30,
                    )
                elif provider == "modal" and inference_config.get("modal_endpoint"):
                    endpoint = inference_config.get("modal_endpoint")
                    model = inference_config.get("model", "HuggingFaceTB/SmolLM3-3B")
                    response = requests.post(
                        endpoint,
                        json={
                            "prompt": prompt,
                            "model": model,
                            "temperature": 0.1,
                            "max_tokens": 100,
                        },
                        timeout=30,
                    )
                else:
                    logger.info(
                        f"Unsupported provider: {provider}, falling back to keyword matching"
                    )
                    return await self._keyword_based_analysis(
                        query, analysis, start_time
                    )

                if response.status_code == 200:
                    # Get the appropriate response field based on provider
                    if provider == "local":
                        result_text = response.json().get("response", "")
                    else:  # modal
                        result_text = response.json().get("text", "")

                    # Parse JSON from response
                    json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                    if json_match:
                        routing_decision = json.loads(json_match.group())
                        # Map routing decision to analysis
                        if routing_decision.get("search_modality") == "video":
                            analysis["needs_video_search"] = True
                        elif routing_decision.get("search_modality") == "text":
                            analysis["needs_text_search"] = True
                        analysis["generation_type"] = routing_decision.get(
                            "generation_type", "raw_results"
                        )
                        analysis["routing_method"] = "llm"
                        total_time = time.time() - start_time
                        logger.info(
                            f"⏱️ [QueryAnalyzer] TOTAL ANALYSIS TIME: {total_time:.3f}s"
                        )
                        return analysis
            except Exception as e:
                logger.info(
                    f"LLM routing failed: {e}, falling back to keyword matching"
                )
                return await self._keyword_based_analysis(query, analysis, start_time)

        # Default: keyword-based analysis
        return await self._keyword_based_analysis(query, analysis, start_time)

    async def _keyword_based_analysis(
        self, query: str, analysis: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """Fallback keyword-based analysis."""
        # Video search indicators
        video_keywords = [
            "video",
            "clip",
            "scene",
            "recording",
            "footage",
            "show me",
            "visual",
            "watch",
            "frame",
            "moment",
            "demonstration",
            "presentation",
            "meeting",
        ]

        text_keywords = [
            "document",
            "report",
            "text",
            "article",
            "information",
            "data",
            "details",
            "summary",
            "analysis",
            "research",
            "study",
        ]

        query_lower = query.lower()

        # Check for video search intent
        if any(keyword in query_lower for keyword in video_keywords):
            analysis["needs_video_search"] = True

        # Check for text search intent
        if any(keyword in query_lower for keyword in text_keywords):
            analysis["needs_text_search"] = True

        # If no explicit intent, default to both
        if not analysis["needs_video_search"] and not analysis["needs_text_search"]:
            analysis["needs_video_search"] = True
            analysis["needs_text_search"] = True

        analysis["routing_method"] = "keyword"
        total_time = time.time() - start_time
        logger.info(f"⏱️ [QueryAnalyzer] TOTAL ANALYSIS TIME: {total_time:.3f}s")
        return analysis

    def _extract_temporal_info(self, query: str) -> Dict[str, Any]:
        """Extract temporal information from query."""
        today = datetime.date.today()
        temporal_info = {}

        # Common temporal patterns
        patterns = {
            r"yesterday": (today - datetime.timedelta(days=1), today),
            r"last week": (today - datetime.timedelta(weeks=1), today),
            r"last month": (today - datetime.timedelta(days=30), today),
            r"this week": (today - datetime.timedelta(days=today.weekday()), today),
            r"this month": (today.replace(day=1), today),
        }

        query_lower = query.lower()

        for pattern, (start_date, end_date) in patterns.items():
            if re.search(pattern, query_lower):
                temporal_info["start_date"] = start_date.strftime("%Y-%m-%d")
                temporal_info["end_date"] = end_date.strftime("%Y-%m-%d")
                temporal_info["detected_pattern"] = pattern
                break

        # Look for specific dates (YYYY-MM-DD)
        date_matches = re.findall(r"\d{4}-\d{2}-\d{2}", query)
        if date_matches:
            if len(date_matches) == 1:
                temporal_info["start_date"] = date_matches[0]
            elif len(date_matches) >= 2:
                temporal_info["start_date"] = date_matches[0]
                temporal_info["end_date"] = date_matches[1]

        return temporal_info


# --- Initialize Tools ---
query_analyzer = QueryAnalysisTool()

# Text search tool - commented out until Elasticsearch is configured
# text_search_tool = EnhancedA2AClientTool(
#     name="TextSearchAgent",
#     description="Searches for information in text documents and reports using hybrid search",
#     agent_url=config.get("text_agent_url"),
#     result_type="text"
# )

video_search_tool = EnhancedA2AClientTool(
    name="VideoSearchAgent",
    description="Searches for video content and scenes using multi-modal search with optional temporal filtering",
    agent_url=config.get("video_agent_url"),
    result_type="video",
)

video_player_tool = VideoPlayerTool(tenant_id="default", config_manager=create_default_config_manager())

# --- Enhanced Composing Agent ---
composing_agent = LlmAgent(
    name="CoordinatorAgent",
    model=config.get("local_llm_model", "deepseek-r1:1.5b"),
    description="""I am an intelligent research coordinator that helps users find information from both text documents and video content.

I analyze user queries to understand what they're looking for, then intelligently route their requests to specialized search agents:
- VideoSearchAgent for video clips, scenes, and visual content
- VideoPlayer for playing videos with frame tagging and timeline markers
(TextSearchAgent support coming soon - requires Elasticsearch setup)

I can also extract temporal information from queries (like "last week" or "yesterday") and apply appropriate filters to search results.

After finding relevant video content, I can play videos with highlighted frames and timeline markers for easy navigation.

My goal is to provide comprehensive, well-organized answers by combining insights from multiple sources.""",
    tools=[
        query_analyzer,
        video_search_tool,
        video_player_tool,
    ],  # text_search_tool commented out for now
)


# --- Direct Tool Execution (Bypass LLM) ---
async def route_and_execute_query(
    query: str, top_k: int = 10, preferred_agent: Optional[str] = None
) -> Dict[str, Any]:
    """Route query to appropriate agent based on analysis or manual preference.

    Args:
        query: The search query
        top_k: Number of results to return
        preferred_agent: Optional manual routing to specific agent URL
    """
    logger.info(f"🔍 Routing query: {query}")
    if preferred_agent:
        logger.info(f"📍 Manual routing requested to: {preferred_agent}")

    try:
        # Check for manual routing first
        if preferred_agent:
            # Manual routing bypasses query analysis
            results = {
                "query": query,
                "query_analysis": {
                    "manual_routing": True,
                    "preferred_agent": preferred_agent,
                },
                "execution_type": "manual_routed",
                "agents_called": [],
                "success": True,
            }

            # Create a custom tool for the preferred agent
            manual_agent_tool = EnhancedA2AClientTool(
                name="ManuallyRoutedAgent",
                description=f"Manually routed to {preferred_agent}",
                agent_url=preferred_agent,
                result_type="video",  # Default to video type
            )

            logger.info(
                f"🎯 Executing query on manually specified agent: {preferred_agent}"
            )
            search_results = await manual_agent_tool.execute(query=query, top_k=top_k)

            results["search_results"] = search_results
            results["agents_called"].append(f"Manual: {preferred_agent}")

            return results

        # Step 1: Analyze the query to determine routing
        logger.info("📊 Step 1: Analyzing query for routing...")
        analysis = await query_analyzer.execute(query)
        logger.info(f"   Analysis result: {analysis}")

        results = {
            "query": query,
            "query_analysis": analysis,
            "execution_type": "routed",
            "agents_called": [],
            "success": True,
        }

        # Step 2: Route based on analysis
        if analysis.get("needs_video_search", False):
            logger.info("🎥 Step 2: Routing to VideoSearchAgent...")
            search_params = {"query": query, "top_k": top_k}

            # Add temporal parameters if found
            if "temporal_info" in analysis and analysis["temporal_info"]:
                temporal = analysis["temporal_info"]
                if "start_date" in temporal:
                    search_params["start_date"] = temporal["start_date"]
                    logger.info(f"   Adding start_date: {temporal['start_date']}")
                if "end_date" in temporal:
                    search_params["end_date"] = temporal["end_date"]
                    logger.info(f"   Adding end_date: {temporal['end_date']}")

            video_results = await video_search_tool.execute(**search_params)
            logger.info(
                f"   VideoSearchAgent found {video_results.get('result_count', 0)} results"
            )
            results["video_search_results"] = video_results
            results["agents_called"].append("VideoSearchAgent")

        if analysis.get("needs_text_search", False):
            logger.info(
                "📄 Step 2b: TextSearchAgent needed but not available (Elasticsearch not configured)"
            )
            results["text_search_results"] = {
                "error": "TextSearchAgent not available - Elasticsearch not configured"
            }
            results["agents_called"].append("TextSearchAgent (unavailable)")

        if not analysis.get("needs_video_search", False) and not analysis.get(
            "needs_text_search", False
        ):
            logger.info(
                "❓ No specific routing determined, defaulting to VideoSearchAgent"
            )
            search_params = {"query": query, "top_k": top_k}
            video_results = await video_search_tool.execute(**search_params)
            results["video_search_results"] = video_results
            results["agents_called"].append("VideoSearchAgent (default)")

        return results

    except Exception as e:
        logger.error("❌ Error in routing execution: {e}")
        return {
            "query": query,
            "execution_type": "routed",
            "error": str(e),
            "success": False,
        }


# --- Main Execution Functions ---
async def run_query_programmatically(query: str) -> Dict[str, Any]:
    """Run a query programmatically and return structured results."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=composing_agent,
        app_name="multi_agent_rag",
        session_service=session_service,
    )

    user_id = "programmatic_user"
    session_id = await session_service.create_session(
        user_id=user_id, app_name="multi_agent_rag"
    )

    results = []
    async for event in runner.run(session_id=session_id, user_input=query):
        results.append(event)

    return {
        "query": query,
        "session_id": session_id,
        "events": results,
        "final_response": results[-1] if results else None,
    }


def start_web_interface():
    """Start the interactive web interface."""
    logger.info("🚀 Starting Multi-Agent RAG System...")
    logger.info("=" * 60)

    # Validate configuration
    missing_config = config.validate_required_config()
    if missing_config:
        logger.error("❌ Configuration Issues Found:")
        for key, description in missing_config.items():
            logger.info(f"   - {key}: {description}")
        print("\n📝 Please fix these issues and try again.")
        return

    logger.info("✅ Configuration validated successfully")
    logger.info("📊 Search Backend: {config.get('search_backend').upper()}")
    logger.info("🔍 Text Agent: {config.get('text_agent_url')}")
    logger.info("🎥 Video Agent: {config.get('video_agent_url')}")
    logger.info(
        f"🌐 Web Interface: http://localhost:{config.get('composing_agent_port', 8000)}"
    )

    print("\n" + "=" * 60)
    logger.info("🎯 Starting ADK Web Interface...")
    logger.info("📖 Instructions:")
    print("   1. Navigate to the web interface URL above")
    print("   2. Select 'CoordinatorAgent' from the dropdown")
    print("   3. Start asking questions about your documents and videos!")
    print(
        "   4. Use video search, then ask to 'play video X' to see results with frame markers"
    )
    logger.info("=" * 60)

    # Start the ADK web interface
    os.system("adk web")


if __name__ == "__main__":
    start_web_interface()
