"""
Enhanced Query Analysis Tool V3 with agent integration and thinking phase.
Integrates with the multi-agent routing system and specialized agents.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from cogniverse_agents.dspy_integration_mixin import DSPyQueryAnalysisMixin
from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_core.common.config_utils import get_config

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""

    SIMPLE = "simple"  # Single intent, direct search
    MODERATE = "moderate"  # Multiple intents or some analysis
    COMPLEX = "complex"  # Multiple modalities, analysis + reporting


class QueryIntent(Enum):
    """Detected query intents"""

    SEARCH = "search"  # Simple search query
    COMPARE = "compare"  # Compare multiple items
    ANALYZE = "analyze"  # Deep analysis required
    SUMMARIZE = "summarize"  # Summarization needed
    REPORT = "report"  # Detailed reporting
    EXPLAIN = "explain"  # Educational/explanatory
    RECOMMEND = "recommend"  # Recommendation needed
    TEMPORAL = "temporal"  # Time-based query
    VISUAL = "visual"  # Visual content focus
    MULTIMODAL = "multimodal"  # Multiple modalities


@dataclass
class QueryContext:
    """Context information for query analysis"""

    conversation_history: List[str] = None
    user_preferences: Dict[str, Any] = None
    previous_results: List[Dict[str, Any]] = None
    session_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.previous_results is None:
            self.previous_results = []
        if self.session_metadata is None:
            self.session_metadata = {}


@dataclass
class QueryAnalysisResult:
    """Comprehensive query analysis result"""

    # Basic analysis
    original_query: str
    cleaned_query: str
    expanded_queries: List[str]

    # Intent and complexity
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    complexity_level: QueryComplexity
    confidence_score: float

    # Modality and search requirements
    needs_video_search: bool
    needs_text_search: bool
    needs_visual_analysis: bool

    # Temporal information
    temporal_filters: Dict[str, Any]

    # Entity extraction
    entities: List[Dict[str, Any]]
    keywords: List[str]

    # Workflow recommendations
    recommended_workflow: str
    workflow_steps: List[Dict[str, Any]]
    required_agents: List[str]

    # Analysis metadata
    thinking_phase: Dict[str, Any]
    analysis_time_ms: float
    routing_method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "original_query": self.original_query,
            "cleaned_query": self.cleaned_query,
            "expanded_queries": self.expanded_queries,
            "primary_intent": self.primary_intent.value,
            "secondary_intents": [intent.value for intent in self.secondary_intents],
            "complexity_level": self.complexity_level.value,
            "confidence_score": self.confidence_score,
            "needs_video_search": self.needs_video_search,
            "needs_text_search": self.needs_text_search,
            "needs_visual_analysis": self.needs_visual_analysis,
            "temporal_filters": self.temporal_filters,
            "entities": self.entities,
            "keywords": self.keywords,
            "recommended_workflow": self.recommended_workflow,
            "workflow_steps": self.workflow_steps,
            "required_agents": self.required_agents,
            "thinking_phase": self.thinking_phase,
            "analysis_time_ms": self.analysis_time_ms,
            "routing_method": self.routing_method,
        }


class QueryAnalysisToolV3(DSPyQueryAnalysisMixin):
    """
    Enhanced Query Analysis Tool with agent integration and thinking phase.

    Features:
    - Advanced intent detection and query understanding
    - Integration with multi-agent routing system
    - Query expansion and refinement
    - Multimodal query support
    - Thinking phase for complex analysis
    - Workflow orchestration
    """

    def __init__(self, **kwargs):
        """Initialize the enhanced query analysis tool"""
        logger.info("Initializing QueryAnalysisToolV3...")
        super().__init__()  # Initialize DSPy mixin

        self.config = get_config()

        # Configuration
        self.enable_thinking_phase = kwargs.get("enable_thinking_phase", True)
        self.enable_query_expansion = kwargs.get("enable_query_expansion", True)
        self.enable_agent_integration = kwargs.get("enable_agent_integration", True)
        self.max_expanded_queries = kwargs.get("max_expanded_queries", 3)

        # Initialize routing agent if integration is enabled
        self.routing_agent = None
        if self.enable_agent_integration:
            try:
                self.routing_agent = RoutingAgent()
                logger.info("Routing agent integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize routing agent: {e}")

        # Query analysis patterns and rules
        self._initialize_analysis_patterns()

        # Statistics
        self.total_analyses = 0
        self.start_time = datetime.now()

        logger.info("QueryAnalysisToolV3 initialization complete")

    def _initialize_analysis_patterns(self):
        """Initialize patterns for query analysis"""
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.SEARCH: [
                r"find|search|show|get|retrieve",
                r"what|where|when|who|how",
                r"videos about|content about",
            ],
            QueryIntent.COMPARE: [
                r"compare|difference|versus|vs|between",
                r"which is better|what's the difference",
            ],
            QueryIntent.ANALYZE: [
                r"analyze|analysis|examine|investigate",
                r"deep dive|detailed look|study",
            ],
            QueryIntent.SUMMARIZE: [
                r"summarize|summary|brief|overview",
                r"key points|main ideas|gist",
            ],
            QueryIntent.REPORT: [
                r"report|detailed report|comprehensive|analysis",
                r"generate report|create report|full analysis",
            ],
            QueryIntent.EXPLAIN: [
                r"explain|how does|why does|what is",
                r"help me understand|clarify|elaborate",
            ],
            QueryIntent.RECOMMEND: [
                r"recommend|suggest|what should",
                r"best|top|ideal|optimal",
            ],
            QueryIntent.TEMPORAL: [
                r"recent|latest|new|today|yesterday",
                r"last week|last month|this year|since",
            ],
            QueryIntent.VISUAL: [
                r"show|display|visualize|image|video",
                r"visual|graphic|chart|diagram",
            ],
            QueryIntent.MULTIMODAL: [
                r"video and text|image and description",
                r"visual and audio|multimedia",
            ],
        }

        # Complexity indicators
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: [
                r"^(find|show|get) .{1,50}$",
                r"^what is .{1,30}$",
            ],
            QueryComplexity.MODERATE: [
                r"compare .* and .*",
                r"summarize.*about",
                r"explain.*how",
            ],
            QueryComplexity.COMPLEX: [
                r"analyze.*and.*report",
                r"comprehensive.*analysis",
                r"detailed.*examination.*with",
            ],
        }

        # Workflow patterns
        self.workflow_patterns = {
            "raw_results": ["simple search", "basic find", "quick lookup"],
            "summary": ["summarize", "overview", "brief", "key points"],
            "detailed_report": ["detailed", "comprehensive", "full analysis", "report"],
        }

    async def analyze(
        self, query: str, context: Optional[QueryContext] = None
    ) -> QueryAnalysisResult:
        """
        Perform comprehensive query analysis with thinking phase.

        Args:
            query: The user query to analyze
            context: Optional context information

        Returns:
            Comprehensive query analysis result
        """
        import time

        start_time = time.time()

        self.total_analyses += 1
        logger.info(
            f"\n🔍 [QueryAnalysisV3] Starting analysis #{self.total_analyses} for: '{query}'"
        )

        try:
            # Phase 1: Thinking phase for complex analysis
            thinking_phase = {}
            if self.enable_thinking_phase:
                thinking_phase = await self._thinking_phase(query, context)

            # Phase 2: Basic query processing
            cleaned_query = self._clean_query(query)

            # Phase 3: Intent detection
            primary_intent, secondary_intents = self._detect_intents(
                query, thinking_phase
            )

            # Phase 4: Complexity assessment
            complexity_level = self._assess_complexity(
                query, primary_intent, secondary_intents
            )

            # Phase 5: Modality detection
            modality_requirements = self._detect_modality_requirements(
                query, thinking_phase
            )

            # Phase 6: Temporal analysis
            temporal_filters = self._extract_temporal_filters(query)

            # Phase 7: Entity and keyword extraction
            entities, keywords = self._extract_entities_and_keywords(query)

            # Phase 8: Query expansion
            expanded_queries = []
            if self.enable_query_expansion:
                expanded_queries = await self._expand_query(
                    query, context, thinking_phase
                )

            # Phase 9: Workflow determination
            workflow_info = await self._determine_workflow(
                query,
                primary_intent,
                secondary_intents,
                complexity_level,
                thinking_phase,
            )

            # Phase 10: Calculate confidence score
            confidence_score = self._calculate_confidence(
                query, primary_intent, complexity_level, len(entities), len(keywords)
            )

            # Create comprehensive result
            analysis_time = (time.time() - start_time) * 1000

            result = QueryAnalysisResult(
                original_query=query,
                cleaned_query=cleaned_query,
                expanded_queries=expanded_queries,
                primary_intent=primary_intent,
                secondary_intents=secondary_intents,
                complexity_level=complexity_level,
                confidence_score=confidence_score,
                needs_video_search=modality_requirements["video"],
                needs_text_search=modality_requirements["text"],
                needs_visual_analysis=modality_requirements["visual_analysis"],
                temporal_filters=temporal_filters,
                entities=entities,
                keywords=keywords,
                recommended_workflow=workflow_info["type"],
                workflow_steps=workflow_info["steps"],
                required_agents=workflow_info["agents"],
                thinking_phase=thinking_phase,
                analysis_time_ms=analysis_time,
                routing_method="enhanced_v3",
            )

            self._log_analysis_result(result, analysis_time / 1000)

            return result

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            raise

    async def _thinking_phase(
        self, query: str, context: Optional[QueryContext]
    ) -> Dict[str, Any]:
        """
        Thinking phase for complex query analysis.

        Args:
            query: The query to analyze
            context: Optional context

        Returns:
            Thinking phase insights
        """
        logger.info("🤔 Starting thinking phase...")

        thinking = {
            "query_length": len(query.split()),
            "has_context": context is not None,
            "query_type_indicators": [],
            "complexity_signals": [],
            "modality_hints": [],
            "temporal_indicators": [],
            "reasoning": "",
        }

        # Analyze query structure
        words = query.lower().split()

        # Look for complexity signals
        complex_terms = [
            "analyze",
            "compare",
            "detailed",
            "comprehensive",
            "report",
            "explain",
        ]
        thinking["complexity_signals"] = [
            term for term in complex_terms if term in words
        ]

        # Look for modality hints
        video_terms = ["video", "watch", "show", "visual", "clip", "footage"]
        text_terms = ["text", "document", "article", "description", "transcript"]
        image_terms = ["image", "picture", "photo", "visual", "graphic"]

        thinking["modality_hints"] = {
            "video": any(term in words for term in video_terms),
            "text": any(term in words for term in text_terms),
            "image": any(term in words for term in image_terms),
        }

        # Look for temporal indicators
        temporal_terms = [
            "recent",
            "latest",
            "new",
            "today",
            "yesterday",
            "last",
            "since",
            "before",
            "after",
        ]
        thinking["temporal_indicators"] = [
            term for term in temporal_terms if term in words
        ]

        # Look for output type indicators
        summary_terms = ["summary", "summarize", "overview", "brief", "key points"]
        report_terms = ["report", "detailed", "comprehensive", "analysis", "full"]

        thinking["query_type_indicators"] = {
            "summary": any(term in words for term in summary_terms),
            "report": any(term in words for term in report_terms),
            "search": not any(term in words for term in summary_terms + report_terms),
        }

        # Generate reasoning
        reasoning_parts = []

        if thinking["query_length"] > 10:
            reasoning_parts.append("Long query suggests complex information need")

        if len(thinking["complexity_signals"]) > 1:
            reasoning_parts.append(
                f"Multiple complexity signals: {', '.join(thinking['complexity_signals'])}"
            )

        if thinking["modality_hints"]["video"] and thinking["modality_hints"]["text"]:
            reasoning_parts.append(
                "Multimodal query requiring both video and text search"
            )

        if thinking["temporal_indicators"]:
            reasoning_parts.append(
                f"Temporal constraints detected: {', '.join(thinking['temporal_indicators'])}"
            )

        if thinking["query_type_indicators"]["report"]:
            reasoning_parts.append(
                "Detailed reporting required based on indicator words"
            )
        elif thinking["query_type_indicators"]["summary"]:
            reasoning_parts.append("Summarization required based on indicator words")

        thinking["reasoning"] = (
            ". ".join(reasoning_parts)
            if reasoning_parts
            else "Simple search query detected"
        )

        logger.info(f"🤔 Thinking phase complete: {thinking['reasoning']}")

        return thinking

    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Basic cleaning
        cleaned = query.strip().lower()

        # Remove extra whitespace
        import re

        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned

    def _detect_intents(
        self, query: str, thinking_phase: Dict[str, Any]
    ) -> Tuple[QueryIntent, List[QueryIntent]]:
        """
        Detect primary and secondary intents in the query.

        Args:
            query: The query to analyze
            thinking_phase: Thinking phase results

        Returns:
            Primary intent and list of secondary intents
        """
        import re

        query_lower = query.lower()
        intent_scores = {}

        # Score each intent based on pattern matching
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score

        # Boost scores based on thinking phase
        if thinking_phase.get("query_type_indicators", {}).get("summary"):
            intent_scores[QueryIntent.SUMMARIZE] += 2

        if thinking_phase.get("query_type_indicators", {}).get("report"):
            intent_scores[QueryIntent.REPORT] += 2

        if thinking_phase.get("complexity_signals"):
            intent_scores[QueryIntent.ANALYZE] += len(
                thinking_phase["complexity_signals"]
            )

        if thinking_phase.get("temporal_indicators"):
            intent_scores[QueryIntent.TEMPORAL] += 1

        if thinking_phase.get("modality_hints", {}).get("video") and thinking_phase.get(
            "modality_hints", {}
        ).get("text"):
            intent_scores[QueryIntent.MULTIMODAL] += 1
        elif thinking_phase.get("modality_hints", {}).get("video"):
            intent_scores[QueryIntent.VISUAL] += 1

        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_intent = QueryIntent.SEARCH

        # Determine secondary intents (score > 0 and not primary)
        secondary_intents = [
            intent
            for intent, score in intent_scores.items()
            if score > 0 and intent != primary_intent
        ]

        return primary_intent, secondary_intents

    def _assess_complexity(
        self,
        query: str,
        primary_intent: QueryIntent,
        secondary_intents: List[QueryIntent],
    ) -> QueryComplexity:
        """
        Assess the complexity level of the query.

        Args:
            query: The query
            primary_intent: Primary detected intent
            secondary_intents: Secondary intents

        Returns:
            Query complexity level
        """
        import re

        # Start with simple
        complexity = QueryComplexity.SIMPLE

        # Check for complexity patterns
        query_lower = query.lower()

        for level, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    if level.value > complexity.value:  # Higher complexity
                        complexity = level
                        break

        # Boost complexity based on intents
        complex_intents = [QueryIntent.ANALYZE, QueryIntent.REPORT, QueryIntent.COMPARE]
        if primary_intent in complex_intents:
            if complexity == QueryComplexity.SIMPLE:
                complexity = QueryComplexity.MODERATE

        # Multiple secondary intents suggest complexity
        if len(secondary_intents) >= 2:
            complexity = QueryComplexity.COMPLEX

        # Multimodal queries are at least moderate
        if QueryIntent.MULTIMODAL in [primary_intent] + secondary_intents:
            if complexity == QueryComplexity.SIMPLE:
                complexity = QueryComplexity.MODERATE

        return complexity

    def _detect_modality_requirements(
        self, query: str, thinking_phase: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Detect what modalities are required for this query.

        Args:
            query: The query
            thinking_phase: Thinking phase results

        Returns:
            Dictionary of modality requirements
        """
        modality_hints = thinking_phase.get("modality_hints", {})

        # Default requirements
        requirements = {
            "video": True,  # Default to video search
            "text": True,  # Default to text search
            "visual_analysis": False,  # Only when specifically needed
        }

        # Adjust based on hints
        if modality_hints.get("video") and not modality_hints.get("text"):
            requirements["text"] = False
        elif modality_hints.get("text") and not modality_hints.get("video"):
            requirements["video"] = False

        # Enable visual analysis for analysis/report tasks
        if any(
            term in query.lower() for term in ["analyze", "detailed", "visual", "image"]
        ):
            requirements["visual_analysis"] = True

        return requirements

    def _extract_temporal_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract temporal filters from the query.

        Args:
            query: The query to analyze

        Returns:
            Dictionary of temporal filters
        """
        import re
        from datetime import datetime, timedelta

        filters = {}
        query_lower = query.lower()

        # Common temporal patterns
        patterns = {
            "today": timedelta(days=0),
            "yesterday": timedelta(days=1),
            "last week": timedelta(weeks=1),
            "last month": timedelta(days=30),
            "last year": timedelta(days=365),
            "recent": timedelta(days=7),
            "latest": timedelta(days=7),
        }

        for term, delta in patterns.items():
            if term in query_lower:
                start_date = (datetime.now() - delta).strftime("%Y-%m-%d")
                filters["start_date"] = start_date
                filters["temporal_term"] = term
                break

        # Look for specific date patterns
        date_pattern = r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        dates = re.findall(date_pattern, query)
        if dates:
            filters["specific_dates"] = dates

        return filters

    def _extract_entities_and_keywords(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Extract entities and keywords from the query.

        Args:
            query: The query to analyze

        Returns:
            Tuple of (entities, keywords)
        """
        import re

        # Simple entity extraction (can be enhanced with NLP libraries)
        entities = []

        # Look for quoted phrases (high-value entities)
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        for phrase in quoted_phrases:
            entities.append(
                {"text": phrase, "type": "quoted_phrase", "confidence": 0.9}
            )

        # Look for capitalized words (potential proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", query)
        for word in capitalized:
            if word.lower() not in [
                "show",
                "find",
                "get",
                "what",
                "when",
                "where",
                "how",
            ]:
                entities.append(
                    {"text": word, "type": "proper_noun", "confidence": 0.6}
                )

        # Extract keywords (remove stop words)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "about",  # Added to filter common preposition
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "have",
            "has",
            "had",
        }
        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return entities, keywords

    async def _expand_query(
        self,
        query: str,
        context: Optional[QueryContext],
        thinking_phase: Dict[str, Any],
    ) -> List[str]:
        """
        Expand the query with related terms and variations.

        Args:
            query: Original query
            context: Optional context
            thinking_phase: Thinking phase results

        Returns:
            List of expanded queries
        """
        expansions = []

        # Enhanced synonym expansion
        synonyms = {
            "video": ["clip", "footage", "recording"],
            "show": ["display", "demonstrate", "present"],
            "find": ["search", "locate", "discover"],
            "analyze": ["examine", "study", "investigate"],
            "explain": ["describe", "clarify", "elaborate"],
            "latest": ["recent", "newest", "current"],
            "developments": ["updates", "progress", "advances"],
            "research": ["studies", "investigations", "analysis"],
            "recent": ["latest", "new", "current"],
        }

        words = query.lower().split()

        # Create variations with synonyms
        for original, synonym_list in synonyms.items():
            if original in words:
                for synonym in synonym_list[:2]:  # Limit to 2 synonyms
                    expanded = query.replace(original, synonym, 1)
                    if (
                        expanded != query
                        and len(expansions) < self.max_expanded_queries
                    ):
                        expansions.append(expanded)

        # If no synonym expansions, create basic variations
        if not expansions and len(words) > 1:
            # Add simple rephrasing
            if query.lower().startswith(("find", "search", "show")):
                alternatives = ["display", "locate", "search for"]
                for alt in alternatives:
                    if len(expansions) < self.max_expanded_queries:
                        expanded = query.replace(query.split()[0].lower(), alt, 1)
                        expansions.append(expanded)
            else:
                # Add search prefixes to queries without them
                prefixes = ["search for", "find information about"]
                for prefix in prefixes:
                    if len(expansions) < self.max_expanded_queries:
                        expansions.append(f"{prefix} {query}")

        # Add context-based expansions if available
        if context and context.conversation_history:
            # Use previous queries for context
            for prev_query in context.conversation_history[-2:]:  # Last 2 queries
                if len(prev_query.split()) <= 3:  # Short previous queries
                    combined = f"{query} {prev_query}"
                    if len(expansions) < self.max_expanded_queries:
                        expansions.append(combined)

        return expansions

    async def _determine_workflow(
        self,
        query: str,
        primary_intent: QueryIntent,
        secondary_intents: List[QueryIntent],
        complexity_level: QueryComplexity,
        thinking_phase: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Determine the recommended workflow for this query.

        Args:
            query: The query
            primary_intent: Primary intent
            secondary_intents: Secondary intents
            complexity_level: Query complexity
            thinking_phase: Thinking phase results

        Returns:
            Workflow information
        """
        # Determine workflow type based on intents and thinking phase
        if primary_intent in [QueryIntent.REPORT, QueryIntent.ANALYZE]:
            workflow_type = "detailed_report"
        elif primary_intent == QueryIntent.SUMMARIZE or thinking_phase.get(
            "query_type_indicators", {}
        ).get("summary"):
            workflow_type = "summary"
        else:
            workflow_type = "raw_results"

        # Use routing agent if available
        if self.routing_agent and self.enable_agent_integration:
            try:
                routing_analysis = await self.routing_agent.route_query(
                    query, context={"thinking_phase": thinking_phase}
                )

                workflow_info = routing_analysis.get("workflow", {})
                workflow_type = workflow_info.get("type", workflow_type)

                return {
                    "type": workflow_type,
                    "steps": workflow_info.get("steps", []),
                    "agents": workflow_info.get("agents", []),
                }
            except Exception as e:
                logger.warning(f"Routing agent integration failed: {e}")

        # Fallback workflow construction
        steps = []
        agents = []

        # Step 1: Search
        steps.append(
            {
                "step": 1,
                "agent": "video_search",
                "action": "search",
                "parameters": {"query": query, "top_k": 10},
            }
        )
        agents.append("video_search")

        # Step 2: Post-processing
        if workflow_type == "summary":
            steps.append(
                {
                    "step": 2,
                    "agent": "summarizer",
                    "action": "summarize",
                    "parameters": {
                        "query": query,
                        "summary_type": (
                            "comprehensive"
                            if complexity_level == QueryComplexity.COMPLEX
                            else "brief"
                        ),
                    },
                }
            )
            agents.append("summarizer")
        elif workflow_type == "detailed_report":
            steps.append(
                {
                    "step": 2,
                    "agent": "detailed_report",
                    "action": "analyze",
                    "parameters": {
                        "query": query,
                        "report_type": "comprehensive",
                        "include_visual_analysis": True,
                    },
                }
            )
            agents.append("detailed_report")

        return {"type": workflow_type, "steps": steps, "agents": agents}

    def _calculate_confidence(
        self,
        query: str,
        primary_intent: QueryIntent,
        complexity_level: QueryComplexity,
        num_entities: int,
        num_keywords: int,
    ) -> float:
        """
        Calculate confidence score for the analysis.

        Args:
            query: Original query text
            primary_intent: Detected primary intent
            complexity_level: Query complexity
            num_entities: Number of detected entities
            num_keywords: Number of keywords

        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.3  # Lower base confidence

        # Check for vague terms that lower confidence
        vague_terms = ["stuff", "things", "some", "anything", "whatever"]
        if any(term in query.lower() for term in vague_terms):
            base_confidence = 0.2  # Very low base for vague queries

        # Boost for clear intents
        if primary_intent in [
            QueryIntent.SEARCH,
            QueryIntent.SUMMARIZE,
            QueryIntent.REPORT,
        ]:
            # Only add boost if we have substantial content (entities/keywords)
            if num_entities > 0 or num_keywords > 3:
                base_confidence += 0.3
            else:
                base_confidence += 0.1  # Small boost for intent but no content

        # Boost for entities and keywords
        entity_boost = min(0.3, num_entities * 0.1)
        keyword_boost = min(0.2, num_keywords * 0.02)

        # Complexity affects confidence differently
        complexity_boost = {
            QueryComplexity.SIMPLE: (
                0.05 if num_entities > 0 else -0.1
            ),  # Simple queries need entities
            QueryComplexity.MODERATE: 0.1,
            QueryComplexity.COMPLEX: 0.15,  # Complex queries show thoughtfulness
        }.get(complexity_level, 0.0)

        confidence = base_confidence + entity_boost + keyword_boost + complexity_boost
        return max(0.1, min(1.0, confidence))  # Ensure at least 0.1

    def _log_analysis_result(self, result: QueryAnalysisResult, execution_time: float):
        """
        Log the analysis result for monitoring.

        Args:
            result: Analysis result
            execution_time: Time taken for analysis
        """
        logger.info(f"🔍 [QueryAnalysisV3] ANALYSIS COMPLETE in {execution_time:.3f}s")
        logger.info(f"   ├─ Primary Intent: {result.primary_intent.value}")
        logger.info(
            f"   ├─ Secondary Intents: {[i.value for i in result.secondary_intents]}"
        )
        logger.info(f"   ├─ Complexity: {result.complexity_level.value}")
        logger.info(f"   ├─ Confidence: {result.confidence_score:.2f}")
        logger.info(f"   ├─ Video Search: {result.needs_video_search}")
        logger.info(f"   ├─ Text Search: {result.needs_text_search}")
        logger.info(f"   ├─ Visual Analysis: {result.needs_visual_analysis}")
        logger.info(f"   ├─ Workflow: {result.recommended_workflow}")
        logger.info(f"   ├─ Required Agents: {result.required_agents}")
        logger.info(
            f"   └─ Entities: {len(result.entities)}, Keywords: {len(result.keywords)}"
        )

        if result.temporal_filters:
            logger.info(f"   └─ Temporal Filters: {result.temporal_filters}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage and performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_analyses": self.total_analyses,
            "uptime_seconds": uptime,
            "analyses_per_minute": (
                (self.total_analyses / uptime * 60) if uptime > 0 else 0
            ),
            "configuration": {
                "thinking_phase_enabled": self.enable_thinking_phase,
                "query_expansion_enabled": self.enable_query_expansion,
                "agent_integration_enabled": self.enable_agent_integration,
                "routing_agent_available": self.routing_agent is not None,
            },
        }


def create_enhanced_query_analyzer(**kwargs) -> QueryAnalysisToolV3:
    """
    Factory function to create an enhanced query analyzer.

    Args:
        **kwargs: Configuration options

    Returns:
        Configured QueryAnalysisToolV3 instance
    """
    return QueryAnalysisToolV3(**kwargs)


# Example usage and integration
async def example_usage():
    """Example of how to use the enhanced query analysis tool"""
    # Create the enhanced analyzer
    analyzer = create_enhanced_query_analyzer(
        enable_thinking_phase=True,
        enable_query_expansion=True,
        enable_agent_integration=True,
    )

    # Example queries of different complexities
    queries = [
        "Show me recent videos about machine learning",  # Simple
        "Compare deep learning frameworks and summarize key differences",  # Moderate
        "Analyze all content about AI ethics from last month and create a detailed report with visual analysis",  # Complex
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        # Analyze the query
        result = await analyzer.analyze(query)

        # Print key results
        print(f"Primary Intent: {result.primary_intent.value}")
        print(f"Complexity: {result.complexity_level.value}")
        print(f"Confidence: {result.confidence_score:.2%}")
        print(f"Workflow: {result.recommended_workflow}")
        print(f"Agents: {', '.join(result.required_agents)}")

        if result.expanded_queries:
            print(f"Expanded Queries: {result.expanded_queries}")

        if result.entities:
            print(f"Entities: {[e['text'] for e in result.entities]}")

        print(f"Thinking: {result.thinking_phase.get('reasoning', 'N/A')}")

    # Get statistics
    stats = analyzer.get_statistics()
    print("\nAnalyzer Statistics:")
    print(f"  Total Analyses: {stats['total_analyses']}")
    print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
    print(f"  Rate: {stats['analyses_per_minute']:.1f} analyses/min")


if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())
