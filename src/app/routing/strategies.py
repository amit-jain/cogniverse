# src/routing/strategies.py
"""
Implementation of various routing strategies for the comprehensive routing system.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from .base import (
    GenerationType,
    RoutingDecision,
    RoutingStrategy,
    SearchModality,
    TemporalExtractor,
)

logger = logging.getLogger(__name__)


class GLiNERRoutingStrategy(RoutingStrategy):
    """
    Routing strategy using GLiNER for named entity recognition.
    Fast and efficient for entity-based routing.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model = None
        self.labels = config.get(
            "gliner_labels",
            [
                "video_content",
                "visual_content",
                "media_content",
                "document_content",
                "text_information",
                "written_content",
                "summary_request",
                "detailed_analysis",
                "report_request",
                "time_reference",
                "date_pattern",
                "temporal_context",
            ],
        )
        self.threshold = config.get("gliner_threshold", 0.3)
        self.model_name = config.get("gliner_model", "urchade/gliner_large-v2.1")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the GLiNER model."""
        try:
            from gliner import GLiNER

            print(f"🔍 Attempting to load GLiNER model: {self.model_name}")
            logger.info(f"Loading GLiNER model: {self.model_name}")
            self.model = GLiNER.from_pretrained(self.model_name)
            print(f"✅ GLiNER model loaded successfully: {self.model_name}")
            logger.info("GLiNER model loaded successfully")
        except ImportError as e:
            print(f"❌ GLiNER import error: {e}")
            logger.warning("GLiNER not available, strategy will be disabled")
            self.model = None
        except Exception as e:
            print(f"❌ Failed to load GLiNER model: {e}")
            logger.error(f"Failed to load GLiNER model: {e}")
            self.model = None

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """Route using GLiNER entity extraction."""
        start_time = time.time()

        if not self.model:
            # Fallback if model not available
            return RoutingDecision(
                search_modality=SearchModality.BOTH,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.0,
                routing_method="gliner_unavailable",
                reasoning="GLiNER model not available",
            )

        try:
            # Extract entities
            entities = self.model.predict_entities(
                query, self.labels, threshold=self.threshold
            )

            # Debug output
            if entities:
                print(
                    f"🔍 GLiNER entities: {[(e['label'], round(e.get('score', 0), 2)) for e in entities]}"
                )

            # Analyze entities for routing decision
            decision = self._analyze_entities(query, entities)

            # Extract temporal information
            temporal_info = TemporalExtractor.extract_temporal_info(query)
            if temporal_info:
                decision.temporal_info = temporal_info

            # Don't override the carefully calculated confidence from _analyze_entities
            print(f"🎯 GLiNER confidence: {decision.confidence_score:.2f}")

            execution_time = time.time() - start_time
            self.record_metrics(query, decision, execution_time, True)

            return decision

        except Exception as e:
            logger.error(f"GLiNER routing failed: {e}")
            execution_time = time.time() - start_time
            fallback_decision = RoutingDecision(
                search_modality=SearchModality.BOTH,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.0,
                routing_method="gliner_error",
                reasoning=f"Error: {str(e)}",
            )
            self.record_metrics(query, fallback_decision, execution_time, False, str(e))
            return fallback_decision

    def _analyze_entities(self, query: str, entities: list[dict]) -> RoutingDecision:
        """Analyze extracted entities to make routing decision."""
        video_score = 0
        text_score = 0
        generation_type = GenerationType.RAW_RESULTS

        entity_types = [e["label"] for e in entities]

        # Count entity types
        for entity in entities:
            label = entity["label"]
            if label in ["video_content", "visual_content", "media_content"]:
                video_score += entity.get("score", 1.0)
            elif label in ["document_content", "text_information", "written_content"]:
                text_score += entity.get("score", 1.0)
            elif label in ["summary_request"]:
                generation_type = GenerationType.SUMMARY
            elif label in ["detailed_analysis", "report_request"]:
                generation_type = GenerationType.DETAILED_REPORT

        # Determine search modality
        if video_score > 0 and text_score > 0:
            search_modality = SearchModality.BOTH
        elif video_score > text_score:
            search_modality = SearchModality.VIDEO
        elif text_score > video_score:
            search_modality = SearchModality.TEXT
        else:
            # Default based on query keywords if no clear entities
            search_modality = self._fallback_modality_detection(query)

        # Check for relationship indicators that GLiNER cannot handle well
        relationship_indicators = [
            "relate",
            "related",
            "connection",
            "between",
            "compare",
            "correlation",
            "similar",
            "difference",
            "contrast",
            "versus",
            "vs",
            "how does",
            "what's the",
            "evolved",
            "changed",
            "pattern",
            "trend",
            "analyze",
            "synthesize",
            "infer",
            "underlying",
            "philosophical",
            "implications",
            "yesterday",
            "previous",
            "last",
            "evolution",
            "contradiction",
        ]

        query_lower = query.lower()
        has_relationship = any(
            indicator in query_lower for indicator in relationship_indicators
        )

        # Calculate confidence based on entity scores
        total_score = video_score + text_score

        # Base confidence from entity detection
        if len(entities) > 0:
            # If query requires relationship extraction, GLiNER should have lower confidence
            if has_relationship:
                # GLiNER detected entities but can't handle relationships well
                confidence = 0.4 + (
                    0.2 * min(total_score, 1.0)
                )  # 0.4-0.6 range - below threshold
            elif video_score > 0 or text_score > 0:
                # Simple entity queries - GLiNER handles these well
                confidence = 0.7 + (0.3 * min(total_score, 1.0))  # 0.7-1.0 range
            elif any(
                e["label"] in ["summary_request", "detailed_analysis", "report_request"]
                for e in entities
            ):
                # Generation type entities also give good confidence
                confidence = 0.75
            else:
                # Other entities provide moderate confidence
                confidence = 0.5 + (0.2 * len(entities))
        else:
            # No entities detected
            if has_relationship:
                confidence = (
                    0.2  # Very low confidence for relationship queries without entities
                )
            else:
                confidence = 0.3  # Low confidence if no entities

        return RoutingDecision(
            search_modality=search_modality,
            generation_type=generation_type,
            confidence_score=confidence,
            routing_method="gliner",
            entities_detected=entities,
            reasoning=f"Detected entities: {entity_types}",
        )

    def _fallback_modality_detection(self, query: str) -> SearchModality:
        """Simple keyword-based fallback for modality detection."""
        query_lower = query.lower()
        video_keywords = ["video", "show", "watch", "visual", "demonstration"]
        text_keywords = ["document", "article", "text", "paper", "report"]

        has_video = any(kw in query_lower for kw in video_keywords)
        has_text = any(kw in query_lower for kw in text_keywords)

        if has_video and has_text:
            return SearchModality.BOTH
        elif has_video:
            return SearchModality.VIDEO
        elif has_text:
            return SearchModality.TEXT
        else:
            return SearchModality.BOTH

    def get_confidence(self, query: str, decision: RoutingDecision) -> float:
        """Calculate confidence based on entity scores."""
        if not decision.entities_detected:
            return 0.3

        scores = [e.get("score", 0.5) for e in decision.entities_detected]
        if not scores:
            return 0.3

        # Average score weighted by number of entities
        avg_score = sum(scores) / len(scores)
        entity_count_factor = min(
            len(scores) / 3, 1.0
        )  # More entities = higher confidence

        return min(avg_score * (0.7 + 0.3 * entity_count_factor), 1.0)


class LLMRoutingStrategy(RoutingStrategy):
    """
    Routing strategy using Large Language Models.
    More sophisticated but slower than other methods.
    Integrates with DSPy optimization system for improved prompts.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.provider = config.get("provider", "local")
        self.model = config.get("model", "gemma2:2b")
        self.endpoint = config.get("endpoint", "http://localhost:11434")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 150)

        # DSPy optimization support
        self.dspy_enabled = config.get("enable_dspy_optimization", False)
        self.optimized_prompts = {}
        self._load_optimized_prompts()

        # Fallback to base prompt if no optimization available
        self.system_prompt = self._get_system_prompt()

    def _load_optimized_prompts(self):
        """Load DSPy-optimized prompts if available."""
        if not self.dspy_enabled:
            return

        try:
            import json
            from pathlib import Path

            # Look for optimized prompts in standard locations
            search_paths = [
                Path("optimized_prompts/routing_prompts.json"),
                Path("src/app/routing/optimized_prompts/routing_prompts.json"),
                Path("routing_prompts.json"),
            ]

            for prompt_file in search_paths:
                if prompt_file.exists():
                    with open(prompt_file, "r") as f:
                        self.optimized_prompts = json.load(f)

                    logger.info(f"Loaded DSPy optimized prompts from {prompt_file}")
                    return

            logger.info("No DSPy optimized routing prompts found, using default prompt")

        except Exception as e:
            logger.warning(f"Failed to load DSPy optimized prompts: {e}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for routing, using DSPy optimization if available."""
        # Use optimized prompt if available
        if self.dspy_enabled and "system_prompt" in self.optimized_prompts:
            optimized_prompt = self.optimized_prompts["system_prompt"]
            logger.debug("Using DSPy-optimized routing prompt")
            return optimized_prompt

        # Fallback to manually improved prompt (better than the original)
        return """You are a precise routing agent for a multi-modal search system.
Analyze the user query and determine:
1. search_modality: Choose ONE of: "video", "text", or "both"
2. generation_type: Choose ONE of: "raw_results", "summary", or "detailed_report"

Rules:
- Use "video" for visual content, demonstrations, tutorials, presentations
- Use "text" for documents, articles, written information, reports
- Use "both" when unclear or when both might be relevant
- Use "raw_results" for simple searches
- Use "summary" for brief overviews
- Use "detailed_report" for comprehensive analysis

Examples:
Query: "show me videos about cats" → {"search_modality": "video", "generation_type": "raw_results", "reasoning": "looking for video content"}
Query: "summarize the document" → {"search_modality": "text", "generation_type": "summary", "reasoning": "text summary requested"}
Query: "detailed analysis of both video and text" → {"search_modality": "both", "generation_type": "detailed_report", "reasoning": "comprehensive analysis needed"}

Respond ONLY with a valid JSON object like the examples above."""

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """Route using LLM inference."""
        start_time = time.time()

        try:
            # Build the prompt
            prompt = self._build_prompt(query, context)
            # Call LLM
            response = await self._call_llm(prompt)

            # Parse response
            decision = self._parse_llm_response(response, query)

            # Extract temporal information
            temporal_info = TemporalExtractor.extract_temporal_info(query)
            if temporal_info:
                decision.temporal_info = temporal_info

            # Calculate confidence
            decision.confidence_score = self.get_confidence(query, decision)

            execution_time = time.time() - start_time
            self.record_metrics(query, decision, execution_time, True)

            return decision

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            execution_time = time.time() - start_time
            fallback_decision = RoutingDecision(
                search_modality=SearchModality.BOTH,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.0,
                routing_method="llm_error",
                reasoning=f"Error: {str(e)}",
            )
            self.record_metrics(query, fallback_decision, execution_time, False, str(e))
            return fallback_decision

    def _build_prompt(self, query: str, context: dict[str, Any] | None) -> str:
        """Build the prompt for the LLM, applying DSPy optimization if available."""
        conversation_history = ""
        if context and "conversation_history" in context:
            conversation_history = (
                f"\nConversation history:\n{context['conversation_history']}\n"
            )

        # Use DSPy-optimized prompt building if available
        if self.dspy_enabled and "prompt_template" in self.optimized_prompts:
            try:
                template = self.optimized_prompts["prompt_template"]
                return template.format(
                    system_prompt=self.system_prompt,
                    conversation_history=conversation_history,
                    query=query,
                )
            except Exception as e:
                logger.warning(f"Failed to apply DSPy optimized prompt template: {e}")

        # Fallback to standard prompt building
        return f"{self.system_prompt}\n{conversation_history}\nUser query: {query}\n\nResponse:"

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM endpoint."""
        import aiohttp

        if self.provider == "local":
            url = f"{self.endpoint}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                "stream": False,
            }
        else:  # modal or other providers
            url = self.endpoint
            payload = {
                "prompt": prompt,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                result = await response.json()

                if self.provider == "local":
                    return result.get("response", "")
                else:
                    return result.get("text", "")

    def _parse_llm_response(self, response: str, query: str) -> RoutingDecision:
        """Parse the LLM response to extract routing decision."""
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in LLM response: {response}")

        try:
            routing_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in LLM response: {json_match.group()}"
            ) from e

        # Validate required fields
        if "search_modality" not in routing_data:
            raise ValueError(
                f"Missing 'search_modality' in LLM response: {routing_data}"
            )
        if "generation_type" not in routing_data:
            raise ValueError(
                f"Missing 'generation_type' in LLM response: {routing_data}"
            )

        # Map to enums with strict validation
        try:
            search_modality = SearchModality(routing_data["search_modality"])
        except ValueError as e:
            raise ValueError(
                f"Invalid search_modality '{routing_data['search_modality']}' in LLM response"
            ) from e

        try:
            generation_type = GenerationType(routing_data["generation_type"])
        except ValueError as e:
            raise ValueError(
                f"Invalid generation_type '{routing_data['generation_type']}' in LLM response"
            ) from e

        reasoning = routing_data.get("reasoning", "LLM routing decision")

        return RoutingDecision(
            search_modality=search_modality,
            generation_type=generation_type,
            routing_method="llm",
            reasoning=reasoning,
        )

    def get_confidence(self, query: str, decision: RoutingDecision) -> float:
        """Calculate confidence for LLM decisions."""
        query_lower = query.lower()

        # Queries that LLM is less confident about (will escalate to Tier 3)
        low_confidence_indicators = [
            "extract specific",
            "parse",
            "structured data",
            "schema",
            "exact format",
            "json",
            "api response",
            "technical specification",
            "regulatory compliance",
            "legal requirements",
            "medical diagnosis",
            "precise extraction",
            "strict format",
            "validate against",
        ]

        base_confidence = 0.6
        if any(indicator in query_lower for indicator in low_confidence_indicators):
            # Lower confidence for structured/technical queries that need Tier 3
            base_confidence = 0.55  # Below 0.6 threshold, will escalate to LangExtract
        elif decision.reasoning and len(decision.reasoning) > 20:
            base_confidence = 0.85
        elif decision.reasoning:
            base_confidence = 0.75

        # Boost confidence if using DSPy-optimized prompts
        if self.dspy_enabled and self.optimized_prompts:
            base_confidence = min(base_confidence * 1.1, 1.0)  # 10% confidence boost

        return base_confidence

    def enable_dspy_optimization(self, enabled: bool = True):
        """Enable or disable DSPy optimization at runtime."""
        self.dspy_enabled = enabled
        if enabled:
            self._load_optimized_prompts()
            self.system_prompt = self._get_system_prompt()
            logger.info("DSPy optimization enabled for LLM routing strategy")
        else:
            self.optimized_prompts = {}
            self.system_prompt = self._get_system_prompt()
            logger.info("DSPy optimization disabled for LLM routing strategy")

    def get_optimization_status(self) -> dict[str, Any]:
        """Get the current DSPy optimization status."""
        return {
            "dspy_enabled": self.dspy_enabled,
            "optimized_prompts_loaded": len(self.optimized_prompts) > 0,
            "available_optimizations": list(self.optimized_prompts.keys()),
            "using_optimized_system_prompt": self.dspy_enabled
            and "system_prompt" in self.optimized_prompts,
        }


class KeywordRoutingStrategy(RoutingStrategy):
    """
    Simple keyword-based routing strategy.
    Fast and deterministic but less sophisticated.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.video_keywords = config.get(
            "video_keywords",
            [
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
                "tutorial",
                "screencast",
                "webinar",
            ],
        )
        self.text_keywords = config.get(
            "text_keywords",
            [
                "document",
                "report",
                "text",
                "article",
                "information",
                "data",
                "details",
                "analysis",
                "research",
                "study",
                "paper",
                "blog",
                "documentation",
                "guide",
                "manual",
                "transcript",
                "transcription",
                "caption",
                "subtitle",
                "written",
                "read",
            ],
        )
        self.summary_keywords = config.get(
            "summary_keywords",
            [
                "summary",
                "summarize",
                "brief",
                "overview",
                "main points",
                "key takeaways",
                "tldr",
                "gist",
                "essence",
            ],
        )
        self.report_keywords = config.get(
            "report_keywords",
            [
                "detailed report",
                "comprehensive analysis",
                "full report",
                "in-depth",
                "thorough",
                "extensive",
                "complete analysis",
            ],
        )

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """Route based on keyword matching."""
        start_time = time.time()
        query_lower = query.lower()

        # Detect search modality
        has_video = any(kw in query_lower for kw in self.video_keywords)
        has_text = any(kw in query_lower for kw in self.text_keywords)

        # Check for comparison keywords that imply both modalities
        comparison_keywords = ["compare", "versus", "vs", "difference between", "both"]
        has_comparison = any(kw in query_lower for kw in comparison_keywords)

        # If comparison detected with at least one modality, assume both
        if has_comparison and (has_video or has_text):
            search_modality = SearchModality.BOTH
        elif has_video and has_text:
            search_modality = SearchModality.BOTH
        elif has_video:
            search_modality = SearchModality.VIDEO
        elif has_text:
            search_modality = SearchModality.TEXT
        else:
            # Default to both if no clear indicators
            search_modality = SearchModality.BOTH

        # Detect generation type
        if any(kw in query_lower for kw in self.report_keywords):
            generation_type = GenerationType.DETAILED_REPORT
        elif any(kw in query_lower for kw in self.summary_keywords):
            generation_type = GenerationType.SUMMARY
        else:
            generation_type = GenerationType.RAW_RESULTS

        # Extract temporal information
        temporal_info = TemporalExtractor.extract_temporal_info(query)

        decision = RoutingDecision(
            search_modality=search_modality,
            generation_type=generation_type,
            routing_method="keyword",
            temporal_info=temporal_info,
            reasoning=f"Matched keywords: video={has_video}, text={has_text}",
        )

        decision.confidence_score = self.get_confidence(query, decision)

        execution_time = time.time() - start_time
        self.record_metrics(query, decision, execution_time, True)

        return decision

    def get_confidence(self, query: str, decision: RoutingDecision) -> float:
        """Calculate confidence based on keyword matches."""
        query_lower = query.lower()
        match_count = 0

        for kw in self.video_keywords + self.text_keywords:
            if kw in query_lower:
                match_count += 1

        # More matches = higher confidence
        if match_count == 0:
            return 0.3  # No matches, low confidence
        elif match_count == 1:
            return 0.6  # Single match, moderate confidence
        elif match_count == 2:
            return 0.75  # Two matches, good confidence
        else:
            return 0.85  # Multiple matches, high confidence


class HybridRoutingStrategy(RoutingStrategy):
    """
    Hybrid routing strategy that combines multiple strategies.
    Uses GLiNER first, falls back to LLM for low confidence, and uses keywords as final fallback.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.gliner_strategy = GLiNERRoutingStrategy(config)
        self.llm_strategy = LLMRoutingStrategy(config)
        self.keyword_strategy = KeywordRoutingStrategy(config)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.use_llm_fallback = config.get("use_llm_fallback", True)

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """Route using hybrid approach."""
        start_time = time.time()

        # Try GLiNER first (fast)
        gliner_decision = await self.gliner_strategy.route(query, context)

        # If GLiNER has high confidence, use it
        if gliner_decision.confidence_score >= self.confidence_threshold:
            gliner_decision.routing_method = "hybrid_gliner"
            execution_time = time.time() - start_time
            self.record_metrics(query, gliner_decision, execution_time, True)
            return gliner_decision

        # If GLiNER confidence is low and LLM is enabled, try LLM
        if self.use_llm_fallback:
            try:
                llm_decision = await self.llm_strategy.route(query, context)
                if llm_decision.confidence_score >= self.confidence_threshold:
                    llm_decision.routing_method = "hybrid_llm"
                    execution_time = time.time() - start_time
                    self.record_metrics(query, llm_decision, execution_time, True)
                    return llm_decision
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")

        # Final fallback to keywords
        keyword_decision = await self.keyword_strategy.route(query, context)
        keyword_decision.routing_method = "hybrid_keyword"

        execution_time = time.time() - start_time
        self.record_metrics(query, keyword_decision, execution_time, True)

        return keyword_decision

    def get_confidence(self, query: str, decision: RoutingDecision) -> float:
        """Return the confidence from the underlying strategy."""
        return decision.confidence_score


class EnsembleRoutingStrategy(RoutingStrategy):
    """
    Ensemble routing strategy that combines predictions from multiple strategies.
    Uses voting or weighted averaging to make final decision.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.strategies = []
        self.weights = config.get("weights", {})
        self.voting_method = config.get(
            "voting_method", "weighted"
        )  # "weighted" or "majority"
        self._initialize_strategies(config)

    def _initialize_strategies(self, config: dict[str, Any]):
        """Initialize the ensemble strategies."""
        enabled_strategies = config.get("enabled_strategies", ["gliner", "keyword"])

        if "gliner" in enabled_strategies:
            self.strategies.append(("gliner", GLiNERRoutingStrategy(config)))
        if "llm" in enabled_strategies:
            self.strategies.append(("llm", LLMRoutingStrategy(config)))
        if "keyword" in enabled_strategies:
            self.strategies.append(("keyword", KeywordRoutingStrategy(config)))

        # Set default weights if not provided
        for name, _ in self.strategies:
            if name not in self.weights:
                self.weights[name] = 1.0

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """Route using ensemble of strategies."""
        start_time = time.time()

        # Collect decisions from all strategies
        decisions = []
        tasks = []

        for _name, strategy in self.strategies:
            tasks.append(strategy.route(query, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (name, _strategy), result in zip(self.strategies, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(f"Strategy {name} failed: {result}")
                continue
            decisions.append((name, result))

        if not decisions:
            # No strategies succeeded, return fallback
            fallback_decision = RoutingDecision(
                search_modality=SearchModality.BOTH,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.0,
                routing_method="ensemble_fallback",
                reasoning="All strategies failed",
            )
            execution_time = time.time() - start_time
            self.record_metrics(
                query, fallback_decision, execution_time, False, "All strategies failed"
            )
            return fallback_decision

        # Combine decisions
        final_decision = self._combine_decisions(decisions)
        final_decision.routing_method = "ensemble"

        # Extract temporal information (use first non-None)
        for _, decision in decisions:
            if decision.temporal_info:
                final_decision.temporal_info = decision.temporal_info
                break

        execution_time = time.time() - start_time
        self.record_metrics(query, final_decision, execution_time, True)

        return final_decision

    def _combine_decisions(
        self, decisions: list[tuple[str, RoutingDecision]]
    ) -> RoutingDecision:
        """Combine multiple decisions into a final decision."""
        if self.voting_method == "majority":
            return self._majority_voting(decisions)
        else:
            return self._weighted_voting(decisions)

    def _majority_voting(
        self, decisions: list[tuple[str, RoutingDecision]]
    ) -> RoutingDecision:
        """Use majority voting to combine decisions."""
        from collections import Counter

        modality_votes = Counter()
        generation_votes = Counter()

        for _name, decision in decisions:
            modality_votes[decision.search_modality] += 1
            generation_votes[decision.generation_type] += 1

        # Get most common choices
        search_modality = modality_votes.most_common(1)[0][0]
        generation_type = generation_votes.most_common(1)[0][0]

        # Average confidence scores
        avg_confidence = sum(d.confidence_score for _, d in decisions) / len(decisions)

        # Combine reasoning
        reasoning = "; ".join(
            f"{name}: {d.reasoning}" for name, d in decisions if d.reasoning
        )

        return RoutingDecision(
            search_modality=search_modality,
            generation_type=generation_type,
            confidence_score=avg_confidence,
            reasoning=reasoning,
        )

    def _weighted_voting(
        self, decisions: list[tuple[str, RoutingDecision]]
    ) -> RoutingDecision:
        """Use weighted voting to combine decisions."""
        from collections import defaultdict

        modality_scores = defaultdict(float)
        generation_scores = defaultdict(float)
        total_weight = 0

        for name, decision in decisions:
            weight = self.weights.get(name, 1.0) * decision.confidence_score
            modality_scores[decision.search_modality] += weight
            generation_scores[decision.generation_type] += weight
            total_weight += weight

        if total_weight == 0:
            # Fallback to majority voting if weights are all zero
            return self._majority_voting(decisions)

        # Get highest scoring choices
        search_modality = max(modality_scores.items(), key=lambda x: x[1])[0]
        generation_type = max(generation_scores.items(), key=lambda x: x[1])[0]

        # Weighted average confidence
        weighted_confidence = sum(
            d.confidence_score * self.weights.get(name, 1.0) for name, d in decisions
        ) / sum(self.weights.get(name, 1.0) for name, _ in decisions)

        # Combine reasoning
        reasoning = "; ".join(
            f"{name}: {d.reasoning}" for name, d in decisions if d.reasoning
        )

        return RoutingDecision(
            search_modality=search_modality,
            generation_type=generation_type,
            confidence_score=weighted_confidence,
            reasoning=reasoning,
        )

    def get_confidence(self, query: str, decision: RoutingDecision) -> float:
        """Return the combined confidence score."""
        return decision.confidence_score


class LangExtractRoutingStrategy(RoutingStrategy):
    """
    Routing strategy using structured extraction with local LLM (via Ollama).
    Provides more structured extraction than basic LLM but uses local models.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # Use local Ollama model instead of Gemini
        self.model_name = config.get("langextract_model", "smollm2:1.7b")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.extractor = None
        self.schema_prompt = """
        Classify query generation type by looking for EXACT keywords:

        If query contains these words → use "raw":
        - extract, get, list, parse
        - timestamps, IDs, JSON, fields
        - specific, exact, all

        If query contains these words → use "summary":
        - summarize, summary
        - brief, overview, gist
        - key, main, points, takeaways
        - quick, recap

        If query contains these words → use "detailed":
        - detailed, comprehensive
        - full, complete, thorough
        - in-depth, analysis, breakdown
        - report, investigation

        DEFAULT: If no keywords match → use "raw"

        Video/Text detection:
        - needs_video: true if mentions video/visual/watch/show
        - needs_text: true if mentions text/document/article/report
        - If neither mentioned: both = true
        """
        self._initialize_extractor()

    def _initialize_extractor(self):
        """Initialize the structured extraction using Ollama."""
        # Skip initialization in CI environment
        import os

        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            print("⏭️ Skipping structured extraction initialization in CI")
            self.extractor = None
            return

        try:
            import httpx

            # Check if Ollama is running and model is available
            client = httpx.Client(timeout=5.0)
            response = client.get(f"{self.ollama_url}/api/tags")

            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in Ollama")
                    print(
                        f"⚠️ Model {self.model_name} not found, available: {model_names}"
                    )
                    self.extractor = None
                    return

                print(
                    f"🔧 Initializing structured extraction with Ollama model: {self.model_name}"
                )
                self.extractor = True  # Just mark as available
                print("✅ Structured extraction initialized successfully")
                logger.info("Structured extraction initialized successfully")
            else:
                logger.warning("Ollama not responding")
                self.extractor = None

        except Exception as e:
            print(f"❌ Failed to initialize structured extraction: {e}")
            logger.error(f"Failed to initialize structured extraction: {e}")
            self.extractor = None

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """Route using structured extraction with local LLM."""
        start_time = time.time()

        if not self.extractor:
            # Fallback if extractor not available
            logger.info("Structured extraction unavailable, using fallback")
            return RoutingDecision(
                search_modality=SearchModality.BOTH,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.1,
                routing_method="langextract_unavailable",
                reasoning="Structured extraction unavailable",
            )

        try:
            # Create a structured prompt for Ollama
            structured_prompt = f"""{self.schema_prompt}

Query: {query}

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{{
    "needs_video": true or false,
    "needs_text": true or false,
    "generation_type": "raw" or "summary" or "detailed",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

STRICT RULES - Check in this order:
1. If query has "extract" or "get" or "list" → generation_type: "raw"
2. If query has "summarize" or "summary" → generation_type: "summary"
3. If query has "detailed" or "comprehensive" → generation_type: "detailed"
4. Otherwise → generation_type: "raw"

Do not include any explanation or text outside the JSON object."""

            logger.debug(f"Structured extraction with model: {self.model_name}")

            # Call Ollama API
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": structured_prompt,
                        "stream": False,
                        "temperature": 0.3,  # Lower temperature for more consistent structured output
                        "format": "json",  # Request JSON format
                    },
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code}")

                result_text = response.json().get("response", "{}")
                logger.debug(f"Structured extraction raw response: {result_text[:200]}")

                # Parse the JSON response
                try:
                    result = json.loads(result_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from the response
                    json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                    else:
                        raise ValueError(
                            "Could not parse LangExtract response"
                        ) from None

            # Determine search modality
            needs_video = result.get("needs_video", False)
            needs_text = result.get("needs_text", False)

            if needs_video and needs_text:
                search_modality = SearchModality.BOTH
            elif needs_video:
                search_modality = SearchModality.VIDEO
            elif needs_text:
                search_modality = SearchModality.TEXT
            else:
                search_modality = SearchModality.BOTH  # Default to both if unclear

            # Determine generation type
            gen_type_str = result.get("generation_type", "raw").lower()
            if gen_type_str == "summary":
                generation_type = GenerationType.SUMMARY
            elif gen_type_str == "detailed" or gen_type_str == "detailed_report":
                generation_type = GenerationType.DETAILED_REPORT
            else:
                generation_type = GenerationType.RAW_RESULTS

            # Get confidence and reasoning
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "LangExtract analysis")

            # Extract temporal information
            temporal_info = TemporalExtractor.extract_temporal_info(query)

            decision = RoutingDecision(
                search_modality=search_modality,
                generation_type=generation_type,
                confidence_score=confidence,
                routing_method="langextract",
                reasoning=reasoning,
                temporal_info=temporal_info,
            )

            execution_time = time.time() - start_time
            self.record_metrics(query, decision, execution_time, True)

            logger.debug(
                f"LangExtract routing: {search_modality.value}, {generation_type.value}, confidence: {confidence:.2f}"
            )

            return decision

        except Exception as e:
            logger.error(f"LangExtract routing failed: {e}")

            # Return fallback decision
            fallback_decision = RoutingDecision(
                search_modality=SearchModality.BOTH,
                generation_type=GenerationType.RAW_RESULTS,
                confidence_score=0.0,
                routing_method="langextract_error",
                reasoning=f"LangExtract failed: {str(e)}",
            )

            execution_time = time.time() - start_time
            self.record_metrics(query, fallback_decision, execution_time, False, str(e))

            return fallback_decision

    def get_confidence(self, query: str, decision: RoutingDecision) -> float:
        """Return the confidence score from LangExtract."""
        return decision.confidence_score
