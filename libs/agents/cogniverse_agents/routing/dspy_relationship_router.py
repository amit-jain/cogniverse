"""
DSPy 3.0 Relationship Router Module

This module integrates DSPy 3.0 signatures with relationship extraction tools
to create an intelligent routing system that leverages entity and relationship
information for enhanced query analysis and routing decisions.

The ComposableQueryAnalysisModule provides two paths:
- Path A (GLiNER fast path): GLiNER extracts high-confidence entities, heuristic
  relationships are inferred, and the LLM only reformulates the query.
- Path B (LLM unified path): A single LLM call performs entity extraction,
  relationship inference, and query reformulation together.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import dspy

from .dspy_routing_signatures import (
    AdvancedRoutingSignature,
    BasicQueryAnalysisSignature,
    QueryReformulationSignature,
    UnifiedExtractionReformulationSignature,
)
from .relationship_extraction_tools import (
    GLiNERRelationshipExtractor,
    SpaCyDependencyAnalyzer,
)

logger = logging.getLogger(__name__)


class ComposableQueryAnalysisModule(dspy.Module):
    """
    Composable DSPy module that combines entity extraction, relationship inference,
    and query reformulation into a single optimizable unit.

    Two paths:
    - Path A (GLiNER fast path): GLiNER entities with avg confidence >= threshold
      → heuristic relationships → LLM reformulation only
    - Path B (LLM unified path): Single LLM call does everything

    Both paths produce the same output shape and are individually optimizable
    via DSPy optimizers (SIMBA, MIPROv2, BootstrapFewShot, GEPA).
    """

    def __init__(
        self,
        gliner_extractor: GLiNERRelationshipExtractor,
        spacy_analyzer: SpaCyDependencyAnalyzer,
        entity_confidence_threshold: float = 0.6,
        min_entities_for_fast_path: int = 1,
    ):
        super().__init__()
        self.gliner_extractor = gliner_extractor
        self.spacy_analyzer = spacy_analyzer
        self.entity_confidence_threshold = entity_confidence_threshold
        self.min_entities_for_fast_path = min_entities_for_fast_path

        # Path A: reformulation only (entities already extracted by GLiNER)
        self.reformulator = dspy.ChainOfThought(QueryReformulationSignature)
        # Path B: unified extraction + reformulation
        self.unified_extractor = dspy.ChainOfThought(
            UnifiedExtractionReformulationSignature
        )

    def forward(self, query: str, search_context: str = "general") -> dspy.Prediction:
        """
        Analyze query: extract entities, infer relationships, and generate
        enhanced query with variants.

        Args:
            query: User query to analyze
            search_context: Search context (general, video, text, multimodal)

        Returns:
            dspy.Prediction with entities, relationships, enhanced_query,
            query_variants, confidence, path_used, domain_classification
        """
        try:
            # Step 1: Try GLiNER entity extraction (sync, fast)
            entities = self.gliner_extractor.extract_entities(query)

            # Step 2: Compute average confidence
            avg_confidence = 0.0
            if entities:
                avg_confidence = sum(e["confidence"] for e in entities) / len(entities)

            # Step 3: Path decision
            gliner_available = self.gliner_extractor.gliner_model is not None
            has_enough_entities = len(entities) >= self.min_entities_for_fast_path
            confidence_above_threshold = (
                avg_confidence >= self.entity_confidence_threshold
            )

            if gliner_available and has_enough_entities and confidence_above_threshold:
                return self._path_a(query, entities, search_context)
            else:
                return self._path_b(query, search_context)

        except Exception as e:
            logger.error(f"ComposableQueryAnalysisModule failed: {e}")
            return self._fallback_prediction(query)

    def _path_a(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        search_context: str,
    ) -> dspy.Prediction:
        """
        Path A: GLiNER fast path.

        GLiNER already extracted high-confidence entities. Infer relationships
        heuristically, enrich with spaCy, then call LLM for reformulation only.
        """
        # Infer relationships from GLiNER entities
        gliner_relationships = self.gliner_extractor.infer_relationships_from_entities(
            query, entities
        )

        # Enrich with spaCy semantic relationships
        spacy_relationships = self.spacy_analyzer.extract_semantic_relationships(query)

        # Deduplicate relationships
        all_relationships = gliner_relationships + spacy_relationships
        relationships = self._deduplicate_relationships(all_relationships)

        # Call LLM reformulator with pre-extracted entities and relationships
        entities_json = json.dumps(entities)
        relationships_json = json.dumps(relationships)

        try:
            result = self.reformulator(
                original_query=query,
                entities=entities_json,
                relationships=relationships_json,
                search_context=search_context,
            )

            query_variants = self._parse_json_field(result.query_variants, [])
            confidence = self._parse_confidence(result.confidence)

            prediction = dspy.Prediction()
            prediction.entities = entities
            prediction.relationships = relationships
            prediction.enhanced_query = result.enhanced_query
            prediction.query_variants = query_variants
            prediction.confidence = confidence
            prediction.path_used = "gliner_fast_path"
            prediction.domain_classification = "unknown"
            prediction.reasoning = getattr(result, "reasoning", "")

            return prediction

        except Exception as e:
            logger.warning(f"Path A reformulation failed: {e}")
            raise

    def _path_b(self, query: str, search_context: str) -> dspy.Prediction:
        """
        Path B: LLM unified path.

        Single LLM call does entity extraction, relationship inference,
        and query reformulation together.
        """
        try:
            result = self.unified_extractor(
                original_query=query,
                search_context=search_context,
            )

            entities = self._parse_json_field(result.entities, [])
            relationships = self._parse_json_field(result.relationships, [])
            query_variants = self._parse_json_field(result.query_variants, [])
            confidence = self._parse_confidence(result.confidence)
            domain_classification = getattr(result, "domain_classification", "unknown")

            prediction = dspy.Prediction()
            prediction.entities = entities
            prediction.relationships = relationships
            prediction.enhanced_query = result.enhanced_query
            prediction.query_variants = query_variants
            prediction.confidence = confidence
            prediction.path_used = "llm_unified_path"
            prediction.domain_classification = domain_classification
            prediction.reasoning = getattr(result, "reasoning", "")

            return prediction

        except Exception as e:
            logger.warning(f"Path B unified extraction failed: {e}")
            return self._fallback_prediction(query)

    def _deduplicate_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate relationships based on subject-relation-object triples."""
        seen = set()
        deduplicated = []
        for rel in relationships:
            key = (
                rel.get("subject", "").lower(),
                rel.get("relation", "").lower(),
                rel.get("object", "").lower(),
            )
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
        return deduplicated

    def _parse_json_field(self, value: Any, default: Any) -> Any:
        """Parse a JSON string field from LLM output, returning default on failure."""
        if isinstance(value, (list, dict)):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, type(default)):
                    return parsed
                return default
            except (json.JSONDecodeError, TypeError):
                return default
        return default

    def _parse_confidence(self, value: Any) -> float:
        """Parse confidence from string or float."""
        if isinstance(value, (int, float)):
            return float(min(1.0, max(0.0, value)))
        if isinstance(value, str):
            try:
                return float(min(1.0, max(0.0, float(value))))
            except (ValueError, TypeError):
                return 0.5
        return 0.5

    def _fallback_prediction(self, query: str) -> dspy.Prediction:
        """Return safe fallback prediction when all paths fail."""
        prediction = dspy.Prediction()
        prediction.entities = []
        prediction.relationships = []
        prediction.enhanced_query = query
        prediction.query_variants = []
        prediction.confidence = 0.0
        prediction.path_used = "fallback"
        prediction.domain_classification = "unknown"
        prediction.reasoning = "All analysis paths failed"
        return prediction


class DSPyBasicRoutingModule(dspy.Module):
    """
    DSPy 3.0 module for basic query analysis and routing using BasicQueryAnalysisSignature.

    Provides fast routing decisions for simple queries without relationship analysis.
    """

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(BasicQueryAnalysisSignature)

    def forward(self, query: str, context: Optional[str] = None) -> dspy.Prediction:
        """
        Perform basic query analysis for routing decisions.

        Args:
            query: User query to analyze
            context: Optional conversation context

        Returns:
            DSPy prediction with routing analysis
        """
        try:
            # Analyze query characteristics
            analysis_result = self._analyze_query_characteristics(query, context)

            # Create prediction with analysis results
            prediction = dspy.Prediction()
            prediction.primary_intent = analysis_result["intent"]
            prediction.complexity_level = analysis_result["complexity"]
            prediction.needs_video_search = analysis_result["needs_video"]
            prediction.needs_text_search = analysis_result["needs_text"]
            prediction.needs_multimodal = analysis_result["needs_multimodal"]
            prediction.recommended_agent = analysis_result["agent"]
            prediction.confidence_score = analysis_result["confidence"]
            prediction.reasoning = analysis_result["reasoning"]

            return prediction

        except Exception as e:
            logger.error(f"Basic routing analysis failed: {e}")

            # Return fallback prediction
            prediction = dspy.Prediction()
            prediction.primary_intent = "search"
            prediction.complexity_level = "moderate"
            prediction.needs_video_search = True
            prediction.needs_text_search = False
            prediction.needs_multimodal = True
            prediction.recommended_agent = "video_search"
            prediction.confidence_score = 0.5
            prediction.reasoning = f"Fallback routing for query: {query[:50]}..."

            return prediction

    def _analyze_query_characteristics(
        self, query: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze query to determine routing characteristics.

        Args:
            query: User query
            context: Optional context

        Returns:
            Dictionary with analysis results
        """
        query_lower = query.lower()

        # Intent analysis
        intent = "search"  # Default
        if any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            intent = "compare"
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            intent = "summarize"
        elif any(word in query_lower for word in ["analyze", "analysis", "explain"]):
            intent = "analyze"
        elif any(
            word in query_lower for word in ["report", "details", "comprehensive"]
        ):
            intent = "report"

        # Complexity analysis
        complexity = "simple"
        if len(query.split()) > 10:
            complexity = "moderate"
        if len(query.split()) > 20 or any(
            word in query_lower for word in ["complex", "detailed", "comprehensive"]
        ):
            complexity = "complex"

        # Search requirements
        needs_video = any(
            word in query_lower
            for word in [
                "video",
                "show",
                "demonstration",
                "visual",
                "watch",
                "footage",
                "clip",
            ]
        )

        needs_text = any(
            word in query_lower
            for word in ["text", "document", "article", "paper", "read", "written"]
        )

        needs_multimodal = needs_video and (
            any(
                word in query_lower
                for word in ["describe", "explain", "analyze", "caption"]
            )
        )

        # Agent recommendation
        if intent == "compare":
            agent = "summarizer"
        elif intent in ["analyze", "report"]:
            agent = "detailed_report"
        elif needs_video:
            agent = "video_search"
        elif needs_text:
            agent = "text_search"
        else:
            agent = "video_search"  # Default to video search

        # Confidence calculation
        confidence = 0.7  # Base confidence
        if any(word in query_lower for word in ["find", "show", "search"]):
            confidence += 0.1
        if complexity == "simple":
            confidence += 0.1
        else:
            confidence -= 0.05

        confidence = min(0.95, max(0.3, confidence))

        # Reasoning
        reasoning = f"Intent: {intent}, Complexity: {complexity}, Video: {needs_video}, Agent: {agent}"

        return {
            "intent": intent,
            "complexity": complexity,
            "needs_video": needs_video,
            "needs_text": needs_text,
            "needs_multimodal": needs_multimodal,
            "agent": agent,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
        }


class DSPyAdvancedRoutingModule(dspy.Module):
    """
    DSPy 3.0 module for advanced routing with full relationship analysis.

    Uses ComposableQueryAnalysisModule for entity extraction, relationship
    analysis, and query enhancement, then makes routing decisions.
    """

    def __init__(
        self,
        analysis_module: Optional[ComposableQueryAnalysisModule] = None,
    ):
        super().__init__()
        self.router = dspy.ChainOfThought(AdvancedRoutingSignature)
        self.basic_module = DSPyBasicRoutingModule()

        # Use provided analysis module or create a default one
        if analysis_module is not None:
            self.analysis_module = analysis_module
        else:
            gliner_extractor = GLiNERRelationshipExtractor()
            spacy_analyzer = SpaCyDependencyAnalyzer()
            self.analysis_module = ComposableQueryAnalysisModule(
                gliner_extractor=gliner_extractor,
                spacy_analyzer=spacy_analyzer,
            )

    def forward(
        self,
        query: str,
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
    ) -> dspy.Prediction:
        """
        Perform advanced routing with relationship-aware analysis.

        Args:
            query: User query to route
            context: Conversation context
            user_preferences: User preferences
            system_state: Current system state

        Returns:
            DSPy prediction with comprehensive routing decision
        """
        try:
            # Step 1: Basic query analysis
            basic_analysis = self.basic_module.forward(query, context)

            # Step 2: Composable query analysis (entities + relationships + enhancement)
            analysis_result = self.analysis_module.forward(query)

            # Step 3: Create comprehensive routing decision
            routing_decision = self._create_routing_decision(
                basic_analysis,
                analysis_result,
                user_preferences,
                system_state,
            )

            # Step 4: Create agent workflow
            agent_workflow = self._create_agent_workflow(routing_decision)

            # Step 5: Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                query, analysis_result
            )

            # Create comprehensive prediction
            prediction = dspy.Prediction()

            # Query analysis
            prediction.query_analysis = {
                "primary_intent": basic_analysis.primary_intent,
                "complexity_level": basic_analysis.complexity_level,
                "domain_classification": analysis_result.domain_classification,
                "confidence": basic_analysis.confidence_score,
            }

            # Extracted information
            prediction.extracted_entities = analysis_result.entities
            prediction.extracted_relationships = analysis_result.relationships
            prediction.enhanced_query = analysis_result.enhanced_query

            # Routing decision
            prediction.routing_decision = routing_decision
            prediction.agent_workflow = agent_workflow
            prediction.optimization_suggestions = optimization_suggestions

            # Overall confidence and reasoning
            prediction.overall_confidence = self._calculate_overall_confidence(
                basic_analysis, analysis_result
            )
            prediction.reasoning_chain = self._generate_reasoning_chain(
                query, basic_analysis, analysis_result
            )

            return prediction

        except Exception as e:
            logger.error(f"Advanced routing failed: {e}")

            # Fallback to basic routing
            basic_prediction = self.basic_module.forward(query, context)

            # Create minimal advanced prediction
            prediction = dspy.Prediction()
            prediction.query_analysis = {"error": str(e)}
            prediction.extracted_entities = []
            prediction.extracted_relationships = []
            prediction.enhanced_query = query
            prediction.routing_decision = {
                "search_modality": "multimodal",
                "generation_type": "raw_results",
                "primary_agent": basic_prediction.recommended_agent,
                "secondary_agents": [],
                "execution_mode": "sequential",
                "confidence": 0.3,
                "reasoning": f"Fallback due to error: {e}",
            }
            prediction.agent_workflow = []
            prediction.optimization_suggestions = []
            prediction.overall_confidence = 0.3
            prediction.reasoning_chain = [f"Error in advanced routing: {e}"]

            return prediction

    def _create_routing_decision(
        self,
        basic_analysis: dspy.Prediction,
        analysis_result: dspy.Prediction,
        user_preferences: Optional[Dict[str, Any]],
        system_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create comprehensive routing decision."""
        # Determine search modality
        if basic_analysis.needs_multimodal:
            search_modality = "multimodal"
        elif basic_analysis.needs_video_search:
            search_modality = "video_only"
        elif basic_analysis.needs_text_search:
            search_modality = "text_only"
        else:
            search_modality = "both"

        # Determine generation type
        if basic_analysis.primary_intent == "summarize":
            generation_type = "summary"
        elif basic_analysis.primary_intent in ["analyze", "report"]:
            generation_type = "detailed_report"
        else:
            generation_type = "raw_results"

        # Primary and secondary agents
        primary_agent = basic_analysis.recommended_agent
        secondary_agents = []

        if basic_analysis.complexity_level == "complex":
            if primary_agent != "summarizer":
                secondary_agents.append("summarizer")
            if (
                generation_type == "detailed_report"
                and primary_agent != "detailed_report"
            ):
                secondary_agents.append("detailed_report")

        # Execution mode
        if len(secondary_agents) > 1:
            execution_mode = "parallel"
        elif len(secondary_agents) == 1:
            execution_mode = "sequential"
        else:
            execution_mode = "single"

        # Overall confidence
        analysis_confidence = analysis_result.confidence
        confidence = basic_analysis.confidence_score * 0.6 + analysis_confidence * 0.4

        # Reasoning
        reasoning = (
            f"Routing based on {basic_analysis.primary_intent} intent, "
            f"{basic_analysis.complexity_level} complexity, "
            f"{len(analysis_result.entities)} entities, "
            f"{len(analysis_result.relationships)} relationships, "
            f"path: {analysis_result.path_used}"
        )

        return {
            "search_modality": search_modality,
            "generation_type": generation_type,
            "primary_agent": primary_agent,
            "secondary_agents": secondary_agents,
            "execution_mode": execution_mode,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
        }

    def _create_agent_workflow(
        self, routing_decision: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create step-by-step agent workflow."""
        workflow = []

        workflow.append(
            {
                "step": 1,
                "agent": routing_decision["primary_agent"],
                "action": "primary_processing",
                "parameters": {
                    "search_modality": routing_decision["search_modality"],
                    "generation_type": routing_decision["generation_type"],
                },
            }
        )

        for i, secondary_agent in enumerate(routing_decision["secondary_agents"]):
            workflow.append(
                {
                    "step": i + 2,
                    "agent": secondary_agent,
                    "action": "secondary_processing",
                    "parameters": {
                        "input_source": "primary_results",
                        "processing_mode": routing_decision["execution_mode"],
                    },
                }
            )

        return workflow

    def _generate_optimization_suggestions(
        self,
        query: str,
        analysis_result: dspy.Prediction,
    ) -> List[str]:
        """Generate suggestions for optimization."""
        suggestions = []

        if len(analysis_result.entities) > 5:
            suggestions.append("Consider entity filtering for large entity sets")
        elif len(analysis_result.entities) == 0:
            suggestions.append("No entities detected - consider query expansion")

        if len(analysis_result.relationships) > 3:
            suggestions.append(
                "Rich relationship context available for enhanced search"
            )
        elif len(analysis_result.relationships) == 0:
            suggestions.append(
                "No relationships detected - consider semantic expansion"
            )

        domain = analysis_result.domain_classification
        if domain in ["artificial_intelligence", "technology"]:
            suggestions.append("Tech domain detected - prioritize technical accuracy")
        elif domain == "sports":
            suggestions.append("Sports domain detected - consider temporal context")

        return suggestions[:5]

    def _calculate_overall_confidence(
        self,
        basic_analysis: dspy.Prediction,
        analysis_result: dspy.Prediction,
    ) -> float:
        """Calculate overall confidence in the routing decision."""
        confidence = (
            basic_analysis.confidence_score * 0.6 + analysis_result.confidence * 0.4
        )
        return round(confidence, 3)

    def _generate_reasoning_chain(
        self,
        query: str,
        basic_analysis: dspy.Prediction,
        analysis_result: dspy.Prediction,
    ) -> List[str]:
        """Generate step-by-step reasoning process."""
        reasoning = []

        reasoning.append(f"Analyzed query: '{query[:50]}...'")
        reasoning.append(f"Detected intent: {basic_analysis.primary_intent}")
        reasoning.append(f"Complexity level: {basic_analysis.complexity_level}")
        reasoning.append(f"Found {len(analysis_result.entities)} entities")
        reasoning.append(
            f"Identified {len(analysis_result.relationships)} relationships"
        )
        reasoning.append(
            f"Domain classification: {analysis_result.domain_classification}"
        )
        reasoning.append(f"Analysis path used: {analysis_result.path_used}")
        reasoning.append(f"Recommended agent: {basic_analysis.recommended_agent}")
        reasoning.append(
            f"Overall confidence: {self._calculate_overall_confidence(basic_analysis, analysis_result)}"
        )

        return reasoning


# Factory functions for easy instantiation


def create_composable_query_analysis_module(
    entity_confidence_threshold: float = 0.6,
    min_entities_for_fast_path: int = 1,
) -> ComposableQueryAnalysisModule:
    """Create ComposableQueryAnalysisModule with default extractors."""
    gliner_extractor = GLiNERRelationshipExtractor()
    spacy_analyzer = SpaCyDependencyAnalyzer()
    return ComposableQueryAnalysisModule(
        gliner_extractor=gliner_extractor,
        spacy_analyzer=spacy_analyzer,
        entity_confidence_threshold=entity_confidence_threshold,
        min_entities_for_fast_path=min_entities_for_fast_path,
    )


def create_basic_routing_module() -> DSPyBasicRoutingModule:
    """Create DSPy basic routing module."""
    return DSPyBasicRoutingModule()


def create_advanced_routing_module(
    analysis_module: Optional[ComposableQueryAnalysisModule] = None,
) -> DSPyAdvancedRoutingModule:
    """Create DSPy advanced routing module."""
    return DSPyAdvancedRoutingModule(analysis_module=analysis_module)
