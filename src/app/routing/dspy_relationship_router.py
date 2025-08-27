"""
DSPy 3.0 Relationship Router Module

This module integrates DSPy 3.0 signatures with relationship extraction tools
to create an intelligent routing system that leverages entity and relationship
information for enhanced query analysis and routing decisions.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import dspy

from .dspy_routing_signatures import (
    AdvancedRoutingSignature,
    BasicQueryAnalysisSignature,
    EntityExtractionSignature,
    RelationshipExtractionSignature,
)
from .relationship_extraction_tools import RelationshipExtractorTool

logger = logging.getLogger(__name__)


class DSPyEntityExtractorModule(dspy.Module):
    """
    DSPy 3.0 module for entity extraction using EntityExtractionSignature.

    Integrates with RelationshipExtractorTool to provide structured entity
    extraction with confidence scores and domain classification.
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(EntityExtractionSignature)
        self.relationship_tool = RelationshipExtractorTool()

    def forward(
        self, query: str, domain_context: Optional[str] = None
    ) -> dspy.Prediction:
        """
        Extract entities from query using DSPy 3.0 + GLiNER integration.

        Args:
            query: Input query text
            domain_context: Optional domain context for better extraction

        Returns:
            DSPy prediction with entity extraction results
        """
        try:
            # Use relationship tool for actual extraction
            # Handle async call properly - check if event loop is running
            try:
                # Try to get the current event loop
                asyncio.get_running_loop()
                # If we're already in an event loop, provide fallback to avoid async issue
                extraction_result = {"entities": [], "confidence": 0.5}
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                extraction_result = asyncio.run(
                    self.relationship_tool.extract_comprehensive_relationships(query)
                )

            entities = extraction_result["entities"]

            # Prepare DSPy inputs
            entities_list = [
                {
                    "text": e["text"],
                    "label": e["label"],
                    "confidence": e["confidence"],
                    "start_pos": e.get("start_pos"),
                    "end_pos": e.get("end_pos"),
                }
                for e in entities
            ]

            entity_types = list(set(e["label"] for e in entities))
            key_entities = [
                e["text"]
                for e in sorted(entities, key=lambda x: x["confidence"], reverse=True)[
                    :5
                ]
            ]

            # Domain classification based on entity types
            domain_classification = self._classify_domain(entity_types, query)

            # Calculate entity density
            entity_density = (
                len(entities) / len(query.split()) if query.split() else 0.0
            )

            # Create mock prediction with real data
            prediction = dspy.Prediction()
            prediction.entities = entities_list
            prediction.entity_types = entity_types
            prediction.key_entities = key_entities
            prediction.domain_classification = domain_classification
            prediction.entity_density = round(entity_density, 3)
            prediction.confidence = extraction_result["confidence"]

            return prediction

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")

            # Return empty prediction on error
            prediction = dspy.Prediction()
            prediction.entities = []
            prediction.entity_types = []
            prediction.key_entities = []
            prediction.domain_classification = "unknown"
            prediction.entity_density = 0.0
            prediction.confidence = 0.0

            return prediction

    def _classify_domain(self, entity_types: List[str], query: str) -> str:
        """
        Classify domain based on entity types and query content.

        Args:
            entity_types: List of entity types found
            query: Original query

        Returns:
            Domain classification
        """
        query_lower = query.lower()

        # Technology domain
        if any(t in entity_types for t in ["TECHNOLOGY", "PRODUCT", "TOOL"]):
            if any(
                word in query_lower
                for word in ["ai", "machine learning", "algorithm", "robot"]
            ):
                return "artificial_intelligence"
            else:
                return "technology"

        # Sports domain
        if any(t in entity_types for t in ["SPORT", "ACTIVITY"]) or any(
            word in query_lower for word in ["playing", "game", "sport", "competition"]
        ):
            return "sports"

        # Entertainment domain
        if any(
            word in query_lower for word in ["video", "show", "movie", "entertainment"]
        ):
            return "entertainment"

        # Education domain
        if any(
            word in query_lower
            for word in ["learn", "teach", "education", "tutorial", "how to"]
        ):
            return "education"

        # Location/Travel domain
        if any(t in entity_types for t in ["LOCATION"]):
            return "location_travel"

        # Business domain
        if any(t in entity_types for t in ["ORGANIZATION"]):
            return "business"

        # Default
        return "general"


class DSPyRelationshipExtractorModule(dspy.Module):
    """
    DSPy 3.0 module for relationship extraction using RelationshipExtractionSignature.

    Takes entities as input and extracts relationships between them using
    both GLiNER inference and spaCy dependency parsing.
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(RelationshipExtractionSignature)
        self.relationship_tool = RelationshipExtractorTool()

    def forward(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        linguistic_context: Optional[str] = None,
    ) -> dspy.Prediction:
        """
        Extract relationships between entities.

        Args:
            query: Original query text
            entities: Previously extracted entities
            linguistic_context: Optional linguistic analysis context

        Returns:
            DSPy prediction with relationship extraction results
        """
        try:
            # Use relationship tool for comprehensive analysis
            # Handle async call properly - check if event loop is running
            try:
                # Try to get the current event loop
                asyncio.get_running_loop()
                # If we're already in an event loop, provide fallback
                extraction_result = {
                    "relationships": [],
                    "relationship_types": [],
                    "confidence": 0.5,
                }
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                extraction_result = asyncio.run(
                    self.relationship_tool.extract_comprehensive_relationships(query)
                )

            relationships = extraction_result["relationships"]

            # Prepare relationship data
            relationships_list = [
                {
                    "subject": r["subject"],
                    "relation": r["relation"],
                    "object": r["object"],
                    "confidence": r["confidence"],
                    "subject_type": r.get("subject_type"),
                    "object_type": r.get("object_type"),
                    "context": r.get("context", ""),
                }
                for r in relationships
            ]

            relationship_types = extraction_result["relationship_types"]
            semantic_connections = extraction_result["semantic_connections"]
            query_structure = extraction_result["query_structure"]
            complexity_indicators = extraction_result["complexity_indicators"]

            # Create prediction
            prediction = dspy.Prediction()
            prediction.relationships = relationships_list
            prediction.relationship_types = relationship_types
            prediction.semantic_connections = semantic_connections
            prediction.query_structure = query_structure
            prediction.complexity_indicators = complexity_indicators
            prediction.confidence = extraction_result["confidence"]

            return prediction

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")

            # Return empty prediction on error
            prediction = dspy.Prediction()
            prediction.relationships = []
            prediction.relationship_types = []
            prediction.semantic_connections = []
            prediction.query_structure = "error"
            prediction.complexity_indicators = []
            prediction.confidence = 0.0

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

    Combines entity extraction, relationship analysis, and query enhancement
    to make sophisticated routing decisions.
    """

    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(AdvancedRoutingSignature)
        self.entity_module = DSPyEntityExtractorModule()
        self.relationship_module = DSPyRelationshipExtractorModule()
        self.basic_module = DSPyBasicRoutingModule()

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

            # Step 2: Entity extraction
            entity_analysis = self.entity_module.forward(query)

            # Step 3: Relationship extraction
            relationship_analysis = self.relationship_module.forward(
                query, entity_analysis.entities
            )

            # Step 4: Query enhancement (basic version)
            enhanced_query = self._enhance_query_with_relationships(
                query, entity_analysis.entities, relationship_analysis.relationships
            )

            # Step 5: Create comprehensive routing decision
            routing_decision = self._create_routing_decision(
                basic_analysis,
                entity_analysis,
                relationship_analysis,
                user_preferences,
                system_state,
            )

            # Step 6: Create agent workflow
            agent_workflow = self._create_agent_workflow(routing_decision)

            # Step 7: Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                query, entity_analysis, relationship_analysis
            )

            # Create comprehensive prediction
            prediction = dspy.Prediction()

            # Query analysis
            prediction.query_analysis = {
                "primary_intent": basic_analysis.primary_intent,
                "complexity_level": basic_analysis.complexity_level,
                "domain_classification": entity_analysis.domain_classification,
                "confidence": basic_analysis.confidence_score,
            }

            # Extracted information
            prediction.extracted_entities = entity_analysis.entities
            prediction.extracted_relationships = relationship_analysis.relationships
            prediction.enhanced_query = enhanced_query

            # Routing decision
            prediction.routing_decision = routing_decision
            prediction.agent_workflow = agent_workflow
            prediction.optimization_suggestions = optimization_suggestions

            # Overall confidence and reasoning
            prediction.overall_confidence = self._calculate_overall_confidence(
                basic_analysis, entity_analysis, relationship_analysis
            )
            prediction.reasoning_chain = self._generate_reasoning_chain(
                query, basic_analysis, entity_analysis, relationship_analysis
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

    def _enhance_query_with_relationships(
        self,
        original_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> str:
        """
        Enhance query with relationship context (basic version).

        Args:
            original_query: Original user query
            entities: Extracted entities
            relationships: Extracted relationships

        Returns:
            Enhanced query string
        """
        if not relationships:
            return original_query

        # Add top relationships as additional context
        top_relations = relationships[:2]  # Top 2 relationships

        enhancements = []
        for rel in top_relations:
            if rel["confidence"] > 0.6:
                enhancement = f"{rel['subject']} {rel['relation'].replace('_', ' ')} {rel['object']}"
                enhancements.append(enhancement)

        if enhancements:
            enhanced = f"{original_query} ({' OR '.join(enhancements)})"
            return enhanced

        return original_query

    def _create_routing_decision(
        self,
        basic_analysis: dspy.Prediction,
        entity_analysis: dspy.Prediction,
        relationship_analysis: dspy.Prediction,
        user_preferences: Optional[Dict[str, Any]],
        system_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create comprehensive routing decision.

        Args:
            basic_analysis: Basic query analysis results
            entity_analysis: Entity extraction results
            relationship_analysis: Relationship extraction results
            user_preferences: User preferences
            system_state: System state

        Returns:
            Routing decision dictionary
        """
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
        confidence = (
            basic_analysis.confidence_score * 0.6
            + entity_analysis.confidence * 0.2
            + relationship_analysis.confidence * 0.2
        )

        # Reasoning
        reasoning = (
            f"Routing based on {basic_analysis.primary_intent} intent, "
            f"{basic_analysis.complexity_level} complexity, "
            f"{len(entity_analysis.entities)} entities, "
            f"{len(relationship_analysis.relationships)} relationships"
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
        """
        Create step-by-step agent workflow.

        Args:
            routing_decision: Routing decision dictionary

        Returns:
            List of workflow steps
        """
        workflow = []

        # Primary agent step
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

        # Secondary agent steps
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
        entity_analysis: dspy.Prediction,
        relationship_analysis: dspy.Prediction,
    ) -> List[str]:
        """
        Generate suggestions for optimization.

        Args:
            query: Original query
            entity_analysis: Entity analysis results
            relationship_analysis: Relationship analysis results

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Entity-based suggestions
        if len(entity_analysis.entities) > 5:
            suggestions.append("Consider entity filtering for large entity sets")
        elif len(entity_analysis.entities) == 0:
            suggestions.append("No entities detected - consider query expansion")

        # Relationship-based suggestions
        if len(relationship_analysis.relationships) > 3:
            suggestions.append(
                "Rich relationship context available for enhanced search"
            )
        elif len(relationship_analysis.relationships) == 0:
            suggestions.append(
                "No relationships detected - consider semantic expansion"
            )

        # Complexity-based suggestions
        if len(relationship_analysis.complexity_indicators) > 2:
            suggestions.append("Complex query - consider multi-step processing")

        # Domain-specific suggestions
        domain = entity_analysis.domain_classification
        if domain in ["artificial_intelligence", "technology"]:
            suggestions.append("Tech domain detected - prioritize technical accuracy")
        elif domain == "sports":
            suggestions.append("Sports domain detected - consider temporal context")

        return suggestions[:5]  # Limit to top 5 suggestions

    def _calculate_overall_confidence(
        self,
        basic_analysis: dspy.Prediction,
        entity_analysis: dspy.Prediction,
        relationship_analysis: dspy.Prediction,
    ) -> float:
        """
        Calculate overall confidence in the routing decision.

        Args:
            basic_analysis: Basic analysis results
            entity_analysis: Entity analysis results
            relationship_analysis: Relationship analysis results

        Returns:
            Overall confidence score
        """
        # Weighted average of component confidences
        weights = {"basic": 0.5, "entity": 0.3, "relationship": 0.2}

        confidence = (
            basic_analysis.confidence_score * weights["basic"]
            + entity_analysis.confidence * weights["entity"]
            + relationship_analysis.confidence * weights["relationship"]
        )

        return round(confidence, 3)

    def _generate_reasoning_chain(
        self,
        query: str,
        basic_analysis: dspy.Prediction,
        entity_analysis: dspy.Prediction,
        relationship_analysis: dspy.Prediction,
    ) -> List[str]:
        """
        Generate step-by-step reasoning process.

        Args:
            query: Original query
            basic_analysis: Basic analysis results
            entity_analysis: Entity analysis results
            relationship_analysis: Relationship analysis results

        Returns:
            List of reasoning steps
        """
        reasoning = []

        reasoning.append(f"Analyzed query: '{query[:50]}...'")
        reasoning.append(f"Detected intent: {basic_analysis.primary_intent}")
        reasoning.append(f"Complexity level: {basic_analysis.complexity_level}")
        reasoning.append(f"Found {len(entity_analysis.entities)} entities")
        reasoning.append(
            f"Identified {len(relationship_analysis.relationships)} relationships"
        )
        reasoning.append(
            f"Domain classification: {entity_analysis.domain_classification}"
        )
        reasoning.append(f"Recommended agent: {basic_analysis.recommended_agent}")
        reasoning.append(
            f"Overall confidence: {self._calculate_overall_confidence(basic_analysis, entity_analysis, relationship_analysis)}"
        )

        return reasoning


# Factory functions for easy instantiation


def create_entity_extractor_module() -> DSPyEntityExtractorModule:
    """Create DSPy entity extractor module."""
    return DSPyEntityExtractorModule()


def create_relationship_extractor_module() -> DSPyRelationshipExtractorModule:
    """Create DSPy relationship extractor module."""
    return DSPyRelationshipExtractorModule()


def create_basic_routing_module() -> DSPyBasicRoutingModule:
    """Create DSPy basic routing module."""
    return DSPyBasicRoutingModule()


def create_advanced_routing_module() -> DSPyAdvancedRoutingModule:
    """Create DSPy advanced routing module."""
    return DSPyAdvancedRoutingModule()
