"""
DSPy 3.0 Signatures for Intelligent Routing

This module defines the core DSPy 3.0 signatures used throughout the routing system.
These signatures leverage DSPy's type system and optimization capabilities to create
intelligent, self-improving routing decisions.
"""

from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel


# Data Models for Structured Outputs
class EntityInfo(BaseModel):
    """Extracted entity information"""

    text: str
    label: str
    confidence: float
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


class RelationshipTuple(BaseModel):
    """Relationship between entities"""

    subject: str
    relation: str
    object: str
    confidence: float
    subject_type: Optional[str] = None
    object_type: Optional[str] = None


class TemporalInfo(BaseModel):
    """Temporal information extracted from query"""

    time_references: List[str]
    date_patterns: List[str]
    temporal_context: str
    has_temporal_constraints: bool


class RoutingDecision(BaseModel):
    """Complete routing decision"""

    search_modality: str  # video_only, text_only, both, multimodal
    generation_type: str  # raw_results, summary, detailed_report
    primary_agent: str
    secondary_agents: List[str]
    execution_mode: str  # sequential, parallel, hybrid
    confidence: float
    reasoning: str


# Core DSPy 3.0 Signatures


class BasicQueryAnalysisSignature(dspy.Signature):
    """Basic query analysis for routing decisions"""

    query: str = dspy.InputField(desc="User query to analyze")
    context: Optional[str] = dspy.InputField(desc="Optional conversation context")

    # Intent and complexity analysis
    primary_intent: str = dspy.OutputField(
        desc="Primary intent: search, compare, analyze, summarize, report, explain"
    )
    complexity_level: str = dspy.OutputField(
        desc="Query complexity: simple, moderate, complex"
    )

    # Search requirements
    needs_video_search: bool = dspy.OutputField(
        desc="Whether video content search is needed"
    )
    needs_text_search: bool = dspy.OutputField(
        desc="Whether text/document search is needed"
    )
    needs_multimodal: bool = dspy.OutputField(
        desc="Whether multimodal processing is required"
    )

    # Routing decision
    recommended_agent: str = dspy.OutputField(
        desc="Primary agent to handle query: video_search, text_search, summarizer, report_generator"
    )
    confidence_score: float = dspy.OutputField(
        desc="Confidence in routing decision (0.0-1.0)"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of routing decision")


class EntityExtractionSignature(dspy.Signature):
    """Enhanced entity extraction with relationship awareness"""

    query: str = dspy.InputField(desc="Query text for entity extraction")
    domain_context: Optional[str] = dspy.InputField(
        desc="Domain context for better extraction"
    )

    # Entity outputs
    entities: List[Dict[str, Any]] = dspy.OutputField(
        desc="Extracted entities with types, positions, and confidence scores"
    )
    entity_types: List[str] = dspy.OutputField(desc="Unique entity types found")
    key_entities: List[str] = dspy.OutputField(
        desc="Most important entities for the query"
    )

    # Context analysis
    domain_classification: str = dspy.OutputField(
        desc="Identified domain/topic of the query"
    )
    entity_density: float = dspy.OutputField(desc="Ratio of entities to total words")
    confidence: float = dspy.OutputField(desc="Overall extraction confidence")


class RelationshipExtractionSignature(dspy.Signature):
    """Extract relationships between entities for query enhancement"""

    query: str = dspy.InputField(desc="Query text for relationship extraction")
    entities: List[Dict[str, Any]] = dspy.InputField(
        desc="Previously extracted entities"
    )
    linguistic_context: Optional[str] = dspy.InputField(
        desc="Linguistic analysis context"
    )

    # Relationship outputs
    relationships: List[Dict[str, Any]] = dspy.OutputField(
        desc="Relationship tuples: subject, relation, object with confidence scores"
    )
    relationship_types: List[str] = dspy.OutputField(
        desc="Types of relationships found"
    )
    semantic_connections: List[str] = dspy.OutputField(
        desc="Semantic connections between entities"
    )

    # Analysis outputs
    query_structure: str = dspy.OutputField(
        desc="Grammatical/semantic structure analysis"
    )
    complexity_indicators: List[str] = dspy.OutputField(
        desc="Factors indicating query complexity"
    )
    confidence: float = dspy.OutputField(desc="Relationship extraction confidence")


class QueryEnhancementSignature(dspy.Signature):
    """Enhance queries using extracted relationships and entities"""

    original_query: str = dspy.InputField(desc="Original user query")
    entities: List[Dict[str, Any]] = dspy.InputField(desc="Extracted entities")
    relationships: List[Dict[str, Any]] = dspy.InputField(
        desc="Extracted relationship tuples"
    )
    search_context: str = dspy.InputField(desc="Target search system context")

    # Enhancement outputs
    enhanced_query: str = dspy.OutputField(
        desc="Query enhanced with relationship context"
    )
    semantic_expansions: List[str] = dspy.OutputField(
        desc="Additional terms for better retrieval"
    )
    relationship_phrases: List[str] = dspy.OutputField(
        desc="Natural language relationship descriptions"
    )

    # Strategy outputs
    enhancement_strategy: str = dspy.OutputField(desc="Strategy used for enhancement")
    search_operators: List[str] = dspy.OutputField(
        desc="Suggested search operators (AND, OR)"
    )
    quality_score: float = dspy.OutputField(desc="Enhancement quality estimate")


class MultiAgentOrchestrationSignature(dspy.Signature):
    """Orchestrate multiple agents for complex queries"""

    query: str = dspy.InputField(desc="Original user query")
    query_analysis: Dict[str, Any] = dspy.InputField(desc="Results from query analysis")
    available_agents: List[str] = dspy.InputField(desc="List of available agent types")
    agent_capabilities: Dict[str, List[str]] = dspy.InputField(
        desc="Capabilities of each agent"
    )

    # Orchestration plan
    execution_plan: Dict[str, Any] = dspy.OutputField(
        desc="Complete execution plan with agent sequence and parameters"
    )
    agent_assignments: List[Dict[str, Any]] = dspy.OutputField(
        desc="Specific tasks assigned to each agent"
    )
    coordination_strategy: str = dspy.OutputField(
        desc="How agents should be coordinated: sequential, parallel, hybrid"
    )

    # Success criteria
    success_metrics: Dict[str, Any] = dspy.OutputField(
        desc="How to measure task success"
    )
    fallback_strategy: str = dspy.OutputField(desc="What to do if primary plan fails")
    expected_processing_time: float = dspy.OutputField(
        desc="Estimated total processing time"
    )


class AdvancedRoutingSignature(dspy.Signature):
    """Advanced routing with relationship-aware query enhancement"""

    query: str = dspy.InputField(desc="User query to route")
    context: Optional[str] = dspy.InputField(desc="Conversation context")
    user_preferences: Optional[Dict[str, Any]] = dspy.InputField(
        desc="User preferences"
    )
    system_state: Optional[Dict[str, Any]] = dspy.InputField(
        desc="Current system state"
    )

    # Comprehensive analysis
    query_analysis: Dict[str, Any] = dspy.OutputField(
        desc="Complete query analysis results"
    )
    extracted_entities: List[Dict[str, Any]] = dspy.OutputField(
        desc="All extracted entities"
    )
    extracted_relationships: List[Dict[str, Any]] = dspy.OutputField(
        desc="All relationship tuples"
    )
    enhanced_query: str = dspy.OutputField(desc="Relationship-enhanced query")

    # Routing decision
    routing_decision: Dict[str, Any] = dspy.OutputField(
        desc="Complete routing decision"
    )
    agent_workflow: List[Dict[str, Any]] = dspy.OutputField(
        desc="Step-by-step agent workflow"
    )
    optimization_suggestions: List[str] = dspy.OutputField(
        desc="Suggestions for better performance"
    )

    # Confidence and quality
    overall_confidence: float = dspy.OutputField(
        desc="Overall confidence in routing decision"
    )
    reasoning_chain: List[str] = dspy.OutputField(desc="Step-by-step reasoning process")


class MetaRoutingSignature(dspy.Signature):
    """Meta-level routing to choose optimal routing strategy"""

    query: str = dspy.InputField(desc="User query")
    query_characteristics: Dict[str, Any] = dspy.InputField(
        desc="Query complexity and features"
    )
    routing_history: List[Dict[str, Any]] = dspy.InputField(
        desc="Recent routing decisions"
    )
    performance_metrics: Dict[str, Any] = dspy.InputField(
        desc="System performance data"
    )

    # Strategy selection
    recommended_strategy: str = dspy.OutputField(
        desc="Optimal routing strategy: fast_path, slow_path, hybrid"
    )
    strategy_confidence: float = dspy.OutputField(desc="Confidence in strategy choice")

    # Optimization
    threshold_adjustments: Dict[str, float] = dspy.OutputField(
        desc="Suggested confidence threshold adjustments"
    )
    learning_opportunities: List[str] = dspy.OutputField(
        desc="Opportunities for system learning and improvement"
    )
    reasoning: str = dspy.OutputField(desc="Reasoning for strategy selection")


class AdaptiveThresholdSignature(dspy.Signature):
    """Learn and adapt routing confidence thresholds"""

    recent_performance: List[Dict[str, Any]] = dspy.InputField(
        desc="Recent routing performance data"
    )
    query_patterns: List[Dict[str, Any]] = dspy.InputField(
        desc="Patterns in recent queries"
    )
    user_feedback: Optional[List[Dict[str, Any]]] = dspy.InputField(
        desc="User satisfaction scores"
    )
    system_metrics: Dict[str, Any] = dspy.InputField(
        desc="System-wide performance metrics"
    )

    # Adaptive thresholds
    fast_path_threshold: float = dspy.OutputField(
        desc="Optimal fast path confidence threshold"
    )
    slow_path_threshold: float = dspy.OutputField(
        desc="Optimal slow path confidence threshold"
    )
    escalation_threshold: float = dspy.OutputField(
        desc="When to escalate between tiers"
    )

    # Learning insights
    performance_insights: List[str] = dspy.OutputField(desc="Key performance insights")
    optimization_opportunities: List[str] = dspy.OutputField(
        desc="Areas for improvement"
    )
    confidence_explanation: str = dspy.OutputField(
        desc="Reasoning for threshold values"
    )


# Utility functions for signature usage


def create_routing_signature(complexity_level: str = "basic") -> type:
    """
    Factory function to create appropriate routing signature based on complexity.

    Args:
        complexity_level: "basic", "advanced", or "meta"

    Returns:
        Appropriate DSPy signature class
    """
    signature_map = {
        "basic": BasicQueryAnalysisSignature,
        "advanced": AdvancedRoutingSignature,
        "meta": MetaRoutingSignature,
    }

    return signature_map.get(complexity_level, BasicQueryAnalysisSignature)


def validate_signature_output(
    signature_output: dspy.Prediction, required_fields: List[str]
) -> bool:
    """
    Validate that signature output contains required fields.

    Args:
        signature_output: Output from DSPy signature
        required_fields: List of required field names

    Returns:
        True if all required fields present
    """
    return all(hasattr(signature_output, field) for field in required_fields)


# Example signature usage for testing
if __name__ == "__main__":
    print("DSPy 3.0 Routing Signatures")
    print("=" * 40)

    # Test signature creation
    signatures = [
        ("BasicQueryAnalysis", BasicQueryAnalysisSignature),
        ("EntityExtraction", EntityExtractionSignature),
        ("RelationshipExtraction", RelationshipExtractionSignature),
        ("QueryEnhancement", QueryEnhancementSignature),
        ("MultiAgentOrchestration", MultiAgentOrchestrationSignature),
        ("AdvancedRouting", AdvancedRoutingSignature),
        ("MetaRouting", MetaRoutingSignature),
        ("AdaptiveThreshold", AdaptiveThresholdSignature),
    ]

    for name, signature_class in signatures:
        try:
            # Test signature creation
            signature_instance = signature_class()

            # Check input and output fields
            input_fields = signature_class.__annotations__.get("inputs", {})
            output_fields = signature_class.__annotations__.get("outputs", {})

            print(f"✅ {name}: Created successfully")
            print(f"   Signature: {signature_class.__name__}")

        except Exception as e:
            print(f"❌ {name}: Failed - {e}")

    print("\n🎉 All DSPy 3.0 routing signatures are ready!")
