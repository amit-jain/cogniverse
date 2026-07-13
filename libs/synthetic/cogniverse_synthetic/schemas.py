"""
Pydantic Schemas for Synthetic Data Generation

Defines schema classes for all optimizer types in the system.
Each schema corresponds to the training data format expected by an optimizer.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProfileSelectionExampleSchema(BaseModel):
    """Training example for ProfileSelectionAgent optimization.

    Output fields mirror ``ProfileSelectionSignature``
    (``libs/agents/cogniverse_agents/profile_selection_agent.py``) and feed
    ``run_profile_optimization`` in
    ``libs/runtime/cogniverse_runtime/optimization_cli.py``, which builds
    a ``dspy.Example`` from these fields and trains the
    ``ProfileSelectionModule`` via teleprompter.
    """

    query: str = Field(..., description="User query text (DSPy input)")
    available_profiles: str = Field(
        ...,
        description="Comma-separated list of available backend profiles (DSPy input)",
    )
    selected_profile: str = Field(
        ..., description="Profile that should be selected for the query"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the selection (0-1)"
    )
    reasoning: str = Field(..., description="Reason for the selection")
    query_intent: str = Field(
        ..., description="text_search, video_search, image_search, etc."
    )
    modality: str = Field(
        ..., description="Target modality: video, image, text, audio, document"
    )
    complexity: str = Field(
        ..., description="Query complexity: simple, medium, complex"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "find a clip about transformer architecture",
                "available_profiles": (
                    "video_colpali_smol500_mv_frame,"
                    "video_colqwen_omni_mv_chunk_30s,"
                    "video_videoprism_base_mv_chunk_30s,"
                    "video_videoprism_large_mv_chunk_30s"
                ),
                "selected_profile": "video_colqwen_omni_mv_chunk_30s",
                "confidence": 0.85,
                "reasoning": "Selected chunk-based profile for medium-complexity video search",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            }
        }
    )


class QueryEnhancementExampleSchema(BaseModel):
    """Training example for QueryEnhancementAgent optimization.

    Feeds ``run_simba_optimization`` in
    ``libs/runtime/cogniverse_runtime/optimization_cli.py``, which builds a
    ``dspy.Example`` from these fields (query is the DSPy input, the rest are
    the enhancement the SIMBA/BootstrapFewShot trainer learns to produce).
    """

    query: str = Field(..., description="Original user query (DSPy input)")
    enhanced_query: str = Field(
        ..., description="Query rewritten with expansion terms (must differ from query)"
    )
    expansion_terms: List[str] = Field(
        default_factory=list, description="Terms added to broaden the query"
    )
    synonyms: List[str] = Field(
        default_factory=list, description="Synonyms for salient query terms"
    )
    context: str = Field("", description="Domain/context the query sits in")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the enhancement (0-1)"
    )
    reasoning: str = Field(..., description="Why the query was enhanced this way")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "transformer architecture",
                "enhanced_query": "transformer architecture attention mechanism self-attention",
                "expansion_terms": ["attention mechanism", "self-attention"],
                "synonyms": ["neural network model"],
                "context": "machine learning",
                "confidence": 0.85,
                "reasoning": "Added attention-related terms for a transformer query",
            }
        }
    )


class EntityExtractionExampleSchema(BaseModel):
    """Training example for EntityExtractionAgent optimization.

    Feeds ``run_entity_extraction_optimization`` in
    ``libs/runtime/cogniverse_runtime/optimization_cli.py``, which builds a
    ``dspy.Example`` from ``query`` (DSPy input) + ``entities`` + ``entity_types``.
    The finetuning evaluator (``adapter_evaluator._check_entity_prediction``)
    scores each ``entities`` item on its ``text`` and ``type``.
    """

    query: str = Field(..., description="Text to extract entities from (DSPy input)")
    entities: List[Dict[str, str]] = Field(
        ..., description="Extracted entities, each with 'text' and 'type'"
    )
    entity_types: str = Field("", description="Comma-separated distinct entity types")
    relationships: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Optional relationships, each {source, target, type}",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "PyTorch was created by Meta AI in Menlo Park",
                "entities": [
                    {"text": "PyTorch", "type": "PRODUCT"},
                    {"text": "Meta AI", "type": "ORG"},
                    {"text": "Menlo Park", "type": "PLACE"},
                ],
                "entity_types": "PRODUCT,ORG,PLACE",
                "relationships": [
                    {"source": "Meta AI", "target": "PyTorch", "type": "created"}
                ],
            }
        }
    )


class RoutingExperienceSchema(BaseModel):
    """Training example representing a routing decision with entity
    extraction and quality metrics."""

    query: str = Field(..., description="User query text")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities from query"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Relationships between entities"
    )
    enhanced_query: str = Field(
        ..., description="Query enhanced with entity information"
    )
    chosen_agent: str = Field(..., description="Agent selected for routing")
    routing_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in routing decision (0-1)"
    )
    search_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Quality of search results (0-1)"
    )
    agent_success: bool = Field(..., description="Whether agent completed successfully")
    user_satisfaction: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Explicit user feedback (0-1)"
    )
    processing_time: float = Field(
        default=0.0, description="Processing time in seconds"
    )
    reward: Optional[float] = Field(default=None, description="Computed reward signal")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this routing occurred",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "find TensorFlow tutorials on neural networks",
                "entities": [
                    {"text": "TensorFlow", "type": "TECHNOLOGY"},
                    {"text": "neural networks", "type": "TOPIC"},
                ],
                "relationships": [
                    {
                        "source": "TensorFlow",
                        "target": "neural networks",
                        "type": "USED_FOR",
                    }
                ],
                "enhanced_query": "find TensorFlow(TECHNOLOGY) tutorials on neural networks(TOPIC)",
                "chosen_agent": "video_search_agent",
                "routing_confidence": 0.85,
                "search_quality": 0.78,
                "agent_success": True,
                "user_satisfaction": 0.9,
            }
        }
    )


class WorkflowExecutionSchema(BaseModel):
    """Training example representing a complete workflow execution with
    performance metrics. Consumed by WorkflowIntelligence."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    query: str = Field(..., description="User query text")
    query_type: str = Field(..., description="Query modality/type")
    execution_time: float = Field(
        ..., ge=0.0, description="Total execution time in seconds"
    )
    success: bool = Field(..., description="Whether workflow completed successfully")
    agent_sequence: List[str] = Field(..., description="Sequence of agents executed")
    task_count: int = Field(..., ge=1, description="Number of tasks in workflow")
    parallel_efficiency: float = Field(
        ..., ge=0.0, le=1.0, description="Parallel execution efficiency (0-1)"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in workflow execution (0-1)",
    )
    user_satisfaction: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="User satisfaction score (0-1)"
    )
    error_details: Optional[str] = Field(
        default=None, description="Error details if failed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When workflow executed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_id": "synthetic_workflow_001",
                "query": "summarize machine learning video and create report",
                "query_type": "VIDEO",
                "execution_time": 3.5,
                "success": True,
                "agent_sequence": [
                    "video_search_agent",
                    "summarizer",
                    "detailed_report",
                ],
                "task_count": 3,
                "parallel_efficiency": 0.85,
                "confidence_score": 0.88,
                "user_satisfaction": 0.9,
            }
        }
    )


class SyntheticDataRequest(BaseModel):
    """Request schema for synthetic data generation endpoint"""

    optimizer: str = Field(
        ...,
        description="Optimizer name (profile, routing, workflow, unified)",
    )
    count: int = Field(
        ..., ge=1, le=10000, description="Number of examples to generate"
    )
    vespa_sample_size: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="Number of documents to sample from Vespa",
    )
    strategies: List[str] = Field(
        default=["diverse"],
        description="Sampling strategies (diverse, temporal_recent, entity_rich, etc.)",
    )
    max_profiles: int = Field(
        default=3, ge=1, le=10, description="Maximum number of backend profiles to use"
    )
    tenant_id: str = Field(..., description="Tenant identifier (required)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "optimizer": "profile",
                "count": 100,
                "vespa_sample_size": 200,
                "strategies": ["diverse"],
                "max_profiles": 3,
                "tenant_id": "acme:production",
            }
        }
    )


class SyntheticDataResponse(BaseModel):
    """Response schema for synthetic data generation endpoint"""

    optimizer: str = Field(..., description="Optimizer name")
    schema_name: str = Field(..., description="Schema class name used")
    count: int = Field(..., description="Number of examples generated")
    selected_profiles: List[str] = Field(..., description="Backend profiles used")
    profile_selection_reasoning: str = Field(
        ..., description="Reasoning for profile selection"
    )
    data: List[Dict[str, Any]] = Field(..., description="Generated synthetic data")
    metadata: Dict[str, Any] = Field(..., description="Generation metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "optimizer": "profile",
                "schema_name": "ProfileSelectionExampleSchema",
                "count": 100,
                "selected_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_videoprism_base_mv_chunk_30s",
                ],
                "profile_selection_reasoning": "Selected frame-based and chunk-based profiles for content diversity",
                "data": [],
                "metadata": {
                    "backend_type": "vespa",
                    "query_strategy": "diverse",
                    "generation_time_ms": 1250,
                },
            }
        }
    )
