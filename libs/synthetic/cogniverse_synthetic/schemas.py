"""
Pydantic Schemas for Synthetic Data Generation

Defines schema classes for all optimizer types in the system.
Each schema corresponds to the training data format expected by an optimizer.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cogniverse_core.approval.training_schema import (
    PROFILE_QUERY_INTENT_VALUES,
    ProfileQueryIntent,
)
from cogniverse_foundation.common.tenant_utils import (
    require_tenant_id,
    validate_tenant_id,
)

SAMPLING_STRATEGIES = frozenset(
    {
        "diverse",
        "temporal_recent",
        "entity_rich",
        "multi_modal_sequences",
    }
)


class ProfileSelectionExampleSchema(BaseModel):
    """Training example for ProfileSelectionAgent optimization.

    Generation executes the production selector and copies its categorical
    decision fields exactly. The optimizer schema intentionally excludes the
    selector's runtime confidence because confidence is not a training target.
    """

    query: str = Field(..., description="User query text (DSPy input)")
    available_profiles: str = Field(
        ...,
        description="Comma-separated list of available backend profiles (DSPy input)",
    )
    selected_profile: str = Field(
        ..., description="Profile that should be selected for the query"
    )
    reasoning: str = Field(..., description="Reason for the selection")
    query_intent: ProfileQueryIntent = Field(
        ..., description=f"Detected intent: {', '.join(PROFILE_QUERY_INTENT_VALUES)}"
    )
    modality: str = Field(
        ...,
        description=(
            "Target modality: audio, code, document, image, text, video, or wiki"
        ),
    )
    complexity: Literal["simple", "medium", "complex"] = Field(
        ..., description="Query complexity: simple, medium, complex"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "query": "find a clip about transformer architecture",
                "available_profiles": (
                    "video_colpali_smol500_mv_frame,"
                    "video_colqwen_omni_mv_chunk_30s,"
                    "video_xclip_sv_chunk_6s"
                ),
                "selected_profile": "video_colqwen_omni_mv_chunk_30s",
                "reasoning": "Selected chunk-based profile for medium-complexity video search",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            }
        },
    )


class QueryEnhancementExampleSchema(BaseModel):
    """Training example for QueryEnhancementAgent optimization.

    Generation executes the production enhancement agent and copies its query,
    expansion, synonym, and reasoning outputs. Runtime confidence is not part
    of the query-enhancement optimizer's training contract.
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
    reasoning: str = Field(..., description="Why the query was enhanced this way")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "query": "transformer architecture",
                "enhanced_query": "transformer architecture attention mechanism self-attention",
                "expansion_terms": ["attention mechanism", "self-attention"],
                "synonyms": ["neural network model"],
                "context": "machine learning",
                "reasoning": "Added attention-related terms for a transformer query",
            }
        },
    )


class EntityExtractionExampleSchema(BaseModel):
    """Training example for EntityExtractionAgent optimization.

    Feeds ``run_entity_extraction_optimization`` in
    ``libs/runtime/cogniverse_runtime/optimization_cli.py``, which builds a
    ``dspy.Example`` from ``query`` (DSPy input) + ``entities`` + ``relationships``.
    The finetuning evaluator (``adapter_evaluator._check_entity_prediction``)
    scores each ``entities`` item on its ``text`` and ``type``.
    """

    query: str = Field(..., description="Text to extract entities from (DSPy input)")
    entities: List[Dict[str, str]] = Field(
        ..., description="Extracted entities, each with 'text' and 'type'"
    )
    relationships: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Optional relationships, each {source, target, type}",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "query": "PyTorch was created by Meta AI in Menlo Park",
                "entities": [
                    {"text": "PyTorch", "type": "PRODUCT"},
                    {"text": "Meta AI", "type": "ORG"},
                    {"text": "Menlo Park", "type": "PLACE"},
                ],
                "relationships": [
                    {"source": "Meta AI", "target": "PyTorch", "type": "created"}
                ],
            }
        },
    )


class RoutingExperienceSchema(BaseModel):
    """Training example containing an observed production gateway decision.

    Fresh generation preserves the gateway's exact routing confidence while
    downstream search quality, agent outcome, and processing time remain
    explicit unobserved sentinels. Query-changing regeneration resets gateway
    confidence to an unobserved sentinel until the gateway is executed again.
    """

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
        extra="forbid",
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
                "chosen_agent": "search_agent",
                "routing_confidence": 0.85,
                "search_quality": 0.78,
                "agent_success": True,
                "user_satisfaction": 0.9,
            }
        },
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
        extra="forbid",
        json_schema_extra={
            "example": {
                "workflow_id": "synthetic_workflow_001",
                "query": "summarize machine learning video and create report",
                "query_type": "VIDEO",
                "execution_time": 3.5,
                "success": True,
                "agent_sequence": [
                    "search_agent",
                    "summarizer_agent",
                    "detailed_report_agent",
                ],
                "task_count": 3,
                "parallel_efficiency": 0.85,
                "confidence_score": 0.88,
                "user_satisfaction": 0.9,
            }
        },
    )


class SyntheticDataRequest(BaseModel):
    """Strict request for one registered synthetic-data optimizer."""

    optimizer: str = Field(
        ...,
        description=(
            "Optimizer name: query_enhancement, entity_extraction, profile, "
            "routing, workflow, unified, or cross_modal"
        ),
    )
    count: int = Field(
        ...,
        strict=True,
        ge=1,
        le=10000,
        description="Number of examples to generate",
    )
    vespa_sample_size: int = Field(
        default=200,
        strict=True,
        ge=1,
        le=10000,
        description="Number of documents to sample from Vespa",
    )
    strategy: str | None = Field(
        default=None,
        description=(
            "Optional sampling strategy override; when omitted, the optimizer's "
            "registered backend query strategy is used"
        ),
    )
    max_profiles: int = Field(
        default=3,
        strict=True,
        ge=1,
        le=10,
        description="Maximum number of backend profiles to use",
    )
    tenant_id: str = Field(..., description="Tenant identifier (required)")

    @field_validator("tenant_id")
    @classmethod
    def validate_and_canonicalize_tenant_id(cls, tenant_id: str) -> str:
        canonical = require_tenant_id(tenant_id, source=cls.__name__)
        validate_tenant_id(canonical)
        return canonical

    @field_validator("strategy", mode="before")
    @classmethod
    def validate_strategy(cls, strategy: object) -> str:
        if strategy is None:
            raise ValueError(
                "strategy must be omitted to use the optimizer default, not null"
            )
        if not isinstance(strategy, str):
            raise ValueError("strategy must be a string")
        if strategy not in SAMPLING_STRATEGIES:
            allowed = ", ".join(sorted(SAMPLING_STRATEGIES))
            raise ValueError(
                f"Unsupported sampling strategy: {strategy}. Allowed: {allowed}"
            )
        return strategy

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "optimizer": "profile",
                "count": 100,
                "vespa_sample_size": 200,
                "strategy": "diverse",
                "max_profiles": 3,
                "tenant_id": "acme:production",
            }
        },
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
        extra="forbid",
        json_schema_extra={
            "example": {
                "optimizer": "profile",
                "schema_name": "ProfileSelectionExampleSchema",
                "count": 100,
                "selected_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_xclip_sv_chunk_6s",
                ],
                "profile_selection_reasoning": "Selected frame-based and chunk-based profiles for content diversity",
                "data": [],
                "metadata": {
                    "backend_type": "vespa",
                    "backend_query_strategy": "diverse",
                    "generation_time_ms": 1250,
                },
            }
        },
    )
