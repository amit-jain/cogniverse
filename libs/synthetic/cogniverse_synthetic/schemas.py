"""
Pydantic Schemas for Synthetic Data Generation

Defines schema classes for all optimizer types in the system.
Each schema corresponds to the training data format expected by an optimizer.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModalityExampleSchema(BaseModel):
    """
    Training example for ModalityOptimizer

    Used to train per-modality routing decisions.
    """

    query: str = Field(..., description="User query text")
    modality: str = Field(..., description="Target modality (VIDEO, DOCUMENT, IMAGE, AUDIO)")
    correct_agent: str = Field(..., description="Agent that should handle this query")
    success: bool = Field(default=True, description="Whether this represents a successful routing")
    modality_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Modality-specific features extracted from query"
    )
    is_synthetic: bool = Field(default=True, description="Whether this is synthetic data")
    synthetic_source: str = Field(
        default="backend_query",
        description="Source of synthetic data (backend_query, template, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "show me machine learning tutorial videos",
                "modality": "VIDEO",
                "correct_agent": "video_search_agent",
                "success": True,
                "is_synthetic": True,
                "synthetic_source": "backend_query"
            }
        }


class FusionHistorySchema(BaseModel):
    """
    Training example for CrossModalOptimizer

    Represents a multi-modal fusion scenario with outcome metrics.
    """

    primary_modality: str = Field(..., description="Primary modality for fusion")
    secondary_modality: str = Field(..., description="Secondary modality for fusion")
    fusion_context: Dict[str, Any] = Field(
        ...,
        description="Context about the fusion (modality_agreement, query_ambiguity, etc.)"
    )
    success: bool = Field(..., description="Whether fusion was successful")
    improvement: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Improvement gained from fusion (0-1)"
    )
    query: Optional[str] = Field(default=None, description="Original query (optional)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this fusion occurred")

    class Config:
        json_schema_extra = {
            "example": {
                "primary_modality": "VIDEO",
                "secondary_modality": "DOCUMENT",
                "fusion_context": {
                    "modality_agreement": 0.75,
                    "query_ambiguity": 0.3,
                    "content_overlap": 0.6
                },
                "success": True,
                "improvement": 0.25
            }
        }


class RoutingExperienceSchema(BaseModel):
    """
    Training example for AdvancedRoutingOptimizer

    Represents a routing decision with entity extraction and quality metrics.
    """

    query: str = Field(..., description="User query text")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted entities from query"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relationships between entities"
    )
    enhanced_query: str = Field(..., description="Query enhanced with entity information")
    chosen_agent: str = Field(..., description="Agent selected for routing")
    routing_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in routing decision (0-1)"
    )
    search_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality of search results (0-1)"
    )
    agent_success: bool = Field(..., description="Whether agent completed successfully")
    user_satisfaction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Explicit user feedback (0-1)"
    )
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    reward: Optional[float] = Field(default=None, description="Computed reward signal")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this routing occurred")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "find TensorFlow tutorials on neural networks",
                "entities": [
                    {"text": "TensorFlow", "type": "TECHNOLOGY"},
                    {"text": "neural networks", "type": "TOPIC"}
                ],
                "relationships": [
                    {"source": "TensorFlow", "target": "neural networks", "type": "USED_FOR"}
                ],
                "enhanced_query": "find TensorFlow(TECHNOLOGY) tutorials on neural networks(TOPIC)",
                "chosen_agent": "video_search_agent",
                "routing_confidence": 0.85,
                "search_quality": 0.78,
                "agent_success": True,
                "user_satisfaction": 0.9
            }
        }


class WorkflowExecutionSchema(BaseModel):
    """
    Training example for WorkflowIntelligence and UnifiedOptimizer

    Represents a complete workflow execution with performance metrics.
    """

    workflow_id: str = Field(..., description="Unique workflow identifier")
    query: str = Field(..., description="User query text")
    query_type: str = Field(..., description="Query modality/type")
    execution_time: float = Field(..., ge=0.0, description="Total execution time in seconds")
    success: bool = Field(..., description="Whether workflow completed successfully")
    agent_sequence: List[str] = Field(..., description="Sequence of agents executed")
    task_count: int = Field(..., ge=1, description="Number of tasks in workflow")
    parallel_efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Parallel execution efficiency (0-1)"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in workflow execution (0-1)"
    )
    user_satisfaction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="User satisfaction score (0-1)"
    )
    error_details: Optional[str] = Field(default=None, description="Error details if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When workflow executed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "synthetic_workflow_001",
                "query": "summarize machine learning video and create report",
                "query_type": "VIDEO",
                "execution_time": 3.5,
                "success": True,
                "agent_sequence": ["video_search_agent", "summarizer", "detailed_report"],
                "task_count": 3,
                "parallel_efficiency": 0.85,
                "confidence_score": 0.88,
                "user_satisfaction": 0.9
            }
        }


class SyntheticDataRequest(BaseModel):
    """Request schema for synthetic data generation endpoint"""

    optimizer: str = Field(
        ...,
        description="Optimizer name (modality, cross_modal, routing, workflow, unified)"
    )
    count: int = Field(..., ge=1, le=10000, description="Number of examples to generate")
    vespa_sample_size: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="Number of documents to sample from Vespa"
    )
    strategies: List[str] = Field(
        default=["diverse"],
        description="Sampling strategies (diverse, temporal_recent, entity_rich, etc.)"
    )
    max_profiles: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of backend profiles to use"
    )
    tenant_id: str = Field(default="default", description="Tenant identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "optimizer": "cross_modal",
                "count": 100,
                "vespa_sample_size": 200,
                "strategies": ["diverse"],
                "max_profiles": 3,
                "tenant_id": "default"
            }
        }


class SyntheticDataResponse(BaseModel):
    """Response schema for synthetic data generation endpoint"""

    optimizer: str = Field(..., description="Optimizer name")
    schema_name: str = Field(..., description="Schema class name used")
    count: int = Field(..., description="Number of examples generated")
    selected_profiles: List[str] = Field(..., description="Backend profiles used")
    profile_selection_reasoning: str = Field(
        ...,
        description="Reasoning for profile selection"
    )
    data: List[Dict[str, Any]] = Field(..., description="Generated synthetic data")
    metadata: Dict[str, Any] = Field(..., description="Generation metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "optimizer": "cross_modal",
                "schema_name": "FusionHistorySchema",
                "count": 100,
                "selected_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_videoprism_base_mv_chunk_30s"
                ],
                "profile_selection_reasoning": "Selected frame-based and chunk-based profiles for content diversity",
                "data": [],
                "metadata": {
                    "backend_type": "vespa",
                    "query_strategy": "cross_modal_pairs",
                    "generation_time_ms": 1250
                }
            }
        }
