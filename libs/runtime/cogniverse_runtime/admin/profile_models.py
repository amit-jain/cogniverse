"""
Pydantic Models for Profile Management API

Request/response models for backend profile CRUD operations.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProfileCreateRequest(BaseModel):
    """Request model for creating a new backend profile."""

    profile_name: str = Field(
        ...,
        description="Unique profile identifier (alphanumeric, underscore, hyphen)",
        min_length=1,
        max_length=100,
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenancy isolation"
    )
    type: str = Field(
        default="video",
        description="Profile type (video, image, audio, text)",
    )
    description: str = Field(
        default="", description="Human-readable profile description"
    )
    schema_name: str = Field(
        ..., description="Base schema name (must have template in configs/schemas/)"
    )
    embedding_model: str = Field(
        ..., description="Embedding model identifier (e.g., 'vidore/colsmol-500m')"
    )
    pipeline_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline configuration (keyframe extraction, transcription, etc.)",
    )
    strategies: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing strategy configurations (segmentation, embedding, etc.)",
    )
    embedding_type: str = Field(
        ...,
        description="Embedding type (frame_based, video_chunks, direct_video_segment, single_vector)",
    )
    schema_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Schema metadata (dimensions, model_name, patches, etc.)",
    )
    model_specific: Optional[Dict[str, Any]] = Field(
        default=None, description="Model-specific parameters"
    )
    deploy_schema: bool = Field(
        default=False, description="Deploy schema to Vespa immediately after creation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "profile_name": "custom_colpali_high_quality",
                "tenant_id": "acme",
                "type": "video",
                "description": "High-quality ColPali with 60 FPS keyframe extraction",
                "schema_name": "video_colpali_smol500_mv_frame",
                "embedding_model": "vidore/colsmol-500m",
                "pipeline_config": {
                    "extract_keyframes": True,
                    "transcribe_audio": True,
                    "generate_descriptions": True,
                    "keyframe_fps": 60.0,
                },
                "strategies": {
                    "segmentation": {
                        "class": "FrameSegmentationStrategy",
                        "params": {"fps": 60.0, "max_frames": 5000},
                    },
                    "embedding": {
                        "class": "MultiVectorEmbeddingStrategy",
                        "params": {},
                    },
                },
                "embedding_type": "frame_based",
                "schema_config": {
                    "schema_name": "video_colpali",
                    "model_name": "ColPali",
                    "num_patches": 1024,
                    "embedding_dim": 128,
                    "binary_dim": 16,
                },
                "deploy_schema": True,
            }
        }


class ProfileCreateResponse(BaseModel):
    """Response model for profile creation."""

    profile_name: str = Field(..., description="Created profile name")
    tenant_id: str = Field(..., description="Tenant identifier")
    schema_deployed: bool = Field(..., description="Whether schema was deployed")
    tenant_schema_name: Optional[str] = Field(
        None, description="Tenant-specific schema name (if deployed)"
    )
    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")
    version: int = Field(..., description="Config version number")


class ProfileSummary(BaseModel):
    """Summary information for a profile (used in list responses)."""

    profile_name: str = Field(..., description="Profile name")
    type: str = Field(..., description="Profile type")
    description: str = Field(..., description="Profile description")
    schema_name: str = Field(..., description="Base schema name")
    embedding_model: str = Field(..., description="Embedding model identifier")
    schema_deployed: bool = Field(..., description="Whether schema is deployed")
    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")


class ProfileListResponse(BaseModel):
    """Response model for listing profiles."""

    profiles: List[ProfileSummary] = Field(..., description="List of profiles")
    total_count: int = Field(..., description="Total number of profiles")
    tenant_id: str = Field(..., description="Tenant identifier")


class ProfileDetail(BaseModel):
    """Detailed profile information (used in GET responses)."""

    profile_name: str
    tenant_id: str
    type: str
    description: str
    schema_name: str
    embedding_model: str
    pipeline_config: Dict[str, Any]
    strategies: Dict[str, Any]
    embedding_type: str
    schema_config: Dict[str, Any]
    model_specific: Optional[Dict[str, Any]] = None
    schema_deployed: bool = Field(..., description="Whether schema is deployed")
    tenant_schema_name: Optional[str] = Field(
        None, description="Tenant-specific schema name (if deployed)"
    )
    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")
    version: int = Field(..., description="Config version number")


class ProfileUpdateRequest(BaseModel):
    """Request model for updating a profile."""

    tenant_id: str = Field(..., description="Tenant identifier")
    pipeline_config: Optional[Dict[str, Any]] = Field(
        None, description="Updated pipeline configuration"
    )
    strategies: Optional[Dict[str, Any]] = Field(
        None, description="Updated strategy configurations"
    )
    description: Optional[str] = Field(None, description="Updated description")
    model_specific: Optional[Dict[str, Any]] = Field(
        None, description="Updated model-specific parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "acme",
                "pipeline_config": {"keyframe_fps": 30.0},
                "description": "Updated description",
            }
        }


class ProfileUpdateResponse(BaseModel):
    """Response model for profile update."""

    profile_name: str = Field(..., description="Updated profile name")
    tenant_id: str = Field(..., description="Tenant identifier")
    updated_fields: List[str] = Field(..., description="List of updated field names")
    version: int = Field(..., description="New config version number")


class ProfileDeleteResponse(BaseModel):
    """Response model for profile deletion."""

    profile_name: str = Field(..., description="Deleted profile name")
    tenant_id: str = Field(..., description="Tenant identifier")
    schema_deleted: bool = Field(..., description="Whether schema was also deleted")
    deleted_at: str = Field(..., description="Deletion timestamp (ISO 8601)")


class SchemaDeploymentRequest(BaseModel):
    """Request model for schema deployment."""

    tenant_id: str = Field(..., description="Tenant identifier")
    force: bool = Field(
        default=False,
        description="Force redeployment even if schema already exists",
    )


class SchemaDeploymentResponse(BaseModel):
    """Response model for schema deployment."""

    profile_name: str = Field(..., description="Profile name")
    tenant_id: str = Field(..., description="Tenant identifier")
    schema_name: str = Field(..., description="Base schema name")
    tenant_schema_name: str = Field(..., description="Tenant-specific schema name")
    deployment_status: str = Field(
        ...,
        description="Deployment status (success, failed, already_deployed)",
    )
    deployed_at: str = Field(..., description="Deployment timestamp (ISO 8601)")
    error_message: Optional[str] = Field(
        None, description="Error message if deployment failed"
    )
