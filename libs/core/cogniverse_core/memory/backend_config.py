"""
Backend Configuration for Mem0

Minimal backend-agnostic configuration for mem0 vector store integration.
"""

from typing import Any

from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    """
    Backend-agnostic configuration for mem0 vector store.

    This config wraps a pre-configured backend client, avoiding direct
    backend-specific dependencies (host, port, url, etc.).
    """

    collection_name: str = Field(
        ..., description="Schema/collection name for memory storage"
    )
    backend_client: Any = Field(
        ..., description="Pre-configured backend client instance"
    )
    embedding_model_dims: int = Field(
        768, description="Embedding vector dimensions"
    )
    tenant_id: str = Field(
        ..., description="Tenant ID for multi-tenant isolation"
    )
    profile: str = Field(
        ..., description="Base schema/profile name (without tenant suffix)"
    )

    class Config:
        arbitrary_types_allowed = True  # Allow backend_client (non-pydantic)
