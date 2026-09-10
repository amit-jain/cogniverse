"""
Backend Configuration for Mem0

Minimal backend-agnostic configuration for mem0 vector store integration.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BackendConfig(BaseModel):
    """
    Backend-agnostic configuration for mem0 vector store.

    Carries the resolver the store calls per operation, avoiding direct
    backend-specific dependencies (host, port, url, etc.). mem0's
    VectorStoreFactory (mem0/utils/factory.py:199) passes this through
    ``model_dump()``, which preserves a callable by identity.
    """

    collection_name: str = Field(
        ..., description="Schema/collection name for memory storage"
    )
    backend_resolver: Any = Field(
        ...,
        description=(
            "Zero-arg callable resolving the backend through BackendRegistry. "
            "A held instance is closed by eviction, an overwriting set or "
            "clear, so the store resolves and leases per operation."
        ),
    )
    embedding_model_dims: int = Field(768, description="Embedding vector dimensions")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenant isolation")
    profile: str = Field(
        ..., description="Base schema/profile name (without tenant suffix)"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True  # Allow backend_resolver (non-pydantic)
    )
