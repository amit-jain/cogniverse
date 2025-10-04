"""
Vespa Configuration for Mem0

Defines the configuration schema for using Vespa as a vector store in Mem0.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class VespaConfig(BaseModel):
    """Configuration for Vespa vector store"""

    collection_name: str = Field(
        "agent_memories", description="Name of the Vespa schema/document type"
    )
    embedding_model_dims: Optional[int] = Field(
        768, description="Dimensions of the embedding model"
    )
    host: str = Field("localhost", description="Vespa endpoint host")
    port: int = Field(8080, description="Vespa endpoint port")

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that no extra fields are provided"""
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. "
                f"Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values
