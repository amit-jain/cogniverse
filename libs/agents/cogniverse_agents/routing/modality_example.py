"""
Modality Example Data Structure

Legacy wrapper for ModalityExampleSchema to maintain backward compatibility.
"""

from typing import Any, Dict, Optional

from cogniverse_agents.search.multi_modal_reranker import QueryModality
from cogniverse_synthetic import ModalityExampleSchema


class ModalityExample:
    """Legacy wrapper for ModalityExampleSchema"""

    def __init__(
        self,
        query: str,
        modality: QueryModality,
        correct_agent: str,
        success: bool,
        modality_features: Optional[Dict[str, Any]] = None,
        is_synthetic: bool = False,
        synthetic_source: Optional[str] = None,
    ):
        self.query = query
        self.modality = modality
        self.correct_agent = correct_agent
        self.success = success
        self.modality_features = modality_features
        self.is_synthetic = is_synthetic
        self.synthetic_source = synthetic_source

    @classmethod
    def from_schema(cls, schema: ModalityExampleSchema) -> "ModalityExample":
        """Create from new schema"""
        # Convert uppercase modality string to enum (VIDEO -> video)
        modality_str = schema.modality.lower()
        return cls(
            query=schema.query,
            modality=QueryModality(modality_str),
            correct_agent=schema.correct_agent,
            success=schema.success,
            modality_features=schema.modality_features,
            is_synthetic=schema.is_synthetic,
            synthetic_source=schema.synthetic_source,
        )
