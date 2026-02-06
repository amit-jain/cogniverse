"""
Adapter Registry Models

Dataclasses for adapter metadata stored in Vespa.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional


@dataclass
class AdapterMetadata:
    """
    Complete metadata for a trained LoRA adapter.

    This is the primary data structure for the model registry, stored in Vespa
    and used for adapter management, versioning, and deployment to agents.

    Attributes:
        adapter_id: Unique identifier for this adapter (UUID)
        tenant_id: Tenant this adapter belongs to
        name: Human-readable name (e.g., "routing_sft_v1")
        version: Semantic version string (e.g., "1.0.0")
        base_model: Base model the adapter was trained on
        model_type: Type of model - "llm" or "embedding"
        agent_type: Target agent type (routing, profile_selection, entity_extraction)
        training_method: Training method used (sft, dpo, embedding)
        adapter_path: Local filesystem path to adapter weights (for local dev)
        adapter_uri: Cloud storage URI (s3://, gs://, modal://, file://) for production
        status: Lifecycle status (active, inactive, deprecated)
        is_active: Whether this is the active adapter for tenant+agent_type
        metrics: Training metrics (loss, accuracy, etc.)
        training_config: Training configuration used (hyperparameters)
        experiment_run_id: Optional MLflow/Phoenix experiment run ID
        created_at: When the adapter was registered
        updated_at: When the metadata was last updated
    """

    adapter_id: str
    tenant_id: str
    name: str
    version: str
    base_model: str
    model_type: Literal["llm", "embedding"]
    agent_type: Optional[str]
    training_method: Literal["sft", "dpo", "embedding", "sft_multi_turn"]
    adapter_path: str
    adapter_uri: Optional[str] = None  # Cloud storage URI for production
    status: Literal["active", "inactive", "deprecated"] = "inactive"
    is_active: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    experiment_run_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_vespa_doc(self) -> Dict[str, Any]:
        """
        Convert to Vespa document format.

        Returns:
            Dict ready for Vespa ingestion
        """
        return {
            "adapter_id": self.adapter_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "version": self.version,
            "base_model": self.base_model,
            "model_type": self.model_type,
            "agent_type": self.agent_type or "",
            "training_method": self.training_method,
            "adapter_path": self.adapter_path,
            "adapter_uri": self.adapter_uri or "",
            "status": self.status,
            "is_active": 1 if self.is_active else 0,
            "metrics": json.dumps(self.metrics),
            "training_config": json.dumps(self.training_config),
            "experiment_run_id": self.experiment_run_id or "",
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_vespa_doc(cls, doc: Dict[str, Any]) -> "AdapterMetadata":
        """
        Create AdapterMetadata from Vespa document.

        Args:
            doc: Vespa document fields

        Returns:
            AdapterMetadata instance
        """
        # Handle nested fields structure from Vespa response
        fields = doc.get("fields", doc)

        # Parse JSON fields
        metrics = fields.get("metrics", "{}")
        if isinstance(metrics, str):
            metrics = json.loads(metrics) if metrics else {}

        training_config = fields.get("training_config", "{}")
        if isinstance(training_config, str):
            training_config = json.loads(training_config) if training_config else {}

        # Parse timestamps
        created_at = fields.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.utcnow()

        updated_at = fields.get("updated_at", "")
        if isinstance(updated_at, str) and updated_at:
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.utcnow()

        return cls(
            adapter_id=fields.get("adapter_id", ""),
            tenant_id=fields.get("tenant_id", ""),
            name=fields.get("name", ""),
            version=fields.get("version", ""),
            base_model=fields.get("base_model", ""),
            model_type=fields.get("model_type", "llm"),
            agent_type=fields.get("agent_type") or None,
            training_method=fields.get("training_method", "sft"),
            adapter_path=fields.get("adapter_path", ""),
            adapter_uri=fields.get("adapter_uri") or None,
            status=fields.get("status", "inactive"),
            is_active=bool(fields.get("is_active", 0)),
            metrics=metrics,
            training_config=training_config,
            experiment_run_id=fields.get("experiment_run_id") or None,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_effective_uri(self) -> str:
        """
        Get the effective URI for loading the adapter.

        Returns adapter_uri if set (for cloud/Modal storage),
        otherwise returns file:// URI from adapter_path.

        Returns:
            URI string (s3://, modal://, file://, etc.)
        """
        if self.adapter_uri:
            return self.adapter_uri
        if self.adapter_path:
            return f"file://{self.adapter_path}"
        return ""

    def __str__(self) -> str:
        """Human-readable string representation."""
        active_marker = " [ACTIVE]" if self.is_active else ""
        return (
            f"Adapter({self.name} v{self.version}{active_marker}, "
            f"tenant={self.tenant_id}, agent={self.agent_type}, "
            f"method={self.training_method}, status={self.status})"
        )
