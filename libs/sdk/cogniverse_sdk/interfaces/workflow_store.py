"""WorkflowStore Abstract Interface

Defines the interface for workflow intelligence storage backends.
Supports multiple implementations: Vespa, Elasticsearch, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionRecord:
    """Workflow execution record."""

    execution_id: str
    tenant_id: str
    workflow_name: str
    status: str
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "execution_id": self.execution_id,
            "tenant_id": self.tenant_id,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        """Create from dictionary."""
        return cls(
            execution_id=data["execution_id"],
            tenant_id=data["tenant_id"],
            workflow_name=data["workflow_name"],
            status=data["status"],
            metrics=data.get("metrics", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class AgentPerformanceRecord:
    """Agent performance record for a single execution."""

    performance_id: str
    tenant_id: str
    agent_type: str
    duration_ms: float
    success: bool
    metrics: Dict[str, Any]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "performance_id": self.performance_id,
            "tenant_id": self.tenant_id,
            "agent_type": self.agent_type,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPerformanceRecord":
        """Create from dictionary."""
        return cls(
            performance_id=data["performance_id"],
            tenant_id=data["tenant_id"],
            agent_type=data["agent_type"],
            duration_ms=data.get("duration_ms", 0.0),
            success=data.get("success", True),
            metrics=data.get("metrics", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class AgentStats:
    """Aggregated agent performance statistics."""

    agent_type: str
    tenant_id: str
    total_executions: int
    avg_duration_ms: float
    success_rate: float
    last_execution: Optional[datetime]


@dataclass
class WorkflowTemplate:
    """Workflow template configuration."""

    template_id: str
    tenant_id: str
    template_name: str
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "template_id": self.template_id,
            "tenant_id": self.tenant_id,
            "template_name": self.template_name,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTemplate":
        """Create from dictionary."""
        return cls(
            template_id=data["template_id"],
            tenant_id=data["tenant_id"],
            template_name=data["template_name"],
            config=data.get("config", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class WorkflowStore(ABC):
    """
    Abstract interface for workflow intelligence storage.

    Implementations:
    - VespaWorkflowStore: Vespa backend storage
    - ElasticsearchWorkflowStore: Elasticsearch backend storage (future)
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the workflow store.

        Creates necessary tables/schemas/indices for storage.
        """
        pass

    # ==================== Execution Records ====================

    @abstractmethod
    def record_execution(
        self,
        tenant_id: str,
        workflow_name: str,
        status: str,
        metrics: Dict[str, Any],
    ) -> str:
        """
        Record workflow execution.

        Args:
            tenant_id: Tenant identifier
            workflow_name: Name of the workflow
            status: Execution status (e.g., "completed", "failed")
            metrics: Execution metrics (duration, steps, etc.)

        Returns:
            execution_id: Unique identifier for this execution
        """
        pass

    @abstractmethod
    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """
        Get execution by ID.

        Args:
            execution_id: Unique execution identifier

        Returns:
            ExecutionRecord if found, None otherwise
        """
        pass

    @abstractmethod
    def list_executions(
        self,
        tenant_id: str,
        workflow_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[ExecutionRecord]:
        """
        List executions with optional filters.

        Args:
            tenant_id: Tenant identifier
            workflow_name: Filter by workflow name (None = all workflows)
            limit: Maximum number of records to return

        Returns:
            List of ExecutionRecord sorted by created_at (newest first)
        """
        pass

    # ==================== Agent Performance ====================

    @abstractmethod
    def record_agent_performance(
        self,
        tenant_id: str,
        agent_type: str,
        duration_ms: float,
        success: bool,
        metrics: Dict[str, Any],
    ) -> str:
        """
        Record agent performance for a single execution.

        Args:
            tenant_id: Tenant identifier
            agent_type: Type of agent (e.g., "routing", "search")
            duration_ms: Execution duration in milliseconds
            success: Whether execution was successful
            metrics: Additional performance metrics

        Returns:
            performance_id: Unique identifier for this record
        """
        pass

    @abstractmethod
    def get_agent_stats(
        self,
        tenant_id: str,
        agent_type: str,
    ) -> Optional[AgentStats]:
        """
        Get aggregated agent statistics.

        Args:
            tenant_id: Tenant identifier
            agent_type: Type of agent

        Returns:
            AgentStats with aggregated metrics, None if no data
        """
        pass

    @abstractmethod
    def list_agent_performance(
        self,
        tenant_id: str,
        agent_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentPerformanceRecord]:
        """
        List agent performance records.

        Args:
            tenant_id: Tenant identifier
            agent_type: Filter by agent type (None = all agents)
            limit: Maximum number of records to return

        Returns:
            List of AgentPerformanceRecord sorted by created_at (newest first)
        """
        pass

    # ==================== Workflow Templates ====================

    @abstractmethod
    def save_template(
        self,
        tenant_id: str,
        template_name: str,
        config: Dict[str, Any],
    ) -> str:
        """
        Save workflow template.

        Creates new or updates existing template.

        Args:
            tenant_id: Tenant identifier
            template_name: Unique name for this template
            config: Template configuration

        Returns:
            template_id: Unique identifier for this template
        """
        pass

    @abstractmethod
    def get_template(
        self,
        tenant_id: str,
        template_name: str,
    ) -> Optional[WorkflowTemplate]:
        """
        Get template by name.

        Args:
            tenant_id: Tenant identifier
            template_name: Template name

        Returns:
            WorkflowTemplate if found, None otherwise
        """
        pass

    @abstractmethod
    def list_templates(self, tenant_id: str) -> List[WorkflowTemplate]:
        """
        List all templates for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of WorkflowTemplate
        """
        pass

    @abstractmethod
    def delete_template(
        self,
        tenant_id: str,
        template_name: str,
    ) -> bool:
        """
        Delete a template.

        Args:
            tenant_id: Tenant identifier
            template_name: Template name

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== Utility Methods ====================

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if storage backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with stats (total executions, templates, etc.)
        """
        pass
