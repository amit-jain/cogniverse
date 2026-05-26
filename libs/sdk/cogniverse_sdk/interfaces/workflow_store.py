"""WorkflowStore interface — typed persistence for workflow intelligence.

Defines the storage contract that ``WorkflowIntelligence`` (the reader, loaded
at orchestrator startup) and the batch optimizer (the writer) share: workflow
executions, agent performance profiles, query-type patterns, and reusable
templates. Implementations register against the ``cogniverse.workflow.stores``
entry-point group and are resolved through ``WorkflowStoreRegistry``.

The domain dataclasses live here rather than in the agents package so the
interface is fully typed to them without the core registry having to import
agents. The data methods are ``async`` because the only backend today
(telemetry/Phoenix via ``ArtifactManager``) and both callers are async; the
trivial lifecycle/health methods stay sync.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class WorkflowExecution:
    """Historical workflow execution record."""

    workflow_id: str
    query: str
    query_type: str
    execution_time: float
    success: bool
    agent_sequence: List[str]
    task_count: int
    parallel_efficiency: float
    confidence_score: float
    user_satisfaction: Optional[float] = None
    error_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowExecution":
        data = dict(data)
        ts = data.get("timestamp")
        if isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts)
        return cls(**data)


@dataclass
class AgentPerformance:
    """Agent performance profile aggregated across executions."""

    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    average_execution_time: float = 0.0
    average_confidence: float = 0.0
    error_rate: float = 0.0
    preferred_query_types: List[str] = field(default_factory=list)
    performance_trend: str = "stable"  # improving, degrading, stable
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPerformance":
        data = dict(data)
        lu = data.get("last_updated")
        if isinstance(lu, str):
            data["last_updated"] = datetime.fromisoformat(lu)
        return cls(**data)


@dataclass
class WorkflowTemplate:
    """Reusable workflow template."""

    template_id: str
    name: str
    description: str
    query_patterns: List[str]
    task_sequence: List[Dict[str, Any]]
    expected_execution_time: float
    success_rate: float
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_used"] = self.last_used.isoformat() if self.last_used else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTemplate":
        data = dict(data)
        ca = data.get("created_at")
        if isinstance(ca, str):
            data["created_at"] = datetime.fromisoformat(ca)
        lu = data.get("last_used")
        if isinstance(lu, str):
            data["last_used"] = datetime.fromisoformat(lu)
        return cls(**data)


class WorkflowStore(ABC):
    """Typed persistence for workflow intelligence.

    Implementations:
    - TelemetryWorkflowStore: persists through the telemetry substrate
      (Phoenix datasets/blobs via ArtifactManager), following whatever
      observability backend the active TelemetryProvider targets.

    Save methods replace the stored set for the tenant (the optimizer rebuilds
    history from spans each batch run); ``save_template`` upserts one template.
    """

    def initialize(self) -> None:
        """Provision backing storage. Default: no-op (lazy creation)."""
        return None

    # ==================== Workflow Executions ====================

    @abstractmethod
    async def save_executions(
        self, tenant_id: str, executions: List[WorkflowExecution]
    ) -> None:
        """Replace the tenant's stored executions with ``executions``."""

    @abstractmethod
    async def load_executions(self, tenant_id: str) -> List[WorkflowExecution]:
        """Load all stored executions for the tenant."""

    # ==================== Agent Performance Profiles ====================

    @abstractmethod
    async def save_agent_profiles(
        self, tenant_id: str, profiles: List[AgentPerformance]
    ) -> None:
        """Replace the tenant's stored agent performance profiles."""

    @abstractmethod
    async def load_agent_profiles(self, tenant_id: str) -> List[AgentPerformance]:
        """Load all stored agent performance profiles for the tenant."""

    # ==================== Query-Type Patterns ====================

    @abstractmethod
    async def save_query_patterns(
        self, tenant_id: str, patterns: Dict[str, List[str]]
    ) -> None:
        """Replace the tenant's query-type → patterns mapping."""

    @abstractmethod
    async def load_query_patterns(self, tenant_id: str) -> Dict[str, List[str]]:
        """Load the tenant's query-type → patterns mapping ({} if none)."""

    # ==================== Workflow Templates ====================

    @abstractmethod
    async def save_template(self, tenant_id: str, template: WorkflowTemplate) -> str:
        """Create or update a template; returns its ``template_id``."""

    @abstractmethod
    async def load_templates(self, tenant_id: str) -> List[WorkflowTemplate]:
        """Load all templates for the tenant."""

    @abstractmethod
    async def delete_template(self, tenant_id: str, template_id: str) -> bool:
        """Delete a template by id; returns False if it did not exist."""

    # ==================== Utility ====================

    @abstractmethod
    def health_check(self) -> bool:
        """Whether the backing store is reachable/usable."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Backend-identifying stats (backend name, cache sizes, …)."""
