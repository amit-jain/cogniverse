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

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _known_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unknown keys so a payload written by a newer schema still
    deserializes — cls(**data) raised TypeError on the first extra key."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _utc_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return value.astimezone(timezone.utc)


def _datetime_from_payload(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be an ISO-8601 string, got {type(value).__name__}"
        )
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not valid ISO-8601: {exc}") from None


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = _utc_datetime(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowExecution":
        data = _known_fields(cls, data)
        if "timestamp" in data:
            data["timestamp"] = _datetime_from_payload(data["timestamp"], "timestamp")
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
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.last_updated = _utc_datetime(self.last_updated, "last_updated")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPerformance":
        data = _known_fields(cls, data)
        if "last_updated" in data:
            data["last_updated"] = _datetime_from_payload(
                data["last_updated"], "last_updated"
            )
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.created_at = _utc_datetime(self.created_at, "created_at")
        if self.last_used is not None:
            self.last_used = _utc_datetime(self.last_used, "last_used")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_used"] = self.last_used.isoformat() if self.last_used else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTemplate":
        data = _known_fields(cls, data)
        if "created_at" in data:
            data["created_at"] = _datetime_from_payload(
                data["created_at"], "created_at"
            )
        if data.get("last_used") is not None:
            data["last_used"] = _datetime_from_payload(data["last_used"], "last_used")
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

    # ==================== Atomic corpus write ====================

    async def save_learning_corpus(
        self,
        tenant_id: str,
        executions: List[WorkflowExecution],
        profiles: List[AgentPerformance],
        patterns: Dict[str, List[str]],
    ) -> None:
        """Write the three learning corpora as one unit.

        The three saves share no transaction, so a mid-sequence outage would
        otherwise leave, e.g., executions saved without matching profiles — and
        the orchestrator reads back execution demos referencing agents whose
        profiles are missing. Two guards: executions (the only corpus that
        references agents) is written LAST, so a failure before it never
        persists a dangling reference; and on any failure the previous corpus is
        restored so a partial write is undone.
        """
        prev_profiles = await self.load_agent_profiles(tenant_id)
        prev_patterns = await self.load_query_patterns(tenant_id)
        prev_executions = await self.load_executions(tenant_id)
        try:
            await self.save_agent_profiles(tenant_id, profiles)
            if patterns:
                await self.save_query_patterns(tenant_id, patterns)
            await self.save_executions(tenant_id, executions)
        except Exception:
            try:
                await self.save_agent_profiles(tenant_id, prev_profiles)
                await self.save_query_patterns(tenant_id, prev_patterns)
                await self.save_executions(tenant_id, prev_executions)
                logger.warning(
                    "Learning-corpus save failed for %s; restored previous corpus",
                    tenant_id,
                )
            except Exception:
                logger.exception(
                    "Learning-corpus save failed for %s and the restore also "
                    "failed; the corpus may be inconsistent",
                    tenant_id,
                )
            raise

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
