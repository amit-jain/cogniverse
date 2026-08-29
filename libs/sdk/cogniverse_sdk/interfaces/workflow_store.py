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

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _canonical_payload(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Require the exact field set emitted by the record's serializer."""
    if not isinstance(data, dict):
        raise ValueError(f"payload must be a dict, got {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    missing = names - set(data)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    return dict(data)


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
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not valid ISO-8601: {exc}") from None
    canonical = _utc_datetime(parsed, field_name)
    if value != canonical.isoformat():
        raise ValueError(
            f"{field_name} must use canonical UTC ISO-8601 form, got {value!r}"
        )
    return canonical


def _require_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {type(value).__name__}")


def _require_nonnegative_integer(value: Any, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field_name} must be a non-negative integer")


def _require_finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{field_name} must be a finite float")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum:g}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be between {minimum:g} and {maximum:g}")


def _require_string_list(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of str")


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
        _require_string(self.workflow_id, "workflow_id")
        _require_string(self.query, "query")
        _require_string(self.query_type, "query_type")
        _require_finite_float(self.execution_time, "execution_time", minimum=0.0)
        if type(self.success) is not bool:
            raise TypeError(
                f"success must be a bool, got {type(self.success).__name__}"
            )
        _require_string_list(self.agent_sequence, "agent_sequence")
        _require_nonnegative_integer(self.task_count, "task_count")
        _require_finite_float(
            self.parallel_efficiency,
            "parallel_efficiency",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_float(
            self.confidence_score,
            "confidence_score",
            minimum=0.0,
            maximum=1.0,
        )
        if self.user_satisfaction is not None:
            _require_finite_float(
                self.user_satisfaction,
                "user_satisfaction",
                minimum=0.0,
                maximum=1.0,
            )
        if self.error_details is not None and not isinstance(self.error_details, str):
            raise TypeError("error_details must be a str or None")
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"metadata must be a dict, got {type(self.metadata).__name__}"
            )
        self.timestamp = _utc_datetime(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowExecution":
        data = _canonical_payload(cls, data)
        data["timestamp"] = _datetime_from_payload(data["timestamp"], "timestamp")
        return cls(**data)


@dataclass
class AgentPerformance:
    """Agent performance profile aggregated across executions."""

    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    average_execution_time: float = 0.0
    average_confidence: Optional[float] = None
    error_rate: float = 0.0
    preferred_query_types: List[str] = field(default_factory=list)
    performance_trend: str = "stable"  # improving, degrading, stable
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _require_string(self.agent_name, "agent_name")
        _require_nonnegative_integer(self.total_executions, "total_executions")
        _require_nonnegative_integer(
            self.successful_executions, "successful_executions"
        )
        if self.successful_executions > self.total_executions:
            raise ValueError("successful_executions cannot exceed total_executions")
        _require_finite_float(
            self.average_execution_time,
            "average_execution_time",
            minimum=0.0,
        )
        if self.average_confidence is not None:
            _require_finite_float(
                self.average_confidence,
                "average_confidence",
                minimum=0.0,
                maximum=1.0,
            )
        _require_finite_float(
            self.error_rate,
            "error_rate",
            minimum=0.0,
            maximum=1.0,
        )
        _require_string_list(self.preferred_query_types, "preferred_query_types")
        if self.performance_trend not in {"improving", "degrading", "stable"}:
            raise ValueError(
                "performance_trend must be improving, degrading, or stable"
            )
        self.last_updated = _utc_datetime(self.last_updated, "last_updated")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPerformance":
        data = _canonical_payload(cls, data)
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
    expected_execution_time: Optional[float]
    success_rate: Optional[float]
    usage_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None

    def __post_init__(self) -> None:
        _require_string(self.template_id, "template_id")
        _require_string(self.name, "name")
        _require_string(self.description, "description")
        _require_string_list(self.query_patterns, "query_patterns")
        if not isinstance(self.task_sequence, list) or any(
            not isinstance(task, dict) for task in self.task_sequence
        ):
            raise TypeError("task_sequence must be a list of dict")
        if self.expected_execution_time is not None:
            _require_finite_float(
                self.expected_execution_time,
                "expected_execution_time",
                minimum=0.0,
            )
        if self.success_rate is not None:
            _require_finite_float(
                self.success_rate,
                "success_rate",
                minimum=0.0,
                maximum=1.0,
            )
        _require_nonnegative_integer(self.usage_count, "usage_count")
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
        data = _canonical_payload(cls, data)
        data["created_at"] = _datetime_from_payload(data["created_at"], "created_at")
        if data["last_used"] is not None:
            data["last_used"] = _datetime_from_payload(data["last_used"], "last_used")
        return cls(**data)


@dataclass(frozen=True)
class WorkflowLearningState:
    """One complete generation of workflow-learning data."""

    executions: List[WorkflowExecution]
    profiles: List[AgentPerformance]
    patterns: Dict[str, List[str]]
    templates: List[WorkflowTemplate]


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
        restored so a partial write is undone. Writes for one tenant are
        serialized; different tenants use independent locks.
        """
        locks = self.__dict__.setdefault("_learning_corpus_locks", {})
        lock = locks.setdefault(tenant_id, asyncio.Lock())
        async with lock:
            await self._save_learning_corpus_locked(
                tenant_id, executions, profiles, patterns
            )

    async def _save_learning_corpus_locked(
        self,
        tenant_id: str,
        executions: List[WorkflowExecution],
        profiles: List[AgentPerformance],
        patterns: Dict[str, List[str]],
    ) -> None:
        prev_profiles = await self.load_agent_profiles(tenant_id)
        prev_patterns = await self.load_query_patterns(tenant_id)
        prev_executions = await self.load_executions(tenant_id)
        try:
            await self.save_agent_profiles(tenant_id, profiles)
            await self.save_query_patterns(tenant_id, patterns)
            await self.save_executions(tenant_id, executions)
        except Exception as forward_error:
            restore_errors = []
            restore_steps = [
                ("agent profiles", self.save_agent_profiles, prev_profiles),
                ("query patterns", self.save_query_patterns, prev_patterns),
                ("executions", self.save_executions, prev_executions),
            ]
            for label, restore, previous in restore_steps:
                try:
                    await restore(tenant_id, previous)
                except Exception as restore_error:
                    restore_error.add_note(
                        f"while restoring {label} for tenant {tenant_id!r}"
                    )
                    restore_errors.append(restore_error)
            if restore_errors:
                logger.error(
                    "Learning-corpus save and restore failed for %s",
                    tenant_id,
                )
                raise ExceptionGroup(
                    f"Learning-corpus save and restore failed for {tenant_id!r}",
                    [forward_error, *restore_errors],
                ) from forward_error
            logger.warning(
                "Learning-corpus save failed for %s; restored previous corpus",
                tenant_id,
            )
            raise

    @abstractmethod
    async def replace_learning_state(
        self,
        tenant_id: str,
        executions: List[WorkflowExecution],
        profiles: List[AgentPerformance],
        patterns: Dict[str, List[str]],
        templates: List[WorkflowTemplate],
    ) -> None:
        """Replace every workflow-learning channel as one coordinated write.

        Implementations must serialize writers for the same tenant across
        processes and restore the exact prior templates, executions, profiles,
        and query patterns before propagating a failed replacement.
        """

    @abstractmethod
    async def load_learning_state(self, tenant_id: str) -> WorkflowLearningState:
        """Load one complete workflow-learning generation.

        Implementations must coordinate this read with replacements for the
        same tenant so the returned channels all belong to one generation.
        """

    # ==================== Workflow Templates ====================

    @abstractmethod
    async def save_template(self, tenant_id: str, template: WorkflowTemplate) -> str:
        """Create or update a template; returns its ``template_id``."""

    @abstractmethod
    async def save_generated_templates(
        self,
        tenant_id: str,
        templates: List[WorkflowTemplate],
    ) -> List[str]:
        """Persist one generated batch atomically and return its ordered IDs."""

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
