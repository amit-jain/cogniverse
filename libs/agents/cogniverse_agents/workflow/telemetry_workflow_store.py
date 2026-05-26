"""Telemetry-backed implementation of the WorkflowStore abstraction.

Persists workflow executions, agent performance, and templates through the
telemetry substrate (Phoenix datasets/blobs via ``ArtifactManager``) — the
same channel the optimizer reads from — so there is no separate backend and
no second source of truth.

The ``WorkflowStore`` interface is synchronous while ``ArtifactManager`` is
async, so each method bridges via ``run_coro_blocking``. The interface is
multi-tenant (tenant_id per call) while ``ArtifactManager`` is tenant-scoped
at construction, so one manager is cached per tenant. ``get_execution`` takes
only an ``execution_id`` (no tenant), so execution ids are tenant-qualified
(``"{tenant_id}|exec|{uuid}"``) and the tenant is recovered from the id.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cogniverse_core.common.utils.async_bridge import run_coro_blocking
from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformanceRecord,
    AgentStats,
    ExecutionRecord,
    WorkflowStore,
    WorkflowTemplate,
)

# Blob coordinates (kind, key) under ArtifactManager.
_KIND = "workflow"
_EXECUTIONS_KEY = "executions"
_PERFORMANCE_KEY = "agent_performance"
_TEMPLATE_KIND = "workflow_template"
_TEMPLATE_INDEX_KEY = "_index"
# Cap blob growth — telemetry storage is a rolling window, not an audit log.
_MAX_RECORDS = 1000
_EXEC_ID_SEP = "|exec|"
_PERF_ID_PREFIX = "perf_"


class TelemetryWorkflowStore(WorkflowStore):
    """WorkflowStore backed by ArtifactManager (Phoenix datasets/blobs)."""

    def __init__(self, telemetry_provider: Any) -> None:
        self._provider = telemetry_provider
        self._am_cache: Dict[str, Any] = {}

    def _am(self, tenant_id: str):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        if tenant_id not in self._am_cache:
            self._am_cache[tenant_id] = ArtifactManager(self._provider, tenant_id)
        return self._am_cache[tenant_id]

    def _load_list(self, tenant_id: str, key: str) -> List[Dict[str, Any]]:
        blob = run_coro_blocking(self._am(tenant_id).load_blob(_KIND, key))
        if not blob:
            return []
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            return []
        return data if isinstance(data, list) else []

    def _save_list(
        self, tenant_id: str, key: str, records: List[Dict[str, Any]]
    ) -> None:
        run_coro_blocking(
            self._am(tenant_id).save_blob(
                _KIND, key, json.dumps(records[-_MAX_RECORDS:])
            )
        )

    def initialize(self) -> None:
        """No provisioning needed — ArtifactManager creates datasets lazily."""
        return None

    # ==================== Execution Records ====================

    def record_execution(
        self, tenant_id: str, workflow_name: str, status: str, metrics: Dict[str, Any]
    ) -> str:
        now = datetime.now(timezone.utc)
        execution_id = f"{tenant_id}{_EXEC_ID_SEP}{uuid.uuid4().hex[:16]}"
        record = ExecutionRecord(
            execution_id=execution_id,
            tenant_id=tenant_id,
            workflow_name=workflow_name,
            status=status,
            metrics=metrics,
            created_at=now,
            updated_at=now,
        )
        records = self._load_list(tenant_id, _EXECUTIONS_KEY)
        records.append(record.to_dict())
        self._save_list(tenant_id, _EXECUTIONS_KEY, records)
        return execution_id

    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        # Recover tenant from the qualified id so the right blob is read.
        tenant_id = execution_id.split(_EXEC_ID_SEP, 1)[0]
        for row in self._load_list(tenant_id, _EXECUTIONS_KEY):
            if row.get("execution_id") == execution_id:
                return ExecutionRecord.from_dict(row)
        return None

    def list_executions(
        self, tenant_id: str, workflow_name: Optional[str] = None, limit: int = 100
    ) -> List[ExecutionRecord]:
        rows = self._load_list(tenant_id, _EXECUTIONS_KEY)
        if workflow_name is not None:
            rows = [r for r in rows if r.get("workflow_name") == workflow_name]
        # Newest first.
        rows = sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)
        return [ExecutionRecord.from_dict(r) for r in rows[:limit]]

    # ==================== Agent Performance ====================

    def record_agent_performance(
        self,
        tenant_id: str,
        agent_type: str,
        duration_ms: float,
        success: bool,
        metrics: Dict[str, Any],
    ) -> str:
        now = datetime.now(timezone.utc)
        performance_id = f"{_PERF_ID_PREFIX}{uuid.uuid4().hex[:16]}"
        record = AgentPerformanceRecord(
            performance_id=performance_id,
            tenant_id=tenant_id,
            agent_type=agent_type,
            duration_ms=duration_ms,
            success=success,
            metrics=metrics,
            created_at=now,
        )
        records = self._load_list(tenant_id, _PERFORMANCE_KEY)
        records.append(record.to_dict())
        self._save_list(tenant_id, _PERFORMANCE_KEY, records)
        return performance_id

    def get_agent_stats(self, tenant_id: str, agent_type: str) -> Optional[AgentStats]:
        rows = [
            r
            for r in self._load_list(tenant_id, _PERFORMANCE_KEY)
            if r.get("agent_type") == agent_type
        ]
        if not rows:
            return None
        total = len(rows)
        avg_duration = sum(float(r.get("duration_ms", 0.0)) for r in rows) / total
        success_rate = sum(1 for r in rows if r.get("success")) / total
        last = max((r.get("created_at", "") for r in rows), default="")
        return AgentStats(
            agent_type=agent_type,
            tenant_id=tenant_id,
            total_executions=total,
            avg_duration_ms=avg_duration,
            success_rate=success_rate,
            last_execution=datetime.fromisoformat(last) if last else None,
        )

    def list_agent_performance(
        self, tenant_id: str, agent_type: Optional[str] = None, limit: int = 100
    ) -> List[AgentPerformanceRecord]:
        rows = self._load_list(tenant_id, _PERFORMANCE_KEY)
        if agent_type is not None:
            rows = [r for r in rows if r.get("agent_type") == agent_type]
        rows = sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)
        return [AgentPerformanceRecord.from_dict(r) for r in rows[:limit]]

    # ==================== Templates ====================

    def _template_key(self, template_name: str) -> str:
        return f"tmpl_{template_name}"

    def save_template(
        self, tenant_id: str, template_name: str, config: Dict[str, Any]
    ) -> str:
        now = datetime.now(timezone.utc)
        am = self._am(tenant_id)
        existing = run_coro_blocking(
            am.load_blob(_TEMPLATE_KIND, self._template_key(template_name))
        )
        created_at = now.isoformat()
        if existing:
            try:
                created_at = json.loads(existing).get("created_at", created_at)
            except (ValueError, TypeError):
                pass
        template_id = f"{tenant_id}:tmpl:{template_name}"
        template = WorkflowTemplate(
            template_id=template_id,
            tenant_id=tenant_id,
            template_name=template_name,
            config=config,
            created_at=datetime.fromisoformat(created_at),
            updated_at=now,
        )
        run_coro_blocking(
            am.save_blob(
                _TEMPLATE_KIND,
                self._template_key(template_name),
                json.dumps(template.to_dict()),
            )
        )
        # Maintain a name index so list_templates can enumerate.
        index = self._template_index(tenant_id)
        if template_name not in index:
            index.append(template_name)
            run_coro_blocking(
                am.save_blob(_TEMPLATE_KIND, _TEMPLATE_INDEX_KEY, json.dumps(index))
            )
        return template_id

    def _template_index(self, tenant_id: str) -> List[str]:
        blob = run_coro_blocking(
            self._am(tenant_id).load_blob(_TEMPLATE_KIND, _TEMPLATE_INDEX_KEY)
        )
        if not blob:
            return []
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            return []
        return [str(n) for n in data] if isinstance(data, list) else []

    def get_template(
        self, tenant_id: str, template_name: str
    ) -> Optional[WorkflowTemplate]:
        blob = run_coro_blocking(
            self._am(tenant_id).load_blob(
                _TEMPLATE_KIND, self._template_key(template_name)
            )
        )
        if not blob:
            return None
        try:
            return WorkflowTemplate.from_dict(json.loads(blob))
        except (ValueError, TypeError, KeyError):
            return None

    def list_templates(self, tenant_id: str) -> List[WorkflowTemplate]:
        templates = []
        for name in self._template_index(tenant_id):
            tmpl = self.get_template(tenant_id, name)
            if tmpl is not None:
                templates.append(tmpl)
        return templates

    def delete_template(self, tenant_id: str, template_name: str) -> bool:
        index = self._template_index(tenant_id)
        if template_name not in index:
            return False
        am = self._am(tenant_id)
        index.remove(template_name)
        run_coro_blocking(
            am.save_blob(_TEMPLATE_KIND, _TEMPLATE_INDEX_KEY, json.dumps(index))
        )
        # Tombstone the config blob (blobs are overwrite-only).
        run_coro_blocking(
            am.save_blob(_TEMPLATE_KIND, self._template_key(template_name), "")
        )
        return True

    # ==================== Health / Stats ====================

    def health_check(self) -> bool:
        return self._provider is not None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "telemetry",
            "tenants_cached": len(self._am_cache),
        }
