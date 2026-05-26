"""Telemetry-backed implementation of the WorkflowStore abstraction.

Persists workflow intelligence through the telemetry substrate (Phoenix
datasets/blobs via ``ArtifactManager``) — the same channel the batch optimizer
and ``WorkflowIntelligence`` already share, so there is no separate backend and
no second source of truth. Executions and agent profiles ride the
demonstration-dataset channel; query patterns and templates ride blobs,
preserving the exact ``(kind, key)`` layout ``load_historical_data`` reads.

The interface is multi-tenant (``tenant_id`` per call) while ``ArtifactManager``
is tenant-scoped at construction, so one manager is cached per tenant. Telemetry
providers are themselves tenant-scoped (``TelemetryRegistry`` is tenant-scoped),
so the store resolves the correct per-tenant provider from the telemetry manager
on demand rather than binding to one at construction — which lets a single
process-wide store serve every tenant correctly. An explicit provider may be
injected (tests / single-provider contexts) to bypass that resolution.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Type, TypeVar

from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformance,
    WorkflowExecution,
    WorkflowStore,
    WorkflowTemplate,
)

# Demonstration-dataset kinds (executions, agent profiles).
_EXECUTIONS_KIND = "workflow"
_PROFILES_KIND = "agent_profiles"
# Blob coordinates — kind "workflow" matches load_historical_data's reads.
_BLOB_KIND = "workflow"
_QUERY_PATTERNS_KEY = "query_patterns"
_TEMPLATE_INDEX_KEY = "template_index"

_T = TypeVar("_T", WorkflowExecution, AgentPerformance)


def _template_key(template_id: str) -> str:
    return f"template_{template_id}"


class TelemetryWorkflowStore(WorkflowStore):
    """WorkflowStore backed by ArtifactManager (Phoenix datasets/blobs)."""

    def __init__(self, telemetry_provider: Any = None) -> None:
        # Explicit provider override (tests / single-provider contexts). When
        # None, each tenant's provider is resolved from the telemetry manager.
        self._provider = telemetry_provider
        self._am_cache: Dict[str, Any] = {}

    def _provider_for(self, tenant_id: str):
        if self._provider is not None:
            return self._provider
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        return get_telemetry_manager().get_provider(tenant_id=tenant_id)

    def _am(self, tenant_id: str):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        if tenant_id not in self._am_cache:
            self._am_cache[tenant_id] = ArtifactManager(
                self._provider_for(tenant_id), tenant_id
            )
        return self._am_cache[tenant_id]

    @staticmethod
    def _parse_demos(demos, cls: Type[_T]) -> List[_T]:
        out: List[_T] = []
        for demo in demos or []:
            try:
                out.append(cls.from_dict(json.loads(demo["input"])))
            except (ValueError, TypeError, KeyError):
                continue
        return out

    # ==================== Workflow Executions ====================

    async def save_executions(
        self, tenant_id: str, executions: List[WorkflowExecution]
    ) -> None:
        demos = [
            {
                "input": json.dumps(e.to_dict(), default=str),
                "output": json.dumps(
                    {"success": e.success, "execution_time": e.execution_time},
                    default=str,
                ),
            }
            for e in executions
        ]
        if demos:
            await self._am(tenant_id).save_demonstrations(_EXECUTIONS_KIND, demos)

    async def load_executions(self, tenant_id: str) -> List[WorkflowExecution]:
        demos = await self._am(tenant_id).load_demonstrations(_EXECUTIONS_KIND)
        return self._parse_demos(demos, WorkflowExecution)

    # ==================== Agent Performance Profiles ====================

    async def save_agent_profiles(
        self, tenant_id: str, profiles: List[AgentPerformance]
    ) -> None:
        demos = [
            {
                "input": json.dumps(p.to_dict(), default=str),
                "output": json.dumps({"agent_name": p.agent_name}, default=str),
            }
            for p in profiles
        ]
        if demos:
            await self._am(tenant_id).save_demonstrations(_PROFILES_KIND, demos)

    async def load_agent_profiles(self, tenant_id: str) -> List[AgentPerformance]:
        demos = await self._am(tenant_id).load_demonstrations(_PROFILES_KIND)
        return self._parse_demos(demos, AgentPerformance)

    # ==================== Query-Type Patterns ====================

    async def save_query_patterns(
        self, tenant_id: str, patterns: Dict[str, List[str]]
    ) -> None:
        await self._am(tenant_id).save_blob(
            _BLOB_KIND, _QUERY_PATTERNS_KEY, json.dumps(dict(patterns))
        )

    async def load_query_patterns(self, tenant_id: str) -> Dict[str, List[str]]:
        blob = await self._am(tenant_id).load_blob(_BLOB_KIND, _QUERY_PATTERNS_KEY)
        if not blob:
            return {}
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    # ==================== Workflow Templates ====================

    async def _template_index(self, tenant_id: str) -> List[str]:
        blob = await self._am(tenant_id).load_blob(_BLOB_KIND, _TEMPLATE_INDEX_KEY)
        if not blob:
            return []
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            return []
        return [str(t) for t in data] if isinstance(data, list) else []

    async def save_template(self, tenant_id: str, template: WorkflowTemplate) -> str:
        am = self._am(tenant_id)
        await am.save_blob(
            _BLOB_KIND,
            _template_key(template.template_id),
            json.dumps(template.to_dict()),
        )
        index = await self._template_index(tenant_id)
        if template.template_id not in index:
            index.append(template.template_id)
            await am.save_blob(_BLOB_KIND, _TEMPLATE_INDEX_KEY, json.dumps(index))
        return template.template_id

    async def load_templates(self, tenant_id: str) -> List[WorkflowTemplate]:
        am = self._am(tenant_id)
        templates: List[WorkflowTemplate] = []
        for tid in await self._template_index(tenant_id):
            blob = await am.load_blob(_BLOB_KIND, _template_key(tid))
            if not blob:
                continue
            try:
                templates.append(WorkflowTemplate.from_dict(json.loads(blob)))
            except (ValueError, TypeError, KeyError):
                continue
        return templates

    async def delete_template(self, tenant_id: str, template_id: str) -> bool:
        index = await self._template_index(tenant_id)
        if template_id not in index:
            return False
        am = self._am(tenant_id)
        index.remove(template_id)
        await am.save_blob(_BLOB_KIND, _TEMPLATE_INDEX_KEY, json.dumps(index))
        # Tombstone the config blob (blobs are overwrite-only).
        await am.save_blob(_BLOB_KIND, _template_key(template_id), "")
        return True

    # ==================== Utility ====================

    def health_check(self) -> bool:
        if self._provider is not None:
            return True
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        return get_telemetry_manager() is not None

    def get_stats(self) -> Dict[str, Any]:
        return {"backend": "telemetry", "tenants_cached": len(self._am_cache)}
