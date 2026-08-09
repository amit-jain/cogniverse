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

import asyncio
import hashlib
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Type, TypeVar

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformance,
    WorkflowExecution,
    WorkflowLearningState,
    WorkflowStore,
    WorkflowTemplate,
)

logger = logging.getLogger(__name__)

# Demonstration-dataset kinds (executions, agent profiles).
_EXECUTIONS_KIND = "workflow"
_PROFILES_KIND = "agent_profiles"
# Blob coordinates — kind "workflow" matches load_historical_data's reads.
_BLOB_KIND = "workflow"
_QUERY_PATTERNS_KEY = "query_patterns"
_TEMPLATE_INDEX_KEY = "template_index"

_T = TypeVar("_T", WorkflowExecution, AgentPerformance)
logger = logging.getLogger(__name__)


def _template_key(template_id: str) -> str:
    return f"template_{template_id}"


class TelemetryWorkflowStore(WorkflowStore):
    """WorkflowStore backed by ArtifactManager (Phoenix datasets/blobs)."""

    _TEMPLATE_LOCK_LEASE_MS = 30_000
    _TEMPLATE_LOCK_WAIT_SECONDS = 30.0

    def __init__(
        self,
        telemetry_provider: Any = None,
        redis_url: str | None = None,
    ) -> None:
        # Explicit provider override (tests / single-provider contexts). When
        # None, each tenant's provider is resolved from the telemetry manager.
        self._provider = telemetry_provider
        self._redis_url = redis_url
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
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug(f"Dropping malformed {cls.__name__} demonstration: {exc}")
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
        am = self._am(tenant_id)
        if demos:
            await am.save_demonstrations(_EXECUTIONS_KIND, demos)
        else:
            await am.clear_demonstrations(_EXECUTIONS_KIND)

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
        am = self._am(tenant_id)
        if demos:
            await am.save_demonstrations(_PROFILES_KIND, demos)
        else:
            await am.clear_demonstrations(_PROFILES_KIND)

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
        return await self._save_template_unlocked(tenant_id, template)

    async def _save_template_unlocked(
        self, tenant_id: str, template: WorkflowTemplate
    ) -> str:
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
        template_ids = await self._template_index(tenant_id)
        # One blob per template — the loads are independent round-trips, so
        # fetch them concurrently instead of serially.
        blobs = await asyncio.gather(
            *(am.load_blob(_BLOB_KIND, _template_key(tid)) for tid in template_ids)
        )
        for tid, blob in zip(template_ids, blobs):
            if not blob:
                continue
            try:
                templates.append(WorkflowTemplate.from_dict(json.loads(blob)))
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug(f"Dropping malformed template blob {tid!r}: {exc}")
                continue
        return templates

    async def delete_template(self, tenant_id: str, template_id: str) -> bool:
        return await self._delete_template_unlocked(tenant_id, template_id)

    async def _delete_template_unlocked(self, tenant_id: str, template_id: str) -> bool:
        index = await self._template_index(tenant_id)
        if template_id not in index:
            return False
        am = self._am(tenant_id)
        # Tombstone the blob (blobs are overwrite-only) BEFORE removing it from
        # the index: if the index write then fails, the id still resolves to an
        # empty blob that load_templates skips, so a torn delete never leaves an
        # index entry pointing at live content nor a non-empty orphan blob. A
        # retry re-tombstones (idempotent) and completes the index removal.
        await am.save_blob(_BLOB_KIND, _template_key(template_id), "")
        index.remove(template_id)
        await am.save_blob(_BLOB_KIND, _TEMPLATE_INDEX_KEY, json.dumps(index))
        return True

    def _configured_redis_url(self) -> str:
        redis_url = self._redis_url
        if redis_url is None:
            from cogniverse_foundation.config.utils import (
                get_config_manager_singleton,
            )

            redis_url = get_config_manager_singleton().get_system_config().redis_url
        if not redis_url or not redis_url.strip():
            raise RuntimeError("Workflow persistence requires SystemConfig.redis_url")
        return redis_url

    @staticmethod
    def _workflow_state_lock_key(tenant_id: str) -> str:
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        return f"cogniverse:workflow:state-write:{digest}:lock"

    @asynccontextmanager
    async def _workflow_state_lock(self, tenant_id: str):
        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError(
                f"Workflow state lock has no owning task for {tenant_id!r}"
            )
        key = self._workflow_state_lock_key(tenant_id)
        owner = secrets.token_hex(16)
        redis = aioredis.from_url(
            self._configured_redis_url(),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )
        deadline = time.monotonic() + self._TEMPLATE_LOCK_WAIT_SECONDS
        try:
            while not await redis.set(
                key,
                owner,
                nx=True,
                px=self._TEMPLATE_LOCK_LEASE_MS,
            ):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out acquiring workflow state lock for {tenant_id!r}"
                    )
                await asyncio.sleep(0.05)
        except RedisError as exc:
            await redis.aclose()
            raise RuntimeError(
                f"Failed to acquire workflow state lock for {tenant_id!r}"
            ) from exc
        except BaseException:
            await redis.aclose()
            raise

        stop = asyncio.Event()
        renewal_failure: list[BaseException] = []

        def raise_renewal_failure() -> None:
            if renewal_failure:
                raise RuntimeError(
                    f"Failed to renew workflow state lock for {tenant_id!r}"
                ) from renewal_failure[0]

        async def renew() -> None:
            interval = self._TEMPLATE_LOCK_LEASE_MS / 3000
            while True:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                    return
                except TimeoutError:
                    pass
                try:
                    renewed = await redis.eval(
                        "if redis.call('get', KEYS[1]) == ARGV[1] then "
                        "return redis.call('pexpire', KEYS[1], ARGV[2]) else "
                        "return 0 end",
                        1,
                        key,
                        owner,
                        self._TEMPLATE_LOCK_LEASE_MS,
                    )
                except RedisError as exc:
                    renewal_failure.append(exc)
                    owner_task.cancel()
                    return
                if renewed != 1:
                    renewal_failure.append(RuntimeError("lock ownership was lost"))
                    owner_task.cancel()
                    return

        renewal_task = asyncio.create_task(renew())
        body_failed = False
        try:
            try:
                yield
            except asyncio.CancelledError:
                raise_renewal_failure()
                raise
            raise_renewal_failure()
        except BaseException:
            body_failed = True
            raise
        finally:
            stop.set()
            await renewal_task
            try:
                released = await redis.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) else return 0 end",
                    1,
                    key,
                    owner,
                )
                if released != 1 and not body_failed:
                    raise RuntimeError(
                        f"Workflow state lock ownership was lost for {tenant_id!r}"
                    )
            except RedisError as exc:
                if body_failed:
                    logger.error(
                        "Failed to release workflow state lock after an error for %s",
                        tenant_id,
                    )
                else:
                    raise RuntimeError(
                        f"Failed to release workflow state lock for {tenant_id!r}"
                    ) from exc
            finally:
                await redis.aclose()

    async def save_generated_templates(
        self,
        tenant_id: str,
        templates: List[WorkflowTemplate],
    ) -> List[str]:
        """Persist one generated batch without clobbering another replica."""
        template_ids = [template.template_id for template in templates]
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("Generated workflow template IDs must be unique")

        async with self._workflow_state_lock(tenant_id):
            candidate_by_id = {template.template_id: template for template in templates}
            previous_by_id = {
                template.template_id: template
                for template in await self.load_templates(tenant_id)
                if template.template_id in candidate_by_id
            }
            written: list[WorkflowTemplate] = []
            try:
                for template in templates:
                    stored_id = await self._save_template_unlocked(tenant_id, template)
                    written.append(template)
                    if stored_id != template.template_id:
                        raise RuntimeError(
                            "Workflow store returned the wrong template identity: "
                            f"expected={template.template_id} actual={stored_id}"
                        )
            except Exception as forward_error:
                restore_errors: list[BaseException] = []
                try:
                    current_by_id = {
                        template.template_id: template
                        for template in await self.load_templates(tenant_id)
                        if template.template_id in candidate_by_id
                    }
                except Exception as restore_error:
                    restore_errors.append(restore_error)
                else:
                    for template in reversed(written):
                        template_id = template.template_id
                        if current_by_id.get(template_id) != template:
                            continue
                        try:
                            previous = previous_by_id.get(template_id)
                            if previous is None:
                                await self._delete_template_unlocked(
                                    tenant_id, template_id
                                )
                            else:
                                await self._save_template_unlocked(tenant_id, previous)
                        except Exception as restore_error:
                            restore_error.add_note(
                                "while restoring generated workflow template "
                                f"{template_id!r} for tenant {tenant_id!r}"
                            )
                            restore_errors.append(restore_error)
                if restore_errors:
                    raise ExceptionGroup(
                        "Generated workflow template save and restore failed",
                        [forward_error, *restore_errors],
                    ) from forward_error
                raise

            return template_ids

    async def replace_learning_state(
        self,
        tenant_id: str,
        executions: List[WorkflowExecution],
        profiles: List[AgentPerformance],
        patterns: Dict[str, List[str]],
        templates: List[WorkflowTemplate],
    ) -> None:
        """Replace templates and their learning corpus under one tenant lease."""
        replacement_templates = {
            template.template_id: template for template in templates
        }
        if len(replacement_templates) != len(templates):
            raise ValueError("Workflow learning template IDs must be unique")

        async with self._workflow_state_lock(tenant_id):
            previous_templates = {
                template.template_id: template
                for template in await self.load_templates(tenant_id)
            }
            previous_profiles = await self.load_agent_profiles(tenant_id)
            previous_patterns = await self.load_query_patterns(tenant_id)
            previous_executions = await self.load_executions(tenant_id)

            try:
                for template_id in previous_templates.keys() - replacement_templates:
                    deleted = await self._delete_template_unlocked(
                        tenant_id, template_id
                    )
                    if not deleted:
                        raise RuntimeError(
                            "Workflow template disappeared during replacement: "
                            f"tenant={tenant_id!r} template={template_id!r}"
                        )
                for template in replacement_templates.values():
                    stored_id = await self._save_template_unlocked(tenant_id, template)
                    if stored_id != template.template_id:
                        raise RuntimeError(
                            "Workflow store returned the wrong template identity: "
                            f"expected={template.template_id} actual={stored_id}"
                        )
                await self.save_agent_profiles(tenant_id, profiles)
                await self.save_query_patterns(tenant_id, patterns)
                await self.save_executions(tenant_id, executions)
            except Exception as forward_error:
                restore_errors: list[BaseException] = []
                try:
                    current_templates = {
                        template.template_id: template
                        for template in await self.load_templates(tenant_id)
                    }
                    for template_id in current_templates.keys() - previous_templates:
                        deleted = await self._delete_template_unlocked(
                            tenant_id, template_id
                        )
                        if not deleted:
                            raise RuntimeError(
                                "Workflow template disappeared during restoration: "
                                f"tenant={tenant_id!r} template={template_id!r}"
                            )
                    for template in previous_templates.values():
                        restored_id = await self._save_template_unlocked(
                            tenant_id, template
                        )
                        if restored_id != template.template_id:
                            raise RuntimeError(
                                "Workflow store restored the wrong template identity: "
                                f"expected={template.template_id} actual={restored_id}"
                            )
                except Exception as restore_error:
                    restore_error.add_note(
                        f"while restoring workflow templates for tenant {tenant_id!r}"
                    )
                    restore_errors.append(restore_error)

                restore_steps = [
                    ("agent profiles", self.save_agent_profiles, previous_profiles),
                    ("query patterns", self.save_query_patterns, previous_patterns),
                    ("executions", self.save_executions, previous_executions),
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
                    raise ExceptionGroup(
                        f"Workflow learning-state save and restore failed for {tenant_id!r}",
                        [forward_error, *restore_errors],
                    ) from forward_error
                raise

    async def load_learning_state(self, tenant_id: str) -> WorkflowLearningState:
        """Load every learning channel under the replacement lease."""
        async with self._workflow_state_lock(tenant_id):
            return WorkflowLearningState(
                executions=await self.load_executions(tenant_id),
                profiles=await self.load_agent_profiles(tenant_id),
                patterns=await self.load_query_patterns(tenant_id),
                templates=await self.load_templates(tenant_id),
            )

    # ==================== Utility ====================

    def health_check(self) -> bool:
        if self._provider is not None:
            return True
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        return get_telemetry_manager() is not None

    def get_stats(self) -> Dict[str, Any]:
        return {"backend": "telemetry", "tenants_cached": len(self._am_cache)}
