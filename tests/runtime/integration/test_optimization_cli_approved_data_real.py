"""Real-boundary integrity checks for optimizer persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def workflow_learning_redis_url():
    port = _free_port()
    container_name = f"cogniverse-workflow-learning-{os.getpid()}"
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Redis: {result.stderr}")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", container_name, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.25)
    else:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        pytest.fail("Redis did not become ready within 30 seconds")

    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def _workflow_state(label: str):
    from cogniverse_sdk.interfaces.workflow_store import (
        AgentPerformance,
        WorkflowExecution,
        WorkflowTemplate,
    )

    timestamp = datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc)
    return {
        "executions": [
            WorkflowExecution(
                workflow_id=f"{label}-execution",
                query=f"find {label} evidence",
                query_type=f"{label}-query",
                execution_time=1.25,
                success=True,
                agent_sequence=[f"{label}-agent"],
                task_count=1,
                parallel_efficiency=1.0,
                confidence_score=0.9,
                timestamp=timestamp,
                metadata={
                    "state": label,
                    "orchestration_pattern": "sequential",
                    "_outcome_metadata": {
                        "observed": True,
                        "required_field_semantics": {
                            "execution_time": "observed_duration_seconds",
                            "success": "observed_execution_outcome",
                            "parallel_efficiency": "observed_parallel_efficiency",
                            "confidence_score": "observed_confidence_score",
                        },
                    },
                },
            )
        ],
        "profiles": [
            AgentPerformance(
                agent_name=f"{label}-agent",
                total_executions=1,
                successful_executions=1,
                average_execution_time=1.25,
                average_confidence=0.9,
                error_rate=0.0,
                preferred_query_types=[f"{label}-query"],
                performance_trend="stable",
                last_updated=timestamp,
            )
        ],
        "patterns": {f"{label}-query": [f"find {label} evidence"]},
        "templates": [
            WorkflowTemplate(
                template_id=f"{label}-template-a",
                name=f"{label} workflow A",
                description=f"First {label} workflow",
                query_patterns=[f"find {label} evidence"],
                task_sequence=[
                    {
                        "agent": f"{label}-agent",
                        "task": "process",
                        "dependencies": [],
                    }
                ],
                expected_execution_time=1.25,
                success_rate=1.0,
                created_at=timestamp,
            ),
            WorkflowTemplate(
                template_id=f"{label}-template-b",
                name=f"{label} workflow B",
                description=f"Second {label} workflow",
                query_patterns=[f"summarize {label} evidence"],
                task_sequence=[
                    {
                        "agent": f"{label}-agent",
                        "task": "process",
                        "dependencies": [],
                    }
                ],
                expected_execution_time=1.5,
                success_rate=0.95,
                created_at=timestamp,
            ),
        ],
    }


def _signed_profile_record(item_id: str, selected_profile: str) -> dict:
    reviewed_at = "2026-08-05T00:00:00+00:00"
    record = {
        "item_id": item_id,
        "confidence": 0.9,
        "status": "approved",
        "created_at": "2026-08-04T00:00:00+00:00",
        "reviewed_at": reviewed_at,
        "query": "find exact text in presentation slides",
        "available_profiles": "video_colpali,document_colpali",
        "selected_profile": selected_profile,
        "reasoning": "Document patch retrieval preserves exact slide text.",
        "query_intent": "document_search",
        "modality": "document",
        "complexity": "complex",
        "metadata.agent_type": "profile_selection",
        "metadata.decision": {
            "reviewer": "reviewer@example.com",
            "feedback": "The selected profile matches the requested content.",
            "corrections": {},
            "timestamp": reviewed_at,
        },
        "context.optimizer": "profile",
    }
    decision_identity = {
        "item_id": item_id,
        "status": "approved",
        "decision": {
            "reviewer": "reviewer@example.com",
            "feedback": "The selected profile matches the requested content.",
            "corrections": {},
        },
    }
    record["metadata.approval_decision_sha256"] = hashlib.sha256(
        json.dumps(
            decision_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    record["metadata.approval_decision_timestamp"] = reviewed_at
    canonical_json = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    record["metadata.approval_record_json"] = canonical_json
    record["metadata.approval_record_sha256"] = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return record


@pytest.mark.asyncio
async def test_fresh_optimizer_provider_rejects_tampered_approved_record(
    phoenix_container,
):
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry
    from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        TelemetryConfig(
            otlp_endpoint=phoenix_container["grpc_endpoint"],
            provider_config={
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    tenant_id = f"optimizer:integrity-{time.time_ns()}"
    dataset_name = f"approved_synthetic_data-{tenant_id}"
    try:
        writer = manager.get_provider(
            tenant_id=tenant_id,
            project_name="approval_writer",
        )
        tampered = _signed_profile_record("tampered-profile", "document_colpali")
        tampered["selected_profile"] = "video_colpali"
        await writer.datasets.create_dataset(
            name=dataset_name,
            data=pd.DataFrame([tampered]),
        )

        fresh_provider = manager.get_provider(
            tenant_id=tenant_id,
            project_name="optimization",
        )
        with pytest.raises(
            RuntimeError,
            match=(
                "Approved dataset item content differs from canonical content: "
                f"tenant={tenant_id} dataset={dataset_name} "
                "row=0 item=tampered-profile"
            ),
        ):
            await _load_approved_synthetic_data(
                fresh_provider,
                tenant_id,
                "profile",
            )
    finally:
        TelemetryManager.reset()
        get_telemetry_registry().clear_cache()


@pytest.mark.asyncio
async def test_workflow_learning_requires_redis_before_phoenix_access():
    from cogniverse_agents.workflow.telemetry_workflow_store import (
        TelemetryWorkflowStore,
    )
    from cogniverse_runtime.optimization_cli import _save_workflow_learning_state

    store = TelemetryWorkflowStore(telemetry_provider=object(), redis_url="")
    phoenix_calls = []

    def unexpected_phoenix_access(tenant_id):
        phoenix_calls.append(tenant_id)
        raise AssertionError("Phoenix must not be touched before Redis validation")

    store._am = unexpected_phoenix_access

    with pytest.raises(RuntimeError, match="requires SystemConfig.redis_url"):
        await _save_workflow_learning_state(
            store,
            tenant_id="acme:prod",
            **_workflow_state("missing-redis"),
        )

    assert phoenix_calls == []


@pytest.mark.asyncio
async def test_failed_workflow_writer_cannot_replace_successful_replica_state(
    telemetry_manager_with_phoenix,
    workflow_learning_redis_url,
):
    from cogniverse_agents.workflow.telemetry_workflow_store import (
        TelemetryWorkflowStore,
    )
    from cogniverse_runtime.optimization_cli import _save_workflow_learning_state

    tenant_id = f"workflow-learning-{time.time_ns()}"
    provider = telemetry_manager_with_phoenix.get_provider(
        tenant_id=tenant_id,
        project_name="workflow_learning_test",
    )
    failing_store = TelemetryWorkflowStore(
        telemetry_provider=provider,
        redis_url=workflow_learning_redis_url,
    )
    successful_store = TelemetryWorkflowStore(
        telemetry_provider=provider,
        redis_url=workflow_learning_redis_url,
    )
    baseline = _workflow_state("baseline")
    failing = _workflow_state("failing")
    successful = _workflow_state("successful")

    for template in baseline["templates"]:
        await failing_store.save_template(tenant_id, template)
    await failing_store.save_learning_corpus(
        tenant_id,
        baseline["executions"],
        baseline["profiles"],
        baseline["patterns"],
    )

    failure_ready = asyncio.Event()
    allow_failure = asyncio.Event()
    successful_writer_entered = asyncio.Event()
    boundary_failure = ConnectionError("Phoenix failed on execution replacement")
    real_failing_save_executions = failing_store.save_executions
    real_successful_load_templates = successful_store.load_templates

    async def fail_execution_replacement(locked_tenant_id, executions):
        if executions == failing["executions"]:
            failure_ready.set()
            await allow_failure.wait()
            raise boundary_failure
        return await real_failing_save_executions(locked_tenant_id, executions)

    async def record_successful_writer_entry(locked_tenant_id):
        successful_writer_entered.set()
        return await real_successful_load_templates(locked_tenant_id)

    failing_store.save_executions = fail_execution_replacement
    successful_store.load_templates = record_successful_writer_entry
    failing_task = asyncio.create_task(
        _save_workflow_learning_state(
            failing_store,
            tenant_id=tenant_id,
            **failing,
        )
    )
    await asyncio.wait_for(failure_ready.wait(), timeout=45)
    successful_task = asyncio.create_task(
        _save_workflow_learning_state(
            successful_store,
            tenant_id=tenant_id,
            **successful,
        )
    )
    try:
        await asyncio.wait_for(successful_writer_entered.wait(), timeout=2)
        entered_before_failure = True
    except TimeoutError:
        entered_before_failure = False
    finally:
        allow_failure.set()

    failure_result, success_result = await asyncio.wait_for(
        asyncio.gather(failing_task, successful_task, return_exceptions=True),
        timeout=120,
    )
    assert entered_before_failure is False
    assert failure_result is boundary_failure
    assert success_result is None
    assert successful_writer_entered.is_set() is True

    reader = TelemetryWorkflowStore(
        telemetry_provider=provider,
        redis_url=workflow_learning_redis_url,
    )
    assert await reader.load_executions(tenant_id) == successful["executions"]
    assert await reader.load_agent_profiles(tenant_id) == successful["profiles"]
    assert await reader.load_query_patterns(tenant_id) == successful["patterns"]
    assert {
        template.template_id: template
        for template in await reader.load_templates(tenant_id)
    } == {template.template_id: template for template in successful["templates"]}


def _assert_workflow_intelligence_state(intelligence, state):
    assert list(intelligence.workflow_history) == state["executions"]
    assert intelligence.agent_performance == {
        profile.agent_name: profile for profile in state["profiles"]
    }
    assert dict(intelligence.query_type_patterns) == state["patterns"]
    assert intelligence.workflow_templates == {
        template.template_id: template for template in state["templates"]
    }


@pytest.mark.asyncio
async def test_reload_replaces_removed_workflow_state_without_duplicates(
    telemetry_manager_with_phoenix,
    workflow_learning_redis_url,
):
    from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
    from cogniverse_agents.workflow.telemetry_workflow_store import (
        TelemetryWorkflowStore,
    )

    tenant_id = f"workflow-reload-{time.time_ns()}"
    provider = telemetry_manager_with_phoenix.get_provider(
        tenant_id=tenant_id,
        project_name="workflow_reload_test",
    )
    store = TelemetryWorkflowStore(
        telemetry_provider=provider,
        redis_url=workflow_learning_redis_url,
    )
    first = _workflow_state("first-generation")
    second = _workflow_state("second-generation")
    second["templates"] = second["templates"][:1]

    await store.replace_learning_state(tenant_id, **first)
    intelligence = WorkflowIntelligence(tenant_id)
    intelligence._store = store
    await intelligence.load_historical_data()
    _assert_workflow_intelligence_state(intelligence, first)

    await store.replace_learning_state(tenant_id, **second)
    await intelligence.load_historical_data()
    _assert_workflow_intelligence_state(intelligence, second)

    await intelligence.load_historical_data()
    _assert_workflow_intelligence_state(intelligence, second)


@pytest.mark.asyncio
async def test_reload_waits_for_complete_generation_and_preserves_prior_on_failure(
    telemetry_manager_with_phoenix,
    workflow_learning_redis_url,
):
    from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
    from cogniverse_agents.workflow.telemetry_workflow_store import (
        TelemetryWorkflowStore,
    )

    tenant_id = f"workflow-reader-{time.time_ns()}"
    provider = telemetry_manager_with_phoenix.get_provider(
        tenant_id=tenant_id,
        project_name="workflow_reader_test",
    )
    writer = TelemetryWorkflowStore(
        telemetry_provider=provider,
        redis_url=workflow_learning_redis_url,
    )
    reader = TelemetryWorkflowStore(
        telemetry_provider=provider,
        redis_url=workflow_learning_redis_url,
    )
    previous = _workflow_state("reader-previous")
    replacement = _workflow_state("reader-replacement")
    await writer.replace_learning_state(tenant_id, **previous)

    intelligence = WorkflowIntelligence(tenant_id)
    intelligence._store = reader
    await intelligence.load_historical_data()
    _assert_workflow_intelligence_state(intelligence, previous)

    writer_paused = asyncio.Event()
    allow_writer = asyncio.Event()
    reader_entered_phoenix = asyncio.Event()
    real_writer_save_profiles = writer.save_agent_profiles
    real_reader_load_executions = reader.load_executions

    async def pause_mid_replacement(locked_tenant_id, profiles):
        writer_paused.set()
        await allow_writer.wait()
        return await real_writer_save_profiles(locked_tenant_id, profiles)

    async def record_reader_entry(locked_tenant_id):
        reader_entered_phoenix.set()
        return await real_reader_load_executions(locked_tenant_id)

    writer.save_agent_profiles = pause_mid_replacement
    reader.load_executions = record_reader_entry
    writer_task = asyncio.create_task(
        writer.replace_learning_state(tenant_id, **replacement)
    )
    await asyncio.wait_for(writer_paused.wait(), timeout=45)
    reader_task = asyncio.create_task(intelligence.load_historical_data())
    try:
        await asyncio.wait_for(reader_entered_phoenix.wait(), timeout=2)
        reader_entered_during_write = True
    except TimeoutError:
        reader_entered_during_write = False
    finally:
        allow_writer.set()

    await asyncio.wait_for(asyncio.gather(writer_task, reader_task), timeout=120)
    assert reader_entered_during_write is False
    assert reader_entered_phoenix.is_set() is True
    _assert_workflow_intelligence_state(intelligence, replacement)

    load_failure = ConnectionError("Phoenix failed while loading agent profiles")

    async def fail_profile_load(locked_tenant_id):
        raise load_failure

    reader.load_agent_profiles = fail_profile_load
    with pytest.raises(ConnectionError) as exc_info:
        await intelligence.load_historical_data()

    assert exc_info.value is load_failure
    _assert_workflow_intelligence_state(intelligence, replacement)
