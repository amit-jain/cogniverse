"""Durable-execution resume for run_triggered_optimization (real Phoenix).

A killed Argo pod re-runs the whole `--mode triggered` invocation. With
durable execution enabled, each successful agent compile is checkpointed to
Phoenix, so the re-run skips the agents that already compiled. This drives
the real CLI path against a real Phoenix Docker: run once where the last
agent crashes (leaving the earlier two checkpointed), then re-run and assert
that ONLY the un-compiled agent is recompiled — the expensive DSPy
compile is stubbed so the test exercises resume wiring, not compile quality.
"""

from __future__ import annotations

import asyncio
import uuid

import pandas as pd
import pytest

from cogniverse_foundation.config.unified_config import DurableExecutionConfig
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

pytestmark = pytest.mark.integration

_AGENTS = ["search", "detailed_report", "summarizer"]


@pytest.fixture
def multi_agent_trigger_dataset(telemetry_manager_with_phoenix, phoenix_container):
    """A Phoenix trigger dataset carrying rows for three agents."""
    from phoenix.client import Client

    dataset_name = f"optimization-trigger-default-{uuid.uuid4().hex[:8]}"
    rows = []
    for agent in _AGENTS:
        for i in range(5):
            rows.append(
                {
                    "agent": agent,
                    "category": "high_scoring",
                    "query": f"{agent} good {i}",
                    "score": 0.9,
                    "output": "{}",
                }
            )
        for i in range(5):
            rows.append(
                {
                    "agent": agent,
                    "category": "low_scoring",
                    "query": f"{agent} bad {i}",
                    "score": 0.1,
                    "output": "{}",
                }
            )
    Client(base_url=phoenix_container["http_endpoint"]).datasets.create_dataset(
        name=dataset_name,
        dataframe=pd.DataFrame(rows),
        input_keys=["agent", "category", "query"],
        output_keys=["score", "output"],
    )
    return dataset_name


async def _await_checkpoint(tenant, workflow_id, predicate, tries=15, delay=1.0):
    from cogniverse_core.durable import PipelineCheckpointStorage

    tm = get_telemetry_manager()
    pc = tm.config.provider_config or {}
    storage = PipelineCheckpointStorage(
        grpc_endpoint=pc.get("grpc_endpoint"),
        http_endpoint=pc.get("http_endpoint"),
        tenant_id=tenant,
        telemetry_manager=tm,
    )
    latest = None
    for _ in range(tries):
        latest = await storage.get_latest_checkpoint(workflow_id)
        if latest is not None and predicate(latest):
            return latest
        await asyncio.sleep(delay)
    return latest


@pytest.mark.asyncio
async def test_resume_skips_already_compiled_agents(
    multi_agent_trigger_dataset, config_manager, phoenix_container, monkeypatch
):
    from cogniverse_runtime import optimization_cli

    tenant = "test:unit"
    trigger = multi_agent_trigger_dataset
    workflow_id = f"opt_triggered:{tenant}:{trigger}"

    # Enable durable execution (the store round-trip is unit-tested separately).
    monkeypatch.setattr(
        config_manager,
        "get_durable_execution_config",
        lambda tenant_id=None, **k: DurableExecutionConfig(
            tenant_id=tenant, enabled=True
        ),
        raising=False,
    )

    # Neutralize the best-effort post-steps (strategy distillation + golden
    # eval) so the test stays fast and focused on the agent-loop resume.
    def _boom(*a, **k):
        raise RuntimeError("disabled in test")

    monkeypatch.setattr("cogniverse_core.memory.manager.Mem0MemoryManager", _boom)
    monkeypatch.setattr("cogniverse_evaluation.quality_monitor.QualityMonitor", _boom)

    async def _kwargs_call(recorder, fail_agent, prefix, *, agent_name, **_kw):
        recorder.append(agent_name)
        if agent_name == fail_agent:
            raise RuntimeError("simulated crash mid-compile")
        return {
            "status": "success",
            "artifact_id": f"{prefix}_{agent_name}",
            "training_examples": 5,
        }

    # --- Run 1: summarizer crashes; search + detailed_report checkpoint. ---
    run1_calls: list[str] = []

    async def fake_run1(**kw):
        return await _kwargs_call(run1_calls, "summarizer", "artifact", **kw)

    monkeypatch.setattr(optimization_cli, "_optimize_agent", fake_run1)

    result1 = await optimization_cli.run_triggered_optimization(
        tenant_id=tenant,
        agents=_AGENTS,
        trigger_dataset=trigger,
        config_manager=config_manager,
        phoenix_endpoint=phoenix_container["http_endpoint"],
    )
    assert result1["search"]["status"] == "success"
    assert result1["detailed_report"]["status"] == "success"
    assert result1["summarizer"]["status"] == "failed"
    assert run1_calls == ["search", "detailed_report", "summarizer"]

    # The checkpoint records the two successful agents (not the crashed one).
    latest = await _await_checkpoint(
        tenant,
        workflow_id,
        lambda c: {"search", "detailed_report"} <= c.completed_unit_keys(),
    )
    assert latest is not None
    assert latest.completed_unit_keys() == {"search", "detailed_report"}
    assert latest.status == "active"

    # --- Run 2 (resume): only summarizer must recompile. ---
    run2_calls: list[str] = []

    async def fake_run2(**kw):
        return await _kwargs_call(run2_calls, None, "artifact2", **kw)

    monkeypatch.setattr(optimization_cli, "_optimize_agent", fake_run2)

    result2 = await optimization_cli.run_triggered_optimization(
        tenant_id=tenant,
        agents=_AGENTS,
        trigger_dataset=trigger,
        config_manager=config_manager,
        phoenix_endpoint=phoenix_container["http_endpoint"],
    )
    # The crux: the two already-compiled agents are NOT recompiled.
    assert run2_calls == ["summarizer"], (
        f"expected only summarizer recompiled on resume, got {run2_calls}"
    )
    # search/detailed_report results come from the run-1 checkpoint; summarizer
    # is freshly compiled in run 2.
    assert result2["search"]["artifact_id"] == "artifact_search"
    assert result2["detailed_report"]["artifact_id"] == "artifact_detailed_report"
    assert result2["summarizer"]["artifact_id"] == "artifact2_summarizer"

    # After a clean run the workflow is marked completed.
    done = await _await_checkpoint(
        tenant, workflow_id, lambda c: c.status == "completed"
    )
    assert done is not None and done.status == "completed"
