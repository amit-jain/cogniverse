"""Durable-execution resume for run_triggered_optimization.

A killed Argo pod re-runs the whole `--mode triggered` invocation. With
durable execution enabled (per-tenant config), each successful agent compile
is checkpointed to Phoenix so the re-run skips the agents that already
compiled. Both tests enable durable execution through the real Vespa config
store and drive the real CLI against real Phoenix:

  * ``test_resume_skips_already_compiled_agents`` stubs the DSPy compile for
    fast, deterministic wiring coverage (mutation-proven).
  * ``test_resume_skips_a_real_dspy_compile`` runs an actual BootstrapFewShot
    compile against a test-managed Ollama LM (``ensure_host_ollama``) and
    proves a resume run does NOT re-run that expensive real compile.
"""

from __future__ import annotations

import asyncio
import uuid

import pandas as pd
import pytest

from cogniverse_foundation.config.unified_config import DurableExecutionConfig
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from tests.fixtures.llm import is_test_lm_available

pytestmark = pytest.mark.integration

skip_if_no_lm = pytest.mark.skipif(
    not is_test_lm_available(),
    reason="Test LM not available for the real-compile durable resume test",
)

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

    # Enable durable execution through the REAL config store (Vespa-backed):
    # exercises set -> store -> the CLI's get_durable_execution_config read,
    # not a monkeypatched flag.
    config_manager.set_durable_execution_config(
        DurableExecutionConfig(enabled=True), tenant_id=tenant
    )
    assert config_manager.get_durable_execution_config(tenant).enabled is True

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


@skip_if_no_lm
@pytest.mark.asyncio
async def test_resume_skips_a_real_dspy_compile(
    multi_agent_trigger_dataset,
    config_manager,
    phoenix_container,
    ensure_host_ollama,
    monkeypatch,
):
    """End-to-end: a REAL DSPy compile is checkpointed and a resume run SKIPS it.

    Run 1 really compiles ``search`` against the test-managed Ollama LM
    (``ensure_host_ollama``), then crashes on ``detailed_report`` — leaving an
    active checkpoint. Run 2 resumes and must NOT re-run search's expensive
    compile: it returns the checkpointed real artifact and retries only the
    failed agent. Durable execution is enabled through the real Vespa config
    store."""
    from cogniverse_runtime import optimization_cli

    tenant = "test:unit"
    trigger = multi_agent_trigger_dataset
    workflow_id = f"opt_triggered:{tenant}:{trigger}"
    agents = ["search", "detailed_report"]

    config_manager.set_durable_execution_config(
        DurableExecutionConfig(enabled=True), tenant_id=tenant
    )

    def _boom(*a, **k):
        raise RuntimeError("disabled in test")

    monkeypatch.setattr("cogniverse_core.memory.manager.Mem0MemoryManager", _boom)
    monkeypatch.setattr("cogniverse_evaluation.quality_monitor.QualityMonitor", _boom)

    real_optimize = optimization_cli._optimize_agent

    # --- Run 1: real compile of search; detailed_report crashes -> active checkpoint. ---
    async def _run1(*, agent_name, **kw):
        if agent_name == "detailed_report":
            raise RuntimeError("simulated crash after search compiled")
        return await real_optimize(agent_name=agent_name, **kw)

    monkeypatch.setattr(optimization_cli, "_optimize_agent", _run1)

    result1 = await optimization_cli.run_triggered_optimization(
        tenant_id=tenant,
        agents=agents,
        trigger_dataset=trigger,
        config_manager=config_manager,
        phoenix_endpoint=phoenix_container["http_endpoint"],
    )
    assert result1["search"]["status"] == "success", result1["search"]
    real_served = result1["search"]["served"]
    assert real_served["served_agent"] == "search_agent", (
        "run 1 did not route the real compile through the serving gate"
    )

    latest = await _await_checkpoint(
        tenant, workflow_id, lambda c: "search" in c.completed_unit_keys()
    )
    assert latest is not None
    assert latest.completed_unit_keys() == {"search"}
    assert latest.status == "active"

    # --- Run 2 (resume): search's real compile must be skipped, not re-run. ---
    run2_calls: list[str] = []

    async def _run2(*, agent_name, **kw):
        run2_calls.append(agent_name)
        return {
            "status": "success",
            "artifact_id": f"r2_{agent_name}",
            "training_examples": 5,
        }

    monkeypatch.setattr(optimization_cli, "_optimize_agent", _run2)

    result2 = await optimization_cli.run_triggered_optimization(
        tenant_id=tenant,
        agents=agents,
        trigger_dataset=trigger,
        config_manager=config_manager,
        phoenix_endpoint=phoenix_container["http_endpoint"],
    )
    # search (real-compiled in run 1) is NOT recompiled; only the failed agent retries.
    assert "search" not in run2_calls, (
        f"resume re-ran search's real compile: {run2_calls}"
    )
    assert run2_calls == ["detailed_report"]
    # search's result comes from the checkpoint — the real run-1 serve outcome.
    assert result2["search"]["served"] == real_served


@pytest.mark.unit
@pytest.mark.ci_fast
def test_from_dict_coerces_malformed_persisted_shapes():
    """A persisted null ``phases`` / list-shaped ``completed_units`` must not
    crash resume later — ``pending_phases`` subscripts phases and
    ``completed_unit_keys`` calls ``.items()``."""
    from cogniverse_core.durable import PipelineCheckpoint

    base = {
        "checkpoint_id": "c1",
        "workflow_id": "w1",
        "tenant_id": "t1",
        "status": "active",
        "phases": None,
        "phase_index": 0,
        "completed_units": ["not", "a", "dict"],
        "metadata": "{}",
        "created_at": "2026-07-15T00:00:00+00:00",
        "resume_count": 0,
    }
    cp = PipelineCheckpoint.from_dict(base)

    assert cp.phases == []
    assert cp.completed_units == {}
    assert cp.pending_phases() == []
    assert cp.completed_unit_keys() == set()
