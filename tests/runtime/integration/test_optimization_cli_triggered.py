"""
Integration test for optimization_cli.py --mode triggered.

Full flow: store trigger dataset in Phoenix → run_triggered_optimization()
→ DSPy compilation + strategy distillation → verify results.

Uses shared vespa_instance + config_manager + real_telemetry from conftest.
"""

import logging

import pandas as pd
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def trigger_dataset_in_phoenix(real_telemetry, phoenix_container):
    """Store a trigger dataset in real Phoenix, return the dataset name."""
    import uuid

    from phoenix.client import Client

    unique = uuid.uuid4().hex[:8]
    dataset_name = f"optimization-trigger-default-{unique}"

    df = pd.DataFrame(
        [
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "man lifting weights in gym",
                "score": 0.95,
                "output": '{"results": [{"video_id": "v1"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "person running on track",
                "score": 0.88,
                "output": '{"results": [{"video_id": "v2"}, {"video_id": "v3"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "what is the dog doing",
                "score": 0.82,
                "output": '{"results": [{"video_id": "v4"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "show me the red car",
                "score": 0.90,
                "output": '{"results": [{"video_id": "v5"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "find the tall building",
                "score": 0.85,
                "output": '{"results": [{"video_id": "v6"}]}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "when did the event happen after the explosion",
                "score": 0.15,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "timeline of events before the crash",
                "score": 0.10,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "sequence during the performance",
                "score": 0.20,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "what happened after the goal",
                "score": 0.18,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "before the sunrise over mountains",
                "score": 0.12,
                "output": '{"results": []}',
            },
        ]
    )

    sync_client = Client(base_url=phoenix_container["http_endpoint"])
    sync_client.datasets.create_dataset(
        name=dataset_name,
        dataframe=df,
        input_keys=["agent", "category", "query"],
        output_keys=["score", "output"],
    )

    yield dataset_name


@pytest.mark.integration
class TestTriggeredOptimization:
    """Test run_triggered_optimization() end-to-end."""

    @pytest.mark.asyncio
    async def test_trigger_dataset_readable_from_phoenix(
        self, trigger_dataset_in_phoenix, phoenix_container
    ):
        """Verify trigger dataset stored in Phoenix is readable — same path
        optimization_cli uses to load training data."""
        from phoenix.client import Client

        sync_client = Client(base_url=phoenix_container["http_endpoint"])
        dataset = sync_client.datasets.get_dataset(dataset=trigger_dataset_in_phoenix)
        df = dataset.to_dataframe()

        # Flatten Phoenix nested format if needed
        if "input" in df.columns:
            flat = []
            for _, row in df.iterrows():
                inp = row.get("input", {}) or {}
                out = row.get("output", {}) or {}
                flat.append({**inp, **out})
            df = pd.DataFrame(flat)

        assert len(df) == 10
        assert "agent" in df.columns
        assert "category" in df.columns
        assert "query" in df.columns

        low = df[df["category"] == "low_scoring"]
        high = df[df["category"] == "high_scoring"]
        assert len(low) == 5
        assert len(high) == 5

    @pytest.mark.asyncio
    async def test_strategy_distillation_from_trigger_dataset(
        self,
        ensure_host_ollama,
        trigger_dataset_in_phoenix,
        memory_manager,
        phoenix_container,
    ):
        """Load trigger dataset from Phoenix and distill strategies into real Vespa memory.

        Uses the module-scoped memory_manager fixture (real Mem0MemoryManager backed
        by the shared Vespa Docker instance) so add_memory calls hit real storage.
        """
        from phoenix.client import Client

        from cogniverse_agents.optimizer.strategy_learner import StrategyLearner

        sync_client = Client(base_url=phoenix_container["http_endpoint"])
        dataset = sync_client.datasets.get_dataset(dataset=trigger_dataset_in_phoenix)
        trigger_df = dataset.to_dataframe()

        # Flatten Phoenix format
        if "input" in trigger_df.columns:
            flat = []
            for _, row in trigger_df.iterrows():
                inp = row.get("input", {}) or {}
                out = row.get("output", {}) or {}
                flat.append({**inp, **out})
            trigger_df = pd.DataFrame(flat)

        learner = StrategyLearner(
            memory_manager=memory_manager,
            tenant_id="test:unit",
        )
        strategies = await learner.learn_from_trigger_dataset(trigger_df)

        assert len(strategies) >= 1, (
            f"Should distill strategies from Phoenix trigger dataset, got {len(strategies)}"
        )

    @pytest.mark.requires_teacher_model
    @pytest.mark.asyncio
    async def test_run_triggered_optimization_end_to_end(
        self,
        teacher_role_on_test_lm,
        trigger_dataset_in_phoenix,
        config_manager,
        phoenix_container,
        monkeypatch,
    ):
        """Call run_triggered_optimization() with injected config_manager
        and phoenix_endpoint. Verifies the full CLI orchestration path.

        The held-out scores are pinned to a winning pair so the run is
        deterministic against the tiny test LM — the scoring math itself is
        unit-tested with exact values; this test proves the real compile →
        gate → versioned serve → overlay pipeline against real Phoenix."""
        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import run_triggered_optimization

        # The served module is scored before the compile, the candidate after.
        pinned = iter([0.20, 0.90])
        monkeypatch.setattr(
            optimization_cli, "_probe_set_score", lambda *a, **k: next(pinned)
        )

        result = await run_triggered_optimization(
            tenant_id="test:unit",
            agents=["search"],
            trigger_dataset=trigger_dataset_in_phoenix,
            config_manager=config_manager,
            phoenix_endpoint=phoenix_container["http_endpoint"],
        )

        # Per-agent result is keyed by agent name (NOT a top-level "status").
        # The search agent must compile and persist its module successfully —
        # asserting the real per-agent outcome, not a top-level key that never
        # exists (the old `result.get("status") != "failed"` was a tautology
        # that masked ArtifactManager construction + store_artifact failures).
        assert "search" in result, f"no per-agent result for search: {result}"
        assert result["search"]["status"] == "success", (
            f"search optimization failed: {result['search']}"
        )
        # _optimize_agent serves via _serve_compiled_prompts through the
        # promote_if_better gate; a winning candidate lands versioned →
        # canary → ACTIVE, so the per-agent result carries "served".
        served = result["search"].get("served")
        assert served, (
            f"no compiled prompts served to the dispatch overlay: {result['search']}"
        )
        assert served["served_agent"] == "search_agent"
        assert served["promoted"] is True
        assert served["active"] is True
        assert served["baseline_score"] == 0.20
        assert served["candidate_score"] == 0.90
        assert isinstance(served["version"], int) and served["version"] >= 1
        assert result["search"]["holdout_examples"] >= 1

        # The published artefact is the WHOLE compiled module, and serving
        # reconstructs exactly the object that was scored: read it back
        # through the request path and load it into a stock served module.
        import json

        import dspy

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_core.agents.base import (
            COMPILED_MODULE_PROMPT_KEY,
            load_compiled_module_state,
        )
        from cogniverse_runtime.optimization_cli import (
            _build_phoenix_provider_for_cli,
        )

        monkeypatch.setenv(
            "TELEMETRY_HTTP_ENDPOINT", phoenix_container["http_endpoint"]
        )
        monkeypatch.setenv(
            "TELEMETRY_OTLP_ENDPOINT", phoenix_container["otlp_endpoint"]
        )
        reader = ArtifactManager(
            _build_phoenix_provider_for_cli("test:unit"), "test:unit"
        )
        overlay = await reader.load_for_request(
            "search_agent", request_seed="triggered-roundtrip"
        )
        assert overlay["served_from"] == "active"
        assert overlay["version"] == served["version"]
        assert list(overlay["prompts"]) == [COMPILED_MODULE_PROMPT_KEY]
        payload = overlay["prompts"][COMPILED_MODULE_PROMPT_KEY]
        published = json.loads(payload)
        assert published["module"] == (
            "cogniverse_agents.search_agent.SearchOptimizationModule"
        )
        assert published["dspy_version"] == dspy.__version__
        module = SearchOptimizationModule()
        load_compiled_module_state(module, payload)
        assert json.loads(json.dumps(module.dump_state())) == published["state"]

        # Strategy distillation should have run.
        assert "strategies_distilled" in result
        assert result["strategies_distilled"] >= 0
        # Both pinned scores were drawn: one pre-score, one candidate score.
        assert list(pinned) == []


# A label no module output can share a token with: every held-out positive
# scores 0.0 and every known-bad probe (whose recorded output is empty) scores
# 1.0, so the served module scores exactly 5/6 on the probe set whatever the
# LM writes.
UNREPRODUCIBLE_LABEL = "qzxv wplk jrtm"
SERVED_PROBE_SCORE = 5 / 6


@pytest.fixture
def gate_trigger_dataset(real_telemetry, phoenix_container):
    """A trigger dataset whose served module scores exactly 5/6."""
    import json
    import uuid

    from phoenix.client import Client

    dataset_name = f"optimization-trigger-gate-{uuid.uuid4().hex[:8]}"
    high = [
        "man lifting weights in gym",
        "person running on track",
        "what is the dog doing",
        "show me the red car",
        "find the tall building",
    ]
    low = [
        "when did the event happen after the explosion",
        "timeline of events before the crash",
        "sequence during the performance",
        "what happened after the goal",
        "before the sunrise over mountains",
    ]
    rows = [
        {
            "agent": "search",
            "category": "high_scoring",
            "query": query,
            "score": 0.9,
            "output": json.dumps(
                {"results": [], "enhanced_query": UNREPRODUCIBLE_LABEL}
            ),
        }
        for query in high
    ] + [
        {
            "agent": "search",
            "category": "low_scoring",
            "query": query,
            "score": 0.1,
            "output": json.dumps({"results": []}),
        }
        for query in low
    ]
    Client(base_url=phoenix_container["http_endpoint"]).datasets.create_dataset(
        name=dataset_name,
        dataframe=pd.DataFrame(rows),
        input_keys=["agent", "category", "query"],
        output_keys=["score", "output"],
    )
    return dataset_name


@pytest.fixture
def publish_gate(teacher_role_on_test_lm, tmp_path, monkeypatch):
    """Publish ``optimization_triggers`` overrides in the config the run reads."""
    import json

    def publish(**triggers):
        config = json.loads(teacher_role_on_test_lm.read_text())
        config.setdefault("automation_rules", {}).setdefault(
            "optimization_triggers", {}
        ).update(triggers)
        path = tmp_path / f"config-gate-{len(triggers)}.json"
        path.write_text(json.dumps(config))
        monkeypatch.setenv("COGNIVERSE_CONFIG", str(path))

    return publish


@pytest.fixture
def optimizer_calls(monkeypatch):
    """Count what an optimization step spends: probe-set scorings, served
    module calls, LM requests and compiles, and which of the LM requests fell
    inside a probe-set scoring."""
    from dspy.teleprompt import BootstrapFewShot

    from cogniverse_agents.search_agent import SearchOptimizationModule
    from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM
    from cogniverse_runtime import optimization_cli

    calls = {
        "scores": [],
        "module_calls": 0,
        "lm_requests": [],
        "lm_requests_while_scoring": 0,
        "compiles": 0,
    }
    scoring = {"active": False}

    real_score = optimization_cli._probe_set_score

    def score(module, holdout, negatives, agent_name):
        scoring["active"] = True
        try:
            value = real_score(module, holdout, negatives, agent_name)
        finally:
            scoring["active"] = False
        calls["scores"].append(
            {"probes": len(holdout) + len(negatives), "score": value}
        )
        return value

    real_module_call = SearchOptimizationModule.__call__

    def module_call(self, *args, **kwargs):
        if scoring["active"]:
            calls["module_calls"] += 1
        return real_module_call(self, *args, **kwargs)

    real_forward = BodyBoundedLM.forward

    def forward(self, *args, **kwargs):
        calls["lm_requests"].append(type(self).__name__)
        if scoring["active"]:
            calls["lm_requests_while_scoring"] += 1
        return real_forward(self, *args, **kwargs)

    real_compile = BootstrapFewShot.compile

    def compile_(self, *args, **kwargs):
        calls["compiles"] += 1
        return real_compile(self, *args, **kwargs)

    monkeypatch.setattr(optimization_cli, "_probe_set_score", score)
    monkeypatch.setattr(SearchOptimizationModule, "__call__", module_call)
    monkeypatch.setattr(BodyBoundedLM, "forward", forward)
    monkeypatch.setattr(BootstrapFewShot, "compile", compile_)
    return calls


def _gate_tenant() -> str:
    import uuid

    return f"gate{uuid.uuid4().hex[:8]}:unit"


async def _search_experiments(real_telemetry, tenant_id):
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    manager = ArtifactManager(
        real_telemetry.get_provider(tenant_id=tenant_id), tenant_id
    )
    return manager, await manager.load_experiments("search_agent")


@pytest.mark.integration
@pytest.mark.requires_teacher_model
class TestServedScoreCeiling:
    """The triggered step scores what is served before it compiles."""

    @pytest.mark.asyncio
    async def test_a_served_score_at_the_ceiling_spends_no_compile(
        self,
        publish_gate,
        gate_trigger_dataset,
        optimizer_calls,
        config_manager,
        real_telemetry,
        phoenix_container,
    ):
        from cogniverse_runtime.optimization_cli import run_triggered_optimization

        publish_gate(optimization_skip_above_score=0.8)
        tenant_id = _gate_tenant()

        result = await run_triggered_optimization(
            tenant_id=tenant_id,
            agents=["search"],
            trigger_dataset=gate_trigger_dataset,
            config_manager=config_manager,
            phoenix_endpoint=phoenix_container["http_endpoint"],
        )

        assert result["search"] == {
            "status": "skipped",
            "reason": "served_score_at_ceiling",
            "served_score": SERVED_PROBE_SCORE,
            "skip_above_score": 0.8,
            "training_examples": 4,
            "holdout_examples": 1,
            "negative_probes": 5,
        }
        # One scoring of the served module over the six probes, and nothing
        # else: no compile, no teacher request, no LM request outside it.
        assert optimizer_calls["scores"] == [{"probes": 6, "score": SERVED_PROBE_SCORE}]
        assert optimizer_calls["module_calls"] == 6
        assert optimizer_calls["compiles"] == 0
        assert set(optimizer_calls["lm_requests"]) == {"BodyBoundedLM"}
        assert (
            len(optimizer_calls["lm_requests"])
            - optimizer_calls["lm_requests_while_scoring"]
            == 0
        )
        # Nothing was offered to the gate or written for the agent.
        manager, experiments = await _search_experiments(real_telemetry, tenant_id)
        assert experiments == []
        assert await manager.load_prompts("search_agent") is None

    @pytest.mark.asyncio
    async def test_below_the_ceiling_the_pre_score_is_the_gates_baseline(
        self,
        publish_gate,
        gate_trigger_dataset,
        optimizer_calls,
        config_manager,
        real_telemetry,
        phoenix_container,
    ):
        from cogniverse_agents.optimizer.artifact_manager import PromotionGate
        from cogniverse_runtime.optimization_cli import run_triggered_optimization

        # Just above the served score: the run compiles.
        publish_gate(optimization_skip_above_score=0.84)
        tenant_id = _gate_tenant()

        result = await run_triggered_optimization(
            tenant_id=tenant_id,
            agents=["search"],
            trigger_dataset=gate_trigger_dataset,
            config_manager=config_manager,
            phoenix_endpoint=phoenix_container["http_endpoint"],
        )

        assert optimizer_calls["compiles"] == 1
        # The served module was scored once, before the compile; the second
        # scoring is the candidate's.
        assert [entry["probes"] for entry in optimizer_calls["scores"]] == [6, 6]
        assert optimizer_calls["scores"][0]["score"] == SERVED_PROBE_SCORE
        assert optimizer_calls["module_calls"] == 12
        candidate_score = optimizer_calls["scores"][1]["score"]
        gate = PromotionGate(
            improvement_threshold=0.05, gap_share=0.5, skip_above_score=0.84
        )
        promoted = gate.promotes(candidate_score, SERVED_PROBE_SCORE, 1.0)

        served = result["search"]["served"]
        assert (
            result["search"]["status"],
            served["served_agent"],
            served["baseline_score"],
            served["candidate_score"],
            served["promoted"],
            served["active"],
        ) == (
            "success",
            "search_agent",
            SERVED_PROBE_SCORE,
            candidate_score,
            promoted,
            promoted,
        )
        # The gate's decision is in the ledger with the gain it required.
        _, experiments = await _search_experiments(real_telemetry, tenant_id)
        assert [
            (
                entry.baseline_score,
                entry.candidate_score,
                entry.promoted,
                entry.extra_metrics["min_improvement"],
                entry.extra_metrics["improvement_threshold"],
                entry.extra_metrics["gap_share"],
                entry.extra_metrics["max_score"],
            )
            for entry in experiments
        ] == [(SERVED_PROBE_SCORE, candidate_score, promoted, 0.05, 0.05, 0.5, 1.0)]
