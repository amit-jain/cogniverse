"""
Integration test for optimization_cli.py --mode triggered.

Full flow: store trigger dataset in Phoenix → run_triggered_optimization()
→ DSPy compilation + strategy distillation → verify results.

Uses shared vespa_instance + config_manager + real_telemetry from conftest.
"""

import logging

import pandas as pd
import pytest

from tests.fixtures.llm import is_test_lm_available

logger = logging.getLogger(__name__)


skip_if_no_lm = pytest.mark.skipif(
    not is_test_lm_available(),
    reason="Test LM not available for triggered optimization",
)


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
@skip_if_no_lm
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
        self, trigger_dataset_in_phoenix, memory_manager, phoenix_container
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

    @pytest.mark.asyncio
    async def test_run_triggered_optimization_end_to_end(
        self, trigger_dataset_in_phoenix, config_manager, phoenix_container
    ):
        """Call run_triggered_optimization() with injected config_manager
        and phoenix_endpoint. Verifies the full CLI orchestration path."""
        from cogniverse_runtime.optimization_cli import run_triggered_optimization

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
        assert result["search"]["artifact_id"], "no artifact persisted"

        # Strategy distillation should have run.
        assert "strategies_distilled" in result
        assert result["strategies_distilled"] >= 0
