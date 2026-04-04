"""Unit tests for QualityMonitor — dual evaluation + threshold + trigger packaging."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_evaluation.quality_monitor import (
    AgentEvalResult,
    AgentType,
    GoldenEvalResult,
    LiveEvalResult,
    OptimizationTrigger,
    QualityMonitor,
    Verdict,
)


@pytest.fixture
def golden_dataset(tmp_path):
    """Create a minimal golden dataset for testing."""
    queries = [
        {
            "query": "man lifting barbell",
            "expected_videos": ["v_-HpCLXdtcas"],
            "ground_truth": "Man lifting a barbell",
            "query_type": "answer_phrase",
            "source": "test",
        },
        {
            "query": "dog playing in park",
            "expected_videos": ["v_dog123"],
            "ground_truth": "Dog playing fetch in a park",
            "query_type": "question",
            "source": "test",
        },
    ]
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(queries))
    return str(path)


@pytest.fixture
def monitor(golden_dataset):
    """Create a QualityMonitor instance for testing."""
    return QualityMonitor(
        tenant_id="test_tenant",
        runtime_url="http://localhost:28000",
        phoenix_http_endpoint="http://localhost:6006",
        llm_base_url="http://localhost:11434",
        llm_model="qwen3:4b",
        golden_dataset_path=golden_dataset,
        argo_api_url="http://localhost:2746",
        argo_namespace="test-ns",
    )


class TestGoldenDatasetLoading:
    def test_load_golden_queries(self, monitor):
        queries = monitor._load_golden_queries()
        assert len(queries) == 2
        assert queries[0]["query"] == "man lifting barbell"
        assert queries[0]["expected_videos"] == ["v_-HpCLXdtcas"]

    def test_load_golden_queries_cached(self, monitor):
        first = monitor._load_golden_queries()
        second = monitor._load_golden_queries()
        assert first is second

    def test_load_golden_queries_file_not_found(self):
        m = QualityMonitor(
            tenant_id="t",
            runtime_url="http://x",
            phoenix_http_endpoint="http://x",
            llm_base_url="http://x",
            llm_model="m",
            golden_dataset_path="/nonexistent/path.json",
        )
        with pytest.raises(FileNotFoundError):
            m._load_golden_queries()


class TestThresholdChecking:
    def test_golden_mrr_drop_triggers_optimize(self, monitor):
        golden = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.4,
            mean_ndcg=0.5,
            mean_precision_at_5=0.3,
            query_count=10,
        )
        # Simulate baseline of 0.6 — drop is 33% > 10% threshold
        with patch.object(
            type(monitor),
            "_last_golden_baseline_mrr",
            new_callable=lambda: property(lambda self: 0.6),
        ):
            verdicts = monitor.check_thresholds(golden, None)
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    def test_golden_no_drop_skips(self, monitor):
        golden = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.58,
            mean_ndcg=0.6,
            mean_precision_at_5=0.5,
            query_count=10,
        )
        # Baseline 0.6, drop is 3% < 10% threshold
        with patch.object(
            type(monitor),
            "_last_golden_baseline_mrr",
            new_callable=lambda: property(lambda self: 0.6),
        ):
            verdicts = monitor.check_thresholds(golden, None)
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP

    def test_live_score_below_floor_triggers(self, monitor):
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.SUMMARY: AgentEvalResult(
                    agent=AgentType.SUMMARY,
                    score=0.3,
                    baseline_score=0.7,
                    degradation_pct=0.57,
                    sample_count=15,
                ),
            },
        )
        verdicts = monitor.check_thresholds(None, live)
        assert verdicts[AgentType.SUMMARY] == Verdict.OPTIMIZE

    def test_insufficient_samples_skips(self, monitor):
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.REPORT: AgentEvalResult(
                    agent=AgentType.REPORT,
                    score=0.1,
                    baseline_score=0.8,
                    degradation_pct=0.87,
                    sample_count=3,  # Below min_samples_for_verdict=10
                ),
            },
        )
        verdicts = monitor.check_thresholds(None, live)
        assert AgentType.REPORT not in verdicts

    def test_multiple_agents_independent_verdicts(self, monitor):
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.8,
                    baseline_score=0.8,
                    degradation_pct=0.0,
                    sample_count=20,
                ),
                AgentType.SUMMARY: AgentEvalResult(
                    agent=AgentType.SUMMARY,
                    score=0.3,
                    baseline_score=0.7,
                    degradation_pct=0.57,
                    sample_count=20,
                ),
            },
        )
        verdicts = monitor.check_thresholds(None, live)
        assert verdicts.get(AgentType.SEARCH) == Verdict.SKIP
        assert verdicts[AgentType.SUMMARY] == Verdict.OPTIMIZE


class TestTriggerBuilding:
    def test_build_trigger_packages_low_and_high_scoring(self, monitor):
        golden = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.3,
            mean_ndcg=0.4,
            mean_precision_at_5=0.2,
            query_count=10,
            low_scoring_queries=[
                {"query": "bad query", "mrr": 0.0}
            ],
            high_scoring_queries=[
                {"query": "good query", "mrr": 1.0}
            ],
        )
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.SUMMARY: AgentEvalResult(
                    agent=AgentType.SUMMARY,
                    score=0.3,
                    baseline_score=0.7,
                    degradation_pct=0.57,
                    sample_count=20,
                    low_scoring_examples=[
                        {"query": "bad summary", "score": 0.2}
                    ],
                    high_scoring_examples=[],
                ),
            },
        )

        trigger = monitor._build_trigger(
            [AgentType.SEARCH, AgentType.SUMMARY], golden, live
        )

        assert AgentType.SEARCH in trigger.agents_to_optimize
        assert AgentType.SUMMARY in trigger.agents_to_optimize
        assert len(trigger.low_scoring_examples[AgentType.SEARCH]) == 1
        assert len(trigger.high_scoring_examples[AgentType.SEARCH]) == 1
        assert len(trigger.low_scoring_examples[AgentType.SUMMARY]) == 1


class TestGoldenSetGrowth:
    @pytest.mark.asyncio
    async def test_grow_golden_set_adds_new_queries(self, monitor, golden_dataset):
        original = monitor._load_golden_queries()
        assert len(original) == 2

        new_queries = [
            {
                "query": "cat sitting on couch",
                "expected_videos": ["v_cat456"],
                "ground_truth": "Cat on couch",
                "query_type": "live_traffic",
                "source": "quality_monitor",
            },
        ]

        await monitor.grow_golden_set(new_queries)

        # Re-read from disk
        with open(golden_dataset) as f:
            updated = json.load(f)
        assert len(updated) == 3
        assert updated[2]["query"] == "cat sitting on couch"

    @pytest.mark.asyncio
    async def test_grow_golden_set_deduplicates(self, monitor, golden_dataset):
        # Try adding a query that already exists
        new_queries = [
            {
                "query": "man lifting barbell",  # Already exists
                "expected_videos": ["v_-HpCLXdtcas"],
            },
        ]

        await monitor.grow_golden_set(new_queries)

        with open(golden_dataset) as f:
            updated = json.load(f)
        assert len(updated) == 2  # No duplicates added


class TestLLMJudgePrompts:
    def test_search_judge_prompt_includes_query_and_results(self, monitor):
        prompt = monitor._build_search_judge_prompt(
            "find videos of dogs",
            [
                {"video_id": "v_123", "score": 0.9},
                {"video_id": "v_456", "score": 0.7},
            ],
        )
        assert "find videos of dogs" in prompt
        assert "v_123" in prompt
        assert "Score: X/10" in prompt

    def test_summary_judge_prompt(self, monitor):
        prompt = monitor._build_summary_judge_prompt(
            "summarize machine learning", "ML is a field of AI..."
        )
        assert "summarize machine learning" in prompt
        assert "ML is a field of AI" in prompt

    def test_report_judge_prompt_truncates(self, monitor):
        long_report = "x" * 5000
        prompt = monitor._build_report_judge_prompt("query", long_report)
        assert len(prompt) < 5000  # Truncated to 2000 chars


class TestArgoSubmission:
    @pytest.mark.asyncio
    async def test_submit_optimization_creates_workflow(self, monitor):
        trigger = OptimizationTrigger(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agents_to_optimize=[AgentType.SEARCH, AgentType.SUMMARY],
            golden_eval=None,
            live_eval=None,
            low_scoring_examples={},
            high_scoring_examples={},
            misrouted_queries=[],
        )

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "metadata": {"name": "quality-triggered-optimization-test"}
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        await monitor.submit_optimization(trigger)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "workflows/test-ns" in call_args[0][0]
        workflow = call_args[1]["json"]["workflow"]
        params = {
            p["name"]: p["value"]
            for p in workflow["spec"]["arguments"]["parameters"]
        }
        assert params["agents"] == "search,summary"
        assert params["tenant-id"] == "test_tenant"


class TestGoldenEvaluation:
    @pytest.mark.asyncio
    async def test_evaluate_golden_set_scores_queries(self, monitor):
        """Test golden eval with mocked HTTP responses."""
        import httpx

        def mock_handler(request):
            body = json.loads(request.content)
            query = body.get("query", "")
            if "barbell" in query:
                return httpx.Response(200, json={
                    "results": [{"source_id": "v_-HpCLXdtcas", "score": 0.9}]
                })
            return httpx.Response(200, json={"results": []})

        monitor._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            base_url="http://testserver",
        )

        with patch.object(monitor, "_store_golden_eval_result", new_callable=AsyncMock):
            result = await monitor.evaluate_golden_set()

        assert result.query_count == 2
        assert result.mean_mrr > 0

        scores = {s["query"]: s for s in result.per_query_scores}
        assert scores["man lifting barbell"]["mrr"] == 1.0
        assert scores["dog playing in park"]["mrr"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_golden_set_handles_http_errors(self, monitor):
        """500 responses are skipped, not crashes."""
        import httpx

        monitor._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(500, text="error")
            ),
            base_url="http://testserver",
        )

        with pytest.raises(RuntimeError, match="No golden queries evaluated"):
            with patch.object(monitor, "_store_golden_eval_result", new_callable=AsyncMock):
                await monitor.evaluate_golden_set()


class TestStoreOperations:
    @pytest.mark.asyncio
    async def test_store_golden_eval_result(self, monitor):
        """Store golden eval calls dataset store."""
        mock_store = AsyncMock()
        mock_store.create_dataset = AsyncMock(return_value="ds_123")
        monitor._dataset_store = mock_store

        result = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.75,
            mean_ndcg=0.70,
            mean_precision_at_5=0.50,
            query_count=10,
        )

        await monitor._store_golden_eval_result(result)
        mock_store.create_dataset.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_live_eval_result(self, monitor):
        """Store live eval calls dataset store with agent data."""
        mock_store = AsyncMock()
        mock_store.create_dataset = AsyncMock(return_value="ds_456")
        monitor._dataset_store = mock_store

        result = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.8,
                    baseline_score=0.85,
                    degradation_pct=0.06,
                    sample_count=20,
                ),
            },
        )

        await monitor._store_live_eval_result(result)
        mock_store.create_dataset.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_trigger_dataset(self, monitor):
        """Store trigger dataset with training examples."""
        mock_store = AsyncMock()
        mock_store.create_dataset = AsyncMock(return_value="ds_789")
        monitor._dataset_store = mock_store

        trigger = OptimizationTrigger(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agents_to_optimize=[AgentType.SEARCH],
            golden_eval=None,
            live_eval=None,
            low_scoring_examples={
                AgentType.SEARCH: [{"query": "bad", "score": 0.1, "output": {}}],
            },
            high_scoring_examples={
                AgentType.SEARCH: [{"query": "good", "score": 0.9, "output": {}}],
            },
            misrouted_queries=[],
        )

        await monitor._store_trigger_dataset(trigger)
        mock_store.create_dataset.assert_called_once()


class TestUpdateBaseline:
    @pytest.mark.asyncio
    async def test_update_baseline_stores_golden(self, monitor):
        with patch.object(monitor, "_store_golden_eval_result", new_callable=AsyncMock) as mock:
            result = GoldenEvalResult(
                timestamp=datetime.utcnow(),
                tenant_id="test_tenant",
                mean_mrr=0.8,
                mean_ndcg=0.75,
                mean_precision_at_5=0.6,
                query_count=10,
            )
            await monitor.update_baseline(golden_result=result)
            mock.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_grow_golden_set_empty_list(self, monitor):
        await monitor.grow_golden_set([])
        # Should not modify file


class TestClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up(self, monitor):
        mock_client = AsyncMock()
        monitor._http_client = mock_client
        await monitor.close()
        mock_client.aclose.assert_called_once()


class TestAgentTypeEnum:
    def test_all_agents_have_span_names(self):
        from cogniverse_evaluation.quality_monitor import SPAN_NAME_BY_AGENT

        for agent_type in AgentType:
            assert agent_type in SPAN_NAME_BY_AGENT

    def test_verdict_ordering(self):
        assert Verdict.SKIP.value < Verdict.OPTIMIZE.value
