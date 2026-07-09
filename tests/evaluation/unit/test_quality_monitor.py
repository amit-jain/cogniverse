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


class TestSpanEvaluatorEndpoint:
    def test_span_evaluator_uses_monitor_phoenix_endpoint(self, golden_dataset):
        """The live-traffic span evaluator must query THIS monitor's Phoenix,
        not the provider default (localhost) — inside an Argo workflow pod
        the default endpoint is unreachable and every span query fails."""
        monitor = QualityMonitor(
            tenant_id="test_tenant",
            runtime_url="http://localhost:28000",
            phoenix_http_endpoint="http://cogniverse-phoenix:6006",
            llm_base_url="http://localhost:11434",
            llm_model="qwen3:4b",
            golden_dataset_path=golden_dataset,
            argo_api_url="http://localhost:2746",
            argo_namespace="test-ns",
        )
        import cogniverse_evaluation.providers as providers_mod

        monitor._make_span_evaluator()
        # The suite-level autouse fixture mocks the factory; the wiring
        # under test is the CALL — the monitor must hand its endpoint over.
        call = providers_mod.get_evaluation_provider.call_args
        assert call.kwargs["tenant_id"] == "test_tenant"
        assert call.kwargs["config"]["http_endpoint"] == (
            "http://cogniverse-phoenix:6006"
        )
        assert call.kwargs["config"]["project_name"] == (
            "cogniverse-test_tenant-runtime"
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
        # Prior baseline 0.6 carried on the result — drop is 33% > 10%.
        golden = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.4,
            mean_ndcg=0.5,
            mean_precision_at_5=0.3,
            query_count=10,
            baseline_mrr=0.6,
        )
        verdicts = monitor.check_thresholds(golden, None)
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    def test_golden_no_drop_skips(self, monitor):
        # Baseline 0.6 carried on the result — drop is 3% < 10%.
        golden = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.58,
            mean_ndcg=0.6,
            mean_precision_at_5=0.5,
            query_count=10,
            baseline_mrr=0.6,
        )
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
            low_scoring_queries=[{"query": "bad query", "mrr": 0.0}],
            high_scoring_queries=[{"query": "good query", "mrr": 1.0}],
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
                    low_scoring_examples=[{"query": "bad summary", "score": 0.2}],
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
            p["name"]: p["value"] for p in workflow["spec"]["arguments"]["parameters"]
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
                return httpx.Response(
                    200,
                    json={"results": [{"source_id": "v_-HpCLXdtcas", "score": 0.9}]},
                )
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
            transport=httpx.MockTransport(lambda r: httpx.Response(500, text="error")),
            base_url="http://testserver",
        )

        with pytest.raises(RuntimeError, match="No golden queries evaluated"):
            with patch.object(
                monitor, "_store_golden_eval_result", new_callable=AsyncMock
            ):
                await monitor.evaluate_golden_set()

    @pytest.mark.asyncio
    async def test_evaluate_golden_set_runs_queries_concurrently(self, monitor):
        """The per-query /search posts run concurrently, not one at a time."""
        import asyncio

        n = len(monitor._load_golden_queries())
        assert n >= 2
        barrier = asyncio.Barrier(n)

        class _FakeResp:
            status_code = 200

            def json(self):
                return {"results": [{"source_id": "v1", "score": 0.9}]}

        class _GatedClient:
            async def post(self, url, json=None):
                # Releases only when all N posts are concurrently in flight; a
                # sequential loop blocks the first one and wait_for times out.
                await barrier.wait()
                return _FakeResp()

        monitor._http_client = _GatedClient()

        with patch.object(monitor, "_store_golden_eval_result", new_callable=AsyncMock):
            result = await asyncio.wait_for(monitor.evaluate_golden_set(), timeout=10)

        assert result.query_count == n


class TestUpdateBaseline:
    @pytest.mark.asyncio
    async def test_update_baseline_stores_golden(self, monitor):
        with patch.object(
            monitor, "_store_golden_eval_result", new_callable=AsyncMock
        ) as mock:
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


class TestEvaluateLiveTraffic:
    @pytest.mark.asyncio
    async def test_evaluate_live_traffic_returns_results(self, monitor):
        """Live traffic eval with mocked span evaluator."""
        import pandas as pd

        mock_spans = pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "attributes": {"query": "test"},
                    "outputs": {"results": []},
                },
            ]
        )

        with patch("cogniverse_evaluation.span_evaluator.SpanEvaluator") as MockEval:
            mock_eval = MagicMock()
            mock_eval.get_recent_spans = AsyncMock(return_value=mock_spans)
            MockEval.return_value = mock_eval

            with patch.object(
                monitor, "_evaluate_agent_spans", new_callable=AsyncMock
            ) as mock_agent:
                mock_agent.return_value = AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.7,
                    baseline_score=0.8,
                    degradation_pct=0.12,
                    sample_count=1,
                )
                with patch.object(
                    monitor, "_store_live_eval_result", new_callable=AsyncMock
                ):
                    result = await monitor.evaluate_live_traffic()

        assert result.tenant_id == "test_tenant"


class TestEvaluateAgentSpans:
    @pytest.mark.asyncio
    async def test_evaluate_search_spans(self, monitor):
        """Score search spans via LLM judge."""
        import pandas as pd

        spans = pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "attributes": {"query": "test video"},
                    "outputs": {"results": [{"video_id": "v1", "score": 0.9}]},
                }
            ]
        )

        mock_judge = MagicMock()
        mock_judge._call_llm = AsyncMock(return_value="Score: 8/10. Good results.")
        mock_judge._extract_score_from_response = MagicMock(return_value=(0.8, "Good"))
        monitor._llm_judge = mock_judge

        with patch.object(
            monitor, "_get_agent_baseline", new_callable=AsyncMock, return_value=0.85
        ):
            result = await monitor._evaluate_agent_spans(AgentType.SEARCH, spans)

        assert result.agent == AgentType.SEARCH
        assert result.score == 0.8
        assert result.sample_count == 1

    @pytest.mark.asyncio
    async def test_unscored_judge_response_is_skipped_not_averaged(self, monitor):
        """A judge reply with no parseable score (LM failure) is skipped, not
        folded into the live quality average as a neutral 0.5."""
        import pandas as pd

        from cogniverse_evaluation.evaluators.llm_judge import LLMJudgeCore

        spans = pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "attributes": {"query": "q1"},
                    "outputs": {"results": [{"video_id": "v1", "score": 0.9}]},
                },
                {
                    "span_id": "s2",
                    "attributes": {"query": "q2"},
                    "outputs": {"results": [{"video_id": "v2", "score": 0.9}]},
                },
            ]
        )

        judge = LLMJudgeCore(model_name="x", base_url="http://unused")
        responses = iter(["Score: 8/10. Good.", "Evaluation failed: refused"])

        async def fake_call_llm(prompt, system_prompt=None, images=None):
            return next(responses)

        judge._call_llm = fake_call_llm
        monitor._llm_judge = judge

        with patch.object(
            monitor, "_get_agent_baseline", new_callable=AsyncMock, return_value=None
        ):
            result = await monitor._evaluate_agent_spans(AgentType.SEARCH, spans)

        # Only the scored span counts; averaging the failed one would give 0.65.
        assert result.sample_count == 1
        assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_agent_spans_scores_concurrently(self, monitor):
        """Each span's judge call runs concurrently, not one at a time."""
        import asyncio

        import pandas as pd

        n = 3
        spans = pd.DataFrame(
            [
                {
                    "span_id": f"s{i}",
                    "attributes": {"query": f"q{i}"},
                    "outputs": {"results": [{"video_id": f"v{i}", "score": 0.9}]},
                }
                for i in range(n)
            ]
        )

        barrier = asyncio.Barrier(n)

        async def gated_call_llm(prompt, system_prompt=None, images=None):
            # Releases only when all N judge calls are concurrently in flight.
            await barrier.wait()
            return "Score: 8/10. Good."

        judge = MagicMock()
        judge._call_llm = gated_call_llm
        judge._extract_score_from_response = MagicMock(return_value=(0.8, "Good"))
        monitor._llm_judge = judge

        with patch.object(
            monitor, "_get_agent_baseline", new_callable=AsyncMock, return_value=None
        ):
            result = await asyncio.wait_for(
                monitor._evaluate_agent_spans(AgentType.SEARCH, spans), timeout=10
            )

        assert result.sample_count == n
        assert result.score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_unreachable_lm_endpoint_skips_span(self, monitor):
        """A real LM transport failure (unreachable endpoint) skips the span
        instead of scoring a fabricated 0.5 — exercises the real _call_llm
        httpx error path, not a stub."""
        import pandas as pd

        from cogniverse_evaluation.evaluators.llm_judge import LLMJudgeCore

        spans = pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "attributes": {"query": "q1"},
                    "outputs": {"results": [{"video_id": "v1", "score": 0.9}]},
                }
            ]
        )
        # Port 1 refuses immediately -> _call_llm catches httpx error and
        # returns its failure sentinel string.
        monitor._llm_judge = LLMJudgeCore(model_name="x", base_url="http://127.0.0.1:1")

        with patch.object(
            monitor, "_get_agent_baseline", new_callable=AsyncMock, return_value=None
        ):
            result = await monitor._evaluate_agent_spans(AgentType.SEARCH, spans)

        assert result.sample_count == 0
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_all_judge_failures_yield_no_samples(self, monitor):
        """Every judgement failing leaves zero samples, not a 0.5 average."""
        import pandas as pd

        from cogniverse_evaluation.evaluators.llm_judge import LLMJudgeCore

        spans = pd.DataFrame(
            [
                {"span_id": "s1", "attributes": {"query": "q1"}, "outputs": {}},
                {"span_id": "s2", "attributes": {"query": "q2"}, "outputs": {}},
            ]
        )

        judge = LLMJudgeCore(model_name="x", base_url="http://unused")

        async def fake_call_llm(prompt, system_prompt=None, images=None):
            return "Evaluation failed: boom"

        judge._call_llm = fake_call_llm
        monitor._llm_judge = judge

        with patch.object(
            monitor, "_get_agent_baseline", new_callable=AsyncMock, return_value=None
        ):
            result = await monitor._evaluate_agent_spans(AgentType.SEARCH, spans)

        assert result.sample_count == 0
        assert result.score == 0.0


class TestGetAgentBaseline:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_baseline(self, monitor):
        with patch("phoenix.client.Client") as mock_client:
            mock_client.side_effect = Exception("no connection")
            result = await monitor._get_agent_baseline(AgentType.SEARCH)
        assert result is None


class TestHTTPClient:
    def test_lazy_init(self, monitor):
        assert monitor._http_client is None
        client = monitor._get_http_client()
        assert client is not None

    def test_load_golden_queries(self, monitor):
        queries = monitor._load_golden_queries()
        assert len(queries) == 2


class TestClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up(self, monitor):
        mock_client = AsyncMock()
        monitor._http_client = mock_client
        await monitor.close()
        mock_client.aclose.assert_called_once()


class TestXGBoostIntegration:
    def test_xgboost_overrides_optimize_to_skip(self, monitor):
        """XGBoost says don't train → override OPTIMIZE to SKIP."""
        mock_model = MagicMock()
        mock_model.should_train.return_value = (False, 0.01)
        monitor._training_decision_model = mock_model
        monitor._telemetry_provider = MagicMock()

        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.3,
                    baseline_score=0.8,
                    degradation_pct=0.6,
                    sample_count=20,
                ),
            },
        )

        verdicts = monitor.check_thresholds(None, live)
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP

    def test_xgboost_upgrades_skip_to_optimize(self, monitor):
        """XGBoost says train is beneficial → upgrade SKIP to OPTIMIZE."""
        mock_model = MagicMock()
        mock_model.should_train.return_value = (True, 0.08)
        monitor._training_decision_model = mock_model
        monitor._telemetry_provider = MagicMock()

        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.75,
                    baseline_score=0.80,
                    degradation_pct=0.06,
                    sample_count=20,
                ),
            },
        )

        verdicts = monitor.check_thresholds(None, live)
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    def test_no_telemetry_provider_skips_xgboost(self, monitor):
        """Without telemetry_provider, XGBoost is not consulted."""
        assert monitor._telemetry_provider is None

        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.3,
                    baseline_score=0.8,
                    degradation_pct=0.6,
                    sample_count=20,
                ),
            },
        )

        verdicts = monitor.check_thresholds(None, live)
        # Without XGBoost, naive check triggers OPTIMIZE
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE


class TestAgentTypeEnum:
    def test_verdict_ordering(self):
        assert Verdict.SKIP.value < Verdict.OPTIMIZE.value


class TestForceOptimizationCycle:
    """verify ``force_optimization_cycle`` runs the full
    eval+trigger+submit chain regardless of thresholds. This is the API
    used by the scheduled distillation Argo CronWorkflow."""

    @pytest.mark.asyncio
    async def test_force_cycle_builds_and_submits_with_argo(self, monitor):
        """Happy path: both evals succeed, trigger is built and submitted."""
        monitor.argo_api_url = "http://argo-server:2746"

        golden_result = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.8,
            mean_ndcg=0.75,
            mean_precision_at_5=0.6,
            query_count=10,
        )
        live_result = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
        )

        with (
            patch.object(
                monitor, "evaluate_golden_set", new_callable=AsyncMock
            ) as mock_golden,
            patch.object(
                monitor, "evaluate_live_traffic", new_callable=AsyncMock
            ) as mock_live,
            patch.object(
                monitor, "_store_trigger_dataset", new_callable=AsyncMock
            ) as mock_store,
            patch.object(
                monitor, "submit_optimization", new_callable=AsyncMock
            ) as mock_submit,
        ):
            mock_golden.return_value = golden_result
            mock_live.return_value = live_result

            result = await monitor.force_optimization_cycle()

        # Both evals were called.
        mock_golden.assert_awaited_once()
        mock_live.assert_awaited_once()
        # Trigger was stored AND submitted.
        mock_store.assert_awaited_once()
        mock_submit.assert_awaited_once()

        assert result["status"] == "ok"
        assert result["submitted_to_argo"] is True
        # ALL agents must be in the trigger, not just degraded ones.
        assert set(result["agents_triggered"]) == {
            "search",
            "summary",
            "report",
            "gateway",
        }

    @pytest.mark.asyncio
    async def test_force_cycle_skips_argo_when_url_missing(self, monitor):
        """Without argo_api_url, the cycle still builds and stores the
        trigger but doesn't try to submit."""
        monitor.argo_api_url = None

        golden_result = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="test_tenant",
            mean_mrr=0.8,
            mean_ndcg=0.75,
            mean_precision_at_5=0.6,
            query_count=10,
        )

        with (
            patch.object(
                monitor, "evaluate_golden_set", new_callable=AsyncMock
            ) as mock_golden,
            patch.object(
                monitor, "evaluate_live_traffic", new_callable=AsyncMock
            ) as mock_live,
            patch.object(monitor, "_store_trigger_dataset", new_callable=AsyncMock),
            patch.object(
                monitor, "submit_optimization", new_callable=AsyncMock
            ) as mock_submit,
        ):
            mock_golden.return_value = golden_result
            mock_live.return_value = LiveEvalResult(
                timestamp=datetime.utcnow(), tenant_id="test_tenant"
            )

            result = await monitor.force_optimization_cycle()

        mock_submit.assert_not_awaited()
        assert result["status"] == "ok"
        assert result["submitted_to_argo"] is False

    @pytest.mark.asyncio
    async def test_force_cycle_returns_no_data_when_both_evals_fail(self, monitor):
        """If both golden and live evals raise, the cycle must return a
        ``no_data`` status without crashing — Argo will retry on next run."""
        with (
            patch.object(
                monitor, "evaluate_golden_set", new_callable=AsyncMock
            ) as mock_golden,
            patch.object(
                monitor, "evaluate_live_traffic", new_callable=AsyncMock
            ) as mock_live,
            patch.object(
                monitor, "_store_trigger_dataset", new_callable=AsyncMock
            ) as mock_store,
            patch.object(
                monitor, "submit_optimization", new_callable=AsyncMock
            ) as mock_submit,
        ):
            mock_golden.side_effect = Exception("phoenix down")
            mock_live.side_effect = Exception("phoenix down")

            result = await monitor.force_optimization_cycle()

        assert result["status"] == "no_data"
        assert result["submitted_to_argo"] is False
        mock_store.assert_not_awaited()
        mock_submit.assert_not_awaited()


class TestSpanNameByAgent:
    """Pin the SPAN_NAME_BY_AGENT convention.

    the previous lookup table had values like
    ``"search_service.search"`` and ``"summarizer_agent.process"`` that did
    NOT match what agents actually emit, so live-traffic eval queried zero
    spans. After, AgentBase emits ``f"{ClassName}.process"``
    for every agent. These tests pin that contract so future renames trip
    a CI failure instead of silently breaking the monitoring loop.
    """

    def test_span_names_match_class_name_dot_process_format(self):
        """Every entry must be of the form ``<ClassName>.process`` so it
        matches what AgentBase._process_span() emits."""
        from cogniverse_evaluation.quality_monitor import SPAN_NAME_BY_AGENT

        for agent_type, span_name in SPAN_NAME_BY_AGENT.items():
            assert span_name.endswith(".process"), (
                f"{agent_type.value} span name '{span_name}' must end in "
                f"'.process' to match AgentBase._process_span() convention"
            )
            class_name = span_name.removesuffix(".process")
            assert class_name and class_name[0].isupper(), (
                f"{agent_type.value} span name '{span_name}' must start "
                f"with a capitalized class name (got '{class_name}')"
            )

    def test_span_names_cover_all_agent_types(self):
        """Every AgentType in the enum must have an entry."""
        from cogniverse_evaluation.quality_monitor import SPAN_NAME_BY_AGENT

        for agent_type in AgentType:
            assert agent_type in SPAN_NAME_BY_AGENT, (
                f"AgentType.{agent_type.name} missing from SPAN_NAME_BY_AGENT — "
                "add an entry or QualityMonitor will skip live eval for it"
            )

    def test_span_names_match_actual_agent_class_names(self):
        """The class-name half of each span name must correspond to a real
        agent class. If an agent gets renamed, this test catches the drift
        between the class name and the lookup table."""
        from cogniverse_evaluation.quality_monitor import SPAN_NAME_BY_AGENT

        expected = {
            AgentType.SEARCH: "SearchAgent",
            AgentType.SUMMARY: "SummarizerAgent",
            AgentType.REPORT: "DetailedReportAgent",
            AgentType.GATEWAY: "GatewayAgent",
        }
        for agent_type, expected_class in expected.items():
            span_name = SPAN_NAME_BY_AGENT[agent_type]
            assert span_name == f"{expected_class}.process", (
                f"{agent_type.value} expected '{expected_class}.process', "
                f"got '{span_name}'"
            )

    def test_span_name_format_matches_agent_base_emission(self):
        """End-to-end pin: instantiate a concrete subclass of AgentBase, call
        process() with a spy telemetry manager, and verify the span name is
        EXACTLY ``f"{ClassName}.process"``. If AgentBase._process_span()
        ever changes the format, this test fails and forces an update to
        SPAN_NAME_BY_AGENT in the same commit."""
        import asyncio
        from contextlib import contextmanager

        from cogniverse_core.agents.base import (
            AgentBase,
            AgentDeps,
            AgentInput,
            AgentOutput,
        )

        class _PinInput(AgentInput):
            query: str
            tenant_id: str = "test:unit"

        class _PinOutput(AgentOutput):
            ok: bool

        class _PinDeps(AgentDeps):
            pass

        class SearchAgent(AgentBase[_PinInput, _PinOutput, _PinDeps]):
            async def _process_impl(self, input: _PinInput) -> _PinOutput:
                return _PinOutput(ok=True)

        captured: list = []

        class _Spy:
            @contextmanager
            def span(self, name, tenant_id, project_name=None, attributes=None):
                captured.append(name)
                yield None

        agent = SearchAgent(deps=_PinDeps())
        agent.set_telemetry_manager(_Spy())
        asyncio.run(agent.process(_PinInput(query="hi", tenant_id="acme")))

        assert captured == ["SearchAgent.process"], (
            f"AgentBase emitted {captured!r}, but SPAN_NAME_BY_AGENT expects "
            f"'SearchAgent.process'. Fix one or the other."
        )


class TestNewThresholdCeilings:
    """golden_ndcg_drop_pct, error_rate_ceiling and latency_p95_ceiling_ms were
    configured but never evaluated; check_thresholds must now honor them."""

    def _golden(self, mean_ndcg, baseline_ndcg):
        return GoldenEvalResult(
            timestamp=datetime.now(),
            tenant_id="t",
            mean_mrr=0.9,
            mean_ndcg=mean_ndcg,
            mean_precision_at_5=0.9,
            query_count=10,
            baseline_mrr=0.9,  # no MRR drop, so only nDCG can trigger
            baseline_ndcg=baseline_ndcg,
        )

    def _live(self, error_rate=None, latency_p95_ms=None):
        agent = AgentEvalResult(
            agent=AgentType.SEARCH,
            score=0.9,  # above floor
            baseline_score=0.9,  # no degradation
            degradation_pct=0.0,
            sample_count=50,
            error_rate=error_rate,
            latency_p95_ms=latency_p95_ms,
        )
        return LiveEvalResult(
            timestamp=datetime.now(),
            tenant_id="t",
            agent_results={AgentType.SEARCH: agent},
        )

    def test_ndcg_drop_triggers_optimize(self, monitor):
        # 0.9 -> 0.7 is a 22% drop, past the 10% default.
        verdicts = monitor.check_thresholds(self._golden(0.7, 0.9), None)
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    def test_ndcg_stable_skips(self, monitor):
        verdicts = monitor.check_thresholds(self._golden(0.88, 0.9), None)
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP

    def test_error_rate_over_ceiling_triggers_optimize(self, monitor):
        # 0.20 > 0.05 default ceiling.
        verdicts = monitor.check_thresholds(None, self._live(error_rate=0.20))
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    def test_error_rate_under_ceiling_skips(self, monitor):
        verdicts = monitor.check_thresholds(None, self._live(error_rate=0.01))
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP

    def test_latency_p95_over_ceiling_triggers_optimize(self, monitor):
        # 5000ms > 1000ms default ceiling.
        verdicts = monitor.check_thresholds(None, self._live(latency_p95_ms=5000.0))
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    def test_latency_p95_under_ceiling_skips(self, monitor):
        verdicts = monitor.check_thresholds(None, self._live(latency_p95_ms=200.0))
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP

    def test_missing_operational_metrics_do_not_trigger(self, monitor):
        # error_rate/latency None (spans carried no status/latency) => skip.
        verdicts = monitor.check_thresholds(None, self._live())
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP


class TestOperationalHealth:
    def test_error_rate_and_p95_from_spans(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "status_code": ["OK", "OK", "ERROR", "OK"],
                "latency_ms": [100.0, 200.0, 300.0, 1000.0],
            }
        )
        err, p95 = QualityMonitor._operational_health(df)
        assert err == 0.25  # 1 of 4 errored
        assert p95 == pytest.approx(1000.0, rel=0.01) or p95 > 700

    def test_missing_columns_return_none(self):
        import pandas as pd

        err, p95 = QualityMonitor._operational_health(pd.DataFrame({"x": [1]}))
        assert err is None and p95 is None
